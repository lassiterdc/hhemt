"""Phase 4 — liveness-gated run-directory reaper tests."""

import hashlib
import subprocess

import pytest

from tests.fixtures._run_dir_reaper import LiveSignals, collect_live_signals, is_reapable, sweep
from tests.fixtures._triton_source_cache import canonical_root, synthetic_runs_root

_NO_SIGNALS = LiveSignals(
    live_worktree_slugs=frozenset(),
    live_branch_slugs=frozenset(),
    locked_slugs=frozenset(),
)


def test_is_reapable_truth_table():
    """PURE — zero subprocess, zero filesystem. Every arm gets a row: live
    worktree blocks; live local branch blocks; held compile lock blocks;
    `main` and `_software` block; TTL-not-elapsed blocks; all-negative reaps.
    Each row asserts the returned REASON, not just the boolean, so a predicate
    that returns the right answer via the wrong arm fails."""
    day = 86400
    now = 1_000_000.0
    base = dict(
        live_worktree_slugs=frozenset(),
        live_branch_slugs=frozenset(),
        locked_slugs=frozenset(),
        ttl_days=7,
        now=now,
        mtime=now - 30 * day,  # old enough that TTL never blocks by default
    )

    # Hard exclusions block regardless of everything else.
    assert is_reapable("main", **base) == (False, "never-reap")
    assert is_reapable("_software", **base) == (False, "never-reap")

    # Each liveness arm blocks and names itself.
    assert is_reapable("s", **{**base, "live_worktree_slugs": frozenset({"s"})}) == (
        False,
        "live-worktree",
    )
    assert is_reapable("s", **{**base, "live_branch_slugs": frozenset({"s"})}) == (
        False,
        "live-branch",
    )
    assert is_reapable("s", **{**base, "locked_slugs": frozenset({"s"})}) == (
        False,
        "compile-lock",
    )

    # TTL not elapsed blocks (young dir), when every liveness arm is negative.
    assert is_reapable("s", **{**base, "mtime": now - 1 * day}) == (
        False,
        "ttl-not-elapsed",
    )

    # All-negative and old enough: reaps.
    assert is_reapable("s", **base) == (True, "reapable")

    # Liveness-first ordering: a live slug that is ALSO young returns the liveness
    # arm, never "ttl-not-elapsed" — order is checked FIRST.
    assert is_reapable(
        "s",
        **{
            **base,
            "live_worktree_slugs": frozenset({"s"}),
            "mtime": now - 1 * day,
        },
    ) == (False, "live-worktree")


def test_sweep_root_has_no_default():
    """Structural guard: a default root would let a test that forgets tmp_path
    sweep the real cache AND still pass."""
    import inspect

    from tests.fixtures._run_dir_reaper import sweep

    p = inspect.signature(sweep).parameters["runs_root"]
    assert p.default is inspect.Parameter.empty, "sweep() must never default its root"


def test_sweep_refuses_real_root_from_a_test_body():
    """Asserts the RAISE, never an empty report — an empty-report assertion
    passes both when the guard fires and when the guard is deleted and nothing
    happened to be reapable."""
    from tests.fixtures._run_dir_reaper import default_runs_root, sweep

    with pytest.raises(RuntimeError, match="refusing to sweep the real cache root"):
        sweep(default_runs_root(), signals=_NO_SIGNALS, ttl_days=7)


def test_sweep_reports_reaped_and_kept(tmp_path):
    """tmp_path-rooted end-to-end: seeds live and dead candidates, asserts the
    dead one is reaped, the live ones are kept with the correct arm named, and
    the canonical store path is never a candidate."""
    # Risk X4: the canonical store is a SIBLING of synthetic_test_runs/, never
    # nested inside it, so it can never be enumerated as a sweep candidate.
    assert not str(canonical_root()).startswith(str(synthetic_runs_root()) + "/")

    # One dead candidate (all liveness signals negative), one blocked by each
    # arm, plus the two hard exclusions.
    for name in ("dead", "live_wt", "live_br", "live_lock", "main", "_software"):
        (tmp_path / name).mkdir()
    (tmp_path / "dead" / "payload.bin").write_bytes(b"x" * 4096)
    # A symlinked child must never be followed or reaped.
    (tmp_path / "a_symlink").symlink_to(tmp_path / "live_wt")

    signals = LiveSignals(
        live_worktree_slugs=frozenset({"live_wt"}),
        live_branch_slugs=frozenset({"live_br"}),
        locked_slugs=frozenset({"live_lock"}),
    )
    report = sweep(tmp_path, signals=signals, ttl_days=0, dry_run=False)

    assert report.reaped == ["dead"]
    assert not (tmp_path / "dead").exists()
    assert report.bytes_reclaimed >= 4096
    assert report.kept == {
        "live_wt": "live-worktree",
        "live_br": "live-branch",
        "live_lock": "compile-lock",
        "main": "never-reap",
        "_software": "never-reap",
    }
    # Every kept candidate — and the untouched symlink — still exists on disk.
    for name in ("live_wt", "live_br", "live_lock", "main", "_software"):
        assert (tmp_path / name).is_dir()
    assert (tmp_path / "a_symlink").is_symlink()


def test_backstop_wrapper_first_run_gates_apply_on_acknowledgment(tmp_path, monkeypatch):
    """The conftest backstop's wrapper must NOT delete until a per-machine
    acknowledgment sentinel exists (friction reaper-backstop-auto-apply-safety).

    Fails against the pre-affordance wrapper, which hard-coded dry_run=False:
    the first call would delete the candidate AND ReapReport had no `applied`.

    `collect_live_signals` is stubbed so no real `git` runs and the candidate is
    unconditionally un-live — the ACK GATE, not liveness, is under test. Rooted
    at tmp_path, so the real cache is never reachable; xdist-safe by tmp_path
    uniqueness."""
    import tests.fixtures._run_dir_reaper as reaper

    monkeypatch.setattr(reaper, "collect_live_signals", lambda *a, **k: _NO_SIGNALS)

    dead = tmp_path / "reaper_test_dead_slug"
    dead.mkdir()
    (dead / "payload.bin").write_bytes(b"x" * 4096)

    # No acknowledgment yet: the backstop runs DRY — the dir SURVIVES, the report
    # names what it WOULD reclaim, and `applied` is False.
    report = reaper.sweep_with_live_signals(tmp_path, ttl_days=0)
    assert report.applied is False
    assert report.reaped == ["reaper_test_dead_slug"]
    assert dead.is_dir(), "backstop deleted before any acknowledgment existed"

    # Operator reviews the dry-run report, then acknowledges on THIS machine.
    reaper._reaper_ack_path(tmp_path).touch()

    # Acknowledged: the backstop now applies — the stale dir is reclaimed.
    report = reaper.sweep_with_live_signals(tmp_path, ttl_days=0)
    assert report.applied is True
    assert report.reaped == ["reaper_test_dead_slug"]
    assert not dead.exists(), "acknowledged backstop did not reclaim the stale dir"


def test_the_path_arm_keeps_the_digest_tier_of_a_live_long_named_worktree(tmp_path):
    """Property: a worktree whose folder name exceeds the digest length is addressed by a digest,
    and the reaper keeps that tier through its path arm, via the tier's `.origin` breadcrumb.

    Class: NEW CAPABILITY -- a 30-character worktree name, which `tier_address` now maps to a
    `path-` digest address that no git worktree folder name and no branch name equals, so the
    worktree arm and the branch arm cannot protect it.
    Why required: moving long-named worktrees onto the digest address is safe only if the
    reaper still keeps their live tier; a path arm that failed to re-derive the address would
    leave a live tier reapable once its 7-day bound passes.
    What kills it: an address rule that mints for the worktree an address the reaper's
    `tier_address_for(origin)` does not re-derive (against the code before this change the
    digest tier is not the address the worktree mints, so the first assertion reds); and a path
    arm that keeps every tier whose breadcrumb names the worktree, which the second tier at an
    address the worktree does not mint (`path-` plus sixteen zeros) detects.
    Second correct implementation: a path arm that compares a re-derived address to the tier
    name by a normalised form rather than by string equality; it keeps the same tiers.
    The `_triton_source_cache.py` stub carries `def checkout_slug(` because without it
    `_checkout_mints_digest_address` returns False and the path arm keeps every tier whose
    breadcrumb names the worktree, so the keep verdict would not depend on the address.
    """
    if ".claude" in tmp_path.resolve().parts:
        pytest.skip("the temporary directory sits under a `.claude` component; the arms cannot be separated")
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=reaper-test", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "seed"], check=True)
    name = "w" * 30
    worktree = repo / ".claude" / "worktrees" / name
    subprocess.run([*git, "worktree", "add", "-q", "-b", "wt-branch", str(worktree)], check=True)
    stub = worktree / "tests" / "fixtures" / "_triton_source_cache.py"
    stub.parent.mkdir(parents=True)
    stub.write_text("def checkout_slug(cwd):\n    return ''\n", encoding="utf-8")

    digest_slug = "path-" + hashlib.sha256(str(worktree).encode("utf-8", "surrogateescape")).hexdigest()[:16]
    foreign_slug = "path-" + "0" * 16
    runs_root = tmp_path / "runs"
    for slug in (digest_slug, foreign_slug):
        (runs_root / slug).mkdir(parents=True)
        (runs_root / slug / ".origin").write_text(f"{worktree}\n", encoding="utf-8")

    signals = collect_live_signals(repo, runs_root=runs_root, candidate_slugs=frozenset({digest_slug, foreign_slug}))
    assert digest_slug in signals.live_path_slugs
    assert foreign_slug not in signals.live_path_slugs

    report = sweep(runs_root, signals=signals, ttl_days=0, dry_run=True)
    assert report.kept[digest_slug] == "live-path"
    assert foreign_slug in report.reaped

    # The unreadable-breadcrumb fail-safe: the digest tier with its `.origin` removed is
    # still kept, because every digest address starts with `path`.
    (runs_root / digest_slug / ".origin").unlink()
    signals = collect_live_signals(repo, runs_root=runs_root, candidate_slugs=frozenset({digest_slug}))
    assert digest_slug in signals.live_path_slugs
