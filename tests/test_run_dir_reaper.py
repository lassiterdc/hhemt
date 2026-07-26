"""Phase 4 — liveness-gated run-directory reaper tests."""

import pytest

from tests.fixtures._run_dir_reaper import LiveSignals, is_reapable, sweep
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
