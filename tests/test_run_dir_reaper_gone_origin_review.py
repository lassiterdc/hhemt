"""Review tests for the gone-origin arm of the run-dir reaper.

Each test varies one property of the gone-origin arm that the coder's file leaves to a
single example: what counts as "named a path that is not a directory", what a gone origin
must NOT take with it, and that the arm acts on the real delete path.
"""

import os
import subprocess
import time

from hhemt.cache.paths import tier_address_for
from tests.fixtures._run_dir_reaper import LiveSignals, collect_live_signals, sweep

_DAY = 86400


def _repo(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _tier(runs_root, slug, breadcrumb=None, age_days=0):
    (runs_root / slug).mkdir(parents=True)
    if breadcrumb is not None:
        (runs_root / slug / ".origin").write_text(breadcrumb, encoding="utf-8")
    if age_days:
        old = time.time() - age_days * _DAY
        os.utime(runs_root / slug, (old, old))


def test_an_origin_that_is_a_file_is_gone_and_a_whitespace_breadcrumb_is_not(tmp_path):
    """Property: "names a path that is not a directory" means `is_dir()` is False, so a recorded
    regular file is gone, and a breadcrumb of only whitespace is empty (stripped) and says nothing.

    Class: NEW CAPABILITY -- the predicate of the `origin-gone` signal in `collect_live_signals`.
    Why required: the predicate is read, non-empty, not a directory.
    What kills it: `exists()` in place of `is_dir()` (the file tier is kept); dropping `.strip()`
    (the whitespace tier is reaped as gone).
    Second correct implementation: `not os.path.isdir(recorded)`; same verdicts.
    The directory-origin tier is the must-not-reap control, and the file tier is the must-reap one.
    """
    repo = _repo(tmp_path)
    a_file = tmp_path / "a_file"
    a_file.write_text("x")
    a_dir = tmp_path / "a_dir"
    a_dir.mkdir()
    runs_root = tmp_path / "runs"
    _tier(runs_root, "path_file", f"{a_file}\n")
    _tier(runs_root, "path_dir", f"{a_dir}\n")
    _tier(runs_root, "path_blank", "  \t\n \n")
    slugs = frozenset({"path_file", "path_dir", "path_blank"})

    signals = collect_live_signals(repo, runs_root=runs_root, candidate_slugs=slugs)

    assert signals.gone_origin_slugs == frozenset({"path_file"})
    report = sweep(runs_root, signals=signals, ttl_days=7, dry_run=True)
    assert report.reaped == ["path_file"]
    assert report.kept == {"path_dir": "live-path", "path_blank": "ttl-not-elapsed"}


def test_a_breadcrumb_that_cannot_be_read_is_never_a_gone_origin(tmp_path):
    """Property: only a breadcrumb that was READ can mark a tier gone. An unreadable one
    (here `.origin` is a directory, so `read_text` raises an `OSError`) keeps the fail-safe, and a
    tier with no breadcrumb whose name is not `path`-prefixed keeps the plain TTL.

    Class: REGRESSION -- the behaviour of the `OSError` branch must survive the arm.
    Why required: the design says "an unreadable breadcrumb still goes to the fail-safe"; a gone
    verdict from a read that failed would reap a live tier no breadcrumb was ever written for.
    What kills it: moving the gone-origin test above the read, or catching the `OSError` as gone.
    A tier with no readable breadcrumb is reaped as `reapable` only once it is past the time limit, never as
    `origin-gone`; younger and unreadable ones stay at `ttl-not-elapsed`. That is the control that the reason
    names the TTL arm and not the gone-origin arm.
    """
    repo = _repo(tmp_path)
    runs_root = tmp_path / "runs"
    _tier(runs_root, "path_unreadable")
    (runs_root / "path_unreadable" / ".origin").mkdir()
    _tier(runs_root, "legacy_unreadable")
    (runs_root / "legacy_unreadable" / ".origin").mkdir()
    _tier(runs_root, "legacy_young")
    _tier(runs_root, "legacy_old", age_days=30)
    _tier(runs_root, "path_removed", f"{tmp_path / 'removed'}\n")
    slugs = frozenset({"path_unreadable", "legacy_unreadable", "legacy_young", "legacy_old", "path_removed"})

    signals = collect_live_signals(repo, runs_root=runs_root, candidate_slugs=slugs)

    assert signals.gone_origin_slugs == frozenset({"path_removed"})
    assert "path_unreadable" in signals.live_path_slugs
    report = sweep(runs_root, signals=signals, ttl_days=7, dry_run=True)
    assert sorted(report.reaped) == ["legacy_old", "path_removed"]
    assert report.kept["path_unreadable"] == "live-path"
    assert report.kept["legacy_unreadable"] == "ttl-not-elapsed"
    assert report.kept["legacy_young"] == "ttl-not-elapsed"


def test_a_superseded_tier_keeps_the_ttl_and_is_not_a_gone_origin(tmp_path):
    """Property: a tier whose origin EXISTS but whose address differs from what that checkout
    derives today is superseded, not gone: it keeps the TTL (kept while young, `reapable` once
    old), and the tier that checkout does derive stays `live-path`.

    Class: REGRESSION -- the approved design says "Superseded tiers keep the TTL".
    Why required: widening the gone predicate to "address differs" would reap a young superseded
    tier a checkout at another commit may still use.
    What kills it: adding a superseded slug to the gone-origin set (the young tier is reaped as
    `origin-gone`).
    The current tier is the must-keep control.
    """
    repo = _repo(tmp_path)
    checkout = tmp_path / "checkout"
    (checkout / "tests" / "fixtures").mkdir(parents=True)
    (checkout / "tests" / "fixtures" / "_triton_source_cache.py").write_text("def checkout_slug(cwd):\n    pass\n")
    checkout = checkout.resolve()
    current = tier_address_for(checkout)
    runs_root = tmp_path / "runs"
    for slug, age in ((current, 0), ("path_superseded_young", 0), ("path_superseded_old", 30)):
        _tier(runs_root, slug, f"{checkout}\n", age_days=age)
    slugs = frozenset({current, "path_superseded_young", "path_superseded_old"})

    signals = collect_live_signals(repo, runs_root=runs_root, candidate_slugs=slugs)

    assert signals.gone_origin_slugs == frozenset()
    assert signals.live_path_slugs == frozenset({current})
    report = sweep(runs_root, signals=signals, ttl_days=7, dry_run=True)
    assert report.reaped == ["path_superseded_old"]
    assert report.kept == {current: "live-path", "path_superseded_young": "ttl-not-elapsed"}


def test_applying_the_sweep_deletes_only_the_gone_origin_tier(tmp_path):
    """Property: with `dry_run=False` the gone-origin tier is removed from disk and counted in
    `bytes_reclaimed`, and a live-origin tier of the same age is left on disk.

    Class: NEW CAPABILITY -- the arm acts on the real delete path, not only on the report.
    Why required: the reaper's job is to free disk, and every earlier test of the arm passes
    `dry_run=True`, so a verdict that never reached `fast_rmtree` would pass them all.
    What kills it: a `reapable` verdict whose branch skips the delete for this reason.
    The surviving live tier is the control.
    """
    repo = _repo(tmp_path)
    live = tmp_path / "checkout"
    live.mkdir()
    runs_root = tmp_path / "runs"
    _tier(runs_root, "path_gone", f"{tmp_path / 'removed'}\n")
    (runs_root / "path_gone" / "payload.bin").write_bytes(b"0123456789")
    _tier(runs_root, "path_live", f"{live}\n")
    slugs = frozenset({"path_gone", "path_live"})

    signals = collect_live_signals(repo, runs_root=runs_root, candidate_slugs=slugs)
    report = sweep(runs_root, signals=signals, ttl_days=7, dry_run=False)

    assert report.reaped == ["path_gone"]
    assert report.bytes_reclaimed >= 10
    assert not (runs_root / "path_gone").exists()
    assert (runs_root / "path_live").is_dir()


def test_every_liveness_signal_still_wins_over_a_gone_origin_through_the_sweep(tmp_path):
    """Property: a slug named gone AND by one liveness signal is kept under that signal's own
    reason when it reaches `sweep`, for each of the four signals, and `main` stays unreapable.

    Class: REGRESSION -- "Every liveness arm still wins", at the `sweep` wiring and not only in
    the pure `is_reapable` truth table.
    Why required: a gone origin reaps whatever the age, so a liveness signal ignored here is the
    one path on which a tier a process is using is deleted.
    What kills it: `sweep` passing the gone set into a position that outranks a liveness arm, or
    reading the gone set into the liveness sets.
    The `reaped_only_by_gone` tier is the must-reap control.
    """
    runs_root = tmp_path / "runs"
    names = ("by_worktree", "by_branch", "by_lock", "by_path", "main", "reaped_only_by_gone")
    for name in names:
        _tier(runs_root, name)
    signals = LiveSignals(
        live_worktree_slugs=frozenset({"by_worktree"}),
        live_branch_slugs=frozenset({"by_branch"}),
        locked_slugs=frozenset({"by_lock"}),
        live_path_slugs=frozenset({"by_path"}),
        gone_origin_slugs=frozenset(names),
    )

    report = sweep(runs_root, signals=signals, ttl_days=7, dry_run=True)

    assert report.reaped == ["reaped_only_by_gone"]
    assert report.kept == {
        "by_worktree": "live-worktree",
        "by_branch": "live-branch",
        "by_lock": "compile-lock",
        "by_path": "live-path",
        "main": "never-reap",
    }
