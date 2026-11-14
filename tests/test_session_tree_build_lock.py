"""The completed session trees are built by one process per node and adopted by the rest.

Problem this file pins. Under `pytest -n K` every xdist worker is its own session, so `synthetic_multisim_completed`
and `synthetic_sensitivity_completed` would each run the multi-minute workflow into one tree unless the workers
exclude each other. `tests/conftest.py::_adopt_or_build_session_tree` holds an inter-process lock on the runs root
while it clears Snakemake's stale state, reads the completion marker and builds.

These tests drive that helper with stand-in callables, never the real fixtures, which execute the solver. The
stand-in `build` counts how many callers enter it and how many are inside at once; the stand-in marker is a file the
build writes, which is how a waiting caller adopts. Threads stand in for workers: each call builds its own lock
object, and two lock objects on one path exclude each other even inside one process.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import tests.conftest as suite_conftest
from hhemt._filelock_compat import resolve_filelock

_WORKERS = 4
_BUILD_SECONDS = 0.3


class _Counter:
    """Counts entries into the stand-in build and the most callers inside it at once."""

    def __init__(self) -> None:
        self.entries = 0
        self.inside = 0
        self.max_inside = 0
        self._guard = threading.Lock()

    def enter(self) -> None:
        """Record one caller entering the build."""
        with self._guard:
            self.entries += 1
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)

    def leave(self) -> None:
        """Record one caller leaving the build."""
        with self._guard:
            self.inside -= 1


def _analysis_dir(tmp_path: Path) -> Path:
    """An analysis directory laid out like the real ones: `{runs root}/{case}/{case}`."""
    path = tmp_path / "runs" / "synth_case" / "synth_case"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_workers(analysis_dir: Path, counter: _Counter) -> list[str]:
    """Run `_WORKERS` concurrent callers of the helper against one tree and return each caller's outcome."""
    marker = analysis_dir / "_status" / "complete.flag"
    marker.parent.mkdir(parents=True, exist_ok=True)
    start = threading.Barrier(_WORKERS)

    def build() -> None:
        """The stand-in build: count the entry, take a moment, write the completion marker."""
        counter.enter()
        try:
            time.sleep(_BUILD_SECONDS)
            marker.write_text("built")
        finally:
            counter.leave()

    def one_worker() -> str:
        """One stand-in worker: wait for the others, then adopt or build through the helper."""
        start.wait()
        suite_conftest._adopt_or_build_session_tree(
            analysis_dir,
            state_of=lambda: "complete" if marker.exists() else "absent",
            build=build,
            on_unsatisfied=lambda: None,
        )
        return "adopted-or-built"

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        return [future.result(timeout=60) for future in [pool.submit(one_worker) for _ in range(_WORKERS)]]


def test_with_the_lock_exactly_one_worker_builds_and_the_rest_adopt(tmp_path):
    """Property: of several workers reaching one cold tree together, exactly one enters the build, never two at once,
    and every worker returns.

    Class: NEW CAPABILITY. The wrong implementation reads the marker outside the lock, or takes no lock, so every
    worker sees "not built" and builds. The count of entries is the observable.

    Why it is required, checkable against the code: `_adopt_or_build_session_tree` reads `state_of()` inside
    `_session_tree_lock`, which is the only reason a waiting worker finds the marker current; the real fixtures call
    it, so a count of one here is the property the fixtures rely on under `-n K`.

    What kills it: removing the `with _session_tree_lock(...)` from the helper, or moving the marker read before it.

    A second correct implementation under which it passes: one that elects a single builder with a different
    primitive (an `O_EXCL` marker), as long as the others wait and adopt."""
    counter = _Counter()
    outcomes = _run_workers(_analysis_dir(tmp_path), counter)
    assert len(outcomes) == _WORKERS
    assert (counter.entries, counter.max_inside) == (1, 1)


def test_without_the_lock_more_than_one_worker_builds(tmp_path, monkeypatch):
    """Must-fire: the same workers with the lock replaced by a no-op enter the build more than once.

    Class: NEW CAPABILITY. Control: the previous test, which differs only in the lock. This is what makes the count
    of one a measurement of the lock and not of timing: with `_BUILD_SECONDS` of work inside the build the workers
    overlap, and this arm shows they do.

    Why it is required, checkable against the code: a count of one that held without the lock would be luck.

    What kills it: nothing in the helper; it kills a stand-in that would pass with no lock.

    A second correct implementation under which it passes: a no-op context manager of any kind."""
    monkeypatch.setattr(suite_conftest, "_session_tree_lock", lambda analysis_dir: _NoLock())
    counter = _Counter()
    _run_workers(_analysis_dir(tmp_path), counter)
    assert counter.entries > 1


class _NoLock:
    """A context manager that excludes nothing."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


def _lock_is_held(analysis_dir: Path) -> bool:
    """Whether another lock object on the tree's lock file can be taken right now (False means it is held)."""
    lock_path = analysis_dir.parent.parent / f".{analysis_dir.parent.name}.session_tree.lock"
    probe = resolve_filelock(lock_path, timeout=0)
    try:
        probe.acquire()
    except Exception:  # noqa: BLE001 - filelock's Timeout is the only expected one; any failure to take it means held
        return True
    probe.release()
    return False


def _removals_of_snakemake_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, lock_around: bool):
    """Run the clearing step and return, per removed `locks`/`incomplete` path, whether the lock was held.

    With `lock_around` the call goes through the helper; without it the clearing function is called bare."""
    analysis_dir = _analysis_dir(tmp_path)
    for name in ("locks", "incomplete"):
        (analysis_dir / ".snakemake" / name).mkdir(parents=True)
    seen: list[tuple[str, bool]] = []
    real = suite_conftest.shutil.rmtree

    def recording_rmtree(path, *args, **kwargs):
        """Record the lock state at the moment of a removal, then remove."""
        seen.append((Path(path).name, _lock_is_held(analysis_dir)))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(suite_conftest.shutil, "rmtree", recording_rmtree)
    if lock_around:
        suite_conftest._adopt_or_build_session_tree(
            analysis_dir, state_of=lambda: "complete", build=lambda: None, on_unsatisfied=lambda: None
        )
    else:
        suite_conftest._clear_stale_snakemake_state(analysis_dir)
    return seen


def test_the_snakemake_state_is_removed_only_while_the_lock_is_held(tmp_path, monkeypatch):
    """Property: both removals (`locks` and `incomplete`) happen while the tree's lock is held.

    Class: NEW CAPABILITY. The wrong implementation clears before taking the lock, as the fixtures did, so a second
    worker clears the directories under the first worker's in-flight build. A stand-in `rmtree` records, at each
    removal, whether another lock object on the same file is blocked.

    Why it is required, checkable against the code: `_clear_stale_snakemake_state` is called from
    `_adopt_or_build_session_tree` after `_session_tree_lock` is entered, and nowhere else in the fixtures.

    What kills it: calling the clearing step before the `with` in the helper. Control, in this test: the same
    removals called bare (no lock) are recorded as not held, so the recorder can see the difference.

    A second correct implementation under which it passes: one that clears from inside the stand-in `build` instead,
    as long as that runs under the lock."""
    bare = _removals_of_snakemake_state(tmp_path / "bare", monkeypatch, lock_around=False)
    assert bare and not any(held for _name, held in bare), "control: a bare call must be seen as unlocked"
    guarded = _removals_of_snakemake_state(tmp_path / "guarded", monkeypatch, lock_around=True)
    assert sorted(name for name, _held in guarded) == ["incomplete", "locks"]
    assert all(held for _name, held in guarded)
