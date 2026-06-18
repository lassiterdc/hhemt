"""Unit tests for version_migration.state - read/write/detect/concurrency."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from hhemt.version_migration import state
from hhemt.version_migration.constants import LAYOUT_VERSION

# Phase 4 (synth-test-isolation-and-runtime): the two stamp-wire regression
# tests at the bottom of this file (test_analysis_run_stamps_version_file and
# test_submit_workflow_stamps_version_file) invoke analysis.run() and
# analysis.submit_workflow() respectively, which launch snakemake subprocesses.
# File-level marker chosen over per-function for uniformity with the rest of
# the phase-4 marker set; the cost of serializing the unit tests above is
# negligible because they are fast.
pytestmark = pytest.mark.requires_snakemake_subprocess


def test_read_returns_none_when_missing(tmp_path: Path) -> None:
    assert state.read_version_file(tmp_path) is None


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    s = state.VersionState.fresh(layout_version=2, toolkit_version="0.7.1")
    state.write_version_file(tmp_path, s)
    out = state.read_version_file(tmp_path)
    assert out is not None
    assert out.layout_version == 2
    assert out.toolkit_version == "0.7.1"
    assert out.migration_history == []


def test_stamp_is_idempotent_at_same_version(tmp_path: Path) -> None:
    state.stamp_new_target(tmp_path, LAYOUT_VERSION)
    first_mtime = (tmp_path / "_version.json").stat().st_mtime
    state.stamp_new_target(tmp_path, LAYOUT_VERSION)  # no-op
    second_mtime = (tmp_path / "_version.json").stat().st_mtime
    assert first_mtime == second_mtime


def test_infer_returns_zero_for_legacy_iloc_prefix(tmp_path: Path) -> None:
    sims = tmp_path / "sims"
    sims.mkdir()
    (sims / "0-event_id.0").mkdir()
    assert state.infer_layout_version(tmp_path) == 0


def test_concurrent_stamping_yields_single_write(tmp_path: Path) -> None:
    """10 threads simultaneously stamp the same dir; exactly one write occurs."""
    barrier = threading.Barrier(10)

    def worker():
        barrier.wait()
        state.stamp_new_target(tmp_path, LAYOUT_VERSION)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    out = state.read_version_file(tmp_path)
    assert out is not None
    assert out.layout_version == LAYOUT_VERSION
    # No partial / corrupt writes:
    json.loads((tmp_path / "_version.json").read_text())


def _mp_stamp_worker(target_dir: str) -> None:
    """Module-scope worker for multiprocessing test; picklable under spawn."""
    from pathlib import Path as _P

    from hhemt.version_migration import state as _state
    from hhemt.version_migration.constants import (
        LAYOUT_VERSION as _LV,
    )

    _state.stamp_new_target(_P(target_dir), _LV)


def test_multiprocess_stamping_yields_single_write(tmp_path: Path) -> None:
    """N separate processes simultaneously stamp the same dir; exactly one write occurs.

    Mirrors the 1_job_many_srun_tasks production case where srun spawns
    separate processes (not threads). filelock must provide cross-process
    safety, which the threading test alone does not exercise. Uses an
    explicit spawn context so the test is portable across Linux (default
    fork) and macOS (default spawn) CI - spawn requires the worker to be
    picklable, which a module-scope function is.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_mp_stamp_worker, args=(str(tmp_path),)) for _ in range(10)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    out = state.read_version_file(tmp_path)
    assert out is not None
    assert out.layout_version == LAYOUT_VERSION
    json.loads((tmp_path / "_version.json").read_text())  # no partial writes


# ---- Stamp-wire regression tests (SE Spec 7 + testing Spec 7) ----


@pytest.mark.slow
def test_analysis_run_stamps_version_file(norfolk_single_sim_analysis_cached) -> None:
    """TRITONSWMM_analysis.run() writes _version.json at analysis_dir on first call.

    Lazy-stamping wire verification: proves the wire in analysis.py fires
    at the documented point and writes to AnalysisPaths.analysis_dir.
    Converts the manual-inspection DoD into an executable assertion.
    """
    analysis = norfolk_single_sim_analysis_cached
    analysis.run(from_scratch=False)
    version_path = analysis.analysis_paths.analysis_dir / "_version.json"
    assert version_path.exists()
    st = state.read_version_file(analysis.analysis_paths.analysis_dir)
    assert st is not None
    assert st.layout_version == LAYOUT_VERSION


@pytest.mark.slow
def test_submit_workflow_stamps_version_file(
    norfolk_single_sim_analysis_cached,
) -> None:
    """TRITONSWMM_analysis.submit_workflow() writes _version.json on first call."""
    analysis = norfolk_single_sim_analysis_cached
    analysis.submit_workflow(dry_run=True)
    version_path = analysis.analysis_paths.analysis_dir / "_version.json"
    assert version_path.exists()
    st = state.read_version_file(analysis.analysis_paths.analysis_dir)
    assert st is not None
    assert st.layout_version == LAYOUT_VERSION


def test_system_init_stamps_version_file(
    norfolk_single_sim_analysis_cached,
) -> None:
    """TRITONSWMM_system.__init__ writes _version.json at system_directory eagerly."""
    analysis = norfolk_single_sim_analysis_cached
    sys_dir = analysis._system.cfg_system.system_directory
    assert (sys_dir / "_version.json").exists()
    st = state.read_version_file(sys_dir)
    assert st is not None
    assert st.layout_version == LAYOUT_VERSION
