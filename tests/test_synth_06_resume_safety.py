"""Phase 6 regression test — resume-safety for per-checkpoint file merge.

When a TRITON-SWMM workflow resumes across SLURM allocations, the per-checkpoint
``performance{N}.txt`` files from the pre-resume allocation MUST be preserved
alongside the post-resume files so that V0008's `_aggregate_perf_tseries` can
reconstruct end-to-end wallclock across the allocation boundary.

This test asserts the invariant codified by the stipulation
``library/docs/stipulations/TRITON-SWMM_toolkit/clear raw triton outputs deferred
until last allocation.md``:

  1. ``test_two_allocation_merge_recovers_full_wallclock`` — when both batches of
     ``performance{N}.txt`` files are present (pre-resume + post-resume),
     ``_aggregate_perf_summary`` recovers the slowest-rank end-to-end wallclock
     across the allocation boundary (reset-detector path).
  2. ``test_pre_resume_files_deleted_undercounts_wallclock`` — when the pre-resume
     batch is deleted (simulating mid-workflow ``_clear_raw_TRITON_outputs``),
     ``_aggregate_perf_summary`` produces a strictly smaller ``Total`` than the
     full-batch summary. The undercount is detectable in CI.

The test calls the module-level ``_aggregate_perf_summary`` helper directly per the
same pattern as ``tests/test_synth_03_perf_tseries_diff.py``; no analysis-instance
harness is required because V0008 + this regression share one source of truth with
the production aggregator (Phase 1 Spec 8 architectural constraint).
"""
from __future__ import annotations

import pytest


def _write_performance_file(perf_dir, tstep, rank0_total, rank1_total):
    """Write a single ``performance{tstep}.txt`` file with two ranks + Average row.

    The Total column is the cumulative wallclock-semantic value for that rank
    at that checkpoint; other columns are populated for parse_performance_file
    compatibility but their per-column values are not asserted on.
    """
    content = "%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total\n"
    # Rank 0: split Total ⅔ Compute, ⅓ SWMM (numerically distinct, all positive).
    r0_compute = round(rank0_total * 2 / 3, 6)
    r0_swmm = round(rank0_total * 1 / 3, 6)
    content += f"0, {r0_compute}, 0, 0, 0, {r0_swmm}, 0, {rank0_total}, 0, {rank0_total}\n"
    # Rank 1: split Total ½ Compute, ½ SWMM.
    r1_compute = round(rank1_total * 1 / 2, 6)
    r1_swmm = round(rank1_total * 1 / 2, 6)
    content += f"1, {r1_compute}, 0, 0, 0, {r1_swmm}, 0, {rank1_total}, 0, {rank1_total}\n"
    # Average row — parse_performance_file expects it and drops it.
    avg_total = (rank0_total + rank1_total) / 2
    content += f"Average, 0, 0, 0, 0, 0, 0, {avg_total}, 0, {avg_total}\n"
    (perf_dir / f"performance{tstep}.txt").write_text(content)


@pytest.fixture
def two_allocation_perf_dir(tmp_path):
    """Build a synthetic ``out_tritonswmm/performance/`` with two allocation batches.

    Allocation 1 (checkpoints 1..5): rank 0 cumulative Total = 15·tstep; rank 1 = 16·tstep.
    Allocation 2 (checkpoints 6..10): TRITON timer reset on resume — cumulative restarts
    at 15·(tstep-5) for rank 0 and 16·(tstep-5) for rank 1. The reset detector in
    ``_aggregate_perf_tseries`` treats the first post-resume row's absolute value as
    the new per-rank cumulative.

    True end-to-end wallclock (max across ranks of summed per-rank deltas):
        Rank 0: 15·5 (pre-resume) + 15·5 (post-resume) = 150
        Rank 1: 16·5 (pre-resume) + 16·5 (post-resume) = 160
        max(Rank) = 160
    """
    perf_dir = tmp_path / "out_tritonswmm" / "performance"
    perf_dir.mkdir(parents=True)
    # Allocation 1: checkpoints 1..5, cumulative growing monotonically.
    for tstep in range(1, 6):
        _write_performance_file(perf_dir, tstep, rank0_total=15 * tstep, rank1_total=16 * tstep)
    # Allocation 2: checkpoints 6..10, TRITON timer reset; cumulative restarts.
    for tstep in range(6, 11):
        post_resume_tstep = tstep - 5
        _write_performance_file(
            perf_dir,
            tstep,
            rank0_total=15 * post_resume_tstep,
            rank1_total=16 * post_resume_tstep,
        )
    return perf_dir


def test_two_allocation_merge_recovers_full_wallclock(two_allocation_perf_dir):
    """With both allocation batches preserved, the aggregator recovers full wallclock.

    The reset detector handles the cumulative-counter drop at the allocation boundary
    (checkpoint 5 → 6: rank 0 cumulative 75 → 15, all-column deltas <= 0). The reset
    detector substitutes the post-resume absolute value as the per-rank delta at the
    reset row, so the per-rank summed deltas equal pre-resume + post-resume work.

    ``max(Rank)`` of the per-rank sums selects rank 1 (slowest at 160s) as the
    end-to-end wallclock.
    """
    from TRITON_SWMM_toolkit.process_simulation import _aggregate_perf_summary

    summary = _aggregate_perf_summary(two_allocation_perf_dir)

    assert summary["Total"].item() == pytest.approx(160.0, rel=1e-6), (
        "With both allocation batches preserved, max(Rank).sum(timestep_min)[Total] "
        "must equal the true end-to-end wallclock (rank 1: 16·5 + 16·5 = 160)."
    )


def test_pre_resume_files_deleted_undercounts_wallclock(two_allocation_perf_dir):
    """With pre-resume batch deleted, the aggregator undercounts wallclock detectably.

    Simulates the mid-workflow ``_clear_raw_TRITON_outputs`` failure mode the
    stipulation forbids: the pre-resume ``performance{1..5}.txt`` files are removed
    before the next allocation runs. ``_aggregate_perf_tseries`` sees only checkpoints
    6..10; its first-checkpoint logic treats the post-resume first checkpoint's
    absolute value as the per-rank cumulative — losing the pre-resume duration.

    Asserts: truncated ``Total`` < full ``Total`` AND truncated ``Total`` equals only
    the post-resume work (rank 1: 16·5 = 80).
    """
    from TRITON_SWMM_toolkit.process_simulation import _aggregate_perf_summary

    # Capture the full summary first as the reference value.
    full_summary = _aggregate_perf_summary(two_allocation_perf_dir)
    full_total = full_summary["Total"].item()

    # Simulate mid-workflow _clear_raw_TRITON_outputs: delete pre-resume batch only.
    for tstep in range(1, 6):
        (two_allocation_perf_dir / f"performance{tstep}.txt").unlink()

    truncated_summary = _aggregate_perf_summary(two_allocation_perf_dir)
    truncated_total = truncated_summary["Total"].item()

    assert truncated_total < full_total, (
        f"Deleting the pre-resume batch must produce a strictly smaller wallclock "
        f"summary; got truncated={truncated_total} vs full={full_total}. If this "
        f"assertion ever fails, the regression silently restored the pre-V0008 "
        f"failure mode (or the aggregator was changed to ignore missing checkpoints "
        f"in a way that mints synthetic wallclock)."
    )
    assert truncated_total == pytest.approx(80.0, rel=1e-6), (
        "With only the post-resume batch present, max(Rank).sum(timestep_min)[Total] "
        "equals only the post-resume work (rank 1: 16·5 = 80); the 80s undercount "
        "vs full (160s) is the pre-resume wallclock lost to mid-workflow "
        "_clear_raw_TRITON_outputs."
    )
