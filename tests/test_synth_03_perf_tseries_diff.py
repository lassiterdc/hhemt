"""Phase 1 regression test — assert per-rank diff aggregation is correct.

The pre-V0008 bug at ``process_simulation.py:435`` used ``pd.diff()`` on a
``(timestep_min, Rank)`` MultiIndex without grouping by Rank, scrambling
per-rank deltas via inter-rank-boundary crossing.

This test:

  1. Generates a synthetic 2-rank performance{N}.txt time series with known
     per-rank cumulative values that grow monotonically.
  2. Runs the module-level ``_aggregate_perf_tseries`` / ``_aggregate_perf_summary``
     helpers (introduced in Phase 1 Spec 8) against the synthetic data.
  3. Asserts that per-rank ``sum(timestep_min)`` of the corrected deltas equals
     the final cumulative per rank — proving the diff stayed within each rank.
  4. Asserts that ``max(Rank)`` of the summary equals the slowest-rank
     cumulative — proving the aggregation is wallclock-semantic.

The helpers are called directly (no analysis-instance harness needed) per Spec 8's
architectural constraint that V0008 + this regression test share one source of truth
with the production aggregator.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def synthetic_perf_dir(tmp_path):
    """Build a synthetic ``out_tritonswmm/performance/`` directory with 2 ranks × 10 checkpoints."""
    perf_dir = tmp_path / "out_tritonswmm" / "performance"
    perf_dir.mkdir(parents=True)
    # Cumulative values per rank, monotonically growing. Choose distinct per-rank
    # values so the test can distinguish per-rank correctness vs cross-rank
    # contamination.
    for tstep in range(1, 11):
        content = "%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total\n"
        # Rank 0: each checkpoint adds 10s Compute, 5s SWMM (Total = 15s/checkpoint).
        content += f"0, {10 * tstep}, 0, 0, 0, {5 * tstep}, 0, {15 * tstep}, 0, {15 * tstep}\n"
        # Rank 1: each checkpoint adds 12s Compute, 4s SWMM (Total = 16s/checkpoint).
        content += f"1, {12 * tstep}, 0, 0, 0, {4 * tstep}, 0, {16 * tstep}, 0, {16 * tstep}\n"
        # Average row — parse_performance_file expects this row to exist and drops it.
        content += f"Average, {11 * tstep}, 0, 0, 0, {4.5 * tstep}, 0, {15.5 * tstep}, 0, {15.5 * tstep}\n"
        (perf_dir / f"performance{tstep}.txt").write_text(content)
    return perf_dir


def test_per_rank_diff_aggregation_is_correct(synthetic_perf_dir):
    """Verify ``max(Rank)`` of summed-deltas equals the slowest-rank final cumulative.

    Synthetic data: rank 0 cumulative grows 10s/checkpoint × 10 checkpoints = 100s
    Compute, 50s SWMM, 150s Total. Rank 1: 120s Compute, 40s SWMM, 160s Total.
    ``max(Rank).sum(timestep_min)`` of correctly-diffed deltas selects the slowest
    rank per column.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    summary = _aggregate_perf_summary(synthetic_perf_dir)

    assert summary["Total"].item() == pytest.approx(160.0, rel=1e-6), (
        "Rank-1 cumulative Total at checkpoint 10 is 160s; max(Rank).sum(timestep_min) "
        "of correctly-diffed deltas must equal this."
    )
    assert summary["Compute"].item() == pytest.approx(120.0, rel=1e-6), (
        "Rank-1 Compute at checkpoint 10 is 120s; max(Rank) selects rank-1."
    )
    assert summary["SWMM"].item() == pytest.approx(50.0, rel=1e-6), (
        "Rank-0 SWMM = 50s; max(Rank) selects rank-0 because rank-0 > rank-1 SWMM."
    )


def test_corrected_reconstruction_matches_final_performance_txt(synthetic_perf_dir):
    """Cross-validate per-rank deltas sum-equal the final cumulative per rank."""
    from hhemt.process_simulation import _aggregate_perf_tseries

    ds = _aggregate_perf_tseries(synthetic_perf_dir)
    rank0_total = ds["Total"].sel(Rank=0).sum(dim="timestep_min").item()
    rank1_total = ds["Total"].sel(Rank=1).sum(dim="timestep_min").item()
    assert rank0_total == pytest.approx(150.0, rel=1e-6), (
        "rank-0 final Total at checkpoint 10 = 15s/checkpoint × 10 = 150s"
    )
    assert rank1_total == pytest.approx(160.0, rel=1e-6), (
        "rank-1 final Total at checkpoint 10 = 16s/checkpoint × 10 = 160s"
    )


def test_reset_detector_handles_resume(synthetic_perf_dir):
    """Inject a non-monotonic checkpoint and assert reset is detected per-rank.

    Overwrite the last checkpoint's content with smaller values to simulate a resume
    that resets the counter; the corrected aggregator's reset-detector branch should
    treat the post-reset row's absolute value as the new cumulative for that rank.
    """
    from hhemt.process_simulation import _aggregate_perf_tseries

    reset_tstep_file = synthetic_perf_dir / "performance10.txt"
    content = "%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total\n"
    content += "0, 5, 0, 0, 0, 3, 0, 8, 0, 8\n"  # reset: smaller than prev rank-0 row
    content += "1, 6, 0, 0, 0, 2, 0, 8, 0, 8\n"  # reset: smaller than prev rank-1 row
    content += "Average, 5.5, 0, 0, 0, 2.5, 0, 8, 0, 8\n"
    reset_tstep_file.write_text(content)

    ds = _aggregate_perf_tseries(synthetic_perf_dir)
    # At the reset row, the absolute value IS the new cumulative for that rank.
    assert ds["Total"].sel(Rank=0, timestep_min=10).item() == pytest.approx(8.0, rel=1e-6)
    assert ds["Total"].sel(Rank=1, timestep_min=10).item() == pytest.approx(8.0, rel=1e-6)
