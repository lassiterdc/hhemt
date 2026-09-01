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


def test_zero_byte_perf_file_is_skipped(synthetic_perf_dir):
    """A 0-byte ``performance{N}.txt`` (left by a hard kill / SIGKILL / the
    deterministic resume harness — the perf dump for a checkpoint created but killed
    before any row was written) must be SKIPPED with a ``UserWarning``, not crash the
    aggregator.

    Pre-fix behavior (the failure this guards): ``parse_performance_file`` ->
    ``pandas.read_csv`` raises ``EmptyDataError`` on the empty file and the whole
    process rule fails (observed 2026-07-28: ``synth_cc_resume_triton`` ``member_gpu_1_r1``
    left ``performance110.txt`` at 0 bytes of 144). The FIRST
    ``_aggregate_perf_tseries`` call below therefore RAISES against pre-fix code — the
    assertion anchors on the raise/return behavior (true in both pre- and post-fix
    worlds), so it discriminates on behavior, not on the warning wording.
    """
    import warnings

    from hhemt.process_simulation import _aggregate_perf_tseries

    # A hard kill leaves a 0-byte perf dump alongside the 10 valid checkpoints.
    (synthetic_perf_dir / "performance11.txt").write_text("")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = _aggregate_perf_tseries(synthetic_perf_dir, resume_steps=[])  # raises EmptyDataError pre-fix

    # The empty file is skipped: the series covers exactly the 10 valid checkpoints.
    assert ds.sizes["timestep_min"] == 10
    # A UserWarning naming the skipped empty file(s) was emitted (mirrors the
    # existing perfs-with-negatives warning pattern).
    assert any(issubclass(w.category, UserWarning) and "empty" in str(w.message).lower() for w in caught)


def test_malformed_perf_file_is_skipped(synthetic_perf_dir):
    """A concurrent-writer-interleaved ``performance{N}.txt`` must be SKIPPED with a
    ``UserWarning``, not crash the aggregator.

    Two real shapes, both measured on Rivanna in the ``synth_sensitivity`` fixture
    corpus (2026-08-23):

      (a) a trailing numeric fragment after the ``Average`` row -- 10 of 1080 files in
          ``member_0``, e.g. ``performance201.txt`` ending in a bare ``.6453`` line.
          ``read_csv`` NaN-pads it into a row and ``df_ranks["Rank"].astype(int)``
          raises ``ValueError: invalid literal for int() with base 10: '.6453'``.
      (b) a mangled ``Average`` row -- ``performance911.txt`` in the ``member_2`` set-aside
          tree reads ``AAverage``, leaving no ``Average`` sentinel, so
          ``df[df["Rank"] == "Average"].iloc[0]`` raises
          ``IndexError: single positional indexer is out-of-bounds``.

    Pre-fix, EITHER shape fails the whole process rule, which costs the member
    its ``d_process`` flag. The assertions below anchor on raise/return behavior, not
    on warning wording, so they discriminate on behavior in both pre- and post-fix
    worlds.
    """
    import warnings

    from hhemt.process_simulation import _aggregate_perf_tseries

    # (a) trailing-fragment interleave, appended to an otherwise valid checkpoint.
    synthetic_perf_dir.joinpath("performance11.txt").write_text(
        "%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total\n"
        "0, 110, 0, 0, 0, 55, 0, 165, 0, 165\n"
        "1, 132, 0, 0, 0, 44, 0, 176, 0, 176\n"
        "Average, 121, 0, 0, 0, 49.5, 0, 170.5, 0, 170.5\n"
        ".6453\n"
    )
    # (b) Average-row interleave, which consumes the sentinel token entirely.
    synthetic_perf_dir.joinpath("performance12.txt").write_text(
        "%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total\n"
        "0, 120, 0, 0, 0, 60, 0, 180, 0, 180\n"
        "1, 144, 0, 0, 0, 48, 0, 192, 0, 192\n"
        "AAverage, 132, 0, 0, 0, 54, 0, 186, 0, 186\n"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = _aggregate_perf_tseries(synthetic_perf_dir, resume_steps=[])  # raises ValueError / IndexError pre-fix

    # Both malformed files are skipped: the series covers exactly the 10 valid
    # checkpoints built by the fixture.
    assert ds.sizes["timestep_min"] == 10
    # The skipped checkpoints do not leak into the reset-row join's iloc set: the
    # per-rank sum still telescopes to the final cumulative of checkpoint 10.
    assert ds["Total"].sum(dim="timestep_min").max(dim="Rank").item() == pytest.approx(160.0, rel=1e-6)
    # A UserWarning naming the malformed file(s) was emitted, and it carries the
    # DIAGNOSIS (concurrent writers), not merely the fact of a skip -- this artifact is
    # the only place duplicate execution is visible, so a generic message would read as
    # noise and delete the detector.
    assert any(issubclass(w.category, UserWarning) and "malformed" in str(w.message).lower() for w in caught)
    assert any(issubclass(w.category, UserWarning) and "concurrent" in str(w.message).lower() for w in caught)


def test_per_rank_diff_aggregation_is_correct(synthetic_perf_dir):
    """Verify ``max(Rank)`` of summed-deltas equals the slowest-rank final cumulative.

    Synthetic data: rank 0 cumulative grows 10s/checkpoint × 10 checkpoints = 100s
    Compute, 50s SWMM, 150s Total. Rank 1: 120s Compute, 40s SWMM, 160s Total.
    ``max(Rank).sum(timestep_min)`` of correctly-diffed deltas selects the slowest
    rank per column.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    summary = _aggregate_perf_summary(synthetic_perf_dir, resume_steps=[])

    assert summary["Total"].item() == pytest.approx(160.0, rel=1e-6), (
        "Rank-1 cumulative Total at checkpoint 10 is 160s; max(Rank).sum(timestep_min) "
        "of correctly-diffed deltas must equal this."
    )
    assert summary["Compute"].item() == pytest.approx(
        120.0, rel=1e-6
    ), "Rank-1 Compute at checkpoint 10 is 120s; max(Rank) selects rank-1."
    assert summary["SWMM"].item() == pytest.approx(
        50.0, rel=1e-6
    ), "Rank-0 SWMM = 50s; max(Rank) selects rank-0 because rank-0 > rank-1 SWMM."


def test_corrected_reconstruction_matches_final_performance_txt(synthetic_perf_dir):
    """Cross-validate per-rank deltas sum-equal the final cumulative per rank."""
    from hhemt.process_simulation import _aggregate_perf_tseries

    ds = _aggregate_perf_tseries(synthetic_perf_dir, resume_steps=[])
    rank0_total = ds["Total"].sel(Rank=0).sum(dim="timestep_min").item()
    rank1_total = ds["Total"].sel(Rank=1).sum(dim="timestep_min").item()
    assert rank0_total == pytest.approx(
        150.0, rel=1e-6
    ), "rank-0 final Total at checkpoint 10 = 15s/checkpoint × 10 = 150s"
    assert rank1_total == pytest.approx(
        160.0, rel=1e-6
    ), "rank-1 final Total at checkpoint 10 = 16s/checkpoint × 10 = 160s"


def _write_resume_boundary(perf_dir):
    """Rewrite checkpoints 9 and 10 so the boundary has PRODUCTION shape.

    The original fixture wrote Init = 0 on every row, which made the reset row's Init
    delta exactly 0 -- satisfying `<= 0` -- so the retired inferred predicate
    `(deltas <= 0).all(axis=1)` detected this boundary. That is precisely the degeneracy
    that made the old test pass while production failed: at a REAL boundary the restarted
    process re-pays initialization and Init INCREASES (measured 0.05694 -> 0.07816 s on
    member_serial_6_r1), one positive column defeats `.all()`, and the reset is missed.

    Giving checkpoint 9 a nonzero Init and checkpoint 10 a LARGER one reproduces that
    shape, so the retired predicate and the ledger join now DISAGREE on this data. That
    disagreement is what lets these tests pin which mechanism ran -- with the old fixture
    both mechanisms produced the same answer and the test could not tell them apart.
    """
    # Checkpoint 9: as the shared fixture writes it, but with a nonzero Init so the
    # boundary's Init delta can be positive rather than trivially zero.
    pre = "%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total\n"
    pre += "0, 90, 0, 0, 0, 45, 0, 135, 0.3, 135\n"
    pre += "1, 108, 0, 0, 0, 36, 0, 144, 0.3, 144\n"
    pre += "Average, 99, 0, 0, 0, 40.5, 0, 139.5, 0.3, 139.5\n"
    (perf_dir / "performance9.txt").write_text(pre)

    # Checkpoint 10: the resumed attempt. Wallclock columns collapse (timer restart) but
    # Init RISES 0.3 -> 0.5, which is what defeats the retired all-column conjunction.
    post = "%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total\n"
    post += "0, 5, 0, 0, 0, 3, 0, 8, 0.5, 8\n"
    post += "1, 6, 0, 0, 0, 2, 0, 8, 0.5, 8\n"
    post += "Average, 5.5, 0, 0, 0, 2.5, 0, 8, 0.5, 8\n"
    (perf_dir / "performance10.txt").write_text(post)


def test_ledger_named_reset_row_is_corrected_despite_rising_init(synthetic_perf_dir):
    """The ledger-named row is corrected even though Init INCREASES across the boundary.

    This is the differential the retired predicate cannot pass. Pre-fix,
    `(deltas <= 0).all(axis=1)` sees Init +0.2 at checkpoint 10, declines to call it a
    reset, and records Total as 8 - 135 = -127 -- a large negative summed as elapsed
    time. Post-fix the boundary comes from the ledger, not from the data's shape, so the
    row is corrected and its absolute value IS the new cumulative.
    """
    from hhemt.process_simulation import _aggregate_perf_tseries

    _write_resume_boundary(synthetic_perf_dir)

    # Allocation 2 resumed FROM checkpoint 9, so the reset row is the first surviving
    # index above it -- checkpoint 10.
    ds = _aggregate_perf_tseries(synthetic_perf_dir, resume_steps=[9])

    assert ds["Total"].sel(Rank=0, timestep_min=10).item() == pytest.approx(8.0, rel=1e-6), (
        "At a ledger-named reset row the absolute value IS the new cumulative. A result "
        "near -127 means the row was diffed against the pre-resume cumulative, i.e. the "
        "boundary was not honoured."
    )
    assert ds["Total"].sel(Rank=1, timestep_min=10).item() == pytest.approx(8.0, rel=1e-6)
    # Init is the column that defeats the retired predicate, so assert it explicitly:
    # the correction must take the absolute (0.5), NOT the +0.2 delta.
    assert ds["Init"].sel(Rank=0, timestep_min=10).item() == pytest.approx(0.5, rel=1e-6), (
        "Init at the reset row must be the restarted attempt's absolute re-init cost "
        "(0.5), not the meaningless +0.2 delta against the previous attempt."
    )


def test_reset_is_not_corrected_when_the_ledger_is_empty(synthetic_perf_dir):
    """Identical bytes, empty ledger -> NO correction. This is what pins the mechanism.

    Same data as the test above; only the ledger differs. If the aggregator still
    corrected checkpoint 10 here, something in it would be inferring resets from the
    data -- which is the behaviour the ledger-driven design exists to remove. No
    data-driven mechanism can satisfy both this test and the one above, because the
    bytes they read are the same.
    """
    from hhemt.process_simulation import _aggregate_perf_tseries

    _write_resume_boundary(synthetic_perf_dir)

    ds = _aggregate_perf_tseries(synthetic_perf_dir, resume_steps=[])

    assert ds["Total"].sel(Rank=0, timestep_min=10).item() == pytest.approx(-127.0, rel=1e-6), (
        "With no ledger entry the row is a plain diff (8 - 135 = -127). A value of 8.0 "
        "here means a reset was inferred from the data despite an empty ledger."
    )
