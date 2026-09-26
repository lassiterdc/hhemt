"""A column emitted for only SOME of a member's allocations must reduce to NaN.

THE DEFECT THIS PINS. `_aggregate_perf_tseries`'s `pd.concat` OUTER-joins the
per-allocation frames and NaN-fills any column an allocation never emitted, which is
exactly what a member spanning a solver rebuild across its own allocations produces.
Under xarray's default `skipna=True` the summary reduction then turns that partly-present
column into a FINITE, UNDERSTATED number rather than into NaN -- a wrong value on a
deposited artifact, and one the only non-finite-raising consumer
(`report_renderers/sensitivity_benchmarking.py`) structurally cannot catch, because the
value it receives is an ordinary float.

WHAT THESE TESTS ASSERT ON, and why it is not the wording. Every assertion here is over a
RETURNED VALUE -- finite versus not, and the arithmetic of the truthful columns -- so each
one discriminates on behaviour in both the pre-fix and post-fix worlds rather than on any
string the fix introduced.

MEASURED THREE-ARM DIFFERENTIAL, pasted rather than predicted, because the middle arm is
what keeps the fix from being satisfied by over-firing:

    arm                                          result
    pre-fix (both reductions skipna=True)        3 failed, 2 passed
    over-broad (both reductions skipna=False)    1 failed, 4 passed
    shipped (skipna=False on the TIME sum only)  5 passed

THE CONTROLS ARE THE POINT OF THIS FILE AS MUCH AS THE DEFECT IS. The symmetric-looking
fix -- `skipna=False` on BOTH the time sum and the rank max -- also repairs the short
column, and it additionally turns the HEADLINE `Total` into NaN on any member whose
allocations ran different rank counts, because `to_xarray` pads the rank axis and that
padding is a reshape artefact rather than a data defect. That is the single failure in the
over-broad arm.

A RAGGED RANK COUNT IS A BEHAVIOUR CHANGE THIS FIX MAKES BEYOND THE SHORT-COLUMN DEFECT,
and it is named here rather than left to be discovered. On a member whose allocations ran
different rank counts, the pre-fix reduction reports the largest per-rank sum among ALL
ranks including one that existed for only part of the run -- it treats that rank's absence
as zero cost. Under the shipped form such a rank sums to NaN and drops out of the max, so
the reported figure becomes the slowest rank that spanned the WHOLE run. Measured on the
fixture below: 104 pre-fix, 88 shipped. 88 is the defensible wallclock and 104 is not, but
the number on the artifact does change for that shape, which is why the ragged-rank test
fails under the pre-fix arm as well as the over-broad one and passes only under the
shipped one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

#: The coupled-timer decomposition a solver rebuild introduces mid-member. Absent from the
#: older build's header, present in the newer one's -- the concrete shape behind every
#: "partly-present column" in this file.
SPLIT_COLS = ("SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER")

_BASE_COLS = ("Compute", "MPI", "IO", "Resize", "SWMM", "Other", "Simulation", "Init", "Total")


def _write_perf_file(perf_dir, tstep, *, ranks, cumulative_total, extra_cols=()):
    """Write one `performance{tstep}.txt` in TRITON's emitted shape.

    `extra_cols` is what makes a header differ between allocations: the older build omits
    them entirely, so the file it wrote has a strictly narrower header, which is the
    condition `pd.concat` NaN-fills.
    """
    cols = list(_BASE_COLS) + list(extra_cols)
    content = "%" + ", ".join(["Rank"] + cols) + "\n"
    for rank in ranks:
        total = cumulative_total(rank, tstep)
        values = []
        for col in cols:
            if col in ("Total", "Simulation"):
                values.append(f"{total}")
            elif col == "Compute":
                values.append(f"{round(total / 2, 6)}")
            elif col in SPLIT_COLS:
                # A small, nonzero, per-column-distinct value: a zero here would make the
                # understated sum and the truthful sum numerically identical and the test
                # would pass under the defect.
                values.append(f"{round(total / 10, 6)}")
            else:
                values.append("0")
        content += f"{rank}, " + ", ".join(values) + "\n"
    # parse_performance_file requires the Average row and drops it.
    content += "Average, " + ", ".join(["0"] * len(cols)) + "\n"
    (perf_dir / f"performance{tstep}.txt").write_text(content)


@pytest.fixture
def rebuild_midway_perf_dir(tmp_path):
    """A member whose allocation 1 predates the column split and whose allocation 2 follows it.

    Checkpoints 1..4 carry the narrow header; 5..8 carry the split columns. Cumulative
    `Total` grows monotonically across the whole run (no resume reset), so the ledger is
    empty and the truthful `Total` is a plain telescope -- which keeps the truthful column
    and the short column independently checkable in one fixture.
    """
    perf_dir = tmp_path / "out_tritonswmm" / "performance"
    perf_dir.mkdir(parents=True)

    def cumulative(rank, tstep):
        return (10 + rank) * tstep

    for tstep in range(1, 5):
        _write_perf_file(perf_dir, tstep, ranks=(0, 1), cumulative_total=cumulative)
    for tstep in range(5, 9):
        _write_perf_file(perf_dir, tstep, ranks=(0, 1), cumulative_total=cumulative, extra_cols=SPLIT_COLS)
    return perf_dir


@pytest.fixture
def uniform_perf_dir(tmp_path):
    """The same member with NO rebuild: every allocation emits the split columns."""
    perf_dir = tmp_path / "out_tritonswmm" / "performance"
    perf_dir.mkdir(parents=True)

    def cumulative(rank, tstep):
        return (10 + rank) * tstep

    for tstep in range(1, 9):
        _write_perf_file(perf_dir, tstep, ranks=(0, 1), cumulative_total=cumulative, extra_cols=SPLIT_COLS)
    return perf_dir


def test_a_partly_emitted_column_reduces_to_nan_not_an_understated_number(rebuild_midway_perf_dir):
    """The defect itself. Pre-fix this column arrives finite; post-fix it arrives NaN.

    A finite value here is not a near-miss -- it is a wrong number on a published artifact
    that reads as a measurement, and every downstream guard treats it as one.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    summary = _aggregate_perf_summary(rebuild_midway_perf_dir, resume_steps=[])

    for col in SPLIT_COLS:
        assert col in summary.data_vars, f"{col} was emitted by allocation 2, so it must be present"
        value = float(summary[col].item())
        assert math.isnan(value), (
            f"{col} was emitted for only 4 of 8 reporting steps, so its cumulative is UNKNOWN. "
            f"It reduced to {value!r} instead, which is the pre-fix understated sum: skipna "
            "treated the four never-emitted steps as zero."
        )


def test_the_truthful_columns_are_unaffected_by_the_short_ones(rebuild_midway_perf_dir):
    """The fix must not spread NaN to columns every allocation emitted.

    This is the arm an over-broad `skipna=False` on both reductions still passes, so it is
    necessary rather than sufficient on its own -- the ragged-rank test below is the one
    that separates them.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    summary = _aggregate_perf_summary(rebuild_midway_perf_dir, resume_steps=[])

    total = float(summary["Total"].item())
    # Rank 1 is the slowest (cumulative 11*tstep), no reset, so the telescope is its final
    # cumulative at checkpoint 8.
    assert total == pytest.approx(11 * 8, rel=1e-9), (
        "'Total' is present in every allocation's header, so the short-column repair must "
        f"leave it exactly truthful; got {total!r}"
    )
    assert math.isfinite(float(summary["Compute"].item()))


def test_a_wholly_absent_column_stays_absent_rather_than_becoming_nan(uniform_perf_dir):
    """The two absence cases must stay distinguishable on the artifact.

    A column NO allocation emitted has no entry at all; a column SOME allocations emitted
    is NaN. Collapsing the two would make the `notes` attr unwritable truthfully and would
    leave a reader unable to tell "not measured" from "measured inconsistently".
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    summary = _aggregate_perf_summary(uniform_perf_dir, resume_steps=[])

    assert "SWMM_STEP" in summary.data_vars
    assert "NO_SUCH_COLUMN" not in summary.data_vars
    for col in SPLIT_COLS:
        value = float(summary[col].item())
        assert math.isfinite(value), (
            f"{col} was emitted by EVERY allocation here, so it must reduce to a real "
            f"number; got {value!r}. A NaN would mean the repair over-fired."
        )


def test_a_ragged_rank_count_does_not_nan_the_headline_total(tmp_path):
    """THE DISCRIMINATING CONTROL: rank padding is a reshape artefact, not missing data.

    When allocations ran different rank counts, `deltas.to_xarray()` pads the rank axis and
    the ranks that did not exist in an allocation read NaN for its timesteps. Propagating
    NaN through the RANK reduction as well as the time one converts that artefact into a
    NaN `Total` on a sound member -- which the non-finite raise in
    `report_renderers/sensitivity_benchmarking.py` then turns into a hard render failure.

    Measured: this test fails under BOTH wrong forms and passes only under the shipped one.
    Under the over-broad form `Total` is NaN and the finiteness assertion fails. Under the
    PRE-FIX form `Total` is finite but reads 104 rather than 88 -- the largest per-rank sum
    among all ranks, including rank 3, which existed for only the second allocation and
    whose absence from the first is being counted as zero cost. The shipped form drops such
    a rank to NaN, so the max is taken over the ranks that ran throughout. It is the only
    assertion in this file that separates the shipped form from the over-broad one.
    """
    from hhemt.process_simulation import _aggregate_perf_summary

    perf_dir = tmp_path / "out_tritonswmm" / "performance"
    perf_dir.mkdir(parents=True)

    def cumulative(rank, tstep):
        return (10 + rank) * tstep

    # Allocation 1 ran 2 ranks; allocation 2 ran 4. No header change and no reset.
    for tstep in range(1, 5):
        _write_perf_file(perf_dir, tstep, ranks=(0, 1), cumulative_total=cumulative)
    for tstep in range(5, 9):
        _write_perf_file(perf_dir, tstep, ranks=(0, 1, 2, 3), cumulative_total=cumulative)

    summary = _aggregate_perf_summary(perf_dir, resume_steps=[])

    total = float(summary["Total"].item())
    assert math.isfinite(total), (
        "A member whose allocations ran different rank counts is SOUND -- the rank axis is "
        "padded by the reshape, not by missing measurement. A NaN 'Total' here means the "
        "repair propagated NaN through the rank reduction as well as the time one."
    )
    # And it is the slowest rank that ran throughout, not a padded one.
    assert total == pytest.approx(11 * 8, rel=1e-9)


def test_the_short_column_survives_a_zarr_round_trip_as_nan(rebuild_midway_perf_dir, tmp_path):
    """Closes review gap 1: every other test in this area stops at an in-memory object.

    The acceptance condition is about a PUBLISHED artifact, so the NaN has to survive the
    project's own encoding path and read back as NaN rather than as a fill value that a
    consumer would see as a number.
    """
    from hhemt.process_simulation import _aggregate_perf_summary
    from hhemt.utils import return_dic_zarr_encodings

    summary = _aggregate_perf_summary(rebuild_midway_perf_dir, resume_steps=[])
    store = tmp_path / "perf_summary.zarr"
    summary.to_zarr(store, mode="w", encoding=return_dic_zarr_encodings(summary), consolidated=False)

    import xarray as xr

    reopened = xr.open_dataset(store, engine="zarr", consolidated=False)
    try:
        assert np.isnan(float(reopened["SWMM_STEP"].values.item())), (
            "the short column read back as a number after the round trip, so the artifact "
            "on disk does not carry the unknown-ness the in-memory object did"
        )
        assert math.isfinite(float(reopened["Total"].values.item()))
    finally:
        reopened.close()
