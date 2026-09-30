"""The rank-COHERENT and `min` families, and the two scalars that name the selection.

THE DEFECT THIS PINS. The performance summary publishes one reduction per column,
``max(dim="Rank")``, and ``max`` is applied PER VARIABLE -- so the published children are a
sum of per-column maxima, which is not the cost of any single rank. Measured over the
52-store non-fixture corpus the children overshoot their own ``Simulation`` parent by
17.17 % to 40.80 % of ``Total``, because no one rank attains the maximum of every child. The
coherent family reads every column at the ONE rank attaining ``max(Total)``, so on that row
the children close their level.

WHAT THESE TESTS ASSERT ON. Every assertion is over a RETURNED VALUE -- which rank a column
was read at, whether a value is finite, and the arithmetic that closes the level -- so each
discriminates on behaviour rather than on any string the change introduced.

THE THREE NaN CLASSES ARE TESTED SEPARATELY AND THAT SEPARATION IS THE POINT. The selector
is ``argmax``, which RAISES where the sibling ``max`` degrades, and the two neighbouring
wrong forms fail in opposite directions:

    input class    selector form          result
    no NaN         default (shipped)      the rank attaining max(Total)          correct
    partial NaN    default (shipped)      the highest VALID rank                 correct
    partial NaN    skipna=False           the NaN rank                           WRONG
    all NaN        default, unguarded     ValueError: All-NaN slice encountered  WRONG
    all NaN        default, guarded       NaN, matching the sibling max          correct

The two WRONG rows are pinned as controls in this file, not merely described: a guard that
never fires and a selector whose default is never exercised are both satisfiable by a
no-op, and only the controls separate the shipped form from them.

THE TWO-SITES TEST IS THE LOAD-BEARING ONE. The reduction is DUPLICATED at
``_export_performance_summary`` (inline, instance method) and ``_aggregate_perf_summary``
(module level), deliberately, and the two are kept in step by an in-source comment rather
than by shared code -- so a one-sided edit is invisible at review. This file drives BOTH
over one input and asserts identical output, which converts that invisible drift into a red
test.
"""

from __future__ import annotations

import contextlib
import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_BASE_COLS = ("Compute", "MPI", "IO", "Resize", "SWMM", "Other", "Simulation", "Init", "Total")


def _summed_over_ranks(total, compute):
    """A time-summed per-rank frame in the shape both reduction sites receive.

    Built directly rather than through ``_aggregate_perf_tseries`` because the NaN classes
    below are properties of the RANK axis after the time sum, and routing them through the
    file parser would make the fixture, not the reduction, decide which NaNs appear.
    """
    return xr.Dataset(
        {
            "Total": (("Rank",), np.asarray(total, dtype=float)),
            "Compute": (("Rank",), np.asarray(compute, dtype=float)),
        },
        coords={"Rank": np.arange(len(total))},
    )


def _reduce_inline(summed):
    """Apply the inline site's reduction to an ALREADY-time-summed frame.

    The inline site consumes a timeseries and sums it first; these NaN-class fixtures are
    already summed, so this helper feeds the rank half only. It is a transcription of the
    shipped block and is NOT the equivalence check -- that is
    ``test_both_reduction_sites_agree_on_one_input``, which calls the two production
    functions themselves.
    """
    out = summed.max(dim="Rank")
    rank_pos = xr.DataArray(np.arange(summed.sizes["Rank"]), dims="Rank")
    selectable = summed["Total"].notnull().any(dim="Rank")
    chosen = rank_pos == summed["Total"].fillna(-np.inf).argmax(dim="Rank")
    coherent = summed.where(chosen).max(dim="Rank")
    for name in summed.data_vars:
        out[f"{name}_coherent"] = coherent[name].where(selectable)
        out[f"{name}_min"] = summed[name].min(dim="Rank")
    out["coherent_rank"] = summed["Rank"].where(chosen).max(dim="Rank").where(selectable)
    out["n_ranks"] = xr.full_like(selectable, summed.sizes["Rank"], dtype="int64")
    return out


#: Per-rank, per-unit-timestep rates. Two properties are deliberate and the test is vacuous
#: without either. (1) The six children SUM to ``Simulation`` on each rank and
#: ``Simulation + Init == Total`` on each rank -- the identity the solver itself enforces on
#: every emitted per-rank row, so the fixture is a legal TRITON frame rather than an
#: arbitrary one. (2) DIFFERENT ranks win different columns: rank 0 wins ``Compute`` and
#: ``Init``, rank 1 wins ``MPI`` and ``Total``. A fixture where one rank won everything is
#: the equality case of the sum-of-maxima inequality, and every overshoot assertion below
#: would pass trivially on it.
_RANK_RATES = {
    0: {"Compute": 8.0, "MPI": 2.0, "IO": 0.0, "Resize": 0.0, "SWMM": 0.0, "Other": 0.0, "Init": 1.0},
    1: {"Compute": 3.0, "MPI": 8.0, "IO": 0.0, "Resize": 0.0, "SWMM": 0.0, "Other": 0.0, "Init": 0.5},
}


def _write_perf_file(perf_dir: Path, tstep: int, *, ranks):
    """Write one ``performance{tstep}.txt`` in TRITON's emitted shape."""
    content = "%" + ", ".join(["Rank"] + list(_BASE_COLS)) + "\n"
    for rank in ranks:
        rates = _RANK_RATES[rank]
        simulation = sum(rates[c] for c in ("Compute", "MPI", "IO", "Resize", "SWMM", "Other"))
        row = {c: rates[c] * tstep for c in rates}
        row["Simulation"] = simulation * tstep
        row["Total"] = (simulation + rates["Init"]) * tstep
        content += f"{rank}, " + ", ".join(f"{row[c]}" for c in _BASE_COLS) + "\n"
    # parse_performance_file requires the Average row and drops it.
    content += "Average, " + ", ".join(["0"] * len(_BASE_COLS)) + "\n"
    (perf_dir / f"performance{tstep}.txt").write_text(content)


@pytest.fixture
def two_rank_perf_dir(tmp_path):
    """A plain two-rank member: no resume, no column split, no rank-count change."""
    perf_dir = tmp_path / "out_tritonswmm" / "performance"
    perf_dir.mkdir(parents=True)
    for tstep in range(1, 5):
        _write_perf_file(perf_dir, tstep, ranks=(0, 1))
    return perf_dir


# --------------------------------------------------------------------------------------
# A2 -- the three NaN classes
# --------------------------------------------------------------------------------------


def test_no_nan_reads_every_column_at_the_rank_attaining_max_total():
    """The coherent row is read at ONE rank, and it is not the per-column max row."""
    out = _reduce_inline(_summed_over_ranks([1.0, 2.0, 1.5], [9.0, 5.0, 7.0]))

    assert float(out["coherent_rank"]) == 1.0, "rank 1 attains max(Total)"
    assert float(out["Total_coherent"]) == 2.0
    assert float(out["Compute_coherent"]) == 5.0, (
        "Compute must be read at the COHERENT rank (5.0), not at its own argmax (9.0) -- "
        "a coherent family that reported 9.0 here would be the defect under a new name"
    )
    assert float(out["Compute"]) == 9.0, "the published max is untouched"
    assert float(out["Compute_min"]) == 5.0
    assert int(out["n_ranks"]) == 3


def test_partial_nan_selects_the_highest_valid_rank_not_the_nan_rank():
    """The default `skipna` is CORRECT for the partial-NaN class, and the control proves it."""
    summed = _summed_over_ranks([1.0, np.nan, 2.0], [9.0, 5.0, 7.0])
    out = _reduce_inline(summed)

    assert float(out["coherent_rank"]) == 2.0, "rank 2 is the highest VALID Total"
    assert float(out["Compute_coherent"]) == 7.0, "read at rank 2, the rank actually chosen"

    # CONTROL -- the named trap. Forcing `skipna=False` onto the selector returns the NaN
    # rank, which would read the whole coherent row at a rank the selector did not choose.
    assert int(summed["Total"].argmax(dim="Rank", skipna=False)) == 1
    assert int(summed["Total"].argmax(dim="Rank")) == 2
    assert math.isnan(float(summed["Total"].isel(Rank=1))), "rank 1 is the NaN rank"


def test_all_nan_degrades_to_nan_exactly_as_the_sibling_max_does():
    """The guard's whole job: degrade, do not raise."""
    summed = _summed_over_ranks([np.nan, np.nan, np.nan], [9.0, 5.0, 7.0])
    out = _reduce_inline(summed)

    assert math.isnan(float(out["Total"])), "the sibling max degrades to NaN"
    assert math.isnan(float(out["Total_coherent"])), "and so must the coherent value"
    assert math.isnan(float(out["Compute_coherent"])), (
        "Compute is finite at every rank, so an unmasked family would publish a finite "
        "number read at a rank `fillna` picked arbitrarily -- that is the silent wrong "
        "answer the mask exists to prevent"
    )
    assert math.isnan(float(out["coherent_rank"])), "no rank was selectable"
    assert int(out["n_ranks"]) == 3, "the rank axis length is still known"

    # CONTROL -- the guard is load-bearing, not decorative: unguarded, this input raises.
    with pytest.raises(ValueError, match="All-NaN slice"):
        summed["Total"].argmax(dim="Rank")


# --------------------------------------------------------------------------------------
# A3 -- the two duplicated sites
# --------------------------------------------------------------------------------------


class _CapturedWrite:
    """A duck-typed `self` for the inline site, capturing what it would have written.

    The inline reduction is an instance method on a class whose real construction needs a
    scenario, an analysis and a system. Nothing in the reduction touches any of those, so
    this stub supplies exactly the four members the method body reaches for -- which is also
    what keeps this test off the compile-bearing path.
    """

    def __init__(self):
        self.captured: xr.Dataset | None = None
        self.log = self

    # -- the `_deferred_log_writes` decorator's requirement
    @contextlib.contextmanager
    def deferred_writes(self):
        yield

    def add_sim_processing_entry(self, *args, **kwargs):
        return None

    def set(self, value):  # the `log_field` stand-in
        return None

    # -- the method body's own gates
    def _already_written(self, f_out):
        return False

    def _summary_usable(self, f_out):
        return True

    def _write_output(self, ds, f_out, compression_level, verbose, mode):
        self.captured = ds
        # `get_file_size_MiB` rglobs a `.zarr` path, so the directory must exist.
        Path(f_out).mkdir(parents=True, exist_ok=True)
        (Path(f_out) / ".zattrs").write_text("{}")


def _run_inline_site(ds, tmp_path):
    from hhemt.process_simulation import TRITONSWMM_sim_post_processing

    stub = _CapturedWrite()
    TRITONSWMM_sim_post_processing._export_performance_summary(
        stub,
        ds=ds,
        fname_out=tmp_path / "captured_summary.zarr",
        compression_level=5,
        verbose=False,
        log_field=stub,
        mode="tritonswmm_performance",
        verify_present=True,
    )
    assert stub.captured is not None, "the inline site did not reach its write"
    return stub.captured


def _strip_attrs(ds):
    out = ds.copy()
    out.attrs = {}
    for name in out.variables:
        out[name].attrs = {}
    return out


def test_both_reduction_sites_agree_on_one_input(two_rank_perf_dir, tmp_path):
    """The duplication's only defence. A one-sided edit fails HERE and nowhere else."""
    from hhemt.process_simulation import _aggregate_perf_summary, _aggregate_perf_tseries

    via_module = _aggregate_perf_summary(two_rank_perf_dir, resume_steps=[])
    via_inline = _run_inline_site(_aggregate_perf_tseries(two_rank_perf_dir, resume_steps=[]), tmp_path)

    assert set(via_module.data_vars) == set(via_inline.data_vars), (
        "the two sites emit different variable SETS -- they have drifted"
    )
    xr.testing.assert_identical(_strip_attrs(via_module), _strip_attrs(via_inline))


def test_the_family_is_present_and_the_coherent_row_closes_its_level(two_rank_perf_dir):
    """The property the family exists for, measured end to end through production code."""
    from hhemt.process_simulation import _aggregate_perf_summary

    out = _aggregate_perf_summary(two_rank_perf_dir, resume_steps=[])

    for base in _BASE_COLS:
        assert f"{base}_coherent" in out.data_vars, f"{base}_coherent missing"
        assert f"{base}_min" in out.data_vars, f"{base}_min missing"
    assert "coherent_rank" in out.data_vars
    assert "n_ranks" in out.data_vars
    assert int(out["n_ranks"]) == 2

    # `Total` at argmax(Total) IS max(Total) -- identically, so no published figure moves.
    assert float(out["Total_coherent"]) == float(out["Total"])

    children = ("Compute", "MPI", "IO", "Resize", "SWMM", "Other")
    coherent_sum = sum(float(out[f"{c}_coherent"]) for c in children)
    assert coherent_sum == pytest.approx(float(out["Simulation_coherent"]), rel=1e-9), (
        "the coherent children must close their own parent -- that is the whole claim"
    )
    # And the published max row does NOT close, which is the defect being repaired.
    published_sum = sum(float(out[c]) for c in children)
    assert published_sum > float(out["Simulation"]), (
        "on this fixture the per-column maxima must overshoot; a fixture where they did "
        "not would make the coherent assertion above vacuous"
    )


def test_no_published_variable_name_or_value_moves(two_rank_perf_dir):
    """Strict additivity: every pre-existing name keeps the value it had."""
    from hhemt.process_simulation import _aggregate_perf_summary, _aggregate_perf_tseries

    summed = _aggregate_perf_tseries(two_rank_perf_dir, resume_steps=[]).sum(dim="timestep_min", skipna=False)
    legacy = summed.max(dim="Rank")  # the pre-change reduction, verbatim
    out = _aggregate_perf_summary(two_rank_perf_dir, resume_steps=[])

    for name in legacy.data_vars:
        assert name in out.data_vars
        assert float(out[name]) == float(legacy[name]), f"{name} moved"
    assert "Rank" not in out.coords, (
        "retaining a Rank carrier on the summary is a separate, later package's decision; "
        "this change must not introduce one as a side effect"
    )


def test_a_lazy_dask_backed_input_produces_the_same_family(two_rank_perf_dir, tmp_path):
    """Why the selection is a mask-and-reduce rather than `isel(Rank=...)`.

    The inline site receives a zarr-backed (lazy) dataset while the module-level site
    receives a numpy-backed one. Vectorized `isel` with a lazy indexer raises
    `IndexError: vindex does not support indexing with dask objects`, so an `isel`-based
    selector would work at one duplicated site and fail at the other.
    """
    from hhemt.process_simulation import _aggregate_perf_tseries

    eager = _aggregate_perf_tseries(two_rank_perf_dir, resume_steps=[])
    store = tmp_path / "tseries.zarr"
    eager.to_zarr(store, mode="w", consolidated=False)
    lazy = xr.open_zarr(store, consolidated=False)
    assert lazy["Total"].chunks is not None, "the lazy arm must actually be chunked"

    from_lazy = _run_inline_site(lazy, tmp_path / "lazy_out")
    from_eager = _run_inline_site(eager, tmp_path / "eager_out")
    xr.testing.assert_allclose(_strip_attrs(from_lazy).compute(), _strip_attrs(from_eager))
