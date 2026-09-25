"""WP-2A: hhemt consumption of TRITON's split SWMM timing columns.

Environment-independent. These exercise the `perf_*` mint (`analysis._perf_row_from_dataset`),
the `PERF_VARS` / `PERF_VARS_ORDERED` contract against the solver's emitted header, and the
CF attribute coverage of the performance summary -- all against synthetic `xr.Dataset`
objects, so none of them compiles or runs TRITON-SWMM.

WHY A MIXED CORPUS IS THE SUBJECT. `PERF_VARS` is a hardcoded list and
`_get_performance_summary_row` indexes the performance summary dataset by it, so a member
produced before a solver-side column addition is indexed for names it does not carry. hhemt
is a library pointed at arbitrary analysis trees, so "every member this version reads is
post-change" is not a property any hhemt version can check; the guard is. The pre-split
fixture below is that case, and it is the regression test for it.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hhemt.analysis import PERF_VARS, PERF_VARS_ORDERED, _perf_row_from_dataset
from hhemt.cf_conventions import _CF_PERFORMANCE_VARIABLES, _auto_long_name, apply_cf_attributes

# The column set TRITON emitted BEFORE the SWMM-timer split, verbatim from the pre-split
# header literal that `tests/test_synth_03_perf_tseries_diff.py` and
# `tests/test_synth_06_resume_safety.py` still write into their synthetic
# `performance{N}.txt` fixtures:
#     "%Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total"
PRE_SPLIT_VARS = [
    "Compute",
    "MPI",
    "IO",
    "Resize",
    "SWMM",
    "Other",
    "Simulation",
    "Init",
    "Total",
]

# The column set TRITON emits AFTER the split, verbatim from the header literal in
# `src/output.h::write_times` on the solver branch `solver-work/replay-precision-accounting`:
#     "%Rank, Compute, MPI, IO, Resize, SWMM, SWMM_XFER, SWMM_MPI, SWMM_STEP, SWMM_OTHER,
#      Other, Simulation, Init, Total"
# `%Rank` is the index column, not a timing variable, so it is not a member here.
POST_SPLIT_VARS = [
    "Compute",
    "MPI",
    "IO",
    "Resize",
    "SWMM",
    "SWMM_XFER",
    "SWMM_MPI",
    "SWMM_STEP",
    "SWMM_OTHER",
    "Other",
    "Simulation",
    "Init",
    "Total",
]

# The three MEASURED children plus the DERIVED residual. The solver enforces
# `XFER + MPI + STEP + OTHER == SWMM` exactly on every emitted per-rank row (and NOT on the
# Average row, which is outside the identity).
SWMM_CHILDREN = ["SWMM_XFER", "SWMM_MPI", "SWMM_STEP", "SWMM_OTHER"]


def _summary_ds(var_names: list[str], value: float = 1.0) -> xr.Dataset:
    """A performance SUMMARY dataset: scalar per variable, dims already reduced away.

    `_export_performance_summary` writes `ds.sum(dim="timestep_min").max(dim="Rank")`, so
    the object `_get_performance_summary_row` reads carries one scalar per variable.
    """
    return xr.Dataset({name: xr.DataArray(np.float64(value)) for name in var_names})


# --------------------------------------------------------------------------------------
# The guard: a pre-split member must blank, never raise.
# --------------------------------------------------------------------------------------


def test_pre_split_member_blanks_the_new_columns_instead_of_raising():
    """The regression test for the whole package.

    Before the guard, this call was `{f"perf_{v}": float(ds[v].values.item()) for v in
    PERF_VARS}` -- an unguarded comprehension inside the `df_status` row loop, whose call
    site `row.update(...)` carries no try/except. Adding the four split names to
    `PERF_VARS` therefore raised `KeyError` on every member produced before the split, and
    a `KeyError` there does not blank a cell: it fails the status frame for the analysis,
    which is upstream of `scenario_status.csv`, the status appendix, the resume-evidence
    candidate filter and three `validate_analysis` members.
    """
    row = _perf_row_from_dataset(_summary_ds(PRE_SPLIT_VARS, value=7.0))

    # Every PERF_VARS name is minted, so the column set is stable across a mixed corpus.
    assert set(row) == {f"perf_{v}" for v in PERF_VARS}
    # The names the pre-split member DOES carry read through as floats.
    for name in PRE_SPLIT_VARS:
        assert row[f"perf_{name}"] == 7.0
    # The names it does not carry blank rather than raise.
    for name in SWMM_CHILDREN:
        assert row[f"perf_{name}"] is None


def test_post_split_member_reads_every_column_as_a_float():
    row = _perf_row_from_dataset(_summary_ds(POST_SPLIT_VARS, value=3.5))

    assert set(row) == {f"perf_{v}" for v in PERF_VARS}
    assert all(isinstance(v, float) for v in row.values())


def test_guard_is_keyed_on_data_vars_not_on_coords():
    """A name present only as a COORD is not a timing measurement and must blank.

    `_write_output` attaches `hhemt_producing_sha` / `hhemt_producing_version` as
    coordinates on the per-scenario summary, so `name in ds` is true for objects that are
    not data variables. The guard tests `ds.data_vars` for that reason; `name in ds` would
    let a coord through to `float(...)` and raise there instead.
    """
    ds = _summary_ds(PRE_SPLIT_VARS)
    ds = ds.assign_coords(SWMM_XFER="not-a-measurement")

    row = _perf_row_from_dataset(ds)

    assert row["perf_SWMM_XFER"] is None


def test_an_empty_dataset_blanks_every_column():
    """The degenerate end of the same guard: nothing emitted, nothing raised."""
    row = _perf_row_from_dataset(_summary_ds([]))

    assert set(row) == {f"perf_{v}" for v in PERF_VARS}
    assert all(v is None for v in row.values())


# --------------------------------------------------------------------------------------
# The contract between PERF_VARS and the solver's emitted header.
# --------------------------------------------------------------------------------------


def test_perf_vars_matches_the_solver_emitted_column_set():
    """`PERF_VARS` is a versioned contract with the solver header, not a discovered payload.

    The PARSE is header-driven (`parse_performance_file`'s bare `pd.read_csv` carries no
    `names=`/`usecols=`), so a new solver column reaches the per-scenario dataset with no
    toolkit edit. The PATH is not: `PERF_VARS` is what decides which of those columns
    becomes a `perf_*` column, so a column the solver emits and this list omits is silently
    unconsumed. This test is what makes that omission loud.
    """
    assert PERF_VARS == POST_SPLIT_VARS


def test_perf_vars_ordered_is_a_permutation_of_perf_vars():
    """The display-order twin must not drift out of the mint set in either direction.

    A name in `PERF_VARS` but not `PERF_VARS_ORDERED` is minted and then demoted into the
    dynamic tail of `scenario_status.csv`; a name in `PERF_VARS_ORDERED` but not
    `PERF_VARS` is never minted and its ordering entry is dead.
    """
    assert sorted(PERF_VARS_ORDERED) == sorted(PERF_VARS)


def test_each_swmm_child_sits_adjacent_to_its_parent_in_the_display_order():
    """The four children are read AS a decomposition of `SWMM`, so they render beside it.

    This also pins the one adjacency a reader could otherwise mistake: `SWMM_MPI` is the
    coupling's own `MPI_Gatherv`/`MPI_Scatterv` cost and is a child of `SWMM`, NOT part of
    the top-level `MPI` column. Ordering it next to `SWMM` rather than next to `MPI` is
    what says so in the table.
    """
    idx = PERF_VARS_ORDERED.index("SWMM")
    assert PERF_VARS_ORDERED[idx : idx + 1 + len(SWMM_CHILDREN)] == ["SWMM", *SWMM_CHILDREN]


# --------------------------------------------------------------------------------------
# CF attributes on the performance artifacts.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["tritonswmm_performance", "triton_only_performance"])
def test_every_performance_column_carries_cf_long_name_and_units(mode):
    """All thirteen, not just the four new ones.

    An uncovered variable falls through to `_auto_long_name`, which renders `IO` as "Io"
    and supplies no `units`. Covering only the split columns would leave the artifact's
    four newest variables with real CF attrs beside nine auto-humanized placeholders.
    """
    ds = apply_cf_attributes(_summary_ds(POST_SPLIT_VARS), mode)

    for name in POST_SPLIT_VARS:
        attrs = ds[name].attrs
        assert attrs.get("units") == "s", f"{name} carries no CF units"
        assert attrs.get("long_name"), f"{name} carries no long_name"
        assert attrs["long_name"] != _auto_long_name(name), (
            f"{name} fell through to the auto-humanized fallback rather than a declared long_name"
        )


@pytest.mark.parametrize("mode", ["tritonswmm_performance", "triton_only_performance"])
def test_performance_columns_declare_no_cell_methods(mode):
    """The omission is load-bearing, so it is asserted rather than left to inspection.

    Both performance artifacts share one mode string: `_export_performance_tseries` writes
    the per-(timestep_min, Rank) series and `_export_performance_summary` writes
    `ds.sum(dim="timestep_min").max(dim="Rank")` of it, and both reach `apply_cf_attributes`
    through `_write_output` with that same mode. A `cell_methods` accurate for the summary
    would therefore be stamped onto the tseries, which has collapsed neither dim.
    """
    ds = apply_cf_attributes(_summary_ds(POST_SPLIT_VARS), mode)

    for name in POST_SPLIT_VARS:
        assert "cell_methods" not in ds[name].attrs, (
            f"{name} declares cell_methods, which would mislabel the per-timestep series that shares this mode string"
        )


def test_performance_cf_entries_are_scoped_to_the_performance_modes():
    """`Total` / `MPI` / `IO` / `Other` are generic names; they must not stamp other modes."""
    ds = apply_cf_attributes(_summary_ds(POST_SPLIT_VARS), "tritonswmm_triton")

    for name in POST_SPLIT_VARS:
        assert "units" not in ds[name].attrs, f"{name} picked up a performance CF entry outside a performance mode"


def test_cf_coverage_tracks_perf_vars_exactly():
    """The CF block and the mint list must not drift apart in either direction."""
    assert sorted(_CF_PERFORMANCE_VARIABLES) == sorted(PERF_VARS)
