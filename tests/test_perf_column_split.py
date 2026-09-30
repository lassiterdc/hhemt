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
from hhemt.cf_conventions import (
    _CF_PERFORMANCE_SUMMARY_VARIABLES,
    _CF_PERFORMANCE_VARIABLES,
    _auto_long_name,
    apply_cf_attributes,
)

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
    """A performance SUMMARY dataset: one value per variable on a RETAINED size-one `Rank`.

    `_export_performance_summary` writes `ds.sum(dim="timestep_min").max(dim="Rank",
    keepdims=True)`, so `timestep_min` is gone entirely while `Rank` survives at size one --
    it is the carrier for the summary's CF `cell_methods` domain, which has no admissible
    name if the axis is fully collapsed.

    THE SHAPE IS PART OF WHAT THESE TESTS COVER. This helper carried 0-d variables until the
    axis was retained; keeping it 0-d would have left every test below asserting against a
    shape the producer no longer writes, and would in particular have let
    `_perf_row_from_dataset`'s `.values.item()` pass here while an axis it had never seen
    reached it in production.
    """
    return xr.Dataset({name: (("Rank",), np.array([value], dtype=np.float64)) for name in var_names})


def _tseries_ds(var_names: list[str], value: float = 1.0) -> xr.Dataset:
    """A performance TIMESERIES dataset: both `timestep_min` and `Rank` intact.

    The artifact `_export_performance_tseries` writes. It has collapsed no axis, which is why
    no `cell_methods` may be stamped on it.
    """
    return xr.Dataset(
        {name: (("timestep_min", "Rank"), np.full((3, 2), value, dtype=np.float64)) for name in var_names}
    )


#: The two artifacts' mode strings, split so each can be addressed separately. The unsuffixed
#: names are the SUMMARY -- `processing_analysis._MODE_CONFIG` already binds them to
#: `output_*_performance_summary` -- and the `_tseries` names are write-path-only.
PERF_SUMMARY_MODES = ["tritonswmm_performance", "triton_only_performance"]
PERF_TSERIES_MODES = ["tritonswmm_performance_tseries", "triton_only_performance_tseries"]


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


@pytest.mark.parametrize("mode", PERF_SUMMARY_MODES + PERF_TSERIES_MODES)
def test_every_performance_column_carries_cf_long_name_and_units(mode):
    """All thirteen, not just the four new ones, and now on all FOUR modes.

    An uncovered variable falls through to `_auto_long_name`, which renders `IO` as "Io"
    and supplies no `units`. Covering only the split columns would leave the artifact's
    four newest variables with real CF attrs beside nine auto-humanized placeholders.

    WIDENED to the `_tseries` modes when the mode strings split: `long_name` and `units` are
    true of the series and of every reduction of it, so a split that dropped them from either
    artifact would be a regression this test is the only one positioned to catch.
    """
    ds = apply_cf_attributes(_summary_ds(POST_SPLIT_VARS), mode)

    for name in POST_SPLIT_VARS:
        attrs = ds[name].attrs
        assert attrs.get("units") == "s", f"{name} carries no CF units"
        assert attrs.get("long_name"), f"{name} carries no long_name"
        assert attrs["long_name"] != _auto_long_name(name), (
            f"{name} fell through to the auto-humanized fallback rather than a declared long_name"
        )


@pytest.mark.parametrize("mode", PERF_TSERIES_MODES)
def test_performance_tseries_columns_declare_no_cell_methods(mode):
    """The omission is load-bearing FOR THE SERIES, so it is asserted rather than inspected.

    RE-TARGETED, NOT WEAKENED. This assertion previously ran on the two unsuffixed modes,
    because both performance artifacts shared one mode string and a `cell_methods` accurate
    for the summary would have been stamped onto the series. The mode strings are now split,
    so the same claim is asserted where it is still true -- the per-(timestep_min, Rank)
    series, which has collapsed NEITHER dim and to which no reduction applies -- and its
    complement is asserted for the summary modes in the sibling test below. The pair covers
    four modes where the single assertion covered two.
    """
    ds = apply_cf_attributes(_tseries_ds(POST_SPLIT_VARS), mode)

    for name in POST_SPLIT_VARS:
        assert "cell_methods" not in ds[name].attrs, (
            f"{name} declares cell_methods on the per-timestep series, which has collapsed neither dim"
        )


@pytest.mark.parametrize("mode", PERF_SUMMARY_MODES)
def test_performance_summary_columns_declare_the_rank_reduction(mode):
    """The summary's whole reason for being separately addressable.

    Asserts the CF-1.13 admissibility condition, not merely the string: section 7.3 allows a
    cell_methods name only if it is "a dimension of the variable, a scalar coordinate
    variable, a valid standard name, or the word `area`", so the name is checked against the
    variable's own dims. That is what makes the retained size-one `Rank` load-bearing rather
    than decorative -- fully collapsing the axis would leave this string naming nothing.

    `timestep_min` IS DELIBERATELY NOT NAMED and its absence is asserted. The time sum is
    real, but after it `timestep_min` is neither a dim, nor a coord, nor a CF standard name,
    nor `area`, so naming it would be inadmissible in exactly the way a scalar `Rank` was.
    """
    ds = apply_cf_attributes(_summary_ds(POST_SPLIT_VARS), mode)

    for name in POST_SPLIT_VARS:
        cm = ds[name].attrs.get("cell_methods")
        assert cm == "Rank: maximum", f"{name} carries cell_methods {cm!r}, not the summary's rank reduction"
        named_axis = cm.split(":")[0]
        assert named_axis in ds[name].dims, (
            f"{name} names axis {named_axis!r} in cell_methods but does not carry it as a dimension; "
            "CF-1.13 section 7.3 admits only a dimension, a scalar coordinate variable, a standard name, or 'area'"
        )
        assert "timestep_min" not in cm, (
            f"{name} names timestep_min, which the time sum removed and which is not a CF standard name"
        )


def test_the_two_performance_artifacts_agree_on_long_name_and_units():
    """The summary descriptors are DERIVED from the series', so only `cell_methods` may differ.

    Pins the derivation rather than the two dicts' contents: a hand-copied second literal
    would drift on the next wording fix, and that drift is invisible to every other test here
    because each one reads only one of the two dicts.
    """
    assert set(_CF_PERFORMANCE_SUMMARY_VARIABLES) == set(_CF_PERFORMANCE_VARIABLES)
    for name, summary_entry in _CF_PERFORMANCE_SUMMARY_VARIABLES.items():
        base_entry = _CF_PERFORMANCE_VARIABLES[name]
        assert base_entry["cell_methods"] is None, f"{name}'s series entry gained a cell_methods"
        assert summary_entry["cell_methods"] == "Rank: maximum"
        for field in ("standard_name", "long_name", "units"):
            assert summary_entry[field] == base_entry[field], (
                f"{name}'s {field} differs between the series and summary descriptors; "
                "the summary dict is derived and only cell_methods may diverge"
            )


def test_performance_cf_entries_are_scoped_to_the_performance_modes():
    """`Total` / `MPI` / `IO` / `Other` are generic names; they must not stamp other modes."""
    ds = apply_cf_attributes(_summary_ds(POST_SPLIT_VARS), "tritonswmm_triton")

    for name in POST_SPLIT_VARS:
        assert "units" not in ds[name].attrs, f"{name} picked up a performance CF entry outside a performance mode"


def test_cf_coverage_tracks_perf_vars_exactly():
    """The CF block and the mint list must not drift apart in either direction."""
    assert sorted(_CF_PERFORMANCE_VARIABLES) == sorted(PERF_VARS)
