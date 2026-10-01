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

from ._cf_cell_methods_grammar import cell_methods_classes

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
    it is the carrier for the `Rank` PAIR of the summary's two-pair CF `cell_methods`, which
    has no admissible name under the spelling `Rank` if the axis is fully collapsed. The TIME
    pair is spelled `time` and is admissible by CF 7.3.4's class 3 precisely BECAUSE no
    dimension or coordinate here is named `time`; that pair needs no retained axis. The
    collapsed-axis argument is about the `Rank` spelling only, and reading it as a general rule
    that a summed-away axis cannot be named is what produced this campaign's defect.

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

    Asserts the CF-1.13 admissibility condition PER PAIR, not merely the string. Section 7.3
    defines the attribute as "a list of blank-separated words of the form `name: method`" and
    allows a name only if it is "a dimension of the variable, a scalar coordinate variable, a
    valid standard name, or the word `area`". The value carries TWO pairs taking TWO DIFFERENT
    classes -- `time` by class 3 (7.3.4's standard-name form) and `Rank` by class 1 (a dimension
    of the variable) -- so the check is per pair and the CLASS SEQUENCE is what is asserted.
    That is what makes the retained size-one `Rank` load-bearing rather than decorative: collapse
    it and the second pair's class drops to None.

    A COLON SPLIT CANNOT EXPRESS THIS and the superseded form used one. Measured,
    `"time: sum Rank: maximum".split(":")` is `['time', ' sum Rank', ' maximum']`, whose middle
    element carries the first pair's method and the second pair's name together -- so the split
    yields no pair structure at all. `_cf_cell_methods_grammar.cell_methods_classes` is the
    per-pair oracle and imports nothing from `cf_conventions`, so a wrong rule cannot agree with
    itself at both ends.

    THE CLASS SEQUENCE IS STRICTLY STRONGER THAN THE `timestep_min`-ABSENCE ASSERTION it
    replaced. That assertion passed on any string lacking the token, `"foo: bar"` included;
    this one pins which of CF's four classes each name takes.
    """
    ds = apply_cf_attributes(_summary_ds(POST_SPLIT_VARS), mode)

    for name in POST_SPLIT_VARS:
        cm = ds[name].attrs.get("cell_methods")
        assert cm == "time: sum Rank: maximum", (
            f"{name} carries cell_methods {cm!r}, not the summary's time sum and rank reduction"
        )
        classes = cell_methods_classes(cm, ds[name], ds, where=name)
        assert classes == ["standard_name", "dimension"], (
            f"{name}'s cell_methods pairs resolve to classes {classes}, not the expected "
            "class-3 time pair followed by the class-1 retained-dimension rank pair"
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
        assert summary_entry["cell_methods"] == "time: sum Rank: maximum"
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


def test_the_rank_axis_family_carries_units_without_cell_methods():
    """`units` is inherited from the parent column; `cell_methods` is declined; neither implies the other.

    The defect this pins had ONE cause and THREE faces. The coherent family is a SELECTION and the
    `min` family names a different method, so neither may carry the summary's `cell_methods` --
    correct, and asserted by its own sibling. What that reasoning does NOT reach is `units`: a
    selection from a seconds-valued column, and a minimum over it, are both still seconds. CF-1.13
    section 3.1.1 makes the omission an assertion rather than a gap ("A variable with no units
    attribute is assumed to be dimensionless"), so dropping `units` alongside `cell_methods`
    publishes 26 durations as dimensionless quantities.

    The `slowest rank` assertion is the third face and the one no other test can see. Seven of the
    thirteen parent `long_name` strings END in `cf_conventions._RANK_ATTRIBUTION_SUFFIX`, which is
    true of a maximum and false of both derived families, so a mechanical suffix onto the raw parent
    string yields seven self-contradictions that read as ordinary prose.

    THIS ARM IS A SUBSTRING TEST AND IS THEREFORE HALF OF THE GUARD, NOT ALL OF IT. It tests for the
    words `slowest rank` while the strip it protects is an `endswith` match on the full constant, so
    it catches a reword that MOVES the phrase off the end and is GREEN on one that REPLACES it --
    measured on both forms. `test_the_rank_attribution_strip_is_live` below closes the replace case
    by measuring the strip's effect rather than any phrase; keep both.

    ASSERTED THROUGH `apply_cf_attributes`, NOT BY IMPORTING THE DESCRIPTOR TABLE. The resolved
    attributes are a property of both the pre-fix and post-fix trees, so this test RUNS and FAILS on
    a tree that has no rank-axis table at all. Importing the table instead would make the pre-fix
    failure an ImportError -- a discrimination on the symbol's existence rather than on the
    behaviour, which is green for any table that merely exists. Measured both ways.
    """
    derived = [f"{n}{s}" for n in POST_SPLIT_VARS for s in ("_coherent", "_min")]
    scalars = ["coherent_rank", "n_ranks"]
    ds = apply_cf_attributes(_summary_ds(POST_SPLIT_VARS + derived + scalars), PERF_SUMMARY_MODES[0])

    for name in POST_SPLIT_VARS:
        parent_units = _CF_PERFORMANCE_VARIABLES[name]["units"]
        for suffix in ("_coherent", "_min"):
            attrs = ds[f"{name}{suffix}"].attrs
            assert attrs.get("units") == parent_units, (
                f"{name}{suffix} does not inherit its parent's units {parent_units!r}; "
                "CF reads an absent units as a positive claim that the variable is dimensionless"
            )
            assert "cell_methods" not in attrs, (
                f"{name}{suffix} declares cell_methods; a selection and a minimum are not the "
                "summary's time sum and rank maximum"
            )
            assert "slowest rank" not in attrs.get("long_name", ""), (
                f"{name}{suffix} attributes itself to the slowest rank, which is the parent's reduction"
            )

    for name in scalars:
        attrs = ds[name].attrs
        assert attrs.get("units") == "1", f"{name} is dimensionless and must declare it explicitly"
        assert "cell_methods" not in attrs, f"{name} declares cell_methods"

    for name in derived + scalars:
        long_name = ds[name].attrs.get("long_name")
        assert long_name and long_name != _auto_long_name(name), (
            f"{name} fell through to the auto-humanized fallback rather than a declared long_name"
        )


def test_the_rank_attribution_strip_is_live():
    """The rank-attribution strip actually FIRES, measured as an effect rather than as a phrase.

    THE DEFECT THIS PINS IS A SILENT ONE AND ITS SIBLING GUARD CANNOT SEE IT.
    `_perf_quantity_phrase` removes the parent's rank attribution before the derived families
    append their own qualifier, and it does so by `endswith` against ONE constant. Its sibling
    above tests the SUBSTRING `"slowest rank"` on the derived names. Those two predicates come
    apart on exactly one class of edit -- a reword that REPLACES the attribution phrase rather
    than extending it -- and that class is the likely one, because rewording for clarity is what
    anyone touching these strings is trying to do. Measured on three parent forms:

        parent form                                  strip fires   substring guard
        "... time, slowest rank for this column"     yes           pass   (shipped)
        "... time at the slowest rank"               NO            FAIL   (caught, loud)
        "... time at the maximum-attaining rank"     NO            pass   (GREEN, broken)

    On the third row every derived name silently becomes a self-contradiction -- it asserts both
    the parent's rank attribution and the family's own, in ordinary-reading prose.

    BOTH ARMS BELOW MEASURE THE FUNCTION'S EFFECT, which is what makes them wording-independent.
    Arm 1 asserts the SET of parents the strip changes, not a count: a count of seven is reachable
    by a different seven, and the whole failure mode here is a silent change of membership. Arm 2
    asserts the post-condition the strip exists to produce -- no derived `long_name` contains its
    parent's full `long_name` -- which is precisely what a no-opped strip violates, under any
    wording, including a future one that drops the words `slowest rank` entirely.

    Arm 3 is the CONTROL, and it is here because arms 1 and 2 are both satisfiable by a strip that
    cannot fail: if `_RANK_ATTRIBUTION_SUFFIX` were the empty string the slice would be a no-op on
    every input, arm 1 would see an empty fired-set against an empty expectation if someone also
    emptied the pin, and nothing would be measured. The control drives a known no-op form through
    the real function and asserts it is detected.
    """
    from hhemt.cf_conventions import (
        _CF_PERFORMANCE_RANK_AXIS_VARIABLES,
        _RANK_ATTRIBUTED_COLUMNS,
        _RANK_ATTRIBUTION_SUFFIX,
        _perf_quantity_phrase,
    )

    # Arm 1 -- the strip's effect, as a SET.
    fired = {
        name
        for name, entry in _CF_PERFORMANCE_VARIABLES.items()
        if _perf_quantity_phrase(entry["long_name"]) != entry["long_name"]
    }
    assert fired == set(_RANK_ATTRIBUTED_COLUMNS), (
        f"the rank-attribution strip fires on {sorted(fired)} but _RANK_ATTRIBUTED_COLUMNS pins "
        f"{sorted(_RANK_ATTRIBUTED_COLUMNS)}. A parent long_name was reworded so it no longer ends "
        f"in {_RANK_ATTRIBUTION_SUFFIX!r}, which makes the strip a NO-OP for that column and leaves "
        "the parent's own rank attribution inside every derived name. Either restore the suffix "
        "form or update the constant AND this pin together -- never this pin alone."
    )
    assert fired, "the strip fires on no parent at all; it has become unconditionally inert"

    # Arm 2 -- the post-condition, stated over containment rather than over any phrase.
    for name in _RANK_ATTRIBUTED_COLUMNS:
        parent_long_name = _CF_PERFORMANCE_VARIABLES[name]["long_name"]
        for suffix in ("_coherent", "_min"):
            derived = _CF_PERFORMANCE_RANK_AXIS_VARIABLES[f"{name}{suffix}"]["long_name"]
            assert parent_long_name not in derived, (
                f"{name}{suffix}'s long_name {derived!r} contains its parent's FULL long_name, so "
                "the attribution strip did not fire and the derived name now carries two "
                "contradictory rank attributions"
            )

    # Arm 3 -- the control: a replace-the-phrase reword is detected by arm 2's predicate even
    # though the substring guard passes it. Driven through the real function, no production state
    # mutated.
    reworded_parent = "Cumulative compute-kernel time at the maximum-attaining rank"
    assert _perf_quantity_phrase(reworded_parent) == reworded_parent, (
        "control precondition: this form must NOT be stripped, or it does not reproduce the no-op"
    )
    broken_derived = f"{_perf_quantity_phrase(reworded_parent)}, read at the rank attaining max(Total)"
    assert "slowest rank" not in broken_derived, (
        "control precondition: the substring guard must PASS on this form, which is what makes it "
        "the silent case rather than the loud one"
    )
    assert reworded_parent in broken_derived, (
        "arm 2's containment predicate must catch the form the substring guard misses"
    )
