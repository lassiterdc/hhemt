"""Guards for `cf_conventions._QUANTITY_PROVENANCE` (Iteration 10, item I).

The computed-quantity descriptor table is hand-authored, so the risk it carries is
DRIFT: a variable gains a CF entry and never gains a descriptor, and the metadata
report renders an em-dash where a reader expects the operation. These tests make that
a CI failure instead of a reading error.

They deliberately do NOT assert that any particular operation string is *correct* --
no test can check prose against a dask expression. They assert coverage, vocabulary,
and non-emptiness, which is the part a machine can decide.
"""

from __future__ import annotations

import pytest

from hhemt.cf_conventions import (
    _ADVERTISABLE_VARIABLES,
    _CF_PERFORMANCE_VARIABLES,
    _DESCRIBED_VARIABLES,
    _PERF_QUANTITY_PROVENANCE,
    _PERF_SUMMARY_PROVENANCE_MODES,
    _QUANTITY_PROVENANCE,
    _QUANTITY_PROVENANCE_BY_MODE,
    quantity_provenance,
)

#: The controlled vocabulary for `spatial_representation`. Kept small on purpose: the
#: column exists to tell a reader what geometry a value describes, and a free-text
#: column would defeat that within two additions.
_SPATIAL_VOCAB = frozenset(
    {
        "grid cell",
        "point (node)",
        "line (conduit)",
        "whole domain (scalar)",
    }
)

_REQUIRED_KEYS = frozenset({"spatial_representation", "source_variables", "operation", "reduced_coordinate"})


def test_quantity_provenance_is_in_bijection_with_the_advertised_union():
    """Every ADVERTISABLE variable has a descriptor, and vice versa.

    This is the drift guard the descriptor columns depend on. A variable present in
    one table and absent from the other is exactly the state that publishes a report
    row asserting a long_name and a unit while saying nothing about how the number was
    computed.

    WIDENED FROM `_CF_VARIABLE_MAP` TO THE ADVERTISED UNION, and the widening is the
    whole reason this guard is load-bearing again. `build_analysis_crate` advertises
    `_ADVERTISABLE_VARIABLES` on its gated path -- the base map PLUS the thirteen
    performance columns -- so a guard scoped to the base map alone is structurally
    blind to thirteen advertised variables. It was green while all thirteen rendered
    an em-dash in the report's three provenance columns, which is precisely the state
    this docstring's own next paragraph names as what the guard exists to prevent.

    The union is IMPORTED from the module that the advertisement site also imports it
    from. That is what makes this guard fail on a drift rather than on a disagreement
    between two independently-maintained unions.
    """
    adv_only = sorted(set(_ADVERTISABLE_VARIABLES) - set(_DESCRIBED_VARIABLES))
    prov_only = sorted(set(_DESCRIBED_VARIABLES) - set(_ADVERTISABLE_VARIABLES))
    assert not adv_only, (
        f"Advertisable variables with no computed-quantity descriptor: {adv_only}. "
        "Add an entry to cf_conventions._QUANTITY_PROVENANCE (base) or "
        "_PERF_QUANTITY_PROVENANCE (performance) grounded in the expression that "
        "computes the variable, not in its cell_methods string."
    )
    assert not prov_only, (
        f"Descriptors for variables absent from the advertised union: {prov_only}. "
        "A descriptor for a variable the pipeline does not emit is published as a "
        "false claim about the data (the same failure the 2026-07-21 removal fixed)."
    )


def test_the_two_descriptor_tables_are_disjoint_so_the_mode_blind_read_is_unambiguous():
    """The mode-blind read path merges two tables; it is sound only if they are disjoint.

    `quantity_provenance(var, mode=None)` resolves `{**_QUANTITY_PROVENANCE,
    **_PERF_QUANTITY_PROVENANCE}`, which is the path the metadata renderer takes
    because a crate carries no mode. A key present in both would silently let the
    performance table shadow a base descriptor, and the renderer would publish the
    wrong operation for a variable that already had a right one. The sibling
    disjointness of the two CF ATTRIBUTE maps is asserted in `tests/test_metadata.py`;
    this is the same property one layer over, for the descriptor tables.
    """
    overlap = sorted(set(_QUANTITY_PROVENANCE) & set(_PERF_QUANTITY_PROVENANCE))
    assert not overlap, (
        f"{overlap} carry a descriptor in BOTH tables, so the mode-blind read is "
        "ambiguous and _PERF_QUANTITY_PROVENANCE silently wins. Either remove the "
        "duplicate or stop resolving the union on the mode=None path."
    )


def test_the_performance_overlay_covers_exactly_the_thirteen_advertised_columns():
    """The overlay's key set is pinned to the CF performance map, not to a count.

    A count of thirteen is reachable by a different thirteen. This compares the SETS,
    so a solver column added to `_CF_PERFORMANCE_VARIABLES` without a descriptor fails
    here with the missing name rather than passing on an unchanged total.
    """
    assert set(_PERF_QUANTITY_PROVENANCE) == set(_CF_PERFORMANCE_VARIABLES), (
        "overlay key set drifted from _CF_PERFORMANCE_VARIABLES: "
        f"missing={sorted(set(_CF_PERFORMANCE_VARIABLES) - set(_PERF_QUANTITY_PROVENANCE))}, "
        f"extra={sorted(set(_PERF_QUANTITY_PROVENANCE) - set(_CF_PERFORMANCE_VARIABLES))}"
    )


def test_the_overlay_is_mode_scoped_and_declines_under_a_non_performance_mode():
    """The mode arm is what the mode-scoping buys, so it is asserted directly.

    The thirteen keys are generic (`Total`, `MPI`, `IO`, `Other`), which is why
    `_CF_PERFORMANCE_VARIABLES` was never merged into `_CF_VARIABLE_MAP`. A caller
    that knows its mode must not be handed a wallclock descriptor for a non-performance
    artifact that happens to carry a same-named variable. Both directions are asserted,
    because a flat merge satisfies the first and silently breaks the second.
    """
    for mode in sorted(_PERF_SUMMARY_PROVENANCE_MODES):
        assert _QUANTITY_PROVENANCE_BY_MODE[mode] is _PERF_QUANTITY_PROVENANCE
        assert quantity_provenance("Total", mode=mode) is not None, f"{mode} must resolve the overlay"
        # The base table stays reachable under a performance mode -- the overlay shadows,
        # it does not replace.
        assert quantity_provenance("max_wlevel_m", mode=mode) is not None

    # A non-performance mode: the generic names must NOT resolve.
    assert quantity_provenance("Total", mode="tritonswmm") is None, (
        "a performance descriptor resolved under a non-performance mode -- the overlay "
        "was merged flat into _QUANTITY_PROVENANCE instead of being mode-scoped"
    )
    assert quantity_provenance("max_wlevel_m", mode="tritonswmm") is not None


def test_the_tseries_modes_carry_no_overlay_because_the_reduction_is_false_of_them():
    """The `*_performance_tseries` modes name the UNREDUCED per-rank series.

    Every `operation` in the overlay names a sum over timesteps and a maximum over
    ranks. The tseries artifact has had neither applied, so handing it these
    descriptors would publish a false claim about how its numbers were computed. The
    absence is a correctness property; this test is what keeps it from being
    "fixed" by a later reader who sees four performance modes and two overlay keys.
    """
    for mode in ("tritonswmm_performance_tseries", "triton_only_performance_tseries"):
        assert mode not in _QUANTITY_PROVENANCE_BY_MODE
        assert quantity_provenance("Total", mode=mode) is None


@pytest.mark.parametrize("var_name", sorted(_PERF_QUANTITY_PROVENANCE))
def test_each_performance_descriptor_names_both_reduced_axes(var_name):
    """`_PERF_SUMMARY_CELL_METHODS` names only `Rank`; this table owes the time sum too.

    CF-1.13 section 7.3 admits no cell_methods name for the summed-away `timestep_min`,
    so `cf_conventions`'s own header nominates this table as the channel carrying it.
    A descriptor that named only the rank reduction would leave the sum documented
    nowhere a reader looks.
    """
    entry = _PERF_QUANTITY_PROVENANCE[var_name]
    assert "timestep_min" in entry["reduced_coordinate"], f"{var_name} does not name the summed time axis"
    assert "Rank" in entry["reduced_coordinate"], f"{var_name} does not name the reduced rank axis"


# PARAMETRIZED OVER THE DESCRIBED UNION, not over the base table. The two structural
# checks below are what make a descriptor renderable at all -- the renderer reads four
# keys and a controlled vocabulary column -- so scoping them to the base table would have
# admitted thirteen performance descriptors with no key check and no vocabulary check.
@pytest.mark.parametrize("var_name", sorted(_DESCRIBED_VARIABLES))
def test_each_descriptor_carries_every_required_key_non_empty(var_name):
    entry = _DESCRIBED_VARIABLES[var_name]
    missing = sorted(_REQUIRED_KEYS - set(entry))
    assert not missing, f"{var_name} descriptor is missing {missing}"
    blank = sorted(k for k in _REQUIRED_KEYS if not str(entry[k]).strip())
    assert not blank, f"{var_name} descriptor has blank {blank}"


@pytest.mark.parametrize("var_name", sorted(_DESCRIBED_VARIABLES))
def test_spatial_representation_uses_the_controlled_vocabulary(var_name):
    value = _DESCRIBED_VARIABLES[var_name]["spatial_representation"]
    assert value in _SPATIAL_VOCAB, (
        f"{var_name} declares spatial_representation={value!r}, which is outside the "
        f"controlled vocabulary {sorted(_SPATIAL_VOCAB)}. Widen the vocabulary "
        "deliberately if a new geometry is genuinely needed."
    )


def test_reader_returns_a_copy_and_none_for_unknown():
    """`quantity_provenance` must not hand out the module table itself.

    The renderer reads this per row; a caller that mutated the returned dict would
    corrupt every subsequent row on the page.
    """
    entry = quantity_provenance("max_wlevel_m")
    assert entry is not None
    entry["operation"] = "MUTATED"
    assert _QUANTITY_PROVENANCE["max_wlevel_m"]["operation"] != "MUTATED"
    assert quantity_provenance("no_such_variable") is None


def test_last_timestep_variable_is_described_as_a_selection_not_a_reduction():
    """Regression guard for the defect that motivated this table.

    `wlevel_m_last_tstep` carries `cell_methods="timestep_min: point"`, but `point`
    in CF means the variable RETAINS the time dimension with no method applied. The
    computation (process_simulation.py, `summarize_triton_simulation_results`) is
    `ds["wlevel_m"].sel(timestep_min=tsteps.max())` -- a selection. If the descriptor
    ever drifts back to reduction language, the table has reacquired the error it was
    written to correct.
    """
    entry = _QUANTITY_PROVENANCE["wlevel_m_last_tstep"]
    assert "select" in entry["operation"].lower()
    assert "maximum" not in entry["operation"].lower()
