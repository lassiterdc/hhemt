"""No-solver tests for the cross-model node variable comparison the synthetic tests use.

The synthetic tests ``test_swmm_cross_model_consistency`` (test_synth_01) and
``test_snakemake_workflow_end_to_end`` (test_synth_04) compare the SWMM-only node store with the coupled
(TRITON-SWMM) one. The coupled store carries ``is_coupling_node`` always and, with
``toggle_coupling_exchange_flow``, ``coupling_exchange_flow`` plus eight statistics, so plain set equality is
wrong by design. ``tests.utils_for_testing.node_var_mismatch`` states the design: every coupled-only name is in
the coupled store and absent from the SWMM-only store, the SWMM-only-only names (the five Runoff Quantity
Continuity rows, which only the standalone model routes runoff to print) may appear in the SWMM-only store and
never in the coupled one, and everything else is equal. Each test docstring gives
Property, Class (REGRESSION for the first: red on plain set equality; NEW CAPABILITY for the rest), Why, and
What kills it.
"""

from __future__ import annotations

from hhemt.constants import LST_RUNOFF_CONTINUITY_VARIABLES, RUNOFF_QUANTITY_CONTINUITY_TERMS
from hhemt.coupling_exchange import STATISTIC_NAMES
from tests.utils_for_testing import coupled_only_node_vars, node_var_mismatch

BASE = {"depth", "inflow_flow", "max_depth"}


def test_the_design_holds_when_only_the_flag_differs():
    """Property: SWMM-only {base} versus coupled {base, is_coupling_node} is consistent with the flux off.

    Class: REGRESSION. Plain `set(swmm) != set(coupled)` rejects exactly this pair, which is the failure on the
    first Rivanna run of the run branch. Why required: this is the shape of every default coupled run.
    What kills it: a comparison that does not allow the flag. Second correct implementation: comparing after
    dropping `coupled_only_node_vars` from the coupled set.
    """
    assert node_var_mismatch(BASE, BASE | {"is_coupling_node"}, exchange_flow_on=False) is None


def test_a_leaked_coupled_only_name_in_the_swmm_only_store_is_reported():
    """Property: `is_coupling_node` in the SWMM-only store is reported by name (must-fire control).

    Class: NEW CAPABILITY. Plausible wrong implementation: a subset check, which passes when the SWMM-only
    store carries extra names. Why required: the SWMM-only run has no coupling nodes. What kills it: loosening
    the check to `swmm <= coupled`. Second correct implementation: asserting the intersection with the
    coupled-only set is empty.
    """
    msg = node_var_mismatch(BASE | {"is_coupling_node"}, BASE | {"is_coupling_node"}, exchange_flow_on=False)
    assert msg is not None and "present in the SWMM-only store" in msg and "is_coupling_node" in msg


def test_a_missing_flag_in_the_coupled_store_is_reported():
    """Property: a coupled store without `is_coupling_node` is reported (must-fire control for the always-on flag).

    Class: NEW CAPABILITY. Plausible wrong implementation: dropping the flag from both sides before comparing.
    Why required: the flag is always written for the coupled model. What kills it: that implementation.
    Second correct implementation: asserting `"is_coupling_node" in coupled` separately.
    """
    msg = node_var_mismatch(BASE, BASE, exchange_flow_on=False)
    assert msg is not None and "absent from the coupled store" in msg and "is_coupling_node" in msg


def test_any_other_difference_is_reported_in_both_directions():
    """Property: a variable only in one store (beyond the coupled-only set) is reported, naming which side.

    Class: NEW CAPABILITY. Plausible wrong implementation: comparing only one direction. Why required: the two
    stores otherwise come from the same parser and must agree. What kills it: a one-sided set difference.
    Second correct implementation: symmetric difference of the two reduced sets.
    """
    only_swmm = node_var_mismatch(BASE | {"extra"}, BASE | {"is_coupling_node"}, exchange_flow_on=False)
    only_coupled = node_var_mismatch(BASE, BASE | {"is_coupling_node", "extra"}, exchange_flow_on=False)
    assert only_swmm is not None and "only in SWMM-only ['extra']" in only_swmm
    assert only_coupled is not None and "only in coupled ['extra']" in only_coupled


def test_with_the_flux_on_the_flux_and_its_eight_statistics_are_coupled_only():
    """Property: with the flux on, the flux and its eight statistics are expected in the coupled store only.

    Class: NEW CAPABILITY. Plausible wrong implementation: expecting only the flag regardless of the toggle.
    Why required: the same comparison must serve a flux-on analysis. What kills it: a fixed one-name set.
    Second correct implementation: building the set from `STATISTIC_NAMES` as here.
    Controls: with the flux OFF the same coupled store is reported (the flux names are then an unexpected difference),
    and a coupled store missing one statistic is reported.
    """
    names = coupled_only_node_vars(exchange_flow_on=True)
    assert names == {"is_coupling_node", "coupling_exchange_flow", *STATISTIC_NAMES}
    assert len(STATISTIC_NAMES) == 8
    assert node_var_mismatch(BASE, BASE | names, exchange_flow_on=True) is None
    assert node_var_mismatch(BASE, BASE | names, exchange_flow_on=False) is not None
    missing_one = BASE | (names - {STATISTIC_NAMES[0]})
    msg = node_var_mismatch(BASE, missing_one, exchange_flow_on=True)
    assert msg is not None and STATISTIC_NAMES[0] in msg


RUNOFF = {
    "runoff_continuity_total_precipitation",
    "runoff_continuity_evaporation_loss",
    "runoff_continuity_infiltration_loss",
    "runoff_continuity_surface_runoff",
    "runoff_continuity_final_storage",
}


def test_the_swmm_only_only_set_is_the_five_runoff_continuity_rows_pinned_to_the_source():
    """Property: the SWMM-only-only node variables are exactly the five `runoff_continuity_*` row stems.

    Class: NEW CAPABILITY. Why required: chunk 1d.6 (D19) deliberately adds them to the SWMM-only node store and the
    coupled model routes no runoff, so the cross-model comparison must expect them. The set is read from the source
    constants and compared with the five names the Rivanna failure reported, so a sixth row added to the source,
    or a rename, turns this red instead of being absorbed. What kills it: a hand-typed copy that drifts from
    `RUNOFF_QUANTITY_CONTINUITY_TERMS`. Second correct implementation: a literal five-name set (as `RUNOFF` here).
    """
    from tests.utils_for_testing import swmm_only_node_vars

    names = swmm_only_node_vars()
    assert names == RUNOFF
    assert names == set(LST_RUNOFF_CONTINUITY_VARIABLES)
    assert names == {f"runoff_continuity_{stem}" for stem in RUNOFF_QUANTITY_CONTINUITY_TERMS.values()}
    assert len(names) == 5
    assert names.isdisjoint(coupled_only_node_vars(exchange_flow_on=True))


def test_the_swmm_only_only_names_are_admitted_in_the_swmm_only_store():
    """Property: SWMM-only {base, five runoff rows} versus coupled {base, flag} is consistent, flux off and on.

    Class: REGRESSION. This is the failure on the 2770a5b8 Rivanna run of `test_swmm_cross_model_consistency` and
    `test_snakemake_workflow_end_to_end`: `only in SWMM-only [five runoff_continuity names], only in coupled []`.
    Why required: the source is right (D19, shape row 36) and the helper's design was stale. What kills it: the
    pre-repair helper, whose `remaining variables differ` arm reports exactly these five. Second correct
    implementation: dropping `swmm_only_node_vars` from the SWMM-only set before comparing.
    """
    assert node_var_mismatch(BASE | RUNOFF, BASE | {"is_coupling_node"}, exchange_flow_on=False) is None
    flux = coupled_only_node_vars(exchange_flow_on=True)
    assert node_var_mismatch(BASE | RUNOFF, BASE | flux, exchange_flow_on=True) is None
    # Admitted, not required: a SWMM-only store that carries none of them is still consistent.
    assert node_var_mismatch(BASE, BASE | {"is_coupling_node"}, exchange_flow_on=False) is None


def test_the_exact_variable_sets_of_the_rivanna_failure_now_pass():
    """Property: the two variable sets behind the reported `node_var_mismatch` return None.

    Class: REGRESSION. The failure text names five SWMM-only names and an empty coupled-only remainder, so the
    stores were {shared, five} and {shared, is_coupling_node}; `shared` here stands for every variable the stores
    have in common (the helper reads no value of it). Why required: it is the observed pair, not a constructed one.
    What kills it: any narrowing of the admitted set below the five.
    """
    shared = {
        "depth",
        "inflow_flow",
        "flooding_rate",
        "routing_continuity_wet_weather_inflow",
        "runoff_continuity_error",
    }
    assert node_var_mismatch(shared | RUNOFF, shared | {"is_coupling_node"}, exchange_flow_on=False) is None


def test_a_name_outside_the_admitted_set_in_only_the_swmm_only_store_is_still_reported():
    """Property: one extra name beside the five is reported, and a sixth `runoff_continuity_` name is not admitted.

    Class: NEW CAPABILITY (must-fire control: the admitted set is exactly five, not a prefix). Plausible wrong
    implementation: admitting every name that starts with `runoff_continuity_`, which would absorb a renamed or
    added row. What kills it: that prefix rule. Second correct implementation: exact-set membership as here.
    """
    msg = node_var_mismatch(BASE | RUNOFF | {"extra"}, BASE | {"is_coupling_node"}, exchange_flow_on=False)
    assert msg is not None and "only in SWMM-only ['extra']" in msg
    msg = node_var_mismatch(
        BASE | RUNOFF | {"runoff_continuity_sixth_row"}, BASE | {"is_coupling_node"}, exchange_flow_on=False
    )
    assert msg is not None and "only in SWMM-only ['runoff_continuity_sixth_row']" in msg
    # The same names are not excused in the coupled direction by the SWMM-only admission.
    msg = node_var_mismatch(BASE, BASE | {"is_coupling_node", "extra"}, exchange_flow_on=False)
    assert msg is not None and "only in coupled ['extra']" in msg


def test_a_swmm_only_only_name_in_the_coupled_store_is_reported():
    """Property: a `runoff_continuity_*` name in the coupled store is reported by name, whichever side else carries it.

    Class: NEW CAPABILITY (the inverse of the leaked coupled-only class). Plausible wrong implementation: dropping
    the five from BOTH sides before the equality, which would let the coupled model silently grow them. Why
    required: the coupled model routes no runoff and must not store them. What kills it: that two-sided drop.
    Second correct implementation: asserting the intersection of the coupled store with the set is empty.
    """
    one = "runoff_continuity_surface_runoff"
    for swmm in (BASE | RUNOFF, BASE):
        msg = node_var_mismatch(swmm, BASE | {"is_coupling_node", one}, exchange_flow_on=False)
        assert msg is not None and "SWMM-only-only variables present in the coupled store" in msg and one in msg
    msg = node_var_mismatch(BASE | RUNOFF, BASE | {"is_coupling_node"} | RUNOFF, exchange_flow_on=False)
    assert msg is not None and all(name in msg for name in RUNOFF)


def test_the_existing_violation_classes_are_unchanged_beside_the_admitted_set():
    """Property: the three prior classes still fire when the SWMM-only store also carries the admitted five.

    Class: REGRESSION guard. Why required: the admission must not mask a leak, a missing flag or a plain
    difference. What kills it: an admission implemented by skipping the comparison when any of the five is present.
    """
    swmm = BASE | RUNOFF
    leaked = node_var_mismatch(swmm | {"is_coupling_node"}, BASE | {"is_coupling_node"}, exchange_flow_on=False)
    assert leaked is not None and "present in the SWMM-only store" in leaked
    absent = node_var_mismatch(swmm, BASE, exchange_flow_on=False)
    assert absent is not None and "absent from the coupled store" in absent
    other = node_var_mismatch(swmm, BASE | {"is_coupling_node", "extra"}, exchange_flow_on=False)
    assert other is not None and "only in coupled ['extra']" in other
