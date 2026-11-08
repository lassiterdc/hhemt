"""No-solver tests for the cross-model node variable comparison the synthetic tests use.

The synthetic tests ``test_swmm_cross_model_consistency`` (test_synth_01) and
``test_snakemake_workflow_end_to_end`` (test_synth_04) compare the SWMM-only node store with the coupled
(TRITON-SWMM) one. The coupled store carries ``is_coupling_node`` always and, with
``toggle_coupling_exchange_flow``, ``coupling_exchange_flow`` plus eight statistics, so plain set equality is
wrong by design. ``tests.utils_for_testing.node_var_mismatch`` states the design: every coupled-only name is in
the coupled store and absent from the SWMM-only store, and everything else is equal. Each test docstring gives
Property, Class (REGRESSION for the first: red on plain set equality; NEW CAPABILITY for the rest), Why, and
What kills it.
"""

from __future__ import annotations

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
