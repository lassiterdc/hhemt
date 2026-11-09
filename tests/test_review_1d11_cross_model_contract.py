"""Reviewer tests for chunk 1d.11: the admitted SWMM-only-only node variables follow the parser, not a copied list.

No solver. Property: the node variables the standalone-SWMM run-level reader adds over the coupled reader are exactly
``swmm_only_node_vars()``, so a reader change that adds a SWMM-only node variable turns the helper stale loudly instead
of the Rivanna tier finding it.
"""

from __future__ import annotations

from hhemt.constants import LST_SWMM_SYSTEM_SERIES, LST_SWMM_SYSTEM_SUMMARY_VARIABLES
from hhemt.swmm_output_parser import _run_level_datasets
from tests.utils_for_testing import node_var_mismatch


def _node_vars(family: str) -> set[str]:
    ds_nodes, _ds_links = _run_level_datasets({}, {}, 0.0, "<none>", has_report=True, family=family)
    return {str(v) for v in ds_nodes.data_vars}


def test_the_admitted_set_is_the_run_level_node_difference_between_the_two_readers():
    """Property: standalone-SWMM run-level node variables minus coupled ones == ``swmm_only_node_vars()``.

    Class: NEW CAPABILITY (derived from the reader, not from the helper's constant). Must-fire control: the difference
    is non-empty, and the coupled reader's set is a strict subset of the standalone one. Plausible wrong
    implementation: a helper set that lags the reader. Killed by: any reader addition or removal on one family only.
    """
    from tests.utils_for_testing import swmm_only_node_vars

    swmm, coupled = _node_vars("swmm"), _node_vars("tritonswmm")
    assert swmm - coupled, "control: the reader difference is empty, so this test sees nothing"
    assert coupled < swmm
    assert swmm - coupled == swmm_only_node_vars()


def test_no_system_series_or_reduction_name_is_admitted_or_read_into_the_node_store():
    """Property: neither the six system series nor their twelve reductions are in the admitted set, and the node
    run-level reader writes none of them, so the helper is never asked to excuse them.

    Class: REGRESSION guard on over-admission. Must-fire control: a caller that does pass one of the eighteen names
    in the SWMM-only store alone is reported.
    """
    from tests.utils_for_testing import swmm_only_node_vars

    names = set(LST_SWMM_SYSTEM_SERIES) | set(LST_SWMM_SYSTEM_SUMMARY_VARIABLES)
    assert len(names) == 18
    assert swmm_only_node_vars().isdisjoint(names)
    assert _node_vars("swmm").isdisjoint(names)
    base = {"depth"}
    for name in sorted(names):
        msg = node_var_mismatch(base | {name}, base | {"is_coupling_node"}, exchange_flow_on=False)
        assert msg is not None and name in msg
