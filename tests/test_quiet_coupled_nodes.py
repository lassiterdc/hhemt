"""No-solver tests for the node-selection criterion of `test_the_flux_integrates_to_the_lateral_inflow_volume`.

The identity test needs coupled nodes whose lateral inflow is the coupling exchange alone. The synthetic model lists
EVERY junction in `[INFLOWS]` with an empty time series and a zero baseline (so the coupling layer can read its node
list there), which the first Rivanna run showed makes a "not listed in [INFLOWS]" criterion select nothing. The
criterion is "no inflow that can be non-zero". Test docstrings give Property, Class (REGRESSION for the first, NEW
CAPABILITY for the rest), Why and What kills it.
"""

from __future__ import annotations

from tests.utils_for_testing import quiet_coupled_nodes as _quiet_coupled_nodes

INP = """[JUNCTIONS]
J1 1 2 0 0 0
J2 1 2 0 0 0
J3 1 2 0 0 0
J4 1 2 0 0 0
J5 1 2 0 0 0
J6 1 2 0 0 0
J7 1 2 0 0 0

[INFLOWS]
;;Node Constituent TimeSeries Type Mfactor Sfactor Baseline Pattern
J1 FLOW "" FLOW 1.0 1 0
J2 FLOW "" FLOW 1.0 1 0.5
J3 FLOW "series_a" FLOW 1.0 1 0
J4 FLOW "" FLOW 1.0 1 0
J7 FLOW "" FLOW 1.0 1 0

[DWF]
J5 FLOW 0.1

[RDII]
J6 UH1 10

[SUBCATCHMENTS]
S1 RG J7 5 100 100 1 0
"""


def _quiet(tmp_path, nodes=("J1", "J2", "J3", "J4", "J5", "J6", "J7")):
    """Write INP to a file and return the quiet coupled nodes among `nodes`."""
    path = tmp_path / "hydraulics.inp"
    path.write_text(INP)
    return _quiet_coupled_nodes(path, list(nodes))


def test_zero_baseline_empty_series_entries_do_not_make_a_node_loaded(tmp_path):
    """Property: J1 and J4 (FLOW entry, empty series, zero baseline) are quiet; so is nothing else here.

    Class: REGRESSION. The criterion "listed in [INFLOWS]" returns [] on the synthetic model and the identity test
    failed with "no coupled node is free of other inflow". Why required: the identity needs such nodes.
    What kills it: counting any [INFLOWS] line as loaded. Second correct implementation: parsing [INFLOWS] with
    pandas and filtering on `Time Series` and `Baseline`.
    """
    assert _quiet(tmp_path) == ["J1", "J4"]


def test_every_other_source_of_inflow_still_excludes_a_node(tmp_path):
    """Property: a non-zero baseline (J2), a series (J3), DWF (J5), RDII (J6) or an outlet (J7) each excludes.

    Class: NEW CAPABILITY. Plausible wrong implementation: dropping the [INFLOWS] check altogether, which would
    select J2 and J3. Why required: a node with another inflow breaks the identity. What kills it: that. Second
    correct implementation: reading the same sections with the toolkit's own parser. Must-fire control: the five
    nodes are each absent from the result of the same call that returns J1 and J4.
    """
    result = _quiet(tmp_path)
    assert not {"J2", "J3", "J5", "J6", "J7"} & set(result)
