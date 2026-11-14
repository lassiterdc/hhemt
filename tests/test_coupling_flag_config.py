"""Configuration surface of the coupling outputs: the exchange-flux toggle exists, the coupling-node flag has no
toggle, the toggle is an experiment field of the reproducer taxonomy, and preflight refuses it without the coupled
model.

None of this needs a solver. Names that this change creates are read lazily inside the test bodies, so that on a
tree without the chunk each test collects and fails with an AssertionError from this file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _analysis_fields() -> dict:
    from hhemt.config.analysis import analysis_config

    return analysis_config.model_fields


def test_the_exchange_flux_has_a_toggle_and_the_coupling_node_flag_has_none() -> None:
    """Property: `toggle_coupling_exchange_flow` is an analysis field defaulting to False, and no field named
    `toggle_coupling_node_flag` exists.

    Class: NEW CAPABILITY. Plausible wrong implementations: adding a flag toggle (the developer's correction was
    that the flag is always written for the coupled model), or naming the flux toggle differently.
    Reachability: the field table `analysis_config.model_fields` is the configuration schema itself.
    Both-states anchor: the assertions turn on the schema's field names and a default, present in any code that
    declares such a field.
    Why required: a flag toggle would let a coupled store be written without the flag the consolidation relies on.
    Killed by: declaring `toggle_coupling_node_flag`, or deleting the flux field. A second correct implementation
    declares the flux field with a different description; the test does not read the description.
    """
    fields = _analysis_fields()
    assert "toggle_coupling_exchange_flow" in fields, "toggle_coupling_exchange_flow is not an analysis field"
    assert fields["toggle_coupling_exchange_flow"].default is False
    assert "toggle_coupling_node_flag" not in fields


def test_the_toggle_is_an_experiment_field_of_the_reproducer_taxonomy() -> None:
    """Property: the reproducer taxonomy buckets `toggle_coupling_exchange_flow` as `experiment`.

    Class: NEW CAPABILITY. Plausible wrong implementation: leaving the field out of the table (then
    `test_field_bucket_is_total` reds, but only if that file runs) or bucketing it `hpc`.
    Reachability: `_FIELD_BUCKET` is the table the totality tests compare every config field against.
    Both-states anchor: asserts a table value, not wording.
    Killed by: omitting the entry or using another bucket. A second correct implementation derives the bucket from
    a field annotation; the test reads the resulting table value.
    """
    from hhemt.config.reprex_taxonomy import _FIELD_BUCKET

    assert _FIELD_BUCKET.get("toggle_coupling_exchange_flow") == "experiment"
    assert _FIELD_BUCKET.get("toggle_consolidate_timeseries") == "experiment"  # control: a neighbour's entry reads


def _preflight_refusals(*, coupled: bool, flux: bool) -> list:
    from hhemt import validation

    check = getattr(validation, "_validate_coupling_exchange_flow", None)
    assert check is not None, "validation._validate_coupling_exchange_flow does not exist"
    result = validation.ValidationResult(context="test")
    check(
        SimpleNamespace(toggle_tritonswmm_model=coupled),
        SimpleNamespace(toggle_coupling_exchange_flow=flux),
        result,
    )
    return [e for e in result.errors if "toggle_coupling_exchange_flow" in str(e)]


@pytest.mark.parametrize(
    "coupled, flux, refused",
    [(False, True, True), (True, True, False), (False, False, False), (True, False, False)],
)
def test_preflight_refuses_the_flux_toggle_only_without_the_coupled_model(coupled, flux, refused) -> None:
    """Property: preflight yields an error exactly when the flux toggle is on and the coupled model is off.

    Class: NEW CAPABILITY. Plausible wrong implementations: never refusing (a silent run with no flux variable),
    or refusing whenever the toggle is on.
    Reachability: the four rows are the whole truth table of the rule.
    Both-states anchor: asserts the error list's emptiness per row.
    Killed by: either wrong implementation (rows (False, True) and (True, True) separate them). A second correct
    implementation raises instead of accumulating; the test would then need to catch it, which is a change of
    contract, not of behaviour.
    """
    assert bool(_preflight_refusals(coupled=coupled, flux=flux)) is refused
