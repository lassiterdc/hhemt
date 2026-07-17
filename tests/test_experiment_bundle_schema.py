"""Schema tests for the experiment-bundle descriptor (`experiment.yaml`).

Covers the five load-bearing behaviors of `hhemt.config.experiment_bundle`:
  (a) a minimal valid descriptor validates;
  (b) an unknown key is rejected (`extra="forbid"`);
  (c) a literal operator `local_path` is rejected (`_check_resolvable`);
  (d) a `DatasetRef` with no resolver (no local_path, no doi/pid) is rejected;
  (e) an input that is neither deposited nor DOI-resolvable is rejected
      (`ExperimentBundle._check_deposit_coverage`).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hhemt.config.experiment_bundle import DatasetRef, ExperimentBundle


def _minimal_bundle_dict() -> dict:
    """A minimal descriptor that validates: one depositable, ${VAR}-templated input."""
    return {
        "experiment_id": "exp_demo",
        "description": "demo experiment",
        "system_config": "system.yaml",
        "analysis_config": "analysis.yaml",
        "toolkit_pin": {"version": "0.1.0"},
        "inputs": [
            {
                "name": "forcing",
                "local_path": "${HHEMT_DATA_ROOT}/weather/design_storm.nc",
                "deposit": True,
            }
        ],
    }


def test_minimal_valid_descriptor_validates():
    bundle = ExperimentBundle.model_validate(_minimal_bundle_dict())
    assert bundle.experiment_id == "exp_demo"
    assert bundle.toolkit_pin.version == "0.1.0"
    assert bundle.inputs[0].deposit is True
    assert bundle.container is None


def test_unknown_key_is_rejected():
    d = _minimal_bundle_dict()
    d["unexpected_key"] = "boom"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExperimentBundle.model_validate(d)


def test_literal_operator_local_path_is_rejected():
    with pytest.raises(ValidationError, match="literal operator path"):
        DatasetRef.model_validate({"name": "forcing", "local_path": "/home/dcl3nd/x.nc", "deposit": True})


def test_unresolvable_datasetref_is_rejected():
    with pytest.raises(ValidationError, match="needs a local_path"):
        DatasetRef.model_validate({"name": "forcing"})


def test_undepositable_input_is_rejected():
    d = _minimal_bundle_dict()
    d["inputs"] = [
        {
            "name": "forcing",
            "local_path": "${HHEMT_DATA_ROOT}/weather/design_storm.nc",
            "deposit": False,
        }
    ]
    with pytest.raises(ValidationError, match="neither deposited nor DOI-resolvable"):
        ExperimentBundle.model_validate(d)
