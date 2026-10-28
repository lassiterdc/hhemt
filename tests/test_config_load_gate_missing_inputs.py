"""Review addition for `assert_both_configs_load`: a declared input that does not exist.

The loader runs in `runnable` mode, so a config naming an absent input file fails the
existence validator. That failure is a schema failure (pydantic ValidationError converted
to ConfigurationError by the loader), not a missing CONFIG file, so the gate must collect
it from BOTH documents rather than raise on the first. No solver is compiled or run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hhemt.exceptions import ConfigurationError
from hhemt.validation import assert_both_configs_load
from tests.test_config_validation import _minimal_analysis_config_dict, _minimal_system_config_dict


def test_absent_declared_inputs_in_both_configs_are_reported_together(tmp_path):
    system_cfg = _minimal_system_config_dict(tmp_path)
    system_cfg["DEM_fullres"] = str(tmp_path / "inputs" / "absent_dem.tif")
    analysis_cfg = _minimal_analysis_config_dict(tmp_path)
    analysis_cfg["weather_timeseries"] = str(tmp_path / "inputs" / "absent_weather.nc")
    system = tmp_path / "system.yaml"
    analysis = tmp_path / "analysis.yaml"
    system.write_text(yaml.safe_dump(system_cfg))
    analysis.write_text(yaml.safe_dump(analysis_cfg))
    with pytest.raises(ConfigurationError) as exc:
        assert_both_configs_load(system, analysis)
    text = str(exc.value)
    assert "absent_dem.tif" in text
    assert "absent_weather.nc" in text


def test_empty_system_file_propagates_value_error_before_analysis_is_read(tmp_path):
    system = tmp_path / "system.yaml"
    system.write_text("")
    analysis = tmp_path / "analysis.yaml"
    analysis.write_text(yaml.safe_dump(_minimal_analysis_config_dict(tmp_path)))
    with pytest.raises(ValueError, match="parsed to None"):
        assert_both_configs_load(Path(system), Path(analysis))
