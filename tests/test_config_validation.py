from pathlib import Path

import pytest
from pydantic import ValidationError

from TRITON_SWMM_toolkit.config.analysis import analysis_config
from TRITON_SWMM_toolkit.config.loaders import load_system_config_from_dict
from TRITON_SWMM_toolkit.config.system import system_config


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")
    return path


def _minimal_system_config_dict(tmp_path: Path) -> dict:
    return {
        "system_directory": str(tmp_path / "system"),
        "watershed_gis_polygon": str(_touch(tmp_path / "inputs" / "watershed.shp")),
        "DEM_fullres": str(_touch(tmp_path / "inputs" / "dem.tif")),
        "SWMM_hydraulics": str(_touch(tmp_path / "inputs" / "hydraulics.inp")),
        "TRITONSWMM_software_directory": str(tmp_path / "triton"),
        "TRITONSWMM_git_URL": "https://example.com/triton.git",
        "SWMM_git_URL": "https://example.com/swmm.git",
        "triton_swmm_configuration_template": str(
            _touch(tmp_path / "inputs" / "TRITONSWMM.cfg")
        ),
        "toggle_use_swmm_for_hydrology": False,
        "toggle_use_constant_mannings": True,
        "toggle_triton_model": False,
        "toggle_tritonswmm_model": True,
        "toggle_swmm_model": False,
        "target_dem_resolution": 5.0,
        "constant_mannings": 0.05,
    }


def _minimal_analysis_config_dict(tmp_path: Path) -> dict:
    return {
        "analysis_id": "analysis_01",
        "weather_event_indices": ["event_id"],
        "weather_timeseries": str(_touch(tmp_path / "inputs" / "weather.nc")),
        "weather_time_series_timestep_dimension_name": "timestep",
        "rainfall_units": "mm/hr",
        "run_mode": "serial",
        "multi_sim_run_method": "local",
        "toggle_sensitivity_analysis": False,
        "toggle_storm_tide_boundary": False,
        "weather_events_to_simulate": str(_touch(tmp_path / "inputs" / "events.csv")),
        "manhole_diameter": 1.0,
        "manhole_loss_coefficient": 0.1,
        "hydraulic_timestep_s": 1.0,
        "TRITON_reporting_timestep_s": 60,
        "open_boundaries": 1,
    }


def test_system_config_forbids_unknown_keys(tmp_path: Path):
    cfg = _minimal_system_config_dict(tmp_path)
    cfg["unexpected_extra"] = "should fail"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_system_config_from_dict(cfg)


def test_system_config_explicit_toggle_dependency(tmp_path: Path):
    cfg = _minimal_system_config_dict(tmp_path)
    cfg["toggle_use_constant_mannings"] = True
    cfg["constant_mannings"] = None

    with pytest.raises(ValidationError, match="constant_mannings"):
        system_config.model_validate(cfg)


def test_analysis_config_explicit_toggle_dependency(tmp_path: Path):
    cfg = _minimal_analysis_config_dict(tmp_path)
    cfg["toggle_storm_tide_boundary"] = True

    with pytest.raises(ValidationError, match="storm_tide_boundary_line_gis"):
        analysis_config.model_validate(cfg)
