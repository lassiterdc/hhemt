"""Tests for the F2-introduced `analysis_config.report` field (R7, R12)."""
from __future__ import annotations
from pathlib import Path
import yaml
import pytest
from pydantic import ValidationError
from TRITON_SWMM_toolkit.config.analysis import analysis_config
from TRITON_SWMM_toolkit.config.loaders import yaml_to_model


def _minimum_valid_cfg_analysis_dict(stub_root: Path) -> dict:
    """Return the smallest dict that satisfies analysis_config's required
    fields. Paths point at stub files under `stub_root` that the caller has
    already created. Mirrors the structure of
    `tests/fixtures/bundles/multi_sim/cfg_analysis.yaml` minus the report block.
    """
    return {
        "analysis_id": "single_sim_test",
        "weather_event_indices": ["event_index"],
        "weather_timeseries": str(stub_root / "weather.nc"),
        "weather_time_series_timestep_dimension_name": "time",
        "weather_time_series_spatial_mean_rainfall_datavar": "RG",
        "rainfall_units": "mm/hr",
        "run_mode": "serial",
        "n_mpi_procs": 1,
        "n_omp_threads": 1,
        "n_gpus": 0,
        "n_nodes": 1,
        "multi_sim_run_method": "local",
        "local_cpu_cores_for_workflow": 2,
        "mem_gb_per_cpu": 2,
        "hpc_mem_allocation_for_sim_output_processing_mb": 12000,
        "hpc_mem_allocation_for_analysis_output_consolidation_mb": 12000,
        "toggle_sensitivity_analysis": False,
        "toggle_storm_tide_boundary": False,
        "storm_tide_units": "m",
        "weather_time_series_storm_tide_datavar": "water_level",
        "weather_events_to_simulate": str(stub_root / "events.csv"),
        "target_processed_output_type": "zarr",
        "TRITON_raw_output_type": "bin",
        "manhole_diameter": 1.2,
        "manhole_loss_coefficient": 0.1,
        "hydraulic_timestep_s": 1.0,
        "TRITON_reporting_timestep_s": 10.0,
        "open_boundaries": 1,
        "analysis_dir": str(stub_root),
        "is_subanalysis": False,
    }


@pytest.fixture
def stubbed_paths(tmp_path: Path) -> Path:
    (tmp_path / "weather.nc").touch()
    (tmp_path / "events.csv").touch()
    return tmp_path


def test_analysis_config_accepts_inline_report(tmp_path, stubbed_paths):
    """R7: cfg_analysis.yaml with a `report:` block round-trips through
    yaml_to_model and exposes
    analysis.cfg_analysis.report.interactive.static_backend == 'matplotlib'."""
    base = _minimum_valid_cfg_analysis_dict(stubbed_paths)
    base["report"] = {"interactive": {"static_backend": "matplotlib"}}
    cfg_path = tmp_path / "cfg_analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(base))
    cfg = yaml_to_model(cfg_path, analysis_config)
    assert cfg.report is not None
    assert cfg.report.interactive.static_backend == "matplotlib"


def test_analysis_config_rejects_missing_report(tmp_path, stubbed_paths):
    """R12 (rev v2): A cfg_analysis.yaml without a `report:` key raises
    pydantic ValidationError at yaml_to_model load time. The field is
    required-no-default; pre-F2 yaml files do not load post-F2."""
    base = _minimum_valid_cfg_analysis_dict(stubbed_paths)
    # Intentionally omit `report:` from base.
    cfg_path = tmp_path / "cfg_analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(base))
    with pytest.raises(ValidationError) as excinfo:
        yaml_to_model(cfg_path, analysis_config)
    missing = {
        e["loc"][0] for e in excinfo.value.errors() if e["type"] == "missing"
    }
    assert "report" in missing


def test_analysis_config_rejects_unknown_report_subkey(tmp_path, stubbed_paths):
    """extra='forbid' on cfgBaseModel propagates into the nested report
    model — unknown keys raise ValidationError."""
    base = _minimum_valid_cfg_analysis_dict(stubbed_paths)
    base["report"] = {
        "interactive": {"static_backend": "plotly"},
        "bogus_key": True,
    }
    cfg_path = tmp_path / "cfg_analysis.yaml"
    cfg_path.write_text(yaml.safe_dump(base))
    with pytest.raises(ValidationError):
        yaml_to_model(cfg_path, analysis_config)
