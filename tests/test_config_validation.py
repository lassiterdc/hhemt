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
        "weather_time_series_spatial_mean_rainfall_datavar": "RG_synth",
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
        "report": {},
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


# ---------------------------------------------------------------------------
# Phase 1 — report_config schema validation
# ---------------------------------------------------------------------------


def _write_report_yaml(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_report_config_loads_default(tmp_path: Path):
    from TRITON_SWMM_toolkit.config.loaders import yaml_to_model
    from TRITON_SWMM_toolkit.config.report import report_config

    yaml_path = _write_report_yaml(
        tmp_path / "report.yaml",
        "figure_defaults:\n  dpi: 120\n",
    )
    cfg = yaml_to_model(yaml_path, report_config)
    assert cfg.figure_defaults.dpi == 120
    assert cfg.sensitivity is None  # F-I-7: default is None


def test_report_config_rejects_unknown_field(tmp_path: Path):
    """Flag 7 — `extra='forbid'` regression test."""
    from TRITON_SWMM_toolkit.config.loaders import yaml_to_model
    from TRITON_SWMM_toolkit.config.report import report_config

    yaml_path = _write_report_yaml(
        tmp_path / "report.yaml",
        "system-map:\n  target_epsg: 4326\n",  # hyphen, not underscore
    )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        yaml_to_model(yaml_path, report_config)


def test_report_config_sensitivity_missing_independent_vars(tmp_path: Path):
    from TRITON_SWMM_toolkit.config.loaders import yaml_to_model
    from TRITON_SWMM_toolkit.config.report import report_config

    yaml_path = _write_report_yaml(
        tmp_path / "report.yaml",
        "sensitivity:\n  mode: benchmarking\n",  # missing independent_vars
    )
    with pytest.raises(ValidationError, match="independent_vars"):
        yaml_to_model(yaml_path, report_config)


def test_validate_sensitivity_independent_vars_missing_columns(tmp_path: Path):
    import pandas as pd

    from TRITON_SWMM_toolkit.config.report import (
        SensitivityReportConfig,
        report_config,
        validate_sensitivity_independent_vars,
    )
    from TRITON_SWMM_toolkit.exceptions import ConfigurationError

    csv_path = tmp_path / "sa.csv"
    pd.DataFrame({"n_omp_threads": [1, 2], "run_mode": ["serial", "parallel"]}).to_csv(
        csv_path, index=False
    )
    cfg = report_config(
        sensitivity=SensitivityReportConfig(independent_vars=["n_omp_threads", "missing_col"])
    )
    with pytest.raises(ConfigurationError) as exc:
        validate_sensitivity_independent_vars(cfg, csv_path)
    assert "missing_col" in str(exc.value)


def test_validate_sensitivity_independent_vars_charset(tmp_path: Path):
    """Flag 17 — Snakemake-safe charset validation."""
    import pandas as pd

    from TRITON_SWMM_toolkit.config.report import (
        SensitivityReportConfig,
        report_config,
        validate_sensitivity_independent_vars,
    )
    from TRITON_SWMM_toolkit.exceptions import ConfigurationError

    csv_path = tmp_path / "sa.csv"
    pd.DataFrame({"bad name": [1, 2]}).to_csv(csv_path, index=False)
    cfg = report_config(
        sensitivity=SensitivityReportConfig(independent_vars=["bad name"])
    )
    with pytest.raises(ConfigurationError, match="charset"):
        validate_sensitivity_independent_vars(cfg, csv_path)


def test_validate_sensitivity_fails_when_block_missing_but_csv_present(tmp_path: Path):
    """F-I-6 — sensitivity CSV present with no sensitivity block raises."""
    from TRITON_SWMM_toolkit.config.report import (
        report_config,
        validate_sensitivity_independent_vars,
    )
    from TRITON_SWMM_toolkit.exceptions import ConfigurationError

    csv_path = tmp_path / "sa.csv"
    csv_path.write_text("col\n1\n")
    cfg = report_config()  # no sensitivity block
    with pytest.raises(ConfigurationError, match="must be set"):
        validate_sensitivity_independent_vars(cfg, csv_path)


def test_validate_sensitivity_fails_when_block_present_but_no_csv(tmp_path: Path):
    from TRITON_SWMM_toolkit.config.report import (
        SensitivityReportConfig,
        report_config,
        validate_sensitivity_independent_vars,
    )
    from TRITON_SWMM_toolkit.exceptions import ConfigurationError

    cfg = report_config(sensitivity=SensitivityReportConfig(independent_vars=["col"]))
    with pytest.raises(ConfigurationError, match="no sensitivity CSV path"):
        validate_sensitivity_independent_vars(cfg, None)


def test_report_artifacts_not_in_globus_exclude_patterns():
    """Flag 14 — R12 automated Globus-exclude audit."""
    from TRITON_SWMM_toolkit.config.globus import DEFAULT_EXCLUDE_PATTERNS

    for bad in ("plots", "report", "analysis_report.html"):
        assert not any(bad in p for p in DEFAULT_EXCLUDE_PATTERNS), (
            f"{bad!r} would match an entry of DEFAULT_EXCLUDE_PATTERNS "
            f"{DEFAULT_EXCLUDE_PATTERNS}; R12 requires report artifacts "
            "to be included in the default Globus transfer."
        )


def test_pydantic_config_field_names_are_snakemake_wildcard_safe():
    """Phase 1 R9 — every system_config and analysis_config field name MUST match
    `^[A-Za-z0-9_.]+$` so the prefixed-column overlay mechanism (which routes
    `system.{field}` / `analysis.{field}` cells into Snakemake wildcards) cannot
    silently break on a future field addition with a hyphen or other unsafe char.

    Developer-facing assertion against the codebase's Pydantic model schemas.
    Failure indicates a toolkit author has introduced a bad field name.
    """
    import re

    charset = re.compile(r"^[A-Za-z0-9_.]+$")
    offenders: list[str] = []
    for model_name, model in [
        ("system_config", system_config),
        ("analysis_config", analysis_config),
    ]:
        for field_name in model.model_fields:
            if not charset.match(field_name):
                offenders.append(f"{model_name}.{field_name}")
    assert not offenders, (
        f"Pydantic field names outside Snakemake-wildcard-safe charset "
        f"^[A-Za-z0-9_.]+$ (toolkit author defect, not a user-config issue): "
        f"{offenders}."
    )
