"""Synth-tier tests for the EDA->report promotion path (reporting-system ADR-11)."""

from __future__ import annotations

import pytest
import yaml

from hhemt.config.static_plots import StaticPlotBaseConfig
from hhemt.eda import (
    promote_eda_plot_to_static_config,
    register_eda_plot_in_reporting_set,
)

# matplotlib-specific base fields that MUST stay at their schema defaults on a
# Plotly-sourced emit (D-2 neutral-only contract).
_MPL_SPECIFIC_DEFAULTS = {
    "colorbar_norm": "linear",
    "colorbar_extend": "neither",
    "set_bad_color": None,
    "vmin": None,
    "vmax": None,
    "vmax_quantile": None,
    "colorbar_boundaries": None,
}


@pytest.fixture
def eda_plot_id(synthetic_sensitivity_completed_isolated):
    """Run .eda() on the synth sensitivity master; return the rendered EDA plot-ID."""
    analysis = synthetic_sensitivity_completed_isolated.master_analysis
    result = analysis.eda()
    assert result.plot_paths, "expected >=1 rendered EDA plot on the sensitivity master"
    return result.plot_paths[0].stem  # the figure stem IS the canonical ADR-2 plot-ID


def test_static_config_roundtrips_with_eda_plot_id(eda_plot_id, tmp_path):
    out = tmp_path / f"{eda_plot_id}.yaml"
    written = promote_eda_plot_to_static_config(eda_plot_id, output_path=out, caption="cross-sim identity")
    assert written == out and written.exists()
    text = written.read_text()
    assert text.lstrip().startswith("#"), "emitted YAML must carry the source-backend header comment"
    loaded = StaticPlotBaseConfig.model_validate(yaml.safe_load(text))
    assert loaded.plot_id == eda_plot_id  # R5 plot-ID single-source


def test_emit_populates_only_neutral_fields(eda_plot_id, tmp_path):
    out = tmp_path / f"{eda_plot_id}.yaml"
    loaded = StaticPlotBaseConfig.model_validate(
        yaml.safe_load(promote_eda_plot_to_static_config(eda_plot_id, output_path=out).read_text())
    )
    for field, default in _MPL_SPECIFIC_DEFAULTS.items():
        assert getattr(loaded, field) == default, f"{field} must stay at its schema default (neutral-only)"
    assert loaded.output_format not in {"pgf", "ps"}  # R3 format portability


def test_invalid_plot_id_charset_raises(tmp_path):
    with pytest.raises(ValueError, match="charset"):
        promote_eda_plot_to_static_config("bad-id-with-hyphen", output_path=tmp_path / "x.yaml")


def test_default_output_path_is_cwd_relative_not_analysis_dir(eda_plot_id, monkeypatch, tmp_path):
    # R4: output_path=None resolves to a cwd-relative promoted_static_configs/{plot_id}.yaml,
    # NEVER under analysis_dir/. chdir into tmp_path so the default dir is hermetic.
    monkeypatch.chdir(tmp_path)
    written = promote_eda_plot_to_static_config(eda_plot_id)
    assert written == tmp_path / "promoted_static_configs" / f"{eda_plot_id}.yaml"
    assert written.exists()
    assert "analysis_dir" not in str(written)  # not nested under any analysis tree


def test_register_in_reporting_set_records_intent():
    rec = register_eda_plot_in_reporting_set("config_diff_maps", "default")
    assert rec.plot_id == "config_diff_maps"
    assert rec.set_name == "default"
    assert "deferred" in rec.routing  # D-1 option (c): routing deferred to eda-skill


def test_register_unknown_set_raises():
    with pytest.raises(ValueError, match="ReportingSet"):
        register_eda_plot_in_reporting_set("config_diff_maps", "no_such_set")
