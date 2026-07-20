"""Regression coverage for report_config.disabled_renderers (Phase 3).

The byte-identity suite (test_synth_reporting_sets_byte_identity.py) proves the
feature is INERT at disabled_renderers=[]. This module guards the ACTIVE behavior:

  * the five-site lockstep — a disabled builder_key drops from emission AND from
    the rule all / render_report enumeration together, so no site disagrees
    (a dispatch-only implementation would leave the plot path in rule all /
    render_report and yield a MissingInputException); and
  * the run-entry unknown-key raise — an unknown disabled_renderers key is a
    ConfigurationError at validate_active_reporting_set() entry, not a silent
    no-op (renderer_active() never matches a typo, so nothing would drop).
"""

from __future__ import annotations

import pytest

from hhemt.config.report import report_config, validate_active_reporting_set
from hhemt.exceptions import ConfigurationError
from hhemt.report_renderers._reporting_sets import renderer_active


def test_renderer_active_predicate():
    """The shared predicate: active unless the key is in the disabled list."""
    assert renderer_active("per_sim", []) is True
    assert renderer_active("per_sim", None) is True
    assert renderer_active("per_sim", ["per_sim"]) is False
    assert renderer_active("metadata", ["per_sim"]) is True


def test_five_site_lockstep_multisim(synth_multi_sim_analysis):
    """disabled_renderers=['per_sim'] drops per_sim from the emitted rule AND
    rule all AND render_report together, leaving other renderers intact."""
    builder = synth_multi_sim_analysis._workflow_builder
    gen_kwargs = dict(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )

    # Positive control: per_sim present when nothing is disabled.
    baseline = builder.generate_snakefile_content(**gen_kwargs)
    assert "rule plot_per_sim_peak_flood_depth" in baseline
    assert "plots/per_sim/" in baseline

    # Disable per_sim and regenerate.
    synth_multi_sim_analysis.cfg_analysis.report.disabled_renderers = ["per_sim"]
    disabled = builder.generate_snakefile_content(**gen_kwargs)

    # Emission: no per_sim plot rules.
    assert "rule plot_per_sim_peak_flood_depth" not in disabled
    assert "rule plot_per_sim_conduit_flow" not in disabled
    # Enumeration: no per_sim path anywhere (rule all, render_report, outputs).
    assert "plots/per_sim/" not in disabled
    # Lockstep control: other renderers survive; render_report still emitted.
    assert "plots/system_overview" in disabled
    assert "plots/metadata" in disabled
    assert "rule render_report:" in disabled


def test_unknown_key_raises_at_run_entry():
    """An unknown disabled_renderers key raises ConfigurationError at run() entry."""
    cfg = report_config(disabled_renderers=["nonexistent_key"])
    with pytest.raises(ConfigurationError) as ei:
        validate_active_reporting_set(cfg, is_sensitivity=False, sensitivity_csv_path=None)
    msg = str(ei.value)
    assert "disabled_renderers" in msg
    assert "nonexistent_key" in msg


def test_known_key_does_not_raise():
    """A key present in the resolved set's renderer_selection validates cleanly."""
    cfg = report_config(disabled_renderers=["per_sim"])  # per_sim is in the default set
    name = validate_active_reporting_set(cfg, is_sensitivity=False, sensitivity_csv_path=None)
    assert name == "default"
