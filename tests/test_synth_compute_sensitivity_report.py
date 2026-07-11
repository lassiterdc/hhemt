"""Phase 4 (R11 + R12): the compute-sensitivity ReportingSet renders the EDA
figures as config-selectable tabs, the in-report EDA adapter passes the
renderer-IO provenance audit, and the combined report assembles via
combine_bundle over two sensitivity-master bundles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hhemt.report_plot_ids import canonical_plot_id, output_ext_for
from hhemt.report_renderers._reporting_sets import get_reporting_set


def test_compute_sensitivity_set_wires_eda_renderer():
    """R11 (data-level, no compile): the compute-sensitivity set carries the EDA
    adapter as a conditional (has_eda_artifact) renderer landing under Key
    Results, and its bundle template co-sources the category + a .html ext."""
    s = get_reporting_set("compute-sensitivity")
    eda_sels = [sel for sel in s.renderer_selection if sel.builder_key == "eda_compute_sensitivity"]
    assert len(eda_sels) == 1, "compute-sensitivity must carry exactly one eda_compute_sensitivity renderer"
    sel = eda_sels[0]
    assert sel.predicate_key == "has_eda_artifact"
    assert len(sel.rule_spec_template) == 1
    tmpl = sel.rule_spec_template[0]
    assert tmpl.renderer_module == "eda_compute_sensitivity"
    assert tmpl.report_kwargs["category"] == "Key Results"
    assert tmpl.output_path_template.startswith("plots/eda/config_diff_maps")
    # The adapter is always-.html under both static backends.
    assert output_ext_for("plotly", "eda_compute_sensitivity") == ".html"
    assert output_ext_for("matplotlib", "eda_compute_sensitivity") == ".html"
    # validator is benchmarking (compute-sensitivity IS a sensitivity analysis).
    assert s.validator_key == "benchmarking"


@pytest.mark.slow
def test_r11_eda_adapter_passes_provenance_audit(rendered_synth_sensitivity):
    """R11 (compile-tier): run the EDA calc+plot on a rendered sensitivity master,
    then render the in-report eda_compute_sensitivity adapter under the renderer-IO
    provenance audit (Gotcha 53). The adapter delegates to render_eda_plots, which
    self-declares its sources, so the audit must NOT raise."""
    from hhemt.report_renderers import eda_compute_sensitivity
    from hhemt.report_renderers._provenance_audit import audit_renderer_io

    analysis = rendered_synth_sensitivity
    analysis.eda()  # produce plots/eda/config_diff_maps.html + its manifest sidecar

    root = Path(analysis.analysis_paths.analysis_dir)
    plot_id = canonical_plot_id("config_diff_maps")
    output_path = root / "plots" / "eda" / f"{plot_id}.html"

    with audit_renderer_io(output_path, root, renderer_name="eda_compute_sensitivity"):
        eda_compute_sensitivity.render(analysis, analysis.cfg_analysis.report, output_path)

    assert output_path.exists(), "adapter must (re)produce the config_diff_maps figure at output_path"
    assert (
        output_path.with_suffix(".html.manifest.json").exists()
        or (output_path.parent / f"{plot_id}.manifest.json").exists()
    )


@pytest.mark.slow
def test_r12_combined_report_assembles_from_two_sensitivity_masters(synthetic_two_sensitivity_bundle_fixture, tmp_path):
    """R12 (compile-tier): combine_bundle accepts two sensitivity-master bundles
    and assembles the single combined report (the compat-only render is
    tree-shape-agnostic)."""
    from hhemt.bundle import CombinedBundle, combine_bundle

    dir_a, dir_b = synthetic_two_sensitivity_bundle_fixture
    out = tmp_path / "combined_sens"
    combined = combine_bundle([dir_a, dir_b], output_path=out)
    assert isinstance(combined, CombinedBundle)
    report = out / "analysis_report.html"
    assert report.exists()
    assert "Cross-Experiment Compatibility" in report.read_text()
    # Regen round-trips against the bundle root (no re-merge; reads the read-model).
    regen = combined.regenerate_report(format="html")
    assert regen.exists()
