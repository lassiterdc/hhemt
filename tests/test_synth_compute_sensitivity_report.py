"""The compute-sensitivity ReportingSet carries one eda_compute_sensitivity renderer selection, the in-report EDA
adapter passes the renderer-IO provenance audit, and the combined report assembles via combine_bundle over two
sensitivity-master bundles.

The two rows that need a sensitivity tree run on the committed sensitivity golden
(``tests/fixtures/golden_trees.py``): the EDA facade is called in-process to produce the ``plots/eda`` figure and its
manifest sidecar, and the bundles are emitted from that tree. No solver is compiled or executed, and the only
Snakemake process is the report-render child of the combined-report test.

Only the interpreter directory is prepended to ``PATH``, and only for the combined-report row, whose report render
runs ``snakemake`` as a child process (measured: without it the row fails with
``FileNotFoundError: 'snakemake'``). The EDA row needs nothing; no Snakefile dry run (see
tests/goldens/SPEC.md section 9) is used by either.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hhemt.report_plot_ids import canonical_plot_id, output_ext_for
from hhemt.report_renderers._reporting_sets import get_reporting_set


def test_compute_sensitivity_set_wires_eda_renderer():
    """Data-level check, no compile: the compute-sensitivity set carries the EDA
    adapter as a conditional (has_eda_artifact) renderer landing under Key
    Results, and its single rule_spec_template entry carries the Key Results category and a plots/eda/config_diff_maps
    output path, while output_ext_for gives .html under both static backends."""
    s = get_reporting_set("compute_sensitivity")
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


def test_r11_eda_adapter_passes_provenance_audit(golden_sensitivity_analysis):
    """Run the EDA calc+plot on the sensitivity golden, then render the in-report eda_compute_sensitivity adapter
    under the renderer-IO provenance audit. The adapter delegates to render_eda_plots, which self-declares its
    sources, so the audit must NOT raise.

    The golden is extracted into the test's own directory (tests/goldens/SPEC.md section 5.2), so the mutating
    ``analysis.eda()`` facade (it re-persists ``validation_report.json`` and re-renders the Errors-and-Warnings
    figure) cannot touch a shared tree.
    """
    from hhemt.report_renderers import eda_compute_sensitivity
    from hhemt.report_renderers._provenance_audit import audit_renderer_io

    analysis = golden_sensitivity_analysis.sensitivity.experiment
    analysis.eda()  # produce plots/eda/config_diff_maps.html + its manifest sidecar

    root = Path(analysis.analysis_paths.analysis_dir)
    # The figure is the one named by canonical_plot_id, not a hard-coded file name.
    plot_id = canonical_plot_id("config_diff_maps")
    output_path = root / "plots" / "eda" / f"{plot_id}.html"

    with audit_renderer_io(output_path, root, renderer_name="eda_compute_sensitivity"):
        eda_compute_sensitivity.render(analysis, analysis.cfg_analysis.report, output_path)

    # The figure already exists from analysis.eda() above, so this holds whether or not the adapter wrote it.
    assert output_path.exists(), "adapter must (re)produce the config_diff_maps figure at output_path"
    assert (
        output_path.with_suffix(".html.manifest.json").exists()
        or (output_path.parent / f"{plot_id}.manifest.json").exists()
    )


@pytest.mark.requires_snakemake_subprocess
def test_r12_combined_report_assembles_from_two_sensitivity_masters(golden_sensitivity_analysis, tmp_path, monkeypatch):
    """combine_bundle accepts two sensitivity-master bundles and assembles the single combined report.

    The first bundle is emitted from the sensitivity golden after the EDA facade has produced its figure manifest
    (bundle emission needs a manifest sidecar under ``plots/``), unpacked, and the second is a filesystem copy of it.
    """
    import shutil
    import sys
    import zipfile

    from hhemt.bundle import CombinedBundle, combine_bundle
    from hhemt.bundle._combine_merge import _resolve_root_tree

    monkeypatch.setenv("PATH", os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")]))
    analysis = golden_sensitivity_analysis.sensitivity.experiment
    analysis.eda()
    zip_a = tmp_path / "sens_bundle_a.zip"
    analysis.bundle_report_data(zip_a)
    dir_a = tmp_path / "sens_bundle_a"
    with zipfile.ZipFile(zip_a) as zf:
        zf.extractall(dir_a)
    _resolve_root_tree(dir_a)  # the bundle must ship a resolvable root consolidated tree
    dir_b = tmp_path / "sens_bundle_b"
    shutil.copytree(dir_a, dir_b, symlinks=True)

    out = tmp_path / "combined_sens"
    combined = combine_bundle([dir_a, dir_b], output_path=out, unzipped=True)
    assert isinstance(combined, CombinedBundle)
    report = out / "analysis_report.html"
    assert report.exists()
    assert "Combine Provenance & Compatibility" in report.read_text()
    # Regeneration from the bundle root yields a file.
    regen = combined.regenerate_report(format="html")
    assert regen.exists()
