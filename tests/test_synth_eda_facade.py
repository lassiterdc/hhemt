"""End-to-end tests for analysis.eda() and Bundle.eda().

Both rows (the two tests below) run on the committed sensitivity golden (``tests/fixtures/golden_trees.py``), which
carries the summary layer the EDA calculation reads and the consolidated tiers (tests/goldens/SPEC.md section 1), so
the facades run in-process with no solver compiled or executed and no Snakemake process started.

None of the reprocess prerequisites are copied. These tests call ``eda()`` and ``bundle_report_data()`` directly, so no
Snakefile, no ``python`` on ``PATH``, no child-interpreter identity shim and no build-gate declaration is used.
"""

from __future__ import annotations

from pathlib import Path

from hhemt.bundle import Bundle
from hhemt.eda import EdaReportResult


def test_analysis_eda_end_to_end(golden_sensitivity_analysis):
    """analysis.eda() runs calc->plots->notebook(+best-effort HTML), returns a populated EdaReportResult."""
    analysis = golden_sensitivity_analysis.sensitivity.experiment
    result = analysis.eda()
    assert isinstance(result, EdaReportResult)
    # The notebook is required; report_path may be None because export_eda_html returns None on any failure.
    assert result.notebook_path is not None and result.notebook_path.exists()
    assert result.notebook_path.name == "eda.ipynb"
    assert result.report_path is None or result.report_path.name == "eda_report.html"
    assert result.plot_paths and all(p.parent.name == "eda" for p in result.plot_paths)
    # Look for the Plotly figure in the per-plot HTML files, which render_eda_plots writes in-process,
    # not only in the aggregate HTML: when notebook execution fails the aggregate is dropped
    # (report_path is None) and the aggregate-HTML check below would then
    # inspect no figure at all.
    eda_html_plots = [p for p in result.plot_paths if p.suffix == ".html"]
    assert eda_html_plots, "no plots/eda/*.html artifact produced"
    assert any(("plotly-graph-div" in p.read_text() or "Plotly.newPlot" in p.read_text()) for p in eda_html_plots), (
        "no plots/eda/*.html artifact carries a rendered Plotly figure"
    )
    # When the best-effort aggregate HTML also exists, it too must carry a figure.
    if result.report_path is not None:
        html = result.report_path.read_text()
        assert "plotly-graph-div" in html or "Plotly.newPlot" in html
    assert result.verdicts  # calc-stage verdicts; analysis.eda() runs calc (the bundle test skips it)


def test_bundle_eda_from_bundle(golden_sensitivity_analysis, tmp_path):
    """Bundle.eda(plots_only=True) re-renders the EDA notebook (and best-effort HTML report).

    The bundle it renders from is emitted AFTER eda().
    """
    analysis = golden_sensitivity_analysis.sensitivity.experiment
    analysis.eda()  # calc + plots; config_diff_maps declares the consolidated sensitivity_datatree.zarr
    bundle_path = analysis.bundle_report_data()  # harvest carries the declared tree source + plots/eda/
    bundle = Bundle.from_directory(bundle_path if bundle_path.is_dir() else _unpack(bundle_path, tmp_path))
    result = bundle.eda(plots_only=True)
    assert result.notebook_path is not None and result.notebook_path.exists()
    assert result.report_path is None or result.report_path.exists()
    assert result.verdicts == []  # calc skipped on the bundle side


def _unpack(zip_path: Path, dest: Path) -> Path:
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest
