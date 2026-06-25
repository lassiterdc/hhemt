"""End-to-end tests for analysis.eda() and Bundle.eda() (ADR-10 facades)."""

from __future__ import annotations

from pathlib import Path

from hhemt.bundle import Bundle
from hhemt.eda import EdaReportResult


def test_analysis_eda_end_to_end(synthetic_sensitivity_completed_isolated):
    """analysis.eda() runs calc->plots->notebook(+best-effort HTML), returns a populated EdaReportResult."""
    analysis = synthetic_sensitivity_completed_isolated.master_analysis
    result = analysis.eda()
    assert isinstance(result, EdaReportResult)
    # The notebook is the ALWAYS-present primary artifact (ADR-14); the HTML is best-effort.
    assert result.notebook_path is not None and result.notebook_path.exists()
    assert result.notebook_path.name == "eda.ipynb"
    assert result.report_path is None or result.report_path.name == "eda_report.html"
    # SE Flag 1 guard: when the best-effort HTML export succeeded, it must carry an
    # actual Plotly figure (not just a path list) — the executed notebook renders the
    # seed-figure cell against live variables.
    if result.report_path is not None:
        html = result.report_path.read_text()
        assert "plotly-graph-div" in html or "Plotly.newPlot" in html
    assert result.plot_paths and all(p.parent.name == "eda" for p in result.plot_paths)
    assert result.verdicts  # the cross-sim-identity verdict


def test_bundle_eda_from_bundle(synthetic_sensitivity_completed_isolated, tmp_path):
    """Bundle.eda(plots_only=True) re-renders the doc from a bundle emitted AFTER eda()."""
    analysis = synthetic_sensitivity_completed_isolated.master_analysis
    analysis.eda()  # calc + plots so the eda/<plot_id>.zarr is declared
    bundle_path = analysis.bundle_report_data()  # harvest carries eda/ zarr + plots/eda/
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
