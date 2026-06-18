"""End-to-end tests for analysis.eda() and Bundle.eda() (ADR-10 facades)."""

from __future__ import annotations

from pathlib import Path

from hhemt.bundle import Bundle
from hhemt.eda import EdaReportResult


def test_analysis_eda_end_to_end(synthetic_sensitivity_completed):
    """analysis.eda() runs calc->plots->doc and returns a populated EdaReportResult."""
    analysis = synthetic_sensitivity_completed.master_analysis
    result = analysis.eda()
    assert isinstance(result, EdaReportResult)
    assert result.report_path.exists()
    assert result.report_path.name == "eda_report.html"
    assert result.plot_paths and all(p.parent.name == "eda" for p in result.plot_paths)
    assert result.verdicts  # the cross-sim-identity verdict


def test_bundle_eda_from_bundle(synthetic_sensitivity_completed, tmp_path):
    """Bundle.eda(plots_only=True) re-renders the doc from a bundle emitted AFTER eda()."""
    analysis = synthetic_sensitivity_completed.master_analysis
    analysis.eda()  # calc + plots so the eda/<plot_id>.zarr is declared
    bundle_path = analysis.bundle_report_data()  # harvest carries eda/ zarr + plots/eda/
    bundle = Bundle.from_directory(bundle_path if bundle_path.is_dir() else _unpack(bundle_path, tmp_path))
    result = bundle.eda(plots_only=True)
    assert result.report_path.exists()
    assert result.verdicts == []  # calc skipped on the bundle side


def _unpack(zip_path: Path, dest: Path) -> Path:
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest
