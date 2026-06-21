"""Iter 9 Phase 7: tests for the analysis-validation report system.

Two layers of testing:

1. **Synthetic-ValidationReport tests** (test_renders_*): build a known
   ValidationReport with hand-crafted CheckResult instances representing
   every failure mode, pass it through the renderer's internal helpers,
   verify the rendered HTML contains the expected scenario × stage ×
   resource rows. This isolates renderer behavior from the complex analysis
   initialization (which re-detects state from disk and overwrites
   directly-mutated log fields).

2. **Real-analysis tests** (test_validation_report_*_failing): build a
   cloned analysis from one of the cached synth fixtures, inject failures
   the analysis init can't undo (file deletions for analysis-summary
   checks; see ``_failing_fixture_helpers.inject_*_at_paths``), assert
   the validator detects them. Some failure modes (compilation flag,
   per-scenario log fields) are re-overwritten by the analysis __init__'s
   resume-detection and aren't reliably triggerable via this path; those
   are covered by the synthetic tests above.
"""

from __future__ import annotations

import pytest

from hhemt.analysis_validation import (
    CheckResult,
    ValidationReport,
    validate_analysis,
)
from hhemt.config.report import DEFAULT_REPORT_CONFIG
from hhemt.report_renderers.errors_and_warnings import (
    _render_aggregate_table,
    _render_granular_failures_table,
    _render_overall_banner,
    _render_resource_mismatches_table,
    _render_system_level_table,
    render,
)

from tests._failing_fixture_helpers import (
    construct_analysis_from_paths,
    inject_multi_sim_failures_at_paths,
    inject_sensitivity_failures_at_paths,
    prepare_clone_dir,
)

pytestmark = pytest.mark.requires_snakemake_subprocess


# ---------------------------------------------------------------------------
# Synthetic ValidationReport tests (deterministic; isolated from analysis init)
# ---------------------------------------------------------------------------


def _synthetic_report() -> ValidationReport:
    """Build a ValidationReport covering every failure mode + a passing one."""
    return ValidationReport(checks=[
        CheckResult(
            name="System setup",
            level="system",
            passed=False,
            summary="System setup FAILED (1 issue(s))",
            details=[{"detail": "TRITON-SWMM compilation failed"}],
        ),
        CheckResult(
            name="Scenarios setup",
            level="aggregate",
            passed=False,
            summary="Scenario setup failed for 1 of 3 scenarios",
            details=[{"sa_id": "sa_0", "scenario": "event_index.0", "scenario_dir": "/path", "detail": "scenario not created"}],
        ),
        CheckResult(
            name="Scenarios ran",
            level="aggregate",
            passed=False,
            summary="Simulation failed for 1 of 3 scenarios",
            details=[{"sa_id": "sa_1", "scenario": "event_index.0", "scenario_dir": "/path", "detail": "simulation did not complete"}],
        ),
        CheckResult(
            name="Timeseries processed",
            level="aggregate",
            passed=False,
            summary="Timeseries processing failed for 1 entries",
            details=[{"sa_id": "sa_2", "scenario": "event_index.0", "scenario_dir": "/path", "detail": "TRITON ts not processed"}],
        ),
        CheckResult(
            name="Analysis summaries created",
            level="system",
            passed=False,
            summary="Analysis summaries missing (1 item(s))",
            details=[{"detail": "TRITONSWMM TRITON summary missing"}],
        ),
        CheckResult(
            name="scenario_status.csv created",
            level="system",
            passed=True,
            summary="scenario_status.csv OK (3 rows)",
        ),
        CheckResult(
            name="Resource usage matches config",
            level="resource",
            passed=False,
            summary="Resource mismatches in 1 scenario(s)",
            details=[{"scenario": "sa_3 / event_index.0", "scenario_dir": "/path", "resource": "OMP threads", "expected": 4, "actual": 1, "detail": "OMP threads: expected 4, actual 1"}],
        ),
    ])


def test_synthetic_report_overall_passed_false():
    report = _synthetic_report()
    assert not report.overall_passed
    assert sum(1 for c in report.checks if c.passed) == 1


def test_synthetic_report_granular_failures_aggregated():
    """granular_failures collects per-scenario rows from aggregate-level checks."""
    report = _synthetic_report()
    rows = report.granular_failures
    sa_ids = {r.get("sa_id") for r in rows}
    stages = {r.get("stage") for r in rows}
    assert {"sa_0", "sa_1", "sa_2"} == sa_ids
    assert {"Scenarios setup", "Scenarios ran", "Timeseries processed"} == stages


def test_renders_overall_banner_failure():
    html = _render_overall_banner(_synthetic_report())
    assert 'class="banner fail"' in html
    assert "6 of 7 checks failed" in html


def test_renders_system_level_table_marks_pass_and_fail():
    report = _synthetic_report()
    html = _render_system_level_table(report.by_level["system"])
    assert "System-Level Checks" in html
    assert "TRITON-SWMM compilation failed" in html
    assert 'class="fail"' in html
    assert 'class="pass"' in html  # scenario_status.csv check passes


def test_renders_aggregate_table_has_three_failed_rows():
    report = _synthetic_report()
    html = _render_aggregate_table(report.by_level["aggregate"])
    assert "Aggregate Per-Scenario Checks" in html
    assert "Scenarios setup" in html
    assert "Scenarios ran" in html
    assert "Timeseries processed" in html
    assert html.count('class="fail"') == 3


def test_renders_granular_failures_table_groups_by_sa_id():
    report = _synthetic_report()
    html = _render_granular_failures_table(report.granular_failures)
    assert "Granular Per-Scenario Failures" in html
    for sa_id in ["sa_0", "sa_1", "sa_2"]:
        assert sa_id in html
    # Each scenario label uses "sa_X / scenario_name" format
    assert "sa_0 / event_index.0" in html


def test_renders_resource_mismatches_table_has_omp_row():
    report = _synthetic_report()
    html = _render_resource_mismatches_table(report.by_level["resource"])
    assert "Resource-Utilization Mismatches" in html
    assert "OMP threads" in html
    assert "<td>4</td>" in html
    assert "<td>1</td>" in html


def test_renders_full_html_doc(tmp_path):
    """End-to-end: render a synthetic ValidationReport via the public render API
    against a minimal-stub analysis (only cfg_analysis.analysis_id is needed)."""
    import json
    from dataclasses import asdict
    from unittest.mock import MagicMock

    from hhemt.analysis_validation import _VALIDATION_REPORT_FILENAME

    fake_analysis = MagicMock()
    fake_analysis.cfg_analysis.analysis_id = "demo_failing_synth"
    # Option-D: the renderer reads the persisted validation_report.json via
    # load_validation_report — it does NOT call validate_analysis at render time
    # (that whole-tree read would trip the renderer-IO provenance audit). So
    # persist the synthetic report to disk (the canonical
    # persist_validation_report shape) and point the fake analysis_dir at it.
    fake_analysis.analysis_paths.analysis_dir = tmp_path
    report = _synthetic_report()
    (tmp_path / _VALIDATION_REPORT_FILENAME).write_text(
        json.dumps({"checks": [asdict(c) for c in report.checks]})
    )

    out_path = tmp_path / "ew.html"
    render(fake_analysis, DEFAULT_REPORT_CONFIG, out_path)

    assert out_path.exists() and out_path.stat().st_size > 0
    html = out_path.read_text()
    assert "Errors and Warnings — demo_failing_synth" in html
    assert "System-Level Checks" in html
    assert "Aggregate Per-Scenario Checks" in html
    assert "Granular Per-Scenario Failures" in html
    assert "Resource-Utilization Mismatches" in html
    assert "OMP threads" in html


# ---------------------------------------------------------------------------
# Real-analysis tests (use cloned cached fixture; assert what's reliably triggerable)
# ---------------------------------------------------------------------------


@pytest.fixture
def failing_synth_multi_sim_analysis(synth_multi_sim_analysis_cached, tmp_path):
    """Clone synth_multi_sim cached fixture, inject failures BEFORE construction."""
    paths = prepare_clone_dir(synth_multi_sim_analysis_cached, tmp_path)
    inject_multi_sim_failures_at_paths(paths)
    return construct_analysis_from_paths(paths)


@pytest.fixture
def failing_synth_sensitivity_analysis(synth_sensitivity_analysis_cached, tmp_path):
    """Clone synth_sensitivity cached fixture, inject failures BEFORE construction."""
    paths = prepare_clone_dir(synth_sensitivity_analysis_cached, tmp_path)
    inject_sensitivity_failures_at_paths(paths)
    return construct_analysis_from_paths(paths)


def test_validation_report_multi_sim_failing_disk_mutations(failing_synth_multi_sim_analysis):
    """Disk mutations the analysis init can't undo: file deletions for analysis-summary checks."""
    report = validate_analysis(failing_synth_multi_sim_analysis)
    assert not report.overall_passed
    failed_check_names = {c.name for c in report.checks if not c.passed}
    # File-deletion mutations are checked at validation time (not at init):
    assert "Analysis summaries created" in failed_check_names


def test_validation_report_sensitivity_failing_disk_mutations(failing_synth_sensitivity_analysis):
    """Sensitivity equivalent: deleted sensitivity_datatree.zarr should fail summaries check."""
    report = validate_analysis(failing_synth_sensitivity_analysis)
    assert not report.overall_passed
    failed_check_names = {c.name for c in report.checks if not c.passed}
    assert "Analysis summaries created" in failed_check_names


def test_errors_and_warnings_renders_html_multi_sim_real(failing_synth_multi_sim_analysis, tmp_path):
    """End-to-end render against a real (cloned + mutated) analysis."""
    out_path = tmp_path / "ew_multi_sim.html"
    render(failing_synth_multi_sim_analysis, DEFAULT_REPORT_CONFIG, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0
    html = out_path.read_text()
    assert "Errors and Warnings" in html
    assert "Analysis summaries created" in html


def test_errors_and_warnings_renders_html_sensitivity_real(failing_synth_sensitivity_analysis, tmp_path):
    """End-to-end render against a real (cloned + mutated) sensitivity analysis."""
    out_path = tmp_path / "ew_sensitivity.html"
    render(failing_synth_sensitivity_analysis, DEFAULT_REPORT_CONFIG, out_path)
    assert out_path.exists() and out_path.stat().st_size > 0
    html = out_path.read_text()
    assert "Errors and Warnings" in html
    assert "Analysis summaries created" in html
