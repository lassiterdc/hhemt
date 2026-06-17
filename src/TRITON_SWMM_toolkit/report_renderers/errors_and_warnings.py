"""Errors and Warnings sidebar renderer.

Calls `analysis_validation.validate_analysis()` and renders the resulting
ValidationReport as an inline-styled HTML document organized into 4 sections
per the user's requested grouping:

1. Overall pass/fail banner.
2. System-Level Checks (compilation, summaries, CSV integrity).
3. Aggregate Per-Scenario Checks (N of M setup / ran / processed).
4. Granular Per-Scenario Failures (table; omitted with "no failures" banner if empty).
5. Resource-Utilization Mismatches (table; omitted with "no mismatches" banner if empty).

Snakemake's report engine renders `.html` outputs in an iframe; the embedded
HTML must carry inline `<style>` for any visual styling (per snakemake-specialist
consult 18:09).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.analysis_validation import CheckResult, ValidationReport
    from TRITON_SWMM_toolkit.config.report import report_config


_INLINE_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       padding: 12px; color: #333; margin: 0; }
h2 { color: #232D4B; border-bottom: 2px solid #232D4B; padding-bottom: 4px; margin-top: 0; }
h3 { color: #232D4B; margin-top: 24px; margin-bottom: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 8px; }
th, td { padding: 6px 10px; border: 1px solid #DADADA; text-align: left; vertical-align: top; }
th { background-color: #232D4B; color: white; font-weight: 600; }
tr:nth-child(even) td { background-color: #F1F1EF; }
tr:hover td { background-color: #FFE4C4; }
td.pass { color: #1F7A1F; font-weight: 600; text-align: center; width: 60px; }
td.fail { color: #B11E1E; font-weight: 600; text-align: center; width: 60px; }
.banner { padding: 10px 14px; border-radius: 6px; margin: 10px 0 18px;
          font-weight: 600; font-size: 14px; }
.banner.pass { background-color: #DDEEDD; color: #1F7A1F; border: 1px solid #1F7A1F; }
.banner.fail { background-color: #F4D4D4; color: #B11E1E; border: 1px solid #B11E1E; }
.banner.info { background-color: #E5EBF5; color: #232D4B; border: 1px solid #232D4B; }
"""


def _render_overall_banner(report: ValidationReport) -> str:
    n_total = len(report.checks)
    n_passed = sum(1 for c in report.checks if c.passed)
    if report.overall_passed:
        return f'<div class="banner pass">✓ All {n_total} checks passed.</div>'
    n_failed = n_total - n_passed
    return f'<div class="banner fail">✗ {n_failed} of {n_total} checks failed. See tables below for details.</div>'


def _render_system_level_table(checks: list[CheckResult]) -> str:
    if not checks:
        return ""
    rows = []
    for c in checks:
        status_cls = "pass" if c.passed else "fail"
        status_glyph = "✓" if c.passed else "✗"
        # Show the summary for both pass and fail; on fail, also list per-issue details
        detail_text = c.summary
        if not c.passed and c.details:
            detail_lines = [d.get("detail", "") for d in c.details]
            detail_text = c.summary + "<br>" + "<br>".join(f"&nbsp;&nbsp;• {d}" for d in detail_lines)
        rows.append(f'<tr><td>{c.name}</td><td class="{status_cls}">{status_glyph}</td><td>{detail_text}</td></tr>')
    return (
        "<h3>System-Level Checks</h3>\n"
        "<table>\n"
        "  <thead><tr><th>Check</th><th>Status</th><th>Details</th></tr></thead>\n"
        "  <tbody>\n    " + "\n    ".join(rows) + "\n  </tbody>\n</table>"
    )


def _render_aggregate_table(checks: list[CheckResult]) -> str:
    if not checks:
        return ""
    rows = []
    for c in checks:
        status_cls = "pass" if c.passed else "fail"
        status_glyph = "✓" if c.passed else "✗"
        rows.append(f'<tr><td>{c.name}</td><td class="{status_cls}">{status_glyph}</td><td>{c.summary}</td></tr>')
    return (
        "<h3>Aggregate Per-Scenario Checks</h3>\n"
        "<table>\n"
        "  <thead><tr><th>Stage</th><th>Status</th><th>Summary</th></tr></thead>\n"
        "  <tbody>\n    " + "\n    ".join(rows) + "\n  </tbody>\n</table>"
    )


def _render_granular_failures_table(granular: list[dict]) -> str:
    if not granular:
        return '<h3>Granular Per-Scenario Failures</h3>\n<div class="banner pass">✓ No per-scenario failures.</div>'
    rows = []
    for d in granular:
        sa_id = d.get("sa_id", "")
        scenario = d.get("scenario", d.get("scenario_dir", ""))
        scenario_label = f"{sa_id} / {scenario}" if sa_id else scenario
        stage = d.get("stage", "")
        detail = d.get("detail", "")
        rows.append(f"<tr><td>{scenario_label}</td><td>{stage}</td><td>{detail}</td></tr>")
    return (
        "<h3>Granular Per-Scenario Failures</h3>\n"
        "<table>\n"
        "  <thead><tr><th>Scenario</th><th>Stage</th><th>Detail</th></tr></thead>\n"
        "  <tbody>\n    " + "\n    ".join(rows) + "\n  </tbody>\n</table>"
    )


def _render_resource_mismatches_table(checks: list[CheckResult]) -> str:
    # checks is list of resource-level CheckResults (typically just the one
    # `Resource usage matches config` check). When that check failed, its
    # `details` list carries the per-scenario per-resource mismatch records.
    if not checks:
        return ""
    all_issues: list[dict] = []
    for c in checks:
        if not c.passed:
            all_issues.extend(c.details)
    if not all_issues:
        return (
            "<h3>Resource-Utilization Mismatches</h3>\n"
            '<div class="banner pass">✓ No resource mismatches — '
            "all scenarios used expected compute resources.</div>"
        )
    rows = []
    for d in all_issues:
        scenario = d.get("scenario", d.get("scenario_dir", ""))
        resource = d.get("resource", "")
        expected = d.get("expected", "")
        actual = d.get("actual", "")
        rows.append(f"<tr><td>{scenario}</td><td>{resource}</td><td>{expected}</td><td>{actual}</td></tr>")
    return (
        "<h3>Resource-Utilization Mismatches</h3>\n"
        "<table>\n"
        "  <thead><tr><th>Scenario</th><th>Resource</th><th>Expected</th><th>Actual</th></tr></thead>\n"
        "  <tbody>\n    " + "\n    ".join(rows) + "\n  </tbody>\n</table>"
    )


def _wrap_html_doc(body: str, analysis_id: str, inline_css: str) -> str:
    return (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<style>{inline_css}</style></head><body>"
        f"<h2>Errors and Warnings — {analysis_id}</h2>"
        f"{body}"
        "</body></html>"
    )


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    """Render the analysis-validation report to output_path (HTML)."""
    static_backend = getattr(
        getattr(report_cfg, "interactive", None),
        "static_backend",
        "plotly",
    )
    if static_backend == "plotly":
        from TRITON_SWMM_toolkit.report_renderers._static_backend_warning import (
            warn_no_plotly_branch,
        )

        warn_no_plotly_branch("errors_and_warnings")

    from TRITON_SWMM_toolkit.analysis_validation import _VALIDATION_REPORT_FILENAME, load_validation_report
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import emit_plot_with_sources
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceLog, ProvenanceRef

    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="validation report (read from persisted validation_report.json, no matplotlib artist)",
    ) as a:
        a.add_channel(
            "data",
            ProvenanceRef(source_path=_VALIDATION_REPORT_FILENAME),
        )
        # Option D (Class-Y resolution): read the persisted read-model artifact
        # produced at consolidation, NOT a render-time validate_analysis() whole-tree
        # inspection. Graceful-absent -> empty report. See analysis_validation.
        report = load_validation_report(analysis)
    by_level = report.by_level
    body_parts = [
        _render_overall_banner(report),
        _render_system_level_table(by_level.get("system", [])),
        _render_aggregate_table(by_level.get("aggregate", [])),
        _render_granular_failures_table(report.granular_failures),
        _render_resource_mismatches_table(by_level.get("resource", [])),
    ]
    analysis_id = str(analysis.cfg_analysis.analysis_id)
    html = _wrap_html_doc(
        "\n".join(b for b in body_parts if b),
        analysis_id,
        report_cfg.errors_and_warnings.render_inline_css(),
    )
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    return emit_plot_with_sources(
        html,
        output_path,
        # Option D: the renderer's sole data source is the persisted ValidationReport
        # read-model (validation_report.json), produced at consolidation. It already
        # folds in the eda/*.verdict.json (validate_analysis appends them at persist
        # time), so the renderer declares exactly one file -> tight audit catch-power
        # + faithful bundle re-render.
        [analysis_dir / _VALIDATION_REPORT_FILENAME],
        analysis_dir=analysis_dir,
        output_format="html",
        manifest_data={
            "renderer": "errors_and_warnings",
            "section_count": sum(1 for b in body_parts if b),
        },
        provenance=prov,
    )
