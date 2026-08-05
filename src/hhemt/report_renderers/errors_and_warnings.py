"""Errors and Warnings sidebar renderer.

Reads the PERSISTED `validation_report.json` read-model via `load_validation_report`
(never `validate_analysis()` at render time — Gotcha 53 Class-Y) and renders the resulting
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
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.analysis_validation import CheckResult, ValidationReport
    from hhemt.config.report import report_config


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
td.pass-qualified { color: #9A6700; font-weight: 600; text-align: center; width: 60px; }
td.na { color: #6E7781; font-weight: 600; text-align: center; width: 60px; }
span.floor-note { color: #6E7781; font-weight: 400; font-style: italic; }
.banner { padding: 10px 14px; border-radius: 6px; margin: 10px 0 18px;
          font-weight: 600; font-size: 14px; }
.banner.pass { background-color: #DDEEDD; color: #1F7A1F; border: 1px solid #1F7A1F; }
.banner.fail { background-color: #F4D4D4; color: #B11E1E; border: 1px solid #B11E1E; }
.banner.info { background-color: #E5EBF5; color: #232D4B; border: 1px solid #232D4B; }
"""


def _render_overall_banner(report: ValidationReport) -> str:
    n_total = len(report.checks)
    applicable = [c for c in report.checks if getattr(c, "applicable", True)]
    n_na = n_total - len(applicable)
    n_passed = sum(1 for c in applicable if c.passed)
    n_qualified = sum(
        1
        for c in applicable
        if c.passed and getattr(c, "instrument", None) not in (None, "raw_rasters")
    )
    if report.overall_passed:
        # Do NOT pool qualified with unqualified passes: a pass computed from the
        # derived summary tier cannot see below its detection floor, so folding it
        # into "all checks passed" restates the defect this banner exists to expose.
        bits = [f"{n_passed - n_qualified} verified at full precision"]
        if n_qualified:
            bits.append(f"{n_qualified} verified only to the derived-summary floor")
        if n_na:
            bits.append(f"{n_na} not applicable")
        cls = "pass" if not (n_qualified or n_na) else "info"
        return f'<div class="banner {cls}">✓ {n_total} checks: ' + "; ".join(bits) + ".</div>"
    n_failed = len(applicable) - n_passed
    return (
        f'<div class="banner fail">✗ {n_failed} of {len(applicable)} applicable '
        "checks failed. See tables below for details.</div>"
    )


def _status_of(c: CheckResult) -> tuple[str, str, str]:
    """(css_class, glyph, qualifier) for one check. Four outcomes, not two.

    A pass from a derived tier discloses its detection floor inline, because at
    float32 (max_wlevel_m) a true 2.4e-08 difference quantizes to ~1.19e-07 and a
    true 1.68e-13 rounds to exactly 0.0 -- so "identical" there means "identical to
    within float32 rounding", which is not what an unqualified green tick says.

    An UNDECLARED instrument is NOT "precision unknown" -- it means the check made
    no precision claim. Most checks are existence/status/config assertions
    ("scenario_status.csv created", "System setup") that perform no numeric
    comparison and therefore have no detection floor; a float32 disclaimer on those
    is a category error. Qualify only where a check DECLARED that it compared
    numerically at a coarser-than-raw resolution. This is the same principle the
    tri-state itself serves, one level up: a disclaimer that fires on every row is
    one no reader attends to, which would destroy the signal for the checks that
    genuinely need it. A check that DOES compare numerically is responsible for
    stamping its instrument, and a regression test pins the one that does today.

    getattr-with-default throughout: a validation_report.json written before these
    fields existed deserializes without them, and must keep rendering.
    """
    if not getattr(c, "applicable", True):
        return ("na", "N/A", "")
    if not c.passed:
        return ("fail", "✗", "")
    instrument = getattr(c, "instrument", None)
    if instrument is None or instrument == "raw_rasters":
        return ("pass", "✓", "")
    floor = getattr(c, "detection_floor", None)
    if floor is None:
        return ("pass-qualified", "✓", f"verified only at the {instrument} tier")
    return ("pass-qualified", "✓", f"identical only to within {floor:.3g} (derived-summary floor)")


def _render_system_level_table(checks: list[CheckResult]) -> str:
    if not checks:
        return ""
    rows = []
    for c in checks:
        status_cls, status_glyph, qualifier = _status_of(c)
        # Show the summary for both pass and fail; on fail, also list per-issue details
        detail_text = c.summary
        if qualifier:
            detail_text += f'<br><span class="floor-note">{qualifier}</span>'
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
        status_cls, status_glyph, qualifier = _status_of(c)
        _summary = c.summary
        if qualifier:
            _summary += f'<br><span class="floor-note">{qualifier}</span>'
        rows.append(f'<tr><td>{c.name}</td><td class="{status_cls}">{status_glyph}</td><td>{_summary}</td></tr>')
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
        from hhemt.report_renderers._static_backend_warning import (
            warn_no_plotly_branch,
        )

        warn_no_plotly_branch("errors_and_warnings")

    from hhemt.analysis_validation import _VALIDATION_REPORT_FILENAME, load_validation_report
    from hhemt.report_renderers._figure_emission import emit_plot_with_sources
    from hhemt.report_renderers._provenance import ProvenanceLog, ProvenanceRef

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
