"""Errors and Warnings sidebar renderer.

Reads the PERSISTED `validation_report.json` read-model via `load_validation_report`
(never `validate_analysis()` at render time — Gotcha 53 Class-Y) and renders the resulting
ValidationReport as an inline-styled HTML document organized into 4 sections
per the user's requested grouping:

1. Overall pass/fail banner.
2. System-Level Checks (compilation, summaries, CSV integrity).
3. Aggregate Per-Scenario Checks (N of M setup / ran / processed).
4. Resource-Utilization Mismatches (status table always; per-scenario mismatch table
   only when a mismatch exists -- no "all clear" banner, it duplicates the status row).
5. Granular Per-Scenario Failures (table when failures exist; the whole section is
   ABSENT when there are none -- it applies no check, so it renders nothing to placehold).

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




#: Display vocabulary for the two `Check` tables ONLY -- the `system`-level and
#: `resource`-level checks. Keyed on `CheckResult.name`, which stays the stable MACHINE
#: key: `cross_experiment_errors_and_warnings.py` joins children across bundles on that
#: string, so a producer-side rename splits the combined matrix into two rows with `-`
#: cells whenever bundles from either side of the rename are combined. The display name
#: and the description are therefore RENDERER-side, and the producer is untouched.
#:
#: Convention, enforced by tests/test_iter7_check_vocabulary.py: the display name is a
#: Sentence-case noun phrase naming the SUBJECT under test, carrying no verb and no raw
#: file/identifier token. The verb belongs in the description, which states what is
#: LOOKED FOR rather than what the check is called.
#:
#: NOT extended to the `aggregate` (`Stage`) table by ruling: the user scoped cross-table
#: naming consistency to the `Check` tables. Eleven further live names route there, five
#: of them minted in `src/hhemt/eda/` and read back via `_read_persisted_eda_verdicts`.
_CHECK_VOCABULARY: dict[str, tuple[str, str]] = {
    "System setup": (
        "System setup",
        "Every enabled model compiled (native mode), and the processed DEM and "
        "Manning roughness rasters exist with matching, correctly-shaped geometry.",
    ),
    "Analysis summaries created": (
        "Analysis summaries",
        "Every consolidated DataTree the analysis owes is present on disk — the "
        "analysis tree, and on a sensitivity master the master tree plus each "
        "sub-analysis tree.",
    ),
    "scenario_status.csv created": (
        "Scenario status export",
        "The per-scenario status export exists, parses, and carries every resource "
        "and performance column its downstream readers require.",
    ),
    "Resource usage matches config": (
        "Resource usage",
        "Every scenario ran on the MPI ranks, OMP threads, GPUs, GPU backend and "
        "build type its configuration requested.",
    ),
}


def _vocab(c: CheckResult) -> tuple[str, str]:
    """(display_name, description) for one check, degrading to the raw name.

    The fallback is NOT dead: a bundle produced by a future toolkit can carry a
    check name this vocabulary predates, and a KeyError there would take the whole
    sidebar down for a cosmetic gap. Falling back to the machine name renders a
    correct-if-plain row, matching the graceful-absent posture the rest of this
    module and `load_validation_report` already take.
    """
    return _CHECK_VOCABULARY.get(c.name, (c.name, ""))


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
        # I7-4: the `Check` cell now DESCRIBES what is looked for; the check's name moves
        # to its own column. Both come from _CHECK_VOCABULARY so the two `Check` tables
        # cannot drift from each other.
        display_name, description = _vocab(c)
        # Show the summary for both pass and fail; on fail, also list per-issue details.
        # `Details` keeps a distinct job from the new description column: this is a
        # per-RUN fact (this run's issues, this run's detection floor), the description
        # is a per-CHECK fact. Both branches below are preserved verbatim -- the failing
        # branch is exercised by no delivered arm, so it must survive by construction.
        detail_text = c.summary
        if qualifier:
            detail_text += f'<br><span class="floor-note">{qualifier}</span>'
        if not c.passed and c.details:
            detail_lines = [d.get("detail", "") for d in c.details]
            detail_text = c.summary + "<br>" + "<br>".join(f"&nbsp;&nbsp;• {d}" for d in detail_lines)
        rows.append(
            f"<tr><td>{display_name}</td><td>{description}</td>"
            f'<td class="{status_cls}">{status_glyph}</td><td>{detail_text}</td></tr>'
        )
    return (
        "<h3>System-Level Checks</h3>\n"
        "<table>\n"
        "  <thead><tr><th>Name</th><th>Check</th><th>Status</th><th>Details</th></tr></thead>\n"
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
        # I7-3 / ledger P9: a section that does not apply is ABSENT, not placeheld. The
        # empty string is filtered by render()'s `if b` join, so heading AND banner both
        # vanish with no call-site edit. Measured: all four delivered arms have every
        # check passing, so `granular_failures` is empty and this is the ONLY branch the
        # delivered generation reaches -- the green "No per-scenario failures" banner was
        # on every page, which is what the user was looking at.
        return ""
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
    # Resource-level checks reached NO status row: this section was written as a
    # mismatch table (it read `c.details` off FAILING checks only), so the single
    # `level="resource"` check -- `Resource usage matches config` -- was counted in
    # the banner's 15 and rendered as one of only 14 rows. That is a different defect
    # from the three aggregate-level siblings, which render as soon as their level is
    # set. Emit the status rows first, then the mismatch detail. Reclassifying the
    # check to `level="system"` would ALSO have made it render -- and would have
    # emptied `by_level["resource"]`, deleting the per-scenario expected-vs-actual
    # detail table below on failure. The status table is the fix that keeps both.
    status_rows = "\n    ".join(
        '<tr><td>{n}</td><td>{w}</td><td class="{cls}">{glyph}</td><td>{d}</td></tr>'.format(
            n=_vocab(c)[0],
            w=_vocab(c)[1],
            cls=_status_of(c)[0],
            glyph=_status_of(c)[1],
            d=c.summary,
        )
        for c in checks
    )
    status_table = (
        "<table>\n"
        "  <thead><tr><th>Name</th><th>Check</th><th>Status</th><th>Details</th></tr></thead>\n"
        "  <tbody>\n    " + status_rows + "\n  </tbody>\n</table>"
    )
    if not all_issues:
        # I7-3: the "No resource mismatches" banner is deleted as redundant with the
        # status row directly above it, which already renders `Resource usage / OK /
        # All scenarios used expected compute resources`. The SECTION and its status
        # table stay: `Resource usage matches config` is a `Check`-table row, so
        # omitting the section here would omit a check. That is the principled
        # asymmetry with the granular section below, which carries no check at all.
        return "<h3>Resource-Utilization Mismatches</h3>\n" + status_table
    rows = []
    for d in all_issues:
        scenario = d.get("scenario", d.get("scenario_dir", ""))
        resource = d.get("resource", "")
        expected = d.get("expected", "")
        actual = d.get("actual", "")
        rows.append(f"<tr><td>{scenario}</td><td>{resource}</td><td>{expected}</td><td>{actual}</td></tr>")
    # The FAILING branch needs the status row as much as the passing one -- more so.
    # This is the only branch a run with a real resource mismatch reaches, and it is
    # the branch no delivered arm exercises (all four pass), so landing only the
    # early-return above would leave the check invisible on precisely the runs the
    # section exists to serve, with no artifact revealing it.
    return (
        "<h3>Resource-Utilization Mismatches</h3>\n"
        + status_table
        + "\n<table>\n"
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
    # I7-3: `Granular Per-Scenario Failures` moves LAST. It is the one section that
    # applies no check -- it re-presents the detail rows of checks already listed above --
    # so putting it between the aggregate and resource tables separated two check tables
    # with a non-check table. Check-bearing sections are now contiguous.
    body_parts = [
        _render_overall_banner(report),
        _render_system_level_table(by_level.get("system", [])),
        _render_aggregate_table(by_level.get("aggregate", [])),
        _render_resource_mismatches_table(by_level.get("resource", [])),
        _render_granular_failures_table(report.granular_failures),
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
