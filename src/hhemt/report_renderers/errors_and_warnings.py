"""Errors and Warnings sidebar renderer.

A finished analysis needs one page saying whether the workflow completed cleanly and, when
it did not, which checks failed and for which scenarios. This module renders that page.

It reads the PERSISTED `validation_report.json` read-model via `load_validation_report`,
never `validate_analysis()` at render time: a render-time call inspects the whole analysis
tree, which a render bundle does not carry, so a bundle could not re-render this page
faithfully. The resulting ValidationReport becomes an inline-styled HTML document in five
sections:

1. Overall pass/fail banner.
2. System-Level Checks (compilation, summaries, CSV integrity).
3. Aggregate Per-Scenario Checks (N of M setup / ran / processed). Carries the same
   Name / Check / Status / Details columns as the other two check tables.
4. Resource-Utilization Mismatches (status table always; per-scenario mismatch table
   only when a mismatch exists -- no "all clear" banner, it duplicates the status row).
5. Granular Per-Scenario Failures (table when failures exist; the whole section is
   ABSENT when there are none -- it applies no check, so it renders nothing to placehold).

Snakemake's report engine renders `.html` outputs in an iframe, so the embedded HTML
must carry inline `<style>` for any visual styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.analysis_validation import CheckResult, ValidationReport
    from hhemt.config.report import report_config


#: Display vocabulary for every check table on this page -- the `system`-, `aggregate`- and
#: `resource`-level checks. Keyed on `CheckResult.name`, which stays the stable MACHINE
#: key: `cross_experiment_errors_and_warnings.py` joins children across bundles on that
#: string, so a producer-side rename splits the combined matrix into two rows with `-`
#: cells whenever bundles from either side of the rename are combined. The display name
#: and the description are therefore RENDERER-side, and the producer is untouched.
#:
#: Convention: the display name is a Sentence-case noun phrase naming the SUBJECT under
#: test, carrying no verb and no raw file/identifier token. The verb belongs in the
#: description, which states what is LOOKED FOR rather than what the check is called.
#: tests/test_iter7_check_vocabulary.py enforces the mechanical half -- Sentence case, no
#: `_` or `.` in the display name, a full-sentence description -- and also requires this map
#: to cover exactly the checks minted at those three levels, including the ones minted in
#: `src/hhemt/eda/` and read back via `_read_persisted_eda_verdicts`, so no check reaches a
#: table without an entry and no entry outlives its check.
_CHECK_VOCABULARY: dict[str, tuple[str, str]] = {
    "System setup": (
        "System setup",
        "Every enabled model is compiled (native mode), and the processed DEM and "
        "Manning roughness rasters exist with matching, correctly-shaped geometry.",
    ),
    "Analysis summaries created": (
        "Analysis summaries",
        "Every consolidated DataTree the analysis owes is present on disk — the "
        "analysis tree, and on a sensitivity master the master tree plus each "
        "member tree.",
    ),
    "scenario_status.csv created": (
        "Scenario status export",
        "The per-scenario status export exists, parses, and carries every resource "
        "and performance column its downstream readers require.",
    ),
    "provenance_completeness": (
        "Version coverage",
        "Every pipeline stage that produced part of this analysis recorded the toolkit "
        "build that ran it, those builds agree with each other, and none of them came "
        "from a modified working tree.",
    ),
    "Resource usage matches config": (
        "Resource usage",
        "Every scenario runs on the MPI ranks, OMP threads, GPUs, GPU backend and "
        "build type its configuration requested.",
    ),
    # --- aggregate-level checks. The aggregate table carries the same Name / Check columns
    # as the system and resource tables, so these need descriptions too, each written from
    # the check's own predicate to the convention stated at the top of this block.
    # (`provenance_completeness`, above, is aggregate-level as well.)
    "Scenarios setup": (
        "Scenario setup",
        "Every scenario the analysis defines exists on disk with its inputs written.",
    ),
    "Scenarios ran": (
        "Scenario runs",
        "Every scenario's simulation runs to completion for each enabled model.",
    ),
    "Timeseries processed": (
        "Timeseries processing",
        "Every scenario carries processed timeseries output for each enabled model.",
    ),
    "Data availability": (
        "Data availability",
        "Every per-scenario artifact class the report reads is present, or its absence is "
        "a deliberate reclamation rather than a loss.",
    ),
    "invalidating-fix registry": (
        "Invalidating-fix registry",
        "No calculation-invalidating fix in the registry applies to the toolkit build that produced this analysis.",
    ),
    "resume validity": (
        "Resume validity",
        "Every resumed coupled sim runs at a solver build with no known resume defect, and "
        "its exchange history replays rather than silently re-initializing.",
    ),
    "Known resume defects": (
        "Known resume defects",
        "The producing solver build carries no defect from the resume-defect registry that "
        "applies to this run's model selection.",
    ),
    "Resume schedule honored": (
        "Resume schedule",
        "Every resumed sim's realized resume count and boundary match the schedule its configuration requested.",
    ),
    "resume event evidence": (
        "Resume evidence",
        "Every sim that attempted a resume left a durable record of each attempt, with no sim "
        "missing that record for some or all of its attempts and no alarm row in its "
        "resume-event record.",
    ),
    "EDA calc ran": (
        "EDA calculation",
        "Every EDA figure enumerated as a report target has a corresponding verdict "
        "artifact, so the calculation actually runs rather than degrading to a placeholder.",
    ),
    "Cross-sim byte-identity": (
        "Cross-sim byte-identity",
        "Members differing only in compute configuration produce bit-identical tracked "
        "variables within each hardware family.",
    ),
    "Raw byte-for-byte identity": (
        "Raw byte-for-byte identity",
        "Each configuration's raw per-timestep rasters are byte-identical to its "
        "within-family reference across every raw output type.",
    ),
    "Rank sensitivity": (
        "Rank sensitivity",
        "Running at N MPI ranks reproduces the single-rank result within the same compute "
        "family, byte-identically or within a disclosed magnitude.",
    ),
    "Resume sensitivity": (
        "Resume sensitivity",
        "A resumed run reproduces its clean counterpart at the same compute configuration, "
        "byte-identically or within a disclosed magnitude.",
    ),
    "Cross-hardware magnitude": (
        "Cross-hardware magnitude",
        "The characterized divergence between one GPU and one serial CPU rank stays within "
        "its disclosed magnitude band.",
    ),
    "forcing tail influence": (
        "Forcing tail influence",
        "No cell reaches its peak water level after that event's forcing ended \u2014 a "
        "regression detector for the window trim, since a simulation running past its "
        "forcing holds the coastal boundary frozen at the last tabulated value.",
    ),
    "Log recoveries": (
        "Log recovery records",
        "No analysis, system, or per-model scenario log document carries a record of having "
        "been rebuilt from a single process's state after failing to parse; every such "
        "recovery is listed, with the quarantine path where the failed bytes were preserved, "
        "because a rebuilt document carries defaults for every field its writer had not set.",
    ),
}

#: The one cell an unregistered check name renders. NOT prose invented for a predicate
#: this module cannot read: a description guessed from a name could state a predicate the
#: check does not have, and a reader acts on a description as the check's own. A name that
#: reaches here came from a persisted `validation_report.json` whose producer we may not
#: have. One fixed string is deterministic, overlaps no registered description, and is the
#: same for every such name.
#:
#: On any tree the vocabulary guard passes this is UNREACHABLE for a toolkit-minted name:
#: tests/test_iter7_check_vocabulary.py fails when a renderable check has no entry. It is
#: reachable only from a bundle produced by a toolkit whose vocabulary is ahead of this one.
_UNDESCRIBED = "No description registered for this check."

#: What the `Details` column may carry, recorded in code. Every check
#: table on this page -- system-level, aggregate, and the resource status table -- renders
#: its last column under this header, and this constant is the single statement of what is
#: admissible there. It is a CONTRACT, not documentation of current behaviour: a future
#: renderer edit that puts something else in the column is violating this line.
#:
#: Admissible, in this order:
#:   1. `CheckResult.summary` -- the per-RUN result sentence. ALWAYS present.
#:   2. On a qualified pass, the precision qualifier from `_status_of` in a
#:      `<span class="floor-note">` -- the detection floor this run's verdict was
#:      computed at. Per-RUN, because the instrument is a property of the run.
#:   3. On a failure, the per-issue `details[].detail` lines as a bulleted list.
#:
#: INADMISSIBLE, and this is the distinction the column exists to preserve: anything that
#: is a per-CHECK fact rather than a per-RUN fact. What a check LOOKS FOR belongs in the
#: `Check` column, sourced from `_CHECK_VOCABULARY`. A `Details` cell that describes the
#: check rather than this run's outcome repeats the `Check` cell beside it, so the row
#: states one per-check fact twice.
_DETAILS_COLUMN_CONTRACT = "summary; optional precision qualifier; per-issue detail lines on failure"


def _vocab(c: CheckResult) -> tuple[str, str]:
    """(display_name, description) for one check, degrading to the raw name.

    The fallback is NOT dead: a bundle produced by a future toolkit can carry a
    check name this vocabulary predates, and a KeyError there would take the whole
    sidebar down for a cosmetic gap. Falling back to the machine name renders a
    correct-if-plain row, matching the graceful-absent posture the rest of this
    module and `load_validation_report` already take.
    """
    return _CHECK_VOCABULARY.get(c.name, (c.name, _UNDESCRIBED))


def _render_overall_banner(report: ValidationReport) -> str:
    # The not-applicable count is deliberately NOT computed here: `render()` states it once,
    # in the omission caption directly beneath this banner, and stating it here as well
    # would print the same figure twice.
    applicable = [c for c in report.checks if getattr(c, "applicable", True)]
    n_passed = sum(1 for c in applicable if c.passed)
    n_qualified = sum(1 for c in applicable if c.passed and getattr(c, "instrument", None) not in (None, "raw_rasters"))
    if report.overall_passed:
        # Deliberately NOT a precision tally: it would spend the banner -- the one line every
        # reader reads -- on a three-way split of a distinction that is already disclosed
        # exactly where it bears, inline beside the row it qualifies (_status_of returns the
        # qualifier; the system and aggregate tables render it). A tally repeated at the top
        # is the redundancy, not the disclosure. A derived-tier pass is still marked, per row,
        # because an unqualified tick over a coarser-than-raw comparison overstates it.
        #
        # The `info` class is used when any pass is qualified, so the banner does not read as
        # an unqualified all-clear. The not-applicable COUNT is stated by the single omission
        # caption `render()` emits, so it appears in exactly one place on the page.
        cls = "pass" if not n_qualified else "info"
        return f'<div class="banner {cls}">✓ {n_passed} of {len(applicable)} applicable checks passed.</div>'
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
    stamping its instrument; several stamp ``"summary_tier"`` today. Not established:
    that a regression test pins every stamper.

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


def _partition_applicable(checks: list[CheckResult]) -> tuple[list[CheckResult], int]:
    """Split off the checks that cannot apply to this analysis. Returns (kept, n_omitted).

    A check whose arm cannot produce the thing being checked would otherwise render a
    full N/A row -- "Resume schedule honored" on a CLEAN arm is N/A on every clean report,
    for every scenario, forever. Those rows are noise proportional to the vocabulary rather
    than to the run.

    Deliberately NOT rendered as a PASS carrying its denominator instead: a green cell over
    a check that examined nothing is a false claim. Omitting the row makes no claim at all.
    What omission WOULD lose is the reader's ability to tell "this check passed" from "this
    check never ran" -- so the page discloses the omitted COUNT once, in the caption
    `render()` places beneath the banner. The count is the disclosure; the rows are the
    noise.

    `applicable` defaults True via getattr: a validation_report.json written before the
    field existed deserializes without it and must keep rendering every row.
    """
    kept = [c for c in checks if getattr(c, "applicable", True)]
    return kept, len(checks) - len(kept)


def _omitted_note(n_omitted: int) -> str:
    """The page's SINGLE omission disclosure. Empty when nothing was cut.

    One short caption, no examples. `render()` calls it exactly once over the whole check
    set; deliberately NOT once per filtered table, because a report omitting checks at two
    levels would then say the same thing twice.

    What it must NOT lose is the COUNT, which is what lets a reader tell "this check
    passed" from "this check never ran". An examined-zero check is N/A rather than a PASS
    carrying its denominator, and an N/A check is omitted from the check tables, so this
    count is how the page still accounts for it.
    """
    if n_omitted <= 0:
        return ""
    plural = "s" if n_omitted != 1 else ""
    return (
        f'\n<p class="floor-note">{n_omitted} check{plural} not applicable to this '
        f"analysis {'were' if n_omitted != 1 else 'was'} omitted.</p>"
    )


def _render_system_level_table(checks: list[CheckResult]) -> str:
    if not checks:
        return ""
    checks, _n_omitted = _partition_applicable(checks)
    if not checks:
        # No per-table caption: `render()` emits ONE omission caption over the whole check
        # set, so a table that filtered every row simply renders nothing rather than
        # repeating the page-level disclosure.
        return ""
    rows = []
    for c in checks:
        status_cls, status_glyph, qualifier = _status_of(c)
        # `Name` and `Check` both come from _CHECK_VOCABULARY through the accessor every
        # check table uses, so no two tables can name or describe one check differently.
        display_name, description = _vocab(c)
        # `Details` carries this run's outcome -- its summary, its detection floor, and on
        # failure each issue -- a per-RUN fact kept apart from the per-CHECK description in
        # the `Check` cell. No passing report reaches the failing branch, so nothing in a
        # passing report would reveal it broken: keep it when editing the passing path.
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
        '<div class="table-scroll">\n<table>\n'
        "  <thead><tr><th>Name</th><th>Check</th><th>Status</th><th>Details</th></tr></thead>\n"
        "  <tbody>\n    " + "\n    ".join(rows) + "\n  </tbody>\n</table>\n</div>"
    )


def _render_aggregate_table(checks: list[CheckResult]) -> str:
    if not checks:
        return ""
    checks, _n_omitted = _partition_applicable(checks)
    if not checks:
        # No per-table caption; render() emits one for the whole page.
        return ""
    rows = []
    for c in checks:
        status_cls, status_glyph, qualifier = _status_of(c)
        # Same columns as the other two check tables, populated from the same registry
        # through the same accessor. The display name replaces the raw `c.name` here too,
        # which is what makes the three tables name one check one way.
        display_name, description = _vocab(c)
        _details = c.summary
        if qualifier:
            _details += f'<br><span class="floor-note">{qualifier}</span>'
        rows.append(
            f"<tr><td>{display_name}</td><td>{description}</td>"
            f'<td class="{status_cls}">{status_glyph}</td><td>{_details}</td></tr>'
        )
    # The header row matches the other two check tables word for word -- `Name`, `Check`,
    # `Status`, `Details` -- because this table's last column carries the same datum
    # (`c.summary`, plus the same optional precision qualifier) that they render under
    # `Details`, and two header words for one datum on one page read as two different things.
    #
    # The `Check` column is fully populated: _CHECK_VOCABULARY covers every aggregate-level
    # check. A blank column WOULD be worse than a naming inconsistency, and what prevents one
    # is the vocabulary guard in tests/test_iter7_check_vocabulary.py, which fails when a
    # renderable check has no entry.
    return (
        "<h3>Aggregate Per-Scenario Checks</h3>\n"
        '<div class="table-scroll">\n<table>\n'
        "  <thead><tr><th>Name</th><th>Check</th><th>Status</th><th>Details</th></tr></thead>\n"
        "  <tbody>\n    " + "\n    ".join(rows) + "\n  </tbody>\n</table>\n</div>"
    )


def _render_granular_failures_table(granular: list[dict]) -> str:
    if not granular:
        # A section that does not apply is ABSENT, not placeheld: it applies no check, so a
        # "no failures" banner would only repeat the overall banner. The empty string is
        # filtered by render()'s `if b` join, so the heading vanishes with no call-site edit.
        # `granular_failures` is empty whenever every check passes, so without this branch
        # every fully passing report would carry a green "No per-scenario failures" banner.
        return ""
    rows = []
    for d in granular:
        member_id = d.get("sa_id", "")
        scenario = d.get("scenario", d.get("scenario_dir", ""))
        scenario_label = f"{member_id} / {scenario}" if member_id else scenario
        stage = d.get("stage", "")
        detail = d.get("detail", "")
        rows.append(f"<tr><td>{scenario_label}</td><td>{stage}</td><td>{detail}</td></tr>")
    return (
        "<h3>Granular Per-Scenario Failures</h3>\n"
        '<div class="table-scroll">\n<table>\n'
        "  <thead><tr><th>Scenario</th><th>Stage</th><th>Detail</th></tr></thead>\n"
        "  <tbody>\n    " + "\n    ".join(rows) + "\n  </tbody>\n</table>\n</div>"
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
    # Every resource-level check gets a status ROW, passing or failing, ahead of any
    # mismatch detail. A table built only from failing checks' `details` would leave a
    # passing `Resource usage matches config` counted in the banner yet shown in no table.
    # Deliberately NOT reclassifying the check to `level="system"` to make it render there:
    # that would empty `by_level["resource"]` and so drop the per-scenario
    # expected-vs-actual detail table below on failure. The status table keeps both.
    status_rows = "\n    ".join(
        f"<tr><td>{_vocab(c)[0]}</td><td>{_vocab(c)[1]}</td>"
        f'<td class="{_status_of(c)[0]}">{_status_of(c)[1]}</td><td>{c.summary}</td></tr>'
        for c in checks
    )
    status_table = (
        '<div class="table-scroll">\n<table>\n'
        "  <thead><tr><th>Name</th><th>Check</th><th>Status</th><th>Details</th></tr></thead>\n"
        "  <tbody>\n    " + status_rows + "\n  </tbody>\n</table>\n</div>"
    )
    if not all_issues:
        # No "all clear" banner: the status row directly above already reports the check
        # as passed. The SECTION and its status table stay even with nothing mismatched:
        # `Resource usage matches config` is a check-table row, so omitting the section here
        # would omit a check. That is the principled asymmetry with the granular section,
        # which carries no check at all and is omitted when empty.
        return "<h3>Resource-Utilization Mismatches</h3>\n" + status_table
    rows = []
    for d in all_issues:
        scenario = d.get("scenario", d.get("scenario_dir", ""))
        resource = d.get("resource", "")
        expected = d.get("expected", "")
        actual = d.get("actual", "")
        rows.append(f"<tr><td>{scenario}</td><td>{resource}</td><td>{expected}</td><td>{actual}</td></tr>")
    # The FAILING branch needs the status row as much as the passing one -- more so. It is
    # the only branch a run with a real resource mismatch reaches, and no passing report
    # exercises it, so a status row present only in the early return above would leave the
    # check invisible on precisely the runs the section exists to serve, with nothing in a
    # passing report revealing it.
    return (
        "<h3>Resource-Utilization Mismatches</h3>\n" + status_table + '\n<div class="table-scroll">\n<table>\n'
        "  <thead><tr><th>Scenario</th><th>Resource</th><th>Expected</th><th>Actual</th></tr></thead>\n"
        "  <tbody>\n    " + "\n    ".join(rows) + "\n  </tbody>\n</table>\n</div>"
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
        # Read the persisted read-model artifact produced at consolidation, NOT a
        # render-time validate_analysis() whole-tree inspection, so this renderer's only
        # input is a file a bundle carries. Graceful-absent -> empty report. See
        # analysis_validation.
        report = load_validation_report(analysis)
    by_level = report.by_level
    # `Granular Per-Scenario Failures` goes LAST. It is the one section that applies no
    # check -- it re-presents the detail rows of checks already listed above -- so placing
    # it between the aggregate and resource tables would separate two check tables with a
    # non-check table. Check-bearing sections stay contiguous.
    # ONE omission caption for the whole page, computed over the whole check set rather
    # than per table. It sits directly under the banner -- which does not carry the
    # not-applicable count -- so the page states the omission exactly once and in the place
    # a reader looking at the headline figure will see it.
    _n_omitted = sum(1 for c in report.checks if not getattr(c, "applicable", True))
    body_parts = [
        _render_overall_banner(report),
        _omitted_note(_n_omitted).lstrip("\n"),
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
        # The renderer's sole data source is the persisted ValidationReport read-model
        # (validation_report.json), produced at consolidation. It already folds in the
        # eda/*.verdict.json (validate_analysis appends them at persist time), so the
        # renderer declares exactly one file: a provenance audit comparing declared against
        # actual reads has one file to match, and a bundle carrying that file re-renders
        # this page faithfully.
        [analysis_dir / _VALIDATION_REPORT_FILENAME],
        analysis_dir=analysis_dir,
        output_format="html",
        manifest_data={
            "renderer": "errors_and_warnings",
            "section_count": sum(1 for b in body_parts if b),
        },
        provenance=prov,
    )
