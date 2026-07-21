"""Cross-experiment errors-and-warnings roll-up renderer (Phase 5, F2).

Restores a DISCOVERABLE top-level cross-experiment health surface. v8/a2 retired the
aggregate combined_errors_and_warnings renderer, leaving each child's E&W reachable only
as a buried {eid}/"Errors and Warnings" subcategory (2-level Snakemake category ceiling).
This renderer reads each child_crates/{eid}/validation_report.json ({"checks":[...]})
DIRECTLY at render time -- the same render-time child-read pattern as
cross_experiment_intercomparison_maps (Option R: no emit-time artifact, CR4-safe) -- and
emits a per-experiment x per-check pass/fail matrix. Honest: it reconstructs each child's
ValidationReport from its .checks (the retired shim wrongly read overall_passed/by_level/
granular_failures off the raw dict, which carries only "checks"). Uniform renderer
signature; emits via emit_plot_with_sources declaring each child JSON as a source.
"""

from __future__ import annotations

import html as _html
import json as _json
from pathlib import Path

from hhemt.report_renderers._figure_emission import emit_plot_with_sources
from hhemt.report_renderers._provenance import ProvenanceLog, ProvenanceRef

_INLINE_STYLE = (
    "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
    "padding:12px;color:#333;margin:0;}h2{color:#232D4B;border-bottom:2px solid #232D4B;"
    "padding-bottom:4px;margin-top:0;}table{border-collapse:collapse;width:100%;font-size:13px;}"
    "th,td{padding:6px 10px;border:1px solid #DADADA;text-align:left;}th{background:#232D4B;"
    "color:#fff;}td.pass{color:#1F7A1F;font-weight:600;text-align:center;}"
    "td.fail{color:#B11E1E;font-weight:600;text-align:center;}</style>"
)


def render(analysis, report_cfg, output_path: Path, **kwargs) -> None:
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    crates = analysis_dir / "child_crates"
    child_reports: list[tuple[str, list[dict]]] = []
    sources: list[Path] = []
    if crates.exists():
        for child in sorted(p for p in crates.iterdir() if p.is_dir()):
            vr = child / "validation_report.json"
            sources.append(vr)  # declared even when absent (info-icon names the expected file)
            checks: list[dict] = []
            if vr.exists():
                try:
                    checks = _json.loads(vr.read_text()).get("checks", [])
                except (OSError, ValueError):
                    checks = []
            child_reports.append((child.name, checks))

    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="cross-experiment errors-and-warnings roll-up (child_crates/*/validation_report.json)",
    ) as artist:
        for src in sources:
            artist.add_channel(
                "data",
                ProvenanceRef(source_path=src.relative_to(analysis_dir).as_posix()),
            )
        html = _render_rollup_html(child_reports)

    emit_plot_with_sources(
        html,
        output_path,
        source_paths=sources,
        analysis_dir=analysis_dir,
        output_format="html",
        allow_empty_sources=not sources,
        provenance=prov,
    )


def _render_rollup_html(child_reports: list[tuple[str, list[dict]]]) -> str:
    # Union of check names across experiments, preserving first-seen order.
    check_names: list[str] = []
    seen: set[str] = set()
    for _eid, checks in child_reports:
        for c in checks:
            name = str(c.get("name", ""))
            if name and name not in seen:
                seen.add(name)
                check_names.append(name)

    if not child_reports or not check_names:
        body = "<p class='note'>No per-experiment validation reports were found in child_crates/.</p>"
        return (
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            + _INLINE_STYLE
            + "</head><body><h2>Cross-Experiment Errors and Warnings</h2>"
            + body
            + "</body></html>"
        )

    header = "".join(f"<th>{_html.escape(eid)}</th>" for eid, _ in child_reports)
    rows: list[str] = []
    for name in check_names:
        cells: list[str] = []
        for _eid, checks in child_reports:
            match = next((c for c in checks if str(c.get("name")) == name), None)
            if match is None:
                cells.append("<td>-</td>")
            elif match.get("passed"):
                cells.append("<td class='pass'>PASS</td>")
            else:
                summ = _html.escape(str(match.get("summary", "")))
                cells.append(f"<td class='fail' title='{summ}'>FAIL</td>")
        rows.append(f"<tr><td>{_html.escape(name)}</td>{''.join(cells)}</tr>")

    table = "<table><thead><tr><th>Check</th>" + header + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        + _INLINE_STYLE
        + "</head><body><section class='cross-experiment-errors-and-warnings'>"
        + "<h2>Cross-Experiment Errors and Warnings</h2>"
        + "<p>Per-experiment validation-check outcomes. Expand an experiment's own section "
        + "for its full Errors and Warnings detail.</p>"
        + table
        + "</section></body></html>"
    )
