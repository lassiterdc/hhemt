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
    "td.fail{color:#B11E1E;font-weight:600;text-align:center;}"
    "td.na{color:#8A8A8A;font-weight:500;text-align:center;font-style:italic;}"
    "td.pass-qualified{color:#1F7A1F;font-weight:600;text-align:center;}"
    "td.pass-qualified::after{content:'';}</style>"
)


def _derived_block_html(derived: dict) -> str:
    """Standalone rendering of the derived cross-experiment finding.

    Used on the no-child-reports path, where there is no matrix to carry a row but the
    finding is still true and still worth surfacing.
    """
    return (
        "<p class='note fail'><b>"
        + _html.escape(str(derived.get("name", "")))
        + "</b> (derived across experiments) &mdash; "
        + _html.escape(str(derived.get("summary", "")))
        + "</p>"
    )


def _swmm_invisible_divergence_finding(combined_root: Path) -> dict | None:
    """The TRITON-visible / SWMM-invisible clean-vs-resume signature, as a warning row.

    Measured on the pure-TRITON campaign: max_wlevel_m differs on 14/14 coupled configs
    while max_flow_cms is identical on 14/14. That asymmetry is the FINGERPRINT of the
    SWMM node-depth scatter defect — the interface error collapses from 1.45e+02 cfs to
    2.1e-03 within ~1000 steps, so SWMM-side maxima are unchanged at reported precision
    while TRITON's own depth field carries a permanent perturbation. A reader looking at
    the intercomparison table sees a wall of near-zero max_abs_diff values and reads
    "resume reproduces clean"; the asymmetry is what says otherwise, and nothing surfaced
    it. Returns None when the signature is absent (either variable missing, or both
    variables agree), so this fires on a PATTERN, never on a mere non-identity.
    """
    rm = combined_root / "combined_intercomparison.json"
    if not rm.exists():
        return None
    try:
        pairs = _json.loads(rm.read_text()).get("pairs", [])
    except (OSError, ValueError):
        return None
    coupled = [p for p in pairs if str(p.get("model", "")) == "TRITON-SWMM"]
    depth = [p for p in coupled if p.get("variable") == "max_wlevel_m"]
    flow = [p for p in coupled if p.get("variable") == "max_flow_cms"]
    if not depth or not flow:
        return None
    n_depth_diff = sum(1 for p in depth if not p.get("identical", True))
    n_flow_diff = sum(1 for p in flow if not p.get("identical", True))
    if not (n_depth_diff == len(depth) and n_flow_diff == 0):
        return None
    return {
        "name": "Coupled clean-vs-resume divergence is TRITON-only",
        "level": "aggregate",
        "passed": False,
        "summary": (
            f"Every coupled config's max_wlevel_m differs clean-vs-resume "
            f"({n_depth_diff}/{len(depth)}) while every config's max_flow_cms is identical "
            f"(0/{len(flow)} differ). This asymmetry is the signature of the SWMM node-depth "
            "SCATTER defect on the resume path: the replayed depths never reach the per-rank "
            "new_depth[], the first post-resume step forces the Case-1 exchange branch at every "
            "manhole, and the perturbation lands permanently in TRITON's depth field while the "
            "interface error decays below SWMM's reported precision within ~1000 steps. "
            "SWMM-side agreement is NOT evidence that the resume reproduced the clean run. "
            "See the per-analysis 'Coupled resume validity' check."
        ),
        "details": [],
    }


def _compat_divergence_finding(combined_root: Path) -> dict | None:
    """Non-expected identity-field divergences, as one warning row.

    Carries the class the compatibility renderer's identity-field section used to
    surface before CP-4 removed it. A BLOCKING divergence aborts combine_bundle
    before any render, so everything reaching this read-model is non-blocking; the
    only rows worth a reader's attention are those `_divergence_is_expected` does
    NOT account for -- today that is layout-version (schemaVersion) skew between
    bundles, which changes cross-bundle field semantics while figures still render.

    Returns None when every divergence is expected, so a clean combine renders
    nothing for this class rather than an empty section (P9).
    """
    from hhemt.report_renderers.cross_experiment_compatibility import (
        _divergence_is_expected,
        _divergence_message,
    )

    rm = combined_root / "combined_compatibility.json"
    if not rm.exists():
        return None
    try:
        divs = _json.loads(rm.read_text()).get("divergences", [])
    except (OSError, ValueError):
        return None
    unexpected = [d for d in divs if not _divergence_is_expected(d)]
    if not unexpected:
        return None
    _lines = "; ".join(
        f"{d.get('field_name')} ({d.get('bundle_a')} vs {d.get('bundle_b')}): "
        f"{_divergence_message(d)}"
        for d in unexpected
    )
    return {
        "name": "Unaccounted identity-field divergence across combined bundles",
        "level": "aggregate",
        "passed": False,
        "summary": (
            f"{len(unexpected)} of {len(divs)} identity-field divergence(s) are not "
            f"accounted for by the combine's intended structure: {_lines}. Figures "
            "render, but cross-bundle field semantics may differ."
        ),
        "details": [],
    }


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

    # S4: a cross-arm derived finding, computed from the shipped combine read-model at
    # render time. The read-model MUST be declared as a source: audit_renderer_io
    # (Gotcha 53) raises ProcessingError on any render-time read outside the declared
    # set, so an undeclared read here would be fatal under HHEMT_ENABLE_PROVENANCE_AUDIT.
    # Declared unconditionally (declared-but-absent only warns; undeclared-but-read is
    # fatal), matching the child_crates/*/validation_report.json treatment above.
    _derived_rm = analysis_dir / "combined_intercomparison.json"
    sources.append(_derived_rm)
    # CP-4C: combined_compatibility.json is declared UNCONDITIONALLY for the same
    # Gotcha-53 reason as the line above -- declared-but-absent only warns, while
    # undeclared-but-read raises ProcessingError under HHEMT_ENABLE_PROVENANCE_AUDIT.
    # This is the read the compatibility renderer gave up when CP-4 removed its
    # identity-field section; without it that class is retired silently.
    _compat_rm = analysis_dir / "combined_compatibility.json"
    sources.append(_compat_rm)
    derived_findings = [
        f
        for f in (
            _swmm_invisible_divergence_finding(analysis_dir),
            _compat_divergence_finding(analysis_dir),
        )
        if f is not None
    ] or None

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
        html = _render_rollup_html(child_reports, derived=derived_findings)

    emit_plot_with_sources(
        html,
        output_path,
        source_paths=sources,
        analysis_dir=analysis_dir,
        output_format="html",
        allow_empty_sources=not sources,
        provenance=prov,
    )


def _render_rollup_html(
    child_reports: list[tuple[str, list[dict]]], *, derived: list[dict] | None = None
) -> str:
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
        for _d in derived or ():
            body += _derived_block_html(_d)
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
            # FOUR producer states arrive; branching on `passed` alone surfaced two.
            # CheckResult (analysis_validation.py:83) carries `applicable`, `instrument` and
            # `detection_floor`, and its docstring names THIS module as the reader that
            # receives them across the bundle boundary. Until 2026-08-10 this renderer read
            # none of them, so `applicable=False, passed=True` -- which all three N/A returns
            # set (:686, :1032, :1259) -- rendered as a green PASS, and a verdict computed at
            # the summary tier rendered identically to one computed on raw rasters. Principle
            # P7: a PASS must mean something. getattr-style .get() with defaults throughout,
            # because a validation_report.json written before these fields existed must keep
            # rendering.
            if match is None:
                cells.append("<td>-</td>")
            elif not match.get("applicable", True):
                # Not a pass and not a failure: the check did not apply to this experiment.
                summ = _html.escape(str(match.get("summary", "")))
                cells.append(f"<td class='na' title='{summ}'>N/A</td>")
            elif match.get("passed"):
                _inst = match.get("instrument")
                _floor = match.get("detection_floor")
                if _inst is None or _inst == "raw_rasters":
                    cells.append("<td class='pass'>PASS</td>")
                else:
                    # QUALIFIED pass. The one producer of this state today is the EDA
                    # cross-sim identity verdict (eda/cross_sim_identity.py:458,
                    # instrument="summary_tier"), and the summary tier stores max_wlevel_m --
                    # the variable carrying the coupled-resume perturbation -- as float32.
                    # float32 eps is 1.19e-07, so a true 1.68e-13 difference rounds to
                    # EXACTLY 0.0 and manufactures a false bit-identical control. Rendering
                    # that as a clean PASS is the drop this branch exists to close.
                    _q = (
                        f"verified only at the {_inst} tier"
                        if _floor is None
                        else f"identical only to within {float(_floor):.3g} ({_inst} floor)"
                    )
                    _t = _html.escape(f"{match.get('summary', '')} — {_q}")
                    cells.append(f"<td class='pass-qualified' title='{_t}'>PASS*</td>")
            else:
                summ = _html.escape(str(match.get("summary", "")))
                cells.append(f"<td class='fail' title='{summ}'>FAIL</td>")
        rows.append(f"<tr><td>{_html.escape(name)}</td>{''.join(cells)}</tr>")

    for _d in derived or ():
        # Spans every experiment column: this finding is DERIVED across the combine, so it
        # belongs to no single child. A per-column cell would assert it about one
        # experiment, which is not what was measured.
        _n_cols = len(child_reports) + 1
        _summ = _html.escape(str(_d.get("summary", "")))
        _nm = _html.escape(str(_d.get("name", "")))
        rows.append(
            f"<tr><td colspan='{_n_cols}' class='fail'>"
            f"<b>{_nm}</b> (derived across experiments) &mdash; {_summ}</td></tr>"
        )

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
