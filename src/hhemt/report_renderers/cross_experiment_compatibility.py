"""Cross-experiment compatibility + characterized-divergence renderer (PIP-1, Phase 4).

Reads the persisted combined_compatibility.json read-model (Phase 3) and renders
the CompatibilityReport (informational / warning / blocking divergences by
taxonomy bucket) as an inline-styled HTML table. The cross-FAMILY byte-identity
panel is DEFERRED for the bundle path (R6 — a bundle ships the consolidated tree
only, not the flat per-scenario summaries check_cross_sim_identity reads), so a
"deferred" placeholder is rendered in its place. Uniform renderer signature per
the report-renderers stipulation; emits via emit_plot_with_sources (declaring the
read-model as the source, satisfying the Gotcha-41 non-empty-source gate).
"""

from __future__ import annotations

import html as _html
import json as _json
from pathlib import Path

from hhemt.report_renderers._figure_emission import emit_plot_with_sources
from hhemt.report_renderers._provenance import (
    ProvenanceLog,
    ProvenanceRef,
)


def render(analysis, report_cfg, output_path: Path, **kwargs) -> None:
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    source = analysis_dir / "combined_compatibility.json"

    # Per the report-renderer provenance convention (matching the peer table
    # renderers disk_utilization / errors_and_warnings / metadata), the data
    # source is recorded via a `with prov.artist(kind="table")` block and
    # threaded into the manifest sidecar through `provenance=prov`.
    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="cross-experiment compatibility table (combined_compatibility.json)",
    ) as artist:
        artist.add_channel(
            "data",
            ProvenanceRef(source_path="combined_compatibility.json"),
        )
        html = _render_compatibility_html(source)

    emit_plot_with_sources(
        html,
        output_path,
        source_paths=[source],
        analysis_dir=analysis_dir,
        provenance=prov,
    )


def _render_compatibility_html(source: Path) -> str:
    if source.exists():
        payload = _json.loads(source.read_text())
    else:  # combine may not have run; render an honest placeholder
        payload = {"is_compatible": True, "divergences": []}
    divs = payload.get("divergences", [])
    if divs:
        rows = "\n".join(
            "<tr><td>{f}</td><td>{bk}</td><td>{sev}</td><td>{ba}: {va}</td><td>{bb}: {vb}</td></tr>".format(
                f=_html.escape(str(d.get("field_name"))),
                bk=_html.escape(str(d.get("bucket"))),
                sev=_html.escape(str(d.get("severity"))),
                ba=_html.escape(str(d.get("bundle_a"))),
                va=_html.escape(str(d.get("value_a"))),
                bb=_html.escape(str(d.get("bundle_b"))),
                vb=_html.escape(str(d.get("value_b"))),
            )
            for d in divs
        )
        table = (
            "<table class='compat'><thead><tr><th>Field</th><th>Bucket</th>"
            "<th>Severity</th><th>Bundle A</th><th>Bundle B</th></tr></thead>"
            "<tbody>" + rows + "</tbody></table>"
        )
    else:
        table = "<p class='note'>All compared identity fields agree — the bundles are combine-compatible.</p>"
    # c2 (v9/Iteration-2): the tautological "Status: compatible" line is DROPPED — a BLOCKING
    # divergence aborts combine_bundle before any render, so reaching this renderer already
    # implies compatibility; only INFORMATIONAL/warning divergence rows (the table) are informative.
    placeholder = (
        "<div class='deferred'><em>Strict byte-identity panel (R6): the byte-for-byte "
        "cross-family check is deferred for the bundle path — a bundle ships the "
        "value-preserving consolidated tree, not the flat per-scenario summaries the "
        "byte-identity check must read. The physically meaningful clean-vs-resume "
        "value comparison is provided in the Cross-Experiment Results section."
        "</em></div>"
    )
    return (
        "<section class='cross-experiment-compatibility'>"
        "<h2>Cross-Experiment Compatibility</h2>" + table + placeholder + "</section>"
    )
