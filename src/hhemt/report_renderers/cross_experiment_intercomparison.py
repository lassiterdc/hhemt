"""Cross-experiment clean-vs-resume intercomparison renderer (PIP-1, Phase 5).

Reads the persisted ``combined_intercomparison.json`` read-model (derived CROSS-BUNDLE
by ``bundle/_combine._write_combined_intercomparison``: clean-vs-resume per-compute-config
byte-identity + ``max_abs_diff``, paired via ``compare_variable_exact``) and renders it as
an inline-styled HTML table. The rich visual encoding (the honest-disclosure magnitude
panel) is owned by the ``/eda-spinup`` design step; until then this shell renders a faithful
table of the per-config pairs so the emit/source-declaration path is exercised end-to-end.

Uniform renderer signature per the ``report renderers accept uniform signature`` stipulation;
reads ONLY ``combined_intercomparison.json`` (so ``CombinedBundle.regenerate_report()``
re-renders with no re-merge) and emits via ``emit_plot_with_sources`` (declaring the
read-model as the sole source, satisfying the non-empty-source gate). Same INERT posture as
the compatibility renderer: consumed only by ``_combine.py``'s emit-time direct-render
dispatch, so no Snakefile rule / caption-RST resolution is involved.
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
    source = analysis_dir / "combined_intercomparison.json"

    # Per the report-renderer provenance convention (matching the peer table
    # renderers), the data source is recorded via a `with prov.artist(kind="table")`
    # block and threaded into the manifest sidecar through `provenance=prov`.
    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="cross-experiment clean-vs-resume intercomparison table (combined_intercomparison.json)",
    ) as artist:
        artist.add_channel(
            "data",
            ProvenanceRef(source_path="combined_intercomparison.json"),
        )
        html = _render_intercomparison_html(source)

    emit_plot_with_sources(
        html,
        output_path,
        source_paths=[source],
        analysis_dir=analysis_dir,
        provenance=prov,
    )


def _render_intercomparison_html(source: Path) -> str:
    if source.exists():
        payload = _json.loads(source.read_text())
    else:  # combine may not have run; render an honest placeholder
        payload = {"experiments": [], "pairs": []}
    experiments = payload.get("experiments", [])
    pairs = payload.get("pairs", [])

    exp_line = (
        ", ".join(f"{_html.escape(str(e.get('experiment')))} ({_html.escape(str(e.get('role')))})" for e in experiments)
        or "(no experiments recorded)"
    )

    if pairs:
        rows = "\n".join(
            "<tr><td>{cfg}</td><td>{var}</td><td>{ev}</td><td>{ident}</td><td>{mad}</td></tr>".format(
                cfg=_html.escape(str(p.get("config"))),
                var=_html.escape(str(p.get("variable"))),
                ev=_html.escape(str(p.get("event_iloc"))),
                ident=("identical" if p.get("identical") else "differs"),
                mad=_html.escape(str(p.get("max_abs_diff"))),
            )
            for p in pairs
        )
        table = (
            "<table class='intercomparison'><thead><tr><th>Compute config</th>"
            "<th>Variable</th><th>Event</th><th>Clean vs resume</th><th>max_abs_diff</th>"
            "</tr></thead><tbody>" + rows + "</tbody></table>"
        )
    else:
        table = (
            "<p class='note'>No paired compute-configs found across the two bundles — "
            "the combined report renders the compatibility half only.</p>"
        )
    placeholder = (
        "<div class='deferred'><em>The honest-disclosure magnitude panel (clean-vs-resume "
        "peak-field perturbation) is owned by the /eda-spinup design step; this table is the "
        "interim faithful projection of combined_intercomparison.json.</em></div>"
    )
    return (
        "<section class='cross-experiment-intercomparison'>"
        "<h2>Cross-Experiment Results: clean vs resume</h2>"
        "<p>Experiments: " + exp_line + "</p>" + table + placeholder + "</section>"
    )
