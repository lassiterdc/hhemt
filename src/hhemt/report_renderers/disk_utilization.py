"""Disk Utilization sidebar card for the analysis report.

Reads `{analysis_dir}/_status/_du.json` via du_sentinels.read_du_sentinel
and renders a compact summary table for the analysis report HTML sidebar.

Per the report-renderer provenance convention (matching the peer table
renderers scenario_status_appendix / errors_and_warnings / per_analysis_summary),
the data source is recorded via a `with prov.artist(kind="table")` block and
threaded into the manifest sidecar through `provenance=prov`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hhemt.du_sentinels import read_du_sentinel
from hhemt.report_renderers._figure_emission import (
    emit_plot_with_sources,
)
from hhemt.report_renderers._provenance import (
    ProvenanceLog,
    ProvenanceRef,
)

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.report import report_config


def _du_table_html(rows: list[tuple[str, int]]) -> str:
    """The scope/bytes breakdown as a sortable Tabulator document.

    Values are carried as MiB FLOATS, never as formatted byte strings. A column
    of "1.1 GiB" strings sorts lexically (9.9 MiB above 1.1 GiB) and cannot be
    summed; a single-unit float sorts numerically, sums correctly, and formats
    through Tabulator's built-in "money" formatter, which needs no JS callable
    and so survives the JSON options round-trip.

    The Total row is Tabulator's own column calculation. The shared surface
    ships no totals feature, but the columns list build_columns_spec returns is
    the caller's to extend, so this needs no change to _tabulator_defaults.
    Measured on all four bundles of the reviewed generation:
    sum(sub_path_breakdown) == disk_utilization_bytes exactly, so the summed
    Total equals the sentinel total this card displayed before.

    `initialSort` preserves the descending-by-size order the caller sorted into
    `rows`; without it Tabulator would render in insertion order and the card's
    "biggest consumers first" reading would be lost on the default view.
    """
    import pandas as pd

    from hhemt.report_renderers._tabulator_defaults import (
        build_columns_spec,
        build_html_document,
        build_options_dict,
    )

    _MIB = 1024.0 * 1024.0
    _SIZE_COL = "Size (MiB)"
    df_du = pd.DataFrame(
        [{"Scope": name, _SIZE_COL: round(b / _MIB, 1)} for name, b in rows],
        columns=["Scope", _SIZE_COL],
    )
    columns_spec = build_columns_spec(
        df_du, visible_columns_default=None, header_filter=True,
    )
    for _spec in columns_spec:
        if _spec.get("field") == _SIZE_COL:
            _spec["bottomCalc"] = "sum"
            _spec["formatter"] = "money"
            _spec["formatterParams"] = {"precision": 1, "symbol": ""}
            _spec["bottomCalcFormatter"] = "money"
            _spec["bottomCalcFormatterParams"] = {"precision": 1, "symbol": ""}
    options = build_options_dict(
        df_du,
        columns_spec=columns_spec,
        table_height="320px",
        pagination_size=0,
        persistence_id="disk_utilization",
        extra_options={"initialSort": [{"column": _SIZE_COL, "dir": "desc"}]},
    )
    return build_html_document(
        title="Disk Utilization",
        container_id="disk-utilization",
        body_heading_html="<h2>Disk Utilization</h2>",
        options=options,
        js_mode="cdn",
        renderer_name="disk_utilization",
    )


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    sentinel_path = analysis_dir / "_status" / "_du.json"

    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="disk utilization summary table (_status/_du.json)",
    ) as a:
        analysis_sentinel = read_du_sentinel(sentinel_path)
        if analysis_sentinel is None:
            html = (
                "<p class='du-missing'>Disk utilization sentinel absent — "
                "re-run processing to populate <code>_status/_du.json</code>.</p>"
            )
            # Declare the expected source unconditionally (ADR-6 D3): _status/_du.json
            # is the named source even when the sentinel is absent.
            source_paths: list[Path] = [sentinel_path]
        else:
            breakdown = analysis_sentinel.get("sub_path_breakdown", {}) or {}
            rows = sorted(
                ((str(name), int(b)) for name, b in breakdown.items()),
                key=lambda r: -r[1],
            )
            html = _du_table_html(rows)
            source_paths = [sentinel_path]
            a.add_channel(
                "data",
                ProvenanceRef(source_path="_status/_du.json"),
            )

    return emit_plot_with_sources(
        html,
        output_path,
        source_paths=source_paths,
        analysis_dir=analysis_dir,
        provenance=prov,
    )
