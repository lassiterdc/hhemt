"""Disk Utilization sidebar card for the analysis report.

Reads `{analysis_dir}/_status/_du.json` via du_sentinels.read_du_sentinel
and renders a compact summary table for the analysis report HTML sidebar.

The sentinel is written when processing finishes, but the report plots are
written afterwards, so the sentinel's own-files breakdown has no row for them.
At render time the analysis's own files are therefore measured again; only the
`members/` and `sims/` directories keep the totals their child sentinels
recorded, because walking them is the expensive scan the sentinel rollup exists
to avoid. The sentinel's `computed_at` is printed above the table.

A page regenerated from a portable bundle is the exception: the bundle root holds
harvested files only (a consolidated zarr is reduced to its metadata documents), so
walking it would measure the bundle, not the analysis. When the root carries
`BUNDLE_MANIFEST_FILENAME`, every row is the sentinel's recorded total and nothing
is walked; the page says so.

Per the report-renderer provenance convention (matching the peer table
renderers scenario_status_appendix / errors_and_warnings / per_analysis_summary),
the data source is recorded via a `with prov.artist(kind="table")` block and
threaded into the manifest sidecar through `provenance=prov`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hhemt.du_sentinels import (
    _EPHEMERAL_TOP_LEVEL_DIRS,
    _walk_root_and_breakdown,
    read_du_sentinel,
)
from hhemt.report_renderers._figure_emission import (
    emit_plot_with_sources,
)
from hhemt.report_renderers._provenance import (
    ProvenanceLog,
    ProvenanceRef,
)
from hhemt.version_migration.constants import BUNDLE_MANIFEST_FILENAME

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.report import report_config


# Directories whose size is the sum of their child sentinels; never walked at render.
_CHILD_SCOPE_DIRS = ("members", "sims")


def _own_files_breakdown(analysis_dir: Path) -> dict[str, int]:
    """Bytes under each top-level entry of `analysis_dir` that is not a child-scope dir.

    Skips `members/` and `sims/`, every entry whose name starts with `_status`
    and the ephemeral top-level directories (`_test/`), and counts top-level
    files: the same entries `du_sentinels.sum_child_sentinels` skips and counts in
    its own-files walk. Entries with zero bytes are omitted.
    """
    skip = set(_CHILD_SCOPE_DIRS) | _EPHEMERAL_TOP_LEVEL_DIRS
    out: dict[str, int] = {}
    try:
        entries = sorted(analysis_dir.iterdir())
    except OSError:
        return out
    for entry in entries:
        if entry.name in skip or entry.name.startswith("_status"):
            continue
        if entry.is_dir():
            size, _breakdown, _errors = _walk_root_and_breakdown(entry)
        elif entry.is_file():
            try:
                size = entry.stat().st_size
            except OSError:
                continue
        else:
            continue
        if size:
            out[entry.name] = size
    return out


def _du_table_html(rows: list[tuple[str, int]], computed_at: str | None = None, *, from_bundle: bool = False) -> str:
    """The scope/bytes breakdown as a sortable Tabulator document.

    Values are carried as MiB FLOATS, never as formatted byte strings. A column
    of "1.1 GiB" strings sorts lexically (9.9 MiB above 1.1 GiB) and cannot be
    summed; a single-unit float sorts numerically, sums correctly, and formats
    through Tabulator's built-in "money" formatter, which needs no JS callable
    and so survives the JSON options round-trip.

    The Total row is Tabulator's own column calculation. The shared surface
    ships no totals feature, but the columns list build_columns_spec returns is
    the caller's to extend, so this needs no change to _tabulator_defaults.
    The Total is the sum of the rows shown, each rounded to 0.1 MiB. It equals the
    sentinel total only while nothing outside `members/` and `sims/` has changed
    since the sentinel was computed, and then only to within that rounding.

    `initialSort` preserves the descending-by-size order the caller sorted into
    `rows`; without it Tabulator would render in insertion order and the card's
    "biggest consumers first" reading would be lost on the default view.

    `computed_at` is the sentinel's timestamp; when given it is printed under the
    heading so a reader can see how old the recorded totals are. With
    `from_bundle` the page also says the rows are the sentinel's record and were
    not re-measured.
    """
    import html as _html

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
        df_du,
        visible_columns_default=None,
        header_filter=True,
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
    heading_html = "<h2>Disk Utilization</h2>"
    if computed_at:
        heading_html += f"<p class='du-computed-at'>Totals recorded at {_html.escape(str(computed_at))}</p>"
    if from_bundle:
        heading_html += (
            "<p class='du-not-remeasured'>Every row is the total recorded at the time above and was not "
            "re-measured, because this page was regenerated from a bundle.</p>"
        )
    return build_html_document(
        title="Disk Utilization",
        container_id="disk-utilization",
        body_heading_html=heading_html,
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
        note="disk utilization summary table (_status/_du.json; own files re-measured unless from a bundle)",
    ) as a:
        analysis_sentinel = read_du_sentinel(sentinel_path)
        if analysis_sentinel is None:
            html = (
                "<p class='du-missing'>The disk use table is not shown: <code>_status/_du.json</code> "
                "is missing, so re-run processing to write it.</p>"
            )
            # Declare the expected source unconditionally (ADR-6 D3): _status/_du.json
            # is the named source even when the sentinel is absent.
            source_paths: list[Path] = [sentinel_path]
        else:
            recorded = analysis_sentinel.get("sub_path_breakdown", {}) or {}
            from_bundle = (analysis_dir / BUNDLE_MANIFEST_FILENAME).exists()
            if from_bundle:
                breakdown = dict(recorded)
            else:
                breakdown = {name: b for name, b in recorded.items() if name in _CHILD_SCOPE_DIRS}
                breakdown.update(_own_files_breakdown(analysis_dir))
            rows = sorted(
                ((str(name), int(b)) for name, b in breakdown.items()),
                key=lambda r: -r[1],
            )
            html = _du_table_html(rows, analysis_sentinel.get("computed_at"), from_bundle=from_bundle)
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
