"""Disk Utilization sidebar card for the analysis report.

Reads `{analysis_dir}/_status/_du.json` via du_sentinels.read_du_sentinel
and renders a compact summary table for the analysis report HTML sidebar.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from TRITON_SWMM_toolkit.du_sentinels import read_du_sentinel
from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
    emit_plot_with_sources,
)

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


def _fmt_bytes(size_bytes: int) -> str:
    size: float = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PiB"


def _render_table_html(rows: list[tuple[str, int]], total_bytes: int) -> str:
    body_rows = "".join(
        f"<tr><td>{name}</td><td style='text-align:right'>{_fmt_bytes(b)}</td></tr>"
        for name, b in rows
    )
    return (
        "<table class='du-table'>"
        "<thead><tr><th>Scope</th>"
        "<th style='text-align:right'>Bytes</th></tr></thead>"
        f"<tbody>{body_rows}</tbody>"
        "<tfoot><tr><th>Total</th>"
        f"<th style='text-align:right'>{_fmt_bytes(total_bytes)}</th></tr></tfoot>"
        "</table>"
    )


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    sentinel_path = analysis_dir / "_status" / "_du.json"
    analysis_sentinel = read_du_sentinel(sentinel_path)
    if analysis_sentinel is None:
        html = (
            "<p class='du-missing'>Disk utilization sentinel absent — "
            "re-run processing to populate <code>_status/_du.json</code>.</p>"
        )
        source_paths: list[Path] = []
    else:
        total = int(analysis_sentinel.get("disk_utilization_bytes", 0))
        breakdown = analysis_sentinel.get("sub_path_breakdown", {}) or {}
        rows = sorted(
            ((str(name), int(b)) for name, b in breakdown.items()),
            key=lambda r: -r[1],
        )
        html = _render_table_html(rows, total)
        source_paths = [sentinel_path]

    return emit_plot_with_sources(
        html,
        output_path,
        source_paths=source_paths,
        analysis_dir=analysis_dir,
    )
