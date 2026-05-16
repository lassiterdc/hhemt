"""Appendix renderer: emit scenario_status.csv as an inline-styled HTML table.

Per Iter 8 agenda item 3 + snakemake-specialist consult 18:09: rule output is
a `.html` file that the Snakemake report engine renders via `<iframe>` (the
JS bundle dispatches `case "html":` to an iframe). The iframe inherits no
parent CSS, so the rendered HTML must carry inline `<style>` to be readable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


_INLINE_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       padding: 12px; color: #333; margin: 0; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 6px 10px; border: 1px solid #DADADA; text-align: left;
         vertical-align: top; }
th { background-color: #232D4B; color: white; font-weight: 600; }
tr:nth-child(even) td { background-color: #F1F1EF; }
tr:hover td { background-color: #FFE4C4; }
"""


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    """Render scenario_status.csv to an HTML table at output_path.

    Sources the CSV from ``analysis.analysis_paths.analysis_dir / scenario_status.csv``
    (written by ``export_scenario_status.py`` as a Snakemake onsuccess/onerror
    hook). When the CSV is missing, emits a placeholder HTML noting the absence
    so the appendix entry is never blank.
    """
    static_backend = getattr(
        getattr(report_cfg, "interactive", None),
        "static_backend",
        "plotly",
    )
    if static_backend == "plotly":
        from TRITON_SWMM_toolkit.report_renderers._static_backend_warning import (
            warn_no_plotly_branch,
        )
        warn_no_plotly_branch("scenario_status_appendix")

    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        _validate_source_path,
        emit_plot_with_sources,
    )
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceLog, ProvenanceRef

    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    csv_path = analysis_dir / "scenario_status.csv"
    prov = ProvenanceLog()
    with prov.artist(
        axes_id="html_section",
        kind="table",
        note="scenario_status table (HTML-rendered, no matplotlib artist)",
    ) as a:
        a.add_channel(
            "data",
            ProvenanceRef(source_path="scenario_status.csv"),
        )
        if csv_path.exists():
            _validate_source_path(csv_path)
            df = pd.read_csv(csv_path)
            table_html = df.to_html(index=False, escape=True, na_rep="—", border=0)
            body = f"<h2>Scenario Status</h2>\n{table_html}"
            row_count = int(len(df))
        else:
            body = (
                "<h2>Scenario Status</h2>\n"
                "<p><em>scenario_status.csv not yet written — workflow may have "
                "been killed before the onsuccess/onerror Snakemake hook ran.</em></p>"
            )
            row_count = 0

    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<style>{report_cfg.scenario_status_appendix.render_inline_css()}</style></head><body>"
        f"{body}"
        "</body></html>"
    )
    return emit_plot_with_sources(
        html,
        output_path,
        [csv_path],
        analysis_dir=analysis_dir,
        output_format="html",
        manifest_data={
            "renderer": "scenario_status_appendix",
            "table_format": "inline-css",
            "row_count": row_count,
            "csv_present": csv_path.exists(),
        },
        provenance=prov,
    )
