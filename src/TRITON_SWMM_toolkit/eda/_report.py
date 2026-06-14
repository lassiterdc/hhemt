"""EDA report assembler (ADR-10): a GENERIC, reusable scrollable-HTML composer
plus the EDA-specific wrapper.

`render_scrollable_report` is EDA-agnostic — a figure-collection + table-collection
assembler over a Jinja2 template — so the parked scrollable-singlepage-report idea
can reuse it. `assemble_eda_report` is the EDA wrapper: it harvests the
plots/eda/*.manifest.json (the figures + their declared datasets) and emits
{root}/eda_report/eda_report.html. The Plotly bundle is inlined EXACTLY ONCE via
`_figure_divs` (first-figure-inline + rest include_plotlyjs=False; full_html=False).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jinja2
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr

from TRITON_SWMM_toolkit.eda._plotting import _cross_sim_identity_figure
from TRITON_SWMM_toolkit.report_plot_ids import canonical_plot_id
from TRITON_SWMM_toolkit.report_renderers._figure_emission import harvest_source_paths

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.config.analysis import analysis_config
    from TRITON_SWMM_toolkit.config.eda import eda_config


# FigureSpec + _figure_divs: the data-visualization-specialist FQ1 VMS (the
# first-figure-inline single-bundle mechanism). Applied verbatim from the master
# plan's Implementation Research; this is the single place the bundle-once
# invariant lives.
@dataclass
class FigureSpec:
    """One Plotly figure destined for the scrollable report."""

    figure: go.Figure
    title: str = ""
    config: dict[str, Any] = field(default_factory=dict)


def _figure_divs(figures: list[FigureSpec], *, plotly_js_mode: str) -> tuple[list[str], str]:
    """Render each FigureSpec to a self-contained <div> snippet, inlining the
    Plotly bundle EXACTLY ONCE.

    The first figure carries the bundle (`include_plotlyjs` resolved from
    `plotly_js_mode`: 'inline' -> True, 'cdn' -> 'cdn'); every subsequent
    figure is emitted with `include_plotlyjs=False`. `full_html=False` is
    MANDATORY on every figure so the snippets concatenate into one document.
    Plotly mints a fresh UUIDv4 div id per call, so ids never collide.

    Returns (list_of_div_html, js_mode_used). Caller drops the divs into the
    page template in order; the first div carries the <script> bundle.
    """
    first_js: bool | str = True if plotly_js_mode == "inline" else "cdn"
    divs: list[str] = []
    for idx, spec in enumerate(figures):
        # Toggle on INDEX, never on figure identity — a reordered/conditional
        # first figure must never silently drop the bundle.
        include_js: bool | str = first_js if idx == 0 else False
        # Ensure an explicit px height so the scrollable stack does not collapse
        # to zero on height:100%-of-auto-height parents.
        if spec.figure.layout.height is None:
            spec.figure.update_layout(height=500)
        divs.append(
            pio.to_html(
                spec.figure,
                include_plotlyjs=include_js,
                full_html=False,
                config={"displaylogo": False, **spec.config},
            )
        )
    return divs, plotly_js_mode


@dataclass
class TableSpec:
    """One Tabulator data table (e.g. the EDA datasets reference)."""

    title: str
    columns: list[str]
    rows: list[list[str]]


def render_scrollable_report(
    figures: list[FigureSpec],
    tables: list[TableSpec],
    *,
    title: str,
    brand: dict[str, Any] | None = None,
    plotly_js_mode: str = "inline",
    tabulator_js_mode: str = "inline",
) -> str:
    """Compose figures + tables into ONE self-contained scrollable HTML string.

    GENERIC (no EDA coupling). The Plotly bundle is inlined once across all
    figures (via _figure_divs); Tabulator loads as an independent global
    (inline or cdn). Returns the full HTML document text.
    """
    # _figure_divs returns (divs, js_mode_used). The Plotly bundle is embedded
    # INSIDE divs[0] (first figure carries include_plotlyjs=True), NOT returned
    # separately — so figure_divs[0] (emitted in <body>) carries the bundle.
    # Do NOT inject the second return value into <head>: it is the mode string,
    # not bundle HTML (master FQ1 VMS: _figure_divs -> tuple[list[str], str]).
    fig_divs, _js_mode_used = _figure_divs(figures, plotly_js_mode=plotly_js_mode)
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("TRITON_SWMM_toolkit", "report_templates"),
        autoescape=jinja2.select_autoescape(["html"]),
    )
    template = env.get_template("eda_report.html.j2")
    return template.render(
        title=title,
        brand=brand or {},
        figure_divs=fig_divs,
        tables=tables,
        tabulator_js_mode=tabulator_js_mode,
    )


def _cross_sim_identity_figure_from_root(root: Path) -> go.Figure:
    """Re-build the cross-sim-identity figure from the carried eda/<plot_id>.zarr.

    The scrollable doc is RE-RENDERED from the data-prep dataset (NOT assembled
    from the saved standalone plots/eda/*.html fragments, which are full_html=True
    documents that cannot be concatenated — master open-exploration (d)). This
    reuses the exact Phase-1 builder so the doc figure matches the per-figure
    artifact.
    """
    plot_id = canonical_plot_id("eda_cross_sim_identity")
    ds = xr.open_zarr(root / "eda" / f"{plot_id}.zarr", consolidated=False)
    return _cross_sim_identity_figure(ds)


#: EDA plot kind -> figure-from-root builder. Parallels eda/_plotting.py's
#: _EDA_RENDERERS (which couples open+emit); this rebuilds the go.Figure alone
#: for the multi-figure bundle-once doc. Extend in lockstep with _EDA_RENDERERS.
_EDA_FIGURE_BUILDERS = {
    "eda_cross_sim_identity": _cross_sim_identity_figure_from_root,
}


def _figures_from_plots_eda(root: Path, *, eda_cfg: eda_config) -> list[FigureSpec]:
    """Rebuild a FigureSpec per enabled EDA plot from its carried zarr.

    Iterates ``eda_cfg.enabled_plots`` (config order = doc order = the order
    render_eda_plots emitted), re-rendering each figure from ``eda/<plot_id>.zarr``
    via the Phase-1 builder. Unknown kinds raise ValueError (fail-fast, mirroring
    render_eda_plots).
    """
    figures: list[FigureSpec] = []
    for kind in eda_cfg.enabled_plots:
        builder = _EDA_FIGURE_BUILDERS.get(kind)
        if builder is None:
            raise ValueError(f"no EDA figure builder for kind {kind!r}; known: {sorted(_EDA_FIGURE_BUILDERS)}")
        figures.append(FigureSpec(figure=builder(root), title=kind))
    return figures


def assemble_eda_report(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
) -> Path:
    """Harvest plots/eda/ and emit {root}/eda_report/eda_report.html.

    Reads the rendered EDA-plot HTML fragments + their declared source datasets
    (via harvest_source_paths over {root}/plots/eda), builds the FigureSpec +
    TableSpec lists, calls render_scrollable_report, and writes the doc.
    """
    plots_eda = root / "plots" / "eda"
    sources_by_stem = harvest_source_paths(plots_eda, root)  # {stem: [Path, ...]}
    figures = _figures_from_plots_eda(root, eda_cfg=eda_cfg)  # rebuilds from eda/<plot_id>.zarr
    datasets_table = TableSpec(
        title="EDA datasets",
        columns=["Plot", "Dataset"],
        rows=[
            [stem, str(p.relative_to(root)) if p.is_relative_to(root) else str(p)]
            for stem, paths in sorted(sources_by_stem.items())
            for p in paths
        ],
    )
    html = render_scrollable_report(
        figures,
        [datasets_table],
        title=f"EDA report — {cfg_analysis.analysis_id}",
        plotly_js_mode=eda_cfg.plotly_js_mode,
        # cdn interim per DECISION-1 Option A (SPAWN); the spawned inline plan
        # flips the eda_config default to "inline" with zero EDA-side rework.
        tabulator_js_mode=eda_cfg.tabulator_js_mode,
    )
    out_dir = root / "eda_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "eda_report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
