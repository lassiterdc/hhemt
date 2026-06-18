"""EDA plotting family (ADR-10 / ADR-2 / ADR-6).

Free functions shared by `analysis.eda()` and `Bundle.eda()` (they take a `root`
Path + the configs, NOT a TRITONSWMM_analysis, so the Bundle non-subclass boundary
is honored). Each EDA plot emits via `emit_plot_with_sources` (HTML branch) under
MASTER-ROOTED `{root}/plots/eda/<plot_id>.html` and declares its
`{root}/eda/<plot_id>.zarr` data-prep artifact as a source - so the existing
harvest chain carries the dataset into a render bundle (D1 Option A). EDA plots
MUST NOT emit under plots/sensitivity/per_sim/sa-{N}/ (harvest re-roots that subtree
against subanalyses/sa_{N}/, which has no eda/ dir; see the master-rooted-emission
stipulation).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go
import xarray as xr

from hhemt.config.viz_vocabulary import validate_plotly_colorscale
from hhemt.report_plot_ids import canonical_plot_id
from hhemt.report_renderers._figure_emission import emit_plot_with_sources

if TYPE_CHECKING:
    from hhemt.config.analysis import analysis_config
    from hhemt.config.eda import eda_config

#: Sequential single-hue colorscale for the cross-sim-identity heatmap. ADR-1:
#: routed through config/viz_vocabulary so the name is validated against plotly's
#: registry. "Reds" is one-directional (0 = pale floor, any nonzero cell pops),
#: matching the from-zero max-abs-diff data; NEVER a rainbow/jet or diverging
#: scale. The /design-figure pass may refine this; the choice is committed here.
_CROSS_SIM_COLORSCALE = "Reds"

#: Per-variable subplot height (px). An explicit height keeps the doc-assembler
#: scrollable stack from collapsing (FQ1 finding).
_EDA_SUBPLOT_HEIGHT_PX = 360


def render_eda_plots(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
) -> list[Path]:
    """Render every plot in ``eda_cfg.enabled_plots`` to ``{root}/plots/eda/``.

    Returns the list of emitted HTML paths. Unknown renderer-kind keys raise
    ValueError (fail-fast at the facade boundary). ``root`` is the analysis_dir
    on the Analysis side and bundle.root on the Bundle side.
    """
    out: list[Path] = []
    for kind in eda_cfg.enabled_plots:
        renderer = _EDA_RENDERERS.get(kind)
        if renderer is None:
            raise ValueError(f"unknown EDA plot kind {kind!r}; known: {sorted(_EDA_RENDERERS)}")
        out.append(renderer(root, cfg_analysis=cfg_analysis, eda_cfg=eda_cfg))
    return out


def _render_cross_sim_identity(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
) -> Path:
    """First EDA plot: a per-(sa_id, event_iloc) pass/fail + max-abs-diff view of
    the completed cross-sim byte-identity artifact (eda/<plot_id>.zarr).

    Reads `{root}/eda/<plot_id>.zarr` (max_abs_diff__{var} + identical__{var} vars
    keyed by (sa_id, event_iloc)) and renders a Plotly heatmap/table. Emits under
    MASTER-ROOTED plots/eda/ declaring the zarr as the source (R3/D1).
    """
    plot_id = canonical_plot_id("eda_cross_sim_identity")
    zarr_path = root / "eda" / f"{plot_id}.zarr"
    ds = xr.open_zarr(zarr_path, consolidated=False)

    fig = _cross_sim_identity_figure(ds)

    output_path = root / "plots" / "eda" / f"{plot_id}.html"
    html_text = _fig_to_html(fig, plotly_js_mode=eda_cfg.plotly_js_mode)
    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths=[zarr_path],  # R3: declare the eda-data-prep artifact
        analysis_dir=root,  # so source_paths_relative = eda/<plot_id>.zarr
        output_format="html",
    )


def _cross_sim_identity_figure(ds: xr.Dataset) -> go.Figure:
    """Build the cross-sim-identity Plotly figure from the diff/identical maps.

    Chart design (master open-exploration (a); empirically grounded - Wilke
    Directory of Visualizations for a 2D-grid amounts message; Moreland 2016 for
    single-hue sequential safety): one faceted go.Heatmap per `max_abs_diff__{var}`
    data_var over the (sa_id, event_iloc) grid, with:
      * zmin=0 anchored at the expected-good value (0.0 everywhere when identical);
      * a SEQUENTIAL single-hue colorscale ('Reds') routed through
        config/viz_vocabulary (ADR-1) - NEVER a rainbow/jet/hsv or a diverging
        scale (the data is one-directional from 0);
      * redundant coding (color + glyph): annotate each cell with the
        max_abs_diff value when NOT identical (blank when identical) so the figure
        is legible without color;
      * axis titles 'event_iloc' / 'sa_id', a colorbar title naming the variable,
        and an explicit layout.height so the scrollable doc stack does not collapse.
    A passing analysis reads as a uniform pale grid (instant 'all identical'); a
    regression surfaces as a single locatable red cell. The /design-figure loop is
    an OPTIONAL aesthetic-refinement pass over this committed design, not the source
    of the chart-type/color decision.
    """
    from plotly.subplots import make_subplots

    colorscale = validate_plotly_colorscale(_CROSS_SIM_COLORSCALE)
    diff_vars = sorted(v for v in ds.data_vars if str(v).startswith("max_abs_diff__"))

    if not diff_vars:
        # Degenerate-but-valid: no tracked variables. Emit an empty figure with an
        # explicit height so the scrollable doc stack still reserves space.
        fig = go.Figure()
        fig.update_layout(
            height=_EDA_SUBPLOT_HEIGHT_PX,
            title="cross-sim byte-identity (no tracked variables)",
        )
        return fig

    n = len(diff_vars)
    vspacing = 0.0 if n == 1 else min(0.08, 0.8 / n)
    plot_h = (1.0 - vspacing * (n - 1)) / n
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=[v.removeprefix("max_abs_diff__") for v in diff_vars],
        vertical_spacing=vspacing,
    )

    for idx, diff_var in enumerate(diff_vars):
        var = diff_var.removeprefix("max_abs_diff__")
        row = idx + 1
        diff_da = ds[diff_var].transpose("sa_id", "event_iloc")
        sa_ids = [str(s) for s in diff_da["sa_id"].values]
        event_ilocs = [str(int(e)) for e in diff_da["event_iloc"].values]
        z = diff_da.values

        identical_name = f"identical__{var}"
        text: list[list[str]] | None = None
        if identical_name in ds.data_vars:
            ident = ds[identical_name].transpose("sa_id", "event_iloc").values
            text = [
                ["" if bool(ident[i][j]) else f"{z[i][j]:.3g}" for j in range(z.shape[1])] for i in range(z.shape[0])
            ]

        # Position this trace's colorbar over its own subplot row so the stacked
        # colorbars do not overlap.
        top = 1.0 - idx * (plot_h + vspacing)
        cbar_y = top - plot_h / 2.0
        heatmap = go.Heatmap(
            x=event_ilocs,
            y=sa_ids,
            z=z,
            zmin=0,
            colorscale=colorscale,
            colorbar=dict(title=f"{var}<br>max abs diff", len=plot_h, y=cbar_y, yanchor="middle"),
        )
        if text is not None:
            heatmap.update(text=text, texttemplate="%{text}")
        fig.add_trace(heatmap, row=row, col=1)
        fig.update_xaxes(title_text="event_iloc", type="category", row=row, col=1)
        fig.update_yaxes(title_text="sa_id", type="category", row=row, col=1)

    fig.update_layout(
        height=_EDA_SUBPLOT_HEIGHT_PX * n,
        title="cross-sim byte-identity: max abs diff per (sa_id, event_iloc)",
    )
    return fig


def _fig_to_html(fig: go.Figure, *, plotly_js_mode: str) -> str:
    """Serialize one figure to an HTML fragment via the FQ1 single-bundle path.

    For a SINGLE figure the simple form is `pio.to_html(fig,
    include_plotlyjs=<True|cdn>, full_html=True)`. The MULTI-figure bundle-once
    composition lives in eda/_report.py::_figure_divs (Phase 2); this single-figure
    emit is what emit_plot_with_sources' HTML branch stores per plot.

    ``plotly_js_mode`` is the eda_config field value ('inline' | 'cdn'); map it to
    plotly's ``include_plotlyjs`` argument (which spells full-inline as ``True``,
    not the literal string 'inline').
    """
    import plotly.io as pio

    include_plotlyjs: bool | str = True if plotly_js_mode == "inline" else "cdn"
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=True)


#: renderer-kind -> renderer function. Extend here when EDA families are added.
_EDA_RENDERERS = {
    "eda_cross_sim_identity": _render_cross_sim_identity,
}
