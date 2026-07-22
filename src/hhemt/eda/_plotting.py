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

from hhemt.eda._dem_resolution_plots import (
    _render_dem_resolution_cost_error,
    _render_dem_resolution_coupling_table,
    _render_dem_resolution_diff_maps,
    _render_dem_resolution_error_ecdf,
)
from hhemt.report_plot_ids import canonical_plot_id
from hhemt.report_renderers._figure_emission import emit_plot_with_sources

if TYPE_CHECKING:
    from hhemt.config.analysis import analysis_config
    from hhemt.config.eda import eda_config


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


def _render_config_diff_maps(
    root: Path,
    *,
    cfg_analysis: analysis_config,
    eda_cfg: eda_config,
) -> Path:
    """Config-diff-maps EDA plot (redesign, /design-figure iteration 1).

    Reads the consolidated `{root}/sensitivity_datatree.zarr` directly (per-cell
    max_wlevel_m + per-conduit max_flow_cms + per-sub compute-config attrs) plus one
    sub's hydraulics.inp for conduit geometry, and renders: a cross-config
    identity+absolute-diff table, and per byte-identical config group the SIGNED diff
    and percent-diff maps (DEM cells + SWMM conduits) vs the serial-CPU baseline with
    serial reference maps. Compute-config labels are derived from config attrs (never
    the sa_id name). Emits under MASTER-ROOTED plots/eda/ as config_diff_maps.html.

    plot_id `config_diff_maps` (== on-disk stem, ADR-2). Existing bundles carrying the
    legacy `eda_cross_sim_identity` enabled_plots key are normalized to this kind by
    config/eda.py::_rewrite_legacy_eda_plot_kind (Bundle.eda() back-compat).
    """
    from hhemt.eda._config_diff import build_config_diff_figure, config_diff_source_paths

    plot_id = canonical_plot_id("config_diff_maps")
    fig = build_config_diff_figure(root)

    output_path = root / "plots" / "eda" / f"{plot_id}.html"
    html_text = _fig_to_html(fig, plotly_js_mode=eda_cfg.plotly_js_mode)
    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths=config_diff_source_paths(root),  # consolidated tree + one hydraulics.inp
        analysis_dir=root,
        output_format="html",
    )


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
    "config_diff_maps": _render_config_diff_maps,
    "dem_resolution_cost_error": _render_dem_resolution_cost_error,
    "dem_resolution_diff_maps": _render_dem_resolution_diff_maps,
    "dem_resolution_error_ecdf": _render_dem_resolution_error_ecdf,
    "dem_resolution_coupling_table": _render_dem_resolution_coupling_table,
}
