"""Per-sim renderer: 3-panel peak-flood-depth + WSE + event hydrology figure.

Iter-3 (2026-04-28) of the per-sim flood-depth figure. Iter-2 user feedback:

- WSE colorbar is too similar to depth's YlGnBu — switch WSE to a perceptually
  distinct colormap (`cividis` — dark-navy → yellow).
- Depth colorbar must be quantized (true discrete bins) at 0.01 / 0.05 / 0.10 /
  0.50 / 1.00 m, matching the user reference at scratch line 1977.
- Move colorbars below each map panel (saves horizontal width and matches the
  reference visual).
- Add a third panel showing the event's rainfall + boundary-condition water
  level as a stacked time-series, matching the "Event hydrology" reference at
  scratch line 2007. Y-axis label on the lower sub-panel reads "Boundary
  condition water level (m)" per user terminology.

Dispatches per `_get_enabled_model_types()` (Gotcha 5) — TRITON-SWMM coupled
fixtures use `output_tritonswmm_triton_summary`; TRITON-only uses
`output_triton_only_summary`; SWMM-only emits a model-type-skip placeholder.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import rioxarray as rxr
import swmmio
import xarray as xr
from matplotlib.colors import Normalize
from plotly.subplots import make_subplots

from hhemt import units, utils
from hhemt.report_renderers._map_bounds import (
    compute_padded_square_bounds,
)
from hhemt.report_renderers.system_overview import _resolve_inp_sources


def _build_discrete_depth_colorscale(
    boundaries,
    vmin: float,
    vmax: float,
    base_cmap_name: str,
):
    """Build a Plotly stepped colorscale matching the matplotlib BoundaryNorm
    behaviour the renderer used pre-iter-19. Returns the colorscale list AND
    the per-band hex colors so callers can mirror them on a legend.

    The colorscale has `len(boundaries)` bands. The first band covers
    `[vmin, boundaries[0])` (lowest depth class); each subsequent band covers
    `[boundaries[i-1], boundaries[i])`; the topmost band covers
    `[boundaries[-1], vmax]`. Plotly's stepped-colorscale idiom is to repeat
    each interior fraction with two stops — one closing the previous band
    color and one opening the next.
    """
    cmap = plt.get_cmap(base_cmap_name)
    n_bands = len(boundaries)
    # Sample the base colormap at evenly-spaced positions in [0.2, 1.0] so the
    # lowest band still has visible saturation against a white panel.
    positions = [0.2 + 0.8 * i / (n_bands - 1) for i in range(n_bands)] if n_bands > 1 else [0.6]
    rgba = [cmap(p) for p in positions]
    hex_colors = [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b, _a in rgba]
    # Normalize boundaries into [0, 1] colorscale fraction space.
    span = max(vmax - vmin, 1e-12)
    norm_b = [max(0.0, min(1.0, (b - vmin) / span)) for b in boundaries]
    cs = [[0.0, hex_colors[0]]]
    for i in range(n_bands - 1):
        cs.append([norm_b[i], hex_colors[i]])
        cs.append([norm_b[i], hex_colors[i + 1]])
    cs.append([1.0, hex_colors[-1]])
    return cs, hex_colors


if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.report import report_config
    from hhemt.config.static_plots import StaticPlotBaseConfig


def _shared_depth_max(analysis, target_crs):
    """Return the global vmax for the peak-flood-depth colorbar across every
    event_iloc (iter-19 user request — depth colorbar must be the same range
    on every per-event figure). vmin is hard-pinned at the user-locked 0.01
    "under" threshold; only the upper bound is computed cross-event. Returns
    None if no event exposes a usable summary (caller falls back to per-event
    range).
    """
    enabled = analysis._get_enabled_model_types()
    if "tritonswmm" not in enabled and "triton" not in enabled:
        return None
    watershed_shp = analysis._system.cfg_system.watershed_gis_polygon
    watershed_gdf = gpd.read_file(watershed_shp)
    g_max = float("-inf")
    try:
        tree = analysis.process.open_datatree()
    except (ValueError, FileNotFoundError):
        return None
    group = "/tritonswmm/triton" if "tritonswmm" in enabled else "/triton_only/triton"
    if group not in tree.groups:
        return None
    ds_all = tree[group].to_dataset()
    for _ev in analysis.df_sims.index:
        try:
            da_ev = ds_all["max_wlevel_m"].sel(event_iloc=int(_ev))
            if da_ev.rio.crs is not None and da_ev.rio.crs != target_crs:
                da_ev = da_ev.rio.reproject(target_crs)
            if da_ev.rio.crs is not None and watershed_gdf.crs is not None and watershed_gdf.crs != da_ev.rio.crs:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir) / "watershed_reprojected.geojson"
                    watershed_gdf.to_crs(da_ev.rio.crs).to_file(tmp_path, driver="GeoJSON")
                    m = utils.create_mask_from_shapefile(da_ev, tmp_path)
            else:
                m = utils.create_mask_from_shapefile(da_ev, watershed_shp)
            d_ev = da_ev.where(m & (da_ev > 0))
            d_max_obj = d_ev.max()
            v_max = float(
                d_max_obj.compute() if hasattr(d_max_obj, "compute") else d_max_obj,
            )
            if np.isfinite(v_max):
                g_max = max(g_max, v_max)
        except KeyError:
            continue
    if not np.isfinite(g_max) or g_max <= 0.01:
        return None
    return g_max


def _shared_wse_range(analysis, target_crs, dem_da, map_cfg=None):
    """Return (vmin, vmax) for the WSE colorbar, computed once across every
    event_iloc so all per-event figures share a colorbar (iter-15 user
    request). Walks every event's TRITON summary, masks depth >
    dry_threshold_m + watershed, builds WSE = depth + DEM, and accumulates
    the global quantile-clipped range across all wetted cells from all
    events. Falls back to per-event range if no events expose a usable
    summary.

    F-I-2 (interactive-report-renderers Phase 3): WSE colorbar range uses
    quantile clip across the union of wetted cells, not min/max. Suppresses
    building-on-top dry-cell artifacts that would otherwise dominate the
    cross-event colorbar scale. Quantiles read from `map_cfg.wse_clip_quantile_lower`
    / `_upper` when `map_cfg` is supplied; falls back to (q01, q99) when
    `map_cfg` is None (preserves callability from sites that don't have the
    cfg in scope).
    """
    enabled = analysis._get_enabled_model_types()
    if "tritonswmm" not in enabled and "triton" not in enabled:
        return None
    watershed_shp = analysis._system.cfg_system.watershed_gis_polygon
    watershed_gdf = gpd.read_file(watershed_shp)
    if map_cfg is not None:
        dry_threshold = float(map_cfg.dry_threshold_m)
        q_lower = float(map_cfg.wse_clip_quantile_lower)
        q_upper = float(map_cfg.wse_clip_quantile_upper)
    else:
        dry_threshold = 0.0
        q_lower, q_upper = 0.01, 0.99
    # F-I-2 building filter: exclude cells whose DEM lands on a wall/building
    # so building-rooftop WSE values do not pollute the cross-event quantile
    # clip. Threshold mirrors system_overview's wall-overlay rule.
    sys_cfg = analysis._system.cfg_system
    wall_th = None
    if sys_cfg.dem_building_height is not None and sys_cfg.dem_outside_watershed_height is not None:
        try:
            buffer_m = analysis.cfg_analysis.report.system_map.elevation_style.wall_threshold_buffer_m
        except AttributeError:
            buffer_m = 40.0
        wall_th = min(sys_cfg.dem_building_height, sys_cfg.dem_outside_watershed_height) - buffer_m
    try:
        tree = analysis.process.open_datatree()
    except (ValueError, FileNotFoundError):
        return None
    group = "/tritonswmm/triton" if "tritonswmm" in enabled else "/triton_only/triton"
    if group not in tree.groups:
        return None
    ds_all = tree[group].to_dataset()
    all_wet_values: list[np.ndarray] = []
    for _ev in analysis.df_sims.index:
        try:
            da_ev = ds_all["max_wlevel_m"].sel(event_iloc=int(_ev))
            if da_ev.rio.crs is not None and da_ev.rio.crs != target_crs:
                da_ev = da_ev.rio.reproject(target_crs)
            if da_ev.rio.crs is not None and watershed_gdf.crs is not None and watershed_gdf.crs != da_ev.rio.crs:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir) / "watershed_reprojected.geojson"
                    watershed_gdf.to_crs(da_ev.rio.crs).to_file(tmp_path, driver="GeoJSON")
                    m = utils.create_mask_from_shapefile(da_ev, tmp_path)
            else:
                m = utils.create_mask_from_shapefile(da_ev, watershed_shp)
            wse_ev = (da_ev + dem_da).where(m & (da_ev > dry_threshold))
            if wall_th is not None:
                wse_ev = wse_ev.where(dem_da < wall_th)
            vals = wse_ev.values
            if hasattr(vals, "compute"):
                vals = vals.compute()
            finite = vals[np.isfinite(vals)]
            if finite.size > 0:
                all_wet_values.append(finite)
        except KeyError:
            continue
    if not all_wet_values:
        return None
    combined = np.concatenate(all_wet_values)
    g_min = float(np.nanquantile(combined, q_lower))
    g_max = float(np.nanquantile(combined, q_upper))
    if not np.isfinite(g_min) or not np.isfinite(g_max) or g_max <= g_min:
        return None
    return (g_min, g_max)


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
    *,
    event_iloc: int | None = None,
    static_cfg: StaticPlotBaseConfig | None = None,
    **kwargs,
) -> Path:
    """Render the 3-panel depth + WSE + hydrology figure for one event_iloc.

    When ``static_cfg`` is provided (publication static-plots path, ADR-8) the
    matplotlib branch is FORCED (publication is matplotlib-only per ADR-3) and
    the figure geometry, depth colormap / out-of-range colors, colorbar norm,
    base typography, and emit format are driven by the PeakFloodDepthStaticConfig
    rather than report_cfg. ``static_cfg=None`` (the report path) is byte-
    unchanged. ``event_iloc`` defaults to None only so the dispatcher can omit it
    without a TypeError; the static generator always threads ``--event-iloc``, so
    a per-sim render still receives it. ``**kwargs`` tolerates dispatcher-passed
    keywords this renderer does not consume.
    """
    from hhemt.config.report import resolve_target_crs
    from hhemt.report_renderers._figure_emission import (
        add_panel_label,
        emit_plot_with_sources,
        per_sim_map_ticks,
    )
    from hhemt.report_renderers._provenance import (
        ProvenanceLog,
        ProvenanceRef,
    )
    from hhemt.report_renderers.per_sim_conduit_flow import (
        _emit_model_type_skip_placeholder,
    )
    from hhemt.report_renderers.system_overview import _apply_rcparams

    _apply_rcparams(report_cfg)
    cfg = report_cfg.per_sim.peak_flood_depth
    prov = ProvenanceLog()

    enabled = analysis._get_enabled_model_types()
    if "tritonswmm" in enabled:
        triton_group = "/tritonswmm/triton"
    elif "triton" in enabled:
        triton_group = "/triton_only/triton"
    else:
        return _emit_model_type_skip_placeholder(
            output_path,
            "peak_flood_depth not applicable for swmm-only analyses",
            report_cfg.figure_defaults.savefig_dpi,
        )

    static_backend = getattr(
        getattr(report_cfg, "interactive", None),
        "static_backend",
        "plotly",
    )
    # Publication path (ADR-3): when static_cfg is provided, FORCE the matplotlib
    # branch regardless of report_cfg.interactive.static_backend (static plots are
    # matplotlib-only) and apply the publication base typography (font_family + a
    # base size) so the publication font fields are not silently inert. Full
    # per-element FontTarget threading + rc_context isolation is routed to the
    # post-Phase-1 follow-up. report-mode (static_cfg is None) is unchanged.
    if static_cfg is not None:
        from hhemt.config.viz_vocabulary import FontTarget

        plt.rcParams["font.family"] = static_cfg.font_family
        plt.rcParams["font.size"] = static_cfg.font_sizes[FontTarget.axis_label]
    elif static_backend == "plotly":
        return _render_plotly_branch(
            analysis,
            report_cfg,
            output_path,
            event_iloc=event_iloc,
            triton_group=triton_group,
            prov=prov,
        )

    target_crs = resolve_target_crs(analysis, report_cfg)
    sys_paths = analysis._system.sys_paths
    triton_summary_path = analysis.analysis_paths.analysis_datatree_zarr

    # ---- Depth raster from consolidated DataTree ------------------------
    tree = analysis.process.open_datatree()
    if triton_group not in tree.groups:
        raise AssertionError(
            f"consolidated tree missing expected group {triton_group}; available: {sorted(tree.groups)}"
        )
    ds = tree[triton_group].to_dataset()
    da = ds["max_wlevel_m"].sel(event_iloc=event_iloc)
    if da.rio.crs is not None and da.rio.crs != target_crs:
        da = da.rio.reproject(target_crs)
    wlevel_attrs = dict(da.attrs)
    wlevel_name = da.name

    # ---- DEM raster -----------------------------------------------------
    dem_da = rxr.open_rasterio(sys_paths.dem_processed).squeeze()
    if dem_da.rio.crs is not None and dem_da.rio.crs != target_crs:
        dem_da = dem_da.rio.reproject(target_crs)
    dem_attrs = dict(dem_da.attrs)

    # ---- Watershed mask -------------------------------------------------
    watershed_shp = analysis._system.cfg_system.watershed_gis_polygon
    watershed_gdf = gpd.read_file(watershed_shp)
    if da.rio.crs is not None and watershed_gdf.crs is not None and watershed_gdf.crs != da.rio.crs:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "watershed_reprojected.geojson"
            watershed_gdf.to_crs(da.rio.crs).to_file(tmp_path, driver="GeoJSON")
            mask = utils.create_mask_from_shapefile(da, tmp_path)
    else:
        mask = utils.create_mask_from_shapefile(da, watershed_shp)
    da_masked = da.where(mask & (da > 0))

    try:
        wse_da = da + dem_da
    except Exception:
        wse_da = da + dem_da.interp_like(da, method="nearest")
    wse_masked = wse_da.where(mask & (da > 0))

    # ---- Per-scenario weather time series for the hydrology panel -------
    from hhemt.report_renderers._hydrology_panel import (
        draw_event_hydrology_panel,
        load_event_hydrology_data,
    )

    weather_event_indexers = analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
    weather_path = Path(analysis.cfg_analysis.weather_timeseries)
    hydro_data = load_event_hydrology_data(
        weather_path,
        analysis.cfg_analysis,
        weather_event_indexers,
    )
    rainfall = hydro_data["rainfall"]

    # ---- Path relpaths --------------------------------------------------
    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    triton_summary_rel = os.path.relpath(
        str(Path(triton_summary_path).resolve()),
        analysis_root,
    )
    watershed_rel = os.path.relpath(
        str(Path(watershed_shp).resolve()),
        analysis_root,
    )
    dem_rel = os.path.relpath(
        str(Path(sys_paths.dem_processed).resolve()),
        analysis_root,
    )
    weather_rel = os.path.relpath(
        str(Path(weather_path).resolve()),
        analysis_root,
    )

    # ---- Figure layout: 1×3 columns, each column with a sub-gridspec ----
    # Subiteration 9.5 — sourced from DEM (same source conduit_flow uses) so
    # both per-sim renderers see IDENTICAL bounds + map_aspect, and explicit
    # set_xlim/set_ylim below produce IDENTICAL tick ranges between toggles.
    bounds = (
        dem_da.rio.bounds()
        if dem_da.rio.crs is not None
        else (
            float(dem_da.x.min()),
            float(dem_da.y.min()),
            float(dem_da.x.max()),
            float(dem_da.y.max()),
        )
    )
    map_aspect = (bounds[2] - bounds[0]) / max(bounds[3] - bounds[1], 1e-9)
    map_cfg = report_cfg.per_sim.map
    if static_cfg is not None:
        # Publication exact dimensions (data-viz OE-1): the user owns the figure
        # size via figure_width/height_inches; bypass the report aspect-math width
        # derivation. layout="constrained" auto-fits panel contents within the
        # declared size, and the publication emit uses bbox_inches=None so the
        # saved figure is exactly this size.
        fig = plt.figure(
            figsize=(static_cfg.figure_width_inches, static_cfg.figure_height_inches),
            layout="constrained",
        )
    else:
        h = float(cfg.figsize_inches[1]) if hasattr(cfg, "figsize_inches") else map_cfg.fallback_h_inches
        fig_width = h * (2 * map_aspect * map_cfg.fig_width_panel_pad + 1.0)
        fig = plt.figure(figsize=(fig_width, h), layout="constrained")
    outer = fig.add_gridspec(1, 3, width_ratios=list(map_cfg.outer_width_ratios), wspace=map_cfg.outer_wspace)
    _MAP_TO_CBAR_HEIGHT_RATIO = map_cfg.map_to_cbar_height_ratio
    gs_depth = outer[0, 0].subgridspec(2, 1, height_ratios=[_MAP_TO_CBAR_HEIGHT_RATIO, 1])
    gs_wse = outer[0, 1].subgridspec(2, 1, height_ratios=[_MAP_TO_CBAR_HEIGHT_RATIO, 1])
    gs_depth_cbar = gs_depth[1, 0].subgridspec(1, 3, width_ratios=list(map_cfg.cbar_inner_width_ratios))
    gs_wse_cbar = gs_wse[1, 0].subgridspec(1, 3, width_ratios=list(map_cfg.cbar_inner_width_ratios))
    # Hydro column: outer 2-row split mirrors map columns so the hydro panel's
    # vertical extent matches the map (not map+colorbar) — iter-4 feedback:
    # "boundary height of the event hydrology figure should match the height
    # of the flood figures". Top row holds the rainfall+BC stack; bottom row
    # is intentionally empty (aligns with the colorbar slot on the map cols).
    gs_hydro_outer = outer[0, 2].subgridspec(
        2,
        1,
        height_ratios=[_MAP_TO_CBAR_HEIGHT_RATIO, 1],
    )
    gs_hydro_inner = gs_hydro_outer[0, 0].subgridspec(2, 1, height_ratios=[1, 1])
    ax_depth = fig.add_subplot(gs_depth[0, 0])
    cax_depth = fig.add_subplot(gs_depth_cbar[0, 1])
    ax_wse = fig.add_subplot(gs_wse[0, 0], sharex=ax_depth, sharey=ax_depth)
    cax_wse = fig.add_subplot(gs_wse_cbar[0, 1])
    ax_rain = fig.add_subplot(gs_hydro_inner[0, 0])
    ax_bc = fig.add_subplot(gs_hydro_inner[1, 0], sharex=ax_rain)

    # ---- Depth panel: YlGnBu continuous linear --------------------------
    # Iter-19 (2026-04-29): switched from discrete BoundaryNorm to a continuous
    # linear Normalize per user request. vmin is pinned at 0.01 m (user-locked
    # "under" threshold — cells below 0.01 render as white via
    # `cmap.set_under("white")`). vmax is computed once across every event_iloc
    # so the depth colorbar shares its range across all per-event figures.
    depth_ref = ProvenanceRef(
        source_path=triton_summary_rel,
        variable=str(wlevel_name) if wlevel_name is not None else "max_wlevel_m",
        attrs=wlevel_attrs,
        selection={"event_iloc": int(event_iloc)},
        transform="masked to watershed and depth>0",
    )
    # vmin/vmax: publication static_cfg.vmin/vmax override the report defaults
    # when set; otherwise the report-path derivation (pinned vmin + shared-or-
    # local vmax) is byte-unchanged.
    if static_cfg is not None and static_cfg.vmin is not None:
        depth_vmin = float(static_cfg.vmin)
    else:
        depth_vmin = map_cfg.depth_vmin
    if static_cfg is not None and static_cfg.vmax is not None:
        depth_vmax = float(static_cfg.vmax)
    else:
        shared_max = _shared_depth_max(analysis, target_crs)
        if shared_max is not None:
            depth_vmax = float(shared_max)
        else:
            d_max_obj = da_masked.max()
            d_max_local = float(
                d_max_obj.compute() if hasattr(d_max_obj, "compute") else d_max_obj,
            )
            depth_vmax = (
                d_max_local if (np.isfinite(d_max_local) and d_max_local > depth_vmin) else map_cfg.depth_vmax_fallback
            )
    # colormap + out-of-range colors: dual-source (report uses map_cfg; publication
    # uses the PeakFloodDepthStaticConfig depth_* fields + optional over/bad).
    _depth_cmap_name = static_cfg.depth_cmap if static_cfg is not None else map_cfg.depth_cmap
    _depth_under = static_cfg.depth_under_color if static_cfg is not None else map_cfg.depth_under_color
    depth_cmap = plt.get_cmap(_depth_cmap_name).copy()
    depth_cmap.set_under(_depth_under)
    if static_cfg is not None and static_cfg.depth_over_color is not None:
        depth_cmap.set_over(static_cfg.depth_over_color)
    if static_cfg is not None and static_cfg.set_bad_color is not None:
        depth_cmap.set_bad(static_cfg.set_bad_color)
    # colorbar norm: report path is a continuous linear Normalize (iter-19). The
    # publication path activates BoundaryNorm/LogNorm/SymLogNorm by colorbar_norm
    # (new conditional code — these controls have NO map_cfg counterpart, so a
    # dual-source swap would render them silently inert).
    _cbar_extend = static_cfg.colorbar_extend if static_cfg is not None else "min"
    if static_cfg is not None and static_cfg.colorbar_norm != "linear":
        from matplotlib.colors import BoundaryNorm, LogNorm, SymLogNorm

        if static_cfg.colorbar_norm == "boundary" and static_cfg.colorbar_boundaries:
            depth_norm = BoundaryNorm(list(static_cfg.colorbar_boundaries), ncolors=depth_cmap.N)
            _norm_label = "BoundaryNorm"
        elif static_cfg.colorbar_norm == "log":
            depth_norm = LogNorm(vmin=max(depth_vmin, 1e-9), vmax=depth_vmax)
            _norm_label = "LogNorm"
        elif static_cfg.colorbar_norm == "symlog":
            depth_norm = SymLogNorm(linthresh=max(depth_vmin, 1e-9), vmin=depth_vmin, vmax=depth_vmax)
            _norm_label = "SymLogNorm"
        else:
            depth_norm = Normalize(vmin=depth_vmin, vmax=depth_vmax)
            _norm_label = "Normalize"
    else:
        depth_norm = Normalize(vmin=depth_vmin, vmax=depth_vmax)
        _norm_label = "Normalize"
    with prov.artist(
        axes_id="ax_depth",
        kind="image",
        note=f"peak flood depth raster ({_norm_label}, event {event_iloc})",
    ) as a:
        a.add_channel("z", depth_ref)
        a.add_channel(
            "color",
            depth_ref,
            cmap=_depth_cmap_name,
            vmin=depth_vmin,
            vmax=depth_vmax,
            norm=_norm_label,
            extend=_cbar_extend,
            under_color=_depth_under,
        )
        depth_img = da_masked.plot(  # noqa: F841
            ax=ax_depth,
            x="x",
            y="y",
            cmap=depth_cmap,
            norm=depth_norm,
            add_colorbar=False,
        )
    cbar_d = fig.colorbar(
        ax_depth.collections[0] if ax_depth.collections else depth_img,
        cax=cax_depth,
        orientation="horizontal",
        extend=_cbar_extend,
    )
    cbar_d.set_label(units.depth_label(analysis._system.cfg_system.crs.vertical_epsg))
    ax_depth.set_aspect("equal")
    ax_depth.set_xlim(bounds[0], bounds[2])
    ax_depth.set_ylim(bounds[1], bounds[3])
    ax_depth.set_title("Peak flood depth")
    from hhemt.config.report import resolve_target_crs

    target_crs_resolved = resolve_target_crs(analysis, report_cfg)
    crs_for_labels = target_crs_resolved.to_epsg()
    ax_depth.tick_params(axis="both", labelsize=map_cfg.tick_labelsize)
    ax_depth.set_xlabel(units.easting_axis_label(crs_for_labels), fontsize=map_cfg.axis_label_fontsize)
    ax_depth.set_ylabel(units.northing_axis_label(crs_for_labels), fontsize=map_cfg.axis_label_fontsize)
    watershed_ref = ProvenanceRef(
        source_path=watershed_rel,
        variable="watershed_polygon",
        attrs={},
    )
    with prov.artist(
        axes_id="ax_depth",
        kind="patch",
        note="watershed boundary overlay",
    ) as a:
        a.add_channel("x", watershed_ref)
        a.add_channel("y", watershed_ref)
        if watershed_gdf.crs is not None:
            watershed_gdf.to_crs(target_crs).boundary.plot(
                ax=ax_depth,
                color=map_cfg.watershed_overlay_color,
                linewidth=map_cfg.watershed_overlay_width,
            )
        else:
            watershed_gdf.boundary.plot(
                ax=ax_depth,
                color=map_cfg.watershed_overlay_color,
                linewidth=map_cfg.watershed_overlay_width,
            )

    # ---- WSE panel: cividis linear --------------------------------------
    # Iter-15 (2026-04-29): the colorbar range is shared across every event
    # in the analysis so the user can compare WSE between event_iloc figures
    # by eye. Falls back to per-event range only if the cross-event scan
    # failed (no other event has a usable summary).
    shared = _shared_wse_range(analysis, target_crs, dem_da, map_cfg=map_cfg)
    if shared is not None:
        wse_min, wse_max = shared
    else:
        wse_min_obj = wse_masked.min()
        wse_max_obj = wse_masked.max()
        wse_min = float(wse_min_obj.compute() if hasattr(wse_min_obj, "compute") else wse_min_obj)
        wse_max = float(wse_max_obj.compute() if hasattr(wse_max_obj, "compute") else wse_max_obj)
        if not np.isfinite(wse_min) or not np.isfinite(wse_max) or wse_max <= wse_min:
            wse_min, wse_max = map_cfg.wse_fallback_range

    wse_ref_depth = ProvenanceRef(
        source_path=triton_summary_rel,
        variable=str(wlevel_name) if wlevel_name is not None else "max_wlevel_m",
        attrs=wlevel_attrs,
        selection={"event_iloc": int(event_iloc)},
        transform="depth, summed with DEM elevation",
    )
    wse_ref_dem = ProvenanceRef(
        source_path=dem_rel,
        variable="dem_elev_m",
        attrs=dem_attrs,
        transform="reprojected to target_crs",
    )
    with prov.artist(
        axes_id="ax_wse",
        kind="image",
        note=f"water surface elevation = depth + DEM (event {event_iloc})",
    ) as a:
        a.add_channel("z", wse_ref_depth)
        a.add_channel("z", wse_ref_dem)
        a.add_channel(
            "color",
            wse_ref_depth,
            cmap=map_cfg.wse_cmap,
            vmin=wse_min,
            vmax=wse_max,
        )
        wse_img = wse_masked.plot(  # noqa: F841
            ax=ax_wse,
            x="x",
            y="y",
            cmap=map_cfg.wse_cmap,
            vmin=wse_min,
            vmax=wse_max,
            add_colorbar=False,
        )
    cbar_w = fig.colorbar(
        ax_wse.collections[0] if ax_wse.collections else wse_img,
        cax=cax_wse,
        orientation="horizontal",
    )
    cbar_w.set_label(units.wse_label(analysis._system.cfg_system.crs.vertical_epsg))
    ax_wse.set_aspect("equal")
    ax_wse.set_title("Water surface elevation")
    # C8 — middle panel shares y-axis with ax_depth (sharey=ax_depth above);
    # hide redundant y-tick labels and drop the ylabel so the gap between
    # the depth and WSE panels collapses to the bare wspace allocation.
    ax_wse.tick_params(axis="both", labelsize=map_cfg.tick_labelsize)
    ax_wse.tick_params(axis="y", labelleft=False)
    ax_wse.set_xlabel(units.easting_axis_label(crs_for_labels), fontsize=map_cfg.axis_label_fontsize)
    ax_wse.set_ylabel("")
    with prov.artist(
        axes_id="ax_wse",
        kind="patch",
        note="watershed boundary overlay",
    ) as a:
        a.add_channel("x", watershed_ref)
        a.add_channel("y", watershed_ref)
        if watershed_gdf.crs is not None:
            watershed_gdf.to_crs(target_crs).boundary.plot(
                ax=ax_wse,
                color=map_cfg.watershed_overlay_color,
                linewidth=map_cfg.watershed_overlay_width,
            )
        else:
            watershed_gdf.boundary.plot(
                ax=ax_wse,
                color=map_cfg.watershed_overlay_color,
                linewidth=map_cfg.watershed_overlay_width,
            )

    # Subiteration 9.5 — explicit identical ticks AFTER both panels render
    # (xarray's `.plot()` resets the axis ticks via auto-locator, clobbering
    # any earlier `set_xticks` / `set_yticks` call). Setting on ax_wse here
    # propagates to ax_depth via the existing `sharex=sharey=ax_depth` link.
    _xticks, _yticks = per_sim_map_ticks(bounds)
    ax_wse.set_xticks(_xticks)
    ax_wse.set_yticks(_yticks)
    # Re-apply lims AFTER set_xticks (matplotlib expands lims to fit ticks);
    # ensures both panels stay bounded to the DEM extent.
    ax_wse.set_xlim(bounds[0], bounds[2])
    ax_wse.set_ylim(bounds[1], bounds[3])

    # ---- Hydrology panel: delegated to shared helper (Subiteration 9.2 C6/C7)
    bc_min, bc_max = draw_event_hydrology_panel(
        ax_rain,
        ax_bc,
        hydro_data=hydro_data,
        weather_rel_path=weather_rel,
        event_iloc=event_iloc,
        cfg_analysis=analysis.cfg_analysis,
        panel_cfg=report_cfg.per_sim.hydrology_panel,
        prov=prov,
    )

    # A4 — panel labels
    add_panel_label(ax_depth, "(a)")
    add_panel_label(ax_wse, "(b)")
    add_panel_label(ax_rain, "(c)")

    source_paths: list[Path] = [
        Path(triton_summary_path),
        Path(watershed_shp),
        Path(sys_paths.dem_processed),
        Path(weather_path),
    ]
    max_obj = da_masked.max()
    wlevel_m_max = max_obj.compute() if hasattr(max_obj, "compute") else max_obj
    cell_count_obj = da_masked.notnull().sum()
    valid_cell_count = cell_count_obj.compute() if hasattr(cell_count_obj, "compute") else cell_count_obj
    return emit_plot_with_sources(
        fig,
        output_path,
        source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        dpi=(static_cfg.savefig_dpi if static_cfg is not None else report_cfg.figure_defaults.savefig_dpi),
        output_format=(static_cfg.output_format if static_cfg is not None else "png"),
        bbox_inches_tight=(static_cfg.bbox_inches_tight if static_cfg is not None else True),
        emit_preview=(static_cfg is None),
        manifest_data={
            "event_iloc": int(event_iloc),
            "depth_m_max": float(wlevel_m_max),
            "valid_cell_count": int(valid_cell_count),
            "wse_m_range": [wse_min, wse_max],
            "rainfall_max_mm_per_hr": float(np.nanmax(rainfall)),
            "bc_water_level_range_m": [bc_min, bc_max],
            "depth_boundaries_m": list(map_cfg.depth_boundaries_m),
        },
        provenance=prov,
    )


def _build_peak_flood_depth_figure(
    analysis,
    report_cfg,
    output_path: Path,
    *,
    event_iloc: int,
    triton_group: str,
    prov,
):
    """Figure-construction seam for the Plotly peak-flood-depth render.

    Builds the `go.Figure` and computes the locals the emission portion of
    `_render_plotly_branch` needs, returning them as a tuple. Extracted
    verbatim from `_render_plotly_branch` (pure-extraction refactor) so a test
    can obtain the figure object before HTML serialization.
    """
    from hhemt.config.report import resolve_target_crs

    # Side-effect import: registers `triton_journal` Plotly template at import time.
    from hhemt.report_renderers import _plotly_theme  # noqa: F401
    from hhemt.report_renderers._hydrology_panel import (
        load_event_hydrology_data,
    )
    from hhemt.report_renderers._provenance import ProvenanceRef

    cfg = report_cfg.per_sim.peak_flood_depth  # noqa: F841 — parity with mpl branch
    map_cfg = report_cfg.per_sim.map
    interactive_cfg = report_cfg.per_sim.interactive

    target_crs = resolve_target_crs(analysis, report_cfg)
    sys_paths = analysis._system.sys_paths
    triton_summary_path = analysis.analysis_paths.analysis_datatree_zarr

    # ---- Data prep (mirror of matplotlib branch) ------------------------
    tree = analysis.process.open_datatree()
    if triton_group not in tree.groups:
        raise AssertionError(
            f"consolidated tree missing expected group {triton_group}; available: {sorted(tree.groups)}"
        )
    ds = tree[triton_group].to_dataset()
    da = ds["max_wlevel_m"].sel(event_iloc=event_iloc)
    if da.rio.crs is not None and da.rio.crs != target_crs:
        da = da.rio.reproject(target_crs)
    wlevel_attrs = dict(da.attrs)
    wlevel_name = da.name

    dem_da = rxr.open_rasterio(sys_paths.dem_processed).squeeze()
    if dem_da.rio.crs is not None and dem_da.rio.crs != target_crs:
        dem_da = dem_da.rio.reproject(target_crs)
    dem_attrs = dict(dem_da.attrs)

    watershed_shp = analysis._system.cfg_system.watershed_gis_polygon
    watershed_gdf = gpd.read_file(watershed_shp)
    if da.rio.crs is not None and watershed_gdf.crs is not None and watershed_gdf.crs != da.rio.crs:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "watershed_reprojected.geojson"
            watershed_gdf.to_crs(da.rio.crs).to_file(tmp_path, driver="GeoJSON")
            mask = utils.create_mask_from_shapefile(da, tmp_path)
    else:
        mask = utils.create_mask_from_shapefile(da, watershed_shp)
    # F-I-3: split mask around configurable dry threshold. Cells inside the
    # watershed with depth at or below `dry_threshold_m` render via a
    # neutral-grey base trace so the watershed shape stays preattentive.
    dry_threshold = map_cfg.dry_threshold_m
    wet_mask = mask & (da > dry_threshold)
    da_masked = da.where(wet_mask)
    # Iter6 (2026-05-17): dry-cell rendering uses a watershed-shaped polygon
    # fill (single go.Scatter with fill="toself") rather than a per-cell
    # Heatmap indicator. Trims ~3 MB of HTML payload (the per-cell indicator
    # raster was the dominant payload contributor; the polygon fill is O(N)
    # in vertex count, ~100s of points). dry_indicator no longer needed.

    try:
        wse_da = da + dem_da
    except Exception:
        wse_da = da + dem_da.interp_like(da, method="nearest")
    wse_masked = wse_da.where(wet_mask)

    weather_event_indexers = analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
    weather_path = Path(analysis.cfg_analysis.weather_timeseries)
    hydro_data = load_event_hydrology_data(
        weather_path,
        analysis.cfg_analysis,
        weather_event_indexers,
    )
    times_min = hydro_data["times_min"]
    rainfall = hydro_data["rainfall"]
    bc_water_level = hydro_data["bc_water_level"]
    rain_attrs = hydro_data["rain_attrs"]
    bc_attrs = hydro_data["bc_attrs"]
    rain_var = hydro_data["rain_var"]
    bc_var = hydro_data["bc_var"]

    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    triton_summary_rel = os.path.relpath(
        str(Path(triton_summary_path).resolve()),
        analysis_root,
    )
    watershed_rel = os.path.relpath(
        str(Path(watershed_shp).resolve()),
        analysis_root,
    )
    dem_rel = os.path.relpath(
        str(Path(sys_paths.dem_processed).resolve()),
        analysis_root,
    )
    weather_rel = os.path.relpath(
        str(Path(weather_path).resolve()),
        analysis_root,
    )

    # ---- Color scale and range setup ------------------------------------
    depth_vmin = map_cfg.depth_vmin
    shared_max = _shared_depth_max(analysis, target_crs)
    if shared_max is not None:
        depth_vmax = float(shared_max)
    else:
        d_max_obj = da_masked.max()
        d_max_local = float(
            d_max_obj.compute() if hasattr(d_max_obj, "compute") else d_max_obj,
        )
        depth_vmax = (
            d_max_local if (np.isfinite(d_max_local) and d_max_local > depth_vmin) else map_cfg.depth_vmax_fallback
        )
    shared_wse = _shared_wse_range(analysis, target_crs, dem_da, map_cfg=map_cfg)
    if shared_wse is not None:
        wse_min, wse_max = shared_wse
    else:
        # F-I-2: clip WSE colorbar to quantiles of WETTED + on-real-terrain
        # cells.  Suppresses building-on-top artifacts that would otherwise
        # dominate the scale and collapse usable WSE range to <5% of the
        # colorbar. Building cells reuse system_overview's wall-threshold
        # pattern: DEM >= min(dem_building_height, dem_outside_watershed_height)
        # - wall_threshold_buffer_m == building/wall.
        sys_cfg = analysis._system.cfg_system
        if sys_cfg.dem_building_height is not None and sys_cfg.dem_outside_watershed_height is not None:
            wall_th = (
                min(sys_cfg.dem_building_height, sys_cfg.dem_outside_watershed_height)
                - report_cfg.system_map.elevation_style.wall_threshold_buffer_m
            )
            terrain_mask = dem_da < wall_th
        else:
            terrain_mask = xr.ones_like(dem_da, dtype=bool)
        wse_for_range = wse_masked.where(terrain_mask)
        wse_values = wse_for_range.values
        if hasattr(wse_values, "compute"):
            wse_values = wse_values.compute()
        finite = wse_values[np.isfinite(wse_values)]
        if finite.size > 0:
            wse_min = float(np.nanquantile(finite, map_cfg.wse_clip_quantile_lower))
            wse_max = float(np.nanquantile(finite, map_cfg.wse_clip_quantile_upper))
            if not np.isfinite(wse_min) or not np.isfinite(wse_max) or wse_max <= wse_min:
                wse_min, wse_max = map_cfg.wse_fallback_range
        else:
            wse_min, wse_max = map_cfg.wse_fallback_range

    # F-I-5: cross-figure bounds parity with system_overview.  Load the same
    # SWMM hydro + hydraulics models and call the shared
    # compute_padded_square_bounds helper so per_sim map panels share their
    # x/y extents with system_overview's panels exactly.
    hydro_inp, hydraulics_inp = _resolve_inp_sources(analysis)
    hydro_model = swmmio.Model(str(hydro_inp))
    hydraulics_model = swmmio.Model(str(hydraulics_inp))
    dem_bounds_raw = (
        dem_da.rio.bounds()
        if dem_da.rio.crs is not None
        else (
            float(dem_da.x.min()),
            float(dem_da.y.min()),
            float(dem_da.x.max()),
            float(dem_da.y.max()),
        )
    )
    bounds = compute_padded_square_bounds(
        dem_bounds_raw,
        hydro_model,
        hydraulics_model,
        padding_frac=0.02,
    )

    # ---- Datashader pre-raster gate -------------------------------------
    # MV scope: single-frame pre-aggregation when depth cell count exceeds threshold.
    cell_count = int(
        da_masked.notnull().sum().compute()
        if hasattr(da_masked.notnull().sum(), "compute")
        else da_masked.notnull().sum()
    )
    use_datashader = cell_count > interactive_cfg.datashader_threshold_cells
    if use_datashader:
        import datashader as ds_lib
        import datashader.reductions as ds_reductions

        canvas = ds_lib.Canvas(plot_width=512, plot_height=512)
        depth_agg = canvas.raster(
            da_masked,
            agg=ds_reductions.max("max_wlevel_m"),
        )
        wse_agg = canvas.raster(
            wse_masked,
            agg=ds_reductions.max(),
        )
        depth_x = depth_agg.x.values
        depth_y = depth_agg.y.values
        depth_z = depth_agg.values
        wse_x = wse_agg.x.values
        wse_y = wse_agg.y.values
        wse_z = wse_agg.values
    else:
        depth_x = da_masked.x.values
        depth_y = da_masked.y.values
        depth_z = da_masked.values
        wse_x = wse_masked.x.values
        wse_y = wse_masked.y.values
        wse_z = wse_masked.values

    # ---- Build figure ---------------------------------------------------
    # 2x3 layout: depth + WSE span 2 rows (cols 1, 2); hydro col splits into
    # rainfall (row 1, col 3) + BC water level (row 2, col 3).
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"rowspan": 2}, {"rowspan": 2}, {}],
            [None, None, {}],
        ],
        row_heights=[1, 1],
        column_widths=[1, 1, 0.8],
        horizontal_spacing=0.06,
        vertical_spacing=0.06,
        subplot_titles=("Peak flood depth", "Water surface elevation", "Flood Drivers"),
    )
    # F-I-8: figure title removed (redundant with Snakemake sidebar/result-row labels).
    # F-I-9: use registered `triton_journal` template instead of plotly_white.
    # F-I-4 Option B: bottom margin expanded to 130 px to clear horizontal
    # colorbars repositioned to y=-0.22.
    # Iteration 2: showlegend=True so the dry-cell legend swatch surfaces.
    fig.update_layout(
        template="triton_journal",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.10,
            x=0.0,
            xanchor="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=40, b=130),
    )

    # Iteration 3: revert to continuous depth colorscale per user feedback
    # ("go back to the previous color bar"). The discrete BoundaryNorm-style
    # quantization from iter2 is left as the helper `_build_discrete_depth_colorscale`
    # for future use, but the live render path passes `map_cfg.depth_cmap` directly.

    # ---- Depth panel ----------------------------------------------------
    depth_ref = ProvenanceRef(
        source_path=triton_summary_rel,
        variable=str(wlevel_name) if wlevel_name is not None else "max_wlevel_m",
        attrs=wlevel_attrs,
        selection={"event_iloc": int(event_iloc)},
        transform=(
            "masked to watershed and depth>0" + ("; datashader pre-rasterized (512x512, max)" if use_datashader else "")
        ),
    )
    # Iter6 (2026-05-17): build watershed polygon coords once; reuse on both
    # map panels as the dry-cell base fill. Plotly's `fill="toself"` connects
    # the last point back to the first per linestring; multi-polygon
    # watersheds use None separators so each ring fills as its own region.
    if watershed_gdf.crs is not None:
        ws_proj_fill = watershed_gdf.to_crs(target_crs)
    else:
        ws_proj_fill = watershed_gdf
    ws_fill_x, ws_fill_y = [], []
    for geom in ws_proj_fill.geometry:
        if geom is None:
            continue
        polygons = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for poly in polygons:
            xs, ys = poly.exterior.coords.xy
            ws_fill_x.extend(list(xs) + [None])
            ws_fill_y.extend(list(ys) + [None])
    dry_ref = ProvenanceRef(
        source_path=watershed_rel,
        variable="watershed_polygon",
        attrs={
            "dry_threshold_m": float(dry_threshold),
            "render_mode": "polygon_fill",
        },
        transform=(
            "within-watershed background fill; wet Heatmap overlays on top "
            "for cells where max_wlevel_m > dry_threshold_m; remaining "
            "watershed cells show through this dry_fill_color polygon"
        ),
    )
    # F-I-3 (iter6): dry-cell base fill rendered as a watershed-shaped polygon
    # via go.Scatter(fill="toself") on the depth panel. Saves ~1.5 MB vs the
    # iter1-5 per-cell Heatmap approach.
    with prov.artist(
        axes_id="ax_depth_plotly",
        kind="patch",
        note="dry-cell watershed fill polygon (depth panel)",
    ) as a:
        a.add_channel("x", dry_ref)
        a.add_channel("y", dry_ref)
        a.add_channel("color", dry_ref, cmap=map_cfg.dry_fill_color)
        fig.add_trace(
            go.Scatter(
                x=ws_fill_x,
                y=ws_fill_y,
                fill="toself",
                fillcolor=map_cfg.dry_fill_color,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                legendgroup="dry",
                name="dry_watershed_depth",
            ),
            row=1,
            col=1,
        )
    with prov.artist(
        axes_id="ax_depth_plotly",
        kind="image",
        note=f"peak flood depth raster (event {event_iloc})"
        + (f"; datashader pre-raster (cell_count={cell_count})" if use_datashader else ""),
    ) as a:
        a.add_channel("z", depth_ref)
        a.add_channel(
            "color",
            depth_ref,
            cmap=map_cfg.depth_cmap,
            vmin=depth_vmin,
            vmax=depth_vmax,
        )
        # F-I-1: depth colormap from config (was hardcoded "YlGnBu").
        # F-I-4 Option B: colorbar pushed to y=-0.22 to clear x-axis labels.
        # Iteration 3: continuous colorscale (reverted from iter2's quantized
        # version per user request — "go back to the previous color bar").
        fig.add_trace(
            go.Heatmap(
                z=depth_z,
                x=depth_x,
                y=depth_y,
                colorscale=map_cfg.depth_cmap,
                zmin=depth_vmin,
                zmax=depth_vmax,
                colorbar=dict(
                    title=units.depth_label(analysis._system.cfg_system.crs.vertical_epsg),
                    orientation="h",
                    y=-0.22,
                    len=0.30,
                    x=0.16,
                    thickness=12,
                ),
                hovertemplate="Depth: %{z:.3f} m<br>x: %{x}<br>y: %{y}<extra></extra>",
                showlegend=False,
                name="depth",
            ),
            row=1,
            col=1,
        )

    # ---- WSE panel ------------------------------------------------------
    wse_ref_depth = ProvenanceRef(
        source_path=triton_summary_rel,
        variable=str(wlevel_name) if wlevel_name is not None else "max_wlevel_m",
        attrs=wlevel_attrs,
        selection={"event_iloc": int(event_iloc)},
        transform=(
            "depth, summed with DEM elevation"
            + ("; datashader pre-rasterized (512x512, max)" if use_datashader else "")
        ),
    )
    wse_ref_dem = ProvenanceRef(
        source_path=dem_rel,
        variable="dem_elev_m",
        attrs=dem_attrs,
        transform="reprojected to target_crs",
    )
    # F-I-3 (iter6): dry-cell base fill on the WSE panel — same polygon-fill
    # approach as the depth panel above; reuses ws_fill_x/ws_fill_y.
    with prov.artist(
        axes_id="ax_wse_plotly",
        kind="patch",
        note="dry-cell watershed fill polygon (WSE panel)",
    ) as a:
        a.add_channel("x", dry_ref)
        a.add_channel("y", dry_ref)
        a.add_channel("color", dry_ref, cmap=map_cfg.dry_fill_color)
        fig.add_trace(
            go.Scatter(
                x=ws_fill_x,
                y=ws_fill_y,
                fill="toself",
                fillcolor=map_cfg.dry_fill_color,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                legendgroup="dry",
                name="dry_watershed_wse",
            ),
            row=1,
            col=2,
        )
    with prov.artist(
        axes_id="ax_wse_plotly",
        kind="image",
        note=f"water surface elevation = depth + DEM (event {event_iloc})"
        + (f"; datashader pre-raster (cell_count={cell_count})" if use_datashader else ""),
    ) as a:
        a.add_channel("z", wse_ref_depth)
        a.add_channel("z", wse_ref_dem)
        a.add_channel(
            "color",
            wse_ref_depth,
            cmap=map_cfg.wse_cmap,
            vmin=wse_min,
            vmax=wse_max,
        )
        # F-I-1: WSE colormap from config (was hardcoded "cividis").
        # F-I-4 Option B: colorbar pushed to y=-0.22 to clear x-axis labels.
        fig.add_trace(
            go.Heatmap(
                z=wse_z,
                x=wse_x,
                y=wse_y,
                colorscale=map_cfg.wse_cmap,
                zmin=wse_min,
                zmax=wse_max,
                colorbar=dict(
                    title=units.wse_label(analysis._system.cfg_system.crs.vertical_epsg),
                    orientation="h",
                    y=-0.22,
                    len=0.30,
                    x=0.52,
                    thickness=12,
                ),
                hovertemplate="WSE: %{z:.3f} m<br>x: %{x}<br>y: %{y}<extra></extra>",
                showlegend=False,
                name="wse",
            ),
            row=1,
            col=2,
        )

    # Iteration 2: dummy-marker legend entry naming the dry-cell fill color +
    # its threshold so the grey is interpretable. Plotly Heatmap traces do not
    # surface in the legend by default; this Scatter trace plots no points
    # (NaN coords) but the marker styling renders the legend swatch.
    dry_legend_ref = ProvenanceRef(
        source_path=watershed_rel,
        variable="watershed_polygon",
        attrs={"dry_threshold_m": float(dry_threshold)},
        transform="legend swatch labelling the dry_fill_color cells",
    )
    with prov.artist(
        axes_id="ax_depth_plotly",
        kind="legend",
        note="dry-cell legend swatch (dummy Scatter, no plotted points)",
    ) as a:
        a.add_channel("color", dry_legend_ref, cmap=map_cfg.dry_fill_color)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color=map_cfg.dry_fill_color, symbol="square", size=12, line=dict(color="darkgrey", width=0.5)
                ),
                name=f"≤ {map_cfg.dry_threshold_m:g} m (dry)",
                showlegend=True,
                legendgroup="dry",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    # ---- Watershed boundary overlay on both maps ------------------------
    watershed_ref = ProvenanceRef(
        source_path=watershed_rel,
        variable="watershed_polygon",
        attrs={},
    )
    if watershed_gdf.crs is not None:
        ws_proj = watershed_gdf.to_crs(target_crs)
    else:
        ws_proj = watershed_gdf
    ws_x, ws_y = [], []
    for geom in ws_proj.boundary.values:
        if geom is None:
            continue
        # Single LineString or MultiLineString
        geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
        for line in geoms:
            xs, ys = line.coords.xy
            ws_x.extend(list(xs) + [None])
            ws_y.extend(list(ys) + [None])
    with prov.artist(
        axes_id="ax_depth_plotly",
        kind="patch",
        note="watershed boundary overlay (depth panel)",
    ) as a:
        a.add_channel("x", watershed_ref)
        a.add_channel("y", watershed_ref)
        fig.add_trace(
            go.Scatter(
                x=ws_x,
                y=ws_y,
                mode="lines",
                line=dict(color=map_cfg.watershed_overlay_color, width=map_cfg.watershed_overlay_width),
                hoverinfo="skip",
                showlegend=False,
                name="watershed",
            ),
            row=1,
            col=1,
        )
    with prov.artist(
        axes_id="ax_wse_plotly",
        kind="patch",
        note="watershed boundary overlay (WSE panel)",
    ) as a:
        a.add_channel("x", watershed_ref)
        a.add_channel("y", watershed_ref)
        fig.add_trace(
            go.Scatter(
                x=ws_x,
                y=ws_y,
                mode="lines",
                line=dict(color=map_cfg.watershed_overlay_color, width=map_cfg.watershed_overlay_width),
                hoverinfo="skip",
                showlegend=False,
                name="watershed",
            ),
            row=1,
            col=2,
        )

    # ---- Hydrology panel: rainfall (row 1, col 3) -----------------------
    rain_units = units.rainfall_provenance_units(analysis.cfg_analysis.rainfall_units)
    bc_units = (
        units.bc_provenance_units(analysis.cfg_analysis.storm_tide_units)
        if analysis.cfg_analysis.storm_tide_units
        else ""
    )
    rain_ref = ProvenanceRef(
        source_path=weather_rel,
        variable=rain_var,
        attrs=rain_attrs,
        selection={"event_iloc": int(event_iloc)},
    )
    panel_cfg = report_cfg.per_sim.hydrology_panel
    # F-I-7: convert times to hours for x-axis display + propagate the units
    # change into the provenance channel so the manifest tells the truth.
    times_hr = np.asarray(times_min, dtype=float) / units.MINUTES_PER_HOUR
    with prov.artist(
        axes_id="ax_rain_plotly",
        kind="bar",
        note="rainfall time series (event hydrology — top sub-panel)",
    ) as a:
        a.add_channel("x", rain_ref, units=units.TIME_AXIS_PROVENANCE_UNITS_HOURS)
        a.add_channel("y", rain_ref, units=rain_units)
        fig.add_trace(
            go.Bar(
                x=times_hr,
                y=rainfall,
                marker=dict(color=panel_cfg.rain_color),
                name="rainfall",
                showlegend=False,
                hovertemplate="t: %{x:.2f} hr<br>rain: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=3,
        )

    # ---- Hydrology panel: BC water level (row 2, col 3) -----------------
    bc_ref = ProvenanceRef(
        source_path=weather_rel,
        variable=bc_var if bc_var is not None else "",
        attrs=bc_attrs,
        selection={"event_iloc": int(event_iloc)},
    )
    with prov.artist(
        axes_id="ax_bc_plotly",
        kind="line2d",
        note="boundary condition water level (event hydrology — bottom sub-panel)",
    ) as a:
        a.add_channel("x", bc_ref, units=units.TIME_AXIS_PROVENANCE_UNITS_HOURS)
        a.add_channel("y", bc_ref, units=bc_units)
        fig.add_trace(
            go.Scatter(
                x=times_hr,
                y=bc_water_level,
                mode="lines",
                line=dict(color=panel_cfg.bc_line_color, width=panel_cfg.bc_line_width),
                name="bc_water_level",
                showlegend=False,
                hovertemplate="t: %{x:.2f} hr<br>BC: %{y:.3f} m<extra></extra>",
            ),
            row=2,
            col=3,
        )

    # ---- Axes setup -----------------------------------------------------
    from hhemt.config.report import resolve_target_crs

    target_crs_resolved = resolve_target_crs(analysis, report_cfg)
    crs_for_labels = target_crs_resolved.to_epsg()
    # F-I-10: link col-2 map axes to col-1 via `matches=` so interactive
    # pan/zoom on either panel synchronises the other. Plotly's
    # `shared_xaxes` in `make_subplots` does not always emit `matches`.
    for col in (1, 2):
        x_kwargs = dict(
            range=[bounds[0], bounds[2]],
            title_text=units.easting_axis_label(crs_for_labels),
            row=1,
            col=col,
        )
        y_kwargs = dict(
            range=[bounds[1], bounds[3]],
            scaleanchor=f"x{col}",
            scaleratio=1.0,
            title_text=units.northing_axis_label(crs_for_labels) if col == 1 else None,
            row=1,
            col=col,
        )
        if col == 2:
            x_kwargs["matches"] = "x"
            y_kwargs["matches"] = "y"
        fig.update_xaxes(**x_kwargs)
        fig.update_yaxes(**y_kwargs)
    # F-I-7: hydrology x-axes now in hours from event start.
    fig.update_xaxes(
        range=[float(times_hr[0]), float(times_hr[-1])],
        title_text="",
        row=1,
        col=3,
    )
    fig.update_yaxes(
        title_text=units.rainfall_axis_label(analysis.cfg_analysis.rainfall_units),
        row=1,
        col=3,
    )
    fig.update_xaxes(
        range=[float(times_hr[0]), float(times_hr[-1])],
        title_text=units.TIME_AXIS_FROM_EVENT_START_HOURS,
        row=2,
        col=3,
    )
    fig.update_yaxes(
        title_text=units.bc_water_level_axis_label(
            analysis.cfg_analysis.storm_tide_units or "m",
        ),
        row=2,
        col=3,
    )

    return (
        fig,
        triton_summary_path,
        watershed_shp,
        sys_paths,
        weather_path,
        da_masked,
        cell_count,
        bc_water_level,
        wse_min,
        wse_max,
        rainfall,
        map_cfg,
        use_datashader,
    )


def _render_plotly_branch(
    analysis,
    report_cfg,
    output_path: Path,
    *,
    event_iloc: int,
    triton_group: str,
    prov,
) -> Path:
    """Plotly MV port (pre-/design-figure): static 3-panel figure with depth raster +
    WSE raster + event hydrology (rainfall bars + BC water level line).
    Informationally congruent with the matplotlib branch — no animation,
    no per-cell hover, no layer-toggle UX. Datashader pre-rasterization fires
    when the depth-frame cell count exceeds `report_cfg.per_sim.interactive.datashader_threshold_cells`.
    """
    from hhemt.report_renderers._figure_emission import (
        emit_plot_with_sources,
    )

    (
        fig,
        triton_summary_path,
        watershed_shp,
        sys_paths,
        weather_path,
        da_masked,
        cell_count,
        bc_water_level,
        wse_min,
        wse_max,
        rainfall,
        map_cfg,
        use_datashader,
    ) = _build_peak_flood_depth_figure(
        analysis,
        report_cfg,
        output_path,
        event_iloc=event_iloc,
        triton_group=triton_group,
        prov=prov,
    )

    # ---- Emit -----------------------------------------------------------
    plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
            "autoScale2d",
            "hoverCompareCartesian",
            "hoverClosestCartesian",
            "toggleSpikelines",
        ],
        "toImageButtonOptions": {
            "format": "svg",
            "filename": "peak_flood_depth",
            "scale": 2,
        },
    }
    html_text = pio.to_html(
        fig,
        include_plotlyjs=report_cfg.interactive.plotly_js_mode,
        full_html=True,
        config=plotly_config,
    )

    # SWMM .inp sources for ADR-6 provenance. The builder consumes these for its
    # swmmio.Model construction; only the resolved paths are needed in the caller
    # scope, so re-calling the pure path resolver (no model build) is cheap.
    hydro_inp, hydraulics_inp = _resolve_inp_sources(analysis)
    source_paths: list[Path] = [
        Path(triton_summary_path),
        Path(watershed_shp),
        Path(sys_paths.dem_processed),
        Path(weather_path),
        Path(hydro_inp),
        Path(hydraulics_inp),
    ]
    max_obj = da_masked.max()
    wlevel_m_max = max_obj.compute() if hasattr(max_obj, "compute") else max_obj
    valid_cell_count = cell_count
    bc_min = float(np.nanmin(bc_water_level))
    bc_max = float(np.nanmax(bc_water_level))

    try:
        fig.write_image(
            output_path.with_suffix(".svg"),
            engine="kaleido",
            width=1400,
            height=600,
            scale=1,
        )
    except Exception as exc:  # noqa: BLE001 — Kaleido failure is non-fatal
        import logging

        logging.getLogger(__name__).warning(
            "Kaleido SVG export skipped for %s: %s",
            output_path.with_suffix(".svg"),
            exc,
        )

    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        output_format="html",
        manifest_data={
            "event_iloc": int(event_iloc),
            "depth_m_max": float(wlevel_m_max),
            "valid_cell_count": int(valid_cell_count),
            "wse_m_range": [wse_min, wse_max],
            "rainfall_max_mm_per_hr": float(np.nanmax(rainfall)),
            "bc_water_level_range_m": [bc_min, bc_max],
            "depth_boundaries_m": list(map_cfg.depth_boundaries_m),
            "datashader_used": bool(use_datashader),
        },
        provenance=prov,
    )
