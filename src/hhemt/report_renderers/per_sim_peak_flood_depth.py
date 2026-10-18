"""Per-sim renderer: where one simulated event flooded, how high the water stood, and what drove it.

The report needs one figure per event that a reader can lay beside any other event's
figure and compare by eye. This module draws it as three panels: the peak flood depth
map, the peak water surface elevation (WSE = depth + DEM) map, and the event's forcing
-- rainfall above, boundary-condition water level below -- as stacked time series. Both
maps carry horizontal colorbars beneath them.

Comparability is the reason most of this file exists. The depth and WSE colorbar ranges
are computed once across EVERY event (`_shared_depth_max`, `_shared_wse_range`), so the
same color means the same depth or elevation on every figure; a per-event range would
make two events' maps look alike when their water was not. Both ranges are
quantile-clipped over wetted cells -- depth by `PerSimMapConfig.depth_clip_quantile_upper`,
WSE by its lower/upper quantile pair -- so a few extreme cells cannot stretch the scale
over the range where most flooding sits. Keep the two clips aligned: they were once
divergent, and the depth map then had no outlier guard at all.

Which model's depth field is drawn is resolved from the enabled model types through the
`_model_arms` matrix; a SWMM-only analysis has no gridded depth field and gets a labeled
placeholder instead of a figure. There are two render paths: the matplotlib branch (a
static image, and the only path for publication figures) and the Plotly branch
(interactive HTML for the report). They share the cross-event range helpers above, but
each prepares its own rasters, and the two differ: the matplotlib branch treats a cell as
wet when depth > 0 and frames the maps on the raw DEM bounds, while the Plotly branch
uses depth > `PerSimMapConfig.dry_threshold_m` and a padded square frame shared with the
system-overview map.
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
from hhemt.figure_panels import (
    WATERSHED_LEGEND_GROUP,
    WATERSHED_LEGEND_LABEL,
    watershed_legend_marker,
)
from hhemt.report_plot_ids import event_page_reference
from hhemt.report_renderers._map_bounds import (
    compute_padded_square_bounds,
)
from hhemt.report_renderers._model_arms import groups_for
from hhemt.report_renderers.system_overview import _resolve_inp_sources


def _build_discrete_depth_colorscale(
    boundaries,
    vmin: float,
    vmax: float,
    base_cmap_name: str,
):
    """Build a Plotly stepped colorscale equivalent to a matplotlib BoundaryNorm
    quantization of depth. Returns the colorscale list AND the per-band hex
    colors so callers can mirror them on a legend. Neither render path calls it:
    both now draw depth on a continuous scale, and this is the discrete
    alternative kept available.

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


def _shared_depth_max(analysis, target_crs, map_cfg=None, model_types=None):
    """Return the global vmax for the peak-flood-depth colorbar across every
    event_iloc, so the depth colorbar has the same range on every per-event
    figure and one color means one depth throughout the report. Only the upper
    bound is computed here; callers take the lower bound from
    `PerSimMapConfig.depth_vmin`, below which cells render in the under-color.
    Returns None if no event exposes a usable summary (caller falls back to
    per-event range).
    """
    enabled = analysis._get_enabled_model_types()
    if "tritonswmm" not in enabled and "triton" not in enabled:
        return None
    watershed_shp = analysis._system.cfg_system.watershed_gis_polygon
    watershed_gdf = gpd.read_file(watershed_shp)
    g_max = float("-inf")
    all_wet_values: list[np.ndarray] = []
    try:
        tree = analysis.process.open_datatree()
    except (ValueError, FileNotFoundError):
        return None
    # Cross-ARM union (not just cross-event). Every applicable model arm is drawn
    # on one scrolling page, so a range computed from a single model's group would
    # be BORROWED by the other arm's figure -- a scale that looks authoritative and
    # is not. The range is therefore accumulated over every (group, event) pair.
    groups = [g for g in groups_for("peak_flood_depth", model_types or enabled) if g in tree.groups]
    if not groups:
        return None
    _ds_by_group = {g: tree[g].to_dataset() for g in groups}
    _pairs = [(_g, _e) for _g in groups for _e in analysis.df_sims.index]
    for _group, _ev in _pairs:
        ds_all = _ds_by_group[_group]
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
            if map_cfg is not None and map_cfg.depth_clip_quantile_upper is not None:
                vals = d_ev.values
                if hasattr(vals, "compute"):
                    vals = vals.compute()
                finite = vals[np.isfinite(vals)]
                if finite.size > 0:
                    all_wet_values.append(finite)
            else:
                d_max_obj = d_ev.max()
                v_max = float(
                    d_max_obj.compute() if hasattr(d_max_obj, "compute") else d_max_obj,
                )
                if np.isfinite(v_max):
                    g_max = max(g_max, v_max)
        except KeyError:
            continue
    if map_cfg is not None and map_cfg.depth_clip_quantile_upper is not None:
        if not all_wet_values:
            return None
        combined = np.concatenate(all_wet_values)
        g_max = float(np.nanquantile(combined, map_cfg.depth_clip_quantile_upper))
    if not np.isfinite(g_max) or g_max <= 0.01:
        return None
    return g_max


def _wall_threshold_m(sys_cfg, elevation_panel):
    """Return the DEM elevation at and above which a cell counts as a building or wall.

    The threshold is the lower of whichever DEM sentinel heights are set, minus the
    configured `wall_threshold_buffer_m`. This is the rule the system-overview map
    uses for its wall overlay, so a user who changes the buffer or sets only one
    sentinel moves both maps' masks together. Returns None only when neither
    sentinel height is set, which callers read as "mask nothing". Deliberately NOT
    read under `except AttributeError` with a hardcoded fallback: that fallback once
    hid a misspelled field name, so the user's buffer was silently replaced by 40 m.
    """
    sentinels = [h for h in (sys_cfg.dem_building_height, sys_cfg.dem_outside_watershed_height) if h is not None]
    if not sentinels:
        return None
    return min(sentinels) - float(elevation_panel.wall_threshold_buffer_m)


def _plotly_map_transforms(dry_threshold, canvas_size, *, use_datashader):
    """Return the provenance `transform` text for the Plotly depth and WSE panels.

    Built from the dry threshold the depth mask used and, when the rasters were
    pre-aggregated, the datashader canvas they were aggregated onto, so the manifest
    states the values the figure was drawn with rather than the defaults.
    """
    suffix = f"; datashader pre-rasterized ({canvas_size[0]}x{canvas_size[1]}, max)" if use_datashader else ""
    depth = f"masked to watershed and depth>{dry_threshold:g} m" + suffix
    wse = "depth, summed with DEM elevation" + suffix
    return depth, wse


def _shared_wse_range(analysis, target_crs, dem_da, map_cfg=None, model_types=None):
    """Return (vmin, vmax) for the WSE colorbar, computed once across every
    event_iloc so all per-event figures share a colorbar and a reader can
    compare water levels between events by eye. Walks every event's TRITON
    summary, masks depth > dry_threshold_m + watershed, builds WSE = depth +
    DEM, and accumulates the global quantile-clipped range across all wetted
    cells from all events. Returns None if no event exposes a usable summary,
    and the caller then falls back to the per-event range.

    The range is a quantile clip across the union of wetted cells, not
    min/max, because building-on-top dry-cell artifacts would otherwise
    dominate the cross-event colorbar scale; cells on a building or wall are also
    excluded outright (see `_wall_threshold_m`). Quantiles read from
    `map_cfg.wse_clip_quantile_lower` / `_upper` when `map_cfg` is supplied;
    falls back to (q01, q99) when `map_cfg` is None (preserves callability
    from sites that don't have the cfg in scope).
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
    # Building filter: exclude cells whose DEM lands on a wall/building so
    # building-rooftop WSE values do not pollute the cross-event quantile clip.
    # Threshold mirrors system_overview's wall-overlay rule.
    wall_th = _wall_threshold_m(analysis._system.cfg_system, analysis.cfg_analysis.report.system_map.elevation_panel)
    try:
        tree = analysis.process.open_datatree()
    except (ValueError, FileNotFoundError):
        return None
    # Cross-ARM union -- see the companion comment in _shared_depth_max. The WSE
    # range matters more than depth here, not less: WSE is an absolute elevation,
    # so two arms drawn on different WSE scales invite a reader to compare
    # water-surface elevations that are not on the same axis at all.
    groups = [g for g in groups_for("peak_flood_depth", model_types or enabled) if g in tree.groups]
    if not groups:
        return None
    _ds_by_group = {g: tree[g].to_dataset() for g in groups}
    all_wet_values: list[np.ndarray] = []
    _pairs = [(_g, _e) for _g in groups for _e in analysis.df_sims.index]
    for _group, _ev in _pairs:
        ds_all = _ds_by_group[_group]
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
    model_type: str | None = None,
    static_cfg: StaticPlotBaseConfig | None = None,
    **kwargs,
) -> Path:
    """Render the 3-panel depth + WSE + hydrology figure for one event_iloc.

    When ``static_cfg`` is provided (the publication-figure path,
    ``TRITONSWMM_analysis.static_plots()``, which passes one per-plot static
    config) the matplotlib branch is FORCED, whatever the report's
    ``static_backend``: publication figures always come from matplotlib, and the
    Plotly branch with its best-effort kaleido SVG serves the interactive report
    only. In that case the figure geometry, depth colormap / out-of-range colors, colorbar norm,
    base typography, and emit format are driven by the PeakFloodDepthStaticConfig
    rather than report_cfg. With ``static_cfg=None`` (the report path) the
    report's ``static_backend`` selects the branch. On the default Plotly branch,
    ``_render_plotly_branch`` emits interactive HTML: its colormaps come from
    ``report_cfg``, its depth colorbar is a continuous colorscale, and its size is
    fixed by the ``PER_SIM_FIG_W_PX`` / ``PER_SIM_FIG_H_PX`` constants. On the
    matplotlib branch the geometry, colormaps and typography come from
    ``report_cfg``, the depth colorbar is a linear Normalize, and the figure is
    emitted as PNG. ``event_iloc`` defaults to None only so the dispatcher can omit it
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
    from hhemt.report_renderers._model_arms import resolve_arm_group
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

    # Model-type dispatch via the single-source arm matrix (_model_arms) --
    # see the companion comment in per_sim_conduit_flow.render(). The retired
    # chain's `elif` was unreachable under a coupled arm, so a three-model
    # analysis silently dropped the standalone-TRITON depth field.
    enabled = analysis._get_enabled_model_types()
    triton_group = resolve_arm_group("peak_flood_depth", enabled, model_type=model_type)
    if triton_group is None:
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
    # Publication path: when static_cfg is provided, FORCE the matplotlib branch
    # regardless of report_cfg.interactive.static_backend (publication figures
    # never come from the Plotly branch) and apply the publication base
    # typography (font_family + a base size) so the publication font fields are
    # not silently inert. Two limits of what this does: only the axis-label entry
    # of static_cfg.font_sizes is applied, as the base size, so the other
    # per-element sizes have no effect yet; and these rcParams are set globally
    # rather than inside an rc_context, so they stay in effect for any later
    # figure drawn in the same process. With static_cfg None (the report path)
    # this block changes nothing: fonts are those `_apply_rcparams(report_cfg)`
    # set above, and static_backend alone selects the branch.
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
    # Consumer-side defense-in-depth: if a partial-completion render reaches this
    # output-owning entrypoint, soft-skip with a VISIBLE labeled placeholder
    # (which still writes output_path) rather than raising, which would leave
    # output_path unwritten and fail the Snakemake rule with an opaque
    # missing-output error. _emit_model_type_skip_placeholder is already
    # imported above for the swmm-only skip path.
    try:
        tree = analysis.process.open_datatree()
    except (ValueError, FileNotFoundError):
        return _emit_model_type_skip_placeholder(
            output_path,
            "peak_flood_depth: consolidated DataTree not available for this event",
            report_cfg.figure_defaults.savefig_dpi,
        )
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
    # FORCING-READ: render-hydrology
    weather_path = Path(analysis.cfg_analysis.weather_timeseries)
    # PATH-ONLY resolution: the free function never instantiates a
    # TRITONSWMM_scenario (whose constructor mkdirs processed/ swmm/ out_swmm/) and
    # creates no directory -- required because this render path runs under the
    # provenance audit. It is NOT read-free: on the no-CSV branch it opens
    # weather_path itself for the axis endpoints. That is audit-safe here ONLY because
    # weather_path is already in this renderer's source_paths for the
    # load_event_hydrology_data call below -- a caller that does not declare it would
    # trip the audit's actual-reads-subset-declared assertion.
    from hhemt.scenario import resolve_event_window

    hydro_data = load_event_hydrology_data(
        weather_path,
        analysis.cfg_analysis,
        weather_event_indexers,
        window=resolve_event_window(
            analysis.cfg_analysis,
            weather_event_indexers,
            cache=analysis.__dict__.setdefault("_event_window_cache", {}),
        ),
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
    # Bounds are sourced from the DEM (same source conduit_flow uses) so
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
        # Publication exact dimensions: the user owns the figure
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
    # vertical extent matches the map (not map+colorbar) and the three panels
    # read as one row of equal-height figures. Top row holds the rainfall+BC
    # stack; bottom row is intentionally empty (aligns with the colorbar slot on
    # the map cols).
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

    # ---- Depth panel: continuous linear color scale ---------------------
    # The report path uses a continuous linear Normalize, not discrete depth
    # bins. vmin comes from `map_cfg.depth_vmin`, and cells below it render in
    # the configured under-color via `cmap.set_under(...)` rather than in the
    # lowest band. vmax is computed once across every event_iloc so the depth
    # colorbar shares its range across all per-event figures.
    depth_ref = ProvenanceRef(
        source_path=triton_summary_rel,
        variable=str(wlevel_name) if wlevel_name is not None else "max_wlevel_m",
        attrs=wlevel_attrs,
        selection={"event_iloc": int(event_iloc)},
        transform="masked to watershed and depth>0",
    )
    # vmin/vmax: publication static_cfg.vmin/vmax override the report defaults
    # when set. Otherwise vmin is `map_cfg.depth_vmin`, and vmax is the cross-event
    # shared max, falling back to this event's own max, then to
    # `map_cfg.depth_vmax_fallback`.
    if static_cfg is not None and static_cfg.vmin is not None:
        depth_vmin = float(static_cfg.vmin)
    else:
        depth_vmin = map_cfg.depth_vmin
    if static_cfg is not None and static_cfg.vmax is not None:
        depth_vmax = float(static_cfg.vmax)
    else:
        shared_max = _shared_depth_max(analysis, target_crs, map_cfg=map_cfg)
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
    # colorbar norm: report path is a continuous linear Normalize. The
    # publication path activates BoundaryNorm/LogNorm/SymLogNorm by colorbar_norm.
    # These controls have NO map_cfg counterpart, so they are read under an
    # explicit static_cfg condition; a dual-source swap would render them
    # silently inert.
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
        note=f"peak flood depth raster ({_norm_label}, {event_page_reference(analysis, event_iloc)})",
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

    # ---- WSE panel: linear color scale ----------------------------------
    # The colorbar range is shared across every event in the analysis so
    # the user can compare WSE between event_iloc figures
    # by eye. Falls back to this event's own range only when the cross-event scan
    # returns no range: no event, this one included, has a usable summary, or the
    # quantile range is degenerate.
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
        note=f"water surface elevation = depth + DEM ({event_page_reference(analysis, event_iloc)})",
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
    # The middle panel shares y-axis with ax_depth (sharey=ax_depth above);
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

    # Explicit identical ticks AFTER both panels render
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

    # ---- Hydrology panel: delegated to the helper shared with conduit_flow ---
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
    `_render_plotly_branch` needs, returning them as a tuple. It is a separate
    function so a test can obtain the figure object before HTML serialization.
    """
    from hhemt.config.report import resolve_target_crs

    # Side-effect import: registers `triton_journal` Plotly template at import time.
    from hhemt.report_renderers import _plotly_theme  # noqa: F401
    from hhemt.report_renderers._figure_emission import (
        PER_SIM_FIG_H_PX,
        PER_SIM_FIG_W_PX,
        PER_SIM_HSPACE,
        PER_SIM_TOP_MARGIN_PX,
        apply_per_sim_map_axes,
        per_sim_bottom_margin_px,
        per_sim_colorbar,
        per_sim_plot_area_h_px,
    )
    from hhemt.report_renderers._hydrology_panel import (
        draw_event_hydrology_panel_plotly,
        load_event_hydrology_data,
    )
    from hhemt.report_renderers._provenance import ProvenanceRef

    cfg = report_cfg.per_sim.peak_flood_depth  # noqa: F841 — parity with mpl branch
    map_cfg = report_cfg.per_sim.map
    interactive_cfg = report_cfg.per_sim.interactive

    target_crs = resolve_target_crs(analysis, report_cfg)
    sys_paths = analysis._system.sys_paths
    triton_summary_path = analysis.analysis_paths.analysis_datatree_zarr

    # ---- Data prep ------------------------------------------------------
    # Not a copy of the matplotlib branch: here a cell is wet when depth >
    # dry_threshold_m (not > 0), and the maps use a padded square frame (not
    # the raw DEM bounds).
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
    # Split mask around configurable dry threshold. Cells inside the
    # watershed with depth at or below `dry_threshold_m` render via a
    # neutral-grey base fill so the watershed shape stays preattentive.
    dry_threshold = map_cfg.dry_threshold_m
    wet_mask = mask & (da > dry_threshold)
    da_masked = da.where(wet_mask)
    # Dry cells get no raster of their own: they show through as one
    # watershed-shaped polygon drawn beneath the heatmaps (built in the depth
    # panel below), whose size grows with its vertex count rather than with the
    # cell count. Deliberately NOT a per-cell dry-indicator Heatmap: that raster
    # was the dominant share of the HTML payload, measured at roughly 3 MB when
    # it was replaced (not re-measured here).

    try:
        wse_da = da + dem_da
    except Exception:
        wse_da = da + dem_da.interp_like(da, method="nearest")
    wse_masked = wse_da.where(wet_mask)

    weather_event_indexers = analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
    # FORCING-READ: render-hydrology
    weather_path = Path(analysis.cfg_analysis.weather_timeseries)
    # PATH-ONLY resolution: the free function never instantiates a
    # TRITONSWMM_scenario (whose constructor mkdirs processed/ swmm/ out_swmm/) and
    # creates no directory -- required because this render path runs under the
    # provenance audit. It is NOT read-free: on the no-CSV branch it opens
    # weather_path itself for the axis endpoints. That is audit-safe here ONLY because
    # weather_path is already in this renderer's source_paths for the
    # load_event_hydrology_data call below -- a caller that does not declare it would
    # trip the audit's actual-reads-subset-declared assertion.
    from hhemt.scenario import resolve_event_window

    hydro_data = load_event_hydrology_data(
        weather_path,
        analysis.cfg_analysis,
        weather_event_indexers,
        window=resolve_event_window(
            analysis.cfg_analysis,
            weather_event_indexers,
            cache=analysis.__dict__.setdefault("_event_window_cache", {}),
        ),
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
    shared_max = _shared_depth_max(analysis, target_crs, map_cfg=map_cfg)
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
        # Clip WSE colorbar to quantiles of WETTED + on-real-terrain
        # cells.  Suppresses building-on-top artifacts that would otherwise
        # dominate the scale and collapse usable WSE range to <5% of the
        # colorbar. Building/wall cells are those at or above
        # `_wall_threshold_m`, the same rule system_overview's wall overlay uses.
        wall_th = _wall_threshold_m(analysis._system.cfg_system, report_cfg.system_map.elevation_panel)
        if wall_th is not None:
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

    # Cross-figure bounds parity with system_overview.  Load the same
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
    # Pre-aggregate the depth and WSE rasters onto a fixed canvas when the depth
    # cell count exceeds the configured threshold, so a large grid is not inlined
    # cell by cell into the HTML. Only the single peak frame exists to aggregate:
    # this figure has no time animation.
    cell_count = int(
        da_masked.notnull().sum().compute()
        if hasattr(da_masked.notnull().sum(), "compute")
        else da_masked.notnull().sum()
    )
    use_datashader = cell_count > interactive_cfg.datashader_threshold_cells
    _depth_transform, _wse_transform = _plotly_map_transforms(
        dry_threshold, interactive_cfg.datashader_canvas_size, use_datashader=use_datashader
    )
    if use_datashader:
        import datashader as ds_lib
        import datashader.reductions as ds_reductions

        canvas = ds_lib.Canvas(
            plot_width=interactive_cfg.datashader_canvas_size[0],
            plot_height=interactive_cfg.datashader_canvas_size[1],
        )
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
        horizontal_spacing=PER_SIM_HSPACE,
        vertical_spacing=0.06,
        subplot_titles=("Peak flood depth", "Water surface elevation", "Flood Drivers"),
    )
    # No figure title: the report's sidebar and result-row labels already name the
    # figure. showlegend=True so the dry-cell legend swatch surfaces.
    fig.update_layout(
        template="triton_journal",
        showlegend=True,
        legend=dict(
            orientation="h",
            # BELOW the colorbars, not above them. The legend goes left of colorbar 1 only
            # if a legend item fits there, and otherwise below. The space there measured
            # 9.9 px against a ~162 px legend item when this placement was chosen (not
            # re-measured here), so it goes below. Sitting at y=-0.10 put this strip
            # directly in the band the colorbar labels occupy.
            yanchor="top",
            y=-0.34,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        # Width/height DECLARED: the layout previously declared
        # neither while `write_image` hardcoded 1400x600, so the browser render and the
        # SVG export were laid out differently and every paper coordinate was correct
        # for at most one of them. Bottom margin is DERIVED from the colorbar band, not
        # the retired literal b=130.
        width=PER_SIM_FIG_W_PX,
        height=PER_SIM_FIG_H_PX,
        margin=dict(
            l=10,
            r=10,
            t=PER_SIM_TOP_MARGIN_PX,
            b=per_sim_bottom_margin_px(),
        ),
    )

    # Map-panel x geometry + centred colorbar paper-x, shared with per_sim_conduit_flow.
    # Called here (not in the axes block below) because the colorbar dicts on the two
    # Heatmaps consume the returned x values.
    _cb_x = apply_per_sim_map_axes(
        fig,
        bounds=bounds,
        crs_epsg=target_crs.to_epsg(),
    )

    # The depth colorscale is continuous: the live render path passes
    # `map_cfg.depth_cmap` directly. The discrete BoundaryNorm-style alternative
    # is available as `_build_discrete_depth_colorscale` and is not called here.

    # ---- Depth panel ----------------------------------------------------
    depth_ref = ProvenanceRef(
        source_path=triton_summary_rel,
        variable=str(wlevel_name) if wlevel_name is not None else "max_wlevel_m",
        attrs=wlevel_attrs,
        selection={"event_iloc": int(event_iloc)},
        transform=_depth_transform,
    )
    # Build the watershed polygon rings once; both map panels reuse them as the
    # dry-cell base fill. Each ring's coordinates end in a None separator so a
    # multi-polygon watershed keeps its rings apart; the path builder below
    # splits on that separator and closes each ring as its own region.
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
    # The dry-cell base fill MUST render BELOW the depth/WSE Heatmaps. Plotly
    # draws SVG vector traces (a Scatter fill) ABOVE the heatmap raster
    # regardless of add_trace order, so a `go.Scatter(fill="toself")` fill
    # occludes the heatmap and the map renders all-grey.
    # A layout shape with layer="below" is the only mechanism that places fill
    # beneath a go.Heatmap; build the watershed multi-polygon as one SVG path
    # (M x,y L x,y ... Z per ring) reusing the same None-separated coords.
    _ws_path_segments: list[str] = []
    _seg_x: list[float] = []
    _seg_y: list[float] = []
    for _px, _py in zip(ws_fill_x, ws_fill_y, strict=False):
        if _px is None or _py is None:
            if _seg_x:
                _seg = (
                    f"M {_seg_x[0]},{_seg_y[0]} "
                    + " ".join(f"L {_x},{_y}" for _x, _y in zip(_seg_x[1:], _seg_y[1:], strict=False))
                    + " Z"
                )
                _ws_path_segments.append(_seg)
                _seg_x, _seg_y = [], []
            continue
        _seg_x.append(_px)
        _seg_y.append(_py)
    if _seg_x:
        _seg = (
            f"M {_seg_x[0]},{_seg_y[0]} "
            + " ".join(f"L {_x},{_y}" for _x, _y in zip(_seg_x[1:], _seg_y[1:], strict=False))
            + " Z"
        )
        _ws_path_segments.append(_seg)
    ws_fill_path = " ".join(_ws_path_segments)
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
    with prov.artist(
        axes_id="ax_depth_plotly",
        kind="patch",
        note="dry-cell watershed fill polygon (depth panel)",
    ) as a:
        a.add_channel("x", dry_ref)
        a.add_channel("y", dry_ref)
        a.add_channel("color", dry_ref, cmap=map_cfg.dry_fill_color)
        # A layout shape with layer="below" so the grey watershed base
        # renders BENEATH the depth Heatmap and shows through only dry (NaN)
        # cells. A go.Scatter fill would draw ABOVE the raster and occlude it.
        fig.add_shape(
            type="path",
            path=ws_fill_path,
            fillcolor=map_cfg.dry_fill_color,
            line=dict(width=0),
            layer="below",
            row=1,
            col=1,
        )
    with prov.artist(
        axes_id="ax_depth_plotly",
        kind="image",
        note=f"peak flood depth raster ({event_page_reference(analysis, event_iloc)})"
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
        fig.add_trace(
            go.Heatmap(
                z=depth_z,
                x=depth_x,
                y=depth_y,
                colorscale=map_cfg.depth_cmap,
                zmin=depth_vmin,
                zmax=depth_vmax,
                colorbar=per_sim_colorbar(
                    units.depth_label(analysis._system.cfg_system.crs.vertical_epsg),
                    _cb_x[1],
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
        transform=_wse_transform,
    )
    wse_ref_dem = ProvenanceRef(
        source_path=dem_rel,
        variable="dem_elev_m",
        attrs=dem_attrs,
        transform="reprojected to target_crs",
    )
    with prov.artist(
        axes_id="ax_wse_plotly",
        kind="patch",
        note="dry-cell watershed fill polygon (WSE panel)",
    ) as a:
        a.add_channel("x", dry_ref)
        a.add_channel("y", dry_ref)
        a.add_channel("color", dry_ref, cmap=map_cfg.dry_fill_color)
        # Same below-layer watershed shape as the depth panel — see its note.
        fig.add_shape(
            type="path",
            path=ws_fill_path,
            fillcolor=map_cfg.dry_fill_color,
            line=dict(width=0),
            layer="below",
            row=1,
            col=2,
        )
    with prov.artist(
        axes_id="ax_wse_plotly",
        kind="image",
        note=f"water surface elevation = depth + DEM ({event_page_reference(analysis, event_iloc)})"
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
        fig.add_trace(
            go.Heatmap(
                z=wse_z,
                x=wse_x,
                y=wse_y,
                colorscale=map_cfg.wse_cmap,
                zmin=wse_min,
                zmax=wse_max,
                colorbar=per_sim_colorbar(
                    units.wse_label(analysis._system.cfg_system.crs.vertical_epsg),
                    _cb_x[2],
                ),
                hovertemplate="WSE: %{z:.3f} m<br>x: %{x}<br>y: %{y}<extra></extra>",
                showlegend=False,
                name="wse",
            ),
            row=1,
            col=2,
        )

    # Dummy-marker legend entry naming the dry-cell fill color +
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
        # Legend-togglability invariant (tests/test_synth_legend_toggle_audit.py): a
        # showlegend=True trace that plots no data must share a legendgroup with a peer
        # so the legend click toggles a grouped element. The grey dry base-fill is an
        # add_shape(layer="below") (occlusion fix, 04faa40) and emits no legend entry,
        # so pair the visible swatch with an invisible peer trace in legendgroup="dry".
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=map_cfg.dry_fill_color, opacity=0),
                legendgroup="dry",
                showlegend=False,
                hoverinfo="skip",
                name="dry_fill_peer",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color=map_cfg.dry_fill_color, symbol="square", size=12, line=dict(color="darkgrey", width=0.5)
                ),
                name=f"≤ {map_cfg.dry_threshold_m:g} m (dry)",
                legendgroup="dry",
                showlegend=True,
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
                # The overlay was once drawn with no legend key, so the boundary read as an
                # unexplained line. Its key is the marker proxy below, labeled to match
                # `eda/_config_diff.py`'s swatch verbatim so one geometry has one name
                # across figure families. The trace is emitted once per PANEL (depth here,
                # WSE below); `legendgroup` is what couples them into one toggle.
                #
                # BOTH boundary traces are showlegend=False and the entry is
                # carried by the marker proxy below. A `mode="lines"` trace renders a LINE
                # glyph in the legend, and plotly exposes no per-trace override for it — so
                # the only way to key this geometry with a RECTANGLE, matching the swatch
                # `figure_panels.watershed_swatch` draws for the config-diff family, is a
                # separate markers-mode trace.
                showlegend=False,
                legendgroup="watershed_extent",
                name="watershed extent (bbox)",
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
                # Same legendgroup as the depth panel's trace, showlegend=False so
                # the shared entry appears exactly ONCE. Toggling it hides both panels'
                # boundaries together, which is the correct coupling — it is one geometry.
                showlegend=False,
                legendgroup="watershed_extent",
                name="watershed extent (bbox)",
            ),
            row=1,
            col=2,
        )
    with prov.artist(
        axes_id="ax_depth_plotly",
        kind="legend",
        note="watershed legend key (dummy Scatter, no plotted points)",
    ) as a:
        a.add_channel("x", watershed_ref)
        # The legend GLYPH, carried by a markers-mode proxy that plots no data.
        # `square-open` is the unfilled square, which is what `watershed_swatch` draws as a
        # rect with `fillcolor="rgba(0,0,0,0)"` — so both figure families now key one
        # geometry with one shape as well as one name.
        #
        # The legend-togglability invariant above (tests/test_synth_legend_toggle_audit.py)
        # is satisfied WITHOUT an invisible peer: this trace shares `watershed_extent` with
        # the two boundary traces, which DO plot data. That is the difference from the dry
        # swatch, whose base fill is an add_shape and therefore had no trace peer to group
        # with. Toggling this entry hides the boundary on both maps.
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=watershed_legend_marker(
                    color=map_cfg.watershed_overlay_color,
                    line_width=map_cfg.watershed_overlay_width,
                ),
                hoverinfo="skip",
                showlegend=True,
                # Glyph, group and label all read `hhemt.figure_panels`, so this key
                # matches the other figure families. Two constraints hold: it stays a
                # markers-mode proxy (a lines-mode trace keys a line, not the rectangle,
                # as noted above), and it is added once, OUTSIDE any per-column loop -- a
                # per-column add would put one legend entry per panel.
                legendgroup=WATERSHED_LEGEND_GROUP,
                name=WATERSHED_LEGEND_LABEL,
            ),
            row=1,
            col=1,
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
    # Convert times to hours for x-axis display + propagate the units
    # change into the provenance channel so the manifest tells the truth.
    times_hr = np.asarray(times_min, dtype=float) / units.MINUTES_PER_HOUR
    bc_ref = ProvenanceRef(
        source_path=weather_rel,
        variable=bc_var if bc_var is not None else "",
        attrs=bc_attrs,
        selection={"event_iloc": int(event_iloc)},
    )
    # Both Flood Drivers traces and both wrapped y titles come from ONE shared helper,
    # sibling to the matplotlib `draw_event_hydrology_panel` it diverged from. The
    # provenance channels stay here because they are per-family; only the trace and
    # axis construction moved.
    with prov.artist(
        axes_id="ax_rain_plotly",
        kind="line2d",
        note="rainfall time series (event hydrology — top sub-panel)",
    ) as a:
        a.add_channel("x", rain_ref, units=units.TIME_AXIS_PROVENANCE_UNITS_HOURS)
        a.add_channel("y", rain_ref, units=rain_units)
    with prov.artist(
        axes_id="ax_bc_plotly",
        kind="line2d",
        note="boundary condition water level (event hydrology — bottom sub-panel)",
    ) as a:
        a.add_channel("x", bc_ref, units=units.TIME_AXIS_PROVENANCE_UNITS_HOURS)
        a.add_channel("y", bc_ref, units=bc_units)
    _col3_panel_h_px = per_sim_plot_area_h_px() * 0.47
    draw_event_hydrology_panel_plotly(
        fig,
        times_hr=times_hr,
        rainfall=rainfall,
        bc_water_level=bc_water_level,
        panel_cfg=panel_cfg,
        rainfall_units=analysis.cfg_analysis.rainfall_units,
        storm_tide_units=analysis.cfg_analysis.storm_tide_units,
        axis_extent_px=_col3_panel_h_px,
        time_axis_title=units.TIME_AXIS_FROM_EVENT_START_HOURS,
    )

    # ---- Axes setup -----------------------------------------------------
    from hhemt.config.report import resolve_target_crs

    target_crs_resolved = resolve_target_crs(analysis, report_cfg)
    crs_for_labels = target_crs_resolved.to_epsg()
    # Link col-2 map axes to col-1 via `matches=` so interactive
    # pan/zoom on either panel synchronises the other. Plotly's
    # `shared_xaxes` in `make_subplots` does not always emit `matches`.
    # Map x-axis range + cleared x titles are already applied by
    # `apply_per_sim_map_axes` above (shared with per_sim_conduit_flow). Only the
    # y-axes and the col-2 `matches=` links remain family-local here.
    for col in (1, 2):
        y_kwargs = dict(
            range=[bounds[1], bounds[3]],
            scaleanchor=f"x{col}",
            scaleratio=1.0,
            title_text=units.northing_axis_label(crs_for_labels) if col == 1 else None,
            row=1,
            col=col,
        )
        if col == 2:
            fig.update_xaxes(matches="x", row=1, col=col)
            y_kwargs["matches"] = "y"
        fig.update_yaxes(**y_kwargs)

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
    """Render the interactive Plotly version of the 3-panel figure — depth raster +
    WSE raster + event hydrology (rainfall bars + BC water level line) — and emit it
    as HTML, with a best-effort SVG export beside it. Shows the same information as
    the matplotlib branch, for the peak frame only (no animation). Datashader
    pre-rasterization fires when the depth-frame cell count exceeds
    `report_cfg.per_sim.interactive.datashader_threshold_cells`. If the consolidated
    DataTree is unavailable, writes a labeled placeholder instead of a figure.
    """
    from hhemt.report_renderers._figure_emission import (
        PER_SIM_FIG_H_PX,
        PER_SIM_FIG_W_PX,
        emit_plot_with_sources,
    )
    from hhemt.report_renderers.per_sim_conduit_flow import (
        _emit_model_type_skip_placeholder,
    )

    # Consumer-side defense-in-depth: the builder opens the consolidated
    # DataTree; if a partial-completion render reaches this output-owning plotly
    # entrypoint, soft-skip with a VISIBLE labeled placeholder (which still
    # writes output_path) rather than raising, which would leave output_path
    # unwritten and fail the Snakemake rule with an opaque missing-output
    # error. The guard wraps the builder CALL (not the open inside
    # it) because the builder returns a 13-tuple this branch unpacks — a
    # placeholder Path returned from inside the builder would break the unpack.
    try:
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
    except (ValueError, FileNotFoundError):
        return _emit_model_type_skip_placeholder(
            output_path,
            "peak_flood_depth: consolidated DataTree not available for this event",
            report_cfg.figure_defaults.savefig_dpi,
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

    # The builder opened both SWMM .inp files (swmmio models for the shared map
    # bounds), so both are declared as sources: the figure's manifest must name
    # every data file the render read. Only the resolved paths are needed in the
    # caller scope, so re-calling the pure path resolver (no model build) is cheap.
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
            width=PER_SIM_FIG_W_PX,
            height=PER_SIM_FIG_H_PX,
            scale=1,
        )
    except Exception as exc:  # noqa: BLE001 — Kaleido failure is non-fatal
        import logging

        logging.getLogger(__name__).info(
            "Kaleido SVG export skipped (HTML emit unaffected) for %s: %s",
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
        preview_figure=(fig if report_cfg.interactive.html_preview_rasterization else None),
    )
