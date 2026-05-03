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
import rioxarray as rxr
import xarray as xr
from matplotlib.colors import Normalize

from TRITON_SWMM_toolkit import utils

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


# Discrete boundaries (m) for the depth colorbar, mirroring the reference
# image at scratch `# Collaborative Figure Design > Flood depth maps`.
_DEPTH_BOUNDARIES = (0.01, 0.05, 0.10, 0.50, 1.00)
_DEPTH_CMAP = "YlGnBu"
# Perceptually-distinct elevation-flavored colormap for the WSE panel
# (dark-navy → yellow), chosen to read as elevation while staying visually
# distinct from the YlGnBu depth ramp.
_WSE_CMAP = "plasma"
_RAIN_COLOR = "#9ecae1"  # light blue, matches the Event hydrology reference
_BC_LINE_COLOR = "black"


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
    for _ev in analysis.df_sims.index:
        try:
            _proc = analysis._retrieve_sim_run_processing_object(int(_ev))
            if "tritonswmm" in enabled:
                _path = _proc.scen_paths.output_tritonswmm_triton_summary
            else:
                _path = _proc.scen_paths.output_triton_only_summary
            with _proc._open(_path) as ds:
                da_ev = ds["max_wlevel_m"].sel(event_iloc=int(_ev))
                if da_ev.rio.crs is not None and da_ev.rio.crs != target_crs:
                    da_ev = da_ev.rio.reproject(target_crs)
                if (
                    da_ev.rio.crs is not None
                    and watershed_gdf.crs is not None
                    and watershed_gdf.crs != da_ev.rio.crs
                ):
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
        except (FileNotFoundError, KeyError):
            continue
    if not np.isfinite(g_max) or g_max <= 0.01:
        return None
    return g_max


def _shared_wse_range(analysis, target_crs, dem_da):
    """Return (vmin, vmax) for the WSE colorbar, computed once across every
    event_iloc so all per-event figures share a colorbar (iter-15 user
    request). Walks every event's TRITON summary, masks depth > 0 + watershed,
    builds WSE = depth + DEM, and accumulates the global min/max. Falls back
    to per-event range if no events expose a usable summary.
    """
    enabled = analysis._get_enabled_model_types()
    if "tritonswmm" not in enabled and "triton" not in enabled:
        return None
    watershed_shp = analysis._system.cfg_system.watershed_gis_polygon
    watershed_gdf = gpd.read_file(watershed_shp)
    g_min, g_max = float("inf"), float("-inf")
    for _ev in analysis.df_sims.index:
        try:
            _proc = analysis._retrieve_sim_run_processing_object(int(_ev))
            if "tritonswmm" in enabled:
                _path = _proc.scen_paths.output_tritonswmm_triton_summary
            else:
                _path = _proc.scen_paths.output_triton_only_summary
            with _proc._open(_path) as ds:
                da_ev = ds["max_wlevel_m"].sel(event_iloc=int(_ev))
                if da_ev.rio.crs is not None and da_ev.rio.crs != target_crs:
                    da_ev = da_ev.rio.reproject(target_crs)
                if (
                    da_ev.rio.crs is not None
                    and watershed_gdf.crs is not None
                    and watershed_gdf.crs != da_ev.rio.crs
                ):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_path = Path(tmp_dir) / "watershed_reprojected.geojson"
                        watershed_gdf.to_crs(da_ev.rio.crs).to_file(tmp_path, driver="GeoJSON")
                        m = utils.create_mask_from_shapefile(da_ev, tmp_path)
                else:
                    m = utils.create_mask_from_shapefile(da_ev, watershed_shp)
                wse_ev = (da_ev + dem_da).where(m & (da_ev > 0))
                wse_min_obj = wse_ev.min()
                wse_max_obj = wse_ev.max()
                v_min = float(
                    wse_min_obj.compute() if hasattr(wse_min_obj, "compute") else wse_min_obj,
                )
                v_max = float(
                    wse_max_obj.compute() if hasattr(wse_max_obj, "compute") else wse_max_obj,
                )
                if np.isfinite(v_min):
                    g_min = min(g_min, v_min)
                if np.isfinite(v_max):
                    g_max = max(g_max, v_max)
        except (FileNotFoundError, KeyError):
            continue
    if not np.isfinite(g_min) or not np.isfinite(g_max) or g_max <= g_min:
        return None
    return (g_min, g_max)


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
    *,
    event_iloc: int,
) -> Path:
    """Render the 3-panel depth + WSE + hydrology figure for one event_iloc."""
    from TRITON_SWMM_toolkit.config.report import resolve_target_crs
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        add_panel_label,
        emit_plot_with_sources,
        per_sim_map_ticks,
    )
    from TRITON_SWMM_toolkit.report_renderers._provenance import (
        ProvenanceLog,
        ProvenanceRef,
    )
    from TRITON_SWMM_toolkit.report_renderers.per_sim_conduit_flow import (
        _emit_model_type_skip_placeholder,
    )
    from TRITON_SWMM_toolkit.report_renderers.system_overview import _apply_rcparams

    _apply_rcparams(report_cfg)
    cfg = report_cfg.per_sim.peak_flood_depth
    prov = ProvenanceLog()

    proc = analysis._retrieve_sim_run_processing_object(event_iloc)

    enabled = analysis._get_enabled_model_types()
    if "tritonswmm" in enabled:
        triton_summary_path = proc.scen_paths.output_tritonswmm_triton_summary
    elif "triton" in enabled:
        triton_summary_path = proc.scen_paths.output_triton_only_summary
    else:
        return _emit_model_type_skip_placeholder(
            output_path,
            "peak_flood_depth not applicable for swmm-only analyses",
            report_cfg.figure_defaults.savefig_dpi,
        )

    target_crs = resolve_target_crs(analysis, report_cfg)
    sys_paths = analysis._system.sys_paths

    # ---- Depth raster from TRITON summary -------------------------------
    with proc._open(triton_summary_path) as ds:
        if ds.sizes.get("event_iloc") != 1:
            raise AssertionError(
                f"per-scenario triton summary expected event_iloc=1, got "
                f"{ds.sizes.get('event_iloc')}"
            )
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
    if (
        da.rio.crs is not None
        and watershed_gdf.crs is not None
        and watershed_gdf.crs != da.rio.crs
    ):
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
    from TRITON_SWMM_toolkit.report_renderers._hydrology_panel import (
        draw_event_hydrology_panel,
        load_event_hydrology_data,
    )
    weather_path = proc.scen_paths.weather_timeseries
    hydro_data = load_event_hydrology_data(weather_path)
    times_min = hydro_data["times_min"]
    rainfall = hydro_data["rainfall"]
    bc_water_level = hydro_data["bc_water_level"]

    # ---- Path relpaths --------------------------------------------------
    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    triton_summary_rel = os.path.relpath(
        str(Path(triton_summary_path).resolve()), analysis_root,
    )
    watershed_rel = os.path.relpath(
        str(Path(watershed_shp).resolve()), analysis_root,
    )
    dem_rel = os.path.relpath(
        str(Path(sys_paths.dem_processed).resolve()), analysis_root,
    )
    weather_rel = os.path.relpath(
        str(Path(weather_path).resolve()), analysis_root,
    )

    # ---- Figure layout: 1×3 columns, each column with a sub-gridspec ----
    # Subiteration 9.5 — sourced from DEM (same source conduit_flow uses) so
    # both per-sim renderers see IDENTICAL bounds + map_aspect, and explicit
    # set_xlim/set_ylim below produce IDENTICAL tick ranges between toggles.
    bounds = dem_da.rio.bounds() if dem_da.rio.crs is not None else (
        float(dem_da.x.min()), float(dem_da.y.min()),
        float(dem_da.x.max()), float(dem_da.y.max()),
    )
    map_aspect = (bounds[2] - bounds[0]) / max(bounds[3] - bounds[1], 1e-9)
    h = float(cfg.figsize_inches[1]) if hasattr(cfg, "figsize_inches") else 6.0
    # iter-5 layout fixes: tighter inter-panel spacing (was 1.05 + 1.4
    # padding, now 1.02 + 1.0); explicit `wspace=0.10` between outer columns
    # so the maps + hydro panel sit closer together. Also `layout=None` so we
    # control gridspec margins directly via `fig.subplots_adjust`.
    fig_width = h * (2 * map_aspect * 1.02 + 1.0)
    fig = plt.figure(figsize=(fig_width, h), layout="constrained")
    outer = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.10)
    # Map columns: outer 2-row split (map + thin colorbar slot).
    # Inner 1×3 split for the colorbar puts narrow margins on either side so
    # the colorbar bar itself doesn't span the full panel width (iter-4
    # feedback: "colorbars are both too wide … should not exceed the width of
    # the figures").
    _MAP_TO_CBAR_HEIGHT_RATIO = 28  # iter-4: was 18; bumps cbar thickness down
    gs_depth = outer[0, 0].subgridspec(2, 1, height_ratios=[_MAP_TO_CBAR_HEIGHT_RATIO, 1])
    gs_wse = outer[0, 1].subgridspec(2, 1, height_ratios=[_MAP_TO_CBAR_HEIGHT_RATIO, 1])
    gs_depth_cbar = gs_depth[1, 0].subgridspec(1, 3, width_ratios=[1, 5, 1])
    gs_wse_cbar = gs_wse[1, 0].subgridspec(1, 3, width_ratios=[1, 5, 1])
    # Hydro column: outer 2-row split mirrors map columns so the hydro panel's
    # vertical extent matches the map (not map+colorbar) — iter-4 feedback:
    # "boundary height of the event hydrology figure should match the height
    # of the flood figures". Top row holds the rainfall+BC stack; bottom row
    # is intentionally empty (aligns with the colorbar slot on the map cols).
    gs_hydro_outer = outer[0, 2].subgridspec(
        2, 1, height_ratios=[_MAP_TO_CBAR_HEIGHT_RATIO, 1],
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
    depth_vmin = 0.01
    shared_max = _shared_depth_max(analysis, target_crs)
    if shared_max is not None:
        depth_vmax = float(shared_max)
    else:
        d_max_obj = da_masked.max()
        d_max_local = float(
            d_max_obj.compute() if hasattr(d_max_obj, "compute") else d_max_obj,
        )
        depth_vmax = d_max_local if (np.isfinite(d_max_local) and d_max_local > depth_vmin) else 1.0
    depth_cmap = plt.get_cmap(_DEPTH_CMAP).copy()
    depth_cmap.set_under("white")
    depth_norm = Normalize(vmin=depth_vmin, vmax=depth_vmax)
    with prov.artist(
        axes_id="ax_depth", kind="image",
        note=f"peak flood depth raster (continuous linear, event {event_iloc})",
    ) as a:
        a.add_channel("z", depth_ref)
        a.add_channel(
            "color", depth_ref,
            cmap=_DEPTH_CMAP,
            vmin=depth_vmin, vmax=depth_vmax,
            norm="Normalize",
            extend="min",
            under_color="white",
        )
        depth_img = da_masked.plot(  # noqa: F841
            ax=ax_depth, x="x", y="y",
            cmap=depth_cmap, norm=depth_norm,
            add_colorbar=False,
        )
    cbar_d = fig.colorbar(
        ax_depth.collections[0] if ax_depth.collections else depth_img,
        cax=cax_depth, orientation="horizontal",
        extend="min",
    )
    cbar_d.set_label("Depth (m)")
    ax_depth.set_aspect("equal")
    # Subiteration 9.5 — explicit lims from DEM bounds (matches conduit_flow's
    # explicit set_xlim/set_ylim, so toggling between figures shows IDENTICAL
    # ranges + ticks regardless of the underlying data extent).
    ax_depth.set_xlim(bounds[0], bounds[2])
    ax_depth.set_ylim(bounds[1], bounds[3])
    ax_depth.set_title("Peak flood depth")
    # iter-5 feedback: restore numeric x/y tick labels on flood maps. Use
    # small font so the labels don't dominate the panel.
    ax_depth.tick_params(axis="both", labelsize=7)
    ax_depth.set_xlabel("Easting (m)", fontsize=8)
    ax_depth.set_ylabel("Northing (m)", fontsize=8)
    watershed_ref = ProvenanceRef(
        source_path=watershed_rel,
        variable="watershed_polygon",
        attrs={},
    )
    with prov.artist(
        axes_id="ax_depth", kind="patch",
        note="watershed boundary overlay",
    ) as a:
        a.add_channel("x", watershed_ref)
        a.add_channel("y", watershed_ref)
        if watershed_gdf.crs is not None:
            watershed_gdf.to_crs(target_crs).boundary.plot(
                ax=ax_depth, color="black", linewidth=1.2,
            )
        else:
            watershed_gdf.boundary.plot(
                ax=ax_depth, color="black", linewidth=1.2,
            )

    # ---- WSE panel: cividis linear --------------------------------------
    # Iter-15 (2026-04-29): the colorbar range is shared across every event
    # in the analysis so the user can compare WSE between event_iloc figures
    # by eye. Falls back to per-event range only if the cross-event scan
    # failed (no other event has a usable summary).
    shared = _shared_wse_range(analysis, target_crs, dem_da)
    if shared is not None:
        wse_min, wse_max = shared
    else:
        wse_min_obj = wse_masked.min()
        wse_max_obj = wse_masked.max()
        wse_min = float(wse_min_obj.compute() if hasattr(wse_min_obj, "compute") else wse_min_obj)
        wse_max = float(wse_max_obj.compute() if hasattr(wse_max_obj, "compute") else wse_max_obj)
        if not np.isfinite(wse_min) or not np.isfinite(wse_max) or wse_max <= wse_min:
            wse_min, wse_max = 0.0, 1.0

    wse_ref_depth = ProvenanceRef(
        source_path=triton_summary_rel,
        variable=str(wlevel_name) if wlevel_name is not None else "max_wlevel_m",
        attrs=wlevel_attrs,
        selection={"event_iloc": int(event_iloc)},
        transform="depth, summed with DEM elevation",
    )
    wse_ref_dem = ProvenanceRef(
        source_path=dem_rel, variable="dem_elev_m",
        attrs=dem_attrs, transform="reprojected to target_crs",
    )
    with prov.artist(
        axes_id="ax_wse", kind="image",
        note=f"water surface elevation = depth + DEM (event {event_iloc})",
    ) as a:
        a.add_channel("z", wse_ref_depth)
        a.add_channel("z", wse_ref_dem)
        a.add_channel(
            "color", wse_ref_depth,
            cmap=_WSE_CMAP, vmin=wse_min, vmax=wse_max,
        )
        wse_img = wse_masked.plot(  # noqa: F841
            ax=ax_wse, x="x", y="y",
            cmap=_WSE_CMAP, vmin=wse_min, vmax=wse_max,
            add_colorbar=False,
        )
    cbar_w = fig.colorbar(
        ax_wse.collections[0] if ax_wse.collections else wse_img,
        cax=cax_wse, orientation="horizontal",
    )
    cbar_w.set_label("WSE (m)")
    ax_wse.set_aspect("equal")
    ax_wse.set_title("Water surface elevation")
    # C8 — middle panel shares y-axis with ax_depth (sharey=ax_depth above);
    # hide redundant y-tick labels and drop the ylabel so the gap between
    # the depth and WSE panels collapses to the bare wspace allocation.
    ax_wse.tick_params(axis="both", labelsize=7)
    ax_wse.tick_params(axis="y", labelleft=False)
    ax_wse.set_xlabel("Easting (m)", fontsize=8)
    ax_wse.set_ylabel("")
    with prov.artist(
        axes_id="ax_wse", kind="patch",
        note="watershed boundary overlay",
    ) as a:
        a.add_channel("x", watershed_ref)
        a.add_channel("y", watershed_ref)
        if watershed_gdf.crs is not None:
            watershed_gdf.to_crs(target_crs).boundary.plot(
                ax=ax_wse, color="black", linewidth=1.2,
            )
        else:
            watershed_gdf.boundary.plot(
                ax=ax_wse, color="black", linewidth=1.2,
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
        ax_rain, ax_bc,
        hydro_data=hydro_data,
        weather_rel_path=weather_rel,
        event_iloc=event_iloc,
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
    valid_cell_count = (
        cell_count_obj.compute() if hasattr(cell_count_obj, "compute") else cell_count_obj
    )
    return emit_plot_with_sources(
        fig, output_path, source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        dpi=report_cfg.figure_defaults.savefig_dpi,
        manifest_data={
            "event_iloc": int(event_iloc),
            "depth_m_max": float(wlevel_m_max),
            "valid_cell_count": int(valid_cell_count),
            "wse_m_range": [wse_min, wse_max],
            "rainfall_max_mm_per_hr": float(np.nanmax(rainfall)),
            "bc_water_level_range_m": [bc_min, bc_max],
            "depth_boundaries_m": list(_DEPTH_BOUNDARIES),
        },
        provenance=prov,
    )


