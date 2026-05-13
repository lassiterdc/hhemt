"""Per-sim renderer: two-panel SWMM conduit flow figure (max-over-full-flow + peak-flow).

Dispatches per `_get_enabled_model_types()` so SWMM-only fixtures pick the
SWMM-only link summary and TRITON-only fixtures emit a model-type-skip
placeholder figure (R6 / Phase 3).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import swmmio
from plotly.subplots import make_subplots

from TRITON_SWMM_toolkit import units

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
    *,
    event_iloc: int,
) -> Path:
    """Render the two-panel conduit flow figure for one event_iloc."""
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        add_panel_label,
        emit_plot_with_sources,
        per_sim_map_ticks,
    )
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceLog
    from TRITON_SWMM_toolkit.report_renderers.system_overview import _apply_rcparams

    _apply_rcparams(report_cfg)
    cfg = report_cfg.per_sim.conduit_flow
    prov = ProvenanceLog()

    static_backend = getattr(
        getattr(report_cfg, "interactive", None),
        "static_backend",
        "plotly",
    )
    if static_backend == "plotly":
        return _render_plotly_branch(
            analysis, report_cfg, output_path,
            event_iloc=event_iloc, prov=prov,
        )

    proc = analysis._retrieve_sim_run_processing_object(event_iloc)

    # Model-type dispatch (Gotcha 5 from the master plan).
    enabled = analysis._get_enabled_model_types()
    if "tritonswmm" in enabled:
        link_group = "/tritonswmm/swmm_link"
    elif "swmm" in enabled:
        link_group = "/swmm_only/swmm_link"
    else:
        return _emit_model_type_skip_placeholder(
            output_path,
            "conduit_flow not applicable for triton-only analyses",
            report_cfg.figure_defaults.savefig_dpi,
        )

    # Read from the consolidated analysis DataTree. The link group carries
    # `max_over_full_flow`, `max_flow_cms`, and `link_id` for every event_iloc.
    link_summary_path = analysis.analysis_paths.analysis_datatree_zarr
    tree = analysis.process.open_datatree()
    if link_group not in tree.groups:
        raise AssertionError(
            f"consolidated tree missing expected group {link_group}; available: "
            f"{sorted(tree.groups)}"
        )
    ds_links = tree[link_group].to_dataset()
    max_over_full_da = ds_links["max_over_full_flow"].sel(event_iloc=event_iloc)
    peak_flow_da = ds_links["max_flow_cms"].sel(event_iloc=event_iloc)
    max_over_full = max_over_full_da.values
    peak_flow = peak_flow_da.values
    link_ids = ds_links["link_id"].values
    max_over_full_attrs = dict(max_over_full_da.attrs)
    max_over_full_name = max_over_full_da.name
    peak_flow_attrs = dict(peak_flow_da.attrs)
    peak_flow_name = peak_flow_da.name

    # Conduit geometry from swmmio. Use the HYDRAULICS .inp (which carries
    # [CONDUITS] + [COORDINATES] sections); the prior version of this code read
    # `swmm_hydro_inp` which is the hydrology-only variant (no [CONDUITS]) and
    # produced a blank figure (iter-2 user feedback 2026-04-27).
    inp_path = Path(
        getattr(proc.scen_paths, "swmm_hydraulics_inp", None)
        or proc.scen_paths.swmm_full_inp
    )
    model = swmmio.Model(str(inp_path))
    coords_df = model.inp.coordinates
    conduits_df = model.inp.conduits
    coords_by_id: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for row in conduits_df.itertuples():
        if row.InletNode in coords_df.index and row.OutletNode in coords_df.index:
            p_in = (
                float(coords_df.at[row.InletNode, "X"]),
                float(coords_df.at[row.InletNode, "Y"]),
            )
            p_out = (
                float(coords_df.at[row.OutletNode, "X"]),
                float(coords_df.at[row.OutletNode, "Y"]),
            )
            coords_by_id[str(row.Index)] = (p_in, p_out)

    # Subiteration 9.2 C6/C7 — switched from 2-column (utilization + peak)
    # layout to 3-column matching `per_sim_peak_flood_depth.py`: utilization
    # map | peak-flow map | Event hydrology stack on the right. Reuse the
    # shared `_hydrology_panel.draw_event_hydrology_panel` helper.
    from TRITON_SWMM_toolkit.report_renderers._hydrology_panel import (
        draw_event_hydrology_panel,
        load_event_hydrology_data,
    )

    weather_event_indexers = analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
    weather_path = Path(analysis.cfg_analysis.weather_timeseries)
    hydro_data = load_event_hydrology_data(
        weather_path, analysis.cfg_analysis, weather_event_indexers,
    )

    # Subiteration 9.4 C7-parity-2 — load DEM bounds (same source peak_flood_depth
    # uses) so map_aspect, fig_width, set_xlim, set_ylim, and ticks all match.
    # This is the root cause of inter-figure popping: peak_flood_depth uses
    # `fig_width = h * (2 * map_aspect * 1.02 + 1.0)` from `da.rio.bounds()`,
    # while conduit_flow previously hardcoded `map_aspect = 1.0`. For non-square
    # DEMs (synth fixture is 150m × 300m → map_aspect=0.5), the two figures had
    # different overall widths and panels popped between toggles.
    import rioxarray as rxr  # noqa: PLC0415
    sys_paths = analysis._system.sys_paths
    _dem_bounds_da = rxr.open_rasterio(sys_paths.dem_processed).squeeze()
    map_bounds = _dem_bounds_da.rio.bounds() if _dem_bounds_da.rio.crs is not None else (
        float(_dem_bounds_da.x.min()), float(_dem_bounds_da.y.min()),
        float(_dem_bounds_da.x.max()), float(_dem_bounds_da.y.max()),
    )
    map_aspect = (map_bounds[2] - map_bounds[0]) / max(map_bounds[3] - map_bounds[1], 1e-9)
    # Subiteration 9.4 — pin h to peak_flood_depth's value (cfg.figsize_inches
    # diverges between the two renderers; using cfg here would break parity).
    map_cfg = report_cfg.per_sim.map
    h = float(report_cfg.per_sim.peak_flood_depth.figsize_inches[1])
    fig_width = h * (2 * map_aspect * map_cfg.fig_width_panel_pad + 1.0)  # exactly matches peak_flood_depth
    fig = plt.figure(figsize=(fig_width, h), layout="constrained")
    outer = fig.add_gridspec(1, 3, width_ratios=list(map_cfg.outer_width_ratios), wspace=map_cfg.outer_wspace)
    _MAP_TO_CBAR_HEIGHT_RATIO = map_cfg.map_to_cbar_height_ratio
    gs_util = outer[0, 0].subgridspec(2, 1, height_ratios=[_MAP_TO_CBAR_HEIGHT_RATIO, 1])
    gs_peak = outer[0, 1].subgridspec(2, 1, height_ratios=[_MAP_TO_CBAR_HEIGHT_RATIO, 1])
    gs_util_cbar = gs_util[1, 0].subgridspec(1, 3, width_ratios=list(map_cfg.cbar_inner_width_ratios))
    gs_peak_cbar = gs_peak[1, 0].subgridspec(1, 3, width_ratios=list(map_cfg.cbar_inner_width_ratios))
    gs_hydro_outer = outer[0, 2].subgridspec(
        2, 1, height_ratios=[_MAP_TO_CBAR_HEIGHT_RATIO, 1],
    )
    gs_hydro_inner = gs_hydro_outer[0, 0].subgridspec(2, 1, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs_util[0, 0])
    cax_util = fig.add_subplot(gs_util_cbar[0, 1])
    ax2 = fig.add_subplot(gs_peak[0, 0], sharex=ax1, sharey=ax1)
    cax_peak = fig.add_subplot(gs_peak_cbar[0, 1])
    ax_rain = fig.add_subplot(gs_hydro_inner[0, 0])
    ax_bc = fig.add_subplot(gs_hydro_inner[1, 0], sharex=ax_rain)

    # Relpaths against analysis_dir for provenance-record portability.
    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    link_summary_rel = os.path.relpath(
        str(Path(link_summary_path).resolve()), analysis_root,
    )
    inp_rel = os.path.relpath(str(inp_path.resolve()), analysis_root)
    weather_rel = os.path.relpath(
        str(Path(weather_path).resolve()), analysis_root,
    )

    # Two-colormap design (iter-2 user feedback): non-overlapping single-color
    # gradations — Blues for utilization (cool / "filling up"), Reds for peak
    # flow magnitude (warm / "intensity"). `cfg.cmap` from report_config is
    # used as a fallback if user has overridden via YAML.
    UTILIZATION_CMAP = map_cfg.utilization_cmap
    PEAK_FLOW_CMAP = map_cfg.peak_flow_cmap
    panels = [
        (ax1, cax_util, max_over_full, max_over_full_name, max_over_full_attrs,
         "max / full flow", 0.0, 1.0, UTILIZATION_CMAP, "ax_utilization"),
        (ax2, cax_peak, peak_flow, peak_flow_name, peak_flow_attrs,
         units.flow_axis_label(),
         (float(cfg.vmin) if cfg.vmin is not None else 0.0),
         (float(cfg.vmax) if cfg.vmax is not None else float(peak_flow.max() or 1.0)),
         PEAK_FLOW_CMAP, "ax_peak_flow"),
    ]
    for ax, cax, values, var_name, var_attrs, label, vmin, vmax, cmap_name, axes_id in panels:
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        # Draw EVERY conduit, regardless of whether it has a value in the
        # link summary (iter-2 user feedback: zero-flow conduits should still
        # show their black outline + colormap-zero fill). Iterate over the
        # geometry so missing-from-summary conduits still appear.
        values_by_id = dict(zip(link_ids, values, strict=True))
        for lid, ((x1, y1), (x2, y2)) in coords_by_id.items():
            val = float(values_by_id.get(lid, 0.0))
            with prov.artist(
                axes_id=axes_id, kind="line2d",
                note=f"conduit {lid}",
            ) as a:
                a.add_swmm_channel(
                    "x", swmm_inp=inp_rel, kind="conduit_coords", link_id=str(lid),
                )
                a.add_swmm_channel(
                    "y", swmm_inp=inp_rel, kind="conduit_coords", link_id=str(lid),
                )
                a.add_channel(
                    "color",
                    _link_summary_ref(
                        link_summary_rel, var_name, var_attrs, lid, event_iloc,
                    ),
                    cmap=cmap_name, vmin=vmin, vmax=vmax,
                )
                # Black boundary underneath (iter-2 user feedback) — slightly
                # wider than the colored line for a thin black outline.
                ax.plot([x1, x2], [y1, y2], color=map_cfg.conduit_outline_color,
                        linewidth=map_cfg.conduit_outline_width,
                        solid_capstyle="round", zorder=2)
                ax.plot([x1, x2], [y1, y2], color=cmap(norm(val)),
                        linewidth=map_cfg.conduit_value_width,
                        solid_capstyle="round", zorder=3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(label)
        ax.set_aspect("equal")
        ax.set_title(label)

    # C7 — middle peak-flow panel shares y with the utilization panel; hide
    # redundant y-tick labels and ylabel so the gap collapses (matches
    # peak_flood_depth.py C8 fix).
    ax2.tick_params(axis="y", labelleft=False)
    ax2.set_ylabel("")
    crs = report_cfg.system_map.target_epsg or analysis._system.cfg_system.crs_epsg
    ax1.set_xlabel(units.easting_axis_label(crs), fontsize=map_cfg.axis_label_fontsize)
    ax1.set_ylabel(units.northing_axis_label(crs), fontsize=map_cfg.axis_label_fontsize)
    ax2.set_xlabel(units.easting_axis_label(crs), fontsize=map_cfg.axis_label_fontsize)
    ax1.tick_params(axis="both", labelsize=map_cfg.tick_labelsize)
    ax2.tick_params(axis="both", labelsize=map_cfg.tick_labelsize)
    # Subiteration 9.4 — explicit shared lims (matches peak_flood_depth's
    # da.rio.bounds()-derived auto-range so x/y ticks align between toggles).
    _xticks, _yticks = per_sim_map_ticks(map_bounds)
    for ax in (ax1, ax2):
        ax.set_xticks(_xticks)
        ax.set_yticks(_yticks)
        # Re-apply lims AFTER set_xticks (matplotlib expands lims to fit ticks).
        ax.set_xlim(map_bounds[0], map_bounds[2])
        ax.set_ylim(map_bounds[1], map_bounds[3])

    # Subiteration 9.4 — TRITON watershed boundary overlay (thin black solid),
    # matching peak_flood_depth's overlay so both per-sim figures show the
    # same domain context.
    import geopandas as gpd  # noqa: PLC0415
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceRef
    watershed_shp = analysis._system.cfg_system.watershed_gis_polygon
    watershed_rel = os.path.relpath(str(Path(watershed_shp).resolve()), analysis_root)
    watershed_gdf = gpd.read_file(watershed_shp)
    with prov.artist(
        axes_id="ax_overview",
        kind="line",
        note="watershed boundary overlay",
    ) as a:
        a.add_channel("geometry", ProvenanceRef(source_path=watershed_rel))
        for ax in (ax1, ax2):
            if watershed_gdf.crs is not None and _dem_bounds_da.rio.crs is not None:
                watershed_gdf.to_crs(_dem_bounds_da.rio.crs).boundary.plot(
                    ax=ax, color=map_cfg.watershed_overlay_color, linewidth=map_cfg.watershed_overlay_width,
                )
            else:
                watershed_gdf.boundary.plot(
                    ax=ax, color=map_cfg.watershed_overlay_color, linewidth=map_cfg.watershed_overlay_width,
                )

    # C6 — Event hydrology panel on the right (delegated to shared helper).
    draw_event_hydrology_panel(
        ax_rain, ax_bc,
        hydro_data=hydro_data,
        weather_rel_path=weather_rel,
        event_iloc=event_iloc,
        cfg_analysis=analysis.cfg_analysis,
        panel_cfg=report_cfg.per_sim.hydrology_panel,
        prov=prov,
    )

    # A4 — panel labels
    add_panel_label(ax1, "(a)")
    add_panel_label(ax2, "(b)")
    add_panel_label(ax_rain, "(c)")

    source_paths: list[Path] = [
        Path(link_summary_path), inp_path, Path(weather_path),
        Path(sys_paths.dem_processed), Path(watershed_shp),
    ]
    return emit_plot_with_sources(
        fig, output_path, source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        dpi=report_cfg.figure_defaults.savefig_dpi,
        output_format="svg" if output_path.suffix == ".svg" else "png",
        provenance=prov,
    )


def _link_summary_ref(source_rel: str, var_name, var_attrs, link_id, event_iloc):
    """Build a `ProvenanceRef` for a link-summary variable / link / event row."""
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceRef

    return ProvenanceRef(
        source_path=source_rel,
        variable=str(var_name) if var_name is not None else None,
        attrs=dict(var_attrs),
        selection={"link_id": str(link_id), "event_iloc": int(event_iloc)},
    )


def _emit_model_type_skip_placeholder(
    output_path: Path, message: str, dpi: int,
) -> Path:
    """Centered-text figure explaining a model-type skip (Gotcha 5).

    Keeps the Snakemake rule output present so the DAG does not fail, while
    making the inapplicability visible in the report.
    """
    fig, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, wrap=True)
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_plotly_branch(
    analysis,
    report_cfg,
    output_path: Path,
    *,
    event_iloc: int,
    prov,
) -> Path:
    """Plotly MV port (pre-/design-figure): static 3-panel figure with utilization
    map + peak-flow map + event hydrology. Geometric layout parity with
    `_render_plotly_branch` in per_sim_peak_flood_depth.py (same 2x3 specs grid).
    Informationally congruent with the matplotlib branch — no hover refinement,
    no max_over_full filter slider, no legend-click magnitude-class toggling.
    """
    import geopandas as gpd
    import rioxarray as rxr
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        emit_plot_with_sources,
    )
    from TRITON_SWMM_toolkit.report_renderers._hydrology_panel import (
        load_event_hydrology_data,
    )
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceRef
    # Side-effect import: registers `triton_journal` Plotly template.
    from TRITON_SWMM_toolkit.report_renderers import _plotly_theme  # noqa: F401
    import matplotlib.cm as mcm
    from matplotlib.colors import Normalize as _MplNormalize

    cfg = report_cfg.per_sim.conduit_flow
    map_cfg = report_cfg.per_sim.map

    proc = analysis._retrieve_sim_run_processing_object(event_iloc)

    # ---- Model-type dispatch (mirror of matplotlib branch) --------------
    enabled = analysis._get_enabled_model_types()
    if "tritonswmm" in enabled:
        link_group = "/tritonswmm/swmm_link"
    elif "swmm" in enabled:
        link_group = "/swmm_only/swmm_link"
    else:
        return _emit_model_type_skip_placeholder(
            output_path,
            "conduit_flow not applicable for triton-only analyses",
            report_cfg.figure_defaults.savefig_dpi,
        )

    # ---- Data prep ------------------------------------------------------
    link_summary_path = analysis.analysis_paths.analysis_datatree_zarr
    tree = analysis.process.open_datatree()
    if link_group not in tree.groups:
        raise AssertionError(
            f"consolidated tree missing expected group {link_group}; available: "
            f"{sorted(tree.groups)}"
        )
    ds_links = tree[link_group].to_dataset()
    max_over_full_da = ds_links["max_over_full_flow"].sel(event_iloc=event_iloc)
    peak_flow_da = ds_links["max_flow_cms"].sel(event_iloc=event_iloc)
    max_over_full = max_over_full_da.values
    peak_flow = peak_flow_da.values
    link_ids = ds_links["link_id"].values
    max_over_full_attrs = dict(max_over_full_da.attrs)
    max_over_full_name = max_over_full_da.name
    peak_flow_attrs = dict(peak_flow_da.attrs)
    peak_flow_name = peak_flow_da.name

    inp_path = Path(
        getattr(proc.scen_paths, "swmm_hydraulics_inp", None)
        or proc.scen_paths.swmm_full_inp
    )
    model = swmmio.Model(str(inp_path))
    coords_df = model.inp.coordinates
    conduits_df = model.inp.conduits
    coords_by_id: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for row in conduits_df.itertuples():
        if row.InletNode in coords_df.index and row.OutletNode in coords_df.index:
            p_in = (
                float(coords_df.at[row.InletNode, "X"]),
                float(coords_df.at[row.InletNode, "Y"]),
            )
            p_out = (
                float(coords_df.at[row.OutletNode, "X"]),
                float(coords_df.at[row.OutletNode, "Y"]),
            )
            coords_by_id[str(row.Index)] = (p_in, p_out)

    weather_event_indexers = analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
    weather_path = Path(analysis.cfg_analysis.weather_timeseries)
    hydro_data = load_event_hydrology_data(
        weather_path, analysis.cfg_analysis, weather_event_indexers,
    )
    times_min = hydro_data["times_min"]
    rainfall = hydro_data["rainfall"]
    bc_water_level = hydro_data["bc_water_level"]
    rain_attrs = hydro_data["rain_attrs"]
    bc_attrs = hydro_data["bc_attrs"]
    rain_var = hydro_data["rain_var"]
    bc_var = hydro_data["bc_var"]

    sys_paths = analysis._system.sys_paths
    _dem_da = rxr.open_rasterio(sys_paths.dem_processed).squeeze()
    map_bounds = _dem_da.rio.bounds() if _dem_da.rio.crs is not None else (
        float(_dem_da.x.min()), float(_dem_da.y.min()),
        float(_dem_da.x.max()), float(_dem_da.y.max()),
    )

    watershed_shp = analysis._system.cfg_system.watershed_gis_polygon
    watershed_gdf = gpd.read_file(watershed_shp)

    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    link_summary_rel = os.path.relpath(
        str(Path(link_summary_path).resolve()), analysis_root,
    )
    inp_rel = os.path.relpath(str(inp_path.resolve()), analysis_root)
    weather_rel = os.path.relpath(
        str(Path(weather_path).resolve()), analysis_root,
    )
    watershed_rel = os.path.relpath(
        str(Path(watershed_shp).resolve()), analysis_root,
    )

    # ---- Build figure ---------------------------------------------------
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"rowspan": 2}, {"rowspan": 2}, {}],
            [None, None, {}],
        ],
        row_heights=[1, 1],
        column_widths=[1, 1, 0.8],
        horizontal_spacing=0.06,
        vertical_spacing=0.06,
        subplot_titles=("Max / full flow", "Peak flow", "Rainfall"),
    )
    fig.update_layout(
        template="plotly_white",
        title=f"Per-sim conduit flow — event_iloc {event_iloc}",
        showlegend=False,
        margin=dict(l=10, r=10, t=80, b=60),
    )

    # ---- Utilization + peak-flow panels (binned-trace approach) ----------
    UTILIZATION_CMAP = map_cfg.utilization_cmap
    PEAK_FLOW_CMAP = map_cfg.peak_flow_cmap
    N_BINS = 20
    panels = [
        {
            "col": 1, "axes_id": "ax_utilization_plotly",
            "values": max_over_full, "var_name": max_over_full_name,
            "var_attrs": max_over_full_attrs,
            "label": "max / full flow", "vmin": 0.0, "vmax": 1.0,
            "cmap_name": UTILIZATION_CMAP, "colorbar_x": 0.16,
        },
        {
            "col": 2, "axes_id": "ax_peak_flow_plotly",
            "values": peak_flow, "var_name": peak_flow_name,
            "var_attrs": peak_flow_attrs,
            "label": units.flow_axis_label(),
            "vmin": float(cfg.vmin) if cfg.vmin is not None else 0.0,
            "vmax": float(cfg.vmax) if cfg.vmax is not None else float(peak_flow.max() or 1.0),
            "cmap_name": PEAK_FLOW_CMAP, "colorbar_x": 0.52,
        },
    ]
    values_by_id_per_panel = {
        p["axes_id"]: dict(zip(link_ids, p["values"], strict=True))
        for p in panels
    }
    for p in panels:
        cmap = mcm.get_cmap(p["cmap_name"])
        norm = _MplNormalize(vmin=p["vmin"], vmax=p["vmax"])
        # Bin conduits into N_BINS equal-value bins so we emit ~20 traces
        # instead of one trace per conduit (would be thousands of traces for
        # large networks). Color is constant per bin (mid-of-bin sampled from
        # cmap). /design-figure iteration may swap binned-traces for a single
        # WebGL trace with per-segment color via go.Scattergl + line.color.
        bin_edges = np.linspace(p["vmin"], p["vmax"], N_BINS + 1)
        bin_to_conduits: dict[int, list[tuple[str, tuple, tuple, float]]] = {
            i: [] for i in range(N_BINS)
        }
        for lid, ((x1, y1), (x2, y2)) in coords_by_id.items():
            val = float(values_by_id_per_panel[p["axes_id"]].get(lid, 0.0))
            bin_idx = int(np.clip(np.searchsorted(bin_edges, val, side="right") - 1, 0, N_BINS - 1))
            bin_to_conduits[bin_idx].append((lid, (x1, y1), (x2, y2), val))
        for bin_idx in range(N_BINS):
            conduits_in_bin = bin_to_conduits[bin_idx]
            if not conduits_in_bin:
                continue
            mid_val = 0.5 * (bin_edges[bin_idx] + bin_edges[bin_idx + 1])
            color_rgba = cmap(norm(mid_val))
            color_hex = "rgba({:.0f},{:.0f},{:.0f},{:.3f})".format(
                color_rgba[0] * 255, color_rgba[1] * 255,
                color_rgba[2] * 255, color_rgba[3],
            )
            xs: list[float | None] = []
            ys: list[float | None] = []
            for _lid, p_in, p_out, _val in conduits_in_bin:
                xs.extend([p_in[0], p_out[0], None])
                ys.extend([p_in[1], p_out[1], None])
            with prov.artist(
                axes_id=p["axes_id"], kind="line2d",
                note=(
                    f"conduit lines (bin {bin_idx}/{N_BINS}, "
                    f"mid_val={mid_val:.3f}, n={len(conduits_in_bin)}) — "
                    f"per-conduit channels registered in inner loop"
                ),
            ) as a:
                # Register per-conduit channels (analogous to matplotlib branch's
                # per-conduit prov.artist blocks; consolidated here under a single
                # binned trace's prov block).
                for lid_inner, _, _, _ in conduits_in_bin:
                    a.add_swmm_channel(
                        "x", swmm_inp=inp_rel, kind="conduit_coords",
                        link_id=str(lid_inner),
                    )
                    a.add_swmm_channel(
                        "y", swmm_inp=inp_rel, kind="conduit_coords",
                        link_id=str(lid_inner),
                    )
                    a.add_channel(
                        "color",
                        _link_summary_ref(
                            link_summary_rel, p["var_name"], p["var_attrs"],
                            lid_inner, event_iloc,
                        ),
                        cmap=p["cmap_name"], vmin=p["vmin"], vmax=p["vmax"],
                    )
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines",
                        line=dict(color=color_hex, width=map_cfg.conduit_value_width),
                        name=f"{p['label']} bin {bin_idx}",
                        legendgroup=p["axes_id"], showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=p["col"],
                )
        # Add an invisible scatter trace to drive the colorbar (Plotly trick:
        # a marker trace with colorscale + cmin/cmax + showscale=True emits a
        # colorbar even when its data points are all NaN).
        with prov.artist(
            axes_id=p["axes_id"], kind="image",
            note=f"colorbar for {p['label']}",
        ):
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(
                        colorscale=_mpl_cmap_to_plotly_colorscale(p["cmap_name"]),
                        cmin=p["vmin"], cmax=p["vmax"],
                        showscale=True, color=[p["vmin"]],
                        colorbar=dict(
                            title=p["label"], orientation="h",
                            y=-0.10, len=0.30, x=p["colorbar_x"], thickness=12,
                        ),
                    ),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=p["col"],
            )

    # ---- Watershed boundary overlay on both maps ------------------------
    target_crs = _dem_da.rio.crs if _dem_da.rio.crs is not None else None
    if watershed_gdf.crs is not None and target_crs is not None:
        ws_proj = watershed_gdf.to_crs(target_crs)
    else:
        ws_proj = watershed_gdf
    ws_x, ws_y = [], []
    for geom in ws_proj.boundary.values:
        if geom is None:
            continue
        geoms = geom.geoms if hasattr(geom, "geoms") else [geom]
        for line in geoms:
            xs_geom, ys_geom = line.coords.xy
            ws_x.extend(list(xs_geom) + [None])
            ws_y.extend(list(ys_geom) + [None])
    watershed_ref = ProvenanceRef(
        source_path=watershed_rel, variable="watershed_polygon", attrs={},
    )
    for col_idx, axes_id in ((1, "ax_utilization_plotly"), (2, "ax_peak_flow_plotly")):
        with prov.artist(
            axes_id=axes_id, kind="patch",
            note="watershed boundary overlay",
        ) as a:
            a.add_channel("x", watershed_ref)
            a.add_channel("y", watershed_ref)
            fig.add_trace(
                go.Scatter(
                    x=ws_x, y=ws_y, mode="lines",
                    line=dict(color=map_cfg.watershed_overlay_color,
                              width=map_cfg.watershed_overlay_width),
                    hoverinfo="skip", showlegend=False, name="watershed",
                ),
                row=1, col=col_idx,
            )

    # ---- Hydrology panel (rainfall row 1 + BC water level row 2) --------
    rain_units = units.rainfall_provenance_units(analysis.cfg_analysis.rainfall_units)
    bc_units = (
        units.bc_provenance_units(analysis.cfg_analysis.storm_tide_units)
        if analysis.cfg_analysis.storm_tide_units else ""
    )
    rain_ref = ProvenanceRef(
        source_path=weather_rel, variable=rain_var,
        attrs=rain_attrs, selection={"event_iloc": int(event_iloc)},
    )
    panel_cfg = report_cfg.per_sim.hydrology_panel
    with prov.artist(
        axes_id="ax_rain_plotly", kind="bar",
        note="rainfall time series (event hydrology — top sub-panel)",
    ) as a:
        a.add_channel("x", rain_ref, units=units.TIME_AXIS_PROVENANCE_UNITS)
        a.add_channel("y", rain_ref, units=rain_units)
        fig.add_trace(
            go.Bar(
                x=times_min, y=rainfall,
                marker=dict(color=panel_cfg.rain_color),
                name="rainfall", showlegend=False,
                hovertemplate="t: %{x} min<br>rain: %{y:.2f}<extra></extra>",
            ),
            row=1, col=3,
        )
    bc_ref = ProvenanceRef(
        source_path=weather_rel,
        variable=bc_var if bc_var is not None else "",
        attrs=bc_attrs, selection={"event_iloc": int(event_iloc)},
    )
    with prov.artist(
        axes_id="ax_bc_plotly", kind="line2d",
        note="boundary condition water level (event hydrology — bottom sub-panel)",
    ) as a:
        a.add_channel("x", bc_ref, units=units.TIME_AXIS_PROVENANCE_UNITS)
        a.add_channel("y", bc_ref, units=bc_units)
        fig.add_trace(
            go.Scatter(
                x=times_min, y=bc_water_level, mode="lines",
                line=dict(color=panel_cfg.bc_line_color,
                          width=panel_cfg.bc_line_width),
                name="bc_water_level", showlegend=False,
                hovertemplate="t: %{x} min<br>BC: %{y:.3f} m<extra></extra>",
            ),
            row=2, col=3,
        )

    # ---- Axes setup -----------------------------------------------------
    crs_for_labels = (
        report_cfg.system_map.target_epsg
        or analysis._system.cfg_system.crs_epsg
    )
    for col in (1, 2):
        fig.update_xaxes(
            range=[map_bounds[0], map_bounds[2]],
            title_text=units.easting_axis_label(crs_for_labels),
            row=1, col=col,
        )
        fig.update_yaxes(
            range=[map_bounds[1], map_bounds[3]],
            scaleanchor=f"x{col}", scaleratio=1.0,
            title_text=units.northing_axis_label(crs_for_labels) if col == 1 else None,
            row=1, col=col,
        )
    fig.update_xaxes(
        range=[float(times_min[0]), float(times_min[-1])],
        title_text="", row=1, col=3,
    )
    fig.update_yaxes(
        title_text=units.rainfall_axis_label(analysis.cfg_analysis.rainfall_units),
        row=1, col=3,
    )
    fig.update_xaxes(
        range=[float(times_min[0]), float(times_min[-1])],
        title_text=units.TIME_AXIS_FROM_EVENT_START, row=2, col=3,
    )
    fig.update_yaxes(
        title_text=units.bc_water_level_axis_label(
            analysis.cfg_analysis.storm_tide_units or "m",
        ),
        row=2, col=3,
    )

    # ---- Emit -----------------------------------------------------------
    plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d", "select2d", "autoScale2d",
            "hoverCompareCartesian", "hoverClosestCartesian",
            "toggleSpikelines",
        ],
        "toImageButtonOptions": {
            "format": "svg", "filename": "conduit_flow", "scale": 2,
        },
    }
    html_text = pio.to_html(
        fig, include_plotlyjs=report_cfg.interactive.plotly_js_mode,
        full_html=True, config=plotly_config,
    )

    source_paths: list[Path] = [
        Path(link_summary_path), inp_path, Path(weather_path),
        Path(sys_paths.dem_processed), Path(watershed_shp),
    ]

    try:
        fig.write_image(
            output_path.with_suffix(".svg"),
            engine="kaleido", width=1400, height=600, scale=1,
        )
    except Exception as exc:  # noqa: BLE001 — Kaleido failure is non-fatal
        import logging
        logging.getLogger(__name__).warning(
            "Kaleido SVG export skipped for %s: %s",
            output_path.with_suffix(".svg"), exc,
        )

    return emit_plot_with_sources(
        html_text, output_path, source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        output_format="html",
        manifest_data={
            "event_iloc": int(event_iloc),
            "max_over_full_max": float(np.nanmax(max_over_full)),
            "peak_flow_max_cms": float(np.nanmax(peak_flow)),
            "conduit_count": int(len(coords_by_id)),
            "binned_traces_per_panel": int(N_BINS),
        },
        provenance=prov,
    )


def _mpl_cmap_to_plotly_colorscale(cmap_name: str, n_samples: int = 32) -> list:
    """Return a Plotly-compatible colorscale list sampled from a matplotlib cmap."""
    import matplotlib.cm as mcm
    cmap = mcm.get_cmap(cmap_name)
    return [
        [
            i / (n_samples - 1),
            "rgb({:.0f},{:.0f},{:.0f})".format(
                cmap(i / (n_samples - 1))[0] * 255,
                cmap(i / (n_samples - 1))[1] * 255,
                cmap(i / (n_samples - 1))[2] * 255,
            ),
        ]
        for i in range(n_samples)
    ]
