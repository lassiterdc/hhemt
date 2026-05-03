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
import swmmio

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

    proc = analysis._retrieve_sim_run_processing_object(event_iloc)

    # Model-type dispatch (Gotcha 5 from the master plan).
    enabled = analysis._get_enabled_model_types()
    if "tritonswmm" in enabled:
        link_summary_path = proc.scen_paths.output_tritonswmm_link_summary
    elif "swmm" in enabled:
        link_summary_path = proc.scen_paths.output_swmm_only_link_summary
    else:
        return _emit_model_type_skip_placeholder(
            output_path,
            "conduit_flow not applicable for triton-only analyses",
            report_cfg.figure_defaults.savefig_dpi,
        )

    # Delegate to proc._open() so engine selection + zarr consolidated=False +
    # decode_timedelta=False are applied uniformly. Wrapped in `with` so file
    # handles release between Hard-STOP iteration renders (gis-specialist note).
    with proc._open(link_summary_path) as ds_links:
        if ds_links.sizes.get("event_iloc") != 1:
            raise AssertionError(
                f"per-scenario link summary expected event_iloc=1, got "
                f"{ds_links.sizes.get('event_iloc')}"
            )
        max_over_full_da = ds_links["max_over_full_flow"].sel(event_iloc=event_iloc)
        peak_flow_da = ds_links["max_flow_cms"].sel(event_iloc=event_iloc)
        max_over_full = max_over_full_da.values
        peak_flow = peak_flow_da.values
        link_ids = ds_links["link_id"].values
        # Cache xarray metadata before the dataset closes — `.attrs`/`.name`
        # are Python-side metadata so they remain valid post-close, but cache
        # them now to make the close-or-not invariant explicit.
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

    weather_path = proc.scen_paths.weather_timeseries
    hydro_data = load_event_hydrology_data(weather_path, analysis.cfg_analysis)

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
    watershed_shp = analysis._system.cfg_system.watershed_gis_polygon
    watershed_gdf = gpd.read_file(watershed_shp)
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
