"""Combined 2-panel system-overview renderer.

Left panel: SWMM model elements (subcatchments, junctions, conduits, outfall,
storm-tide BC line) — every element traces to a real SWMM `.inp` attribute
(no synthesised geometry).

Right panel: DEM elevation raster with a bathtub-aware colormap clipping so
the interior gradient stays visible when outlier-wall cells dominate.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import swmmio

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


_JUNCTION_FILL = "#1f77b4"
_OUTFALL_FILL = "#d62728"
_DRAINAGE_LINE_COLOR = _JUNCTION_FILL  # same blue as junction fill


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    from TRITON_SWMM_toolkit.config.report import resolve_target_crs
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        emit_plot_with_sources,
    )

    cfg_ana = analysis.cfg_analysis
    sys_paths = analysis._system.sys_paths
    map_cfg = report_cfg.system_map

    _apply_rcparams(report_cfg)
    target_crs = resolve_target_crs(analysis, report_cfg)

    # ---- Load DEM + BC up front so the figure can be sized to data aspect.
    dem = rxr.open_rasterio(sys_paths.dem_processed)
    if dem.rio.crs is not None and dem.rio.crs != target_crs:
        dem = dem.rio.reproject(target_crs)
    dem_bounds = dem.rio.bounds()

    # Two-panel figure with shared x- and y-axes pinned to the DEM bounds
    # (iter-2 mid-iteration fix 2026-04-26). With `aspect="equal"` on both
    # panels, the figure width must match the data aspect ratio of two
    # side-by-side panels — otherwise matplotlib shrinks each panel to data
    # aspect and the surplus slot space becomes a wide gap between them.
    w, h = map_cfg.figsize_inches
    dem_x_extent = dem_bounds[2] - dem_bounds[0]
    dem_y_extent = dem_bounds[3] - dem_bounds[1]
    panel_aspect = dem_x_extent / dem_y_extent if dem_y_extent else 1.0
    fig_width = max(2 * h * panel_aspect * 1.15, h * 1.3)
    fig, (ax_model, ax_elev) = plt.subplots(
        1, 2, figsize=(fig_width, h), sharex=True, sharey=True,
    )
    # `bottom=0.20` leaves room below each axes for a 2-column legend
    # placed clear of the x-tick labels.
    fig.subplots_adjust(left=0.05, right=0.97, top=0.92,
                       bottom=0.20, wspace=0.04)
    bc_path: Path | None = None
    if cfg_ana.toggle_storm_tide_boundary and cfg_ana.storm_tide_boundary_line_gis:
        bc_path = Path(cfg_ana.storm_tide_boundary_line_gis)
        if not bc_path.exists():
            bc_path = None

    # ---- Load SWMM topology (primary + optional conduits supplement) ----
    primary_inp, conduits_inp = _resolve_inp_sources(analysis)
    primary = swmmio.Model(str(primary_inp))
    coords_df = primary.inp.coordinates
    junctions_df = primary.inp.junctions
    outfalls_df = primary.inp.outfalls
    subcatch_df = getattr(primary.inp, "subcatchments", None)
    polygons_df = getattr(primary.inp, "polygons", None)
    conduits_df = primary.inp.conduits
    if conduits_inp is not None:
        conduits_df = swmmio.Model(str(conduits_inp)).inp.conduits

    # ---- Left panel: model elements -------------------------------------
    _draw_model_elements_panel(
        ax_model,
        dem_bounds, bc_path, target_crs, map_cfg,
        coords_df, junctions_df, outfalls_df,
        subcatch_df, polygons_df, conduits_df,
    )

    # ---- Right panel: elevation map -------------------------------------
    _draw_elevation_panel(
        ax_elev, dem, dem_bounds, bc_path, target_crs, map_cfg,
    )

    fig.suptitle(f"System overview — {analysis.cfg_analysis.analysis_id}")

    source_paths: list[Path] = [
        sys_paths.dem_processed,
        primary_inp,
    ]
    if conduits_inp is not None:
        source_paths.append(conduits_inp)
    if bc_path is not None:
        source_paths.append(bc_path)

    manifest_data = _build_manifest_data(
        analysis_id=analysis.cfg_analysis.analysis_id,
        ax_model=ax_model,
        ax_elev=ax_elev,
        dem_bounds=dem_bounds,
        coords_df=coords_df,
        junctions_df=junctions_df,
        outfalls_df=outfalls_df,
        subcatch_df=subcatch_df,
        polygons_df=polygons_df,
        conduits_df=conduits_df,
        bc_present=bc_path is not None,
    )
    return emit_plot_with_sources(
        fig, output_path, source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        dpi=report_cfg.figure_defaults.savefig_dpi,
        manifest_data=manifest_data,
    )


def _build_manifest_data(
    analysis_id, ax_model, ax_elev, dem_bounds,
    coords_df, junctions_df, outfalls_df, subcatch_df, polygons_df, conduits_df,
    bc_present: bool,
) -> dict:
    n_subcatchments = (
        int(len(polygons_df.index.unique())) if polygons_df is not None and len(polygons_df) > 0 else 0
    )
    # TRITON-extent rectangle removed from both panels per iter-2 feedback;
    # storm-tide BC removed from SWMM panel per iter-2 feedback.
    legend_model: list[str] = []
    if n_subcatchments:
        legend_model.append("Subcatchments")
    if len(conduits_df):
        legend_model.append("SWMM conduits")
    if len(junctions_df):
        legend_model.append("SWMM junction")
    # TRITON-extent rectangle removed from elevation panel per iteration-2 feedback.
    legend_elev: list[str] = []
    if bc_present:
        legend_elev.append("Storm tide BC")
    return {
        "analysis_id": str(analysis_id),
        "panels": [
            {
                "name": "swmm_elements",
                "title": ax_model.get_title(),
                "axis_extents": {
                    "xlim": list(ax_model.get_xlim()),
                    "ylim": list(ax_model.get_ylim()),
                },
                "element_counts": {
                    "junctions": int(len(junctions_df)),
                    "outfalls": int(len(outfalls_df)),
                    "conduits": int(len(conduits_df)),
                    "subcatchments_with_polygons": n_subcatchments,
                },
                "legend_labels": legend_model,
            },
            {
                "name": "triton_dem",
                "title": ax_elev.get_title(),
                "axis_extents": {
                    "xlim": list(ax_elev.get_xlim()),
                    "ylim": list(ax_elev.get_ylim()),
                },
                "dem_bounds": list(dem_bounds),
                "legend_labels": legend_elev,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Left panel: SWMM model elements
# ---------------------------------------------------------------------------


def _draw_model_elements_panel(
    ax, dem_bounds, bc_path, target_crs, map_cfg,
    coords_df, junctions_df, outfalls_df, subcatch_df, polygons_df, conduits_df,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # TRITON-extent rectangle removed from BOTH panels per iteration-2 feedback
    # (mid-iteration addendum 2026-04-26 00:11 EDT — user: "I want the TRITON
    # extent boundary to be removed from the SWMM elements plot").
    legend_handles: list = []
    # Storm-tide BC drawing removed from this panel per iteration-2 feedback
    # (BC is a TRITON forcing, not a SWMM element). It remains on the
    # elevation panel for spatial context.

    # Subcatchments from [POLYGONS]
    drew_any_subcatchment = False
    if polygons_df is not None and len(polygons_df) > 0:
        drew_any_subcatchment = _draw_subcatchments_and_drainage_lines(
            ax, polygons_df, subcatch_df, coords_df,
        )
    if drew_any_subcatchment:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="#d62728", hatch="////",
                  label="Subcatchments")
        )

    # Conduits with slope labels
    connected_nodes = _collect_connected_nodes(conduits_df)
    _draw_conduits_with_slope_labels(
        ax, conduits_df, junctions_df, outfalls_df, coords_df,
    )
    if len(conduits_df) > 0:
        legend_handles.append(
            Line2D([], [], color=map_cfg.swmm_link_color, linewidth=1.2,
                   label="SWMM conduits")
        )

    # Junctions as circles
    if len(junctions_df):
        jx = [float(coords_df.at[n, "X"]) for n in junctions_df.index]
        jy = [float(coords_df.at[n, "Y"]) for n in junctions_df.index]
        ax.scatter(jx, jy, marker="o", s=70, color=_JUNCTION_FILL,
                   edgecolor="black", linewidths=0.8, zorder=6)
        legend_handles.append(
            Line2D([], [], color=_JUNCTION_FILL, marker="o", linestyle="None",
                   markersize=8, markeredgecolor="black", label="SWMM junction")
        )

    # Outfalls: upward triangle, NO legend entry (per iteration-4 feedback).
    if len(outfalls_df):
        ox = [float(coords_df.at[n, "X"]) for n in outfalls_df.index]
        oy = [float(coords_df.at[n, "Y"]) for n in outfalls_df.index]
        ax.scatter(ox, oy, marker="^", s=100, color=_OUTFALL_FILL,
                   edgecolor="black", linewidths=0.8, zorder=7)

    # Node labels
    _draw_node_labels(ax, coords_df, junctions_df, outfalls_df, connected_nodes)

    ax.set_aspect("equal")
    # Pin the SWMM panel's data extent to the DEM bounds (iter-2 mid-iteration
    # fix 2026-04-26): without this the auto-xlim shrinks the SWMM panel to
    # its element x-extent and `aspect="equal"` then renders it narrower
    # than the DEM panel, opening a visual gap.
    ax.set_xlim(dem_bounds[0], dem_bounds[2])
    ax.set_ylim(dem_bounds[1], dem_bounds[3])
    ax.set_title("SWMM elements")
    # Legend below the panel, 2 columns, pushed clear of x-tick labels
    # (iter-2 user feedback: legend was overlapping with axis tick labels).
    ax.legend(handles=legend_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=8, framealpha=0.9)


def _draw_subcatchments_and_drainage_lines(ax, polygons_df, subcatch_df, coords_df) -> bool:
    from matplotlib.patches import Polygon as MplPolygon

    drew = False
    for sc_name in polygons_df.index.unique():
        rows = polygons_df.loc[[sc_name]]
        verts = list(zip(rows["X"].astype(float), rows["Y"].astype(float), strict=True))
        if len(verts) < 3:
            continue
        ax.add_patch(MplPolygon(
            verts, closed=True,
            facecolor="none", edgecolor="#d62728", linewidth=1.0,
            hatch="////", zorder=2,
        ))
        drew = True
        if subcatch_df is None or sc_name not in subcatch_df.index:
            continue
        outlet_name = subcatch_df.at[sc_name, "Outlet"]
        if outlet_name not in coords_df.index:
            continue
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        ox = float(coords_df.at[outlet_name, "X"])
        oy = float(coords_df.at[outlet_name, "Y"])
        ax.plot([cx, ox], [cy, oy],
                color=_DRAINAGE_LINE_COLOR, linestyle="--", linewidth=1.0,
                zorder=3)
    return drew


def _draw_conduits_with_slope_labels(ax, conduits_df, junctions_df, outfalls_df,
                                     coords_df):
    inverts: dict[str, float] = {}
    for name, row in junctions_df.iterrows():
        inverts[name] = float(row["InvertElev"])
    for name, row in outfalls_df.iterrows():
        inverts[name] = float(row["InvertElev"])

    for row in conduits_df.itertuples():
        if row.InletNode not in coords_df.index or row.OutletNode not in coords_df.index:
            continue
        p_in = (float(coords_df.at[row.InletNode, "X"]),
                float(coords_df.at[row.InletNode, "Y"]))
        p_out = (float(coords_df.at[row.OutletNode, "X"]),
                 float(coords_df.at[row.OutletNode, "Y"]))
        ax.plot([p_in[0], p_out[0]], [p_in[1], p_out[1]],
                color="#555555", linewidth=1.2, zorder=4)
        length_m = float(getattr(row, "Length", 0.0))
        inv_in = inverts.get(row.InletNode)
        inv_out = inverts.get(row.OutletNode)
        if length_m > 0 and inv_in is not None and inv_out is not None:
            slope_pct = 100.0 * (inv_in - inv_out) / length_m
            mx = (p_in[0] + p_out[0]) / 2.0
            my = (p_in[1] + p_out[1]) / 2.0
            ax.annotate(
                f"{row.Index}\nSlope: {slope_pct:.2f}%",
                xy=(mx, my), xytext=(5, 5), textcoords="offset points",
                fontsize=6, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8),
            )


def _collect_connected_nodes(conduits_df) -> set[str]:
    nodes: set[str] = set()
    for row in conduits_df.itertuples():
        nodes.add(row.InletNode)
        nodes.add(row.OutletNode)
    return nodes


def _draw_node_labels(ax, coords_df, junctions_df, outfalls_df, connected_nodes):
    label_offset = (8, -6)
    rows = []
    for name, row in junctions_df.iterrows():
        if name not in coords_df.index:
            continue
        invert = float(row["InvertElev"])
        maxd = float(row.get("MaxDepth", 0.0))
        rim = invert + maxd
        rows.append((name, invert, rim, "junction"))
    for name, row in outfalls_df.iterrows():
        if name not in coords_df.index:
            continue
        rows.append((name, float(row["InvertElev"]), None, "outfall"))
    for name, invert, rim, kind in rows:
        x = float(coords_df.at[name, "X"])
        y = float(coords_df.at[name, "Y"])
        dx, dy = label_offset
        if kind == "outfall":
            label_text = f"{name}"
        elif rim is not None:
            label_text = f"{name}\nRim: {rim:.2f}\nInv: {invert:.2f}"
        else:
            label_text = f"{name}\nInv: {invert:.2f}"
        ax.annotate(
            label_text,
            xy=(x, y), xytext=(dx, dy), textcoords="offset points",
            fontsize=6, zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.75),
        )


# ---------------------------------------------------------------------------
# Right panel: elevation raster
# ---------------------------------------------------------------------------


def _draw_elevation_panel(ax, dem, dem_bounds, bc_path, target_crs, map_cfg):
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    arr = dem.squeeze().values
    valid = arr[np.isfinite(arr)]
    if valid.size:
        median = float(np.median(valid))
        p5 = float(np.percentile(valid, 5))
        p95 = float(np.percentile(valid, 95))
        # Bimodal-wall detection: p95 >> median indicates outlier/wall cluster.
        if p95 > 5 * median:
            vmax = max(3.0 * median, p5 + 0.1)
        else:
            vmax = p95
        vmin = p5
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    cmap = cm.get_cmap("terrain").copy()
    cmap.set_over("#808080")  # medium grey against white background
    im = ax.imshow(
        arr,
        cmap=cmap, vmin=vmin, vmax=vmax,
        extent=(dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]),
        origin="upper", aspect="equal",
    )
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, extend="max")
    cbar.set_label("Elevation (m)")

    # TRITON-extent rectangle removed from this panel per iteration-2 feedback
    # (the colored DEM cells already convey the extent).
    legend_handles = []

    if bc_path is not None:
        bc_gdf = gpd.read_file(bc_path)
        if bc_gdf.crs is not None:
            bc_gdf.to_crs(target_crs).plot(
                ax=ax, color=map_cfg.bc_color, linewidth=2.5,
            )
            legend_handles.append(
                Line2D([], [], color=map_cfg.bc_color, linewidth=2.5,
                       label="Storm tide BC")
            )

    ax.set_title("TRITON DEM")
    if legend_handles:
        # Match the SWMM panel's legend offset (iter-2 user feedback: same
        # legend overlap fix needed on the right panel).
        ax.legend(handles=legend_handles,
                  loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=1, fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_inp_sources(analysis: TRITONSWMM_analysis) -> tuple[Path, Path | None]:
    """Return `(primary, conduits_supplement_or_None)`.

    Primary has polygons + subcatchments + coordinates + junctions + outfalls.
    If primary = `full.inp`, supplement is None. Otherwise `hydraulics.inp`
    supplements the conduits.
    """
    def _pick(scenario_paths):
        full = getattr(scenario_paths, "swmm_full_inp", None)
        if full is not None and Path(full).exists():
            return Path(full), None
        hydro = getattr(scenario_paths, "swmm_hydro_inp", None)
        hydraulics = getattr(scenario_paths, "swmm_hydraulics_inp", None)
        hydraulics_path = Path(hydraulics) if hydraulics and Path(hydraulics).exists() else None
        if hydro is not None and Path(hydro).exists():
            return Path(hydro), hydraulics_path
        if hydraulics_path is not None:
            return hydraulics_path, None
        return Path(scenario_paths.swmm_hydro_inp), None

    if getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False):
        subs = analysis.sensitivity.sub_analyses
        first_sub = subs[next(iter(subs))]
        return _pick(first_sub._retrieve_sim_runs(0)._scenario.scen_paths)
    return _pick(analysis._retrieve_sim_runs(0)._scenario.scen_paths)


def _apply_rcparams(report_cfg: report_config) -> None:
    fd = report_cfg.figure_defaults
    plt.rcParams.update({
        "font.family": fd.font_family,
        "font.size": fd.font_size,
        "figure.dpi": fd.dpi,
        "savefig.dpi": fd.savefig_dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
