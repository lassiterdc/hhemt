"""Three-panel system-overview renderer.

Iteration 3 (2026-04-27): the original two-panel design conflated two domain
views in a single SWMM panel. Iteration 3 splits them so each panel's artists
trace cleanly to one source `.inp` file:

- **Hydrology panel** (left) — subcatchment polygons, drainage lines from each
  polygon centroid to its outlet node, and small outlet markers. All artists
  trace to `swmm_hydro.inp`.
- **Hydraulics panel** (middle) — junctions, outfalls, conduits with slope
  labels, node Rim/Inv labels. All artists trace to `swmm_hydraulics.inp`.
- **TRITON DEM panel** (right) — elevation raster with bathtub-aware colormap
  clipping, sea-wall row in grey via `cmap.set_over`, and the storm-tide BC
  line overlay. Unchanged from iteration 2.

All three panels share x and y axes pinned to the DEM bounds.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import rioxarray as rxr
import swmmio
from plotly.subplots import make_subplots

from TRITON_SWMM_toolkit import swmm_schema as _ss, units

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
) -> Path:
    from TRITON_SWMM_toolkit.config.report import resolve_target_crs
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        emit_plot_with_sources,
    )
    from TRITON_SWMM_toolkit.report_renderers._provenance import ProvenanceLog

    cfg_ana = analysis.cfg_analysis
    sys_paths = analysis._system.sys_paths
    map_cfg = report_cfg.system_map
    static_backend = getattr(
        getattr(report_cfg, "interactive", None),
        "static_backend",
        "plotly",
    )
    use_plotly = (static_backend == "plotly")

    if not use_plotly:
        _apply_rcparams(report_cfg)
    target_crs = resolve_target_crs(analysis, report_cfg)
    prov = ProvenanceLog()

    # ---- Load DEM + BC up front so the figure can be sized to data aspect.
    dem = rxr.open_rasterio(sys_paths.dem_processed)
    if dem.rio.crs is not None and dem.rio.crs != target_crs:
        dem = dem.rio.reproject(target_crs)
    dem_bounds = dem.rio.bounds()

    bc_path: Path | None = None
    if cfg_ana.toggle_storm_tide_boundary and cfg_ana.storm_tide_boundary_line_gis:
        bc_path = Path(cfg_ana.storm_tide_boundary_line_gis)
        if not bc_path.exists():
            bc_path = None

    # ---- Resolve hydro + hydraulics .inp paths (both required, with full.inp
    # fallback for either when only the combined .inp exists).
    hydro_inp, hydraulics_inp = _resolve_inp_sources(analysis)

    # Relpaths against analysis_dir for provenance-record portability.
    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    hydro_rel = os.path.relpath(str(Path(hydro_inp).resolve()), analysis_root)
    hydraulics_rel = os.path.relpath(
        str(Path(hydraulics_inp).resolve()), analysis_root,
    )
    dem_rel = os.path.relpath(str(Path(sys_paths.dem_processed).resolve()), analysis_root)
    bc_rel = (
        os.path.relpath(str(Path(bc_path).resolve()), analysis_root)
        if bc_path is not None else None
    )

    # ---- Load each model independently so swmmio's empty-section warnings
    # are scoped (hydraulics has no [POLYGONS]; hydro has no populated
    # [CONDUITS]) and panel artists trace cleanly to the right file.
    hydro_model = swmmio.Model(str(hydro_inp))
    hydraulics_model = swmmio.Model(str(hydraulics_inp))

    # ---- Persistent GeoJSON exports for downstream GIS workflows --------
    # Written under <system_dir>/gis/ so the layers are durable outputs of
    # the system inputs (not analysis-run outputs). Idempotent — same .inp
    # data produces the same files. See _swmm_gis_layers.export_swmm_gis_layers.
    from TRITON_SWMM_toolkit.report_renderers._swmm_gis_layers import (
        export_swmm_gis_layers,
    )

    system_dir = Path(sys_paths.dem_processed).parent
    gis_dir = system_dir / "gis"
    export_swmm_gis_layers(
        hydro_model, hydraulics_model, gis_dir, target_crs=target_crs,
    )

    # Source-paths and manifest-data are shared across both branches.
    source_paths: list[Path] = [
        sys_paths.dem_processed,
        Path(hydro_inp),
        Path(hydraulics_inp),
    ]
    if bc_path is not None:
        source_paths.append(bc_path)

    manifest_data = _build_manifest_data(
        analysis_id=analysis.cfg_analysis.analysis_id,
        dem_bounds=dem_bounds,
        hydro_model=hydro_model,
        hydraulics_model=hydraulics_model,
        bc_present=bc_path is not None,
    )

    if use_plotly:
        # Pull the optional `dem_building_height` from cfg_system. The static
        # `plot_system.py::create_dem_plot` masks DEM cells equal to this
        # value before plotting (per the docstring on `SystemConfig`); the
        # Plotly branch does the same so the colorbar range reflects ground
        # elevations rather than the building-elevation plateau.
        building_height = getattr(
            analysis._system.cfg_system, "dem_building_height", None,
        )
        return _render_plotly_branch(
            output_path=output_path,
            source_paths=source_paths,
            analysis_dir=analysis.analysis_paths.analysis_dir,
            dem=dem, dem_bounds=dem_bounds,
            hydro_model=hydro_model, hydraulics_model=hydraulics_model,
            hydro_rel=hydro_rel, hydraulics_rel=hydraulics_rel, dem_rel=dem_rel,
            bc_path=bc_path, bc_rel=bc_rel, target_crs=target_crs,
            map_cfg=map_cfg, manifest_data=manifest_data, prov=prov,
            plotly_js_mode=report_cfg.interactive.plotly_js_mode,
            dem_building_height=building_height,
        )

    # Matplotlib branch (legacy / interactive.enabled=False default).
    _, h = map_cfg.figsize_inches
    dem_x_extent = dem_bounds[2] - dem_bounds[0]
    dem_y_extent = dem_bounds[3] - dem_bounds[1]
    panel_aspect = dem_x_extent / dem_y_extent if dem_y_extent else 1.0
    fig_width = max(
        3 * h * panel_aspect * map_cfg.fig_width_panel_pad,
        h * map_cfg.fig_width_min_factor,
    )
    fig, (ax_hydro, ax_hydraulics, ax_dem) = plt.subplots(
        1, 3, figsize=(fig_width, h), sharex=True, sharey=True,
    )
    fig.subplots_adjust(**map_cfg.subplots_adjust)

    _draw_hydrology_panel(
        ax_hydro, hydro_model, hydro_rel, dem_bounds, map_cfg, prov,
    )
    _draw_hydraulics_panel(
        ax_hydraulics, hydraulics_model, hydraulics_rel, dem_bounds, map_cfg, prov,
    )
    _draw_elevation_panel(
        ax_dem, dem, dem_bounds, bc_path, bc_rel, target_crs, map_cfg,
        prov, dem_source=dem_rel,
    )

    return emit_plot_with_sources(
        fig, output_path, source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        dpi=report_cfg.figure_defaults.savefig_dpi,
        manifest_data=manifest_data,
        provenance=prov,
    )


def _build_manifest_data(
    analysis_id, dem_bounds,
    hydro_model, hydraulics_model, bc_present: bool,
) -> dict:
    polygons_df = getattr(hydro_model.inp, "polygons", None)
    n_subcatchments = (
        int(len(polygons_df.index.unique()))
        if polygons_df is not None and len(polygons_df) > 0 else 0
    )
    panel_extents = {
        "xlim": [float(dem_bounds[0]), float(dem_bounds[2])],
        "ylim": [float(dem_bounds[1]), float(dem_bounds[3])],
    }
    return {
        "analysis_id": str(analysis_id),
        "panels": [
            {
                "name": "hydrology",
                "title": "Hydrology",
                "axis_extents": panel_extents,
                "element_counts": {
                    "subcatchments_with_polygons": n_subcatchments,
                    "subcatchment_rows": int(len(hydro_model.inp.subcatchments)),
                },
                "legend_labels": (
                    ["Subcatchments", "Drains to"] if n_subcatchments else []
                ),
            },
            {
                "name": "hydraulics",
                "title": "Hydraulics",
                "axis_extents": panel_extents,
                "element_counts": {
                    "junctions": int(len(hydraulics_model.inp.junctions)),
                    "outfalls": int(len(hydraulics_model.inp.outfalls)),
                    "conduits": int(len(hydraulics_model.inp.conduits)),
                },
                "legend_labels": ["SWMM conduits", "SWMM junction"],
            },
            {
                "name": "triton_dem",
                "title": "TRITON DEM",
                "axis_extents": panel_extents,
                "dem_bounds": [float(b) for b in dem_bounds],
                "legend_labels": ["Storm tide BC"] if bc_present else [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Hydrology panel — subcatchments + drainage lines (from swmm_hydro.inp)
# ---------------------------------------------------------------------------


def _draw_hydrology_panel(
    ax, hydro_model, hydro_rel: str, dem_bounds, map_cfg, prov,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    hp = map_cfg.hydrology_panel
    coords_df = hydro_model.inp.coordinates
    subcatch_df = getattr(hydro_model.inp, "subcatchments", None)
    polygons_df = getattr(hydro_model.inp, "polygons", None)

    legend_handles: list = []
    drew_any_subcatchment = False
    outlets_drawn: set[str] = set()

    if polygons_df is not None and len(polygons_df) > 0:
        drew_any_subcatchment = _draw_subcatchments_and_drainage_lines(
            ax, polygons_df, subcatch_df, coords_df, hp, prov,
            axes_id="ax_hydro", swmm_inp_rel=hydro_rel,
            outlets_drawn=outlets_drawn,
        )
    if drew_any_subcatchment:
        legend_handles.append(
            Patch(facecolor="none", edgecolor=hp.subcatchment_edge_color, hatch=hp.subcatchment_hatch,
                  label="Subcatchments")
        )
        legend_handles.append(
            Line2D([], [], color=hp.drainage_line_color, linestyle=hp.drainage_line_style,
                   linewidth=hp.drainage_line_width, label="Drains to")
        )

    if outlets_drawn:
        ox = [float(coords_df.at[n, _ss.COORDS_X]) for n in sorted(outlets_drawn)
              if n in coords_df.index]
        oy = [float(coords_df.at[n, _ss.COORDS_Y]) for n in sorted(outlets_drawn)
              if n in coords_df.index]
        if ox:
            with prov.artist(
                axes_id="ax_hydro", kind="scatter",
                note=f"subcatchment outlet markers ({len(ox)})",
            ) as a:
                a.add_swmm_channel("x", swmm_inp=hydro_rel, kind="outlet_node_coords")
                a.add_swmm_channel("y", swmm_inp=hydro_rel, kind="outlet_node_coords")
                ax.scatter(ox, oy, marker="o", s=hp.outlet_marker_size,
                           color=hp.outlet_marker_fill,
                           edgecolor="black", linewidths=hp.outlet_marker_edgewidth, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(dem_bounds[0], dem_bounds[2])
    ax.set_ylim(dem_bounds[1], dem_bounds[3])
    ax.set_title("Hydrology")
    if legend_handles:
        ax.legend(handles=legend_handles,
                  loc=map_cfg.legend_loc, bbox_to_anchor=map_cfg.legend_bbox_to_anchor,
                  ncol=2, fontsize=map_cfg.legend_fontsize, framealpha=map_cfg.legend_framealpha)


def _draw_subcatchments_and_drainage_lines(
    ax, polygons_df, subcatch_df, coords_df, hp, prov,
    *, axes_id: str, swmm_inp_rel: str, outlets_drawn: set[str],
) -> bool:
    from matplotlib.patches import Polygon as MplPolygon

    drew = False
    for sc_name in polygons_df.index.unique():
        rows = polygons_df.loc[[sc_name]]
        verts = list(zip(rows[_ss.COORDS_X].astype(float), rows[_ss.COORDS_Y].astype(float), strict=True))
        if len(verts) < 3:
            continue
        with prov.artist(
            axes_id=axes_id, kind="patch",
            note=f"subcatchment polygon {sc_name}",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=swmm_inp_rel,
                               kind="subcatchment_polygon", node_id=str(sc_name))
            a.add_swmm_channel("y", swmm_inp=swmm_inp_rel,
                               kind="subcatchment_polygon", node_id=str(sc_name))
            ax.add_patch(MplPolygon(
                verts, closed=True,
                facecolor="none", edgecolor=hp.subcatchment_edge_color, linewidth=hp.subcatchment_linewidth,
                hatch=hp.subcatchment_hatch, zorder=2,
            ))
        drew = True
        if subcatch_df is None or sc_name not in subcatch_df.index:
            continue
        outlet_name = subcatch_df.at[sc_name, _ss.SUBCATCH_OUTLET]
        if outlet_name not in coords_df.index:
            continue
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        ox = float(coords_df.at[outlet_name, _ss.COORDS_X])
        oy = float(coords_df.at[outlet_name, _ss.COORDS_Y])
        with prov.artist(
            axes_id=axes_id, kind="line2d",
            note=f"drainage line: {sc_name} → {outlet_name}",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=swmm_inp_rel,
                               kind="subcatchment_outlet",
                               node_id=str(sc_name))
            a.add_swmm_channel("y", swmm_inp=swmm_inp_rel,
                               kind="subcatchment_outlet",
                               node_id=str(sc_name))
            ax.plot([cx, ox], [cy, oy],
                    color=hp.drainage_line_color, linestyle=hp.drainage_line_style, linewidth=hp.drainage_line_width,
                    zorder=3)
        outlets_drawn.add(str(outlet_name))
    return drew


# ---------------------------------------------------------------------------
# Hydraulics panel — junctions + outfalls + conduits (from swmm_hydraulics.inp)
# ---------------------------------------------------------------------------


def _draw_hydraulics_panel(
    ax, hydraulics_model, hydraulics_rel: str, dem_bounds, map_cfg, prov,
) -> None:
    from matplotlib.lines import Line2D

    hp = map_cfg.hydraulics_panel
    coords_df = hydraulics_model.inp.coordinates
    junctions_df = hydraulics_model.inp.junctions
    outfalls_df = hydraulics_model.inp.outfalls
    conduits_df = hydraulics_model.inp.conduits

    legend_handles: list = []
    connected_nodes = _collect_connected_nodes(conduits_df)

    _draw_conduits_with_slope_labels(
        ax, conduits_df, junctions_df, outfalls_df, coords_df, hp,
        prov, hydraulics_rel,
    )
    if len(conduits_df) > 0:
        legend_handles.append(
            Line2D([], [], color=hp.conduit_color, linewidth=hp.conduit_linewidth,
                   label="SWMM conduits")
        )

    if len(junctions_df):
        jx = [float(coords_df.at[n, _ss.COORDS_X]) for n in junctions_df.index]
        jy = [float(coords_df.at[n, _ss.COORDS_Y]) for n in junctions_df.index]
        with prov.artist(
            axes_id="ax_hydraulics", kind="scatter",
            note=f"junctions ({len(junctions_df)})",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=hydraulics_rel, kind="junction_coords")
            a.add_swmm_channel("y", swmm_inp=hydraulics_rel, kind="junction_coords")
            ax.scatter(jx, jy, marker="o", s=hp.junction_marker_size, color=hp.junction_fill,
                       edgecolor="black", linewidths=hp.junction_marker_edgewidth, zorder=6)
        legend_handles.append(
            Line2D([], [], color=hp.junction_fill, marker="o", linestyle="None",
                   markersize=8, markeredgecolor="black", label="SWMM junction")
        )

    if len(outfalls_df):
        ox = [float(coords_df.at[n, _ss.COORDS_X]) for n in outfalls_df.index]
        oy = [float(coords_df.at[n, _ss.COORDS_Y]) for n in outfalls_df.index]
        with prov.artist(
            axes_id="ax_hydraulics", kind="scatter",
            note=f"outfalls ({len(outfalls_df)})",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=hydraulics_rel, kind="outfall_coords")
            a.add_swmm_channel("y", swmm_inp=hydraulics_rel, kind="outfall_coords")
            ax.scatter(ox, oy, marker=hp.outfall_marker, s=hp.outfall_marker_size, color=hp.outfall_fill,
                       edgecolor="black", linewidths=hp.outfall_marker_edgewidth, zorder=7)

    _draw_node_labels(ax, coords_df, junctions_df, outfalls_df, connected_nodes, hp)

    ax.set_aspect("equal")
    ax.set_xlim(dem_bounds[0], dem_bounds[2])
    ax.set_ylim(dem_bounds[1], dem_bounds[3])
    ax.set_title("Hydraulics")
    ax.legend(handles=legend_handles,
              loc=map_cfg.legend_loc, bbox_to_anchor=map_cfg.legend_bbox_to_anchor,
              ncol=2, fontsize=map_cfg.legend_fontsize, framealpha=map_cfg.legend_framealpha)


def _draw_conduits_with_slope_labels(ax, conduits_df, junctions_df, outfalls_df,
                                     coords_df, hp, prov, swmm_inp_rel: str):
    inverts: dict[str, float] = {}
    for name, row in junctions_df.iterrows():
        inverts[name] = float(row[_ss.JUNC_INVERT_ELEV])
    for name, row in outfalls_df.iterrows():
        inverts[name] = float(row[_ss.OUTFALL_INVERT_ELEV])

    for row in conduits_df.itertuples():
        if row.InletNode not in coords_df.index or row.OutletNode not in coords_df.index:
            continue
        p_in = (float(coords_df.at[row.InletNode, _ss.COORDS_X]),
                float(coords_df.at[row.InletNode, _ss.COORDS_Y]))
        p_out = (float(coords_df.at[row.OutletNode, _ss.COORDS_X]),
                 float(coords_df.at[row.OutletNode, _ss.COORDS_Y]))
        with prov.artist(
            axes_id="ax_hydraulics", kind="line2d",
            note=f"conduit {row.Index}: {row.InletNode} → {row.OutletNode}",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=swmm_inp_rel,
                               kind="conduit_coords", link_id=str(row.Index))
            a.add_swmm_channel("y", swmm_inp=swmm_inp_rel,
                               kind="conduit_coords", link_id=str(row.Index))
            ax.plot([p_in[0], p_out[0]], [p_in[1], p_out[1]],
                    color=hp.conduit_color, linewidth=hp.conduit_linewidth, zorder=4)
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
                fontsize=hp.slope_label_fontsize, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8),
            )


def _collect_connected_nodes(conduits_df) -> set[str]:
    nodes: set[str] = set()
    for row in conduits_df.itertuples():
        nodes.add(row.InletNode)
        nodes.add(row.OutletNode)
    return nodes


def _draw_node_labels(ax, coords_df, junctions_df, outfalls_df, connected_nodes, hp):
    label_offset = hp.node_label_offset
    rows = []
    for name, row in junctions_df.iterrows():
        if name not in coords_df.index:
            continue
        invert = float(row[_ss.JUNC_INVERT_ELEV])
        maxd = float(row.get(_ss.JUNC_MAX_DEPTH, 0.0))
        rim = invert + maxd
        rows.append((name, invert, rim, "junction"))
    for name, row in outfalls_df.iterrows():
        if name not in coords_df.index:
            continue
        rows.append((name, float(row[_ss.OUTFALL_INVERT_ELEV]), None, "outfall"))
    for name, invert, rim, kind in rows:
        x = float(coords_df.at[name, _ss.COORDS_X])
        y = float(coords_df.at[name, _ss.COORDS_Y])
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
            fontsize=hp.node_label_fontsize, zorder=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.75),
        )


# ---------------------------------------------------------------------------
# DEM panel — elevation raster + storm-tide BC overlay (unchanged from iter-2)
# ---------------------------------------------------------------------------


def _draw_elevation_panel(ax, dem, dem_bounds, bc_path, bc_rel, target_crs, map_cfg,
                          prov, dem_source: str):
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    dem_squeezed = dem.squeeze()
    arr = dem_squeezed.values
    valid = arr[np.isfinite(arr)]
    # Iter-10 wall-aware vmin/vmax: walls (DEM `_WALL_ELEV` ≈ 50 m) dominate
    # the cell count in the synth fixture (~95% of cells are walls), making
    # the iter-2 percentile-based bimodal detection fail (median ≈ p95 ≈ 50).
    # Replace with explicit wall detection: cells within 10% of the global
    # max are walls; vmin/vmax span only the modeled area, so the corridor's
    # interior gradient + dropoff occupy the full color range. Walls trigger
    # `cmap.set_over` (grey, "#808080") and the colorbar's `extend="max"`
    # arrow makes the modeled-area extent unambiguous on the figure.
    ep = map_cfg.elevation_panel
    if valid.size:
        arr_max = float(valid.max())
        wall_threshold = arr_max * ep.wall_threshold_fraction if arr_max > 0 else 1.0
        modeled = valid[valid < wall_threshold]
        if modeled.size > 0:
            vmin = float(modeled.min())
            vmax = float(modeled.max())
        else:
            vmin, vmax = float(valid.min()), arr_max
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    cmap = cm.get_cmap(ep.cmap).copy()
    cmap.set_over(ep.over_color)
    with prov.artist(
        axes_id="ax_dem", kind="image", note="DEM elevation raster",
    ) as a:
        a.add_xarray_channel(
            "z", dem_squeezed, source_path=dem_source,
            transform="modeled-area vmin/vmax clipping (walls → set_over grey)",
        )
        a.add_xarray_channel(
            "color", dem_squeezed, source_path=dem_source,
            transform="modeled-area vmin/vmax clipping (walls → set_over grey)",
            cmap=ep.cmap, vmin=vmin, vmax=vmax, set_over=ep.over_color,
        )
        im = ax.imshow(
            arr,
            cmap=cmap, vmin=vmin, vmax=vmax,
            extent=(dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]),
            origin="upper", aspect="equal",
        )
    cbar = plt.colorbar(im, ax=ax, shrink=ep.cbar_shrink, pad=ep.cbar_pad, extend="max")
    cbar.set_label(units.DEM_ELEV_LABEL)

    legend_handles = []

    if bc_path is not None:
        from TRITON_SWMM_toolkit.report_renderers._provenance import (
            ProvenanceRef,
        )

        bc_gdf = gpd.read_file(bc_path)
        if bc_gdf.crs is not None:
            target_crs_str = (
                target_crs.to_string()
                if hasattr(target_crs, "to_string")
                else str(target_crs)
            )
            bc_ref = ProvenanceRef(
                source_path=bc_rel if bc_rel is not None else str(bc_path),
                variable="storm_tide_boundary",
                attrs={},
                transform=f"reproject to {target_crs_str}",
            )
            with prov.artist(
                axes_id="ax_dem", kind="line2d",
                note="storm tide boundary line",
            ) as a:
                a.add_channel("x", bc_ref)
                a.add_channel("y", bc_ref)
                bc_gdf.to_crs(target_crs).plot(
                    ax=ax, color=map_cfg.bc_color, linewidth=ep.bc_line_width,
                )
            legend_handles.append(
                Line2D([], [], color=map_cfg.bc_color, linewidth=ep.bc_line_width,
                       label="Storm tide BC")
            )

    ax.set_title("TRITON DEM")
    if legend_handles:
        ax.legend(handles=legend_handles,
                  loc=map_cfg.legend_loc, bbox_to_anchor=map_cfg.legend_bbox_to_anchor,
                  ncol=1, fontsize=map_cfg.legend_fontsize, framealpha=map_cfg.legend_framealpha)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_inp_sources(analysis: TRITONSWMM_analysis) -> tuple[Path, Path]:
    """Return `(hydro_inp, hydraulics_inp)` — both required.

    When only the combined `swmm_full_inp` exists (some scenarios), it
    serves as the source for both panels — the panels still render
    cleanly but their `source_path` provenance entries cite the same
    file.
    """
    def _pick(scenario_paths) -> tuple[Path, Path]:
        full = getattr(scenario_paths, "swmm_full_inp", None)
        full_path = Path(full) if full and Path(full).exists() else None
        hydro = getattr(scenario_paths, "swmm_hydro_inp", None)
        hydro_path = Path(hydro) if hydro and Path(hydro).exists() else None
        hydraulics = getattr(scenario_paths, "swmm_hydraulics_inp", None)
        hydraulics_path = (
            Path(hydraulics) if hydraulics and Path(hydraulics).exists() else None
        )
        if hydro_path is not None and hydraulics_path is not None:
            return hydro_path, hydraulics_path
        if full_path is not None:
            return full_path, full_path
        if hydro_path is not None:
            return hydro_path, hydro_path
        if hydraulics_path is not None:
            return hydraulics_path, hydraulics_path
        raise FileNotFoundError(
            "system_overview renderer requires at least one of "
            "swmm_full_inp / swmm_hydro_inp / swmm_hydraulics_inp to exist"
        )

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


# ===========================================================================
# Plotly branch (interactive.enabled=True) — informationally congruent with
# the matplotlib branch above. Authored as the "minimum-viable Plotly port"
# step described in `Phase 2` plan doc § Pre-/design-figure congruence step.
# Transcribes matplotlib choices 1:1; interactive-UX refinement (hover field
# set, layer toggles, attribute filters, color-scale type, palette) is
# /design-figure's iteration scope.
# ===========================================================================


def _render_plotly_branch(
    output_path: Path,
    source_paths: list[Path],
    *,
    analysis_dir,
    dem,
    dem_bounds,
    hydro_model,
    hydraulics_model,
    hydro_rel: str,
    hydraulics_rel: str,
    dem_rel: str,
    bc_path,
    bc_rel,
    target_crs,
    map_cfg,
    manifest_data,
    prov,
    plotly_js_mode: str,
    dem_building_height: float | None = None,
) -> Path:
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        emit_plot_with_sources,
    )

    # Side-effect import: registers `triton_journal` Plotly template.
    from TRITON_SWMM_toolkit.report_renderers import _plotly_theme  # noqa: F401

    # Three subplots, equal aspect, shared spatial extent (DEM bounds).
    fig = make_subplots(
        rows=1, cols=3, shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=("Hydrology", "Hydraulics", "TRITON DEM"),
    )
    fig.update_layout(
        template="triton_journal",
        title="System overview — Hydrology, Hydraulics, TRITON DEM",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=80),
    )
    # Lock equal-aspect on all panels and pin to DEM bounds.
    # Axis titles label the projected CRS so readers can identify the
    # seven-digit easting/northing values; the CRS authority string is
    # appended to the middle-panel x-title to show once across the figure.
    crs_authority = (
        target_crs.to_string() if hasattr(target_crs, "to_string")
        else str(target_crs)
    )
    for col in (1, 2, 3):
        x_title = "Easting (m)" if col != 2 else f"Easting (m) — {crs_authority}"
        fig.update_xaxes(
            range=[dem_bounds[0], dem_bounds[2]],
            title_text=x_title,
            row=1, col=col,
        )
        fig.update_yaxes(
            range=[dem_bounds[1], dem_bounds[3]],
            scaleanchor=f"x{col}", scaleratio=1.0,
            title_text="Northing (m)" if col == 1 else None,
            row=1, col=col,
        )

    _draw_hydrology_panel_plotly(
        fig, hydro_model, hydro_rel, map_cfg, prov, col=1,
    )
    _draw_hydraulics_panel_plotly(
        fig, hydraulics_model, hydraulics_rel, map_cfg, prov, col=2,
    )
    _draw_elevation_panel_plotly(
        fig, dem, dem_bounds, bc_path, bc_rel, target_crs, map_cfg,
        prov, dem_source=dem_rel, col=3,
        dem_building_height=dem_building_height,
    )

    # Filter the default Plotly modebar to the buttons that map onto the
    # message of a three-panel pinned-aspect spatial figure (zoom / pan /
    # reset / static export). Drops lasso/select/spike-lines/compare-on-hover
    # which add Tab-stops without audience purpose (I3 accessibility + I5
    # interaction grammar). `toImageButtonOptions` couples the modebar
    # download to the SVG sibling produced below (I8).
    plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d", "select2d", "autoScale2d",
            "hoverCompareCartesian", "hoverClosestCartesian",
            "toggleSpikelines",
        ],
        "toImageButtonOptions": {
            "format": "svg", "filename": "system_overview", "scale": 2,
        },
    }
    html_text = pio.to_html(
        fig, include_plotlyjs=plotly_js_mode, full_html=True,
        config=plotly_config,
    )
    # Inject an accessible <title> tag so screen-readers narrating the
    # iframe-embedded figure announce the document name rather than the URL
    # (I3 accessibility floor).
    html_text = html_text.replace(
        '<head><meta charset="utf-8" /></head>',
        '<head><meta charset="utf-8" />'
        '<title>System overview — Hydrology, Hydraulics, TRITON DEM</title>'
        '</head>',
        1,
    )

    # Sibling SVG export for journal-supplement archival (I8). Best-effort:
    # Kaleido is the Plotly static-export engine and is an optional dep — if
    # missing, log and continue; the HTML deliverable remains valid.
    try:
        fig.write_image(
            output_path.with_suffix(".svg"),
            engine="kaleido", width=1400, height=500, scale=1,
        )
    except Exception as exc:  # noqa: BLE001 — Kaleido failure is non-fatal
        import logging
        logging.getLogger(__name__).warning(
            "Kaleido SVG export skipped for %s: %s",
            output_path.with_suffix(".svg"), exc,
        )

    return emit_plot_with_sources(
        html_text, output_path, source_paths,
        analysis_dir=analysis_dir,
        output_format="html",
        manifest_data=manifest_data,
        provenance=prov,
    )


def _draw_hydrology_panel_plotly(
    fig, hydro_model, hydro_rel: str, map_cfg, prov, *, col: int,
) -> None:
    hp = map_cfg.hydrology_panel
    coords_df = hydro_model.inp.coordinates
    subcatch_df = getattr(hydro_model.inp, "subcatchments", None)
    polygons_df = getattr(hydro_model.inp, "polygons", None)

    polygon_x: list[float | None] = []
    polygon_y: list[float | None] = []
    drainage_x: list[float | None] = []
    drainage_y: list[float | None] = []
    outlet_xs: list[float] = []
    outlet_ys: list[float] = []
    outlet_names: list[str] = []
    drew_any = False

    if polygons_df is not None and len(polygons_df) > 0:
        for sc_name in polygons_df.index.unique():
            rows = polygons_df.loc[[sc_name]]
            verts = list(zip(
                rows[_ss.COORDS_X].astype(float),
                rows[_ss.COORDS_Y].astype(float),
                strict=True,
            ))
            if len(verts) < 3:
                continue
            with prov.artist(
                axes_id="ax_hydro_plotly", kind="scatter_path",
                note=f"subcatchment polygon {sc_name}",
            ) as a:
                a.add_swmm_channel(
                    "x", swmm_inp=hydro_rel,
                    kind="subcatchment_polygon", node_id=str(sc_name),
                )
                a.add_swmm_channel(
                    "y", swmm_inp=hydro_rel,
                    kind="subcatchment_polygon", node_id=str(sc_name),
                )
                # Close polygon by repeating first vertex.
                xs = [v[0] for v in verts] + [verts[0][0], None]
                ys = [v[1] for v in verts] + [verts[0][1], None]
                polygon_x.extend(xs)
                polygon_y.extend(ys)
            drew_any = True

            if subcatch_df is None or sc_name not in subcatch_df.index:
                continue
            outlet_name = subcatch_df.at[sc_name, _ss.SUBCATCH_OUTLET]
            if outlet_name not in coords_df.index:
                continue
            cx = sum(v[0] for v in verts) / len(verts)
            cy = sum(v[1] for v in verts) / len(verts)
            ox = float(coords_df.at[outlet_name, _ss.COORDS_X])
            oy = float(coords_df.at[outlet_name, _ss.COORDS_Y])
            with prov.artist(
                axes_id="ax_hydro_plotly", kind="scatter_path",
                note=f"drainage line: {sc_name} → {outlet_name}",
            ) as a:
                a.add_swmm_channel(
                    "x", swmm_inp=hydro_rel,
                    kind="subcatchment_outlet", node_id=str(sc_name),
                )
                a.add_swmm_channel(
                    "y", swmm_inp=hydro_rel,
                    kind="subcatchment_outlet", node_id=str(sc_name),
                )
                drainage_x.extend([cx, ox, None])
                drainage_y.extend([cy, oy, None])
            if str(outlet_name) not in outlet_names:
                outlet_names.append(str(outlet_name))
                outlet_xs.append(ox)
                outlet_ys.append(oy)

    if drew_any:
        with prov.artist(
            axes_id="ax_hydro_plotly", kind="scatter_path",
            note="subcatchment polygons (consolidated trace; per-polygon channels registered in inner loop)",
        ):
            fig.add_trace(
                go.Scatter(
                    x=polygon_x, y=polygon_y, mode="lines",
                    line=dict(
                        color=hp.subcatchment_edge_color,
                        width=hp.subcatchment_linewidth,
                    ),
                    name="Subcatchments",
                    legendgroup="hydrology",
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )
        with prov.artist(
            axes_id="ax_hydro_plotly", kind="scatter_path",
            note="drainage lines (consolidated trace; per-line channels registered in inner loop)",
        ):
            fig.add_trace(
                go.Scatter(
                    x=drainage_x, y=drainage_y, mode="lines",
                    line=dict(
                        color=hp.drainage_line_color,
                        width=hp.drainage_line_width,
                        dash="dash" if hp.drainage_line_style in ("--", ":", "dashed") else "solid",
                    ),
                    name="Drains to",
                    legendgroup="hydrology",
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )
    if outlet_xs:
        with prov.artist(
            axes_id="ax_hydro_plotly", kind="scatter",
            note=f"subcatchment outlet markers ({len(outlet_xs)})",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=hydro_rel, kind="outlet_node_coords")
            a.add_swmm_channel("y", swmm_inp=hydro_rel, kind="outlet_node_coords")
            fig.add_trace(
                go.Scatter(
                    x=outlet_xs, y=outlet_ys, mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=max(int(hp.outlet_marker_size ** 0.5 * 2), 4),
                        color=hp.outlet_marker_fill,
                        line=dict(color="black", width=hp.outlet_marker_edgewidth),
                    ),
                    name="Subcatchment outlets",
                    legendgroup="hydrology",
                    customdata=outlet_names,
                    hovertemplate=(
                        "<b>%{customdata}</b> (outlet)<br>"
                        "Easting: %{x:.1f} m<br>Northing: %{y:.1f} m"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1, col=col,
            )


def _draw_hydraulics_panel_plotly(
    fig, hydraulics_model, hydraulics_rel: str, map_cfg, prov, *, col: int,
) -> None:
    hp = map_cfg.hydraulics_panel
    # Use the hydrology-outlet marker-size formula uniformly across both
    # panels for visual parity (matplotlib's default `junction_marker_size=70`
    # and `outfall_marker_size=100` produce visibly oversized Plotly markers
    # relative to the hydrology-panel outlets at `outlet_marker_size=22`).
    # Phase C iteration may revisit per-symbol sizing if a specialist
    # recommends emphasis differentiation.
    hp_node_size = max(
        int(map_cfg.hydrology_panel.outlet_marker_size ** 0.5 * 2), 4,
    )
    coords_df = hydraulics_model.inp.coordinates
    junctions_df = hydraulics_model.inp.junctions
    outfalls_df = hydraulics_model.inp.outfalls
    conduits_df = hydraulics_model.inp.conduits

    inverts: dict[str, float] = {}
    for name, row in junctions_df.iterrows():
        inverts[name] = float(row[_ss.JUNC_INVERT_ELEV])
    for name, row in outfalls_df.iterrows():
        inverts[name] = float(row[_ss.OUTFALL_INVERT_ELEV])

    conduit_x: list[float | None] = []
    conduit_y: list[float | None] = []
    slope_xs: list[float] = []
    slope_ys: list[float] = []
    slope_texts: list[str] = []
    # Conduit midpoint hover surface (Plotly polylines cannot mix per-feature
    # hover with None-separated coords; expose each conduit's attributes via
    # an invisible-marker midpoint trace overlaid on the polyline).
    midpoint_xs: list[float] = []
    midpoint_ys: list[float] = []
    midpoint_customdata: list[list] = []

    for row in conduits_df.itertuples():
        if (
            row.InletNode not in coords_df.index
            or row.OutletNode not in coords_df.index
        ):
            continue
        p_in = (
            float(coords_df.at[row.InletNode, _ss.COORDS_X]),
            float(coords_df.at[row.InletNode, _ss.COORDS_Y]),
        )
        p_out = (
            float(coords_df.at[row.OutletNode, _ss.COORDS_X]),
            float(coords_df.at[row.OutletNode, _ss.COORDS_Y]),
        )
        with prov.artist(
            axes_id="ax_hydraulics_plotly", kind="scatter_path",
            note=f"conduit {row.Index}: {row.InletNode} → {row.OutletNode}",
        ) as a:
            a.add_swmm_channel(
                "x", swmm_inp=hydraulics_rel,
                kind="conduit_coords", link_id=str(row.Index),
            )
            a.add_swmm_channel(
                "y", swmm_inp=hydraulics_rel,
                kind="conduit_coords", link_id=str(row.Index),
            )
            conduit_x.extend([p_in[0], p_out[0], None])
            conduit_y.extend([p_in[1], p_out[1], None])
        length_m = float(getattr(row, "Length", 0.0))
        inv_in = inverts.get(row.InletNode)
        inv_out = inverts.get(row.OutletNode)
        mid_x = (p_in[0] + p_out[0]) / 2.0
        mid_y = (p_in[1] + p_out[1]) / 2.0
        if length_m > 0 and inv_in is not None and inv_out is not None:
            slope_pct = 100.0 * (inv_in - inv_out) / length_m
            slope_xs.append(mid_x)
            slope_ys.append(mid_y)
            slope_texts.append(f"{row.Index}<br>Slope: {slope_pct:.2f}%")
        else:
            slope_pct = float("nan")
        midpoint_xs.append(mid_x)
        midpoint_ys.append(mid_y)
        midpoint_customdata.append([
            str(row.Index),
            slope_pct,
            length_m,
            float(inv_in) if inv_in is not None else float("nan"),
            float(inv_out) if inv_out is not None else float("nan"),
        ])

    if conduit_x:
        with prov.artist(
            axes_id="ax_hydraulics_plotly", kind="scatter_path",
            note="SWMM conduit lines (consolidated trace; per-conduit channels registered in inner loop)",
        ):
            fig.add_trace(
                go.Scatter(
                    x=conduit_x, y=conduit_y, mode="lines",
                    line=dict(color=hp.conduit_color, width=hp.conduit_linewidth),
                    name="SWMM conduits",
                    legendgroup="hydraulics",
                    hoverinfo="skip",
                ),
                row=1, col=col,
            )
    if midpoint_xs:
        with prov.artist(
            axes_id="ax_hydraulics_plotly", kind="scatter",
            note="conduit midpoint hover markers (consolidated trace; per-conduit channels registered in inner loop)",
        ):
            fig.add_trace(
                go.Scatter(
                    x=midpoint_xs, y=midpoint_ys, mode="markers",
                    marker=dict(size=6, color="rgba(0,0,0,0)"),
                    customdata=midpoint_customdata,
                    hovertemplate=(
                        "<b>Conduit %{customdata[0]}</b><br>"
                        "Slope: %{customdata[1]:.2f}%<br>"
                        "Length: %{customdata[2]:.1f} m<br>"
                        "Inv in: %{customdata[3]:.2f} m<br>"
                        "Inv out: %{customdata[4]:.2f} m"
                        "<extra></extra>"
                    ),
                    name="Conduit details (hover)",
                    legendgroup="hydraulics",
                    showlegend=False,
                ),
                row=1, col=col,
            )
    if slope_xs:
        # Hidden by default (legend-toggleable) — the 1000+ text labels overlap
        # at default zoom and obscure the conduit/node geometry. Click the legend
        # entry "Conduit slopes" to show them. /design-figure Phase C iteration
        # will likely replace these with hover-only display.
        with prov.artist(
            axes_id="ax_hydraulics_plotly", kind="text",
            note="conduit slope text labels (consolidated trace; per-conduit channels registered in inner loop)",
        ):
            fig.add_trace(
                go.Scatter(
                    x=slope_xs, y=slope_ys, mode="text",
                    text=slope_texts,
                    textposition="top right",
                    textfont=dict(size=hp.slope_label_fontsize),
                    name="Conduit slopes",
                    legendgroup="hydraulics_labels",
                    hoverinfo="skip",
                    showlegend=True,
                    visible="legendonly",
                ),
                row=1, col=col,
            )

    if len(junctions_df):
        jx = [float(coords_df.at[n, _ss.COORDS_X]) for n in junctions_df.index]
        jy = [float(coords_df.at[n, _ss.COORDS_Y]) for n in junctions_df.index]
        j_customdata = []
        for n in junctions_df.index:
            inv_j = float(junctions_df.at[n, _ss.JUNC_INVERT_ELEV])
            maxd = float(junctions_df.at[n, _ss.JUNC_MAX_DEPTH])
            j_customdata.append([str(n), inv_j, inv_j + maxd])
        with prov.artist(
            axes_id="ax_hydraulics_plotly", kind="scatter",
            note=f"junctions ({len(junctions_df)})",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=hydraulics_rel, kind="junction_coords")
            a.add_swmm_channel("y", swmm_inp=hydraulics_rel, kind="junction_coords")
            fig.add_trace(
                go.Scatter(
                    x=jx, y=jy, mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=hp_node_size,
                        color=hp.junction_fill,
                        line=dict(
                            color="black",
                            width=hp.junction_marker_edgewidth,
                        ),
                    ),
                    name="SWMM junction",
                    legendgroup="hydraulics",
                    customdata=j_customdata,
                    hovertemplate=(
                        "<b>Junction %{customdata[0]}</b><br>"
                        "Invert: %{customdata[1]:.2f} m<br>"
                        "Rim: %{customdata[2]:.2f} m"
                        "<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )

    if len(outfalls_df):
        ox = [float(coords_df.at[n, _ss.COORDS_X]) for n in outfalls_df.index]
        oy = [float(coords_df.at[n, _ss.COORDS_Y]) for n in outfalls_df.index]
        o_customdata = [
            [str(n), float(outfalls_df.at[n, _ss.OUTFALL_INVERT_ELEV])]
            for n in outfalls_df.index
        ]
        with prov.artist(
            axes_id="ax_hydraulics_plotly", kind="scatter",
            note=f"outfalls ({len(outfalls_df)})",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=hydraulics_rel, kind="outfall_coords")
            a.add_swmm_channel("y", swmm_inp=hydraulics_rel, kind="outfall_coords")
            fig.add_trace(
                go.Scatter(
                    x=ox, y=oy, mode="markers",
                    marker=dict(
                        symbol=_matplotlib_marker_to_plotly(hp.outfall_marker),
                        size=hp_node_size,
                        color=hp.outfall_fill,
                        line=dict(
                            color="black",
                            width=hp.outfall_marker_edgewidth,
                        ),
                    ),
                    name="SWMM outfall",
                    legendgroup="hydraulics",
                    customdata=o_customdata,
                    hovertemplate=(
                        "<b>Outfall %{customdata[0]}</b><br>"
                        "Invert: %{customdata[1]:.2f} m"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1, col=col,
            )

    # Per-node Rim/Inv labels (matplotlib branch annotates each node).
    label_xs: list[float] = []
    label_ys: list[float] = []
    label_texts: list[str] = []
    for name, row in junctions_df.iterrows():
        if name not in coords_df.index:
            continue
        invert = float(row[_ss.JUNC_INVERT_ELEV])
        maxd = float(row.get(_ss.JUNC_MAX_DEPTH, 0.0))
        rim = invert + maxd
        label_xs.append(float(coords_df.at[name, _ss.COORDS_X]))
        label_ys.append(float(coords_df.at[name, _ss.COORDS_Y]))
        label_texts.append(f"{name}<br>Rim: {rim:.2f}<br>Inv: {invert:.2f}")
    for name, row in outfalls_df.iterrows():
        if name not in coords_df.index:
            continue
        label_xs.append(float(coords_df.at[name, _ss.COORDS_X]))
        label_ys.append(float(coords_df.at[name, _ss.COORDS_Y]))
        label_texts.append(f"{name}")
    if label_xs:
        # Hidden by default (legend-toggleable) — see "Conduit slopes" above for
        # rationale. Click the legend entry "Node Rim/Inv labels" to show them.
        with prov.artist(
            axes_id="ax_hydraulics_plotly", kind="text",
            note="node Rim/Inv text labels (consolidated trace; per-node channels registered in inner loop)",
        ):
            fig.add_trace(
                go.Scatter(
                    x=label_xs, y=label_ys, mode="text",
                    text=label_texts,
                    textposition="bottom right",
                    textfont=dict(size=hp.node_label_fontsize),
                    name="Node Rim/Inv labels",
                    legendgroup="hydraulics_labels",
                    hoverinfo="skip",
                    showlegend=True,
                    visible="legendonly",
                ),
                row=1, col=col,
            )


def _draw_elevation_panel_plotly(
    fig, dem, dem_bounds, bc_path, bc_rel, target_crs, map_cfg,
    prov, *, dem_source: str, col: int,
    dem_building_height: float | None = None,
) -> None:
    ep = map_cfg.elevation_panel
    dem_squeezed = dem.squeeze()
    arr = dem_squeezed.values
    # Mask building cells (DEM gridcells whose elevation was assigned the
    # `cfg_system.dem_building_height` sentinel) BEFORE deriving vmin/vmax
    # so the colorbar range reflects ground elevations only. The walls and
    # bathtub fill are handled separately by the over-color overlay below.
    if dem_building_height is not None:
        valid_for_range = arr[
            np.isfinite(arr) & (arr != dem_building_height)
        ]
    else:
        valid_for_range = arr[np.isfinite(arr)]
    if valid_for_range.size:
        vmin, vmax, wall_threshold = _resolve_dem_color_range(
            valid_for_range, ep,
        )
    else:
        vmin, vmax, wall_threshold = 0.0, 1.0, 1.0

    # Two-heatmap overlay reproduces matplotlib's `cmap.set_over` behavior:
    # bottom layer shows modeled-area gradient (walls masked NaN); top layer
    # shows walls only as a single grey color (matplotlib's set_over("#808080")).
    arr_modeled = np.where(arr < wall_threshold, arr, np.nan)
    arr_walls = np.where(arr >= wall_threshold, 1.0, np.nan)

    # Build x/y coordinate vectors for the heatmap (cell centers).
    n_y, n_x = arr.shape
    x0, y0, x1, y1 = (
        dem_bounds[0], dem_bounds[1], dem_bounds[2], dem_bounds[3],
    )
    xs = np.linspace(x0, x1, n_x)
    # DEM is origin="upper" in matplotlib; for Plotly Heatmap with explicit
    # y vector starting at y1 and decreasing, the rows render top-to-bottom
    # to match matplotlib's `extent` + `origin="upper"` behavior.
    ys = np.linspace(y1, y0, n_y)

    cmap_plotly = _matplotlib_cmap_to_plotly_colorscale(ep.cmap)
    with prov.artist(
        axes_id="ax_dem_plotly", kind="image", note="DEM elevation raster",
    ) as a:
        a.add_xarray_channel(
            "z", dem_squeezed, source_path=dem_source,
            transform="modeled-area vmin/vmax clipping (walls → grey overlay)",
        )
        a.add_xarray_channel(
            "color", dem_squeezed, source_path=dem_source,
            transform="modeled-area vmin/vmax clipping (walls → grey overlay)",
            cmap=ep.cmap, vmin=vmin, vmax=vmax, set_over=ep.over_color,
        )
        fig.add_trace(
            go.Heatmap(
                z=arr_modeled, x=xs, y=ys,
                colorscale=cmap_plotly,
                zmin=vmin, zmax=vmax, zauto=False,
                showscale=True,
                colorbar=dict(
                    title=units.DEM_ELEV_LABEL,
                    len=ep.cbar_shrink, x=1.005,
                ),
                name="DEM elevation (modeled area)",
                hovertemplate=(
                    "Elevation: %{z:.2f} m<br>"
                    "Easting: %{x:.1f} m<br>Northing: %{y:.1f} m"
                    "<extra></extra>"
                ),
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Heatmap(
                z=arr_walls, x=xs, y=ys,
                colorscale=[[0, ep.over_color], [1, ep.over_color]],
                zmin=0.0, zmax=1.0, zauto=False,
                showscale=False,
                name="DEM walls",
                hovertemplate=(
                    "Wall / out-of-range cell<br>"
                    "Easting: %{x:.1f} m<br>Northing: %{y:.1f} m"
                    "<extra></extra>"
                ),
            ),
            row=1, col=col,
        )

    if bc_path is not None:
        from TRITON_SWMM_toolkit.report_renderers._provenance import (
            ProvenanceRef,
        )

        bc_gdf = gpd.read_file(bc_path)
        if bc_gdf.crs is not None:
            target_crs_str = (
                target_crs.to_string()
                if hasattr(target_crs, "to_string") else str(target_crs)
            )
            bc_ref = ProvenanceRef(
                source_path=bc_rel if bc_rel is not None else str(bc_path),
                variable="storm_tide_boundary",
                attrs={},
                transform=f"reproject to {target_crs_str}",
            )
            with prov.artist(
                axes_id="ax_dem_plotly", kind="scatter_path",
                note="storm tide boundary line",
            ) as a:
                a.add_channel("x", bc_ref)
                a.add_channel("y", bc_ref)
                bc_proj = bc_gdf.to_crs(target_crs)
                bc_x: list[float | None] = []
                bc_y: list[float | None] = []
                for geom in bc_proj.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    if geom.geom_type == "MultiLineString":
                        for sub in geom.geoms:
                            xs_, ys_ = sub.xy
                            bc_x.extend([float(v) for v in xs_])
                            bc_y.extend([float(v) for v in ys_])
                            bc_x.append(None)
                            bc_y.append(None)
                    elif geom.geom_type == "LineString":
                        xs_, ys_ = geom.xy
                        bc_x.extend([float(v) for v in xs_])
                        bc_y.extend([float(v) for v in ys_])
                        bc_x.append(None)
                        bc_y.append(None)
                fig.add_trace(
                    go.Scatter(
                        x=bc_x, y=bc_y, mode="lines",
                        line=dict(
                            color=map_cfg.bc_color,
                            width=ep.bc_line_width,
                        ),
                        name="Storm tide BC",
                        legendgroup="dem",
                        hovertemplate=(
                            "Storm tide boundary<br>"
                            "Easting: %{x:.1f} m<br>Northing: %{y:.1f} m"
                            "<extra></extra>"
                        ),
                    ),
                    row=1, col=col,
                )


def _matplotlib_marker_to_plotly(matplotlib_marker: str) -> str:
    """Map matplotlib marker strings to Plotly marker symbol names.

    1:1 transcription for the common matplotlib markers system_overview.py
    uses today (`s` for square, `^` for triangle-up, etc.). Falls back to
    "circle" for unrecognized markers — Plotly's default and a safe choice
    since the matplotlib branch is the one that defines the marker contract.
    """
    mapping = {
        "o": "circle",
        "s": "square",
        "^": "triangle-up",
        "v": "triangle-down",
        "D": "diamond",
        "*": "star",
        "+": "cross-thin",
        "x": "x-thin",
    }
    return mapping.get(matplotlib_marker, "circle")


def _matplotlib_cmap_to_plotly_colorscale(
    cmap_name: str, n_samples: int = 32,
) -> list[list]:
    """Sample a matplotlib named colormap into a Plotly colorscale list.

    Returns a list of `[t, "rgb(r,g,b)"]` entries with `t` in [0, 1]. This
    preserves the matplotlib visual identity (used by the static branch of
    this renderer) instead of falling back to a Plotly-named approximation.
    Important for cmaps without a Plotly equivalent — e.g., `terrain`,
    which is the system_overview default.
    """
    import matplotlib.cm as mcm

    try:
        cmap = mcm.get_cmap(cmap_name)
    except (ValueError, KeyError):
        cmap = mcm.get_cmap("viridis")
    ts = np.linspace(0.0, 1.0, n_samples)
    colorscale: list[list] = []
    for t in ts:
        r, g, b, _ = cmap(float(t))
        colorscale.append(
            [float(t), f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"]
        )
    return colorscale


def _resolve_dem_color_range(
    valid_values: "np.ndarray", ep,
    vmax_percentile: float = 90.0,
) -> tuple[float, float, float]:
    """Resolve (vmin, vmax, overlay_threshold) for the DEM heatmap.

    Approach: the dominant DEM regime is real elevations (low values). Walls
    and bathtub fill cluster well above the elevation regime — for the UVA
    bundle, real elevations cluster near 0–10 m with some high-terrain spots
    up to 80 m, while walls span 3000–9000 m and the bathtub fill is 9999 m;
    for the synth fixture real elevations max near 10 m and walls/bathtub
    are 50 m. In both cases, restricting to cells below `arr_max / 2`
    cleanly isolates the real-elevation regime, and the `vmax_percentile`
    of that subset gives a tight color range that reveals inside-area
    variation. Cells above vmax but below `arr_max / 2` (e.g., the UVA
    bundle's 43–80 m high-terrain spots) render at the colormap's max color
    via Plotly's `zmax` clip — informationally "elevated, exact value
    suppressed."

    `overlay_threshold = arr_max / 2` (decoupled from vmax) so the
    wall-overlay heatmap renders only walls + bathtub as grey, NOT high-
    but-real terrain. This differs from the matplotlib branch's strict
    `cmap.set_over` semantics (which would grey-out everything above vmax)
    — the trade is intentional: with vmax tightened for inside-area
    contrast, equating "above vmax = walls" would smear high terrain into
    the wall category.
    """
    arr_max = float(valid_values.max())
    arr_min = float(valid_values.min())
    if arr_max <= arr_min:
        return arr_min, arr_min + 1.0, arr_max + 1.0

    half_max = (arr_min + arr_max) / 2.0
    low_regime = valid_values[valid_values < half_max]
    if low_regime.size == 0:
        # All cells lie in the upper half — pathological for a DEM. Fall
        # back to `arr_max * wall_threshold_fraction` for symmetry with the
        # matplotlib branch's defensive default.
        wall_threshold = (
            arr_max * ep.wall_threshold_fraction if arr_max > 0 else 1.0
        )
        modeled = valid_values[valid_values < wall_threshold]
        if modeled.size > 0:
            return float(modeled.min()), float(modeled.max()), wall_threshold
        return arr_min, arr_max, arr_max + 1.0

    vmin = float(low_regime.min())
    vmax = float(np.percentile(low_regime, vmax_percentile))
    if vmax <= vmin:
        vmax = float(low_regime.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax, half_max
