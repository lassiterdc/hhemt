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
import rioxarray as rxr
import swmmio

if TYPE_CHECKING:
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.config.report import report_config


_JUNCTION_FILL = "#1f77b4"
_OUTFALL_FILL = "#d62728"
_DRAINAGE_LINE_COLOR = _JUNCTION_FILL
_OUTLET_MARKER_FILL = _JUNCTION_FILL  # small dots on hydrology panel


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

    _apply_rcparams(report_cfg)
    target_crs = resolve_target_crs(analysis, report_cfg)
    prov = ProvenanceLog()

    # ---- Load DEM + BC up front so the figure can be sized to data aspect.
    dem = rxr.open_rasterio(sys_paths.dem_processed)
    if dem.rio.crs is not None and dem.rio.crs != target_crs:
        dem = dem.rio.reproject(target_crs)
    dem_bounds = dem.rio.bounds()

    # Three-panel figure with shared x/y axes (all panels are maps of the same
    # spatial extent). Width sized to fit three equal-aspect panels.
    _, h = map_cfg.figsize_inches
    dem_x_extent = dem_bounds[2] - dem_bounds[0]
    dem_y_extent = dem_bounds[3] - dem_bounds[1]
    panel_aspect = dem_x_extent / dem_y_extent if dem_y_extent else 1.0
    fig_width = max(3 * h * panel_aspect * 1.1, h * 1.6)
    fig, (ax_hydro, ax_hydraulics, ax_dem) = plt.subplots(
        1, 3, figsize=(fig_width, h), sharex=True, sharey=True,
    )
    fig.subplots_adjust(left=0.04, right=0.97, top=0.92,
                        bottom=0.20, wspace=0.04)

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

    # ---- Panels --------------------------------------------------------
    _draw_hydrology_panel(
        ax_hydro, hydro_model, hydro_rel, dem_bounds, prov,
    )
    _draw_hydraulics_panel(
        ax_hydraulics, hydraulics_model, hydraulics_rel, dem_bounds, map_cfg, prov,
    )
    _draw_elevation_panel(
        ax_dem, dem, dem_bounds, bc_path, bc_rel, target_crs, map_cfg,
        prov, dem_source=dem_rel,
    )

    fig.suptitle(f"System overview — {analysis.cfg_analysis.analysis_id}")

    source_paths: list[Path] = [
        sys_paths.dem_processed,
        Path(hydro_inp),
        Path(hydraulics_inp),
    ]
    if bc_path is not None:
        source_paths.append(bc_path)

    manifest_data = _build_manifest_data(
        analysis_id=analysis.cfg_analysis.analysis_id,
        ax_hydro=ax_hydro,
        ax_hydraulics=ax_hydraulics,
        ax_dem=ax_dem,
        dem_bounds=dem_bounds,
        hydro_model=hydro_model,
        hydraulics_model=hydraulics_model,
        bc_present=bc_path is not None,
    )
    return emit_plot_with_sources(
        fig, output_path, source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        dpi=report_cfg.figure_defaults.savefig_dpi,
        manifest_data=manifest_data,
        provenance=prov,
    )


def _build_manifest_data(
    analysis_id, ax_hydro, ax_hydraulics, ax_dem, dem_bounds,
    hydro_model, hydraulics_model, bc_present: bool,
) -> dict:
    polygons_df = getattr(hydro_model.inp, "polygons", None)
    n_subcatchments = (
        int(len(polygons_df.index.unique()))
        if polygons_df is not None and len(polygons_df) > 0 else 0
    )
    return {
        "analysis_id": str(analysis_id),
        "panels": [
            {
                "name": "hydrology",
                "title": ax_hydro.get_title(),
                "axis_extents": {
                    "xlim": list(ax_hydro.get_xlim()),
                    "ylim": list(ax_hydro.get_ylim()),
                },
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
                "title": ax_hydraulics.get_title(),
                "axis_extents": {
                    "xlim": list(ax_hydraulics.get_xlim()),
                    "ylim": list(ax_hydraulics.get_ylim()),
                },
                "element_counts": {
                    "junctions": int(len(hydraulics_model.inp.junctions)),
                    "outfalls": int(len(hydraulics_model.inp.outfalls)),
                    "conduits": int(len(hydraulics_model.inp.conduits)),
                },
                "legend_labels": ["SWMM conduits", "SWMM junction"],
            },
            {
                "name": "triton_dem",
                "title": ax_dem.get_title(),
                "axis_extents": {
                    "xlim": list(ax_dem.get_xlim()),
                    "ylim": list(ax_dem.get_ylim()),
                },
                "dem_bounds": list(dem_bounds),
                "legend_labels": ["Storm tide BC"] if bc_present else [],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Hydrology panel — subcatchments + drainage lines (from swmm_hydro.inp)
# ---------------------------------------------------------------------------


def _draw_hydrology_panel(
    ax, hydro_model, hydro_rel: str, dem_bounds, prov,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    coords_df = hydro_model.inp.coordinates
    subcatch_df = getattr(hydro_model.inp, "subcatchments", None)
    polygons_df = getattr(hydro_model.inp, "polygons", None)

    legend_handles: list = []
    drew_any_subcatchment = False
    outlets_drawn: set[str] = set()

    if polygons_df is not None and len(polygons_df) > 0:
        drew_any_subcatchment = _draw_subcatchments_and_drainage_lines(
            ax, polygons_df, subcatch_df, coords_df, prov,
            axes_id="ax_hydro", swmm_inp_rel=hydro_rel,
            outlets_drawn=outlets_drawn,
        )
    if drew_any_subcatchment:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="#d62728", hatch="////",
                  label="Subcatchments")
        )
        legend_handles.append(
            Line2D([], [], color=_DRAINAGE_LINE_COLOR, linestyle="--",
                   linewidth=1.0, label="Drains to")
        )

    # Small unlabeled outlet markers — provide visible endpoints for the
    # drainage lines without duplicating the hydraulics-panel schematic.
    if outlets_drawn:
        ox = [float(coords_df.at[n, "X"]) for n in sorted(outlets_drawn)
              if n in coords_df.index]
        oy = [float(coords_df.at[n, "Y"]) for n in sorted(outlets_drawn)
              if n in coords_df.index]
        if ox:
            with prov.artist(
                axes_id="ax_hydro", kind="scatter",
                note=f"subcatchment outlet markers ({len(ox)})",
            ) as a:
                a.add_swmm_channel("x", swmm_inp=hydro_rel, kind="outlet_node_coords")
                a.add_swmm_channel("y", swmm_inp=hydro_rel, kind="outlet_node_coords")
                ax.scatter(ox, oy, marker="o", s=22, color=_OUTLET_MARKER_FILL,
                           edgecolor="black", linewidths=0.5, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(dem_bounds[0], dem_bounds[2])
    ax.set_ylim(dem_bounds[1], dem_bounds[3])
    ax.set_title("Hydrology")
    if legend_handles:
        ax.legend(handles=legend_handles,
                  loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=2, fontsize=8, framealpha=0.9)


def _draw_subcatchments_and_drainage_lines(
    ax, polygons_df, subcatch_df, coords_df, prov,
    *, axes_id: str, swmm_inp_rel: str, outlets_drawn: set[str],
) -> bool:
    from matplotlib.patches import Polygon as MplPolygon

    drew = False
    for sc_name in polygons_df.index.unique():
        rows = polygons_df.loc[[sc_name]]
        verts = list(zip(rows["X"].astype(float), rows["Y"].astype(float), strict=True))
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
                    color=_DRAINAGE_LINE_COLOR, linestyle="--", linewidth=1.0,
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

    coords_df = hydraulics_model.inp.coordinates
    junctions_df = hydraulics_model.inp.junctions
    outfalls_df = hydraulics_model.inp.outfalls
    conduits_df = hydraulics_model.inp.conduits

    legend_handles: list = []
    connected_nodes = _collect_connected_nodes(conduits_df)

    # Conduits with slope labels
    _draw_conduits_with_slope_labels(
        ax, conduits_df, junctions_df, outfalls_df, coords_df,
        prov, hydraulics_rel,
    )
    if len(conduits_df) > 0:
        legend_handles.append(
            Line2D([], [], color=map_cfg.swmm_link_color, linewidth=1.2,
                   label="SWMM conduits")
        )

    # Junctions as filled circles
    if len(junctions_df):
        jx = [float(coords_df.at[n, "X"]) for n in junctions_df.index]
        jy = [float(coords_df.at[n, "Y"]) for n in junctions_df.index]
        with prov.artist(
            axes_id="ax_hydraulics", kind="scatter",
            note=f"junctions ({len(junctions_df)})",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=hydraulics_rel, kind="junction_coords")
            a.add_swmm_channel("y", swmm_inp=hydraulics_rel, kind="junction_coords")
            ax.scatter(jx, jy, marker="o", s=70, color=_JUNCTION_FILL,
                       edgecolor="black", linewidths=0.8, zorder=6)
        legend_handles.append(
            Line2D([], [], color=_JUNCTION_FILL, marker="o", linestyle="None",
                   markersize=8, markeredgecolor="black", label="SWMM junction")
        )

    # Outfalls: upward triangle, NO legend entry (per iteration-4 historical feedback).
    if len(outfalls_df):
        ox = [float(coords_df.at[n, "X"]) for n in outfalls_df.index]
        oy = [float(coords_df.at[n, "Y"]) for n in outfalls_df.index]
        with prov.artist(
            axes_id="ax_hydraulics", kind="scatter",
            note=f"outfalls ({len(outfalls_df)})",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=hydraulics_rel, kind="outfall_coords")
            a.add_swmm_channel("y", swmm_inp=hydraulics_rel, kind="outfall_coords")
            ax.scatter(ox, oy, marker="^", s=100, color=_OUTFALL_FILL,
                       edgecolor="black", linewidths=0.8, zorder=7)

    # Node labels
    _draw_node_labels(ax, coords_df, junctions_df, outfalls_df, connected_nodes)

    ax.set_aspect("equal")
    ax.set_xlim(dem_bounds[0], dem_bounds[2])
    ax.set_ylim(dem_bounds[1], dem_bounds[3])
    ax.set_title("Hydraulics")
    ax.legend(handles=legend_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, fontsize=8, framealpha=0.9)


def _draw_conduits_with_slope_labels(ax, conduits_df, junctions_df, outfalls_df,
                                     coords_df, prov, swmm_inp_rel: str):
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
        with prov.artist(
            axes_id="ax_hydraulics", kind="line2d",
            note=f"conduit {row.Index}: {row.InletNode} → {row.OutletNode}",
        ) as a:
            a.add_swmm_channel("x", swmm_inp=swmm_inp_rel,
                               kind="conduit_coords", link_id=str(row.Index))
            a.add_swmm_channel("y", swmm_inp=swmm_inp_rel,
                               kind="conduit_coords", link_id=str(row.Index))
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
    if valid.size:
        arr_max = float(valid.max())
        wall_threshold = arr_max * 0.9 if arr_max > 0 else 1.0
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

    cmap = cm.get_cmap("terrain").copy()
    cmap.set_over("#808080")
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
            cmap="terrain", vmin=vmin, vmax=vmax, set_over="#808080",
        )
        im = ax.imshow(
            arr,
            cmap=cmap, vmin=vmin, vmax=vmax,
            extent=(dem_bounds[0], dem_bounds[2], dem_bounds[1], dem_bounds[3]),
            origin="upper", aspect="equal",
        )
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, extend="max")
    cbar.set_label("Elevation (m)")

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
                    ax=ax, color=map_cfg.bc_color, linewidth=2.5,
                )
            legend_handles.append(
                Line2D([], [], color=map_cfg.bc_color, linewidth=2.5,
                       label="Storm tide BC")
            )

    ax.set_title("TRITON DEM")
    if legend_handles:
        ax.legend(handles=legend_handles,
                  loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=1, fontsize=8, framealpha=0.9)


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
