"""Per-sim renderer: peak flood depth raster from `tritonswmm_TRITON_summary['max_wlevel_m']`.

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

from TRITON_SWMM_toolkit import utils
from TRITON_SWMM_toolkit.plot_utils import plot_continuous_raster

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
    """Render peak flood depth raster for one event_iloc."""
    from TRITON_SWMM_toolkit.config.report import resolve_target_crs
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        emit_plot_with_sources,
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

    # Reproject watershed shapefile to the raster's CRS before masking. Use
    # TemporaryDirectory for exception-safe cleanup (SE F-I Flag 11). When the
    # raster carries no CRS (e.g., the synth fixture's TRITON summary zarr is
    # built from a plain numpy raster without rioxarray metadata), assume the
    # watershed and raster share a coord system and mask without reprojection.
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

    # Relpaths against analysis_dir for provenance-record portability.
    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    triton_summary_rel = os.path.relpath(
        str(Path(triton_summary_path).resolve()), analysis_root,
    )
    watershed_rel = os.path.relpath(
        str(Path(watershed_shp).resolve()), analysis_root,
    )

    fig, ax = plt.subplots(figsize=cfg.figsize_inches, layout="constrained")
    wlevel_ref = ProvenanceRef(
        source_path=triton_summary_rel,
        variable=str(wlevel_name) if wlevel_name is not None else "max_wlevel_m",
        attrs=wlevel_attrs,
        selection={"event_iloc": int(event_iloc)},
        transform="masked to watershed and depth>0",
    )
    with prov.artist(
        axes_id="ax_depth", kind="image",
        note="peak flood depth raster",
    ) as a:
        a.add_channel("z", wlevel_ref)
        a.add_channel(
            "color", wlevel_ref,
            cmap=cfg.cmap, vmin=cfg.vmin, vmax=cfg.vmax,
        )
        plot_continuous_raster(
            da_masked,
            cbar_lab="peak flood depth (m)",
            cmap=cfg.cmap,
            watershed_shapefile=watershed_shp,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            ax=ax,
        )
    # The watershed-overlay polygon artist is drawn inside `plot_continuous_raster`;
    # record its lineage as a separate `ProvenanceLog.artist(...)` block so the
    # manifest captures the shapefile source independently of the raster lineage.
    with prov.artist(
        axes_id="ax_depth", kind="patch",
        note="watershed boundary overlay",
    ) as a:
        a.add_channel(
            "x",
            ProvenanceRef(
                source_path=watershed_rel,
                variable="watershed_polygon",
                attrs={},
            ),
        )
        a.add_channel(
            "y",
            ProvenanceRef(
                source_path=watershed_rel,
                variable="watershed_polygon",
                attrs={},
            ),
        )

    ax.set_title(
        f"Peak flood depth — {analysis.cfg_analysis.analysis_id} — event_iloc {event_iloc}"
    )

    source_paths: list[Path] = [Path(triton_summary_path), Path(watershed_shp)]
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
            "wlevel_m_max": float(wlevel_m_max),
            "valid_cell_count": int(valid_cell_count),
        },
        provenance=prov,
    )
