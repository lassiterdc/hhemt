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
        emit_plot_with_sources,
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

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=cfg.figsize_inches, layout="constrained",
    )

    # Relpaths against analysis_dir for provenance-record portability.
    analysis_root = str(Path(analysis.analysis_paths.analysis_dir).resolve())
    link_summary_rel = os.path.relpath(
        str(Path(link_summary_path).resolve()), analysis_root,
    )
    inp_rel = os.path.relpath(str(inp_path.resolve()), analysis_root)

    # Two-colormap design (iter-2 user feedback): non-overlapping single-color
    # gradations — Blues for utilization (cool / "filling up"), Reds for peak
    # flow magnitude (warm / "intensity"). `cfg.cmap` from report_config is
    # used as a fallback if user has overridden via YAML.
    UTILIZATION_CMAP = "Blues"
    PEAK_FLOW_CMAP = "Reds"
    panels = [
        (ax1, max_over_full, max_over_full_name, max_over_full_attrs,
         "max / full flow", 0.0, 1.0, UTILIZATION_CMAP, "ax_utilization"),
        (ax2, peak_flow, peak_flow_name, peak_flow_attrs,
         "peak flow (m³/s)",
         (float(cfg.vmin) if cfg.vmin is not None else 0.0),
         (float(cfg.vmax) if cfg.vmax is not None else float(peak_flow.max() or 1.0)),
         PEAK_FLOW_CMAP, "ax_peak_flow"),
    ]
    for ax, values, var_name, var_attrs, label, vmin, vmax, cmap_name, axes_id in panels:
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
                ax.plot([x1, x2], [y1, y2], color="black", linewidth=4.5,
                        solid_capstyle="round", zorder=2)
                ax.plot([x1, x2], [y1, y2], color=cmap(norm(val)), linewidth=3.0,
                        solid_capstyle="round", zorder=3)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=label, shrink=0.7, pad=0.02)
        ax.set_aspect("equal")
        ax.set_title(label)

    fig.suptitle(
        f"SWMM conduit flow — {analysis.cfg_analysis.analysis_id} — event_iloc {event_iloc}"
    )

    source_paths: list[Path] = [Path(link_summary_path), inp_path]
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
