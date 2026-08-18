"""One scrolling results page per scenario, sectioned by model arm (Piece 2).

Answers the user's requirement directly: one tab per event, every applicable
plot on that page, organised under model headers, no click between conduit
results and peak-flood results.

Why this is ONE `report()` output rather than a subcategory grouping of N
figures: Snakemake's report viewer shows one figure at a time -- which is why
`_react_surgery` carries an auto-pop-first-figure shim and a row-click shim at
all -- so no arrangement of `category`/`subcategory` produces a scroll. The page
must therefore be a single self-contained document whose INTERNAL structure
carries the headers.

Why the model axis is inside this renderer rather than a rule wildcard: a
per-model wildcard would emit N files per event that nothing consumes and then
embed their content a second time here. The loop belongs where the page is.

Plotly-only by construction (ADR-3 boundary): the `go.Figure` builder seams this
calls exist only on each renderer's plotly branch, and `render_scrollable_report`
is `pio.to_html` end to end. `analysis.run()` rejects the matplotlib backend for
this renderer at entry rather than silently falling back -- see the run-entry
guard -- because a user who configured matplotlib and received the legacy
two-row layout would reasonably believe they had the page they asked for.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis
    from hhemt.config.report import report_config


def render(
    analysis: TRITONSWMM_analysis,
    report_cfg: report_config,
    output_path: Path,
    *,
    event_iloc: int | None = None,
    **kwargs,
) -> Path:
    """Compose every applicable model arm's per-sim figure into one page.

    `**kwargs` tolerates dispatcher-passed keywords this renderer does not
    consume, matching the uniform renderer signature stipulation.
    """
    from hhemt.eda._report import FigureSpec, render_scrollable_report
    from hhemt.report_renderers._figure_emission import emit_plot_with_sources
    from hhemt.report_renderers._model_arms import MODEL_DISPLAY_NAMES, page_sections
    from hhemt.report_renderers._provenance import ProvenanceLog
    from hhemt.report_renderers.per_sim_conduit_flow import (
        _build_conduit_flow_figure,
        _emit_model_type_skip_placeholder,
    )
    from hhemt.report_renderers.per_sim_peak_flood_depth import _build_peak_flood_depth_figure
    from hhemt.report_renderers.system_overview import _apply_rcparams

    _apply_rcparams(report_cfg)
    prov = ProvenanceLog()

    _RENDERER_LABELS = {
        "peak_flood_depth": "Peak flood depth",
        "conduit_flow": "Conduit flow",
    }

    sections = page_sections(analysis._get_enabled_model_types())
    if not sections:
        return _emit_model_type_skip_placeholder(
            output_path,
            "no per-simulation figures are applicable to this analysis",
            report_cfg.figure_defaults.savefig_dpi,
        )

    figures: list[FigureSpec] = []
    for model_type, renderer_kind, group in sections:
        if renderer_kind == "peak_flood_depth":
            built = _build_peak_flood_depth_figure(
                analysis,
                report_cfg,
                output_path,
                event_iloc=event_iloc,
                triton_group=group,
                prov=prov,
            )
        else:
            built = _build_conduit_flow_figure(
                analysis,
                report_cfg,
                output_path,
                event_iloc=event_iloc,
                link_group=group,
                prov=prov,
            )
        # Both builders are Path-polymorphic: on a soft-skip (partial-completion
        # tree) they return the placeholder Path they already wrote rather than a
        # tuple. Skipping the SECTION rather than the PAGE is the right degradation
        # -- one unavailable arm must not blank the other three.
        if isinstance(built, Path):
            continue
        figures.append(
            FigureSpec(
                figure=built[0],
                title=f"{MODEL_DISPLAY_NAMES[model_type]} - {_RENDERER_LABELS[renderer_kind]}",
            )
        )

    if not figures:
        return _emit_model_type_skip_placeholder(
            output_path,
            "per-simulation figures are not yet available for this scenario",
            report_cfg.figure_defaults.savefig_dpi,
        )

    interactive = report_cfg.interactive
    html_text = render_scrollable_report(
        figures,
        [],
        title=f"Simulation results - event {event_iloc}",
        brand=None,
        plotly_js_mode=interactive.plotly_js_mode,
        tabulator_js_mode=interactive.tabulator_js_mode,
    )

    # Declared sources are the UNION over every section, written explicitly
    # rather than harvested from the builders' return tuples by index -- the
    # renderer-IO audit (Gotcha 53) compares declared against ACTUAL reads, and
    # an index-derived list would silently drift when a builder's tuple changes.
    sys_paths = analysis._system.sys_paths
    proc = analysis._retrieve_sim_run_processing_object(event_iloc)
    source_paths = [
        analysis.analysis_paths.analysis_datatree_zarr,
        Path(sys_paths.dem_processed),
        Path(analysis.cfg_analysis.weather_timeseries),
    ]
    watershed = analysis._system.cfg_system.watershed_gis_polygon
    if watershed:
        source_paths.append(Path(watershed))
    inp = getattr(proc.scen_paths, "swmm_hydraulics_inp", None) or proc.scen_paths.swmm_full_inp
    if inp:
        source_paths.append(Path(inp))

    return emit_plot_with_sources(
        html_text,
        output_path,
        source_paths,
        analysis_dir=analysis.analysis_paths.analysis_dir,
        output_format="html",
        manifest_data={
            "event_iloc": int(event_iloc) if event_iloc is not None else None,
            "sections": [{"model_type": m, "renderer_kind": k, "datatree_group": g} for m, k, g in sections],
            "figure_count": len(figures),
        },
        provenance=prov,
        emit_preview=False,
    )
