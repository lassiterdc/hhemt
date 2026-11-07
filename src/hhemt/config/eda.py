"""EDA-loop configuration (ADR-10): which EDA plots appear in the standalone
updateable eda_report.html, and the doc's JS-bundling mode.

Attached INLINE (not as a path field) on analysis_config so it travels in
cfg_analysis.yaml automatically — Bundle.eda() reads it from the bundled cfg
with zero extra carry/repoint wiring (mirrors the post-F2 `report` inline-field
decision; config/analysis.py). Optional with a code-frozen default: an old
cfg_analysis.yaml without an `eda:` block loads cleanly and gets the default
member set.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hhemt.config.base import cfgBaseModel

#: EDA figure kinds the render refuses, shared source of truth for (a) the combine catch-all guard
#: (bundle/combined_snakefile_generator.py::_figures_of) and (b) the render-time lookup of a
#: predecessor backing artifact of a figure now under another stem (eda/_plotting.py). A listed kind in
#: ``enabled_plots`` is refused at render time like any unknown kind: a MOVED figure renders
#: under its new stem and a DROPPED kind has no within-primary replacement (F8: the
#: clean-vs-resume result lives on the hhemt-combine cross_experiment_intercomparison
#: surface). Add an entry when an EDA figure is removed or moves to another stem.
_EDA_RENAMES: dict[str, str] = {
    "eda_cross_sim_identity": "config_diff_maps",
    "b4b_clean_identity": "raw_byte_identity",
}
_EDA_DROPS: frozenset[str] = frozenset({"b4b_clean_vs_resume"})
#: Every retired FIGURE stem (rename originals + drops). The combine catch-all guard skips
#: any plots/eda/{stem}.html whose stem is in this set so a no-wipe re-render never surfaces
#: a retired figure. NEVER key an eda/*.zarr prune on this set: eda_cross_sim_identity.zarr
#: is the LIVE backing artifact for config_diff_maps (_plotting._RENDERER_BACKING_ARTIFACT).
_RETIRED_EDA_FIGURE_STEMS: frozenset[str] = frozenset(_EDA_RENAMES) | _EDA_DROPS


class eda_config(cfgBaseModel):
    """EDA-loop config: selected plots + the eda_report.html JS-bundling mode."""

    enabled_plots: list[str] = Field(
        default_factory=lambda: [
            "config_diff_maps",
        ],
        # NOTE (combine-first): eda_resume_sensitivity is a general per-primary figure that
        # requires a single primary carrying BOTH clean and resume rows (check_resume_sensitivity
        # pairs them). The synthetic compute-sensitivity experiment is run as two SEPARATE
        # single-arm primaries (clean sweep + resume sweep), so the member always SKIPS here; the
        # clean-vs-resume comparison is produced at COMBINE level (cross_experiment_intercomparison).
        # Kept in _EDA_RENDERERS and analysis.eda() as an opt-in capability for a future both-arms
        # primary; enable it explicitly via enabled_plots to render it there.
        description=(
            "Renderer-type keys of the EDA plots to render into eda_report.html, in order. The default is "
            "the single shipped first member, config_diff_maps, the config-diff visualization read from "
            "the consolidated sensitivity_datatree.zarr. The dem-resolution family is the other shipped "
            "selection and holds dem_resolution_cost_error, dem_resolution_diff_maps, "
            "dem_resolution_error_ecdf, and dem_resolution_coupling_table. Pair the family with "
            "report.reporting_set='dem_resolution' and a system.target_dem_resolution sweep. The "
            "compute-sensitivity and dem-resolution families are mutually exclusive per experiment, "
            "because config_diff_maps requires a UNIFORM grid across members and raises a named "
            "ProcessingError on a mixed-resolution primary. Membership is VALIDATED AT RENDER TIME and "
            "never at config-load. An unknown key, and a key of a figure type the render refuses, raises "
            "ValueError in render_eda_plots before any figure renders, for every key of the list that the "
            "call receives. `Bundle.eda(plots_only=True)` passes the whole list, and `analysis.eda()` "
            "passes the list once a calc member produced an artifact, so an analysis whose calc members "
            "all skipped renders no figure and refuses no key. The report-render rule passes only the keys "
            "of the figure of the rule. A cfg_analysis.yaml that lists a moved figure's earlier type is "
            "edited to list the new type. Config-load validation is deliberately omitted here, because the "
            "renderer registry `_EDA_RENDERERS` lives in eda/_plotting.py and a @field_validator import of "
            "the registry would create a config->plotting dependency edge. The render-time gate is the "
            "chosen enforcement point."
        ),
    )
    plotly_js_mode: Literal["cdn", "inline"] = Field(
        "inline",
        description=(
            "Plotly JS bundling for the standalone eda_report.html. Defaults to 'inline', against the "
            "'cdn' default of report.interactive.plotly_js_mode, so the EDA doc is self-contained and "
            "offline/archival-safe, because the doc is a portable artifact and no server-hosted report. "
            "'cdn' is available for size-sensitive online-only viewing."
        ),
    )
    dem_resolution_tolerance_m: float | None = Field(
        None,
        gt=0,
        description=(
            "Depth-error tolerance T, in meters, that the analyst declares for a DEM-resolution sweep. "
            "dem_resolution_error_ecdf draws T as a vertical line labelled with the value on the error "
            "axis. None draws no line. T comes from the accuracy requirement of the analyst and is never "
            "fitted to the data."
        ),
    )
