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

from pydantic import Field, model_validator

from hhemt.config.base import cfgBaseModel

#: Retired EDA figure kinds, shared source of truth for (a) the config-load self-heal
#: below and (b) the combine catch-all guard (bundle/combined_snakefile_generator.py::
#: _figures_of). A RENAMED kind keeps rendering under its new stem; a DROPPED kind has no
#: within-master replacement (F8: the clean-vs-resume result lives on the hhemt-combine
#: cross_experiment_intercomparison surface). Add an entry when an EDA figure is retired.
_EDA_RENAMES: dict[str, str] = {"eda_cross_sim_identity": "config_diff_maps"}
_EDA_DROPS: frozenset[str] = frozenset({"b4b_clean_vs_resume"})
#: Every retired FIGURE stem (rename originals + drops). The combine catch-all guard skips
#: any plots/eda/{stem}.html whose stem is in this set so a no-wipe re-render never surfaces
#: a retired figure. NEVER key an eda/*.zarr prune on this set: eda_cross_sim_identity.zarr
#: is the LIVE backing artifact for config_diff_maps (_plotting._RENDERER_BACKING_ARTIFACT).
_RETIRED_EDA_FIGURE_STEMS: frozenset[str] = frozenset(_EDA_RENAMES) | _EDA_DROPS


class eda_config(cfgBaseModel):
    """EDA-loop config: selected plots + the eda_report.html JS-bundling mode."""

    @model_validator(mode="before")
    @classmethod
    def _rewrite_legacy_eda_plot_kind(cls, data):
        """One-cycle config-format compat: the config-diff EDA plot was renamed
        eda_cross_sim_identity -> config_diff_maps (the on-disk figure stem +
        _EDA_RENDERERS key). A pre-rename cfg_analysis.yaml (or a baked render
        bundle) still carries enabled_plots: [eda_cross_sim_identity]. Rewrite
        the legacy key to config_diff_maps with an actionable DeprecationWarning
        so Bundle.eda() renders instead of raising the render_eda_plots
        "unknown EDA plot kind" ValueError. Pure string rewrite on the raw dict
        (no _EDA_RENDERERS import) so no config->plotting dependency edge.
        Mirrors config/report.py::_rewrite_legacy_mode_key (Gotcha-45).
        Remove next cycle."""
        if isinstance(data, dict) and isinstance(data.get("enabled_plots"), list):
            plots = data["enabled_plots"]
            present_retired = [p for p in plots if p in _RETIRED_EDA_FIGURE_STEMS]
            if present_retired:
                import warnings

                rewritten = [_EDA_RENAMES.get(p, p) for p in plots if p not in _EDA_DROPS]
                data = {**data, "enabled_plots": rewritten}
                renamed = sorted({p for p in present_retired if p in _EDA_RENAMES})
                dropped = sorted({p for p in present_retired if p in _EDA_DROPS})
                warnings.warn(
                    "eda.enabled_plots carried retired EDA figure kind(s): "
                    f"renamed {renamed} (eda_cross_sim_identity -> config_diff_maps); "
                    f"dropped {dropped} (F8: no within-master replacement — the clean-vs-"
                    "resume result lives on the hhemt-combine cross_experiment_intercomparison "
                    "surface). The legacy keys are rewritten this cycle and will be rejected "
                    "in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        return data

    enabled_plots: list[str] = Field(
        default_factory=lambda: [
            "config_diff_maps",
        ],
        # NOTE (combine-first): eda_resume_sensitivity is a general per-master figure that
        # requires a single master carrying BOTH clean and resume rows (check_resume_sensitivity
        # pairs them). The synthetic compute-sensitivity experiment is run as two SEPARATE
        # single-arm masters (clean sweep + resume sweep), so the member always SKIPS here; the
        # clean-vs-resume comparison is produced at COMBINE level (cross_experiment_intercomparison).
        # Kept in _EDA_RENDERERS and analysis.eda() as an opt-in capability for a future both-arms
        # master; enable it explicitly via enabled_plots to render it there.
        description=(
            "Renderer-kind keys of the EDA plots to render into eda_report.html, "
            "in order. Default is the single shipped first member (config_diff_maps, "
            "the config-diff visualization read from the consolidated "
            "sensitivity_datatree.zarr). The dem-resolution family is the other "
            "shipped selection: dem_resolution_cost_error, dem_resolution_diff_maps, "
            "dem_resolution_error_ecdf, dem_resolution_coupling_table -- pair it with "
            "report.reporting_set='dem-resolution' and a system.target_dem_resolution "
            "sweep. The two families are mutually exclusive per experiment: "
            "config_diff_maps requires a UNIFORM grid across sub-analyses and raises "
            "a named ProcessingError on a mixed-resolution master. Membership is "
            "VALIDATED AT RENDER TIME (not config-load): an unknown key raises "
            "ValueError in render_eda_plots, which fails fast at the eda() facade "
            "boundary. Config-load validation is deliberately NOT done here — the "
            "renderer registry (_EDA_RENDERERS) lives in eda/_plotting.py, and a "
            "@field_validator import of it would create a config->plotting "
            "dependency edge; the render-time gate is the chosen enforcement point."
        ),
    )
    plotly_js_mode: Literal["cdn", "inline"] = Field(
        "inline",
        description=(
            "Plotly JS bundling for the standalone eda_report.html. Defaults to "
            "'inline' (vs report.interactive.plotly_js_mode's 'cdn' default) so the "
            "EDA doc is self-contained and offline/archival-safe — it is a portable "
            "artifact, not a server-hosted report. 'cdn' available for size-sensitive "
            "online-only viewing."
        ),
    )
    tabulator_js_mode: Literal["cdn", "inline"] = Field(
        "cdn",
        description=(
            "Tabulator JS bundling for the EDA datasets table in eda_report.html. "
            "Defaults to 'cdn' (INTERIM, per DECISION-1 Option A / SPAWN): the "
            "toolkit's inline-Tabulator path is an unimplemented stub today "
            "(_tabulator_defaults.build_html_document(js_mode='inline') warns + "
            "falls back to CDN). Completing inline-Tabulator toolkit-wide is the "
            "SEPARATE reporting-system_inline-tabulator plan (scoped under the "
            "reporting-system system design); when it lands, this default flips to "
            "'inline' and the EDA table becomes fully offline-safe. Note: 'inline' "
            "is NOT yet functional — it currently still resolves to CDN."
        ),
    )
