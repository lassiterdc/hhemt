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


class eda_config(cfgBaseModel):
    """EDA-loop config: selected plots + the eda_report.html JS-bundling mode."""

    enabled_plots: list[str] = Field(
        default_factory=lambda: ["eda_cross_sim_identity"],
        description=(
            "Renderer-kind keys of the EDA plots to render into eda_report.html, "
            "in order. Default is the single shipped first member (the cross-sim "
            "byte-identity visualization over eda/<plot_id>.zarr). Membership is "
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
