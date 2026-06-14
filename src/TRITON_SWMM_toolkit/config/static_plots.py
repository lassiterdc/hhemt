"""Publication static-plot config family (ADR-4: shared base + per-function override).

Parallel to config/report.py (different requiredness contract: report sub-models
default everything; publication models combine sensible defaults with strict-
required-no-default fields per user O-f). Style defaults are sourced from the
ADR-1 viz_vocabulary module, NOT inherited from report_config — reuse-by-shared-
vocabulary, not reuse-by-inheritance (ADR-3/ADR-4).

This plan is SCHEMA-ONLY and matplotlib-shaped (ADR-3): every field maps to a
matplotlib publication control. The matplotlib renderer that consumes these
models is the downstream reporting-system_static-plots-entrypoint-and-distribution
plan. Per ADR-4's "partly speculative" caveat (D1->Option B), this plan authors
the base + the single richest exemplar (PeakFloodDepthStaticConfig); the
remaining per-function override models + the renderer_kind->model registry are
authored by the renderer plan with the renderer in hand.
"""

from __future__ import annotations

import re
import warnings
from typing import Literal

from pydantic import Field, field_validator, model_validator

from TRITON_SWMM_toolkit.config.base import cfgBaseModel
from TRITON_SWMM_toolkit.config.viz_vocabulary import (
    FontFamily,
    FontTarget,
    MplColor,
    MplColormap,
    PanelScalePolicy,
    ValueEncodingPolicy,
    VminVmaxStrategy,
)

# ADR-2 plot-ID charset, local mirror of config/report.py:14 (D3). The whole
# canonical plot ID is segments joined by '__' with '.' within a segment
# (sa.{id}, evt.{id}); BOTH separators are inside the charset, so a single
# fullmatch over the whole string suffices and no separator stripping is needed.
# Mirrors report.py rather than importing from the LAYOUT-RELEVANT
# report_plot_ids.py (editing that file to add a charset export would trip CI
# Check B for zero benefit).
_WILDCARD_SAFE = re.compile(r"^[A-Za-z0-9_.]+$")


class CvdAdvisoryWarning(UserWarning):
    """Non-blocking advisory: a chosen colormap is outside the CVD-safe allowlist.

    Dedicated category so a deployment can silence it narrowly
    (`warnings.filterwarnings("ignore", category=CvdAdvisoryWarning)`) without
    muting unrelated UserWarnings. The set is ADVISORY, not legal — every entry
    in viz_vocabulary.MplColormap remains accepted; this only nudges toward
    perceptually-uniform CVD-safe choices (Moreland 2016 / Wilke 2019 Ch. 19).
    """


# Perceptually-uniform, CVD-safe sequential/diverging colormaps. Local module
# constant — NOT viz_vocabulary, which holds LEGAL-VALUE sets (membership
# validation), not advisory subsets (D4). Curated from the matplotlib-shipped
# perceptually-uniform family + ColorBrewer CVD-safe maps the report.py
# precedents already chose (cividis, plasma, YlGnBu, Blues, Reds).
_CVD_SAFE_COLORMAPS: frozenset[str] = frozenset(
    {
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "Blues",
        "Greens",
        "Greys",
        "Oranges",
        "Purples",
        "Reds",
        "YlGnBu",
        "YlOrRd",
        "PuBu",
        "BuPu",
        "GnBu",
        "RdBu",
        "PuOr",
        "BrBG",
    }
)


class StaticPlotBaseConfig(cfgBaseModel):
    """Cross-cutting publication fields shared by every per-function static config.

    Defined ONCE (user O-d). Per-function subclasses add only content-specific
    knobs (a SPECIFIC colormap + its under/over/bad colors). Every field maps to
    a matplotlib publication control (ADR-3).
    """

    # --- strict-required-no-default (user O-f) ---
    plot_id: str = Field(
        ...,
        description=(
            "Canonical ADR-2 plot ID; the figure-output stem and manifest plot_id. "
            "The whole ID matches ^[A-Za-z0-9_.]+$ ('.' within a segment, '__' between)."
        ),
    )

    # --- caption / panel labeling (publication) ---
    caption: str | None = Field(None, description="Figure caption text.")
    subplot_label: str | None = Field(None, description="(a)/(b)/1/2 publication panel label.")

    # --- exact figure geometry (publication: exact dimensions, no reflow) ---
    figure_width_inches: float = Field(7.0, gt=0.0)
    figure_height_inches: float = Field(5.0, gt=0.0)
    savefig_dpi: int = Field(300, gt=0, description="Publication raster dpi.")
    output_format: Literal["pdf", "svg", "ps", "eps", "pgf", "png"] = Field(
        "pdf", description="Vector publication format. matplotlib backends: pdf|svg|ps|eps|pgf; png for raster."
    )
    bbox_inches_tight: bool = Field(
        False,
        description=(
            "Publication default False: matplotlib savefig(bbox_inches='tight') "
            "REFLOWS the figure and silently overrides figure_*_inches. Set True "
            "only when exact dimensions are NOT required."
        ),
    )

    # --- typography ---
    font_family: FontFamily = Field(
        "DejaVu Sans",
        description=(
            "Validated non-empty via viz_vocabulary.FontFamily (matplotlib has NO "
            "authoritative font registry — it silently falls back to DejaVu Sans "
            "for an unavailable family, so membership cannot be checked; only "
            "non-empty is enforced). Single-source per the viz_vocabulary "
            "stipulation; FontFamily's validator imports no matplotlib."
        ),
    )
    font_sizes: dict[FontTarget, int] = Field(
        default_factory=lambda: {
            FontTarget.figure_title: 12,
            FontTarget.axis_label: 10,
            FontTarget.tick_label: 8,
            FontTarget.legend_title: 10,
            FontTarget.legend_text: 9,
            FontTarget.callout: 9,
            FontTarget.caption: 9,
            FontTarget.subplot_label: 11,
        },
        description="Per-text-element independent font sizes, keyed by the ADR-1 FontTarget enum (user O-e).",
    )

    # --- cross-panel scaling (multi-panel figures) ---
    panel_scale_policy: PanelScalePolicy = Field(
        PanelScalePolicy.shared,
        description=(
            "Shared vs independent colorbar scale ACROSS multi-panel figures. "
            "Owns the cross-panel axis exclusively (D2)."
        ),
    )

    # --- colorbar bound derivation (single colorbar, within a panel) ---
    vmin_vmax_strategy: VminVmaxStrategy = Field(
        VminVmaxStrategy.absolute,
        description=(
            "How a single colorbar's [vmin, vmax] is derived: 'absolute' uses the "
            "vmin/vmax fields (or data min/max when None); 'quantile' uses "
            "vmax_quantile. RESTRICTED to {absolute, quantile} at the base (D2): "
            "per_panel_max / shared_across_panels are cross-PANEL concerns owned "
            "by panel_scale_policy."
        ),
    )
    vmin: float | None = Field(None, description="Absolute colorbar lower bound; None -> data min.")
    vmax: float | None = Field(
        None, description="Absolute colorbar upper bound; None -> data max (or vmax_quantile when strategy='quantile')."
    )
    vmax_quantile: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Upper-quantile colorbar bound when vmin_vmax_strategy='quantile'. Clips top-end outliers.",
    )

    # --- colorbar norm + out-of-range + NaN (matplotlib-native; FQ2 completeness) ---
    colorbar_norm: Literal["linear", "log", "symlog", "boundary"] = Field(
        "linear",
        description=(
            "Explicit matplotlib norm-type selector. 'boundary' activates "
            "BoundaryNorm over colorbar_boundaries; 'log'/'symlog' map to "
            "LogNorm/SymLogNorm. The boundaries field ALONE cannot disambiguate "
            "linear-with-under/over from discrete-BoundaryNorm — this selector does (FQ2)."
        ),
    )
    colorbar_boundaries: tuple[float, ...] | None = Field(
        None,
        description="BoundaryNorm discrete bin edges (matplotlib-native). Consumed only when colorbar_norm='boundary'.",
    )
    colorbar_extend: Literal["neither", "min", "max", "both"] = Field(
        "neither",
        description=(
            "matplotlib colorbar(extend=...) — the triangular out-of-range arrow(s) "
            "that pair with set_under/set_over (FQ2)."
        ),
    )
    set_bad_color: MplColor | None = Field(
        None,
        description=(
            "matplotlib cmap.set_bad(...) — NaN/masked-cell color (FQ2). "
            "None leaves matplotlib's default (transparent)."
        ),
    )

    # --- value->encoding policy (user O-e) ---
    value_encoding_policy: ValueEncodingPolicy = Field(
        ValueEncodingPolicy.colorscale,
        description=(
            "colorscale | one_to_one_dict | value_to_shape — how a data variable maps to a visual encoding (user O-e)."
        ),
    )

    @field_validator("plot_id", mode="after")
    @classmethod
    def _plot_id_wildcard_safe(cls, v: str) -> str:
        # ADR-2: '__' separates segments, '.' separates within a segment
        # (sa.{id}, evt.{id}). BOTH separators are inside ^[A-Za-z0-9_.]+$, so a
        # full-string match suffices; no separator stripping. '-' is NOT legal.
        if not _WILDCARD_SAFE.fullmatch(v):
            raise ValueError(
                f"plot_id={v!r} must match ^[A-Za-z0-9_.]+$ (ADR-2 canonical "
                "plot-ID charset; '-' is not legal — use '.' as the within-segment "
                "separator, e.g. 'per_sim_peak_flood_depth__sa.0__evt.year.9')."
            )
        return v

    @model_validator(mode="after")
    def _restrict_vmin_vmax_strategy_to_single_colorbar(self) -> StaticPlotBaseConfig:
        # D2: per_panel_max / shared_across_panels are cross-PANEL concerns owned
        # by panel_scale_policy. Reject them on vmin_vmax_strategy so the two
        # fields can never contradict on the cross-panel axis.
        if self.vmin_vmax_strategy in (
            VminVmaxStrategy.per_panel_max,
            VminVmaxStrategy.shared_across_panels,
        ):
            raise ValueError(
                f"vmin_vmax_strategy={self.vmin_vmax_strategy.value!r} is a cross-PANEL "
                "policy and is not valid here — vmin_vmax_strategy derives a single "
                "colorbar's bounds and accepts only {'absolute', 'quantile'}. Use "
                "panel_scale_policy for cross-panel sharing (D2)."
            )
        return self

    @model_validator(mode="after")
    def _bound_derivation_inert_under_boundary_norm(self) -> StaticPlotBaseConfig:
        # Companion to D2: when colorbar_norm='boundary', matplotlib's BoundaryNorm
        # derives the mapping from colorbar_boundaries and IGNORES vmin/vmax — so a
        # quantile/absolute bound-derivation set alongside it is silently discarded.
        # Fail fast rather than render a figure whose requested bound-clipping no-ops
        # (the contradictory-colorbar-settings guard promised in Target Outcome).
        if self.colorbar_norm == "boundary" and (
            self.vmin_vmax_strategy is not VminVmaxStrategy.absolute
            or self.vmin is not None
            or self.vmax is not None
            or self.vmax_quantile is not None
        ):
            raise ValueError(
                "colorbar_norm='boundary' uses BoundaryNorm over colorbar_boundaries "
                "and ignores vmin/vmax bound-derivation — but a non-default "
                "vmin_vmax_strategy/vmin/vmax/vmax_quantile was set, which would be "
                "silently discarded. Either set colorbar_norm to a continuous norm "
                "(linear/log/symlog) to honor the bound-derivation fields, or remove "
                "the vmin/vmax/vmax_quantile/vmin_vmax_strategy overrides and rely on "
                "colorbar_boundaries."
            )
        return self

    @model_validator(mode="after")
    def _cvd_advisory(self) -> StaticPlotBaseConfig:
        # D4: NON-BLOCKING advisory. Inspect any field whose value is a colormap
        # name (duck-typed: named 'cmap'/'colormap' or ending in '_cmap') and warn
        # if outside the CVD-safe allowlist. Never raises — every
        # viz_vocabulary.MplColormap remains legal.
        for fname, value in self.__dict__.items():
            if not isinstance(value, str):
                continue
            if fname in ("cmap", "colormap") or fname.endswith("_cmap"):
                # A trailing '_r' reverses a colormap without changing its
                # perceptual-uniformity / CVD-safety, and reversed maps are a
                # standard publication choice for depth/elevation (dark = deep).
                # Strip it before the membership test so e.g. 'viridis_r' /
                # 'YlGnBu_r' do not spuriously fire the advisory.
                base_name = value[:-2] if value.endswith("_r") else value
                if base_name not in _CVD_SAFE_COLORMAPS:
                    warnings.warn(
                        f"colormap {value!r} on field {fname!r} is outside the "
                        f"CVD-safe advisory allowlist. It remains legal; consider a "
                        "perceptually-uniform CVD-safe map (viridis/plasma/cividis or "
                        "a ColorBrewer single-hue) for publication (Moreland 2016 / "
                        "Wilke 2019 Ch. 19).",
                        category=CvdAdvisoryWarning,
                        stacklevel=2,
                    )
        return self


class PeakFloodDepthStaticConfig(StaticPlotBaseConfig):
    """Publication static config for the peak-flood-depth map (exemplar override).

    The single per-function model THIS plan authors (ADR-4 "partly speculative"
    caveat, D1->Option B — the remaining 3 chart-renderer override models are
    authored by the downstream matplotlib-renderer plan with the renderer in
    hand). Adds only the SPECIFIC depth-colormap content knobs; all cross-cutting
    publication mechanics live on the base.
    """

    depth_cmap: MplColormap = Field("YlGnBu", description="Depth colormap (CVD-safe by default).")
    depth_under_color: MplColor = Field(
        "white", description="cmap.set_under(...) — cells below vmin (dry) render white."
    )
    depth_over_color: MplColor | None = Field(
        None, description="cmap.set_over(...) — cells above vmax (matplotlib-native)."
    )
