"""Pydantic v2 models for the report_config.yaml schema."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from TRITON_SWMM_toolkit.config.base import cfgBaseModel
from TRITON_SWMM_toolkit.exceptions import ConfigurationError

_WILDCARD_SAFE = re.compile(r"^[A-Za-z0-9_.]+$")


class FigureDefaults(cfgBaseModel):
    font_family: str = Field("DejaVu Sans", description="matplotlib font.family rcParam")
    font_size: int = Field(10, description="matplotlib font.size rcParam")
    dpi: int = Field(100, description="matplotlib figure.dpi rcParam (interactive)")
    savefig_dpi: int = Field(150, description="matplotlib savefig.dpi rcParam (file output)")


class HydrologyPanelConfig(cfgBaseModel):
    """Event-hydrology side-panel formatting (per_sim_peak_flood_depth +
    per_sim_conduit_flow share this via `_hydrology_panel`).
    Defaults match the prior `_RAIN_COLOR` / `_BC_LINE_COLOR` constants in
    `_hydrology_panel.py` lines 23-24.
    """

    rain_color: str = Field(
        "#03152e",
        description=(
            "Rainfall bar color (hex). Near-black navy for maximum legibility "
            "against Plotly's white panel background; iter3 darkened past "
            "ColorBrewer Blues 9/9 (#08306b) because the thinner Plotly bars "
            "still read as pale at scale. Mid-tone alternatives like #3182bd "
            "and the deeper #08306b were both flagged as too light."
        ),
    )
    bc_line_color: str = Field("black", description="BC water-level line color.")
    bc_line_width: float = Field(1.5, description="BC line width (pt).")
    rain_ylim_min_cap: float = Field(
        1.0,
        description=(
            "Lower cap applied to the rainfall y-axis upper bound: "
            "`max(nanmax(rainfall) * 1.1, rain_ylim_min_cap)`. "
            "Verbatim hardcoded `1.0` in _hydrology_panel.py line 108."
        ),
    )
    bc_flat_threshold: float = Field(
        0.05,
        description=(
            "BC range below this is treated as 'flat' and the y-axis is "
            "expanded to ±1 around the centered integer. Verbatim hardcoded "
            "`0.05` in _hydrology_panel.py line 142."
        ),
    )
    panel_title: str = Field(
        "Event hydrology",
        description=(
            "Title for the rainfall sub-panel (also acts as the panel-stack "
            "title). Verbatim default matches `panel_title` keyword default "
            "on `draw_event_hydrology_panel`."
        ),
    )
    tick_labelsize: int = Field(7, description="Tick label fontsize (pt).")


class PerSimMapConfig(cfgBaseModel):
    """Shared map formatting across per_sim_peak_flood_depth + per_sim_conduit_flow."""

    depth_cmap: str = Field("YlGnBu")
    depth_under_color: str = Field("white")
    depth_vmin: float = Field(
        0.01,
        description="Lower bound for depth colorbar (cells below render as `under_color`).",
    )
    depth_vmax_fallback: float = Field(
        1.0,
        description="Fallback when no event has a usable summary or local max < depth_vmin.",
    )
    depth_boundaries_m: tuple[float, ...] = Field(
        (0.01, 0.05, 0.10, 0.50, 1.00),
        description="Discrete legend boundaries cited in the manifest (m).",
    )
    wse_cmap: str = Field(
        "plasma",
        description=(
            "WSE colormap. Perceptually uniform, CVD-safe. Default plasma per "
            "user preference (iter4 of Phase 3 design-figure); the Plotly "
            "branch's earlier hardcoded `cividis` is also CVD-safe and was "
            "the iter1-3 transient default, but user prefers plasma's warmer "
            "range for the WSE-on-real-terrain colorbar after iter3's "
            "building-cell exclusion (which produces WSE [1.81, 6.05] m on "
            "the norfolk fixture — well within plasma's high-contrast band)."
        ),
    )
    wse_fallback_range: tuple[float, float] = Field((0.0, 1.0))
    dry_threshold_m: float = Field(
        0.0025,
        description=(
            "Within-watershed depth-mask threshold in meters. Cells with "
            "max_wlevel_m < this value render as `dry_fill_color` (neutral "
            "grey) rather than as wet-color or transparent. Default 0.0025 m "
            "= 2.5 mm (a standard puddle-vs-sheen threshold)."
        ),
    )
    dry_fill_color: str = Field(
        "#d9d9d9",
        description=(
            "Neutral light grey fill for within-watershed dry cells. Preserves "
            "watershed shape as preattentive background context per Wilke "
            "Ch. 23."
        ),
    )
    wse_clip_quantile_upper: float = Field(
        0.99,
        description=(
            "Upper-quantile clip on WSE colorbar (computed across wetted "
            "cells). Suppresses building-on-top dry-cell artifacts that would "
            "otherwise dominate the colorbar scale and collapse usable range "
            "below 5%."
        ),
    )
    wse_clip_quantile_lower: float = Field(
        0.01,
        description=("Lower-quantile clip on WSE colorbar (computed across wetted " "cells)."),
    )
    utilization_cmap: str = Field("Blues")
    peak_flow_cmap: str = Field("Reds")
    conduit_outline_color: str = Field("black")
    conduit_outline_width: float = Field(4.5)
    conduit_value_width: float = Field(3.0)
    watershed_overlay_color: str = Field("black")
    watershed_overlay_width: float = Field(1.2)
    map_to_cbar_height_ratio: int = Field(28)
    outer_width_ratios: tuple[float, float, float] = Field((1.0, 1.0, 0.95))
    outer_wspace: float = Field(0.10)
    cbar_inner_width_ratios: tuple[float, float, float] = Field((1.0, 5.0, 1.0))
    map_tick_step: float = Field(50.0)
    axis_label_fontsize: int = Field(8)
    tick_labelsize: int = Field(7)
    fig_width_panel_pad: float = Field(
        1.02,
        description=(
            "Per_sim figure width formula: `h * (2 * map_aspect * pad + 1.0)`. "
            "Verbatim hardcoded `1.02` in per_sim_peak_flood_depth.py line 284 "
            "and per_sim_conduit_flow.py line 133."
        ),
    )
    fallback_h_inches: float = Field(
        6.0,
        description=(
            "Fallback figure height when `cfg.figsize_inches` is absent. "
            "Verbatim per_sim_peak_flood_depth.py line 279."
        ),
    )


class HydraulicsPanelStyle(cfgBaseModel):
    """system_overview hydraulics panel formatting."""

    junction_fill: str = Field("#1f77b4")
    outfall_fill: str = Field("#d62728")
    junction_marker_size: float = Field(70.0)
    junction_marker_edgewidth: float = Field(0.8)
    outfall_marker: str = Field("^")
    outfall_marker_size: float = Field(100.0)
    outfall_marker_edgewidth: float = Field(0.8)
    conduit_color: str = Field("#555555")
    conduit_linewidth: float = Field(1.2)
    slope_label_fontsize: int = Field(6)
    node_label_fontsize: int = Field(6)
    node_label_offset: tuple[int, int] = Field((8, -6))


class HydrologyMapPanelStyle(cfgBaseModel):
    """system_overview hydrology panel formatting."""

    subcatchment_edge_color: str = Field("#d62728")
    subcatchment_hatch: str = Field("////")
    subcatchment_linewidth: float = Field(1.0)
    drainage_line_color: str = Field("#1f77b4")
    drainage_line_style: str = Field("--")
    drainage_line_width: float = Field(1.0)
    outlet_marker_fill: str = Field("#1f77b4")
    outlet_marker_size: float = Field(22.0)
    outlet_marker_edgewidth: float = Field(0.5)


class ElevationPanelStyle(cfgBaseModel):
    """system_overview DEM panel formatting."""

    cmap: str = Field(
        "cividis",
        description=(
            "Perceptually-uniform CVD-safe sequential colormap for DEM "
            "elevation. Previous default 'terrain' is a cartographic-mimic "
            "palette with green-brown luminance non-monotonicity and fails "
            "deuteranope/protanope CVD simulation (per Moreland 2016 / "
            "Wilke 2019 Ch. 19). 'cividis' is the CVD-optimized viridis-"
            "family variant. Overridable per-deployment via report_config.yaml."
        ),
    )
    over_color: str = Field(
        "white",
        description=(
            "Color applied to DEM cells exceeding `wall_threshold_fraction` (buildings) "
            "AND used implicitly via `plot_bgcolor` for out-of-watershed cells (which are "
            "NaN-masked). White unifies the two non-modeled-area cell classes visually."
        ),
    )
    wall_threshold_buffer_m: float = Field(
        40.0,
        description=(
            "Subtractive buffer in meters applied to the cfg-derived wall threshold "
            "(`min(dem_building_height, dem_outside_watershed_height) - buffer`). "
            "Catches DEM cells whose coarsened elevation falls below the building "
            "sentinel but is still building-dominated due to resampling. With Norfolk's "
            "default `dem_building_height=80.0` and a 40 m buffer, threshold = 40 m, "
            "which empirically catches coarsened-but-still-building cells (Round 4 "
            "feedback: 50 m still left yellow-rendered building cells; 40 m clears them)."
        ),
    )
    wall_threshold_fraction: float = Field(
        0.9,
        description=(
            "Cells within this fraction of the DEM max are walls (recolored "
            "via `cmap.set_over`). Verbatim `0.9` system_overview.py line 506."
        ),
    )
    bc_line_width: float = Field(2.5)
    cbar_shrink: float = Field(0.7)
    cbar_pad: float = Field(0.02)


class SystemMapConfig(cfgBaseModel):
    target_epsg: int | None = Field(
        None,
        description=(
            "Target CRS for the system map and downstream per-sim renderers. "
            "Resolved via `resolve_target_crs()` precedence: this field -> "
            "system_config.crs.horizontal_epsg -> DEM .rio.crs. When None, falls through to "
            "the next precedence level."
        ),
    )
    figsize_inches: tuple[float, float] = Field((10.0, 8.0))
    fig_width_panel_pad: float = Field(
        1.1,
        description=(
            "Three-panel width formula: `max(3 * h * panel_aspect * pad, h * 1.6)`. "
            "Verbatim `1.1` system_overview.py line 73."
        ),
    )
    fig_width_min_factor: float = Field(
        1.6,
        description=("Minimum figure-width multiplier on `h`. Verbatim `1.6` " "system_overview.py line 73."),
    )
    subplots_adjust: dict = Field(
        default_factory=lambda: {"left": 0.04, "right": 0.97, "top": 0.92, "bottom": 0.20, "wspace": 0.04},
        description="`fig.subplots_adjust` kwargs — verbatim system_overview.py lines 77-78.",
    )
    legend_loc: str = Field("upper center")
    legend_bbox_to_anchor: tuple[float, float] = Field((0.5, -0.10))
    legend_fontsize: int = Field(8)
    legend_framealpha: float = Field(0.9)
    watershed_color: str = Field("red")
    dem_extent_color: str = Field("blue")
    bc_marker: str = Field("o")
    bc_color: str = Field("orange")
    swmm_node_color: str = Field("black")
    swmm_node_size: float = Field(8.0)
    swmm_link_color: str = Field("gray")
    swmm_link_width: float = Field(0.6)
    hydrology_panel: HydrologyMapPanelStyle = Field(default_factory=HydrologyMapPanelStyle)
    hydraulics_panel: HydraulicsPanelStyle = Field(default_factory=HydraulicsPanelStyle)
    elevation_panel: ElevationPanelStyle = Field(default_factory=ElevationPanelStyle)


class PerSimFigureSpec(cfgBaseModel):
    figsize_inches: tuple[float, float] = Field((10.0, 8.0))
    cmap: str = Field("viridis")
    vmin: float | None = Field(None)
    vmax: float | None = Field(None)
    vmax_quantile: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional quantile (in [0, 1]) used to derive the colorbar upper bound "
            "when `vmax` is None. When set, the renderer computes "
            "`np.nanquantile(values, vmax_quantile)` as the colorbar max. This "
            "clips out top-end outliers so the bulk of the value distribution is "
            "visible with better contrast. When both `vmax` and `vmax_quantile` "
            "are None, the renderer falls back to the absolute max. Not all "
            "renderers consume this field; check the renderer's docstring."
        ),
    )


class InteractiveBackendConfig(cfgBaseModel):
    """Top-level interactive-output toggles. Applies to ALL renderers that
    emit HTML; matplotlib-PNG renderers ignore this block."""

    enabled: bool = Field(
        True,
        description=(
            "Master switch. When False, renderers emit static PNG via the "
            "matplotlib branch of emit_plot_with_sources (legacy behavior). "
            "When True (default), renderers with an HTML emit-path emit HTML "
            "via the str branch of emit_plot_with_sources. Flipped from "
            "False to True at Phase 9 of the interactive_report_renderers PWI "
            "after Phase 8.5's cleanup_stale_metadata mechanism landed, so "
            "the first post-flip invocation against an existing analysis_dir "
            "handles the one-shot rule-rename cleanup cascade silently."
        ),
    )
    static_backend: Literal["matplotlib", "plotly"] = Field(
        "plotly",
        description=(
            "Static-figure rendering backend for renderers that have "
            "BOTH a matplotlib branch and a Plotly branch. When "
            "'plotly', renderers emit SVG via "
            "fig.write_image(engine='kaleido') as their static artifact "
            "and the matplotlib branch is bypassed. When 'matplotlib', "
            "current behavior. Renderers without a Plotly branch fall "
            "back to matplotlib regardless of this flag (with a "
            "one-time warning per renderer per process — see "
            "report_renderers/_static_backend_warning.py, introduced "
            "in Plan Phase 5). Default 'plotly' per Plan Phase 2 D3 + "
            "Decision 4: the bundle workflow's headline use case is "
            "interactive Plotly reports, so the default matches the "
            "headline experience. Users on environments without "
            "kaleido installed should either install the viz-export "
            "extra (`pip install -e '.[viz-export]'`) or set this "
            "field to 'matplotlib' in cfg_analysis.yaml."
        ),
    )
    plotly_js_mode: Literal["cdn", "inline"] = Field(
        "cdn",
        description=(
            "Plotly JS bundling. 'cdn' writes a plain <script src=\"https://"
            'cdn.plot.ly/plotly-<version>.min.js"> tag (no SRI integrity '
            "attribute — plotly.py 6.x does NOT emit SRI by default, and the "
            "per-version SHA-256 maintenance is out of scope for this "
            "renderer; see follow-up idea in scratch). Tiny HTML files, "
            "requires online viewer at view time. 'inline' embeds the full "
            "~3 MB bundle into every HTML file — works offline, large files, "
            "archival-safe."
        ),
    )
    tabulator_js_mode: Literal["cdn", "inline"] = Field(
        "cdn",
        description=(
            "Tabulator JS bundling. 'cdn' references "
            "https://cdn.jsdelivr.net/npm/tabulator-tables@6.4.0/dist/js/tabulator.min.js. "
            "'inline' embeds the ~420 KB TabulatorFull bundle."
        ),
    )
    report_html_mode: Literal["html", "zip", "auto"] = Field(
        "auto",
        description=(
            "Pass-through for analysis.render_report(format=...). 'auto' picks "
            "'html' when total report-flagged size < 15 MB else 'zip'. "
            "Snakemake's stated ceiling is ~10-20 MB total report-flagged "
            "size; base64 encoding inflates by ~33%."
        ),
    )

    @model_validator(mode="after")
    def _check_interactive_consistency(self) -> InteractiveBackendConfig:
        """Cross-field rules for InteractiveBackendConfig.

        Rule 1 (CDN-inside-ZIP orphan): if `report_html_mode == "zip"`, a CDN
        js mode produces a ZIP that, once extracted on a viewer's machine,
        requires online access for the JS bundle. The ZIP fallback exists
        specifically because Snakemake's report ceiling is ~10–20 MB; in that
        regime the user is choosing self-contained portability, so CDN-mode
        JS is incoherent with ZIP output. Reject the combination.

        Rule 2 (auto -> zip promotion when both bundles inline): retired.
        The auto→zip eager-promotion rule mutated the validated model
        post-construction, conflicting with the Pydantic field-immutability
        contract. The 15 MB auto-picker at render time in
        `analysis.render_report()` handles the inline-bundles-blow-budget
        case naturally without requiring construction-time coercion.

        Rule 3 (orphan-config when disabled): if `enabled is False`, the
        three mode fields are inert (matplotlib-PNG branch ignores them).
        We do not error — leaving non-default mode values when disabled is a
        legitimate "pre-set my preferred interactive config; flip enabled
        later" workflow.
        """
        if self.enabled and self.report_html_mode == "zip":
            if self.plotly_js_mode == "cdn":
                raise ValueError(
                    "InteractiveBackendConfig: report_html_mode='zip' is "
                    "incompatible with plotly_js_mode='cdn' — a ZIP "
                    "fallback is chosen for self-contained portability, "
                    "but CDN-mode JS requires online access at view time. "
                    "Set plotly_js_mode='inline' or report_html_mode='html'/'auto'."
                )
            if self.tabulator_js_mode == "cdn":
                raise ValueError(
                    "InteractiveBackendConfig: report_html_mode='zip' is "
                    "incompatible with tabulator_js_mode='cdn' — see above."
                )
        return self


class PerSimMapInteractiveConfig(cfgBaseModel):
    """Interactive overrides for per_sim_peak_flood_depth + per_sim_conduit_flow."""

    time_animation: bool = Field(
        True,
        description=(
            "Render flood-depth raster as a Plotly animation with frames over "
            "the time dim of max_wlevel_m + slider/play. False preserves the "
            "legacy 'max-over-time' single-frame static map."
        ),
    )
    datashader_threshold_cells: int = Field(
        25_000,
        description=(
            "Pre-rasterize via Datashader Canvas.raster() when per-frame cell "
            "count exceeds this. Below the threshold, frames go directly to "
            "go.Heatmap. Tuned during Phase 3 design-figure closeout "
            "(2026-05-17): the prior 1_000_000 default skipped Datashader on "
            "the norfolk fixture (29,542 valid cells), pushing per-figure "
            "HTML to 6.76 MB > 5 MB DoD budget; lowering to 25,000 fires the "
            "branch on any norfolk-scale and larger fixture, trims the per-"
            "figure HTML by replacing the 29,542-cell raster JSON with a "
            "512×512 datashader aggregate."
        ),
    )
    visible_layers_default: list[
        Literal[
            "depth_raster",
            "watershed_boundary",
            "swmm_conduits",
            "swmm_nodes",
            "rainfall_inset",
        ]
    ] = Field(
        default_factory=lambda: [
            "depth_raster",
            "watershed_boundary",
            "swmm_conduits",
            "swmm_nodes",
        ],
        description=(
            "Trace layers visible on initial render. User toggles via Plotly "
            "legend (click hides, double-click isolates)."
        ),
    )
    colorbar_range_lock: bool = Field(
        False,
        description=(
            "When True, colorbar range is fixed at [depth_vmin, depth_vmax_fallback] "
            "(existing PerSimMapConfig fields). When False, a Plotly RangeSlider "
            "widget under the colorbar lets the user adjust [vmin, vmax] live "
            "via Plotly.restyle."
        ),
    )


class TableInteractiveConfig(cfgBaseModel):
    """Tabulator defaults for per_analysis_summary + scenario_status_appendix."""

    pagination_size: int = Field(
        50,
        description=(
            "Rows per page (Tabulator option `paginationSize`). Set to 0 to "
            "disable pagination and use virtual-scroll over full dataset "
            "(requires height/minHeight/maxHeight to engage virtual DOM)."
        ),
    )
    visible_columns_default: list[str] | None = Field(
        None,
        description=(
            "Column slugs to mark visible:true initially. None means all " "visible. User toggles via headerMenu."
        ),
    )
    header_filter: bool = Field(
        True,
        description=(
            "Per-column header-filter input. Built-in match types =, !=, like, "
            "keywords, starts, ends, <, <=, >, >=, in, regex."
        ),
    )
    table_height: str = Field(
        "70vh",
        description=(
            "Tabulator `height` option (CSS length string). Required for the "
            "virtual DOM to engage — per tabulator-architecture: 'tables MUST "
            "have height set or virtual DOM disengages and rendering becomes "
            "slow at scale'. RowManager.initializeRenderer reads "
            "options.height once at construction; iframe-external CSS height "
            "is not seen by Tabulator. Use a CSS viewport-height ('70vh') so "
            "the iframe scales naturally."
        ),
    )
    persistence_id: str | None = Field(
        None,
        description=(
            "Tabulator localStorage key. When set, user's filter/sort/"
            "column-visibility state persists across browser reloads."
        ),
    )

    @field_validator("persistence_id")
    @classmethod
    def _persistence_id_charset(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", v):
            raise ValueError(
                f"persistence_id={v!r} must match ^[A-Za-z0-9_.\\-]+$ " f"(Tabulator localStorage-key safe)."
            )
        return v

    @field_validator("visible_columns_default")
    @classmethod
    def _visible_columns_charset(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        for col in v:
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", col):
                raise ValueError(
                    f"visible_columns_default contains {col!r} which must "
                    f"match ^[A-Za-z0-9_.\\-]+$ (Tabulator column-id safe)."
                )
        return v

    def resolve_persistence_key(self, *, renderer_name: str, analysis_id: str) -> str | None:
        """Resolve the effective Tabulator persistenceID for a renderer.

        When ``persistence_id`` is set by the user, returns
        ``f"{renderer_name}__{persistence_id}"`` so multiple renderers
        sharing this config (per_analysis_summary + scenario_status_appendix)
        don't collide on the same localStorage key. When unset, derives
        from ``analysis_id`` via the same sanitization rule as the renderer
        previously applied inline. Returns ``None`` only when both
        ``persistence_id`` is unset AND ``analysis_id`` is empty/unsafe
        after sanitization — in which case the renderer should not enable
        persistence.
        """
        from TRITON_SWMM_toolkit.report_renderers._tabulator_defaults import (
            sanitize_persistence_id,
        )

        effective = self.persistence_id or sanitize_persistence_id(analysis_id)
        if not effective or effective == "_":
            return None
        return f"{renderer_name}__{effective}"


class PerSimConfig(cfgBaseModel):
    map: PerSimMapConfig = Field(default_factory=PerSimMapConfig)
    hydrology_panel: HydrologyPanelConfig = Field(default_factory=HydrologyPanelConfig)
    peak_flood_depth: PerSimFigureSpec = Field(default_factory=PerSimFigureSpec)
    conduit_flow: PerSimFigureSpec = Field(default_factory=lambda: PerSimFigureSpec(cmap="plasma", vmax_quantile=0.95))
    interactive: PerSimMapInteractiveConfig = Field(default_factory=PerSimMapInteractiveConfig)


class PerAnalysisSummaryConfig(cfgBaseModel):
    metrics: list[
        Literal[
            "n_sims",
            "n_successful",
            "n_pending",
            "n_failed",
            "enabled_model_types",
            "sensitivity_mode",
        ]
    ] = Field(
        default_factory=lambda: [
            "n_sims",
            "n_successful",
            "n_pending",
            "n_failed",
            "enabled_model_types",
            "sensitivity_mode",
        ]
    )
    table_scale: tuple[float, float] = Field(
        (1.0, 1.5),
        description=(
            "matplotlib `Table.scale(x, y)` arguments — verbatim "
            "`table.scale(1, 1.5)` per_analysis_summary.py line 208."
        ),
    )
    figure_height_per_row_inches: float = Field(
        0.4,
        description="Per-row height contribution. Verbatim `0.4` line 150.",
    )
    figure_height_padding_inches: float = Field(
        0.7,
        description="Constant additional figure height. Verbatim `0.7` line 150.",
    )
    figure_width_inches: float = Field(8.0)
    interactive: TableInteractiveConfig = Field(default_factory=TableInteractiveConfig)


class SensitivityReportConfig(cfgBaseModel):
    independent_vars: list[str] = Field(
        ...,
        description=(
            "Column names from the sensitivity CSV. Validated at analysis.run() "
            "entry against the actual CSV columns; unknown names raise "
            "ConfigurationError. Each name must match the Snakemake-safe charset "
            "`^[A-Za-z0-9_.]+$` because it becomes a wildcard in generated rule "
            "output paths."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _rewrite_legacy_mode_key(cls, data):
        """One-cycle config-format compat: the ADR-5 ReportingSet registry
        retired SensitivityReportConfig.mode; set selection now lives on
        report_config.reporting_set. A pre-conversion report_config.yaml may
        still carry sensitivity: {mode: benchmarking, ...}. Strip the legacy key
        with an actionable DeprecationWarning rather than letting extra="forbid"
        raise an opaque extra_forbidden error. Remove next cycle."""
        if isinstance(data, dict) and "mode" in data:
            import warnings

            data = {k: v for k, v in data.items() if k != "mode"}
            warnings.warn(
                "report_config.sensitivity.mode is retired (ADR-5 ReportingSet "
                "registry). Reporting-set selection now lives on "
                "report_config.reporting_set (default 'benchmarking' for "
                "sensitivity analyses). The legacy `mode:` key is ignored this "
                "cycle and will be rejected in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        return data

    dependent_var: str = Field(
        "performance.Total",
        description=(
            "Path into the per-scenario performance summary. Default "
            "'performance.Total' uses the Total column of the restart-safe "
            "per-scenario summary. For SWMM-only sub-analyses, the renderer "
            "routes to the .rpt 'Total elapsed time' value."
        ),
    )
    aggregation: Literal["mean", "median", "min", "max"] = Field("mean")
    group_by_var: str | None = Field(
        None,
        description=(
            "Optional secondary CSV column whose values become a categorical "
            "grouping (color) on the benchmarking plot — one line/marker series "
            "per group, x=independent_var, y=dependent_var. Typical value: "
            "'run_mode'. None → single ungrouped series. Validated at run-entry "
            "against the actual CSV columns alongside independent_vars."
        ),
    )
    show_gridlines: bool = Field(
        True,
        description=(
            "When True, all panels render light-grey major-axis gridlines on x "
            "and y. Useful for reading absolute values off the panels."
        ),
    )
    # ---- Style knobs (verbatim defaults from sensitivity_benchmarking.py) --
    cpu_marker: str = Field("o", description="Verbatim `_CPU_MARKER` line 47.")
    gpu_marker: str = Field("^", description="Verbatim `_GPU_MARKER` line 48.")
    point_size: float = Field(110.0, description="Verbatim `_POINT_SIZE` line 49.")
    line_style: str = Field("--")
    line_width: float = Field(1.0)
    palette: tuple[str, ...] = Field(
        (
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#CC79A7",
            "#56B4E9",
            "#D55E00",
            "#F0E442",
            "#000000",
        ),
        description="Okabe-Ito CVD-safe palette — verbatim `_OKABE_ITO` lines 54-63.",
    )
    independent_var_labels: dict[str, str] = Field(
        default_factory=lambda: {"n_devices": "Number of Devices (CPUs or GPUs)"},
        description="Verbatim `_INDEP_VAR_LABELS` line 65.",
    )
    figsize_inches: tuple[float, float] = Field(
        (7.0, 14.0),
        description="Verbatim `figsize=(7, 14)` line 125.",
    )
    title: str = Field(
        "Wall-clock, compute-cost, speedup, and efficiency\nvs. number of devices, by run mode",
        description="Verbatim title text line 171.",
    )
    title_fontsize: int = Field(11)
    title_pad: float = Field(4.0)
    annotation_fontsize: int = Field(8)
    footnote_text: str = Field(
        "* number next to hybrid scenarios indicates number of MPI processes",
        description="Verbatim footnote line 182.",
    )
    footnote_fontsize: int = Field(7)
    ideal_line_color: str = Field("red")
    ideal_line_width: float = Field(1.0)
    gridline_color: str = Field("lightgrey")
    gridline_width: float = Field(0.5)


_HTML_TABLE_STYLE_TEMPLATE = """\
body {{ font-family: {font_family};
       padding: {body_padding_px}px; color: {body_text_color}; margin: 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: {table_font_size_px}px; }}
th, td {{ padding: {cell_padding_v_px}px {cell_padding_h_px}px; border: 1px solid {cell_border_color};
         text-align: left; vertical-align: top; }}
th {{ background-color: {primary_color}; color: {th_text_color}; font-weight: {th_font_weight}; }}
tr:nth-child(even) td {{ background-color: {row_alt_bg_color}; }}
tr:hover td {{ background-color: {row_hover_bg_color}; }}
"""


_HtmlTableStyle_DEFAULT_FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'


class _HtmlTableStyleBase(cfgBaseModel):
    """Shared HTML-table styling fields used by both sidebar/appendix HTML renderers."""

    font_family: str = Field(_HtmlTableStyle_DEFAULT_FONT_FAMILY)
    body_padding_px: int = Field(12)
    body_text_color: str = Field("#333")
    # Brand-derived defaults below are OVERRIDDEN from the resolved brand_theme at
    # analysis.run() (D-5, analysis.py); the literals here are the no-theme-context
    # fallback so a bare report_config() still constructs. th_text_color/body_text_color
    # and the ErrorsAndWarningsConfig semantic pass/fail colors are NOT theme-driven.
    primary_color: str = Field("#232D4B")
    th_text_color: str = Field("white")
    th_font_weight: int = Field(600)
    cell_border_color: str = Field("#DADADA")
    cell_padding_v_px: int = Field(6)
    cell_padding_h_px: int = Field(10)
    table_font_size_px: int = Field(13)
    row_alt_bg_color: str = Field("#F1F1EF")
    row_hover_bg_color: str = Field("#FFE4C4")


_ERRORS_AND_WARNINGS_EXTRA_CSS = """\
h2 {{ color: {primary_color}; border-bottom: {h2_border_width_px}px solid {primary_color}; padding-bottom: {h2_padding_bottom_px}px; margin-top: 0; }}
h3 {{ color: {primary_color}; margin-top: {h3_margin_top_px}px; margin-bottom: {h3_margin_bottom_px}px; }}
table {{ margin-bottom: {table_margin_bottom_px}px; }}
td.pass {{ color: {pass_text_color}; font-weight: {th_font_weight}; text-align: center; width: {passfail_cell_width_px}px; }}
td.fail {{ color: {fail_text_color}; font-weight: {th_font_weight}; text-align: center; width: {passfail_cell_width_px}px; }}
.banner {{ padding: {banner_padding_v_px}px {banner_padding_h_px}px; border-radius: {banner_border_radius_px}px; margin: 10px 0 18px;
          font-weight: {th_font_weight}; font-size: {banner_font_size_px}px; }}
.banner.pass {{ background-color: {pass_bg_color}; color: {pass_text_color}; border: 1px solid {pass_text_color}; }}
.banner.fail {{ background-color: {fail_bg_color}; color: {fail_text_color}; border: 1px solid {fail_text_color}; }}
.banner.info {{ background-color: {info_bg_color}; color: {primary_color}; border: 1px solid {primary_color}; }}
"""


class ErrorsAndWarningsConfig(_HtmlTableStyleBase):
    """HTML inline-style overrides for the errors_and_warnings sidebar."""

    pass_text_color: str = Field("#1F7A1F")
    fail_text_color: str = Field("#B11E1E")
    pass_bg_color: str = Field("#DDEEDD")
    fail_bg_color: str = Field("#F4D4D4")
    info_bg_color: str = Field("#E5EBF5")
    h2_border_width_px: int = Field(2)
    h2_padding_bottom_px: int = Field(4)
    h3_margin_top_px: int = Field(24)
    h3_margin_bottom_px: int = Field(8)
    table_margin_bottom_px: int = Field(8)
    passfail_cell_width_px: int = Field(60)
    banner_padding_v_px: int = Field(10)
    banner_padding_h_px: int = Field(14)
    banner_border_radius_px: int = Field(6)
    banner_font_size_px: int = Field(14)

    def render_inline_css(self) -> str:
        fields = self.model_dump()
        return _HTML_TABLE_STYLE_TEMPLATE.format(**fields) + _ERRORS_AND_WARNINGS_EXTRA_CSS.format(**fields)


class ScenarioStatusAppendixConfig(_HtmlTableStyleBase):
    """HTML inline-style overrides for the scenario_status_appendix renderer."""

    interactive: TableInteractiveConfig = Field(default_factory=TableInteractiveConfig)

    def render_inline_css(self) -> str:
        return _HTML_TABLE_STYLE_TEMPLATE.format(**self.model_dump(exclude={"interactive"}))


class report_config(cfgBaseModel):
    figure_defaults: FigureDefaults = Field(default_factory=FigureDefaults)
    system_map: SystemMapConfig = Field(default_factory=SystemMapConfig)
    per_sim: PerSimConfig = Field(default_factory=PerSimConfig)
    per_analysis_summary: PerAnalysisSummaryConfig = Field(default_factory=PerAnalysisSummaryConfig)
    errors_and_warnings: ErrorsAndWarningsConfig = Field(default_factory=ErrorsAndWarningsConfig)
    scenario_status_appendix: ScenarioStatusAppendixConfig = Field(default_factory=ScenarioStatusAppendixConfig)
    sensitivity: SensitivityReportConfig | None = Field(
        None,
        description=(
            "Required when the analysis is a sensitivity analysis; ignored for "
            "main analyses. Cross-field validation occurs at analysis.run() entry."
        ),
    )
    reporting_set: str = Field(
        "default",
        description=(
            "ADR-5/ADR-7 layer-3 active reporting-set selector. Names an entry in "
            "the ReportingSet registry (report_renderers/_reporting_sets.py). The "
            "sentinel 'default' resolves at analysis.run() entry to 'benchmarking' "
            "when toggle_sensitivity_analysis is True, else to the standard set. "
            "Validated against the registry at run-entry (NOT at field-construction "
            "time — that would create a config.report -> report_renderers import "
            "cycle). A future named set is selected by writing its registered name "
            "here, with no code edit (TO-8)."
        ),
    )
    interactive: InteractiveBackendConfig = Field(default_factory=InteractiveBackendConfig)


def validate_sensitivity_independent_vars(
    cfg: report_config,
    sensitivity_csv_path: Path | None,
) -> None:
    """Cross-validate report_config.sensitivity against the sensitivity CSV.

    Fail-fast semantics:
      * `cfg.sensitivity is None` AND a sensitivity CSV is present -> raise, because
        sensitivity analyses require an explicit `sensitivity:` block in
        report_config.yaml (F-I-6 / F-I-7).
      * `cfg.sensitivity` is set but no CSV present -> raise, because the cross-validation
        has nothing to check against.
      * Each `independent_vars` entry must be a column in the CSV AND match the
        Snakemake-safe charset `^[A-Za-z0-9_.]+$` (Flag 17).
    """
    import pandas as pd

    if cfg.sensitivity is None:
        if sensitivity_csv_path is not None:
            raise ConfigurationError(
                field="sensitivity",
                message=(
                    "report_config.sensitivity must be set for sensitivity analyses. "
                    f"Detected sensitivity CSV at {sensitivity_csv_path}. "
                    "Add a sensitivity: block to report_config.yaml with at least "
                    "independent_vars: [<CSV column names>]."
                ),
                config_path=None,
            )
        return

    if sensitivity_csv_path is None:
        raise ConfigurationError(
            field="sensitivity",
            message=(
                "report_config.sensitivity is set but the analysis is not a "
                "sensitivity analysis (no sensitivity CSV path)."
            ),
            config_path=None,
        )

    bad_charset = [v for v in cfg.sensitivity.independent_vars if not _WILDCARD_SAFE.match(v)]
    if bad_charset:
        raise ConfigurationError(
            field="sensitivity.independent_vars",
            message=(
                f"independent_vars contains names that violate the Snakemake-safe "
                f"charset `^[A-Za-z0-9_.]+$`: {bad_charset}. These names become "
                "wildcards in generated Snakefile rule output paths and must match "
                "the charset."
            ),
            config_path=None,
        )

    df = (
        pd.read_csv(sensitivity_csv_path)
        if sensitivity_csv_path.suffix == ".csv"
        else pd.read_excel(sensitivity_csv_path)
    )
    csv_columns = set(df.columns)
    # Derived columns the renderer constructs from CSV columns and accepts as
    # independent_vars even though they are not literal CSV columns. Keep this
    # list aligned with sensitivity_benchmarking._ensure_n_devices_column().
    derived_columns = {"n_devices"}
    missing = [v for v in cfg.sensitivity.independent_vars if v not in csv_columns and v not in derived_columns]
    if missing:
        raise ConfigurationError(
            field="sensitivity.independent_vars",
            message=(
                f"independent_vars contains names not present in sensitivity CSV "
                f"{sensitivity_csv_path}: {missing}. Available columns: "
                f"{sorted(csv_columns)}."
            ),
            config_path=None,
        )


def resolve_active_reporting_set_name(
    cfg: report_config,
    *,
    is_sensitivity: bool,
) -> str:
    """Resolve the active reporting-set NAME, CSV-free (F-B-1).

    `cfg.reporting_set == "default"` resolves to "benchmarking" when
    is_sensitivity else "default" (the standard set). A non-"default" value is
    taken verbatim. The resolved name is validated against the ReportingSet
    registry (imported LAZILY here so this module never imports report_renderers
    at module load — that would create the cycle config.report ->
    report_renderers._reporting_sets -> config.report). Returns the resolved set
    name.

    This is the CSV-free resolver: it performs no sensitivity-CSV
    cross-validation, so it is safe to call from the render-without-run() path
    (analysis.render_report's surgery block) where no sensitivity CSV is
    available. validate_active_reporting_set delegates name resolution here and
    layers the run-entry CSV cross-validation on top.
    """
    from TRITON_SWMM_toolkit.report_renderers._reporting_sets import (  # lazy: cycle-break
        REPORTING_SETS,
    )

    name = cfg.reporting_set
    if name == "default":
        name = "benchmarking" if is_sensitivity else "default"
    if name not in REPORTING_SETS:
        raise ConfigurationError(
            field="reporting_set",
            message=(
                f"report_config.reporting_set='{cfg.reporting_set}' resolves to "
                f"unknown set '{name}'. Registered sets: {sorted(REPORTING_SETS)}."
            ),
            config_path=None,
        )
    return name


def validate_active_reporting_set(
    cfg: report_config,
    *,
    is_sensitivity: bool,
    sensitivity_csv_path: Path | None,
) -> str:
    """Resolve and validate the active reporting set at analysis.run() entry.

    Delegates name resolution + registry validation to
    resolve_active_reporting_set_name (CSV-free). For a set whose validator_key
    is "benchmarking", layers the run-entry CSV cross-validation by delegating to
    validate_sensitivity_independent_vars (ADR-5: benchmarking's validation IS
    that function). Returns the resolved set name for the caller to thread to the
    renderer CLI / surgery.
    """
    from TRITON_SWMM_toolkit.report_renderers._reporting_sets import (  # lazy: cycle-break
        REPORTING_SETS,
    )

    name = resolve_active_reporting_set_name(cfg, is_sensitivity=is_sensitivity)
    if REPORTING_SETS[name].validator_key == "benchmarking":
        validate_sensitivity_independent_vars(cfg, sensitivity_csv_path)
    return name


DEFAULT_REPORT_CONFIG = report_config()


def resolve_target_crs(analysis, report_cfg: report_config):
    """Resolve the target CRS for renderers.

    Precedence (first non-None wins):
      1. report_cfg.system_map.target_epsg [DEPRECATED — removal scheduled in
         the FOLLOW-UP plan; retained one cycle for back-compat with existing
         report_config.yaml files that explicitly set this override.]
      2. analysis._system.cfg_system.crs.horizontal_epsg (canonical)
      3. analysis._system.sys_paths.dem_processed's .rio.crs (last-resort)
    """
    import pyproj
    import rioxarray as rxr

    if report_cfg.system_map.target_epsg is not None:
        return pyproj.CRS.from_epsg(report_cfg.system_map.target_epsg)

    cfg_sys = analysis._system.cfg_system
    horizontal = cfg_sys.crs.horizontal_epsg
    if horizontal is not None:
        return pyproj.CRS.from_epsg(horizontal)

    dem_path = analysis._system.sys_paths.dem_processed
    dem = rxr.open_rasterio(dem_path)
    if dem.rio.crs is not None:
        return dem.rio.crs

    raise ConfigurationError(
        field="report_cfg.system_map.target_epsg",
        message=(
            "Cannot resolve target CRS: report_cfg.system_map.target_epsg is None, "
            "system_config.crs.horizontal_epsg is None, and the processed DEM at "
            f"{dem_path} has no CRS metadata."
        ),
        config_path=None,
    )
