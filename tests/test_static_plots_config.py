"""Unit tests for the publication static-plot config family (config/static_plots.py)."""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from hhemt.config.static_plots import (
    _CVD_SAFE_COLORMAPS,
    STATIC_PLOT_CONFIG_REGISTRY,
    ConduitFlowStaticConfig,
    CvdAdvisoryWarning,
    PeakFloodDepthStaticConfig,
    SensitivityBenchmarkingStaticConfig,
    StaticPlotBaseConfig,
    SystemOverviewStaticConfig,
)
from hhemt.config.viz_vocabulary import FontTarget, VminVmaxStrategy

_VALID_PLOT_ID = "per_sim_peak_flood_depth__evt.year.9"
_RENDERER_KIND = "per_sim_peak_flood_depth"


def _base(**overrides):
    kw = {"plot_id": _VALID_PLOT_ID, "renderer_kind": _RENDERER_KIND}
    kw.update(overrides)
    return kw


# R2 — plot_id strict-required
def test_plot_id_required():
    with pytest.raises(ValidationError):
        StaticPlotBaseConfig()


# R3 — plot_id charset
def test_plot_id_rejects_hyphen():
    with pytest.raises(ValidationError):
        StaticPlotBaseConfig(plot_id="per_sim_peak_flood_depth__sa-0", renderer_kind=_RENDERER_KIND)


def test_plot_id_accepts_adr2_dot_form():
    cfg = StaticPlotBaseConfig(plot_id="per_sim_peak_flood_depth__sa.0__evt.year.9", renderer_kind=_RENDERER_KIND)
    assert cfg.plot_id == "per_sim_peak_flood_depth__sa.0__evt.year.9"


# R7 (renderer plan) — renderer_kind strict-required + registry
def test_renderer_kind_required():
    with pytest.raises(ValidationError):
        StaticPlotBaseConfig(plot_id=_VALID_PLOT_ID)


def test_registry_maps_exemplar_kind_to_subclass():
    assert STATIC_PLOT_CONFIG_REGISTRY[_RENDERER_KIND] is PeakFloodDepthStaticConfig
    assert all(issubclass(model, StaticPlotBaseConfig) for model in STATIC_PLOT_CONFIG_REGISTRY.values())


# R5 — output_format Literal
def test_output_format_rejects_unknown():
    with pytest.raises(ValidationError):
        StaticPlotBaseConfig(**_base(output_format="jpeg"))


def test_output_format_default_pdf():
    assert StaticPlotBaseConfig(**_base()).output_format == "pdf"


# R6 — bbox_inches_tight default False
def test_bbox_inches_tight_default_false():
    assert StaticPlotBaseConfig(**_base()).bbox_inches_tight is False


# R7 — D2 vmin_vmax_strategy restriction
@pytest.mark.parametrize("strat", [VminVmaxStrategy.per_panel_max, VminVmaxStrategy.shared_across_panels])
def test_vmin_vmax_strategy_rejects_cross_panel(strat):
    with pytest.raises(ValidationError):
        StaticPlotBaseConfig(**_base(vmin_vmax_strategy=strat))


@pytest.mark.parametrize("strat", [VminVmaxStrategy.absolute, VminVmaxStrategy.quantile])
def test_vmin_vmax_strategy_accepts_single_colorbar(strat):
    assert StaticPlotBaseConfig(**_base(vmin_vmax_strategy=strat)).vmin_vmax_strategy == strat


# R11 — font_sizes
def test_font_sizes_default_all_targets():
    assert set(StaticPlotBaseConfig(**_base()).font_sizes) == set(FontTarget)


def test_font_sizes_accepts_partial():
    cfg = StaticPlotBaseConfig(**_base(font_sizes={FontTarget.figure_title: 14}))
    assert cfg.font_sizes == {FontTarget.figure_title: 14}


# extra=forbid (inherited from cfgBaseModel)
def test_extra_forbidden():
    with pytest.raises(ValidationError):
        StaticPlotBaseConfig(**_base(not_a_field=1))


# R10 — exemplar inherits base + adds content knobs
def test_exemplar_inherits_and_extends():
    cfg = PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID, renderer_kind=_RENDERER_KIND)
    assert cfg.bbox_inches_tight is False
    assert cfg.output_format == "pdf"
    assert cfg.depth_cmap == "YlGnBu"
    assert cfg.depth_under_color == "white"
    assert cfg.depth_over_color is None


# Phase 2 — system_overview per-function model + registry entry
_SYSOV_PLOT_ID = "system_overview"
_SYSOV_RENDERER_KIND = "system_overview"


def test_registry_maps_system_overview_kind_to_subclass():
    assert STATIC_PLOT_CONFIG_REGISTRY[_SYSOV_RENDERER_KIND] is SystemOverviewStaticConfig
    assert issubclass(SystemOverviewStaticConfig, StaticPlotBaseConfig)


def test_system_overview_inherits_and_extends():
    # dem_cmap default "terrain" is non-CVD-safe -> the base _cvd_advisory fires a
    # (non-blocking) warning on construction; suppress it for the field-default asserts.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CvdAdvisoryWarning)
        cfg = SystemOverviewStaticConfig(
            plot_id=_SYSOV_PLOT_ID, renderer_kind=_SYSOV_RENDERER_KIND
        )
    assert cfg.bbox_inches_tight is False  # inherited base default
    assert cfg.output_format == "pdf"  # inherited base default
    assert cfg.dem_cmap == "terrain"  # added content knob
    assert cfg.dem_over_color is None  # added content knob


def test_system_overview_default_cmap_fires_cvd_advisory():
    with pytest.warns(CvdAdvisoryWarning):
        SystemOverviewStaticConfig(
            plot_id=_SYSOV_PLOT_ID, renderer_kind=_SYSOV_RENDERER_KIND
        )


# Phase 3 — per_sim_conduit_flow per-function model + registry entry
_CONDUIT_PLOT_ID = "per_sim_conduit_flow__evt.year.9"
_CONDUIT_RENDERER_KIND = "per_sim_conduit_flow"


def test_registry_maps_conduit_flow_kind_to_subclass():
    assert STATIC_PLOT_CONFIG_REGISTRY[_CONDUIT_RENDERER_KIND] is ConduitFlowStaticConfig
    assert issubclass(ConduitFlowStaticConfig, StaticPlotBaseConfig)


def test_conduit_flow_inherits_and_extends():
    cfg = ConduitFlowStaticConfig(plot_id=_CONDUIT_PLOT_ID, renderer_kind=_CONDUIT_RENDERER_KIND)
    assert cfg.bbox_inches_tight is False  # inherited base default
    assert cfg.output_format == "pdf"  # inherited base default
    assert cfg.utilization_cmap == "YlOrRd"  # added content knob (fixed [0,1] panel)
    assert cfg.peak_flow_cmap == "viridis"  # added content knob (data-ranged panel)
    assert cfg.peak_flow_vmax is None  # added content knob; None -> data max


def test_conduit_flow_default_cmaps_are_cvd_safe_silent():
    # Both defaults (YlOrRd utilization, viridis peak-flow) are in the CVD-safe
    # allowlist, so construction with defaults fires NO advisory (contrast with
    # system_overview's "terrain" default which does).
    assert "YlOrRd" in _CVD_SAFE_COLORMAPS
    assert "viridis" in _CVD_SAFE_COLORMAPS
    with warnings.catch_warnings():
        warnings.simplefilter("error", CvdAdvisoryWarning)
        ConduitFlowStaticConfig(plot_id=_CONDUIT_PLOT_ID, renderer_kind=_CONDUIT_RENDERER_KIND)


def test_conduit_flow_peak_flow_vmax_rejects_negative():
    with pytest.raises(ValidationError):
        ConduitFlowStaticConfig(
            plot_id=_CONDUIT_PLOT_ID, renderer_kind=_CONDUIT_RENDERER_KIND, peak_flow_vmax=-1.0
        )


# Phase 4 — sensitivity_benchmarking per-function model + registry entry
_SENS_PLOT_ID = "sensitivity_benchmarking"
_SENS_RENDERER_KIND = "sensitivity_benchmarking"


def test_registry_maps_sensitivity_benchmarking_kind_to_subclass():
    assert STATIC_PLOT_CONFIG_REGISTRY[_SENS_RENDERER_KIND] is SensitivityBenchmarkingStaticConfig
    assert issubclass(SensitivityBenchmarkingStaticConfig, StaticPlotBaseConfig)


def test_sensitivity_benchmarking_inherits_and_extends():
    cfg = SensitivityBenchmarkingStaticConfig(plot_id=_SENS_PLOT_ID, renderer_kind=_SENS_RENDERER_KIND)
    assert cfg.bbox_inches_tight is False  # inherited base default
    assert cfg.output_format == "pdf"  # inherited base default
    assert cfg.series_palette[0] == "#000000"  # Okabe-Ito default (CVD-safe)
    assert len(cfg.series_palette) == 8
    assert cfg.cpu_marker == "o"  # added content knob
    assert cfg.gpu_marker == "s"  # added content knob
    assert cfg.log_y is False  # added content knob


def test_sensitivity_benchmarking_palette_bypasses_cvd_advisory():
    # Documents the known bypass (data-viz/hhemt/SE specialists, routed to follow-up):
    # _cvd_advisory inspects only str fields ending in _cmap, so series_palette (a
    # tuple) is never inspected — even a deliberately non-CVD-safe palette fires no
    # advisory. The default is Okabe-Ito (CVD-safe), so the bypass is acceptable for v1.
    with warnings.catch_warnings():
        warnings.simplefilter("error", CvdAdvisoryWarning)
        SensitivityBenchmarkingStaticConfig(
            plot_id=_SENS_PLOT_ID,
            renderer_kind=_SENS_RENDERER_KIND,
            series_palette=("red", "lime"),  # non-CVD-safe but valid; bypass => no warning
        )


def test_registry_has_all_four_renderer_kinds():
    # Phase 4 DoD: all four renderer kinds registered.
    assert set(STATIC_PLOT_CONFIG_REGISTRY) == {
        "per_sim_peak_flood_depth",
        "system_overview",
        "per_sim_conduit_flow",
        "sensitivity_benchmarking",
    }
    assert all(issubclass(m, StaticPlotBaseConfig) for m in STATIC_PLOT_CONFIG_REGISTRY.values())


# R8 — bad colormap raises (via the viz_vocabulary MplColormap AfterValidator)
def test_bad_colormap_raises():
    with pytest.raises(ValidationError):
        PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID, renderer_kind=_RENDERER_KIND, depth_cmap="not_a_real_cmap")


# R9 — CVD advisory is non-blocking
def test_cvd_advisory_warns_non_cvd_safe():
    assert "terrain" not in _CVD_SAFE_COLORMAPS  # valid mpl cmap, not CVD-safe
    with pytest.warns(CvdAdvisoryWarning):
        cfg = PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID, renderer_kind=_RENDERER_KIND, depth_cmap="terrain")
    assert cfg.depth_cmap == "terrain"  # construction succeeded


def test_cvd_advisory_silent_for_cvd_safe_default():
    with warnings.catch_warnings():
        warnings.simplefilter("error", CvdAdvisoryWarning)
        PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID, renderer_kind=_RENDERER_KIND)  # default YlGnBu — no warning


# R9 — the base (no colormap field) never fires the duck-typed CVD loop
def test_base_constructs_without_cvd_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", CvdAdvisoryWarning)
        StaticPlotBaseConfig(**_base())  # no *_cmap field on the base — silent


# R9 — a reversed (_r) variant of a CVD-safe map does NOT spuriously warn
def test_cvd_advisory_silent_for_reversed_cvd_safe():
    with warnings.catch_warnings():
        warnings.simplefilter("error", CvdAdvisoryWarning)
        PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID, renderer_kind=_RENDERER_KIND, depth_cmap="YlGnBu_r")


# D2-companion — colorbar_norm='boundary' + bound-derivation override raises
@pytest.mark.parametrize(
    "overrides",
    [
        {"vmax_quantile": 0.95},
        {"vmin_vmax_strategy": VminVmaxStrategy.quantile},
        {"vmin": 0.0},
        {"vmax": 5.0},
    ],
)
def test_boundary_norm_rejects_bound_derivation(overrides):
    with pytest.raises(ValidationError):
        StaticPlotBaseConfig(**_base(colorbar_norm="boundary", **overrides))


def test_boundary_norm_default_bounds_ok():
    cfg = StaticPlotBaseConfig(**_base(colorbar_norm="boundary"))
    assert cfg.colorbar_norm == "boundary"


def test_continuous_norm_allows_bound_derivation():
    cfg = StaticPlotBaseConfig(**_base(colorbar_norm="linear", vmax_quantile=0.95))
    assert cfg.vmax_quantile == 0.95
