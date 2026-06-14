"""Unit tests for the publication static-plot config family (config/static_plots.py)."""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from TRITON_SWMM_toolkit.config.static_plots import (
    _CVD_SAFE_COLORMAPS,
    CvdAdvisoryWarning,
    PeakFloodDepthStaticConfig,
    StaticPlotBaseConfig,
)
from TRITON_SWMM_toolkit.config.viz_vocabulary import FontTarget, VminVmaxStrategy

_VALID_PLOT_ID = "per_sim_peak_flood_depth__evt.year.9"


def _base(**overrides):
    kw = {"plot_id": _VALID_PLOT_ID}
    kw.update(overrides)
    return kw


# R2 — plot_id strict-required
def test_plot_id_required():
    with pytest.raises(ValidationError):
        StaticPlotBaseConfig()


# R3 — plot_id charset
def test_plot_id_rejects_hyphen():
    with pytest.raises(ValidationError):
        StaticPlotBaseConfig(plot_id="per_sim_peak_flood_depth__sa-0")


def test_plot_id_accepts_adr2_dot_form():
    cfg = StaticPlotBaseConfig(plot_id="per_sim_peak_flood_depth__sa.0__evt.year.9")
    assert cfg.plot_id == "per_sim_peak_flood_depth__sa.0__evt.year.9"


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
    cfg = PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID)
    assert cfg.bbox_inches_tight is False
    assert cfg.output_format == "pdf"
    assert cfg.depth_cmap == "YlGnBu"
    assert cfg.depth_under_color == "white"
    assert cfg.depth_over_color is None


# R8 — bad colormap raises (via the viz_vocabulary MplColormap AfterValidator)
def test_bad_colormap_raises():
    with pytest.raises(ValidationError):
        PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID, depth_cmap="not_a_real_cmap")


# R9 — CVD advisory is non-blocking
def test_cvd_advisory_warns_non_cvd_safe():
    assert "terrain" not in _CVD_SAFE_COLORMAPS  # valid mpl cmap, not CVD-safe
    with pytest.warns(CvdAdvisoryWarning):
        cfg = PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID, depth_cmap="terrain")
    assert cfg.depth_cmap == "terrain"  # construction succeeded


def test_cvd_advisory_silent_for_cvd_safe_default():
    with warnings.catch_warnings():
        warnings.simplefilter("error", CvdAdvisoryWarning)
        PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID)  # default YlGnBu — no warning


# R9 — the base (no colormap field) never fires the duck-typed CVD loop
def test_base_constructs_without_cvd_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", CvdAdvisoryWarning)
        StaticPlotBaseConfig(**_base())  # no *_cmap field on the base — silent


# R9 — a reversed (_r) variant of a CVD-safe map does NOT spuriously warn
def test_cvd_advisory_silent_for_reversed_cvd_safe():
    with warnings.catch_warnings():
        warnings.simplefilter("error", CvdAdvisoryWarning)
        PeakFloodDepthStaticConfig(plot_id=_VALID_PLOT_ID, depth_cmap="YlGnBu_r")


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
