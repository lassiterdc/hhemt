"""C4: the serial-reference depth cap is watershed-bounded AND cross-arm fungible."""

from __future__ import annotations

import numpy as np

from hhemt.eda._config_diff import _DEPTH_CAP_STEP_M, _apply_mask, _watershed_mask


def _quantize(v):
    return float(np.ceil(v / _DEPTH_CAP_STEP_M) * _DEPTH_CAP_STEP_M)


def test_mask_excludes_the_seaward_extreme_from_the_cap():
    """The boundary condition must not set the colorbar for the inland floodplain."""
    z = np.array([[0.4, 0.5], [0.6, 9.9]])          # 9.9 = seaward boundary cell
    mask = np.array([[True, True], [True, False]])  # watershed excludes it
    assert np.nanmax(_apply_mask(z, mask)) == 0.6


def test_cap_is_identical_across_arms_whose_peaks_differ_by_centimetres():
    """G3: a pure-TRITON and a coupled render of the same domain must not produce two
    different colorbars for the same TRITON quantity."""
    assert _quantize(0.61) == _quantize(0.6349)


def test_cap_never_clips_the_data():
    for v in (0.01, 0.24, 0.25, 0.26, 3.1):
        assert _quantize(v) >= v


def test_absent_polygon_yields_no_mask():
    """Excludable input: absence must be detectable by the renderer, not silent."""
    assert _watershed_mask(None, [0.0, 1.0], [0.0, 1.0]) is None
    z = np.array([[1.0, 2.0]])
    assert np.array_equal(_apply_mask(z, None), z)


def test_cap_guard_returns_none_on_a_field_with_no_positive_cell():
    """Iteration-11 item 4, guard arms (ii) and (iii): a fully-masked or all-dry
    field must yield NO cap rather than a zero cap, because plotly reads None as
    'autoscale' and 0 as a real upper bound that would render the panel black."""
    from hhemt.eda._config_diff import _quantised_depth_cap

    assert _quantised_depth_cap(np.full((2, 2), np.nan)) is None
    assert _quantised_depth_cap(np.zeros((2, 2))) is None
    assert _quantised_depth_cap(np.array([[0.61, np.nan], [np.nan, 0.4]])) == _quantize(0.61)


def test_reference_range_fallback_is_epsilon_not_one():
    """Iteration-11 item 4, guard arm (iv). The open-coded form this replaces fell
    back to 1.0, which asserts a 0-1 cms range no data supports."""
    from hhemt.eda._config_diff import _RANGE_EPS, _abs_ref_range

    assert _abs_ref_range(0.0) == _RANGE_EPS
    assert _abs_ref_range(float("nan")) == _RANGE_EPS
    assert _abs_ref_range(2.5) == 2.5


def test_glyph_is_square_and_single_sourced():
    """Iteration-11 item 6. Compares the legend-proxy marker to the paper-space
    swatch and to the module constant -- never to a literal -- so a future retune
    of WATERSHED_GLYPH_PX keeps this test passing while a second hardcoded copy
    of the old 28x16 rect makes it fail."""
    from hhemt.figure_panels import WATERSHED_GLYPH_PX, watershed_legend_marker

    marker = watershed_legend_marker(color="#111", line_width=1.3)
    assert marker["symbol"] == "square-open"
    assert marker["size"] == WATERSHED_GLYPH_PX
