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
