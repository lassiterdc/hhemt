"""Phase 2: the cross-resolution regrid kernel + its six guards + the additive keys.

Pure-numpy/xarray unit tests -- no compile, no HPC, no synth fixture build. The guards
are the point: each negative case asserts a LOUD failure where the naive implementation
returns a silent false-PASS.
"""

from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

from hhemt.eda._dem_resolution import compare_resolution_pair, regrid_to_fine
from hhemt.eda.compute_sensitivity import compute_magnitude
from hhemt.exceptions import ProcessingError

_EPSG = 32618  # UTM 18N -- realistic coordinates; a synthetic x=0..6 grid declared as
# UTM makes reproject_match return all-nodata SILENTLY (coords outside the CRS's valid
# domain). Guard 5 catches it, but the fixtures use real coordinates so the POSITIVE
# tests exercise the real path.
_X0, _Y0 = 500000.0, 4100000.0


def _grid(res: float, n: int, fill: float = 1.0) -> xr.DataArray:
    x = _X0 + res * (np.arange(n) + 0.5)
    y = _Y0 - res * (np.arange(n) + 0.5)
    da = xr.DataArray(np.full((n, n), fill, dtype="float64"), dims=("y", "x"), coords={"y": y, "x": x})
    return da.rio.write_crs(_EPSG).rio.write_nodata(np.nan)


def test_regrid_shape_matches_the_fine_template():
    base, test, diag = regrid_to_fine(_grid(4.0, 8), _grid(2.0, 16), horizontal_epsg=_EPSG)
    assert base.shape == test.shape == (16, 16)
    assert diag["direction"] == "coarse_to_fine"
    assert diag["coarse_res_m"] == 4.0 and diag["fine_res_m"] == 2.0


def test_average_degenerates_to_nearest_on_upsample():
    """The MEASURED fact D2b's two strikes rest on. Assert it rather than assume it."""
    from rasterio.enums import Resampling

    coarse = _grid(4.0, 4)
    coarse.values[:2, :2] = 10.0
    fine = _grid(2.0, 8)
    out_avg = coarse.rio.reproject_match(fine, resampling=Resampling.average)
    out_near = coarse.rio.reproject_match(fine, resampling=Resampling.nearest)
    out_max = coarse.rio.reproject_match(fine, resampling=Resampling.max)
    assert np.array_equal(np.asarray(out_avg), np.asarray(out_near), equal_nan=True)
    assert np.array_equal(np.asarray(out_max), np.asarray(out_near), equal_nan=True)
    # No partial averages: every finite output value is a SOURCE value.
    assert set(np.unique(np.asarray(out_avg))) <= set(np.unique(np.asarray(coarse)))


def test_guard1_crs_mismatch_raises():
    coarse = _grid(4.0, 8).rio.write_crs(32618)
    fine = _grid(2.0, 16).rio.write_crs(4326)
    with pytest.raises(ProcessingError, match="CRS mismatch"):
        regrid_to_fine(coarse, fine, horizontal_epsg=_EPSG)


def test_none_crs_grids_are_filled_from_horizontal_epsg():
    """Production grids arrive rio.crs is None; horizontal_epsg FILLS the missing CRS
    (fill-only, not override), so the regrid proceeds. Covers the `is None` branch that
    the guard1 (CRS-present) test does not, and locks the semantics Guard 1 rests on."""

    def _raw(res: float, n: int) -> xr.DataArray:
        x = _X0 + res * (np.arange(n) + 0.5)
        y = _Y0 - res * (np.arange(n) + 0.5)
        return xr.DataArray(np.full((n, n), 1.0, dtype="float64"), dims=("y", "x"), coords={"y": y, "x": x})

    coarse, fine = _raw(4.0, 8), _raw(2.0, 16)
    assert coarse.rio.crs is None and fine.rio.crs is None
    base, test, diag = regrid_to_fine(coarse, fine, horizontal_epsg=_EPSG)
    assert base.shape == test.shape == (16, 16)
    assert diag["direction"] == "coarse_to_fine"


def test_guard2_disjoint_extents_raise_rather_than_reporting_perfect_agreement():
    """THE false-PASS this guard set exists for: without it, compute_magnitude's
    empty-domain branch returns max_abs=0.0 -- a wrong pair reads as perfect."""
    coarse = _grid(4.0, 8)
    far = _grid(2.0, 16)
    far = far.assign_coords(x=far.x + 50_000.0, y=far.y - 50_000.0).rio.write_crs(_EPSG)
    with pytest.raises(ProcessingError, match="disjoint extents"):
        regrid_to_fine(coarse, far, horizontal_epsg=_EPSG)


def test_guard4_swapped_direction_raises():
    with pytest.raises(ProcessingError, match="regrid direction violated"):
        regrid_to_fine(_grid(2.0, 16), _grid(4.0, 8), horizontal_epsg=_EPSG)


def test_empty_domain_would_be_perfect_agreement_without_guard5():
    """Pins the mechanism guard 5 defends against, at the compute_magnitude layer."""
    empty = np.full((4, 4), np.nan)
    m = compute_magnitude(empty, empty)
    assert m["max_abs_diff_m"] == 0.0 and m["rmse_wetted_m"] == 0.0
    assert m["csi"] is None, "CSI must be None on an empty domain, never 1.0"


def test_additive_keys_are_present_and_signed():
    base = np.full((4, 4), 0.50)
    test = np.full((4, 4), 0.40)  # coarse UNDER-estimates by 20%
    m = compute_magnitude(base, test)
    assert m["pct_diff_p95_signed"] < 0.0, "under-estimation must read NEGATIVE"
    assert m["pct_diff_p05_signed"] < 0.0
    assert m["pct_diff_p95"] > 0.0, "the abs-summarized secondary stays positive"
    assert m["csi"] == pytest.approx(1.0)


def test_existing_member_keys_are_unchanged():
    """R3 is additive-only: the fixed _artifact_vars allowlist must still resolve."""
    m = compute_magnitude(np.full((4, 4), 0.5), np.full((4, 4), 0.4))
    for k in ("max_abs_diff_m", "rmse_wetted_m", "pct_diff_p95", "n_newly_wet", "n_extent_disagree"):
        assert k in m


def test_compare_resolution_pair_threads_the_disclosed_threshold():
    out = compare_resolution_pair(
        _grid(4.0, 8, 0.05), _grid(2.0, 16, 0.05), horizontal_epsg=_EPSG, dry_threshold_m=0.10
    )
    assert out["dry_threshold_m"] == 0.10, "the 2.5 mm module default must NOT carry over"
    assert out["direction"] == "coarse_to_fine"
