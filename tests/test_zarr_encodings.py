"""Phase 3 — return_dic_zarr_encodings float32 + explicit time-chunk behavior (R6/R7).

Verifies:
- store_float32=True sets dtype float32 on float data-vars (and not on int/coord vars).
- store_float32=False preserves float64 (no dtype override).
- time_chunk sets the on-disk chunk size on the timestep_min axis; None preserves
  first-write-extent chunking.
- The dtype encoding is orthogonal to CF attributes: round-tripping float32 vs
  float64 yields identical long_name/standard_name/units/cell_methods (R6).
"""

import numpy as np
import xarray as xr

from TRITON_SWMM_toolkit.utils import return_dic_zarr_encodings

_CF_ATTRS = {
    "long_name": "water surface elevation",
    "standard_name": "sea_surface_height_above_geopotential_datum",
    "units": "m",
    "cell_methods": "timestep_min: point",
}


def _make_ds(ntsteps: int = 6, ny: int = 4, nx: int = 5) -> xr.Dataset:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((ntsteps, ny, nx)).astype(np.float64)
    da = xr.DataArray(
        data,
        dims=["timestep_min", "y", "x"],
        coords={"timestep_min": np.arange(ntsteps), "y": np.arange(ny), "x": np.arange(nx)},
        attrs=dict(_CF_ATTRS),
    )
    return xr.Dataset({"wlevel": da})


def test_store_float32_sets_float32_dtype_in_encoding():
    ds = _make_ds()
    enc = return_dic_zarr_encodings(ds, store_float32=True)
    assert enc["wlevel"]["dtype"] == "float32"


def test_store_float32_false_preserves_float64():
    ds = _make_ds()
    enc = return_dic_zarr_encodings(ds, store_float32=False)
    assert "dtype" not in enc["wlevel"]  # no override -> source float64 preserved


def test_time_chunk_sets_timestep_axis_chunk():
    ds = _make_ds(ntsteps=6, ny=4, nx=5)
    enc = return_dic_zarr_encodings(ds, time_chunk=2)
    # timestep_min is axis 0 of (timestep_min, y, x)
    assert enc["wlevel"]["chunks"] == (2, 4, 5)


def test_time_chunk_none_preserves_first_write_extent():
    ds = _make_ds()
    enc = return_dic_zarr_encodings(ds, time_chunk=None)
    assert "chunks" not in enc["wlevel"]


def test_float32_roundtrip_preserves_cf_attrs(tmp_path):
    """R6: dtype lives in the encoding dict; CF attrs are identical float32 vs float64."""
    ds = _make_ds()

    store32 = tmp_path / "f32.zarr"
    ds.to_zarr(store32, mode="w", encoding=return_dic_zarr_encodings(ds, store_float32=True))
    back32 = xr.open_zarr(store32)
    assert back32["wlevel"].dtype == np.float32

    store64 = tmp_path / "f64.zarr"
    ds.to_zarr(store64, mode="w", encoding=return_dic_zarr_encodings(ds, store_float32=False))
    back64 = xr.open_zarr(store64)
    assert back64["wlevel"].dtype == np.float64

    # CF attributes are identical regardless of storage dtype.
    for key in _CF_ATTRS:
        assert back32["wlevel"].attrs[key] == _CF_ATTRS[key]
        assert back64["wlevel"].attrs[key] == _CF_ATTRS[key]
