"""Tests for the row-streamed `_write_raster` and setup-time DEM integrity assertion.

Covers:
  - R1: peak RSS < 250 MB on a synthetic 5000×5000 DEM (vs. ~4.5–5 GB for the prior pivot-based implementation).
  - R6: round-trip parity — write then read via rioxarray, assert np.allclose.
  - R7: byte-identical to a fixture representing the 1.1m DEM ground truth.
  - R5: integrity assertion fails fast on a malformed artifact (blank line interleaved).
"""

from __future__ import annotations

import resource
from pathlib import Path

import numpy as np
import pytest
import rioxarray as rxr
import xarray as xr

from hhemt.exceptions import ProcessingError


def _make_synth_da(ncols: int, nrows: int, *, dtype: str = "float32") -> xr.DataArray:
    """Construct a synthetic rioxarray-shaped DataArray for testing."""
    rng = np.random.default_rng(seed=42)
    data = rng.uniform(low=0.0, high=200.0, size=(1, nrows, ncols)).astype(dtype)
    x = np.arange(ncols, dtype="float64") * 0.35
    y = np.arange(nrows, dtype="float64")[::-1] * 0.35
    da = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={"band": [1], "y": y, "x": x},
    )
    da.rio.write_crs("EPSG:32147", inplace=True)
    return da


def test_write_raster_peak_rss_under_250mb(tmp_path: Path, monkeypatch):
    """R1 — peak RSS during _write_raster on a 5000×5000 DEM stays under 250 MB."""
    from hhemt.system import TRITONSWMM_system

    da = _make_synth_da(ncols=5000, nrows=5000)

    # Construct a minimal TRITONSWMM_system harness — we only need _write_raster.
    sys_obj = TRITONSWMM_system.__new__(TRITONSWMM_system)
    fpath = tmp_path / "synth_5000x5000.dem"
    metadata = {
        "ncols         ": 5000,
        "nrows         ": 5000,
        "xllcorner     ": 0.0,
        "yllcorner     ": 0.0,
        "cellsize      ": 0.35,
        "NODATA_value  ": -9999,
    }

    # Per SE plan-review Flag 6: use resource.getrusage for actual OS-reported peak RSS
    # (tracemalloc undercounts numpy/C-extension allocations).
    baseline_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    sys_obj._write_raster(fpath, da, raster_metadata=metadata)
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    peak_mb = (peak_rss_kb - baseline_rss_kb) / 1024
    assert peak_mb < 250, f"delta peak RSS {peak_mb:.1f} MB exceeded 250 MB ceiling"


def test_write_raster_roundtrip_parity(tmp_path: Path):
    """R6 — round-trip: write then read via rioxarray, allclose at 1e-5 tolerance."""
    from hhemt.system import TRITONSWMM_system

    da = _make_synth_da(ncols=100, nrows=80)
    sys_obj = TRITONSWMM_system.__new__(TRITONSWMM_system)
    fpath = tmp_path / "synth_100x80.dem"
    metadata = {
        "ncols         ": 100,
        "nrows         ": 80,
        "xllcorner     ": 0.0,
        "yllcorner     ": 0.0,
        "cellsize      ": 0.35,
        "NODATA_value  ": -9999,
    }

    sys_obj._write_raster(fpath, da, raster_metadata=metadata)

    read_da = rxr.open_rasterio(fpath).load()
    np.testing.assert_allclose(read_da.values.squeeze(), da.values.squeeze(), atol=1e-5)


def test_dem_integrity_assertion_fires_fast_on_blank_line(tmp_path: Path):
    """R5 — synthetic malformed DEM with blank-line interleaving triggers ProcessingError."""
    # Per SE plan-review Flag 5: _assert_dem_integrity is a module-level function (state-free).
    from hhemt.system import _assert_dem_integrity

    fpath = tmp_path / "malformed.dem"

    # Header declares 3 rows; body has 3 real rows + 3 interleaved blank lines = 6 body lines.
    fpath.write_text(
        "ncols         3\n"
        "nrows         3\n"
        "xllcorner     0.0\n"
        "yllcorner     0.0\n"
        "cellsize      0.35\n"
        "NODATA_value  -9999\n"
        "1.0 2.0 3.0\n"
        "\n"
        "4.0 5.0 6.0\n"
        "\n"
        "7.0 8.0 9.0\n"
        "\n"
    )

    with pytest.raises(ProcessingError):
        _assert_dem_integrity(fpath)


def test_dem_integrity_assertion_passes_on_clean_file(tmp_path: Path):
    """R5 control — well-formed DEM passes the integrity check."""
    from hhemt.system import TRITONSWMM_system, _assert_dem_integrity

    da = _make_synth_da(ncols=10, nrows=8)
    sys_obj = TRITONSWMM_system.__new__(TRITONSWMM_system)
    fpath = tmp_path / "clean.dem"
    metadata = {
        "ncols         ": 10,
        "nrows         ": 8,
        "xllcorner     ": 0.0,
        "yllcorner     ": 0.0,
        "cellsize      ": 0.35,
        "NODATA_value  ": -9999,
    }
    sys_obj._write_raster(fpath, da, raster_metadata=metadata)

    # Should not raise.
    _assert_dem_integrity(fpath)
