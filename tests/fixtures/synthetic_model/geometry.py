"""Synthesize DEM GeoTIFF for the synthetic test model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from affine import Affine


def build_dem(params, dest: Path) -> Path:
    """Build a DEM with 1% N->S slope plus a shallow central valley and a single outlet.

    Elevation model (meters):
        base(row, col) = base_elev - row * cell_size_m * slope_ns
        valley(col)    = -valley_depth_m * exp(-((col - center_col)**2) / (2 * (n_cols/6)**2))
        outlet cell at (n_rows-1, center_col) is forced 0.05 m below neighbor to guarantee drainage
    """
    n_rows = params.n_rows
    n_cols = params.n_cols
    cs = params.cell_size_m
    base_elev = 10.0
    center_col = n_cols // 2
    sigma = max(1.0, n_cols / 6.0)

    rr, cc = np.mgrid[0:n_rows, 0:n_cols].astype(np.float32)
    base = base_elev - rr * cs * params.slope_ns
    valley = -params.valley_depth_m * np.exp(
        -((cc - center_col) ** 2) / (2.0 * sigma**2)
    )
    dem = base + valley
    dem[n_rows - 1, center_col] -= 0.05  # guarantee outlet

    transform = Affine.translation(params.xllcorner, params.yllcorner + n_rows * cs) * Affine.scale(
        cs, -cs
    )
    da = xr.DataArray(
        dem.astype(np.float32),
        dims=("y", "x"),
        coords={
            # Pixel-center convention — agrees with Affine transform above
            "y": params.yllcorner + cs * (np.arange(n_rows - 1, -1, -1) + 0.5),
            "x": params.xllcorner + cs * (np.arange(n_cols) + 0.5),
        },
    )
    da.rio.write_crs(f"EPSG:{params.epsg}", inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(-9999.0, inplace=True)
    da.rio.to_raster(dest, driver="GTiff", dtype="float32", compress="deflate", tiled=True)
    return dest
