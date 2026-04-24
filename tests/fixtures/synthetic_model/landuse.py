"""Synthesize landuse GeoTIFF and Manning's lookup CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine

IMPERVIOUS_ID = 1
PERVIOUS_ID = 2


def build_landuse(params, dest: Path) -> Path:
    """Upper half impervious (class 1), lower half pervious (class 2)."""
    n_rows = params.n_rows
    n_cols = params.n_cols
    cs = params.cell_size_m
    arr = np.full((n_rows, n_cols), PERVIOUS_ID, dtype=np.int32)
    arr[: n_rows // 2, :] = IMPERVIOUS_ID

    transform = Affine.translation(params.xllcorner, params.yllcorner + n_rows * cs) * Affine.scale(
        cs, -cs
    )
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            # Pixel-center convention — agrees with Affine transform above
            "y": params.yllcorner + cs * (np.arange(n_rows - 1, -1, -1) + 0.5),
            "x": params.xllcorner + cs * (np.arange(n_cols) + 0.5),
        },
    )
    da.rio.write_crs(f"EPSG:{params.epsg}", inplace=True)
    da.rio.write_transform(transform, inplace=True)
    # nodata=0 is safe: synthetic classes are 1 (impervious) and 2 (pervious); class 0 reserved as nodata.
    da.rio.write_nodata(0, inplace=True)
    da.rio.to_raster(dest, driver="GTiff", dtype="int32")
    return dest


def build_lookup(params, dest: Path) -> Path:
    """Lookup CSV with columns: landuse_class_id, mannings, description."""
    df = pd.DataFrame(
        {
            "landuse_class_id": [IMPERVIOUS_ID, PERVIOUS_ID],
            "mannings": [params.impervious_mannings, params.pervious_mannings],
            "description": ["impervious", "pervious"],
        }
    )
    df.to_csv(dest, index=False)
    return dest
