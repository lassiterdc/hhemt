"""Synthesize DEM GeoTIFF for the synthetic test model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from affine import Affine

# DEM design constants (shared with swmm_template.py so node rims pin to DEM).
_TOP_ELEV = 7.0
_PRE_DROPOFF_BOTTOM_ELEV = 1.5  # elevation at matrix_row = _PRE_DROPOFF_LAST_MR
_POST_DROPOFF_ELEV = 0.5        # below dropoff — at the BC interface
_DROPOFF_AMOUNT_M = 1.0         # visual reference — equals _PRE_DROPOFF_BOTTOM_ELEV - _POST_DROPOFF_ELEV
_WALL_ELEV = 50.0
_WALL_THICKNESS = 2
# Iteration-2 layout (sea wall divides slope from dropoff):
#   matrix_row 0..1    : top wall (_WALL_ELEV)
#   matrix_row 2..25   : gradual interior slope (_TOP_ELEV → _PRE_DROPOFF_BOTTOM_ELEV)
#   matrix_row 26      : sea-wall row (_WALL_ELEV) — interior peak between slope and dropoff
#   matrix_row 27..29  : dropoff/BC zone (_POST_DROPOFF_ELEV)
_INTERIOR_TOP_MR = 2
_PRE_DROPOFF_LAST_MR = 25
_SEA_WALL_MR = 26


def dem_elev_at(params, col: int, row_from_bottom: int) -> float:
    """Return the DEM cell-center elevation at (col, row_from_bottom).

    Mirrors `build_dem`'s bathtub + dropoff logic so SWMM node rim elevations
    can be pinned to the DEM at build time (see cache.py rim==DEM assertion).

    Parameters mirror `_NODES` coordinate conventions in `swmm_template.py`:
    col is 0..n_cols-1, row_from_bottom is 0..n_rows-1 (0 = bottom row).
    """
    n_rows = params.n_rows
    n_cols = params.n_cols
    matrix_row = n_rows - 1 - row_from_bottom
    # Walls: top `_WALL_THICKNESS` rows (except bottom, which is BC exit);
    # left/right `_WALL_THICKNESS` columns.
    if matrix_row < _WALL_THICKNESS and matrix_row != n_rows - 1:
        return _WALL_ELEV
    if col < _WALL_THICKNESS or col >= n_cols - _WALL_THICKNESS:
        return _WALL_ELEV
    # Sea-wall row: interior peak between gradual slope and dropoff (iter 2).
    if matrix_row == _SEA_WALL_MR:
        return _WALL_ELEV
    if matrix_row > _SEA_WALL_MR:
        return float(_POST_DROPOFF_ELEV)
    # Linear interior gradient from _TOP_ELEV (at _INTERIOR_TOP_MR) down to
    # _PRE_DROPOFF_BOTTOM_ELEV (at _PRE_DROPOFF_LAST_MR).
    span = _PRE_DROPOFF_LAST_MR - _INTERIOR_TOP_MR
    frac = (matrix_row - _INTERIOR_TOP_MR) / float(span)
    return float(_TOP_ELEV - (_TOP_ELEV - _PRE_DROPOFF_BOTTOM_ELEV) * frac)


def build_dem(params, dest: Path) -> Path:
    """Build a 'bathtub' DEM with a 1-m elevation dropoff above the BC.

    Regions (matrix-row index, 0 at top):
      * Walls (top `_WALL_THICKNESS` rows excluding bottom, and left/right
        `_WALL_THICKNESS` columns): `_WALL_ELEV` (50 m).
      * Interior gradient (rows `_INTERIOR_TOP_MR`..`_PRE_DROPOFF_LAST_MR`):
        linear from `_TOP_ELEV` down to `_PRE_DROPOFF_BOTTOM_ELEV`.
      * Dropoff zone (rows `_PRE_DROPOFF_LAST_MR+1`..n_rows-1): flat at
        `_POST_DROPOFF_ELEV`, one-m below the pre-dropoff interior floor.
        This emulates a sea-water BC: J4 sits in this zone, downstream of
        the dropoff.
    """
    n_rows = params.n_rows
    n_cols = params.n_cols
    cs = params.cell_size_m

    dem = np.zeros((n_rows, n_cols), dtype=np.float32)
    for mr in range(n_rows):
        for col in range(n_cols):
            dem[mr, col] = dem_elev_at(params, col, n_rows - 1 - mr)

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
