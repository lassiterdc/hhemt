"""Synthesize DEM GeoTIFF for the synthetic test model.

DEM design (2026-06-15 deterministic redesign — replaces the iter-2..17
hand-tuned 16x30 corridor; all region boundaries derive from
``n_rows``/``n_cols``/``cell_size`` + fractional constants, no hardcoded
absolute footprint):

  * **Bathtub border.** The upstream (north) modelled area is enclosed by a
    continuous high-elevation rectangle (``_WALL_ELEV``): the top
    ``_WALL_THICKNESS`` rows, the left/right ``_WALL_THICKNESS`` columns, and a
    FULL-WIDTH sea-wall row as the rectangle's southern edge.
  * **Downstream BC shelf.** The ``_BC_ZONE_ROWS`` rows south of the sea wall
    are a flat low shelf (``_BC_FLOOR_ELEV``) open to the variable-water-level
    (storm-tide) boundary condition. The sea wall blocks the BC from the
    upstream area; the ONLY hydraulic connection is the SWMM culvert from the
    ``collector`` junction (just upstream of the wall) to the ``sewer_outflow``
    interaction junction in this shelf — so tide backwater enters SWMM at the
    interaction node and surcharges up the chain, flooding every upstream node's
    rim (BC 2.0 m > all rims).
  * **Floodplain + river.** The upstream interior is a floodplain whose base
    surface slopes gently N->S (``_FLOODPLAIN_TOP_ELEV`` -> ``_FLOODPLAIN_BOT_ELEV``),
    DEPRESSED toward the conduit centerlines (the "river") by a deterministic
    buffer: a cell within ``_FLOODPLAIN_BUFFER_M`` of the NEAREST conduit is
    lowered by ``_FLOODPLAIN_GRADE * (buffer - distance)`` — a ~1.5% cross-slope
    toward the river. Using distance to the NEAREST conduit makes overlapping
    buffers take the MINIMUM elevation (deepest depression) WITHOUT compounding,
    exactly as specified. SWMM node rims sit on the centerline (distance 0 ->
    full depression) and are pinned rim==DEM (cache._assert_rim_matches_dem).
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from affine import Affine

# --- Bathtub / wall ---------------------------------------------------------
_WALL_ELEV = 50.0  # bathtub rim + sea wall; artificially high — excluded
# from the DEM colorbar's dynamic range (renderer R10).
_WALL_THICKNESS = 2  # cells of top + left/right border (vectors.py mirrors this).

# --- Downstream storm-tide BC shelf (south of the sea wall) -----------------
_BC_ZONE_ROWS = 4  # rows of open low shelf below the full-width sea wall.
_BC_FLOOR_ELEV = 0.0  # flat shelf elevation — well below the 2.0 m tide BC so
# the BC floods the interaction node's cell.

# --- Upstream floodplain base surface (gentle N->S slope) -------------------
_FLOODPLAIN_TOP_ELEV = 1.5  # base elevation at the north interior edge.
_FLOODPLAIN_BOT_ELEV = 0.5  # base elevation just upstream of the sea wall (the
# floodplain funnels down to the collector here).

# --- River buffer (the conduit "valley") ------------------------------------
_FLOODPLAIN_BUFFER_M = 30.0  # buffer half-width either side of a conduit centerline.
_FLOODPLAIN_GRADE = 0.015  # cross-slope toward the river (~1.5%); centerline
# depression = _FLOODPLAIN_GRADE * _FLOODPLAIN_BUFFER_M.

_INTERIOR_TOP_MR = _WALL_THICKNESS  # first interior matrix-row below the top wall.


def _sea_wall_mr(params) -> int:
    """Full-width sea-wall matrix-row = southern edge of the upstream bathtub.
    The ``_BC_ZONE_ROWS`` rows below it are the open storm-tide BC shelf."""
    return params.n_rows - 1 - _BC_ZONE_ROWS


@functools.cache
def _conduit_network(params):
    """Lazy-built (and cached) shapely union of all conduit centerlines, used by
    the floodplain river-depression. Cached because `build_dem` queries the
    network distance once per cell (n_rows*n_cols times). Node centers match
    `swmm_template._nodes` coordinate convention so a node point lies exactly on
    the line (distance 0 -> full centerline depression)."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from .swmm_template import _conduits, _nodes

    cs = params.cell_size_m
    node_xy = {name: (cs * (col + 0.5), cs * (row_from_bottom + 0.5)) for name, col, row_from_bottom in _nodes(params)}
    lines = [
        LineString([node_xy[from_node], node_xy[to_node]]) for _name, from_node, to_node, _length in _conduits(params)
    ]
    if not lines:
        return None
    return unary_union(lines)


def _base_floodplain_elev(params, matrix_row: int) -> float:
    """Linear N->S base floodplain surface from ``_FLOODPLAIN_TOP_ELEV`` at the
    north interior edge to ``_FLOODPLAIN_BOT_ELEV`` just upstream of the sea
    wall (so the floodplain longitudinally funnels toward the collector)."""
    top = _INTERIOR_TOP_MR
    bot = _sea_wall_mr(params) - 1
    span = max(bot - top, 1)
    frac = min(max((matrix_row - top) / float(span), 0.0), 1.0)
    return float(_FLOODPLAIN_TOP_ELEV - (_FLOODPLAIN_TOP_ELEV - _FLOODPLAIN_BOT_ELEV) * frac)


def _river_depression(params, col: int, row_from_bottom: int) -> float:
    """Buffer depression at a cell: deepest (``grade*buffer``) on a conduit
    centerline, ramping linearly to 0 at ``_FLOODPLAIN_BUFFER_M``. Distance to
    the NEAREST conduit => overlapping buffers take the min elevation (no
    compounding)."""
    from shapely.geometry import Point

    network = _conduit_network(params)
    if network is None:
        return 0.0
    cs = params.cell_size_m
    d = network.distance(Point(cs * (col + 0.5), cs * (row_from_bottom + 0.5)))
    if d >= _FLOODPLAIN_BUFFER_M:
        return 0.0
    return float(_FLOODPLAIN_GRADE * (_FLOODPLAIN_BUFFER_M - d))


def dem_elev_at(params, col: int, row_from_bottom: int) -> float:
    """Return the DEM cell-center elevation at (col, row_from_bottom).

    Mirrors `build_dem`'s region logic so SWMM node rims can be pinned to the
    DEM at build time (cache._assert_rim_matches_dem). col is 0..n_cols-1,
    row_from_bottom is 0..n_rows-1 (0 = bottom/southern row).
    """
    n_rows = params.n_rows
    n_cols = params.n_cols
    matrix_row = n_rows - 1 - row_from_bottom
    sea_wall = _sea_wall_mr(params)

    # Downstream storm-tide BC shelf (south of the sea wall): open, flat, low.
    if matrix_row > sea_wall:
        return float(_BC_FLOOR_ELEV)
    # Top wall (bathtub north edge).
    if matrix_row < _WALL_THICKNESS:
        return float(_WALL_ELEV)
    # Full-width sea-wall row (southern edge of the bathtub).
    if matrix_row == sea_wall:
        return float(_WALL_ELEV)
    # Left/right bathtub walls (upstream region only).
    if col < _WALL_THICKNESS or col >= n_cols - _WALL_THICKNESS:
        return float(_WALL_ELEV)
    # Upstream interior floodplain, depressed toward the river.
    return _base_floodplain_elev(params, matrix_row) - _river_depression(params, col, row_from_bottom)


def build_dem(params, dest: Path) -> Path:
    """Build the bathtub-floodplain DEM (see module docstring)."""
    n_rows = params.n_rows
    n_cols = params.n_cols
    cs = params.cell_size_m

    dem = np.zeros((n_rows, n_cols), dtype=np.float32)
    for mr in range(n_rows):
        for col in range(n_cols):
            dem[mr, col] = dem_elev_at(params, col, n_rows - 1 - mr)

    transform = Affine.translation(params.xllcorner, params.yllcorner + n_rows * cs) * Affine.scale(cs, -cs)
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
