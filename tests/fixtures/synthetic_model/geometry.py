"""Synthesize DEM GeoTIFF for the synthetic test model."""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401  (registers .rio accessor)
import xarray as xr
from affine import Affine

# DEM design constants (shared with swmm_template.py so node rims pin to DEM).
#
# Iter-6 of `per_sim_peak_flood_depth` (2026-04-28): the upstream interior
# is now FLAT at `_PRE_DROPOFF_BOTTOM_ELEV` (1.5 m), with internal channel
# walls confining the wettable upstream zone to columns
# `_CHANNEL_COL_MIN`..`_CHANNEL_COL_MAX` (cols 8..12, 5 cells wide). All
# other interior cells (cols 2..7 and 13..17 in matrix-rows 2..25) are walls
# at `_WALL_ELEV`. This concentrates SWMM-pipe-backwater surcharge water
# into a narrow channel so 30-min sim with bumped 0.2 m pipes produces
# visibly substantial upstream flooding (was: thin <25 mm uniform spread
# across 16-cell-wide gradient row, barely visible).
#
# Iter-2 history: _TOP_ELEV had been reduced 7.0→3.0 to shallow the upstream
# gradient. With iter-6's flat upstream interior, _TOP_ELEV is unused for
# the channel cells (held to _PRE_DROPOFF_BOTTOM_ELEV throughout) but kept
# as a constant for any downstream consumer that still references it.
_TOP_ELEV = 1.5  # legacy constant; superseded by _UPSTREAM_TOP_ELEV below
_PRE_DROPOFF_BOTTOM_ELEV = 1.0  # elevation at matrix_row = _PRE_DROPOFF_LAST_MR
_POST_DROPOFF_ELEV = 0.5        # iter-17 peak_flood_depth (2026-04-29): per user,
                                # "the elevation of the portion south of the
                                # sea wall to equal the minimum elevation of
                                # the portion upstream of the sea wall". The
                                # minimum upstream elev is the buffer-lowered
                                # swale at the bottom row of the corridor:
                                # gradient 1.0 m at mr=25 minus swale 0.5 m
                                # = 0.5 m. Dropoff matches that.
_DROPOFF_AMOUNT_M = 0.5         # visual reference — equals _PRE_DROPOFF_BOTTOM_ELEV - _POST_DROPOFF_ELEV
_WALL_ELEV = 50.0
_WALL_THICKNESS = 2
# Iter-8 peak_flood_depth (2026-04-28): rather than a rectangular channel
# band (iter-6/7 cols 8..12), the wettable upstream area now follows the Y
# shape of the conduit network. Cells whose centers are within
# `_Y_CORRIDOR_HALF_WIDTH_M` of any conduit centerline are inside the Y
# corridor (~3 cells perpendicular per branch); cells outside the corridor
# but inside the interior gradient region are walls. Cells within
# `_CONDUIT_BUFFER_RADIUS_M` of a centerline (the swale, ~1 cell
# perpendicular) are additionally lowered by `_CONDUIT_BUFFER_LOWERING_M`.
# The legacy `_CHANNEL_COL_MIN/MAX` are retained as no-op constants for any
# downstream consumer that still references them.
_CHANNEL_COL_MIN = 0
_CHANNEL_COL_MAX = 0
_Y_CORRIDOR_HALF_WIDTH_M = 15.0  # ~1.5 cells either side of centerline
# Iter-8 sloped upstream gradient: surface drops from `_UPSTREAM_TOP_ELEV`
# at matrix_row = _INTERIOR_TOP_MR down to `_UPSTREAM_BOT_ELEV` at
# matrix_row = _PRE_DROPOFF_LAST_MR. With Y corridor walls confining
# wettable area, the gradient creates a real downhill flow direction so
# water has somewhere to go; combined with the swale lowering, the figure
# shows variation in depth along the network rather than a uniform pool.
# Iter-16 (2026-04-29): per user, "now the DEM is too flat. I want there to
# be some slope, but i want it to be very gradual. Like maybe from 1.5m to
# 1m or something." Restored a small gradient: TOP 1.5 → BOT 1.0 over the
# 23-row corridor (0.5 m drop / 230 m linear distance ≈ 0.22 % surface
# slope). With buffer-lowered swale (-0.5 m), post-swale rims:
#   J1, J2 (mr=7):  1.391 - 0.5 = 0.891
#   J3     (mr=16): 1.196 - 0.5 = 0.696
#   J4     (mr=25): 1.000 - 0.5 = 0.500
#   sewer_outflow  (mr=27, in dropoff zone): 1.000 (no swale)
_UPSTREAM_TOP_ELEV = 1.5
_UPSTREAM_BOT_ELEV = 1.0
# Layout (matrix-row index, 0 at top):
#   matrix_row 0..1    : top wall (_WALL_ELEV)
#   matrix_row 2..25   : interior, flat at `_PRE_DROPOFF_BOTTOM_ELEV` inside
#                        the channel band; `_WALL_ELEV` outside the band
#   matrix_row 26      : sea-wall row (_WALL_ELEV)
#   matrix_row 27..29  : dropoff/BC zone (_POST_DROPOFF_ELEV)
_INTERIOR_TOP_MR = 2
_PRE_DROPOFF_LAST_MR = 25
_SEA_WALL_MR = 26

# Iter-7 peak_flood_depth (2026-04-28): every grid cell touching a 5-m buffer
# around any conduit centerline is lowered by `_CONDUIT_BUFFER_LOWERING_M`.
# The lowered cells form a "swale" along the pipe alignments so SWMM
# surcharge water from the pipes pools in the swale instead of spreading
# uniformly across the (already narrow) channel band. Junction rims still
# follow rim==DEM, so SWMM rims drop with the swale; junction depths are
# left as set in `swmm_template._NODE_DEPTHS_M` and inverts auto-recompute
# (rim − depth). Pipe slopes stay ≥1 % because the lowering is uniform along
# the pipe alignment (every junction cell drops by the same 0.5 m).
_CONDUIT_BUFFER_RADIUS_M = 5.0
_CONDUIT_BUFFER_LOWERING_M = 0.5


def _conduit_network(params):
    """Lazy-built shapely union of all conduit centerlines (used by both the
    Y-corridor cell-set and the swale buffer-lowering cell-set)."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    from .swmm_template import _CONDUITS, _NODES

    cs = params.cell_size_m
    node_xy = {
        name: (cs * (col + 0.5), cs * (row_from_bottom + 0.5))
        for name, col, row_from_bottom in _NODES
    }
    lines = [
        LineString([node_xy[from_node], node_xy[to_node]])
        for _name, from_node, to_node, _length in _CONDUITS
    ]
    if not lines:
        return None
    return unary_union(lines)


@functools.cache
def _y_corridor_cells(params) -> frozenset:
    """Cells whose centers are within `_Y_CORRIDOR_HALF_WIDTH_M` of any
    conduit centerline. Defines the wettable upstream area under iter-8's
    Option B Y-corridor design — cells outside this set inside the interior
    gradient region become walls."""
    from shapely.geometry import Point

    network = _conduit_network(params)
    if network is None:
        return frozenset()
    cs = params.cell_size_m
    cells: set[tuple[int, int]] = set()
    for col in range(params.n_cols):
        for row_from_bottom in range(params.n_rows):
            pt = Point(cs * (col + 0.5), cs * (row_from_bottom + 0.5))
            if network.distance(pt) <= _Y_CORRIDOR_HALF_WIDTH_M:
                cells.add((col, row_from_bottom))
    return frozenset(cells)


@functools.cache
def _buffered_cells(params) -> frozenset:
    """Cells whose extent intersects the `_CONDUIT_BUFFER_RADIUS_M`-wide
    buffer around any conduit centerline (the swale: ~1 cell perpendicular,
    visible as a darker stripe outlining the conduit centerlines on the DEM
    figure)."""
    from shapely.geometry import box

    network = _conduit_network(params)
    if network is None:
        return frozenset()
    buf = network.buffer(_CONDUIT_BUFFER_RADIUS_M)
    cs = params.cell_size_m
    cells: set[tuple[int, int]] = set()
    for col in range(params.n_cols):
        for row_from_bottom in range(params.n_rows):
            cell = box(
                col * cs, row_from_bottom * cs,
                (col + 1) * cs, (row_from_bottom + 1) * cs,
            )
            if cell.intersects(buf):
                cells.add((col, row_from_bottom))
    return frozenset(cells)


def _interior_gradient_elev(matrix_row: int) -> float:
    """Linear surface gradient from `_UPSTREAM_TOP_ELEV` at
    `_INTERIOR_TOP_MR` down to `_UPSTREAM_BOT_ELEV` at
    `_PRE_DROPOFF_LAST_MR`. Cells inside the Y corridor return this elev
    (minus swale lowering if applicable); cells outside are walls."""
    span = _PRE_DROPOFF_LAST_MR - _INTERIOR_TOP_MR
    frac = (matrix_row - _INTERIOR_TOP_MR) / float(span)
    return float(_UPSTREAM_TOP_ELEV - (_UPSTREAM_TOP_ELEV - _UPSTREAM_BOT_ELEV) * frac)


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
    # Iter-11 base elevation:
    #   - dropoff zone (matrix_row > _SEA_WALL_MR): constant `_POST_DROPOFF_ELEV`
    #     (0.0m); buffer-lowering does NOT apply here (early return).
    #   - upstream interior, INSIDE Y corridor: linear sloped gradient
    #   - upstream interior, OUTSIDE Y corridor: WALL (Option B Y-shaped channel)
    if matrix_row > _SEA_WALL_MR:
        return float(_POST_DROPOFF_ELEV)
    if (col, row_from_bottom) in _y_corridor_cells(params):
        base = _interior_gradient_elev(matrix_row)
    else:
        return _WALL_ELEV
    # Iter-7 buffer-lowering: swale cells (within ~5 m of conduit centerline)
    # drop by 0.5 m, outlining the Y on the DEM. Applies inside the corridor
    # only — iter-11 keeps the dropoff zone uniformly flat per user feedback.
    if (col, row_from_bottom) in _buffered_cells(params):
        base -= _CONDUIT_BUFFER_LOWERING_M
    return base


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
