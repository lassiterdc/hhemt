"""Synthesize watershed polygon and storm-tide boundary line as GeoJSON."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Polygon


def _extent(params):
    cs = params.cell_size_m
    x0 = params.xllcorner
    y0 = params.yllcorner
    x1 = x0 + params.n_cols * cs
    y1 = y0 + params.n_rows * cs
    return x0, y0, x1, y1


def build_watershed(params, dest: Path) -> Path:
    """Polygon covering the DEM extent inset by one cell on all sides."""
    x0, y0, x1, y1 = _extent(params)
    cs = params.cell_size_m
    poly = Polygon(
        [
            (x0 + cs, y0 + cs),
            (x1 - cs, y0 + cs),
            (x1 - cs, y1 - cs),
            (x0 + cs, y1 - cs),
        ]
    )
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs=f"EPSG:{params.epsg}")
    gdf.to_file(dest, driver="GeoJSON", engine="pyogrio")
    return dest


def build_boundary(params, dest: Path) -> Path:
    """Storm-tide boundary along the interior (non-wall) portion of the DEM's
    southern edge. Walls occupy the outer `_WALL_THICKNESS` columns on each
    side, so the BC line spans x from `xllcorner + 2*cs` to `x1 - 2*cs` —
    matching the low-elevation interior where the bathtub drains.
    """
    x0, y0, x1, _ = _extent(params)
    cs = params.cell_size_m
    wall_thickness = 2  # matches geometry.py::_WALL_THICKNESS
    y_bc = y0 + 0.5 * cs
    line = LineString([
        (x0 + wall_thickness * cs, y_bc),
        (x1 - wall_thickness * cs, y_bc),
    ])
    gdf = gpd.GeoDataFrame({"bc_id": [1]}, geometry=[line], crs=f"EPSG:{params.epsg}")
    gdf.to_file(dest, driver="GeoJSON", engine="pyogrio")
    return dest
