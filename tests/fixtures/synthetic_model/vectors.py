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
    """Horizontal line along the southern edge at the outlet row."""
    x0, y0, x1, _ = _extent(params)
    line = LineString([(x0, y0), (x1, y0)])
    gdf = gpd.GeoDataFrame({"bc_id": [1]}, geometry=[line], crs=f"EPSG:{params.epsg}")
    gdf.to_file(dest, driver="GeoJSON", engine="pyogrio")
    return dest
