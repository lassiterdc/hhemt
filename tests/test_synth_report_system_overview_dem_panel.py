"""The system-overview DEM and Manning's panels record and draw what the raster holds.

Two properties are checked on a small synthetic raster. First, the provenance note
for the DEM colour channel names the colour walls are actually painted, which is
the configurable `ElevationPanelStyle.over_color`; a note naming a fixed colour
goes false when a user changes it. Second, the interactive heatmaps place each
cell at the raster's own cell centre, so the drawn map covers exactly the raster's
extent rather than reaching half a cell past it on every side.

These tests build panels in memory from a three-by-four raster. They compile no
solver and run no simulation.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from plotly.subplots import make_subplots

from hhemt.config.report import ElevationPanelStyle, SystemMapConfig
from hhemt.report_renderers._provenance import ProvenanceLog
from hhemt.report_renderers.system_overview import (
    _draw_elevation_panel,
    _draw_elevation_panel_plotly,
    _draw_mannings_panel_plotly,
)

_RES = 2.0


def _raster():
    """A 3-row, 4-column north-up raster with 2 m cells; the outer edges are x 100..108, y 200..206."""
    xs = 100.0 + _RES / 2 + _RES * np.arange(4)
    ys = 206.0 - _RES / 2 - _RES * np.arange(3)
    da = xr.DataArray(np.arange(12, dtype=float).reshape(3, 4) + 1.0, coords={"y": ys, "x": xs}, dims=("y", "x"))
    return da.rio.write_crs("EPSG:32618")


def _transforms(prov):
    """Every `transform` string the provenance log recorded, wherever it sits in the record."""
    found = []

    def walk(node):
        """Collect `transform` strings from nested dicts and lists, depth first."""
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "transform" and isinstance(v, str):
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(prov.serialize())
    return found


@pytest.mark.parametrize("over_color", ["white", "magenta"], ids=["default", "non-default"])
def test_dem_wall_note_names_the_configured_over_color(over_color):
    """Property: the DEM colour channel's provenance note names the colour walls are
    painted with, in both the static and the interactive panel.

    Class: the default `over_color` and a non-default one. The default differs from
    the colour the note used to state, so both cases discriminate.

    Why required: the note is a reader's record of how the panel was coloured, and a
    note naming one colour while the walls are painted another misreports the figure.

    What kills it: a fixed colour word in any of the four wall notes.

    A second correct implementation that still passes: wording the note differently
    (for example "walls painted magenta") as long as it names the configured colour.
    """
    da = _raster()
    map_cfg = SystemMapConfig(elevation_panel=ElevationPanelStyle(over_color=over_color))

    prov = ProvenanceLog()
    fig, ax = plt.subplots()
    _draw_elevation_panel(ax, da, da.rio.bounds(), None, None, da.rio.crs, map_cfg, prov, dem_source="dem.tif")
    plt.close(fig)
    prov_plotly = ProvenanceLog()
    _draw_elevation_panel_plotly(
        make_subplots(rows=1, cols=1),
        da,
        da.rio.bounds(),
        None,
        None,
        da.rio.crs,
        map_cfg,
        prov_plotly,
        dem_source="dem.tif",
        col=1,
        dem_building_height=10.0,
    )

    wall_notes = [t for t in _transforms(prov) + _transforms(prov_plotly) if "walls" in t]
    assert len(wall_notes) == 4, wall_notes
    assert all(over_color in t for t in wall_notes), wall_notes
    assert not any("grey" in t for t in wall_notes), wall_notes


def _heatmap_extents(fig):
    """For each heatmap trace: (first x centre, x spacing, first y centre, y spacing)."""
    return [(t.x0, t.dx, t.y0, t.dy) for t in fig.data if t.type == "heatmap"]


def test_plotly_heatmaps_place_cells_at_the_raster_cell_centres():
    """Property: each interactive heatmap's first cell centre and cell spacing equal
    the raster's own, so the drawn map spans exactly the raster's extent.

    Class: both heatmap-drawing panels — elevation (its modeled-area and walls layers)
    and Manning's n — on a north-up raster whose cell size is not 1.

    Why required: the heatmaps were spaced by dividing the raster's OUTER edges into
    n points, which put the first centre on the edge and stretched the spacing to
    (x1 - x0) / (n - 1), so each map drew one cell larger than the raster.

    What kills it: placing cells from `rio.bounds()` edges, or any spacing other than
    the raster's resolution.

    A second correct implementation that still passes: computing the centres from
    `rio.bounds()` plus half a cell, with spacing from `rio.resolution()`.
    """
    da = _raster()
    expected = (float(da.x[0]), _RES, float(da.y[0]), -_RES)
    map_cfg = SystemMapConfig()

    fig = make_subplots(rows=1, cols=1)
    _draw_elevation_panel_plotly(
        fig,
        da,
        da.rio.bounds(),
        None,
        None,
        da.rio.crs,
        map_cfg,
        ProvenanceLog(),
        dem_source="dem.tif",
        col=1,
        dem_building_height=10.0,
    )
    fig_m = make_subplots(rows=1, cols=1)
    _draw_mannings_panel_plotly(fig_m, da, "mannings.tif", None, ProvenanceLog(), row=1, col=1)

    extents = _heatmap_extents(fig) + _heatmap_extents(fig_m)
    assert len(extents) == 3, extents
    for got in extents:
        assert got == pytest.approx(expected), (got, expected)
