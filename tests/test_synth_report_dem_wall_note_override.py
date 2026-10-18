"""The static DEM panel's wall note names the colour the walls are actually painted.

The static system-overview DEM panel paints wall cells with a colormap "over" colour
and records, in its provenance log, a note saying which colour that was. The colour
has two sources: the report's elevation-panel style, and a publication figure's own
override, which wins when set. A note built from the style alone is correct for every
report render and wrong for every publication render that overrides the colour, so
this test sets an override that differs from the style and checks the note against
the colour the drawn image actually uses.

It draws one panel in memory from a small synthetic raster. It compiles no solver and
runs no simulation.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from hhemt.config.report import ElevationPanelStyle, SystemMapConfig
from hhemt.report_renderers._provenance import ProvenanceLog
from hhemt.report_renderers.system_overview import _draw_elevation_panel


def _raster():
    """A 3-row, 4-column north-up raster with 2 m cells, one of them a tall wall."""
    xs = 101.0 + 2.0 * np.arange(4)
    ys = 205.0 - 2.0 * np.arange(3)
    values = np.arange(12, dtype=float).reshape(3, 4) + 1.0
    values[0, 0] = 100.0
    da = xr.DataArray(values, coords={"y": ys, "x": xs}, dims=("y", "x"))
    return da.rio.write_crs("EPSG:32618")


def _wall_notes(prov):
    """Every provenance `transform` string that describes how walls were coloured."""
    found = []

    def walk(node):
        """Collect `transform` strings from nested dicts and lists, depth first."""
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "transform" and isinstance(value, str):
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(prov.serialize())
    return [note for note in found if "walls" in note]


def test_static_wall_note_names_the_publication_override_colour():
    """Property: when a publication override sets the wall colour, the static DEM
    panel's wall notes name that colour, and it is the colour the drawn image uses.

    Class: an override colour (cyan) that differs from the elevation-panel style's
    colour (white), so a note built from either source alone is distinguishable. The
    assertion reads the notes from the provenance log and the over colour from the
    image the panel drew, both of which the panel produces before and after the change.

    Why required: a publication figure is the one most likely to override the wall
    colour, and its provenance note must describe the figure as drawn, not as the
    report would have drawn it.

    What kills it: building the note from the elevation-panel style's colour while the
    colormap uses the override, or naming any fixed colour.

    A second correct implementation that still passes: wording the note differently,
    for example "walls painted cyan", as long as it names the colour the image uses.
    """
    da = _raster()
    map_cfg = SystemMapConfig(elevation_panel=ElevationPanelStyle(over_color="white"))
    prov = ProvenanceLog()
    fig, ax = plt.subplots()
    try:
        _draw_elevation_panel(
            ax,
            da,
            da.rio.bounds(),
            None,
            None,
            da.rio.crs,
            map_cfg,
            prov,
            dem_source="dem.tif",
            dem_over_color_override="cyan",
        )
        drawn_over = {mcolors.to_hex(im.get_cmap().get_over()) for im in ax.images}
    finally:
        plt.close(fig)

    assert mcolors.to_hex("cyan") in drawn_over, drawn_over
    notes = _wall_notes(prov)
    assert len(notes) == 2, notes
    assert all("cyan" in note for note in notes), notes
    assert not any("white" in note for note in notes), notes
