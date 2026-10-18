"""The per-sim WSE colorbar excludes the wall cells the user's config defines.

The per-sim peak-flood-depth figure clips its water-surface-elevation colorbar to the
range of wetted cells, and it drops cells whose DEM elevation marks them as a
building or wall so that rooftop values do not stretch the scale. Which cells count
as walls depends on two system-config sentinel heights and on the report config's
`system_map.elevation_panel.wall_threshold_buffer_m`. A unit test of the threshold
helper alone cannot see a call site that reads the buffer from the wrong place or
adds its own condition on the sentinels, so this test drives the call site in
`_shared_wse_range` and reads back the range it returns.

The analysis is a stand-in object carrying only what that function reads. It
compiles no solver and runs no simulation.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import hhemt.report_renderers.per_sim_peak_flood_depth as pfd
from hhemt.config.report import ElevationPanelStyle

# Three wetted cells, 1 m deep, on ground at 10 m, 50 m and 75 m, so their WSE values
# are 11, 51 and 76 m. With a 10 m buffer and an 80 m sentinel the threshold is 70 m,
# so only the 75 m cell is a wall; a 40 m buffer would also drop the 50 m cell.
_DEM = xr.DataArray(np.array([[10.0, 50.0, 75.0]]), dims=("y", "x"), coords={"y": [0.0], "x": [0.0, 1.0, 2.0]})
_DEPTH = xr.DataArray(
    np.ones((1, 1, 3)), dims=("event_iloc", "y", "x"), coords={"event_iloc": [0], "y": [0.0], "x": [0.0, 1.0, 2.0]}
)
_GROUP = "/triton_only/triton"


class _Tree:
    """The consolidated tree: one model group carrying the per-event peak water level."""

    groups = (_GROUP,)

    def __getitem__(self, group):
        return SimpleNamespace(to_dataset=lambda: xr.Dataset({"max_wlevel_m": _DEPTH}))


def _analysis(building, outside, buffer_m):
    """A stand-in analysis carrying the sentinel heights and the report-config buffer."""
    return SimpleNamespace(
        _get_enabled_model_types=lambda: ["triton"],
        _system=SimpleNamespace(
            cfg_system=SimpleNamespace(
                watershed_gis_polygon="watershed.shp",
                dem_building_height=building,
                dem_outside_watershed_height=outside,
            )
        ),
        cfg_analysis=SimpleNamespace(
            report=SimpleNamespace(
                system_map=SimpleNamespace(elevation_panel=ElevationPanelStyle(wall_threshold_buffer_m=buffer_m))
            )
        ),
        process=SimpleNamespace(open_datatree=lambda: _Tree()),
        df_sims=SimpleNamespace(index=[0]),
    )


@pytest.mark.parametrize(
    ("building", "outside", "expected"),
    [
        pytest.param(80.0, 100.0, (11.0, 51.0), id="both-set"),
        pytest.param(80.0, None, (11.0, 51.0), id="building-only"),
        pytest.param(None, 80.0, (11.0, 51.0), id="outside-only"),
        pytest.param(None, None, (11.0, 76.0), id="neither-set"),
    ],
)
def test_shared_wse_range_drops_the_configured_wall_cells(building, outside, expected, monkeypatch):
    """Property: the WSE range the per-sim figure shares across events leaves out
    exactly the cells at or above the smaller set sentinel height less the buffer
    the user configured, and leaves out nothing when neither sentinel is set.

    Class: the four sentinel states (both set, building height only, outside-watershed
    height only, neither), each with a non-default 10 m buffer read from the
    analysis's report config at `system_map.elevation_panel`. The assertion reads the
    returned (min, max) range, which the function returns before and after the change.

    Why required: the overview map paints walls from whichever sentinel is set, less
    the configured buffer, and the per-sim map's colorbar must exclude the same cells;
    otherwise a rooftop WSE stretches the scale on one map and not on the other.

    What kills it: the call site keeping its own both-sentinels-set condition (the
    one-sentinel cases return (11, 76)); passing a default `ElevationPanelStyle()`
    or a constant 40 m instead of the configured panel (the threshold drops to 40 m,
    the 50 m cell goes too, and the range collapses to None); or reading the buffer
    from a field the config does not declare under `except AttributeError`.

    A second correct implementation that still passes: computing the threshold
    inline in `_shared_wse_range` from the two heights and the configured buffer
    without calling `_wall_threshold_m`.
    """
    monkeypatch.setattr(pfd.gpd, "read_file", lambda path: SimpleNamespace(crs=None))
    monkeypatch.setattr(pfd.utils, "create_mask_from_shapefile", lambda da, path: xr.ones_like(da, dtype=bool))
    map_cfg = SimpleNamespace(dry_threshold_m=0.0, wse_clip_quantile_lower=0.0, wse_clip_quantile_upper=1.0)

    result = pfd._shared_wse_range(_analysis(building, outside, 10.0), None, _DEM, map_cfg=map_cfg)

    assert result == pytest.approx(expected)
