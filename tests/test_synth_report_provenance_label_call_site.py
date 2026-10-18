"""The per-sim depth panel's manifest label carries the values the figure was drawn with.

The Plotly peak-flood-depth figure records, for its depth panel, a provenance
`transform` describing how the plotted raster was produced: the dry threshold below
which cells were masked, and the datashader canvas a large raster was pre-aggregated
onto. Both come from the report config. A test of the function that formats the label
cannot see the figure builder handing it the wrong values -- a default config, or the
canvas of some other setting -- so this test drives the figure builder itself and reads
the label off the provenance reference it constructs.

Every input the builder reads is replaced by a small in-memory stand-in, and the run is
stopped as soon as the depth panel's provenance reference is built. It compiles no
solver and runs no simulation.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

import hhemt.config.report as report_mod
import hhemt.report_renderers._hydrology_panel as hydrology_mod
import hhemt.report_renderers._provenance as provenance_mod
import hhemt.report_renderers.per_sim_peak_flood_depth as pfd
import hhemt.scenario as scenario_mod
from hhemt.config.report import report_config

_GROUP = "/triton_only/triton"
_COORDS = {"y": [3.0, 2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]}


class _StopAtDepthPanel(Exception):
    """Raised by the recording ProvenanceRef once the depth panel's reference exists."""


class _Tree:
    """A consolidated tree with one group holding the peak water level of event 0."""

    groups = (_GROUP,)

    def __getitem__(self, group):
        depth = xr.DataArray(
            np.full((1, 4, 4), 0.5), dims=("event_iloc", "y", "x"), coords={"event_iloc": [0], **_COORDS}
        )
        return SimpleNamespace(to_dataset=lambda: xr.Dataset({"max_wlevel_m": depth}))


def _analysis(tmp_path):
    """A stand-in analysis carrying only what the builder reads before the depth panel."""
    return SimpleNamespace(
        _system=SimpleNamespace(
            sys_paths=SimpleNamespace(dem_processed=str(tmp_path / "dem.tif")),
            cfg_system=SimpleNamespace(watershed_gis_polygon=str(tmp_path / "watershed.shp")),
        ),
        analysis_paths=SimpleNamespace(
            analysis_datatree_zarr=str(tmp_path / "analysis_datatree.zarr"), analysis_dir=str(tmp_path)
        ),
        process=SimpleNamespace(open_datatree=lambda: _Tree()),
        cfg_analysis=SimpleNamespace(weather_timeseries=str(tmp_path / "weather.nc")),
        _retrieve_weather_indexer_using_integer_index=lambda iloc: {"event": iloc},
    )


def _report_cfg(dry_threshold_m, canvas, threshold_cells):
    """The default report config with the three settings this test varies."""
    data = report_config().model_dump()
    data["per_sim"]["map"]["dry_threshold_m"] = dry_threshold_m
    data["per_sim"]["interactive"]["datashader_canvas_size"] = canvas
    data["per_sim"]["interactive"]["datashader_threshold_cells"] = threshold_cells
    return report_config.model_validate(data)


@pytest.fixture
def depth_transform(monkeypatch, tmp_path):
    """Run the figure builder up to the depth panel and return its provenance transform."""
    captured = []

    def _recording_ref(**kwargs):
        captured.append(kwargs)
        raise _StopAtDepthPanel

    dem = xr.DataArray(np.arange(16.0).reshape(4, 4), dims=("y", "x"), coords=_COORDS)
    monkeypatch.setattr(provenance_mod, "ProvenanceRef", _recording_ref)
    monkeypatch.setattr(report_mod, "resolve_target_crs", lambda analysis, cfg: CRS.from_epsg(32618))
    monkeypatch.setattr(pfd.rxr, "open_rasterio", lambda path: dem)
    monkeypatch.setattr(pfd.gpd, "read_file", lambda path: SimpleNamespace(crs=None))
    monkeypatch.setattr(pfd.utils, "create_mask_from_shapefile", lambda da, path: xr.ones_like(da, dtype=bool))
    monkeypatch.setattr(scenario_mod, "resolve_event_window", lambda *a, **k: None)
    monkeypatch.setattr(
        hydrology_mod,
        "load_event_hydrology_data",
        lambda *a, **k: {
            "times_min": np.array([0.0, 1.0]),
            "rainfall": np.array([0.0, 0.0]),
            "bc_water_level": np.array([0.0, 0.0]),
            "rain_attrs": {},
            "bc_attrs": {},
            "rain_var": "rain",
            "bc_var": "bc",
        },
    )
    monkeypatch.setattr(pfd, "_shared_depth_max", lambda *a, **k: 1.0)
    monkeypatch.setattr(pfd, "_shared_wse_range", lambda *a, **k: (0.0, 20.0))
    monkeypatch.setattr(pfd, "_resolve_inp_sources", lambda analysis: ("hydro.inp", "hydraulics.inp"))
    monkeypatch.setattr(pfd.swmmio, "Model", lambda path: None)
    monkeypatch.setattr(pfd, "compute_padded_square_bounds", lambda *a, **k: (0.0, 0.0, 3.0, 3.0))

    def _run(report_cfg):
        with pytest.raises(_StopAtDepthPanel):
            pfd._build_peak_flood_depth_figure(
                _analysis(tmp_path), report_cfg, tmp_path / "fig.html", event_iloc=0, triton_group=_GROUP, prov=None
            )
        assert len(captured) == 1, "the builder built no depth-panel provenance reference"
        return captured.pop()["transform"]

    return _run


def test_depth_label_names_the_configured_threshold_and_canvas(depth_transform):
    """Property: when the depth raster is pre-aggregated, the depth panel's manifest
    `transform` names the dry threshold and the datashader canvas size the report
    config set, not the defaults.

    Class: a non-default dry threshold (0.05 m, default 0.0025 m) and a non-default
    canvas (64x48, default 512x512), read by the figure builder from the report config
    and handed to whatever formats the label. The assertion reads the transform off the
    ProvenanceRef the builder constructs, which it constructs before and after the change.

    Why required: the manifest is the record a reader consults to learn how a panel's
    numbers were produced, so a label naming values other than the ones the mask and the
    canvas used misreports the figure.

    What kills it: the builder passing a default config's threshold or canvas, or a
    hardcoded one, to the label; or the label ignoring either value.

    A second correct implementation that still passes: formatting the label inline at the
    ProvenanceRef call, in any wording that states both values.
    """
    transform = depth_transform(_report_cfg(0.05, (64, 48), threshold_cells=0))

    assert "0.05" in transform
    assert "64x48" in transform
    assert "512x512" not in transform


def test_depth_label_carries_no_canvas_when_the_raster_is_not_aggregated(depth_transform):
    """Property: when the depth raster is drawn cell by cell, the depth panel's manifest
    `transform` names the configured dry threshold and claims no pre-aggregation.

    Class: the same non-default threshold and canvas, with the aggregation threshold
    set above the grid's cell count so datashader does not run.

    Why required: a label claiming a datashader canvas for a raster drawn at full
    resolution tells the reader the plotted values were aggregated when they were not.

    What kills it: attaching the canvas suffix whatever the gate decided, or dropping
    the configured threshold from the label.

    A second correct implementation that still passes: the same inline formatting as
    the previous test's, omitting the aggregation clause when datashader did not run.
    """
    transform = depth_transform(_report_cfg(0.05, (64, 48), threshold_cells=10**9))

    assert "0.05" in transform
    assert "datashader" not in transform
