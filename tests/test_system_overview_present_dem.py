"""The system-overview renderer reads a DEM the bundle actually built.

A sensitivity bundle may overlay ``system.target_dem_resolution`` on every row, so the master system config
names a resolution (1.10 m) that no member ever processed while the system directory holds only the members'
(3.50 m) DEM and Manning's rasters. The renderer used to open the master's resolution unconditionally and died with
``RasterioIOError ... elevation_1.10m.dem: No such file or directory``.

These tests drive ``system_overview.render`` over a synthetic system directory holding one resolution, with the
Plotly SVG export (kaleido) stubbed out. They compile no solver and run no simulation.

Class: every (master resolution, set of resolutions on disk) pair the bundle can present.
Why required: a report that cannot draw its system overview loses the whole "System Information" category.
What kills it: reverting ``_system_holding_present_dem`` so ``render`` opens ``analysis._system.sys_paths`` again.
Second correct implementation: scanning ``system_directory`` for ``elevation_*m.dem`` and taking the finest one
that also has a Manning's raster passes every test here, because each asserts on the files read and not on how
they were found.
"""

import json
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import rioxarray as rxr
import xarray as xr

import hhemt.analysis  # noqa: F401  (the renderer imports it lazily; importing it here keeps that cost out of the first test)
from hhemt.config.report import report_config
from hhemt.exceptions import ProcessingError
from hhemt.report_renderers import system_overview

_EPSG = 32618


def _write_raster(path: Path, resolution: float) -> None:
    """Write a 4x5 north-up GeoTIFF of ``resolution``-metre cells under a ``.dem`` name, as TRITON's rasters are."""
    xs = 500000.0 + resolution / 2 + resolution * np.arange(5)
    ys = 4100000.0 - resolution / 2 - resolution * np.arange(4)
    da = xr.DataArray(
        np.linspace(1.0, 20.0, 20).reshape(4, 5),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
    ).rio.write_crs(f"EPSG:{_EPSG}")
    da.rio.to_raster(path, driver="GTiff")


class _System(types.SimpleNamespace):
    """A stub system whose ``mannings_rds`` opens its own Manning's raster, as the real system's does."""

    @property
    def mannings_rds(self):
        return rxr.open_rasterio(self.sys_paths.mannings_processed)


def _system(system_dir: Path, resolution: float) -> _System:
    """A system object carrying what ``render`` reads: paths and a config naming this resolution."""
    return _System(
        sys_paths=types.SimpleNamespace(
            dem_processed=system_dir / f"elevation_{resolution:.2f}m.dem",
            mannings_processed=system_dir / f"mannings_{resolution:.2f}m.dem",
        ),
        cfg_system=types.SimpleNamespace(
            target_dem_resolution=resolution,
            dem_building_height=None,
            dem_outside_watershed_height=None,
            crs=types.SimpleNamespace(horizontal_epsg=_EPSG, vertical_epsg=None),
            system_directory=system_dir,
        ),
    )


def _analysis(tmp_path: Path, master_resolution: float, member_resolutions: tuple[float, ...] = ()):
    """An analysis stub: a master system at ``master_resolution`` and, when given, sensitivity targets beside it."""
    system_dir = tmp_path / "system"
    system_dir.mkdir(exist_ok=True)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    targets = [types.SimpleNamespace(system=_system(system_dir, r), member_systems={}) for r in member_resolutions]
    sensitive = bool(member_resolutions)
    return types.SimpleNamespace(
        cfg_analysis=types.SimpleNamespace(
            toggle_sensitivity_analysis=sensitive,
            toggle_external_boundaries=False,
            external_boundary_gis=None,
            analysis_id="aware",
        ),
        _system=_system(system_dir, master_resolution),
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
        sensitivity=types.SimpleNamespace(unique_system_targets=targets, members={}),
        _get_enabled_model_types=lambda: ["triton"],
    )


def _built(tmp_path: Path, *resolutions: float) -> None:
    """Write the DEM and Manning's rasters a processing step leaves for each resolution."""
    system_dir = tmp_path / "system"
    system_dir.mkdir(exist_ok=True)
    for r in resolutions:
        _write_raster(system_dir / f"elevation_{r:.2f}m.dem", r)
        _write_raster(system_dir / f"mannings_{r:.2f}m.dem", r)


@pytest.fixture(autouse=True)
def _no_kaleido_no_build_stamp(monkeypatch):
    """Stub the SVG export (kaleido) and the build-identity stamp, neither of which is the subject of these tests.

    The Plotly branch then runs without a headless browser, and the manifest sidecar does not shell out to git and
    re-parse ``pyproject.toml`` on every render.
    """
    import plotly.graph_objects as go

    from hhemt import provenance

    monkeypatch.setattr(go.Figure, "write_image", lambda self, *a, **k: None)
    monkeypatch.setattr(
        provenance,
        "producing_stamp",
        lambda: {"hhemt_sha": "deadbeef0000", "hhemt_version": "0+test", "hhemt_dirty": "false"},
    )


def _render(analysis, tmp_path: Path) -> dict:
    """Render the system overview and return the emitted manifest."""
    out = tmp_path / "plots" / "system_overview.html"
    out.parent.mkdir(exist_ok=True)
    system_overview.render(analysis, report_config(), out)
    return json.loads(out.with_suffix(".manifest.json").read_text())


def _declared_sources(manifest: dict) -> list[str]:
    """The source paths the manifest declares for the figure."""
    return [str(s) for s in manifest["source_paths_relative"]]


def test_bundle_with_only_member_resolution_renders_from_that_resolution(tmp_path):
    """The aware-bundle defect: master names 1.10 m, only the members' 3.50 m rasters exist, and 3.50 m is drawn."""
    _built(tmp_path, 3.5)
    analysis = _analysis(tmp_path, master_resolution=1.1, member_resolutions=(3.5, 3.5))

    manifest = _render(analysis, tmp_path)

    sources = _declared_sources(manifest)
    assert any(s.endswith("elevation_3.50m.dem") for s in sources), sources
    assert not any("1.10m" in s for s in sources), sources
    titles = [p["title"] for p in manifest["renderer_data"]["panels"]]
    assert "Manning's n (3.5m)" in titles, titles


def test_master_resolution_is_kept_when_its_rasters_exist(tmp_path):
    """A built master resolution is drawn as before, even beside finer member resolutions."""
    _built(tmp_path, 1.1, 3.5)
    analysis = _analysis(tmp_path, master_resolution=3.5, member_resolutions=(1.1, 3.5))

    manifest = _render(analysis, tmp_path)

    sources = _declared_sources(manifest)
    assert any(s.endswith("elevation_3.50m.dem") for s in sources), sources
    assert not any("1.10m" in s for s in sources), sources


def test_finest_member_resolution_is_drawn_when_master_is_absent_and_several_exist(tmp_path):
    """With the master's resolution unbuilt and two member resolutions built, the finest built grid is drawn."""
    _built(tmp_path, 1.1, 3.5)
    analysis = _analysis(tmp_path, master_resolution=2.0, member_resolutions=(3.5, 1.1))

    manifest = _render(analysis, tmp_path)

    sources = _declared_sources(manifest)
    assert any(s.endswith("elevation_1.10m.dem") for s in sources), sources
    assert not any("3.50m" in s for s in sources), sources


def test_a_resolution_with_a_dem_but_no_mannings_raster_is_not_chosen(tmp_path):
    """The overview draws both rasters of one grid, so a DEM with no Manning's raster is skipped for a complete pair."""
    _built(tmp_path, 3.5)
    _write_raster(tmp_path / "system" / "elevation_1.10m.dem", 1.1)
    analysis = _analysis(tmp_path, master_resolution=1.1, member_resolutions=(1.1, 3.5))

    manifest = _render(analysis, tmp_path)

    sources = _declared_sources(manifest)
    assert any(s.endswith("elevation_3.50m.dem") for s in sources), sources
    assert not any("1.10m" in s for s in sources), sources


def test_no_built_resolution_fails_naming_the_dems_present_and_the_ones_asked_for(tmp_path):
    """A system directory with no complete DEM/Manning's pair raises a ProcessingError that names the DEMs present."""
    system_dir = tmp_path / "system"
    system_dir.mkdir()
    _write_raster(system_dir / "elevation_5.00m.dem", 5.0)  # a stray DEM with no Manning's raster
    analysis = _analysis(tmp_path, master_resolution=1.1, member_resolutions=(3.5,))

    with pytest.raises(ProcessingError) as excinfo:
        _render(analysis, tmp_path)

    message = str(excinfo.value)
    assert "elevation_5.00m.dem" in message, message
    assert "elevation_1.10m.dem" in message and "elevation_3.50m.dem" in message, message
