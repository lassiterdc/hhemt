"""A bundle carries a shapefile input's reader sidecars, and ingest refuses a shapefile that lacks them.

No solver. A synthetic shapefile is written with geopandas (``.shp .shx .dbf .prj .cpg``) beside decoys (``.sbn``,
``.shp.xml``); the real ``_copy_declared_inputs`` and ``_harvest_sources`` carry it, and the real
``_assert_declared_inputs_exist`` gates a reconstituted bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from hhemt.bundle import _emit
from hhemt.exceptions import ConfigurationError, ProcessingError
from tests.test_publish_deposit_paths import _world


def _write_shapefile(directory: Path, name: str = "boundary") -> Path:
    """A shapefile at ``directory/{name}.shp`` with its sidecars and two decoys that must not be carried."""
    directory.mkdir(parents=True, exist_ok=True)
    shp = directory / f"{name}.shp"
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]}, geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 0)])], crs=32147
    )
    gdf.to_file(shp)
    (directory / f"{name}.sbn").write_bytes(b"index")
    (directory / f"{name}.shp.xml").write_text("<metadata><author>someone</author></metadata>")
    return shp


def _names(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()}


def _emit_copy(tmp_path: Path, shp: Path) -> tuple[Path, Path]:
    world = _world(tmp_path, system_over={"watershed_gis_polygon": str(shp)})
    staging = tmp_path / "staging"
    staging.mkdir()
    _emit._copy_declared_inputs(world.analysis, staging)
    carried = next(staging.rglob("boundary.shp"))
    return staging, carried


def test_the_declared_input_copy_carries_the_reader_sidecars_and_nothing_else(tmp_path):
    """Property: the copy of a declared `.shp` holds `.shx .dbf .prj .cpg` beside it, no `.sbn` and no `.shp.xml`, and
    geopandas reads the carried copy back equal to the source.

    Must-fire control: with the sidecar helper returning nothing (the former single-file copy) the copied `.shp`
    cannot be read by geopandas.
    Kills it: a copy that carries the `.shp` alone, or that carries the metadata or index files.
    """
    shp = _write_shapefile(tmp_path / "src")
    _staging, carried = _emit_copy(tmp_path, shp)
    names = _names(carried.parent)
    assert {"boundary.shp", "boundary.shx", "boundary.dbf", "boundary.prj", "boundary.cpg"} <= names
    assert "boundary.sbn" not in names and "boundary.shp.xml" not in names
    assert gpd.read_file(carried).equals(gpd.read_file(shp))


def test_without_the_sidecar_helper_the_carried_copy_is_unreadable(tmp_path, monkeypatch):
    """Control for the test above: the former behaviour (one file) yields a copy geopandas cannot open."""
    shp = _write_shapefile(tmp_path / "src")
    monkeypatch.setattr(_emit, "shapefile_sidecars", lambda path: [])
    _staging, carried = _emit_copy(tmp_path, shp)
    assert {n for n in _names(carried.parent) if n.startswith("boundary")} == {"boundary.shp"}
    with pytest.raises(Exception, match="shx|boundary"):
        gpd.read_file(carried)


def test_the_renderer_source_fallback_carries_the_sidecars_through_the_same_helper(tmp_path):
    """Property: a `.shp` declared as a renderer source outside the analysis directory is planned as
    `external/boundary.shp` with the same sidecars, and no `.sbn` or `.shp.xml`."""
    shp = _write_shapefile(tmp_path / "outside")
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    entries, absent, _shallow = _emit._harvest_sources({"plot": [shp]}, analysis_dir)
    assert not absent
    arcnames = {k for k in entries}
    assert {f"external/boundary.{e}" for e in ("shp", "shx", "dbf", "prj", "cpg")} <= arcnames
    assert not [a for a in arcnames if a.endswith((".sbn", ".xml"))]


def test_an_upper_case_sidecar_extension_is_carried_under_its_own_spelling(tmp_path):
    """Property: sidecar extensions match case-insensitively (a `.SHX` beside a `.shp` is carried)."""
    shp = _write_shapefile(tmp_path / "src")
    (tmp_path / "src" / "boundary.shx").rename(tmp_path / "src" / "boundary.SHX")
    assert {p.name for p in _emit.shapefile_sidecars(shp)} >= {"boundary.SHX", "boundary.dbf"}


def test_emit_refuses_a_shapefile_without_its_shx_or_dbf_naming_it(tmp_path):
    """Property: a `.shp` lacking `.shx` (or `.dbf`) at emit time is refused, naming the file."""
    shp = _write_shapefile(tmp_path / "src")
    (tmp_path / "src" / "boundary.shx").unlink()
    with pytest.raises(ConfigurationError, match=r"boundary\.shp.*\.shx"):
        _emit.shapefile_sidecars(shp)
    shp2 = _write_shapefile(tmp_path / "src2")
    (tmp_path / "src2" / "boundary.dbf").unlink()
    with pytest.raises(ConfigurationError, match=r"\.dbf"):
        _emit.shapefile_sidecars(shp2)


def _ingest_root(tmp_path: Path, shp: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(exist_ok=True)
    (root / "bundle_manifest.json").write_text(json.dumps({"analysis_id": "x"}))
    (root / "cfg_system.yaml").write_text(f"watershed_gis_polygon: {shp}\n")
    (root / "cfg_analysis.yaml").write_text("weather_timeseries: null\n")
    return root


def test_ingest_fails_closed_naming_a_shapefile_without_its_shx(tmp_path):
    """Property: a reconstituted `.shp` that lacks its `.shx` makes the declared-input gate raise, naming the file;
    with the sidecars present the same gate passes (control)."""
    from hhemt.experiments import TRITON_SWMM_experiment as Exp

    shp = _write_shapefile(tmp_path / "bundle" / "external")
    root = _ingest_root(tmp_path, shp)
    Exp._assert_declared_inputs_exist(root / "cfg_system.yaml", root / "cfg_analysis.yaml")
    (shp.parent / "boundary.shx").unlink()
    with pytest.raises(ProcessingError, match=r"boundary\.shp.*\.shx"):
        Exp._assert_declared_inputs_exist(root / "cfg_system.yaml", root / "cfg_analysis.yaml")
