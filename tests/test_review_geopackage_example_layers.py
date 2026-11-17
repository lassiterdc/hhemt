"""Reviewer tests for change 1116-1523: the Norfolk example reads its two map layers as GeoPackage files.

Written by the reviewing seat, independently of the coder's tests. Nothing here compiles or runs a solver.
The staged layers live outside the repository (``~/hydroshare_staging_norfolk/gpkg``), so the tests that read
them skip where that directory is absent.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import geopandas as gpd
import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_CASE = _REPO / "test_data" / "norfolk_coastal_flooding"
_STAGED = Path.home() / "hydroshare_staging_norfolk" / "gpkg"
_LAYERS = {
    "variable_bc.gpkg": "d0dd01c87c2d3b3312d51117b4d72711e7357f4e1ec1293adcb97c96484b10b8",
    "norfolk_wshed_epsg32147_state_plane_m.gpkg": "ffd624e1d7b0bafac5941f444665d937ef30328867376684fa0f52452cb9129b",
}
_needs_staged = pytest.mark.skipif(not _STAGED.is_dir(), reason="the staged GeoPackage layers are not on this machine")


def _yaml(name: str) -> dict:
    return yaml.safe_load((_CASE / name).read_text())


def test_both_template_fields_name_the_two_gpkg_basenames_under_the_external_folder():
    """Property: each field holds ``${DATA_DIR}/external/<layer>.gpkg`` for exactly the published layer names."""
    assert _yaml("template_system_config.yaml")["watershed_gis_polygon"] == (
        "${DATA_DIR}/external/norfolk_wshed_epsg32147_state_plane_m.gpkg"
    )
    assert _yaml("template_analysis_config.yaml")["external_boundary_gis"] == "${DATA_DIR}/external/variable_bc.gpkg"


def test_the_manifest_holds_exactly_twelve_rows_and_both_layers_at_the_stated_digests():
    """Property: the manifest set and the two layer digests, compared against a literal list typed here."""
    manifest = _yaml("case.yaml")["manifest"]
    assert len(manifest) == 12
    layer_rows = {k.rsplit("/", 1)[1]: v for k, v in manifest.items() if k.endswith(".gpkg")}
    assert layer_rows == _LAYERS
    # Control that varies the property: no row carries a shapefile-family suffix.
    assert [k for k in manifest if re.search(r"\.(shp|shx|dbf|prj|cpg|vrt)$", k)] == []
    assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in manifest.values())


@_needs_staged
@pytest.mark.parametrize("name", sorted(_LAYERS))
def test_the_staged_file_bytes_hash_to_the_manifest_digest(name):
    """Property: the file the toolkit would read has the digest the manifest pins."""
    digest = hashlib.sha256((_STAGED / name).read_bytes()).hexdigest()
    assert digest == _LAYERS[name]
    assert _yaml("case.yaml")["manifest"]["data/contents/external/" + name] == digest


@_needs_staged
def test_the_boundary_layer_reads_through_geopandas_with_type_outfalls_and_epsg_32147():
    """Property: ``gpd.read_file`` returns the columns the analysis template names and the template's CRS."""
    analysis = _yaml("template_analysis_config.yaml")
    layer = gpd.read_file(_STAGED / "variable_bc.gpkg")
    assert analysis["external_boundary_type_attribute"] in layer.columns
    assert analysis["water_level_boundary_outfall_attribute"] in layer.columns
    assert {"type", "outfalls"} <= set(layer.columns)
    assert layer.crs.to_epsg() == _yaml("template_system_config.yaml")["crs"]["horizontal_epsg"] == 32147
    assert list(layer["type"]) == ["water_level"]
    assert set(layer["outfalls"].iloc[0].split()) == {"E143250", "E147007"}
    # Control: the property fires on a frame lacking the columns.
    assert "type" not in layer.drop(columns=["type"]).columns


@_needs_staged
def test_the_watershed_layer_reads_through_geopandas_as_one_polygon_layer_in_epsg_32147():
    """Property: the watershed GeoPackage yields polygon geometry in the CRS the system template declares."""
    layer = gpd.read_file(_STAGED / "norfolk_wshed_epsg32147_state_plane_m.gpkg")
    assert len(layer) >= 1
    assert layer.crs.to_epsg() == _yaml("template_system_config.yaml")["crs"]["horizontal_epsg"]
    assert set(layer.geom_type) <= {"Polygon", "MultiPolygon"}


def test_no_tracked_file_outside_the_records_names_a_removed_layer_file_or_the_old_layer_folders():
    """Property: a different matcher (regex, per file) finds no shapefile-family name of either layer.

    Population: tracked text under the whole repository less the two historical sets (rendered bundle reports,
    zarr reference stores). Unit: occurrences, ``git grep -o``.
    """
    names = [
        f"{stem}.{ext}"
        for stem in ("variable_bc", "norfolk_wshed_epsg32147_state_plane_m")
        for ext in ("shp", "shx", "dbf", "prj", "cpg")
    ] + ["shp_watershed/", "shp_boundary_condition/"]
    flags = [arg for name in names for arg in ("-e", name)]
    base = ["git", "grep", "-I", "-o", "-F", *flags, "--", "tests", "docs", "src", "test_data"]
    found = subprocess.run(
        [*base, ":!docs/reference/bundles", ":!test_data/swmm_refactoring_reference/*.zarr/*"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    if found.returncode == 128:
        pytest.skip("not a git work tree")
    # This file's own docstring and the older retired-layout assertions are the only places the old folder names
    # appear on purpose.
    allowed = (
        "tests/test_review_geopackage_example_layers.py",
        "tests/test_experiments_hydroshare_bag_validation.py",
        "tests/test_review_m1_case_manifest.py",
    )
    stray = [line for line in found.stdout.splitlines() if not line.startswith(allowed)]
    assert stray == []
    # Control that varies the property: the same flags fire on a tracked file that holds an old folder name.
    base_control = ["git", "grep", "-I", "-o", "-F", *flags, "--", "tests/test_review_m1_case_manifest.py"]
    control = subprocess.run(base_control, cwd=_REPO, capture_output=True, text=True).stdout
    assert "shp_watershed/" in control
