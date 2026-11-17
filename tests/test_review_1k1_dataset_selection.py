"""Reviewer tests for chunk 1k.1: the Norfolk case dataset chosen from a reprex bundle.

The dataset itself lives outside the repository. What a laptop run can hold is the
dataset's FILE LIST (12 files under ``data/contents/external/``: the two map layers are
GeoPackage files), against the production surfaces that name dataset files: the Norfolk config
templates and every tracked text file under ``tests``, ``docs``, ``src`` and ``test_data``.
Nothing here compiles or runs a solver and nothing reads the dataset's bytes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_CASE_DIR = _REPO / "test_data" / "norfolk_coastal_flooding"

# The staged dataset's members by basename, as listed by `find <stage> -type f -printf "%f\n"`
# over the staged tree (12 entries). Basenames, not relative paths, so the pin survives the folder
# layout the case fetch path settles on.
_STAGED_BASENAMES = frozenset(
    {
        "b_sub_ids_and_mrms_rain_col.csv",
        "event_windows_observed.csv",
        "fullres_dem_m.tif",
        "hague_model.inp",
        "hurricane_irene_event_index.csv",
        "landuse_clipped_to_triton_extent.tif",
        "landuse_lookup_tables.csv",
        "norfolk_wshed_epsg32147_state_plane_m.gpkg",
        "observed_event_events.csv",
        "observed_events_named_subset.csv",
        "obs_event_tseries_from_yrs_with_complete_coverage.nc",
        "variable_bc.gpkg",
    }
)

# The one `${DATA_DIR}` value that no dataset file backs: the case loaders overwrite it with an
# event-index CSV name before the config is read.
_LOADER_FILLED_PLACEHOLDER = "PLACEHOLDER.csv"


def _data_dir_basenames(template_name: str) -> list[str]:
    """Basename of every uncommented ``${DATA_DIR}/...`` value in one Norfolk config template."""
    cfg = yaml.safe_load((_CASE_DIR / template_name).read_text())
    values = [v for v in cfg.values() if isinstance(v, str) and v.startswith("${DATA_DIR}/")]
    return [Path(v).name for v in values]


@pytest.mark.parametrize("template_name", ["template_system_config.yaml", "template_analysis_config.yaml"])
def test_every_template_data_dir_path_names_a_staged_dataset_file(template_name: str) -> None:
    """Quantifies over: every ``${DATA_DIR}/...`` value of a Norfolk config template.

    Each must name, by basename, a file the staged dataset carries, so a reader who fetches the
    dataset never meets a template path that resolves to nothing. The assertion turns on the
    basename, a property that holds both under the current folder layout and under any layout the
    fetch path later settles on, so it does not reward one layout.

    Why required: the fetch path expands ``${DATA_DIR}`` to the extracted resource's data folder
    and nothing else supplies the file, so a template naming a file absent from the dataset
    fails at the first existence check of config load.

    Killed by: adding to a template a ``${DATA_DIR}`` value whose basename is not in the dataset
    (for example a file the dataset's bundle did not carry).
    """
    basenames = [b for b in _data_dir_basenames(template_name) if b != _LOADER_FILLED_PLACEHOLDER]
    assert basenames, f"{template_name} carries no ${{DATA_DIR}} value, so the matcher read nothing"
    absent = sorted(set(basenames) - _STAGED_BASENAMES)
    assert not absent, f"{template_name} names dataset files the staged dataset lacks: {absent}"


def test_the_two_template_layer_fields_name_geopackage_files() -> None:
    """Quantifies over: the template fields that name a vector layer.

    Those are ``watershed_gis_polygon`` and ``external_boundary_gis``.

    Each must name a ``.gpkg`` file, because both Norfolk layers are projected in EPSG:32147 and a GeoPackage
    carries that projection in one file, where a shapefile needs four sidecars and a reprojecting GeoJSON would
    move the coordinates the boundary validators compare against the model.

    Why required: the published resource holds the layers as ``.gpkg`` only, so a template naming any other
    suffix resolves to nothing at the first existence check of config load.

    Killed by: pointing either field back at a shapefile.
    A second correct implementation: any other layer format the resource gains later, with this suffix tuple
    widened, since the assertion turns on the suffix and not on the folder.
    """
    fields = {
        "template_system_config.yaml": "watershed_gis_polygon",
        "template_analysis_config.yaml": "external_boundary_gis",
    }
    named = {
        field: Path(yaml.safe_load((_CASE_DIR / template).read_text())[field]).suffix
        for template, field in fields.items()
    }
    assert named == {"watershed_gis_polygon": ".gpkg", "external_boundary_gis": ".gpkg"}


_LAYER_STEMS = ("variable_bc", "norfolk_wshed_epsg32147_state_plane_m")
_REMOVED_SHAPEFILE_SUFFIXES = ("shp", "shx", "dbf", "prj", "cpg")
# Rendered bundle reports are historical records, and the zarr reference stores record the paths of the run that
# produced them; neither is edited.
_HISTORICAL = (":!docs/reference/bundles", ":!test_data/swmm_refactoring_reference/*.zarr/*")


def _layer_file_mentions(suffixes: tuple[str, ...]) -> list[tuple[str, str]]:
    """Every ``(tracked path, "<stem>.<suffix>")`` mention of a Norfolk layer file, one per occurrence.

    One fixed-string ``git grep -o`` call (a regex alternation measured 2.5 times slower) over ``tests``, ``docs``,
    ``src`` and ``test_data`` less the historical records, so the population is the tracked text of those trees and
    the unit is the occurrence.
    """
    names = [f"{stem}.{ext}" for stem in _LAYER_STEMS for ext in (*suffixes, "gpkg")]
    found = subprocess.run(
        [
            "git",
            "grep",
            "-I",
            "-o",
            "-F",
            *(arg for name in names for arg in ("-e", name)),
            "--",
            "tests",
            "docs",
            "src",
            "test_data",
            *_HISTORICAL,
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
    )
    if found.returncode == 128:
        pytest.skip("not a git work tree")
    return [tuple(line.split(":", 1)) for line in found.stdout.splitlines()]


def test_no_tracked_file_names_a_removed_shapefile_of_either_norfolk_layer() -> None:
    """Quantifies over: every tracked file under ``tests``, ``docs``, ``src`` and ``test_data``, less the records.

    The historical records are the rendered bundle reports and the zarr reference stores.

    None may name a file of either removed layer, that is a stem of the two layers followed by a shapefile
    suffix, because the resource no longer holds those files and a test, doc or template that names one
    describes a download that fails. The pattern is built from parts, so this file does not name a removed file.

    Must-fire control: the same call also matches the GeoPackage suffix, and it must name this file and a
    template for the files that exist, so a zero for the removed suffixes is not an instrument that reads nothing.
    Killed by: naming ``<stem>.<shapefile suffix>`` in any tracked file of the four trees.
    A second correct implementation: walking the tracked file list in Python and matching per line.
    """
    mentions = _layer_file_mentions(_REMOVED_SHAPEFILE_SUFFIXES)
    removed = [(path, name) for path, name in mentions if name.rpartition(".")[2] in _REMOVED_SHAPEFILE_SUFFIXES]
    assert removed == []
    named_files = {path for path, name in mentions if name.endswith(".gpkg")}
    assert "tests/test_review_1k1_dataset_selection.py" in named_files
    assert any(path.startswith("test_data/norfolk_coastal_flooding/") for path in named_files)
