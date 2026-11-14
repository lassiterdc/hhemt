"""Reviewer tests for chunk 1k.1: the Norfolk case dataset chosen from a reprex bundle.

The dataset itself lives outside the repository. What a laptop run can hold is the
dataset's FILE LIST, measured by the reviewer on the staged tree (23 files under
``data/contents/external/``), against the two production surfaces that name dataset
files: the Norfolk config templates and the shapefile sidecar rule of the bundle emitter.
Nothing here compiles or runs a solver and nothing reads the dataset's bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hhemt.bundle._emit import SHAPEFILE_REQUIRED_SIDECARS

_CASE_DIR = Path(__file__).resolve().parents[1] / "test_data" / "norfolk_coastal_flooding"

# The staged dataset's members by basename, as listed by `find <stage> -type f -printf "%f\n"`
# over the staged tree (23 entries). Basenames, not relative paths, so the pin survives the folder
# layout the case fetch path settles on.
_STAGED_BASENAMES = frozenset(
    {
        "b_sub_ids_and_mrms_rain_col.csv",
        "event_windows_observed.csv",
        "fullres_dem_m.tif",
        "hague_full_model_template.inp",
        "hague_hydraulics_model_template.inp",
        "hague_hydrology_model_template.inp",
        "hurricane_irene_event_index.csv",
        "landuse_clipped_to_triton_extent.tif",
        "landuse_lookup_tables.csv",
        "norfolk_wshed_epsg32147_state_plane_m.cpg",
        "norfolk_wshed_epsg32147_state_plane_m.dbf",
        "norfolk_wshed_epsg32147_state_plane_m.prj",
        "norfolk_wshed_epsg32147_state_plane_m.shp",
        "norfolk_wshed_epsg32147_state_plane_m.shx",
        "observed_event_events.csv",
        "observed_events_named_subset.csv",
        "obs_event_tseries_from_yrs_with_complete_coverage.nc",
        "TRITON_SWMM_definition_template.cfg",
        "variable_bc.cpg",
        "variable_bc.dbf",
        "variable_bc.prj",
        "variable_bc.shp",
        "variable_bc.shx",
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


def test_every_staged_shapefile_carries_the_sidecars_the_emitter_requires() -> None:
    """Quantifies over: every ``.shp`` member of the staged dataset.

    Each must sit with the reader sidecars the bundle emitter's own rule names
    (``SHAPEFILE_REQUIRED_SIDECARS``), because a ``.shp`` without them cannot be read by
    geopandas. Two earlier bundles carried the ``.shp`` alone, and that is why the newest bundle
    was chosen. The assertion reads the production rule rather than a copy of it, so it turns on
    the sidecar contract in both the current and any widened state of that rule.

    Killed by: dropping a required sidecar from the dataset's file list, or widening the
    emitter's required set to an extension the dataset does not carry.
    """
    shapefiles = sorted(n for n in _STAGED_BASENAMES if n.endswith(".shp"))
    assert shapefiles, "the staged dataset lists no shapefile, so the matcher read nothing"
    missing = {
        shp: [ext for ext in SHAPEFILE_REQUIRED_SIDECARS if Path(shp).with_suffix(ext).name not in _STAGED_BASENAMES]
        for shp in shapefiles
    }
    missing = {shp: exts for shp, exts in missing.items() if exts}
    assert not missing, f"staged shapefiles lack required sidecars: {missing}"
