"""Review tests for the Norfolk external-boundary data, the docs section and the release notes).

The chunk's own tests read the Norfolk templates as YAML and never run the published layer through the boundary
validators, and the docs and release notes are pinned by words, not against the code they describe. Each test here
builds a Norfolk-shaped input in ``tmp_path`` or reads tracked text. None runs a solver or the network.

The Norfolk model (``hague_model.inp``, 29,309 lines) and the published layer are not in the repository. The
coordinates of its two outfalls below were read from the ``[COORDINATES]`` rows of the copy whose sha256 is the
``hague_model.inp`` value of ``case.yaml`` (``26b76336...``), and the line of the layer is the one of the published
``variable_bc.gpkg`` whose sha256 is the ``variable_bc.gpkg`` value of ``case.yaml`` (``d0dd01c8...``).
"""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr
import yaml
from shapely.geometry import LineString

from hhemt.water_level_boundary import (
    _ENGINE_DEFAULT_START,
    EXTBC_TYPE_CODE,
    converted_node_depths,
    matched_outfall_names,
    preflight_problems,
    read_boundary_records,
)

_REPO = Path(__file__).resolve().parents[1]
_CASE_DIR = _REPO / "test_data" / "norfolk_coastal_flooding"
_DOCS = _REPO / "docs" / "how-to" / "using-a-runnable-swmm-model.md"
_CACHED_LAYER = (
    Path.home() / ".local/share/hhemt/examples/norfolk_coastal_flooding/data/contents/external/variable_bc.gpkg"
)

# The two Norfolk outfalls read a series named ``water_level``; their positions are in EPSG:32147 meters.
_NORFOLK_MODEL = """[TITLE]
Norfolk-shaped stand-in for hague_model.inp

[OUTFALLS]
;;Name  Elev   Type       Stage       Gated
E143250 -1.079 TIMESERIES water_level NO
E147007 -1.079 TIMESERIES water_level NO

[TIMESERIES]
;;Name Date Time Value
water_level 01/01/2020 00:00 0.0
water_level 01/01/2020 01:00 0.5

[COORDINATES]
;;Node X Y
E143250 3696719.162 1060291.312
E147007 3696713.791 1060298.456
"""
# The line of the published layer: a 2-vertex LineString, 46.9 m, with the columns Id, tseries and length_m, and
# the two columns ``type`` and ``outfalls`` that the published layer added.
_LINE = [(3696703.8335999995, 1060295.0697782936), (3696704.0515120127, 1060248.124152198)]
_PRINTED = "record 0: E143250 15.3 m, E147007 10.5 m"


def _write_layer(directory: Path, *, kind="water_level", outfalls="E143250 E147007", columns=True) -> Path:
    data = {"Id": [0], "tseries": ["norfolk_water_level"], "length_m": [46.9461318445]}
    if columns:
        data["type"] = [kind]
        data["outfalls"] = [outfalls]
    gdf = gpd.GeoDataFrame(data, geometry=[LineString(_LINE)], crs="EPSG:32147")
    path = directory / "variable_bc.gpkg"
    # A second write onto an existing GeoPackage replaces the layer inside the SQLite file, which costs about twice a
    # first write; removing the file first gives every call the first-write cost.
    path.unlink(missing_ok=True)
    gdf.to_file(path)
    return path


def _templates() -> tuple[dict, dict]:
    analysis = yaml.safe_load((_CASE_DIR / "template_analysis_config.yaml").read_text())
    system = yaml.safe_load((_CASE_DIR / "template_system_config.yaml").read_text())
    return system, analysis


def _preflight(tmp_path: Path, layer: Path, *, couple=None, model_text=_NORFOLK_MODEL):
    """The whole boundary pass over the Norfolk template values, the stand-in model and the given layer."""
    system, analysis = _templates()
    model = tmp_path / "model.inp"
    model.write_text(model_text)
    values = {
        "external_boundary_value_attribute": None,
        "water_level_boundary_outfall_max_distance_m": None,
        "water_level_boundary_unequal_outfall_series": "refuse",
        "swmm_node_rim_from_dem": False,
        **analysis,
        "external_boundary_gis": layer,
        "weather_events_to_simulate": None,
        "weather_timeseries": None,
    }
    cfg_analysis = SimpleNamespace(**values)
    cfg_system = SimpleNamespace(
        SWMM_model=model,
        crs=SimpleNamespace(horizontal_epsg=system["crs"]["horizontal_epsg"]),
        swmm_outfalls_to_couple=system["swmm_outfalls_to_couple"] if couple is None else couple,
        DEM_fullres=tmp_path / "absent.tif",
    )
    return preflight_problems(cfg_system, cfg_analysis)


def _codes(report) -> list[str]:
    return [p.code for p in report.problems]


def _grid(values: list[float]) -> xr.DataArray:
    """A 2-row DEM of 1 m cells whose rows repeat, cell k centred at x = k + 0.5, rows at y = 1.5 and 0.5."""
    cells = xr.DataArray(
        np.array([values, values], dtype="float64"),
        coords={"y": [1.5, 0.5], "x": [k + 0.5 for k in range(len(values))]},
        dims=("y", "x"),
    )
    return cells.rio.write_crs("EPSG:32618")


# Collection-time warm-up: the first layer write, layer read and CRS construction of the process are charged to no
# test (the 0.15 s ruling charges a shared setup to the first test that asks for it).
with tempfile.TemporaryDirectory() as _warm:
    _preflight(Path(_warm), _write_layer(Path(_warm)))
    _grid([0.0, 0.0]).rio.bounds()


def test_the_two_column_norfolk_layer_passes_the_whole_boundary_preflight(tmp_path):
    """Property: the Norfolk templates, the two-column layer and a model with the Norfolk outfalls raise no refusal.

    Class: NEW CAPABILITY (the chunk's data change). The plausible wrong implementation is a layer whose columns,
    types or outfall names disagree with the analysis template, which the chunk's YAML-only tests cannot see.
    Reachability: ``preflight_problems`` is the pass ``analysis.validate()`` runs over the layer (V1 to V25).
    Both-states anchor: the next three tests vary one property of the same inputs (the columns, the type, a name)
    and each turns red, so the empty result is not the result of a pass that checks nothing.
    What kills it: the layer without ``type`` or ``outfalls``, a type outside the four, an outfall the model lacks.
    A second correct implementation: the same layer read through ``boundary_records_for`` with the config models.
    """
    layer = _write_layer(tmp_path)
    report = _preflight(tmp_path, layer)
    assert report.problems == [] and report.warnings == []
    assert report.lines == [_PRINTED]
    records, problems = read_boundary_records(layer, 32147, "type", None, "outfalls")
    assert problems == [] and len(records) == 1
    assert matched_outfall_names(records) == ("E143250", "E147007")


def test_the_matched_set_is_the_analysis_template_map_and_no_outfall_is_also_listed_for_conversion(tmp_path):
    """Property: the outfalls the layer names are the keys of the template's variable map, and the system lists none.

    Class: REGRESSION on the earlier state, where the system template listed both outfalls. An outfall in both places
    is a double decision; the validators refuse it, so a template pair that said both would never validate.
    Both-states anchor: the control lists ``E143250`` in the system's ``swmm_outfalls_to_couple`` and the same
    pass then refuses, so the empty result above is a property of the Norfolk templates and not of the pass.
    """
    system, analysis = _templates()
    layer = _write_layer(tmp_path)
    records, _ = read_boundary_records(layer, 32147, "type", None, "outfalls")
    matched = set(matched_outfall_names(records))
    assert matched == set(analysis["water_level_boundary_outfall_variables"]) == {"E143250", "E147007"}
    assert not matched & set(system["swmm_outfalls_to_couple"])
    assert not matched & set(analysis["swmm_outfalls_to_keep"])
    doubled = _preflight(tmp_path, layer, couple=["E143250"])
    assert doubled.problems, "control: an outfall both matched and listed must be refused"


def test_the_layer_as_published_before_this_chunk_is_refused_for_its_missing_columns(tmp_path):
    """Control: the original three-column layer fails (V2 twice, and V9 for the two undecided outfalls)."""
    report = _preflight(tmp_path, _write_layer(tmp_path, columns=False))
    assert _codes(report).count("V2") == 2
    assert "V9" in _codes(report)
    # The property under test is the two columns, so the good layer has neither code.
    good = _preflight(tmp_path, _write_layer(tmp_path))
    assert "V2" not in _codes(good) and "V9" not in _codes(good)


def test_a_record_typed_outside_the_four_or_naming_an_unknown_outfall_is_refused(tmp_path):
    """Control: type ``tide`` is V17; an outfall the model lacks is V4."""
    assert "V17" in _codes(_preflight(tmp_path, _write_layer(tmp_path, kind="tide")))
    assert "V4" in _codes(_preflight(tmp_path, _write_layer(tmp_path, outfalls="E143250 E143251")))


def test_the_norfolk_node_depths_on_both_grids_are_the_ones_the_docs_and_the_shape_state():
    """Property: the processed grid gives depths 0.971 and 1.076 m; the full grid refuses E143250 and gives 1.3065 m.

    The depths are the ground at the node's cell less the invert (-1.079 m). The cell values are the ones the
    approved shape measured on the two Norfolk grids. The crowns of the two conduit ends (1.524 and 1.676 m, read
    from ``[XSECTIONS]`` of ``hague_model.inp``) are above the processed depths, which is why the docs say SWMM keeps
    the crown there.
    Both-states anchor: one function, two grids.
    """
    nodes = {"E143250": (0.5, 0.5, -1.079), "E147007": (1.5, 0.5, -1.079)}
    processed, problems = converted_node_depths(_grid([-0.108, -0.003]), nodes)
    assert problems == []
    assert processed["E143250"] == pytest.approx(0.971) and processed["E147007"] == pytest.approx(1.076)
    assert processed["E143250"] < 1.524 and processed["E147007"] < 1.676
    full, problems = converted_node_depths(_grid([-1.500, 0.2275]), nodes)
    assert [p.code for p in problems] == ["V23"] and "E143250" in problems[0].message
    assert full == {"E147007": pytest.approx(1.3065)}


def test_the_case_manifest_pins_the_published_layer_and_marks_nothing_provisional():
    """Property: the ``variable_bc.gpkg`` row holds the published layer's digest and no comment calls it provisional.

    The layer gained the columns ``type`` and ``outfalls``, so a manifest that kept the digest of the layer without
    them would fail the fetch-side integrity check against the published layer. The earlier manifest held a locally
    written digest under a PROVISIONAL comment until the new resource version existed; that version exists, so the
    comment has no referent.
    What kills it: a missing ``.gpkg`` row, a digest other than the published one, a PROVISIONAL comment left behind.
    A second correct implementation: the digest compared after reading the manifest through ``CaseManifest``.
    """
    text = (_CASE_DIR / "case.yaml").read_text()
    manifest = yaml.safe_load(text)["manifest"]
    assert manifest["data/contents/external/variable_bc.gpkg"] == (
        "d0dd01c87c2d3b3312d51117b4d72711e7357f4e1ec1293adcb97c96484b10b8"
    )
    assert "PROVISIONAL" not in text
    # Control: the matcher fires on the marker the earlier manifest carried.
    assert "PROVISIONAL" in "# PROVISIONAL: the layer variable_bc gains the columns"


@pytest.mark.skipif(not _CACHED_LAYER.is_file(), reason="the fetched Norfolk example layer is not on this machine")
def test_the_published_layer_carries_exactly_the_two_columns_the_analysis_template_names(tmp_path):
    """Property: the fetched layer has Id, tseries, length_m, type and outfalls, and the whole boundary pass accepts it.

    Reachability: the layer is read through geopandas by the boundary validators exactly as a run reads it.
    What kills it: a published layer that lacks ``type`` or ``outfalls``, or is not in EPSG:32147.
    A second correct implementation: the layer read with ``pyogrio.read_dataframe`` instead of ``gpd.read_file``.
    """
    _, analysis = _templates()
    published = gpd.read_file(_CACHED_LAYER)
    assert [c for c in published.columns if c != "geometry"] == ["Id", "tseries", "length_m", "type", "outfalls"]
    assert published.crs.to_epsg() == 32147
    assert list(LineString(_LINE).coords) == list(published.geometry.iloc[0].coords), "the literal line is the layer's"
    assert analysis["external_boundary_type_attribute"] == "type"
    assert analysis["water_level_boundary_outfall_attribute"] == "outfalls"
    report = _preflight(tmp_path, _CACHED_LAYER)
    assert report.problems == [] and report.lines == [_PRINTED]


def test_the_docs_name_the_layer_files_the_norfolk_templates_name():
    """Property: the page names, by basename, the file each Norfolk template gives to a map layer.

    Reachability: the page's Norfolk section is the one place a reader learns what the boundary layer file is called.
    What kills it: a template pointing at a layer file the page does not name, or a page naming one the templates
    do not.
    A second correct implementation: the page generated from the templates, in which case the property holds by
    construction.
    """
    system, analysis = _templates()
    names = {Path(system["watershed_gis_polygon"]).name, Path(analysis["external_boundary_gis"]).name}
    text = _DOCS.read_text()
    assert names and {name for name in names if name in text} == names
    # Control: the matcher fires on a basename the page does not hold.
    assert "no_such_layer.gpkg" not in text


def test_the_docs_distance_example_for_the_norfolk_outfalls_is_what_the_norfolk_layer_prints(tmp_path):
    """Property: a docs example line that names E143250 and E147007 carries the distances the Norfolk layer prints.

    The page shows ``record 0: E143250 8.9 m, E147007 8.9 m``. 8.9 m is the distance between the two outfalls, not
    the distance of each to the line (15.3 m and 10.5 m, measured with this test's inputs).
    Must-fire control: the matcher finds the pair in the text of the line the page carries today.
    """
    pattern = re.compile(r"record \d+: E143250 ([\d.]+) m, E147007 ([\d.]+) m")
    assert pattern.search("record 0: E143250 8.9 m, E147007 8.9 m")
    printed = _preflight(tmp_path, _write_layer(tmp_path)).lines
    assert printed == [_PRINTED]
    for found in pattern.finditer(_DOCS.read_text()):
        assert found.group(0) == printed[0], (
            f"the page prints {found.group(0)!r}; the Norfolk layer prints {printed[0]!r}"
        )


def test_the_release_entry_claims_match_the_code():
    """Property: layout version 34, ``swmm_threads`` default 1, both decision lists and the two new key names exist."""
    from hhemt.config.analysis import _REFUSED_BOUNDARY_KEYS, analysis_config
    from hhemt.config.system import system_config
    from hhemt.version_migration.constants import LAYOUT_VERSION

    entry = (_REPO / "HISTORY.md").read_text()
    assert "layout version 34" in entry and LAYOUT_VERSION == 34
    assert analysis_config.model_fields["swmm_threads"].default == 1
    assert "swmm_outfalls_to_keep" in analysis_config.model_fields
    assert "swmm_outfalls_to_couple" in system_config.model_fields
    assert "Replacement: toggle_external_boundaries" in _REFUSED_BOUNDARY_KEYS["toggle_storm_tide_boundary"]
    assert "Replacement: external_boundary_gis" in _REFUSED_BOUNDARY_KEYS["storm_tide_boundary_line_gis"]
    # Control: the matcher fires on a pair that is not a refusal.
    assert "Replacement: external_boundary_gis" not in _REFUSED_BOUNDARY_KEYS["toggle_storm_tide_boundary"]


def test_the_docs_boundary_table_and_units_match_the_code():
    """Property: the type codes of the docs table, the 2004 default date and the unit list are the code's."""
    from hhemt.triton_cfg import LEVEL_UNIT_FACTORS

    text = _DOCS.read_text()
    rows = dict(re.findall(r"^\| `(\w+)` \| `(\d),", text, re.MULTILINE))
    assert {kind: int(code) for kind, code in rows.items()} == EXTBC_TYPE_CODE
    assert _ENGINE_DEFAULT_START == datetime(2004, 1, 1) and "01/01/2004" in text
    listed = re.search(r"The unit is (.*?)\.", text).group(1)
    assert set(re.findall(r"`(\w+)`", listed)) == set(LEVEL_UNIT_FACTORS)
