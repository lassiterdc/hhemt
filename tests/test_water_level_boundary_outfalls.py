"""The outfall decisions, the series identity rule and the series files of the external-boundary layer.

Every test is text and a few rows, no solver and no synthetic analysis, so each runs well under 0.15 s.
Class of each test: NEW CAPABILITY, the plausible wrong implementation named in its docstring.
"""

from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr

import hhemt.scenario  # noqa: F401 -- imported at collection, so no test is charged for the module's import time
from hhemt.swmm_runnable import _parse, derive_role_model
from hhemt.water_level_boundary import (
    COMPLETE_MARKER,
    BoundaryRecord,
    extract_model_series,
    model_outfalls,
    model_series,
    outfall_decision_problems,
    resolve_series_rows,
    series_identity_problems,
    series_pairs,
    variable_problems,
)

MODEL = """[OPTIONS]
FLOW_ROUTING DYNWAVE

[JUNCTIONS]
J1 0 1 0 0 0

[OUTFALLS]
OF_A 0 TIMESERIES series_a
OF_B 0 TIMESERIES series_b
OF_F 0 FREE NO

[CONDUITS]
C1 J1 OF_A 10 0.01 0 0 0 0

[TIMESERIES]
series_a 01/01/2020 0 0.5
series_a 1 0.7
series_b 01/01/2020 00:00 0.5
series_b 01/01/2020 01:00 0.7
tmp_ts 01/01/2020 00:00 10.0
"""


def _dem_array(nx: int = 10, ny: int = 10, cell: float = 10.0):
    """A flat in-memory DEM, ground 0.0, cell centres at 5, 15, ... in both axes."""
    xs = np.arange(nx) * cell + cell / 2
    ys = (np.arange(ny) * cell + cell / 2)[::-1]
    da = xr.DataArray(np.zeros((ny, nx), dtype="float32"), coords={"y": ys, "x": xs}, dims=("y", "x"))
    return da.rio.write_crs("EPSG:32618")


# Built once at collection: the first CRS construction of a process is slow and no test should be charged for it.
DEM = _dem_array()


def _dem(path):
    """The same DEM as a GeoTIFF on disk."""
    DEM.rio.to_raster(path)
    return path


def _outfalls(text: str = MODEL):
    return model_outfalls(_parse(text))


def test_every_undecided_outfall_is_named_once_with_the_line_to_write():
    """NEW CAPABILITY. Wrong implementation: report only the first undecided outfall, or default them silently.

    Three undecided outfalls and one unknown keep name raise in one pass; the variable and the plain outfalls
    are separate problem kinds, each naming every offender and carrying the YAML to paste.
    """
    problems = outfall_decision_problems(_outfalls(), matched=[], to_couple=[], to_keep=["NOPE"])
    kinds = {p.field: p for p in problems}
    variable = kinds["analysis.swmm_outfalls_to_keep"]
    assert "OF_A (TIMESERIES series_a)" in variable.message and "OF_B (TIMESERIES series_b)" in variable.message
    assert variable.config_line == "swmm_outfalls_to_keep: [OF_A, OF_B]"
    assert "OF_F (FREE)" in kinds["system.swmm_outfalls_to_couple"].message
    assert any("NOPE" in p.message for p in problems)


def test_an_outfall_stated_in_each_of_the_three_places_is_accepted():
    """NEW CAPABILITY. Wrong implementation: refuse a model whose outfalls are all decided, one per place."""
    assert outfall_decision_problems(_outfalls(), matched=["OF_A"], to_couple=["OF_F"], to_keep=["OF_B"]) == []


def test_a_variable_outfall_in_the_couple_list_and_a_matched_name_listed_twice_are_refused():
    """NEW CAPABILITY. Wrong implementation: let swmm_outfalls_to_couple convert a TIMESERIES outfall (V13) or let a
    record's outfall also be listed there (V6)."""
    problems = outfall_decision_problems(_outfalls(), matched=["OF_A"], to_couple=["OF_A", "OF_B", "OF_F"], to_keep=[])
    codes = sorted(p.code for p in problems)
    assert codes == ["V13", "V6"]
    assert "OF_B" in next(p for p in problems if p.code == "V13").message


def _record(*outfalls: str) -> BoundaryRecord:
    return BoundaryRecord(0, "water_level", None, tuple(outfalls), ((0.0, 0.0), (1.0, 0.0)))


def test_two_names_with_equal_resolved_rows_are_identical_series():
    """NEW CAPABILITY. Wrong implementation: compare series by name or by raw text.

    series_a (a date row, then a decimal-hour row) and series_b (repeated dates, hh:mm times) are different text and
    the same series, so V12 passes; a differing value is refused naming the first differing row.
    """
    sections = _parse(MODEL)
    assert (
        series_identity_problems(
            [_record("OF_A", "OF_B")], model_series(sections, None), _outfalls(), "weather_time_series", "refuse"
        )
        == []
    )
    changed = MODEL.replace("series_b 01/01/2020 01:00 0.7", "series_b 01/01/2020 01:00 0.9")
    sections = _parse(changed)
    problems = series_identity_problems(
        [_record("OF_A", "OF_B")],
        model_series(sections, None),
        model_outfalls(sections),
        "weather_time_series",
        "refuse",
    )
    assert [p.code for p in problems] == ["V12"]
    assert "row 1" in problems[0].message and "does not apply" in problems[0].fix_hint


def test_the_averaging_option_is_offered_only_under_the_model_source():
    """NEW CAPABILITY. Wrong implementation: offer averaging under the weather source, where no run reads the series."""
    changed = MODEL.replace("series_b 01/01/2020 01:00 0.7", "series_b 01/01/2020 01:00 0.9")
    sections = _parse(changed)
    args = ([_record("OF_A", "OF_B")], model_series(sections, None), model_outfalls(sections))
    weather = series_identity_problems(*args, "weather_time_series", "refuse")[0]
    model = series_identity_problems(*args, "swmm_model", "refuse")[0]
    assert "average" not in weather.fix_hint.replace("averaging option does not apply", "")
    assert "average" in model.fix_hint


@pytest.mark.parametrize(
    ("rows_a", "rows_b"),
    [
        ([["01/01/2020", "1.5", "2"]], [["01/01/2020", "01:30", "2"]]),
        ([["01/01/2020", "0", "1", "1", "2"]], [["01/01/2020", "0", "1"], ["1", "2"]]),
    ],
)
def test_resolution_reads_decimal_hours_and_clock_times_and_carried_dates_alike(rows_a, rows_b):
    """NEW CAPABILITY. Wrong implementation: compare tokens, so 1.5 and 01:30 or a carried date differ."""
    assert resolve_series_rows(rows_a) == resolve_series_rows(rows_b)


def test_extraction_writes_each_inline_series_once_and_resolves_to_the_base_pairs(tmp_path):
    """NEW CAPABILITY. Wrong implementation: write the extracted file with the base row's raw tokens only, losing a
    carried date, or rebuild the directory at every call.

    The file's first line is a ;; comment whose first three tokens are short whatever the name's length (SWMM
    reads three tokens into 50-character buffers), it opens with the engine's date, and a second call writes
    nothing new.
    """
    long_name = "s" * 60
    text = MODEL.replace("series_a", long_name)
    model = tmp_path / "model.inp"
    model.write_text(text)
    pointers = extract_model_series(model, tmp_path / "system")
    extracted = tmp_path / "system" / "swmm_model_series"
    directory = next(extracted.iterdir())
    assert (directory / COMPLETE_MARKER).is_file()
    first = (directory / "000.dat").read_text().splitlines()[0]
    assert first.split()[:3] == [";;", "hhemt", "extracted"] and long_name in first
    from hhemt.water_level_boundary import ModelSeries

    base = model_series(_parse(text), tmp_path)[long_name.upper()]
    file_series = ModelSeries(long_name, (), directory / "000.dat")
    assert series_pairs(file_series) == series_pairs(base)
    mtimes = {f.name: f.stat().st_mtime_ns for f in directory.iterdir()}
    again = extract_model_series(model, tmp_path / "system")
    assert again == pointers
    assert {f.name: f.stat().st_mtime_ns for f in directory.iterdir()} == mtimes


def test_a_derived_role_holds_a_file_pointer_per_series_and_no_series_row(tmp_path):
    """NEW CAPABILITY. Wrong implementation: point only the referenced series, or keep the rows beside the pointer."""
    model = tmp_path / "model.inp"
    model.write_text(MODEL)
    pointers = extract_model_series(model, tmp_path / "system")
    for role in ("full", "hydraulics", "hydrology"):
        out = derive_role_model(MODEL, role, ["OF_F"], series_pointers=pointers)
        rows = [r for r in _rows(out, "TIMESERIES")]
        assert {r[0] for r in rows} == {"series_a", "series_b", "tmp_ts"}
        assert all(r[1] == "FILE" and len(r) == 3 for r in rows)


def _rows(text: str, section: str) -> list[list[str]]:
    from hhemt.swmm_runnable import _node_rows

    return _node_rows(_parse(text), section)


def test_a_converted_outfall_takes_its_dem_depth_and_a_matched_one_reads_its_record_series():
    """NEW CAPABILITY. Wrong implementation: write MaxDepth 0 for a converted outfall, or leave the stage token."""
    full = derive_role_model(MODEL, "full", [], stage_series={"of_a": "hhemt_water_level_0"})
    assert ["OF_A", "0", "TIMESERIES", "hhemt_water_level_0"] in _rows(full, "OUTFALLS")
    hydraulics = derive_role_model(MODEL, "hydraulics", ["OF_A"], node_depths={"of_a": 0.971})
    assert ["OF_A", "0", "0.971", "0", "0", "0"] in _rows(hydraulics, "JUNCTIONS")
    assert "OF_A" not in {r[0] for r in _rows(hydraulics, "OUTFALLS")}


def test_the_weather_source_map_must_name_every_matched_outfall_and_a_known_unit():
    """NEW CAPABILITY. Wrong implementation: accept a map that misses an outfall, or an unknown unit."""
    records = [_record("OF_A")]
    problems = variable_problems({"OTHER": "v"}, ["OF_A"], records, ["v"], {"v": "furlong"}, {"v": 0.0})
    assert sorted(p.code for p in problems) == ["V14", "V22"]
    ok = variable_problems({"OF_A": "v"}, ["OF_A"], records, ["v"], {"v": "ft"}, {"v": 0.0})
    assert ok == []


# ----------------------------------------------------------------------------------------------------------------
# The layer, its records, and the files the writers produce from them
# ----------------------------------------------------------------------------------------------------------------

LAYER_MODEL = """[OPTIONS]
FLOW_ROUTING DYNWAVE

[JUNCTIONS]
J1 0 1 0 0 0

[OUTFALLS]
OF_A -1.0 TIMESERIES lvl
OF_B -1.0 TIMESERIES lvl
OF_F 0 FREE NO

[CONDUITS]
C1 J1 OF_A 10 0.01 0 0 0 0

[TIMESERIES]
lvl 01/01/2020 00:00 0.5
lvl 01/01/2020 01:00 0.7

[COORDINATES]
J1 25 25
OF_A 15 8
OF_B 25 8
OF_F 85 85
"""


def _layer(path, rows, epsg=32618, columns=("type", "outfalls", "value")):
    """A tiny GeoJSON of straight line records: each row is ``(vertices, *column values)``.

    With ``epsg=None`` the layer is a shapefile with no ``.prj``, which is how a layer holds no CRS (GeoJSON
    assumes one).
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    data = {c: [r[i + 1] for r in rows] for i, c in enumerate(columns)}
    frame = gpd.GeoDataFrame(data, geometry=[LineString(r[0]) for r in rows], crs=f"EPSG:{epsg}" if epsg else None)
    if epsg is None:
        path = path.with_suffix(".shp")
        frame.to_file(path, driver="ESRI Shapefile")
    else:
        frame.to_file(path, driver="GeoJSON")
    return path


BOTTOM = [(5.0, 5.0), (35.0, 5.0)]


def test_a_layer_without_a_crs_without_a_column_or_with_a_bad_geometry_is_refused_by_kind(tmp_path):
    """NEW CAPABILITY. Wrong implementation: read a CRS-less layer as the system CRS, or raise on the first problem.

    V1 (no CRS, no record, a point where a line belongs) and V2 (a named column the layer lacks) stop the read.
    """
    from hhemt.water_level_boundary import read_boundary_records

    rows = [(BOTTOM, "water_level", "OF_A", None)]
    no_crs = read_boundary_records(_layer(tmp_path / "a.geojson", rows, epsg=None), 32618, "type", None, "outfalls")
    assert [p.code for p in no_crs[1]] == ["V1"] and "no CRS" in no_crs[1][0].message
    missing = read_boundary_records(_layer(tmp_path / "b.geojson", rows), 32618, "kind", "slope", "outfalls")
    assert sorted(p.field for p in missing[1]) == [
        "analysis.external_boundary_type_attribute",
        "analysis.external_boundary_value_attribute",
    ]
    ok_records, ok_problems = read_boundary_records(
        _layer(tmp_path / "c.geojson", rows), 32618, "type", None, "outfalls"
    )
    assert ok_problems == [] and [r.kind for r in ok_records] == ["water_level"]


def test_a_layer_in_another_crs_is_reprojected_and_its_endpoints_are_the_system_crs_ones(tmp_path):
    """NEW CAPABILITY. Wrong implementation: use the layer's own coordinates as if they were the system CRS."""
    from pyproj import Transformer

    from hhemt.water_level_boundary import read_boundary_records

    to_geographic = Transformer.from_crs(32618, 4326, always_xy=True)
    vertices = [to_geographic.transform(x, y) for x, y in ((500000.0, 4000000.0), (500030.0, 4000000.0))]
    layer = _layer(tmp_path / "geo.geojson", [(vertices, "water_level", "OF_A", None)], epsg=4326)
    (record,), problems = read_boundary_records(layer, 32618, "type", None, "outfalls")
    assert problems == []
    assert record.vertices[0] == pytest.approx((500000.0, 4000000.0), abs=0.01)
    assert record.vertices[1] == pytest.approx((500030.0, 4000000.0), abs=0.01)
    # Control that varies the property (the layer's CRS equals the system's), not the call: no coordinate moves.
    same = _layer(
        tmp_path / "same.geojson", [([(500000.0, 4000000.0), (500030.0, 4000000.0)], "water_level", "OF_A", None)]
    )
    assert read_boundary_records(same, 32618, "type", None, "outfalls")[0][0].vertices[0] == (500000.0, 4000000.0)


def test_each_record_type_agrees_with_its_value_and_outfall_columns():
    """NEW CAPABILITY. Wrong implementation: let a slope record carry no number, a level record carry one, or a free
    outflow record name an outfall (V3, V17, V18)."""
    from hhemt.water_level_boundary import record_problems

    v = ((0.0, 0.0), (1.0, 0.0))
    good = [
        BoundaryRecord(0, "water_level", None, ("OF_A",), v),
        BoundaryRecord(1, "normal_slope", 0.01, (), v),
        BoundaryRecord(2, "froude_number", 1.5, (), v),
        BoundaryRecord(3, "free_outflow", None, (), v),
    ]
    assert record_problems(good, "outfalls", "value", "layer.geojson") == []
    bad = [
        BoundaryRecord(0, "water_level", None, (), v),
        BoundaryRecord(1, "normal_slope", None, (), v),
        BoundaryRecord(2, "free_outflow", 3.0, ("OF_A",), v),
        BoundaryRecord(3, "tide_gauge", None, (), v),
    ]
    problems = record_problems(bad, "outfalls", "value", "layer.geojson")
    assert sorted({p.code for p in problems}) == ["V17", "V18", "V3"]
    edits = [p.layer_edit for p in problems if p.layer_edit]
    assert any("column value of layer.geojson, records [1]" in e for e in edits)
    assert all(e.startswith(("column ", "the ")) and "{" not in e for e in edits)


def test_each_matched_outfall_must_exist_be_a_level_outfall_be_placed_and_be_near(tmp_path):
    """NEW CAPABILITY. Wrong implementation: skip a named junction, an outfall with no coordinates, or the distance
    maximum (V4, V5, V19, V20); and print no distance when no maximum is set."""
    from hhemt.water_level_boundary import distance_lines, matched_outfall_problems, model_coordinates

    sections = _parse(LAYER_MODEL + "\n")
    outfalls, coords = model_outfalls(sections), model_coordinates(sections)
    near = BoundaryRecord(0, "water_level", None, ("OF_A", "OF_B"), ((5.0, 5.0), (35.0, 5.0)))
    problems, distances = matched_outfall_problems([near], outfalls, coords, None)
    assert problems == []
    assert distance_lines(distances) == ["record 0: OF_A 3.0 m, OF_B 3.0 m"]
    far, _ = matched_outfall_problems([near], outfalls, coords, 2.0)
    assert [p.code for p in far] == ["V19"] and "OF_A" in far[0].message
    wrong = BoundaryRecord(0, "water_level", None, ("J1", "OF_F", "NOPE", "OF_NOXY"), near.vertices)
    codes = sorted(p.code for p in matched_outfall_problems([wrong], outfalls, coords, None)[0])
    assert codes == ["V4", "V5"]
    placeless = {**coords}
    del placeless["OF_A"]
    assert [p.code for p in matched_outfall_problems([near], outfalls, placeless, None)[0]] == ["V20"]


def test_a_cell_two_segments_share_and_a_one_cell_segment_are_refused(tmp_path):
    """NEW CAPABILITY. Wrong implementation: let two segments write one corner cell, or accept a one-cell one (V10)."""
    from hhemt.water_level_boundary import snap_segments

    dem = DEM
    a = BoundaryRecord(0, "water_level", None, ("OF_A",), ((5.0, 5.0), (35.0, 5.0)))
    apart = BoundaryRecord(1, "free_outflow", None, (), ((55.0, 5.0), (85.0, 5.0)))
    snapped, problems = snap_segments([a, apart], dem)
    assert problems == [] and [s[1][0] for s in snapped] == ["bottom", "bottom"]
    sharing = BoundaryRecord(1, "free_outflow", None, (), ((35.0, 5.0), (65.0, 5.0)))
    assert [p.code for p in snap_segments([a, sharing], dem)[1]] == ["V10"]
    assert "stop one cell short of a corner another segment uses" in snap_segments([a, sharing], dem)[1][0].message
    one = BoundaryRecord(0, "free_outflow", None, (), ((5.0, 5.0), (6.0, 5.0)))
    assert "single DEM cell" in snap_segments([one], dem)[1][0].message


def test_a_depth_from_the_dem_is_the_ground_at_the_nodes_cell_minus_its_invert_and_a_low_ground_is_refused(tmp_path):
    """NEW CAPABILITY. Wrong implementation: read the cell by rounding the coordinate, or accept a node whose ground
    sits at or below its invert (V23)."""
    from hhemt.water_level_boundary import converted_node_depths

    dem = DEM  # ground 0.0 everywhere
    depths, problems = converted_node_depths(dem, {"OF_A": (15.0, 8.0, -1.0)})
    assert problems == [] and depths == {"OF_A": 1.0}
    depths, problems = converted_node_depths(
        dem, {"OF_A": (15.0, 8.0, -1.0), "LOW": (25.0, 8.0, 0.5), "OUT": (500.0, 8.0, 0.0)}
    )
    assert depths == {"OF_A": 1.0} and [p.code for p in problems] == ["V23"]
    assert "LOW (invert 0.500 m, cell elevation 0.000 m)" in problems[0].message and "OUT" in problems[0].message


def test_a_model_object_under_the_reserved_prefix_and_a_missing_series_file_are_refused(tmp_path):
    """NEW CAPABILITY. Wrong implementation: let a base model name an object under hhemt_, or defer a missing FILE
    series to the engine (V11, V21)."""
    from types import SimpleNamespace

    from hhemt.swmm_utils import swmm_model_problems

    model = tmp_path / "m.inp"
    model.write_text(
        LAYER_MODEL.replace("J1 0 1 0 0 0", "hhemt_J1 0 1 0 0 0").replace(
            "lvl 01/01/2020 00:00 0.5", 'gone FILE "missing.dat"\nlvl 01/01/2020 00:00 0.5'
        )
    )
    cfg = SimpleNamespace(
        SWMM_model=model,
        swmm_outfalls_to_couple=[],
        subcatchment_raingage_mapping=None,
        subcatchment_raingage_mapping_gage_id_colname=None,
    )
    messages = [p[1] for p in swmm_model_problems(cfg)]
    assert any("hhemt_J1" in m and "reserved prefix" in m for m in messages)
    assert any("missing.dat" in m for m in messages)
    model.write_text(LAYER_MODEL)
    assert swmm_model_problems(cfg) == []


def test_three_undecided_outfalls_and_one_unknown_name_raise_once_with_two_issues_that_name_all_four():
    """NEW CAPABILITY. Wrong implementation: raise on the first offender, or one issue per outfall."""
    from hhemt.exceptions import ConfigurationError
    from hhemt.validation import ValidationResult, _add_boundary_problems

    model = LAYER_MODEL.replace("OF_F 0 FREE NO", "OF_F 0 FREE NO\nOF_G 0 NORMAL NO")
    problems = outfall_decision_problems(_outfalls(model), matched=[], to_couple=[], to_keep=["NOPE"])
    result = ValidationResult(context="preflight")
    _add_boundary_problems(result, [p for p in problems if p.code == "V9"])
    undecided = [e for e in result.errors if "no decision" in e.message or "neither converted nor kept" in e.message]
    assert len(undecided) == 2
    with pytest.raises(ConfigurationError) as raised:
        result.raise_if_invalid()
    text = str(raised.value)
    assert all(name in text for name in ("OF_A", "OF_B", "OF_F", "OF_G", "NOPE"))
    assert text.count("Write:") == 2


def _scenario(tmp_path, layer_rows, *, source="weather_time_series", units=None, offsets=None, outfall_vars=None):
    """A scenario stand-in with the attributes the boundary writers read, and a recording scenario log."""
    from types import SimpleNamespace

    times = np.array(["2020-01-01T00:00", "2020-01-01T00:30", "2020-01-01T01:00"], dtype="datetime64[ns]")
    ds = xr.Dataset({"wl": ("time", np.array([1.0, 2.0, 3.0]))}, coords={"time": times})
    model = tmp_path / "model.inp"
    model.write_text(LAYER_MODEL)
    layer = _layer(tmp_path / "layer.geojson", layer_rows)
    cfg_analysis = SimpleNamespace(
        toggle_external_boundaries=True,
        external_boundary_gis=layer,
        external_boundary_type_attribute="type",
        external_boundary_value_attribute="value",
        water_level_boundary_outfall_attribute="outfalls",
        water_level_boundary_series_source=source,
        water_level_boundary_outfall_variables=outfall_vars
        if outfall_vars is not None
        else {"OF_A": "wl", "OF_B": "wl"},
        water_level_boundary_variable_units=units if units is not None else {"wl": "m"},
        water_level_boundary_variable_datum_offsets=offsets if offsets is not None else {"wl": 0.0},
        water_level_boundary_unequal_outfall_series="refuse",
        water_level_boundary_outfall_max_distance_m=None,
        swmm_threads=1,
    )
    cfg_system = SimpleNamespace(
        SWMM_model=model, crs=SimpleNamespace(horizontal_epsg=32618), swmm_outfalls_to_couple=[]
    )
    recorded: dict = {}

    class _Field:
        def __init__(self, name):
            self.name = name

        def set(self, value):
            recorded[self.name] = value

    extbc = tmp_path / "extbc"
    extbc.mkdir()
    dats = tmp_path / "dats"
    dats.mkdir()
    paths = SimpleNamespace(
        extbc_dir=extbc,
        extbc_loc=extbc / "loc.extbc",
        dir_weather_datfiles=dats,
    )
    from hhemt.paths import ScenarioPaths

    paths.extbc_tseries = lambda k: ScenarioPaths.extbc_tseries(paths, k)
    paths.water_level_dat = lambda k: ScenarioPaths.water_level_dat(paths, k)
    scenario = SimpleNamespace(
        _analysis=SimpleNamespace(cfg_analysis=cfg_analysis),
        _system=SimpleNamespace(
            cfg_system=cfg_system, sys_paths=SimpleNamespace(dem_processed=_dem(tmp_path / "dem.tif"))
        ),
        ds_event_ts=ds,
        scen_paths=paths,
        sim_id_str="sim0",
        log=SimpleNamespace(
            extbc_tseries_created=_Field("tseries"),
            extbc_loc_created=_Field("loc"),
            external_boundary_segments=_Field("segments"),
            water_level_boundary_dats=_Field("dats"),
        ),
        _resolve_event_window=lambda: (
            times[0].astype("datetime64[s]").item(),
            times[-1].astype("datetime64[s]").item(),
        ),
    )
    return scenario, recorded


FOUR = [
    (BOTTOM, "water_level", "OF_A OF_B", None),
    ([(45.0, 5.0), (65.0, 5.0)], "free_outflow", None, None),
    ([(5.0, 95.0), (35.0, 95.0)], "normal_slope", None, 0.01),
    ([(45.0, 95.0), (65.0, 95.0)], "froude_number", None, 1.5),
]


def test_a_four_record_layer_writes_four_rows_with_their_codes_and_one_series_file(tmp_path):
    """NEW CAPABILITY. Wrong implementation: write only record 0, give every row the type-1 code, or write a series
    file for a record that is not a water_level record."""
    from hhemt.scenario_inputs import ScenarioInputGenerator

    scenario, recorded = _scenario(tmp_path, FOUR, units={"wl": "ft"}, offsets={"wl": 1.0})
    ScenarioInputGenerator(scenario).create_external_boundary_condition_files()
    loc = (tmp_path / "extbc" / "loc.extbc").read_text().splitlines()
    assert loc[0].startswith("% BC Type")
    assert loc[1:] == [
        '1,5.0,5.0,35.0,5.0,"tseries_000.txt"',
        "0,45.0,5.0,65.0,5.0,0.0",
        "2,5.0,95.0,35.0,95.0,0.01",
        "3,45.0,95.0,65.0,95.0,1.5",
    ]
    assert sorted(p.name for p in (tmp_path / "extbc").glob("tseries_*.txt")) == ["tseries_000.txt"]
    body = [ln for ln in (tmp_path / "extbc" / "tseries_000.txt").read_text().splitlines() if not ln.startswith("%")]
    assert body == ["0.0,1.3048", "0.5,1.6096", "1.0,1.9144"]  # ft, then the datum offset, in meters
    assert [s["type"] for s in recorded["segments"]] == ["water_level", "free_outflow", "normal_slope", "froude_number"]
    assert recorded["segments"][0]["outfall_distance_m"] == {"OF_A": 3.0, "OF_B": 3.0}


def test_both_level_files_carry_the_same_meters_and_a_toggle_off_writes_a_header_only_location_file(tmp_path):
    """NEW CAPABILITY. Wrong implementation: convert for one file only, or write a level file with the toggle off."""
    from hhemt.scenario_inputs import ScenarioInputGenerator
    from hhemt.swmm_runoff_modeling import SWMMRunoffModeler

    scenario, recorded = _scenario(tmp_path, FOUR[:1], units={"wl": "in"}, offsets={"wl": 0.5})
    ScenarioInputGenerator(scenario).create_external_boundary_condition_files()
    scenario.cfg_analysis = scenario._analysis.cfg_analysis
    SWMMRunoffModeler(scenario).write_swmm_waterlevel_dat_files()
    triton = [
        ln.split(",")[1]
        for ln in (tmp_path / "extbc" / "tseries_000.txt").read_text().splitlines()
        if not ln.startswith("%")
    ]
    swmm = [
        ln.split("\t")[2]
        for ln in (tmp_path / "dats" / "water_level_000.dat").read_text().splitlines()
        if not ln.startswith(";")
    ]
    assert triton == swmm == ["0.5254", "0.5508", "0.5762"]
    assert list(recorded["dats"]) == [0]
    off, off_recorded = _scenario(tmp_path / "off", FOUR[:1]) if (tmp_path / "off").mkdir() is None else (None, None)
    off._analysis.cfg_analysis.toggle_external_boundaries = False
    ScenarioInputGenerator(off).create_external_boundary_condition_files()
    assert (tmp_path / "off" / "extbc" / "loc.extbc").read_text().splitlines() == ["% BC Type, X1, Y1, X2, Y2, BC"]
    assert not list((tmp_path / "off" / "extbc").glob("tseries_*"))
    assert "tseries" not in off_recorded


def test_the_model_source_reads_the_outfalls_series_clips_it_to_the_window_and_averages_only_on_request(tmp_path):
    """NEW CAPABILITY. Wrong implementation: write rows before the window start, accept a series that does not cover
    the window, or average without being asked."""
    from hhemt.exceptions import ProcessingError
    from hhemt.water_level_boundary import boundary_level_series

    scenario, _ = _scenario(tmp_path, FOUR[:1], source="swmm_model", outfall_vars=None)
    record = BoundaryRecord(0, "water_level", None, ("OF_A",), BOTTOM)
    level = boundary_level_series(scenario, record)
    assert level.meters == (0.5, 0.7) and not level.source.endswith(", averaged")
    assert level.times[0].isoformat() == "2020-01-01T00:00:00"
    narrow = tmp_path / "narrow"
    narrow.mkdir()
    inner, _ = _scenario(narrow, FOUR[:1], source="swmm_model")
    inner._resolve_event_window = lambda: (level.times[0].replace(minute=15), level.times[0].replace(minute=45))
    assert boundary_level_series(inner, record).meters == pytest.approx((0.55, 0.65))  # the ends are interpolated
    short = tmp_path / "short"
    short.mkdir()
    late, _ = _scenario(short, FOUR[:1], source="swmm_model")
    late._resolve_event_window = lambda: (level.times[0], level.times[0].replace(hour=3))
    with pytest.raises(ProcessingError, match="does not cover"):
        boundary_level_series(late, record)
    unequal = tmp_path / "unequal"
    unequal.mkdir()
    diff, _ = _scenario(unequal, FOUR[:1], source="swmm_model")
    diff._analysis.cfg_analysis.water_level_boundary_unequal_outfall_series = "average"
    model = unequal / "model.inp"
    model.write_text(
        LAYER_MODEL.replace("OF_B -1.0 TIMESERIES lvl", "OF_B -1.0 TIMESERIES other").replace(
            "lvl 01/01/2020 01:00 0.7",
            "lvl 01/01/2020 01:00 0.7\nother 01/01/2020 00:00 0.9\nother 01/01/2020 01:00 1.1",
        )
    )
    averaged = boundary_level_series(diff, BoundaryRecord(0, "water_level", None, ("OF_A", "OF_B"), BOTTOM))
    assert averaged.meters == pytest.approx((0.7, 0.9)) and averaged.source.endswith(", averaged")


def test_the_converted_set_adds_the_matched_outfalls_and_a_derived_role_holds_a_pointer_for_each_series(tmp_path):
    """NEW CAPABILITY. Wrong implementation: convert only swmm_outfalls_to_couple, or leave inline rows in a role."""
    from hhemt.swmm_utils import converted_outfalls, swmm_role_text

    scenario, _ = _scenario(tmp_path, FOUR[:1])
    cfg_system, cfg_analysis = scenario._system.cfg_system, scenario._analysis.cfg_analysis
    cfg_system.system_directory = tmp_path / "system"
    cfg_system.swmm_outfalls_to_couple = ["OF_F"]
    assert converted_outfalls(cfg_system, cfg_analysis) == ("OF_F", "OF_A", "OF_B")
    text = swmm_role_text(cfg_system, "hydraulics", cfg_analysis)
    assert {r[0] for r in _rows(text, "JUNCTIONS")} >= {"OF_A", "OF_B", "OF_F"}
    assert all(r[1] == "FILE" for r in _rows(text, "TIMESERIES"))
    full = swmm_role_text(cfg_system, "full", cfg_analysis)
    assert ["OF_A", "-1.0", "TIMESERIES", "hhemt_water_level_0"] in _rows(full, "OUTFALLS")
    assert ["OF_F", "0", "FREE", "NO"] in _rows(full, "OUTFALLS")


def _preflight_configs(tmp_path, rows=None, **analysis):
    """The two configs the preflight pass reads, over the one-layer scenario stand-in and a processed DEM on disk."""
    import shutil

    scenario, _ = _scenario(tmp_path, rows if rows is not None else FOUR[:1])
    cfg_system, cfg_analysis = scenario._system.cfg_system, scenario._analysis.cfg_analysis
    cfg_system.system_directory = tmp_path / "system"
    cfg_system.system_directory.mkdir()
    cfg_system.target_dem_resolution = 10.0
    cfg_system.DEM_fullres = tmp_path / "dem.tif"
    shutil.copy(tmp_path / "dem.tif", cfg_system.system_directory / "elevation_10.00m.dem")
    cfg_analysis.__dict__.update(
        external_boundary_value_attribute=None,
        swmm_outfalls_to_keep=["OF_F"],
        swmm_node_rim_from_dem=False,
        dem_rim_resolution="processed",
        weather_timeseries=None,
        weather_events_to_simulate=None,
    )
    cfg_analysis.__dict__.update(analysis)
    return cfg_system, cfg_analysis


def test_a_valid_boundary_configuration_has_no_problem_and_prints_the_distances_and_snapped_cells(tmp_path):
    """NEW CAPABILITY. Wrong implementation: refuse a configuration that states every decision, or print nothing for
    the user to check the match against."""
    from hhemt.water_level_boundary import preflight_problems

    report = preflight_problems(*_preflight_configs(tmp_path))
    assert [p.message for p in report.problems] == []
    assert "record 0: OF_A 3.0 m, OF_B 3.0 m" in report.lines
    assert any(
        "record 0 (water_level)" in line and "snapped to bottom edge cells (5.0, 5.0) to (35.0, 5.0)" in line
        for line in report.lines
    )


def test_an_unchosen_dem_and_each_missing_layer_setting_are_one_problem_each_and_full_prints_its_notice(tmp_path):
    """NEW CAPABILITY. Wrong implementation: default the DEM, or let the layer's missing settings through (DR-39)."""
    from hhemt.water_level_boundary import FULL_RESOLUTION_NOTICE, preflight_problems

    unset = (
        preflight_problems(*_preflight_configs(tmp_path / "a", dem_rim_resolution=None))
        if (tmp_path / "a").mkdir() is None
        else None
    )
    assert [p.field for p in unset.problems] == ["analysis.dem_rim_resolution"]
    assert unset.problems[0].config_line == "dem_rim_resolution: processed"
    (tmp_path / "b").mkdir()
    bare = preflight_problems(
        *_preflight_configs(
            tmp_path / "b",
            water_level_boundary_series_source=None,
            water_level_boundary_outfall_attribute=None,
            external_boundary_value_attribute="value",
        )
    )
    fields = sorted(p.field for p in bare.problems)
    assert "analysis.water_level_boundary_series_source" in fields
    assert "analysis.water_level_boundary_outfall_attribute" in fields
    assert "analysis.external_boundary_value_attribute" in fields  # no slope or Froude record, so the column is refused
    (tmp_path / "c").mkdir()
    full = preflight_problems(*_preflight_configs(tmp_path / "c", dem_rim_resolution="full"))
    assert FULL_RESOLUTION_NOTICE in full.lines
    (tmp_path / "d").mkdir()
    unused = preflight_problems(
        *_preflight_configs(
            tmp_path / "d", swmm_outfalls_to_keep=["OF_A", "OF_B", "OF_F"], toggle_external_boundaries=False
        )
    )
    assert [p.field for p in unused.problems] == ["analysis.dem_rim_resolution"]
    assert "no outfall is converted" in unused.problems[0].message


def test_full_resolution_rims_write_every_coupling_nodes_depth_and_the_triton_key_is_not_written(tmp_path):
    """NEW CAPABILITY. Wrong implementation: set the depth of the converted outfalls only, or write the TRITON key
    beside depths the toolkit wrote (TRITON would overwrite them from the processed grid) (DR-35, DR-39)."""
    from types import SimpleNamespace

    from hhemt.scenario import swmm_node_rim_key_is_written
    from hhemt.swmm_utils import swmm_role_text

    scenario, _ = _scenario(tmp_path, FOUR[:1])
    cfg_system, cfg_analysis = scenario._system.cfg_system, scenario._analysis.cfg_analysis
    cfg_system.system_directory = tmp_path / "system"
    cfg_system.swmm_coupling_mode = "all_junctions"
    (DEM + 2.0).rio.to_raster(tmp_path / "high.tif")  # ground at 2.0 m everywhere
    cfg_system.DEM_fullres = tmp_path / "high.tif"
    cfg_analysis.dem_rim_resolution = "full"
    cfg_analysis.swmm_node_rim_from_dem = True
    rows = {r[0]: float(r[2]) for r in _rows(swmm_role_text(cfg_system, "hydraulics", cfg_analysis), "JUNCTIONS")}
    assert rows == {"J1": 2.0, "OF_A": 3.0, "OF_B": 3.0}  # ground 2.0 minus each invert (0, -1, -1)
    assert swmm_node_rim_key_is_written(cfg_analysis) is False
    # Control that varies the property (the DEM chosen), not the call: the processed DEM keeps the key and only the
    # converted outfalls take a depth.
    cfg_analysis.dem_rim_resolution = "processed"
    assert swmm_node_rim_key_is_written(cfg_analysis) is True
    assert (
        swmm_node_rim_key_is_written(SimpleNamespace(swmm_node_rim_from_dem=False, dem_rim_resolution="processed"))
        is False
    )


def test_the_raster_behind_the_node_depths_is_read_once_per_path_and_mtime(tmp_path, monkeypatch):
    """NEW CAPABILITY. Wrong implementation: open the DEM on every ``swmm_role_text`` call (six run-entry sites and
    three roles per scenario), which re-reads a whole full-resolution raster each time.

    Counts ``rioxarray.open_rasterio`` calls over five calls of the role text; a touched DEM is read again.
    """
    import os

    from hhemt import swmm_utils

    scenario, _ = _scenario(tmp_path, FOUR[:1])
    cfg_system, cfg_analysis = scenario._system.cfg_system, scenario._analysis.cfg_analysis
    cfg_system.system_directory = tmp_path / "system"
    cfg_system.DEM_fullres = tmp_path / "dem.tif"
    cfg_analysis.dem_rim_resolution = "full"
    cfg_analysis.swmm_node_rim_from_dem = False
    swmm_utils._loaded_raster.cache_clear()
    reads: list[str] = []
    real = rioxarray.open_rasterio
    monkeypatch.setattr(rioxarray, "open_rasterio", lambda *a, **k: (reads.append(str(a[0])), real(*a, **k))[1])
    for role in ("full", "hydraulics", "hydrology", "hydraulics", "hydraulics"):
        swmm_utils.swmm_role_text(cfg_system, role, cfg_analysis)
    assert reads == [str(tmp_path / "dem.tif")]
    stat = (tmp_path / "dem.tif").stat()
    os.utime(tmp_path / "dem.tif", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    swmm_utils.swmm_role_text(cfg_system, "hydraulics", cfg_analysis)
    assert len(reads) == 2, "control: a changed DEM is read again"
