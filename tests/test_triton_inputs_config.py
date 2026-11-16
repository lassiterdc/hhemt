"""The optional TRITON inputs a user supplies: observation points and initial-condition rasters.

Pure text and tiny arrays, laptop tier, no solver and no compile. A two-point CSV and a few-cell DEM header
stand in for a study site, so every assertion here reads a rendered cfg line, a written location file or a
refusal message rather than a solver run. Each test is held to 0.15 s over setup, call and teardown.

Why the inputs are checked in the toolkit and not left to TRITON: a point outside the grid prints an error and
the run continues, a header row mistaken for a point is read as one, and an initial-condition raster with the
wrong grid is read onto the wrong cells without a message. The solver reports none of these as a failure.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal, get_args, get_origin

import pytest
from pydantic import ValidationError

# Imported at collection so the lazy import inside `triton_input_settings` costs no test time.
import hhemt.system  # noqa: F401
from hhemt.bundle._path_policy import _EXCLUDABLE_CATALOG, _PATH_FIELD_POLICY, PathPolicy
from hhemt.config.analysis import analysis_config
from hhemt.config.reprex_taxonomy import _FIELD_BUCKET, field_bucket
from hhemt.config.system import system_config
from hhemt.exceptions import ConfigurationError
from hhemt.triton_cfg import (
    _INITIAL_CONDITIONS,
    LEVEL_UNIT_FACTORS,
    OBSERVATION_FILE_NAME,
    ObservationPoint,
    observation_cell,
    observation_point_problems,
    parse_observation_points,
    render_triton_cfg,
    triton_input_settings,
)
from hhemt.validation import ValidationResult, _validate_observation_interval, _validate_observation_points_file
from tests.test_triton_cfg_render import MAPPING, _analysis

# A 3-row, 4-column DEM, cell size 10, lower-left corner (1000, 2000): x spans 1000-1040 and y spans 2000-2030.
# Row 0 is the NORTH row (the writer's order). Row 1 holds a building cell (80) and a NODATA cell (-9999).
HEADER = {"ncols": 4, "nrows": 3, "xllcorner": 1000.0, "yllcorner": 2000.0, "cellsize": 10.0}
ELEVATIONS = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 80.0, 7.0, -9999.0],
    [9.0, 10.0, 11.0, 12.0],
]
# The four settings that shape the depth raster, read by name from the analysis config.
_DEPTH_SETTINGS = (
    "TRITON_initial_depth_raster_kind",
    "TRITON_initial_depth_raster_units",
    "TRITON_initial_depth_raster_datum_offset",
    "TRITON_initial_depth_nodata_fill",
)
IN_NORTH_WEST = ("a", 1005.0, 2025.0)  # row 0, col 0
IN_SOUTH_EAST = ("b", 1035.0, 2005.0)  # row 2, col 3
ON_BUILDING = ("bldg", 1015.0, 2015.0)  # row 1, col 1
ON_NODATA = ("hole", 1035.0, 2015.0)  # row 1, col 3
EAST_OF_GRID = ("east", 1041.0, 2025.0)
NORTH_OF_GRID = ("north", 1005.0, 2031.0)


def _dem(tmp_path, header=HEADER, rows=ELEVATIONS):
    """A processed-DEM text file in the writer's layout: six header lines, then the rows north to south."""
    lines = [
        f"ncols         {header['ncols']}",
        f"nrows         {header['nrows']}",
        f"xllcorner     {header['xllcorner']}",
        f"yllcorner     {header['yllcorner']}",
        f"cellsize      {header['cellsize']}",
        "NODATA_value  -9999",
    ]
    lines += [" ".join(f"{v:.5f}" for v in row) for row in rows]
    path = tmp_path / "dem.dem"
    path.write_text("\n".join(lines) + "\n")
    return path


def _points_csv(tmp_path, points):
    path = tmp_path / "points.csv"
    path.write_text("id,x,y\n" + "".join(f"{i},{x},{y}\n" for i, x, y in points))
    return path


def _inputs(tmp_path, points=None, **analysis):
    """``triton_input_settings`` for stand-in configs: a system with ``points``, an analysis with ``analysis``."""
    cfg_system = SimpleNamespace(
        observation_points=_points_csv(tmp_path, points) if points is not None else None,
        dem_outside_watershed_height=90.0,
        dem_building_height=80.0,
    )
    cfg_analysis = SimpleNamespace(
        TRITON_observation_interval_s=analysis.get("interval"),
        TRITON_initial_depth_raster=analysis.get("depth"),
        TRITON_initial_qx_raster=analysis.get("qx"),
        TRITON_initial_qy_raster=analysis.get("qy"),
        TRITON_initial_depth_raster_kind=analysis.get("kind"),
        TRITON_initial_depth_raster_units=analysis.get("units"),
        TRITON_initial_depth_raster_datum_offset=analysis.get("offset"),
        TRITON_initial_depth_nodata_fill=analysis.get("fill"),
    )
    return triton_input_settings(
        cfg_system,
        cfg_analysis,
        dem_path=_dem(tmp_path),
        observation_file=tmp_path / "sim" / OBSERVATION_FILE_NAME,
    )


def _cfg_lines(settings):
    return render_triton_cfg({**MAPPING, **settings}, coupled=True).splitlines()


def _raster(tmp_path, name, values=None, **over):
    """An ESRI ASCII initial-condition grid whose header is the DEM's unless ``over`` changes a key.

    ``values`` are the rows, north first, and default to zeros on the DEM's 3 by 4 shape. The header carries the
    no-data value -9999, as a grid exported from a GIS does.
    """
    header = {**HEADER, **over}
    rows = values if values is not None else [[0] * HEADER["ncols"]] * HEADER["nrows"]
    path = tmp_path / name
    path.write_text(
        f"ncols {header['ncols']}\nnrows {header['nrows']}\nxllcorner {header['xllcorner']}\n"
        f"yllcorner {header['yllcorner']}\ncellsize {header['cellsize']}\nNODATA_value -9999\n"
        + "".join(" ".join(str(v) for v in row) + "\n" for row in rows)
    )
    return path


def _normalized_rows(settings, placeholder):
    """The value rows of the file a cfg key names, as lists of the tokens as written."""
    return [line.split(" ") for line in Path(settings[placeholder]).read_text().split("\n") if line]


def test_two_points_and_an_interval_render_the_three_keys_and_write_the_file_in_csv_order(tmp_path):
    """Property: the cfg names the location file and the interval; the file holds ``x,y`` rows in CSV order.

    Class: any valid point list with an interval set. Why: TRITON reads the file with no names, so the CSV's row
    order is the identity of every point downstream. What kills it: writing the rows sorted, with the header, or
    leaving ``time_series_flag`` at 0. Second correct implementation: writing the file with a trailing space or
    CRLF, which the assertions on the parsed rows tolerate.
    """
    settings = _inputs(tmp_path, [IN_SOUTH_EAST, IN_NORTH_WEST], interval=120)
    lines = _cfg_lines(settings)
    location_file = tmp_path / "sim" / OBSERVATION_FILE_NAME
    assert lines.count("time_series_flag=1") == 1
    assert lines.count(f'observation_loc_file="{location_file}"') == 1
    assert lines.count("print_observation=120") == 1
    assert [row.split(",") for row in location_file.read_text().split()] == [["1035", "2005"], ["1005", "2025"]]


def test_no_points_and_no_rasters_leave_the_cfg_unchanged_and_write_nothing(tmp_path):
    """Property: with nothing set the rendered cfg is byte-identical to the render without these settings.

    Class: the default configuration. Control: with points set the same comparison differs, so the equality is
    not vacuous. Why: the defaults must not add a line TRITON reads. What kills it: always writing
    ``time_series_flag=1`` or always writing the location file. Second correct implementation: returning the
    placeholders under different default objects, which the text comparison does not see.
    """
    settings = _inputs(tmp_path)
    assert render_triton_cfg({**MAPPING, **settings}, coupled=True) == render_triton_cfg(MAPPING, coupled=True)
    assert not (tmp_path / "sim").exists()
    with_points = _inputs(tmp_path, [IN_NORTH_WEST])
    assert render_triton_cfg({**MAPPING, **with_points}, coupled=True) != render_triton_cfg(MAPPING, coupled=True)


def test_points_with_no_interval_write_no_print_observation(tmp_path):
    """Class: points set, interval unset. Why: unset means TRITON's own default, the reporting step.

    What kills it: writing ``print_observation=None`` or a zero. Second correct implementation: emitting the
    reporting interval explicitly, which would fail this test and is excluded because the shape says unset.
    """
    lines = _cfg_lines(_inputs(tmp_path, [IN_NORTH_WEST]))
    assert "time_series_flag=1" in lines
    assert not [ln for ln in lines if ln.startswith("print_observation")]


@pytest.mark.parametrize(
    ("bad", "named", "fragment"),
    [
        (EAST_OF_GRID, "east", "outside the processed DEM"),
        (NORTH_OF_GRID, "north", "outside the processed DEM"),
        (ON_BUILDING, "bldg", "fill height"),
        (ON_NODATA, "hole", "NODATA"),
    ],
)
def test_a_point_off_the_grid_or_on_a_dead_cell_is_refused_naming_its_id(tmp_path, bad, named, fragment):
    """Class: one point per way a point can fail the DEM, among valid ones that must not be named.

    Why: TRITON prints an out-of-bounds error and continues, and computes nothing on a fill cell, so the toolkit
    is the only place either is caught. What kills it: checking only one axis, ignoring the fill heights, or
    naming a valid point. Second correct implementation: judging the cell by an equality with the fill height
    rather than ``>=``, which still refuses the 80 m building cell here.
    """
    with pytest.raises(ConfigurationError) as caught:
        _inputs(tmp_path, [IN_NORTH_WEST, bad, IN_SOUTH_EAST])
    message = str(caught.value)
    assert f"point '{named}'" in message and fragment in message
    assert "point 'a'" not in message and "point 'b'" not in message


def test_a_terrain_cell_below_every_fill_height_is_accepted(tmp_path):
    """Control for the dead-cell refusal, with a differently positioned input: the highest terrain cell passes.

    Class: a cell whose elevation is high but below the smaller fill height. Why: it proves the refusal turns on
    the fill heights and not on elevation in general. What kills it: refusing any cell above some constant.
    Second correct implementation: any threshold at the fill height itself.
    """
    assert _inputs(tmp_path, [("high", 1025.0, 2005.0)])["TIME_SERIES_FLAG"] == 1  # row 2, col 2: 11.0


def test_the_cell_of_a_point_is_triton_s_own_formula_including_its_edges():
    """Property: ``observation_cell`` is the solver's column and row formula, ``ceil(v + 1e-16) - 1`` of each.

    Here ``v`` is ``(x-xll)/cs`` for the column and ``nrows-(y-yll)/cs`` for the row. Class: points inside a
    cell, on the north-west and south-east corners, and one cell past the east edge. Why: the toolkit refuses a
    point from the cell it computes, so a different formula refuses a point TRITON accepts or accepts one it
    drops. Values below are the solver's ``calc_src_col`` and ``calc_src_row`` worked by hand at faba2232. What
    kills it: floor instead of ceil minus one, or counting rows from the south. Second correct implementation:
    the same formula with ``numpy.ceil``.
    """
    cell = lambda x, y: observation_cell(ObservationPoint("p", x, y), HEADER)  # noqa: E731
    assert cell(1005.0, 2025.0) == (0, 0)
    assert cell(1000.0, 2030.0) == (0, 0)  # the north-west corner belongs to cell (0, 0)
    assert cell(1040.0, 2000.0) == (2, 3)  # the south-east corner: ceil(4.0) - 1 = 3 and ceil(3.0) - 1 = 2, inside
    assert cell(1035.0, 2005.0) == (2, 3)
    assert cell(1041.0, 2025.0) == (0, 4)  # one cell past the east edge


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("name,x,y\na,1,2\n", "must be exactly 'id,x,y'"),
        ("id,x,y\n", "lists no points"),
        ("id,x,y\na,1,2\na,3,4\n", "id 'a' appears in data rows 1 and 2"),
        ("id,x,y\n,1,2\n", "empty id"),
        ("id,x,y\nq,nan,2\n", "point 'q': x is 'nan'"),
        ("id,x,y\nq,1,inf\n", "point 'q': y is 'inf'"),
        ("id,x,y\nq,1\n", "has 2 fields"),
        ("", "file is empty"),
    ],
)
def test_a_malformed_observation_file_is_refused_with_its_reason(text, fragment):
    """Class: one input per refusal rule over the CSV's format, header first.

    Why: TRITON's location file has no header, so a header read as a point puts a point at NaN, and TRITON
    reads the first line unconditionally, so an empty file crashes it. What kills it: ignoring the header,
    accepting a repeated id, or accepting a non-finite number. Second correct implementation: collecting every
    problem instead of stopping at the first, which these single-defect inputs cannot tell apart.
    """
    points, problems = parse_observation_points(text)
    assert any(fragment in problem for problem in problems), problems


def test_a_well_formed_file_parses_to_its_points_in_order():
    """Control for the refusals above: valid rows give no problem, and quoted ids and CRLF line ends survive."""
    points, problems = parse_observation_points('id,x,y\r\n"gage 7",1005.5,2025\r\nb,1035,2005\r\n')
    assert problems == []
    assert points == [ObservationPoint("gage 7", 1005.5, 2025.0), ObservationPoint("b", 1035.0, 2005.0)]


def test_the_preflight_names_the_file_problem_and_the_orphan_interval(tmp_path):
    """Class: a bad file at the system level, and an interval with no points at the pair level.

    Why: the file is judged before any scenario exists, and the interval lives in a different config from the
    points, so only the pair can refuse it. What kills it: judging the interval alone, or skipping the file read.
    Second correct implementation: raising at config load instead, which would not be reachable from the pair.
    """
    bad = tmp_path / "points.csv"
    bad.write_text("name,x,y\n")
    result = ValidationResult()
    _validate_observation_points_file(SimpleNamespace(observation_points=bad), result)
    assert [e.field for e in result.errors] == ["system.observation_points"]
    assert "'id,x,y'" in result.errors[0].message
    orphan = ValidationResult()
    _validate_observation_interval(
        SimpleNamespace(observation_points=None), SimpleNamespace(TRITON_observation_interval_s=60.0), orphan
    )
    assert [e.field for e in orphan.errors] == ["analysis.TRITON_observation_interval_s"]
    paired = ValidationResult()
    _validate_observation_interval(
        SimpleNamespace(observation_points=bad), SimpleNamespace(TRITON_observation_interval_s=60.0), paired
    )
    assert paired.errors == []
    assert observation_point_problems(SimpleNamespace(observation_points=None)) == []


@pytest.mark.parametrize("value", [0, -5, float("inf")])
def test_a_non_positive_or_infinite_interval_is_refused_at_load(tmp_path, value):
    """Class: the three values a positive finite number excludes. Why: ``print_observation`` is a cadence in
    seconds. What kills it: checking ``>= 0`` or ignoring infinity. Second correct implementation: a pydantic
    ``gt=0`` constraint, which this assertion on the field name also meets.
    """
    with pytest.raises(ValidationError, match="TRITON_observation_interval_s"):
        _analysis(tmp_path, TRITON_observation_interval_s=value)
    assert _analysis(tmp_path, TRITON_observation_interval_s=30).TRITON_observation_interval_s == 30


@pytest.mark.parametrize(("which", "key"), [("depth", "h_infile"), ("qx", "qx_infile"), ("qy", "qy_infile")])
def test_a_raster_on_the_dem_grid_renders_its_key_and_the_unset_ones_render_none(tmp_path, which, key):
    """Property: a set raster writes its own cfg key, quoted; the two unset rasters write nothing.

    Class: each of the three rasters set in turn, with a header equal to the DEM's. Why: the keys are the only way
    TRITON learns of an initial condition, and an unset field must leave the cfg as it was. What kills it: writing
    all three keys, a misspelled key, or writing the key bare. Second correct implementation: writing the path
    relative, which this assertion on the key and quote accepts only if the path matches, so it pins the absolute form.
    The path is the normalized copy under ``triton_initial_conditions/`` beside the DEM, never the user's grid.
    """
    raster = _raster(tmp_path, f"{which}.asc")
    settings = _inputs(tmp_path, **{which: raster}, kind="depth")
    lines = _cfg_lines(settings)
    normalized = settings[{"depth": "H_INFILE", "qx": "QX_INFILE", "qy": "QY_INFILE"}[which]]
    assert Path(normalized).parent == tmp_path / "triton_initial_conditions"
    assert lines.count(f'{key}="{normalized}"') == 1
    assert [ln.split("=")[0] for ln in lines if ln.startswith(("h_infile", "qx_infile", "qy_infile"))] == [key]


@pytest.mark.parametrize(
    ("over", "key"),
    [
        ({"ncols": 5}, "ncols"),
        ({"nrows": 2}, "nrows"),
        ({"xllcorner": 1010.0}, "xllcorner"),
        ({"yllcorner": 1990.0}, "yllcorner"),
        ({"cellsize": 5.0}, "cellsize"),
    ],
)
def test_a_raster_whose_grid_differs_is_refused_naming_the_raster_and_the_key(tmp_path, over, key):
    """Class: each of the five grid keys differing, one at a time. Why: TRITON lays the raster over the DEM with
    no check, so a shifted or resized grid places the water in the wrong cells silently. What kills it: comparing
    only the dimensions. Second correct implementation: comparing a tuple of the five keys at once, which names
    the differing key identically.
    """
    raster = _raster(tmp_path, "qx.asc", **over)
    with pytest.raises(ConfigurationError, match=f"TRITON_initial_qx_raster.*'{key}'"):
        _inputs(tmp_path, qx=raster)


def test_a_raster_that_names_cell_centres_is_refused(tmp_path):
    """Class: a centre-registered header. Why: TRITON reads corners, so accepting it shifts the grid half a cell.

    What kills it: ignoring the centre keys and reading the missing corner as absent only. Second correct
    implementation: converting the centre to a corner, which the shape does not ask for.
    """
    raster = tmp_path / "c.asc"
    raster.write_text("ncols 4\nnrows 3\nxllcenter 1005\nyllcenter 2005\ncellsize 10\n0 0 0 0\n")
    with pytest.raises(ConfigurationError, match="names cell centres"):
        _inputs(tmp_path, depth=raster, kind="depth")


def test_the_names_the_toolkit_reads_are_fields_of_the_real_models(tmp_path):
    """Pin: ``triton_input_settings`` reads its config attributes with ``getattr`` so stand-in configs read unset.

    Why: that tolerance would also read a misspelled name as unset, so each name is held against the models' own
    fields. What kills it: renaming a field in the model and not in the table. Second correct implementation:
    reading the attributes directly, which would make this test unnecessary and break the stand-in configs.
    """
    for _placeholder, _key, field in _INITIAL_CONDITIONS:
        assert field in analysis_config.model_fields
    assert "TRITON_observation_interval_s" in analysis_config.model_fields
    for field in _DEPTH_SETTINGS:
        assert field in analysis_config.model_fields
    for field in ("observation_points", "dem_outside_watershed_height", "dem_building_height"):
        assert field in system_config.model_fields


def test_each_new_input_has_its_bundle_and_taxonomy_row():
    """Property: every new Path field is bundled relative-or-none, catalogued, and bucketed; the interval too.

    Class: the four new Path fields and the interval. Why: a Path field with no policy row stops bundle emission,
    and the catalog and taxonomy tests demand totality. What kills it: a row under the wrong policy. Second
    correct implementation: ``BUNDLE_RELATIVE`` for a field that is required, which these optional fields are not.
    """
    path_fields = (
        "observation_points",
        "TRITON_initial_depth_raster",
        "TRITON_initial_qx_raster",
        "TRITON_initial_qy_raster",
    )
    for name in path_fields:
        assert _PATH_FIELD_POLICY[name] is PathPolicy.BUNDLE_RELATIVE_OR_NONE
        assert _EXCLUDABLE_CATALOG[name].excludable
        assert field_bucket(name) == "experiment"
    assert _FIELD_BUCKET["TRITON_observation_interval_s"] == "experiment"
    for name in _DEPTH_SETTINGS:
        assert _FIELD_BUCKET[name] == "experiment"


def test_a_missing_raster_or_points_file_is_refused_at_load(tmp_path):
    """Class: a path that does not exist, for the analysis rasters. Why: the shared path validator is what makes
    the new fields fail at load and not mid-run. What kills it: marking the field toolkit-owned. Second correct
    implementation: a field validator of its own.
    """
    with pytest.raises(ValidationError, match="TRITON_initial_depth_raster"):
        _analysis(tmp_path, TRITON_initial_depth_raster=str(tmp_path / "absent.asc"))
    present = tmp_path / "h.asc"
    present.write_text("x")
    loaded = _analysis(tmp_path, TRITON_initial_depth_raster=str(present), TRITON_initial_depth_raster_kind="depth")
    assert loaded.TRITON_initial_depth_raster == present


def _wire_inputs(scen, tmp_path, points, **analysis):
    """Give a stand-in scenario a real processed DEM and the observation and raster settings under test."""
    scen._system.sys_paths.dem_processed = _dem(tmp_path)
    scen._system.cfg_system.observation_points = _points_csv(tmp_path, points)
    scen._system.cfg_system.dem_outside_watershed_height = 90.0
    scen._system.cfg_system.dem_building_height = 80.0
    scen._analysis.cfg_analysis.TRITON_observation_interval_s = analysis.get("interval")
    for field in ("depth", "qx", "qy"):
        setattr(scen._analysis.cfg_analysis, f"TRITON_initial_{field}_raster", analysis.get(field))
    scen._analysis.cfg_analysis.TRITON_initial_depth_raster_kind = analysis.get("kind")


def test_the_coupled_cfg_generator_writes_the_location_file_and_the_three_keys(tmp_path, monkeypatch):
    """Property: the coupled generator runs the inputs through ``triton_input_settings`` and renders the result.

    Class: a coupled scenario with two points, an interval and one raster, on stand-in collaborators. Why: the
    pure functions above are only worth anything if the scenario calls them, and the stand-in configs of the
    other scenario tests carry none of these fields, so this is the one test of the call itself. What kills it:
    leaving the mapping at the placeholders' defaults, or writing the location file elsewhere than beside the cfg.
    Second correct implementation: computing the file path from ``sim_folder``, which is the cfg's folder.
    """
    from tests.test_scenario_boundary_guards import _scenario_for_coupled_cfg

    reads_both = "a146fb6ed7c22790fb9a15ff854c446db4519ad2"
    scen, cfg = _scenario_for_coupled_cfg(
        tmp_path,
        monkeypatch,
        rim_from_dem=False,
        environment="container",
        branch_key=reads_both,
        recorded_sha=reads_both,
    )
    depth = _raster(tmp_path, "h.asc")
    _wire_inputs(scen, tmp_path, [IN_SOUTH_EAST, IN_NORTH_WEST], interval=90, depth=depth, kind="depth")
    scen._generate_TRITON_SWMM_cfg()
    lines = cfg.read_text().splitlines()
    location_file = cfg.parent / OBSERVATION_FILE_NAME
    assert "time_series_flag=1" in lines
    assert f'observation_loc_file="{location_file}"' in lines
    assert "print_observation=90" in lines
    normalized = next(tmp_path.glob("triton_initial_conditions/TRITON_initial_depth_raster_*.txt"))
    assert f'h_infile="{normalized}"' in lines
    assert location_file.read_text() == "1035,2005\n1005,2025\n"


def test_the_triton_only_cfg_generator_writes_the_same_location_file_and_keys(tmp_path, monkeypatch):
    """Class: the TRITON-only generator, which fills a second mapping from the same template.

    Why: one template feeds two generators, and a one-sided edit would leave the other cfg without its points.
    What kills it: wiring only the coupled generator. Second correct implementation: one shared helper that
    builds both mappings, which this assertion on the written cfg accepts.
    """
    from tests.test_scenario_boundary_guards import _scenario_for_coupled_cfg

    reads_both = "a146fb6ed7c22790fb9a15ff854c446db4519ad2"
    scen, cfg = _scenario_for_coupled_cfg(
        tmp_path,
        monkeypatch,
        rim_from_dem=False,
        environment="container",
        branch_key=reads_both,
        recorded_sha=reads_both,
    )
    scen._system.cfg_system.toggle_triton_model = True
    scen.scen_paths.triton_cfg = cfg.parent / "TRITON.cfg"
    scen.log = SimpleNamespace(triton_cfg_created=SimpleNamespace(set=lambda value: None))
    _wire_inputs(scen, tmp_path, [IN_NORTH_WEST], interval=None)
    scen._generate_TRITON_cfg()
    lines = scen.scen_paths.triton_cfg.read_text().splitlines()
    assert "time_series_flag=1" in lines
    assert f'observation_loc_file="{cfg.parent / OBSERVATION_FILE_NAME}"' in lines
    assert not [ln for ln in lines if ln.startswith(("print_observation", "h_infile"))]
    assert (cfg.parent / OBSERVATION_FILE_NAME).read_text() == "1005,2025\n"


# ---------------------------------------------------------------------------------------------------------------
# Initial-condition rasters reach TRITON in the form it reads
# ---------------------------------------------------------------------------------------------------------------


def _float_rows(settings, placeholder):
    """The value rows of the normalized file a cfg key names, as floats."""
    return [[float(token) for token in row] for row in _normalized_rows(settings, placeholder)]


def test_a_grid_in_any_layout_is_rewritten_as_one_header_less_file_of_fixed_point_tokens(tmp_path):
    """Property: the file the cfg names has ``nrows`` lines of ``ncols`` single-space tokens, each with a point.

    Class: a grid written with a tab, a run of spaces, a leading space, CRLF line ends and the value ``1e-05``.
    Why: TRITON counts the columns as the single-space tokens of line 1 and the rows as the lines, reads a token
    with no point as an integer, and exits on a mismatch, so a header or a tab breaks the run and ``1e-05`` is read
    wrong. What kills it: handing TRITON the user's grid, writing the header, or writing ``1e-05`` as written.
    Second correct implementation: writing six decimals instead of five, which the float comparison accepts.
    """
    raster = tmp_path / "h.asc"
    raster.write_bytes(
        b"ncols 4\r\nnrows 3\r\nxllcorner 1000.0\r\nyllcorner 2000.0\r\ncellsize 10.0\r\nNODATA_value -9999\r\n"
        b"1\t2  3 1e-05\r\n 4 5 6 7\r\n8 9 10 11\r\n"
    )
    settings = _inputs(tmp_path, depth=raster, kind="depth")
    text = Path(settings["H_INFILE"]).read_bytes()
    rows = _normalized_rows(settings, "H_INFILE")
    assert b"\r" not in text and text.endswith(b"\n")
    assert [len(row) for row in rows] == [4, 4, 4]
    assert all("." in token for row in rows for token in row)
    assert _float_rows(settings, "H_INFILE") == [[1, 2, 3, 1e-05], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_a_value_row_of_the_wrong_length_or_a_word_that_is_no_number_is_refused_naming_its_line(tmp_path):
    """Class: a short row, and a word that is not a number. Why: a ragged grid reaches TRITON as a column-count
    exit, and the message must name the line so the user finds it. What kills it: padding the short row, or
    reading the word as 0. Second correct implementation: checking the row counts against the header first.
    """
    short = _raster(tmp_path, "short.asc", values=[[1, 2, 3, 4], [5, 6, 7], [9, 10, 11, 12]])
    with pytest.raises(ConfigurationError, match=r"TRITON_initial_qx_raster \(short.asc\): line 8 holds 3 values"):
        _inputs(tmp_path, qx=short)
    word = _raster(tmp_path, "word.asc", values=[[1, 2, 3, 4], [5, "n/a", 7, 8], [9, 10, 11, 12]])
    with pytest.raises(ConfigurationError, match=r"line 8 holds 'n/a', which is not a number"):
        _inputs(tmp_path, qx=word)


@pytest.mark.parametrize(
    ("rows", "count", "first"),
    [
        ([[1, 2, -9999, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 1, "row 0, col 2"),
        ([[1, 2, 3, 4], [5, 6, 7, 8], [-9999, 10, 11, -9999]], 2, "row 2, col 0"),
    ],
)
def test_a_depth_no_data_cell_takes_the_fill_or_the_raster_is_refused_with_the_line_to_write(
    tmp_path, capsys, rows, count, first
):
    """Property: with a fill every no-data cell takes it and the filled count is logged; without one the raster is
    refused naming the count, the first cell and the line to write.

    Class: one no-data cell, and two with the first not in row 0. Why: TRITON has no no-data handling, so -9999
    would flood the cell to 9999 m. What kills it: filling with 0 when the fill is unset, counting every cell, or
    naming the last cell. Second correct implementation: reporting the first cell in column-major order, which the
    second input separates from the row-major first cell asserted here.
    """
    raster = _raster(tmp_path, "h.asc", values=rows)
    with pytest.raises(ConfigurationError) as caught:
        _inputs(tmp_path, depth=raster, kind="depth")
    message = str(caught.value)
    assert f"{count} no-data cells, the first at ({first})" in message
    assert "Write: TRITON_initial_depth_nodata_fill: 0" in message
    settings = _inputs(tmp_path, depth=raster, kind="depth", fill=0.25)
    expected = [[0.25 if v == -9999 else float(v) for v in row] for row in rows]
    assert _float_rows(settings, "H_INFILE") == expected
    assert f"{count} no-data cells set to 0.25" in capsys.readouterr().out


def test_a_non_finite_cell_is_a_no_data_cell_even_when_the_header_declares_none(tmp_path):
    """Class: a header with no ``NODATA_value`` and a ``nan`` cell. Why: the shape detects no-data by the declared
    value or by a value that is not finite, and a grid with no declaration still holds the second kind. What kills
    it: reading ``nan`` as a number or checking the declaration only. Second correct implementation: ``inf`` in
    place of ``nan``, which is not finite either.
    """
    raster = tmp_path / "h.asc"
    raster.write_text(
        "ncols 4\nnrows 3\nxllcorner 1000.0\nyllcorner 2000.0\ncellsize 10.0\n1 2 3 4\n5 nan 7 8\n9 10 11 12\n"
    )
    with pytest.raises(ConfigurationError, match=r"1 no-data cells, the first at \(row 1, col 1\)"):
        _inputs(tmp_path, depth=raster, kind="depth")
    assert _float_rows(_inputs(tmp_path, depth=raster, kind="depth", fill=0), "H_INFILE")[1] == [5, 0, 7, 8]


def test_a_negative_depth_is_refused_with_its_count_and_first_cell_and_is_not_clipped(tmp_path):
    """Property: under ``depth`` any negative non-no-data cell refuses the raster.

    Class: two negative cells, the first in row 0. Why: a negative depth is a unit or kind mistake, and a clip
    would hide it. What kills it: clipping to 0, or counting the no-data cell as negative (the -9999 cell here must
    not be counted). Second correct implementation: refusing on the first negative cell alone, which the count of 2
    asserted here separates.
    """
    raster = _raster(tmp_path, "h.asc", values=[[1, -2, 3, 4], [5, 6, -7, -9999], [9, 10, 11, 12]])
    with pytest.raises(ConfigurationError) as caught:
        _inputs(tmp_path, depth=raster, kind="depth", fill=0)
    assert "2 cells are negative, the first at (row 0, col 1) with value -2" in str(caught.value)


@pytest.mark.parametrize("which", ["qx", "qy"])
def test_a_discharge_no_data_cell_takes_zero_and_a_negative_discharge_is_kept(tmp_path, which):
    """Property: no-data cells of a discharge raster become 0 with no fill set; negative flow is legal.

    Class: both discharge rasters, with a no-data cell and a negative cell. Why: discharge has no fill setting and
    flows run in either direction. What kills it: refusing the no-data cell, or applying the depth rule to flow.
    Second correct implementation: zero-filling by ``np.nan_to_num``, which would also zero ``inf`` where this
    implementation refuses nothing and still writes 0.
    """
    raster = _raster(tmp_path, f"{which}.asc", values=[[1, -2, -9999, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    settings = _inputs(tmp_path, **{which: raster})
    assert _float_rows(settings, f"{which.upper()}_INFILE")[0] == [1, -2, 0, 4]


def test_a_water_surface_raster_becomes_depths_above_the_dem_with_below_ground_cells_dry(tmp_path, capsys):
    """Property: depth = elevation in meters plus the offset, less the DEM, floored at 0; no-data takes the fill.

    Class: a raster of feet with an offset of 0.5 m, cells above, at and below the DEM, a negative elevation, and
    a no-data cell whose source value would be far below the ground if it were converted. Why: TRITON reads depths,
    so an elevation raster must lose the ground height, and the no-data cell must be set after the subtraction or
    it floods. What kills it: skipping the offset, skipping the unit factor, subtracting the DEM from the no-data
    cell, or writing a negative depth. Second correct implementation: computing in feet and converting last,
    which the tolerance accepts.
    """
    feet = 0.3048
    elevations = [[10, 1.5 / feet, 1, 30], [20, 100, 30, -9999], [40, 33, 50, -5]]
    # Independent arithmetic, cell by cell: meters, plus 0.5, less the DEM, floor 0. Cell (0, 1) is 1.5 m, +0.5 = 2.0 m,
    # exactly the DEM's 2.0, so it is dry.
    expected = [
        [10 * feet + 0.5 - 1.0, 0, 0, 30 * feet + 0.5 - 4.0],
        [20 * feet + 0.5 - 5.0, 0, 30 * feet + 0.5 - 7.0, 0.1],
        [40 * feet + 0.5 - 9.0, 33 * feet + 0.5 - 10.0, 50 * feet + 0.5 - 11.0, 0],
    ]
    raster = _raster(tmp_path, "h.asc", values=elevations)
    settings = _inputs(tmp_path, depth=raster, kind="water_surface_elevation", units="ft", offset=0.5, fill=0.1)
    got = _float_rows(settings, "H_INFILE")
    assert [v for row in got for v in row] == pytest.approx([v for row in expected for v in row], abs=2e-5)
    assert "cells below the ground set dry" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("unit", "factor"),
    [("m", 1.0), ("cm", 0.01), ("mm", 0.001), ("in", 0.0254), ("ft", 0.3048), ("ftUS", 1200 / 3937)],
)
def test_each_unit_of_the_closed_set_converts_by_its_own_factor(tmp_path, unit, factor):
    """Property: an elevation of ``2000 / factor`` in ``unit`` over a 12 m DEM cell is a depth of 1988 m.

    Class: each of the six units, one at a time. Why: the factors are the only thing standing between a value in
    feet and a depth that is wrong by the unit. What kills it: a swapped or missing factor, or a US survey foot read
    as an international foot (the two differ by 2 parts per million, which is 4 mm over 2000 m, more than the 1 mm
    tolerance here). Second correct implementation: a table of factors from meters to the unit, applied by division.
    """
    value = 2000 / factor
    raster = _raster(tmp_path, "h.asc", values=[[value] * 4] * 3)
    settings = _inputs(tmp_path, depth=raster, kind="water_surface_elevation", units=unit, offset=0.0)
    assert _float_rows(settings, "H_INFILE")[2][3] == pytest.approx(1988.0, abs=1e-3)


def test_the_unit_set_of_the_config_field_and_the_factor_table_are_the_same_six_names():
    """Pin: ``TRITON_initial_depth_raster_units`` accepts exactly the keys of ``LEVEL_UNIT_FACTORS``.

    Why: the Literal on the field and the table in the renderer are two statements of one closed set, and a unit
    added to one alone is either refused at load or fails at preparation. What kills it: adding a unit to either.
    Second correct implementation: deriving the Literal from the table, which this comparison then confirms.
    """
    annotation = analysis_config.model_fields["TRITON_initial_depth_raster_units"].annotation
    literal = next(arg for arg in get_args(annotation) if get_origin(arg) is Literal)
    assert set(get_args(literal)) == set(LEVEL_UNIT_FACTORS)
    assert LEVEL_UNIT_FACTORS["ftUS"] == 1200 / 3937


def test_a_depth_raster_with_no_kind_is_refused_by_the_preparation_step_too(tmp_path):
    """Class: a stand-in config with a depth raster and no kind. Why: the real config refuses it at load, and the
    stand-in configs of the generator tests bypass that, so the entry refuses it as well rather than guessing.
    What kills it: defaulting to ``depth``. Second correct implementation: raising from the config only.
    """
    with pytest.raises(ConfigurationError, match="TRITON_initial_depth_raster_kind is not set"):
        _inputs(tmp_path, depth=_raster(tmp_path, "h.asc"))


def test_the_normalized_file_is_reused_and_a_changed_setting_names_a_new_file(tmp_path):
    """Property: the same inputs name the same file, which is not rewritten; a different fill names another file.

    Class: two scenario preparations with one raster, then one with another fill. Why: every scenario of an
    analysis prepares the same raster, so the copy is keyed by what shapes it and skipped when present. What kills
    it: a name that omits the settings (a stale copy is reused), or a rewrite per scenario. Second correct
    implementation: keying by a hash of the output values, which this assertion on the equal names accepts.
    """
    raster = _raster(tmp_path, "h.asc", values=[[1, -9999, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    first = _inputs(tmp_path, depth=raster, kind="depth", fill=0.5)["H_INFILE"]
    stamp = Path(first).stat().st_mtime_ns
    again = _inputs(tmp_path, depth=raster, kind="depth", fill=0.5)["H_INFILE"]
    assert again == first and Path(first).stat().st_mtime_ns == stamp
    other = _inputs(tmp_path, depth=raster, kind="depth", fill=0.75)["H_INFILE"]
    assert other != first and _float_rows({"H": other}, "H")[0][1] == 0.75


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"TRITON_initial_depth_raster_kind": None}, "TRITON_initial_depth_raster_kind is required"),
        (
            {"TRITON_initial_depth_raster_kind": "water_surface_elevation"},
            "TRITON_initial_depth_raster_units is required",
        ),
        (
            {"TRITON_initial_depth_raster_kind": "water_surface_elevation", "TRITON_initial_depth_raster_units": "ft"},
            "TRITON_initial_depth_raster_datum_offset is required",
        ),
        ({"TRITON_initial_depth_raster_units": "ft"}, "TRITON_initial_depth_raster_units must be unset"),
        ({"TRITON_initial_depth_raster_datum_offset": 1.0}, "TRITON_initial_depth_raster_datum_offset must be unset"),
        ({"TRITON_initial_depth_nodata_fill": -1.0}, "TRITON_initial_depth_nodata_fill is -1.0"),
        ({"TRITON_initial_depth_raster_kind": "elevation"}, "TRITON_initial_depth_raster_kind"),
    ],
)
def test_the_depth_raster_settings_are_refused_at_load_when_they_do_not_fit_its_kind(tmp_path, over, named):
    """Class: each way the four settings can disagree with the raster and its kind, from a valid depth-kind base.

    Why: a grid of depths and a grid of elevations look alike, so the kind has no default, and a unit set under
    ``depth`` would be silently ignored. What kills it: defaulting the kind, or ignoring a setting that the kind
    does not use. Second correct implementation: one combined message, which the substring match accepts.
    """
    present = tmp_path / "h.asc"
    present.write_text("x")
    base = {"TRITON_initial_depth_raster": str(present), "TRITON_initial_depth_raster_kind": "depth"}
    with pytest.raises(ValidationError, match=named):
        _analysis(tmp_path, **{**base, **over})
    ok = _analysis(
        tmp_path,
        TRITON_initial_depth_raster=str(present),
        TRITON_initial_depth_raster_kind="water_surface_elevation",
        TRITON_initial_depth_raster_units="ftUS",
        TRITON_initial_depth_raster_datum_offset=0.0,
        TRITON_initial_depth_nodata_fill=0,
    )
    assert ok.TRITON_initial_depth_raster_units == "ftUS"


def test_a_depth_setting_with_no_depth_raster_is_refused_at_load(tmp_path):
    """Class: each of the four settings alone, with no raster. Why: a setting with nothing to apply to is a
    mistake in the config, not a default. What kills it: ignoring settings when the raster is unset. Second correct
    implementation: refusing only the kind, which the loop over all four separates.
    """
    for name, value in (
        ("TRITON_initial_depth_raster_kind", "depth"),
        ("TRITON_initial_depth_raster_units", "m"),
        ("TRITON_initial_depth_raster_datum_offset", 0.0),
        ("TRITON_initial_depth_nodata_fill", 0.0),
    ):
        with pytest.raises(ValidationError, match=f"{name} is set, and TRITON_initial_depth_raster is not"):
            _analysis(tmp_path, **{name: value})


@pytest.mark.parametrize("which", ["depth", "qx", "qy"])
def test_a_grid_that_starts_with_a_byte_order_mark_is_read_as_the_same_grid(tmp_path, which):
    """Property: a grid saved with a UTF-8 byte-order mark yields the same values as the grid without one.

    Class: each of the three raster fields, with the three bytes EF BB BF before ``ncols``. Why: a Windows tool
    writes the mark, and without handling the first header word reads as an unknown word and the grid is refused
    with a message that prints the word without the invisible mark. What kills it: reading with plain
    ``utf-8``. Second correct implementation: stripping the mark from the first line after the read, which gives
    the same values.
    """
    plain = _raster(tmp_path, "plain.asc", values=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    marked = tmp_path / "marked.asc"
    marked.write_bytes(b"\xef\xbb\xbf" + plain.read_bytes())
    placeholder = {"depth": "H_INFILE", "qx": "QX_INFILE", "qy": "QY_INFILE"}[which]
    got = _inputs(tmp_path, **{which: marked}, kind="depth")
    assert _float_rows(got, placeholder) == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
