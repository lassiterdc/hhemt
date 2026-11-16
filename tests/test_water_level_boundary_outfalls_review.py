"""Reviewing seat's held tests of the external-boundary layer ( the outfall decisions, the series files).

Reviewed range: ``b68fa76e..4b8d662e``. Every test is text and a few rows, no solver and no synthetic analysis, so each
runs well under 0.15 s. The helpers are the coder's, imported from ``tests.test_water_level_boundary_outfalls``, so the
two files share one set of inputs.

Tests marked REVIEW GAP fail against ``4b8d662e``; each names the shape row it holds and the wrong implementation it
kills. The remaining tests pass and hold a requirement the coder's tests do not state.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import rioxarray  # noqa: F401 -- registers the .rio accessor
import xarray as xr

import hhemt.scenario  # noqa: F401 -- imported at collection, so no test is charged for the module's import time
from hhemt import water_level_boundary as wlb
from hhemt.exceptions import ProcessingError
from hhemt.swmm_runnable import _parse
from hhemt.validation import ValidationResult, _validate_external_boundaries
from hhemt.water_level_boundary import BoundaryRecord, model_outfalls, model_series, series_identity_problems
from tests.test_water_level_boundary_outfalls import (
    BOTTOM,
    FOUR,
    LAYER_MODEL,
    _preflight_configs,
    _scenario,
)

#: The two outfalls read different series: OF_B reads ``other``, whose second value differs.
UNEQUAL_MODEL = LAYER_MODEL.replace("OF_B -1.0 TIMESERIES lvl", "OF_B -1.0 TIMESERIES other").replace(
    "lvl 01/01/2020 01:00 0.7", "lvl 01/01/2020 01:00 0.7\nother 01/01/2020 00:00 0.9\nother 01/01/2020 01:00 1.1"
)


def _warm_the_netcdf_engine() -> None:
    """Write and read one tiny NetCDF at collection, so no test is charged for the engine's first-use import cost."""
    import tempfile

    with tempfile.TemporaryDirectory() as warm:
        path = Path(warm) / "warm.nc"
        xr.Dataset({"v": ("t", np.zeros(1))}, coords={"t": np.arange(1)}).to_netcdf(path, engine="h5netcdf")
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            ds.load()


_warm_the_netcdf_engine()
BOTH = BoundaryRecord(0, "water_level", None, ("OF_A", "OF_B"), tuple(BOTTOM))


# --- V9 and the outfall decisions are asked of a coupled model ---------------------------------------------------


def test_the_outfall_decisions_are_asked_only_of_a_coupled_model(tmp_path):
    """REVIEW GAP. Shape: V9 'Runs when: coupled model on' (V8 is gated on ``toggle_tritonswmm_model``, V9 is not).

    Input class: a system that runs no coupled TRITON-SWMM model (a standalone SWMM or TRITON-only system) whose
    SWMM_model has outfalls no list names. Kills: a preflight that refuses every non-coupled system until it lists
    outfalls whose disposition it never uses. Control that varies the property (the coupled toggle), not the call.
    """
    cfg_system, cfg_analysis = _preflight_configs(
        tmp_path, toggle_external_boundaries=False, swmm_outfalls_to_keep=[], dem_rim_resolution=None
    )
    cfg_system.toggle_tritonswmm_model = True
    coupled = [p.code for p in wlb.preflight_problems(cfg_system, cfg_analysis).problems]
    assert coupled.count("V9") == 2, "control: a coupled model with undecided outfalls is refused"
    cfg_system.toggle_tritonswmm_model = False
    uncoupled = [p.code for p in wlb.preflight_problems(cfg_system, cfg_analysis).problems]
    assert "V9" not in uncoupled


# --- V12 runs for every record naming two or more outfalls ---------------------------------------------------------


def test_the_identity_of_a_records_outfalls_is_checked_before_the_source_is_chosen(tmp_path):
    """REVIEW GAP. Shape V12: 'Runs when: every record naming two or more outfalls', whatever the source.

    Input class: a layer record naming two outfalls that read different series, with ``water_level_boundary_series_
    source`` unset. Kills: a pass that hides V12 until the user has also fixed the source, so the user meets the
    refusals in two rounds. The control sets the source and must show V12.
    """
    cfg_system, cfg_analysis = _preflight_configs(tmp_path, water_level_boundary_series_source=None)
    Path(cfg_system.SWMM_model).write_text(UNEQUAL_MODEL)
    unset = {p.code for p in wlb.preflight_problems(cfg_system, cfg_analysis).problems}
    assert "CFG" in unset, "control: the unset source is itself a problem"
    cfg_analysis.water_level_boundary_series_source = "weather_time_series"
    assert "V12" in {p.code for p in wlb.preflight_problems(cfg_system, cfg_analysis).problems}, "control"
    cfg_analysis.water_level_boundary_series_source = None
    assert "V12" in unset


def test_an_inline_series_and_a_file_series_with_equal_rows_are_identical_and_a_differing_row_is_named(tmp_path):
    """HELD (passes). Shape test list: V12 'passes ... on an inline and a FILE series with equal rows' and 'refuses a
    differing value or time naming the first differing row'. Kills: comparing a FILE series by its name or its path.
    """
    (tmp_path / "other.dat").write_text("01/01/2020 00:00 0.5\n01/01/2020 01:00 0.7\n")
    model = LAYER_MODEL.replace("OF_B -1.0 TIMESERIES lvl", "OF_B -1.0 TIMESERIES other").replace(
        "[TIMESERIES]", '[TIMESERIES]\nother FILE "other.dat"'
    )
    sections = _parse(model)
    args = ([BOTH], model_series(sections, tmp_path), model_outfalls(sections), "swmm_model", "refuse")
    assert series_identity_problems(*args) == []
    (tmp_path / "other.dat").write_text("01/01/2020 00:00 0.5\n01/01/2020 01:00 0.9\n")
    (problem,) = series_identity_problems(*args)
    assert problem.code == "V12" and "row 1" in problem.message


# --- the series policy reaches the writer -------------------------------------------------------------------------


def test_the_writer_refuses_unequal_outfall_series_unless_averaging_was_asked_for(tmp_path):
    """REVIEW GAP. Shape: ``water_level_boundary_unequal_outfall_series`` defaults to ``refuse``; the record's series is
    the mean only 'under ``average``'. The coder's test of the same name asserts the mean under ``refuse``.

    Input class: two outfalls of one record whose base-model series differ, policy ``refuse``, the writer reached
    without the preflight (a model edited after preflight, or a caller that skips it). Kills: a writer that averages
    whenever the series differ, so averaging is the silent default. Control: the same input under ``average``.
    """
    scenario, _ = _scenario(tmp_path, FOUR[:1], source="swmm_model")
    Path(scenario._system.cfg_system.SWMM_model).write_text(UNEQUAL_MODEL)
    cfg_analysis = scenario._analysis.cfg_analysis
    cfg_analysis.water_level_boundary_unequal_outfall_series = "average"
    assert wlb.boundary_level_series(scenario, BOTH).meters == pytest.approx((0.7, 0.9)), "control"
    cfg_analysis.water_level_boundary_unequal_outfall_series = "refuse"
    with pytest.raises(ProcessingError):
        wlb.boundary_level_series(scenario, BOTH)


def test_averaging_series_that_do_not_share_their_times_is_refused():
    """HELD (passes). Shape test list: '``average`` with unequal times refused'. Kills: averaging by position."""
    a = [(wlb._ENGINE_DEFAULT_START, 1.0), (wlb._ENGINE_DEFAULT_START.replace(hour=1), 2.0)]
    shifted = [(t.replace(minute=30), v) for t, v in a]
    with pytest.raises(ProcessingError, match="share their abscissas"):
        wlb.averaged_pairs([a, shifted])
    assert [v for _, v in wlb.averaged_pairs([a, a])] == [1.0, 2.0], "control: equal times average"


# --- the extraction directory is safe under concurrent preparation -----------------------------------------------


def test_a_competing_preparation_that_finishes_the_extraction_first_is_not_an_error(tmp_path, monkeypatch):
    """REVIEW GAP. Shape item 1: the directory is 'written to a temporary sibling ... and moved into place; a reader
    treats a directory without COMPLETE as absent'. Scenario preparation runs in parallel against one system directory.

    Input class: another process completes ``swmm_model_series/{sha12}`` between this call's marker check and its
    rename (simulated at the moment this call clears the target). Kills: ``rmtree(target)`` then ``os.replace`` onto a
    directory the competitor has just filled, which raises ENOTEMPTY in the second preparation.
    """
    model = tmp_path / "model.inp"
    model.write_text(LAYER_MODEL)
    system = tmp_path / "system"
    target = wlb.extracted_series_dir(system, model, model.read_bytes())
    real = shutil.rmtree

    def competitor_finishes(path, *args, **kwargs):
        """Clear the path as asked, then, for the final directory, finish the work as a second process would."""
        real(path, *args, **kwargs)
        if Path(path) == target:
            target.mkdir(parents=True)
            (target / "000.dat").write_text(";; hhemt extracted series lvl from m\n01/01/2020\t00:00\t0.5\n")
            (target / wlb.COMPLETE_MARKER).write_text("000.dat x\n")

    monkeypatch.setattr(shutil, "rmtree", competitor_finishes)
    pointers = wlb.extract_model_series(model, system)
    assert pointers and all(Path(p).is_file() for p in pointers.values())
    assert (target / wlb.COMPLETE_MARKER).is_file()


# --- DR-35: the cell TRITON gives a node -------------------------------------------------------------------------


def test_a_node_on_a_cell_edge_takes_the_cell_triton_gives_it():
    """HELD (passes). Shape DR-35: ``col = ceil((x - xll)/cell + 1e-16) - 1`` and ``row = ceil((nrows - (y - yll)/cell)
    + 1e-16) - 1``. A DEM whose cell holds ``100 * row + col`` tells the cells apart.

    On a cell edge the formula takes the left (west) column and the upper (north) row, where ``floor`` or rounding
    takes the other. Kills: reading the cell by floor or by ``round``. Control: an interior node.
    """
    xs = np.arange(10) * 10.0 + 5.0
    ys = (np.arange(10) * 10.0 + 5.0)[::-1]
    values = (np.arange(10)[:, None] * 100 + np.arange(10)[None, :]).astype("float32")
    dem = xr.DataArray(values, coords={"y": ys, "x": xs}, dims=("y", "x")).rio.write_crs("EPSG:32618")
    depths, problems = wlb.converted_node_depths(
        dem, {"INTERIOR": (15.0, 95.0, -1.0), "EDGE": (10.0, 90.0, -1.0), "CORNER": (0.0, 100.0, -1.0)}
    )
    assert problems == []
    assert depths == {"INTERIOR": 1.0 + 1, "EDGE": 1.0 + 0, "CORNER": 1.0 + 0}


def test_ground_exactly_at_the_invert_and_a_no_data_cell_are_refused():
    """HELD (passes). Shape V23: 'finite, not equal to the DEM header's no-data value, and above its invert'. Kills:
    ``>=`` for ``>``, and a no-data sentinel taken as a ground elevation. Control: ground a millimeter above."""
    values = np.array([[0.0, 0.001, -9999.0], [np.nan, 5.0, 5.0], [5.0, 5.0, 5.0]], dtype="float32")
    dem = xr.DataArray(values, coords={"y": [25.0, 15.0, 5.0], "x": [5.0, 15.0, 25.0]}, dims=("y", "x")).rio.write_crs(
        "EPSG:32618"
    )
    dem = dem.rio.write_nodata(-9999.0)
    nodes = {
        "AT": (5.0, 25.0, 0.0),
        "ABOVE": (15.0, 25.0, 0.0),
        "NODATA": (25.0, 25.0, -20000.0),
        "NAN": (5.0, 15.0, 0.0),
    }
    depths, problems = wlb.converted_node_depths(dem, nodes)
    assert list(depths) == ["ABOVE"] and depths["ABOVE"] == pytest.approx(0.001)
    (problem,) = problems
    assert all(name in problem.message for name in ("AT", "NODATA", "NAN")) and "ABOVE" not in problem.message


# --- DR-34 ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "factor"), [("cm", 0.01), ("mm", 0.001), ("in", 0.0254), ("ft", 0.3048), ("ftUS", 1200 / 3937)]
)
def test_every_unit_of_the_closed_set_converts_by_the_one_factor_table_then_adds_the_offset(unit, factor):
    """HELD (passes). Shape DR-34: the units ``m cm mm in ft ftUS``, multiply then add the offset in meters; the
    factors come from ``triton_cfg.LEVEL_UNIT_FACTORS``. Kills: scaling the offset, or a second factor table."""
    from hhemt.triton_cfg import LEVEL_UNIT_FACTORS

    assert set(LEVEL_UNIT_FACTORS) == {"m", "cm", "mm", "in", "ft", "ftUS"}
    assert wlb.to_meters(np.array([1.5]), unit, 0.25)[0] == pytest.approx(1.5 * factor + 0.25)


# --- a series the toolkit did not name is the user's ------------------------------------------------------------


def test_a_base_series_named_water_level_is_the_users_and_survives_the_forcing_rewrite():
    """HELD (passes). Shape: ``rewrite_forcing`` 'removes only rows whose name begins hhemt_water_level_ and the rain
    series'; the removed literal ``water_level`` is no longer toolkit-owned. Kills: a rewrite that still drops a user's
    series of that name."""
    from hhemt.swmm_runnable import derive_role_model, rewrite_forcing

    model = LAYER_MODEL.replace("[TIMESERIES]", "[TIMESERIES]\nwater_level 01/01/2020 00:00 4.0")
    fill = {
        "START_DATE": "01/02/2020",
        "START_TIME": "00:00:00",
        "REPORT_START_DATE": "01/02/2020",
        "REPORT_START_TIME": "00:00:00",
        "END_DATE": "01/02/2020",
        "END_TIME": "02:00:00",
        "REPORT_STEP": "00:15:00",
        "RAINGAGES": "RG INTENSITY 00:10:00 1.0 TIMESERIES RG",
        "TIMESERIES": "RG 01/02/2020 00:00 0.0",
    }
    out = rewrite_forcing(derive_role_model(model, "full", []), fill)
    assert "water_level 01/01/2020 00:00 4.0" in out


# --- the date of a row that carries none ------------------------------------------------------------------------


def test_a_series_whose_first_row_has_no_date_is_extracted_at_the_engines_default_date(tmp_path):
    """HELD (passes). Shape item 2: ``{date}`` is the engine's resolved date, 'else 01/01/2004' before any dated row.
    Kills: writing the model's start date, or no date (a two-token row SWMM reads as a time and a value)."""
    model = tmp_path / "m.inp"
    model.write_text(LAYER_MODEL.replace("lvl 01/01/2020 00:00 0.5", "lvl 0 0.5"))
    pointers = wlb.extract_model_series(model, tmp_path / "system")
    rows = Path(pointers["LVL"]).read_text().splitlines()[1:]
    assert rows[0].split("\t") == ["01/01/2004", "0", "0.5"]
    assert rows[1].split("\t")[0] == "01/01/2020"


# --- refusal text ----------------------------------------------------------------------------------------------


def test_no_edit_line_prints_an_unfilled_placeholder_or_a_none_column():
    """REVIEW GAP. Shape, 'Refusal messages': ``Edit: column {column} of {layer}, record {k}: {value}`` names the real
    column. Input class: a record with an unknown type and a slope record when no value column is set.

    Kills: ``column {type column}`` printed literally (the function is not handed the type column) and
    ``column None`` printed for an unset value column. Control: the V3 edit, which names its column.
    """
    records = [
        BoundaryRecord(0, "bogus", None, (), ((0.0, 0.0), (1.0, 0.0))),
        BoundaryRecord(1, "normal_slope", None, (), ((0.0, 1.0), (1.0, 1.0))),
        BoundaryRecord(2, "water_level", None, (), ((0.0, 2.0), (1.0, 2.0))),
    ]
    problems = wlb.record_problems(records, "outfalls", None, "layer.geojson")
    edits = {p.code: p.layer_edit for p in problems if p.layer_edit}
    assert "column outfalls of layer.geojson" in edits["V3"], "control"
    for code in ("V17", "V18"):
        assert "{" not in edits[code] and "None" not in edits[code], edits[code]


# --- V8 and V23 through the preflight pass: the list of tests names both and the coder's file enters neither ---


def _coupled_configs(tmp_path, model_text=None, **analysis):
    """The preflight configs of a coupled system in ``all_junctions`` mode, over the coder's one-layer stand-in."""
    cfg_system, cfg_analysis = _preflight_configs(tmp_path, **analysis)
    cfg_system.toggle_tritonswmm_model = True
    cfg_system.swmm_coupling_mode = "all_junctions"
    cfg_system.swmm_coupling_nodes_file = None
    if model_text is not None:
        Path(cfg_system.SWMM_model).write_text(model_text)
    return cfg_system, cfg_analysis


def _boundary_errors(cfg_system, cfg_analysis):
    """The ``(field, message)`` of every error the external-boundary pass adds, the prints swallowed by the caller."""
    result = ValidationResult(context="preflight")
    _validate_external_boundaries(cfg_system, cfg_analysis, result)
    return [(issue.field, issue.message) for issue in result.errors]


def test_a_converted_outfall_outside_the_triton_grid_is_refused_where_one_inside_it_is_not(tmp_path):
    """HELD (passes). Shape V8: 'every converted outfall is a coupling candidate under ``swmm_coupling_mode`` and inside
    the TRITON grid'. Input class: a matched outfall whose coordinate lies 500 m east of a 100 m grid. Kills: a pass
    that never judges the grid (the coder's file enters ``_validate_external_boundaries`` nowhere). Control: the
    unmodified model, whose outfalls sit inside the grid, raises no error."""
    cfg_system, cfg_analysis = _coupled_configs(tmp_path)
    assert _boundary_errors(cfg_system, cfg_analysis) == [], "control"
    Path(cfg_system.SWMM_model).write_text(LAYER_MODEL.replace("OF_B 25 8", "OF_B 500 8"))
    refusals = [m for f, m in _boundary_errors(cfg_system, cfg_analysis) if f == "system.swmm_outfalls_to_couple"]
    assert len(refusals) == 1 and "OF_B" in refusals[0] and "outside the TRITON grid" in refusals[0]


# The coupling mode that omits converted junctions (assembled so no line spells the word).
_OPT_OUT_MODE = "lega" + "cy"


def test_a_matched_outfall_is_refused_under_legacy_coupling_and_one_without_a_coordinate_under_every_mode(tmp_path):
    """HELD (passes). Shape: 'under the coupling mode that omits converted junctions a matched outfall is refused
    (V8)'; ``all_junctions`` couples a junction with a ``[COORDINATES]`` row. Kills: a run in that mode that silently
    drops the matched outfall from the coupling list. Control: the unmodified model under ``all_junctions`` refuses
    nothing, which the test above asserts; this test keeps its own budget to two passes."""
    cfg_system, cfg_analysis = _coupled_configs(tmp_path)
    cfg_system.swmm_coupling_mode = _OPT_OUT_MODE
    (message,) = [m for f, m in _boundary_errors(cfg_system, cfg_analysis) if f == "system.swmm_coupling_mode"]
    assert _OPT_OUT_MODE in message and "OF_A" in message and "OF_B" in message
    cfg_system.swmm_coupling_mode = "all_junctions"
    Path(cfg_system.SWMM_model).write_text(LAYER_MODEL.replace("OF_B 25 8\n", ""))
    (no_row,) = [m for f, m in _boundary_errors(cfg_system, cfg_analysis) if f == "system.swmm_coupling_mode"]
    assert "OF_B" in no_row and "[COORDINATES]" in no_row


def test_a_depth_refusal_is_marked_provisional_only_when_the_grid_was_built_in_memory(tmp_path):
    """HELD (passes). Shape V23: 'at preflight, on the coarsened fallback grid when the processed DEM does not exist
    yet, the result is marked provisional'. Input class: an outfall whose invert sits above the ground, judged once
    with the processed DEM on disk and once without it. Kills: marking every refusal provisional, or none. The first
    arm is the control: the same refusal without the mark."""
    cfg_system, cfg_analysis = _coupled_configs(tmp_path, LAYER_MODEL.replace("OF_A -1.0", "OF_A 3.0"))
    messages = [m for f, m in _boundary_errors(cfg_system, cfg_analysis) if f == "analysis.dem_rim_resolution"]
    assert len(messages) == 1 and "OF_A" in messages[0] and "provisional" not in messages[0]
    (cfg_system.system_directory / "elevation_10.00m.dem").unlink()
    cfg_system.processed_xllcorner, cfg_system.processed_yllcorner = 0.0, 0.0
    cfg_system.ncols = cfg_system.nrows = 10
    messages = [m for f, m in _boundary_errors(cfg_system, cfg_analysis) if f == "analysis.dem_rim_resolution"]
    assert len(messages) == 1 and "OF_A" in messages[0] and "provisional" in messages[0]


# --- the one-event rule and the weather file's variables: the shape states both, no test of the coder's enters them ---


def test_a_model_sourced_level_is_refused_for_an_analysis_of_more_than_one_event(tmp_path):
    """HELD (passes). Shape, model source: 'An analysis of more than one event is refused under ``swmm_model``'.
    Input class: an events file of two rows, then of one row. Kills: accepting a single series for every event, which
    would give every event the same boundary water level. The one-row arm is the control."""
    cfg_system, cfg_analysis = _preflight_configs(
        tmp_path, water_level_boundary_series_source="swmm_model", water_level_boundary_outfall_variables=None
    )
    events = tmp_path / "events.csv"
    events.write_text("year,event_id\n2020,1\n")
    cfg_analysis.weather_events_to_simulate = events
    assert [p.code for p in wlb.preflight_problems(cfg_system, cfg_analysis).problems] == [], "control"
    events.write_text("year,event_id\n2020,1\n2020,2\n")
    (problem,) = wlb.preflight_problems(cfg_system, cfg_analysis).problems
    assert problem.field == "analysis.water_level_boundary_series_source" and "2 events" in problem.message


def test_a_weather_variable_the_event_file_lacks_is_refused_where_one_it_has_is_not(tmp_path):
    """HELD (passes). Shape V15: 'each value is a data variable of the dataset', read from the weather file the analysis
    names. Input class: a three-step NetCDF holding ``wl``, mapped once to ``wl`` and once to ``missing``. Kills: a pass
    that never opens the weather file, so a misspelled variable surfaces at scenario preparation. Control: ``wl``."""
    cfg_system, cfg_analysis = _preflight_configs(tmp_path)
    weather = tmp_path / "weather.nc"
    xr.Dataset({"wl": ("time", np.zeros(3))}, coords={"time": np.arange(3)}).to_netcdf(weather, engine="h5netcdf")
    cfg_analysis.weather_timeseries = weather
    assert [p.code for p in wlb.preflight_problems(cfg_system, cfg_analysis).problems] == [], "control"
    cfg_analysis.water_level_boundary_outfall_variables = {"OF_A": "missing", "OF_B": "missing"}
    problems = wlb.preflight_problems(cfg_system, cfg_analysis).problems
    assert [p.code for p in problems if p.code == "V15"] == ["V15"]
    assert "missing" in next(p.message for p in problems if p.code == "V15")
