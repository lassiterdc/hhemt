"""Scenario preparation writes the runnable-model roles with the same forcing the template path wrote.

Laptop tier, stub scenario, no solver. The forcing VALUES ``create_swmm_inp_from_template`` wrote for a
fixed stub scenario (the two-step event of ``tests/test_swmm_inp_file_path_length.py``) were recorded at base commit
``0a6b2dc0`` by running that function; they are the constants below. After the cut-over the runnable path
(``swmm_utils._swmm_fill_values`` then ``swmm_runnable.rewrite_forcing``) must write those same values into the sections
it owns. A mutation of one value in ``_swmm_fill_values`` turns the first test red.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.exceptions import ConfigurationError
from hhemt.swmm_utils import _swmm_fill_values, swmm_role_text, write_swmm_role_model
from tests.fixtures.swmm_runnable_inputs import runnable_source_text
from tests.test_swmm_inp_file_path_length import _scenario

# Recorded at base 0a6b2dc0: window, rain-gauge row, series rows and REPORT_STEP of the stub scenario.
BASE_VALUES = {
    "START_DATE": "01/01/2020",
    "START_TIME": "00:00:00",
    "REPORT_START_DATE": "01/01/2020",
    "REPORT_START_TIME": "00:00:00",
    "END_DATE": "01/01/2020",
    "END_TIME": "00:05:00",
    "REPORT_STEP": "00:01:00",
    "RAINGAGES": "rg1 INTENSITY 00:05:00 1.0 TIMESERIES rg1",
    "TIMESERIES": 'rg1 FILE "/r/rain.dat"\n\nhhemt_water_level_0 FILE "/r/level.dat"',
}


def _stub(tmp_path: Path, model: Path, outfalls=()):
    scenario = _scenario(tmp_path, rain="/r/rain.dat", storm="/r/level.dat")
    scenario._system = SimpleNamespace(
        cfg_system=SimpleNamespace(
            SWMM_model=model, swmm_outfalls_to_couple=list(outfalls), system_directory=tmp_path / "system"
        )
    )
    return scenario


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "model.inp"
    path.write_text(runnable_source_text())
    return path


def test_the_fill_values_equal_the_template_path_values_recorded_at_base(tmp_path, model):
    assert _swmm_fill_values(_stub(tmp_path, model), tmp_path / "o.inp") == BASE_VALUES


@pytest.mark.parametrize("role", ["full", "hydrology"])
def test_a_written_role_carries_the_recorded_forcing(tmp_path, model, role):
    scenario = _stub(tmp_path, model)
    dest = tmp_path / "out" / f"{role}.inp"
    write_swmm_role_model(scenario, role, dest)
    lines = dest.read_text().splitlines()
    options = {ln.split()[0]: ln.split()[1] for ln in lines if ln.startswith(("START_", "END_", "REPORT_"))}
    assert {k: options[k] for k in BASE_VALUES if k in options} == {
        k: v for k, v in BASE_VALUES.items() if k in options
    }
    assert len(options) == 7
    assert BASE_VALUES["RAINGAGES"] in lines
    assert 'rg1 FILE "/r/rain.dat"' in lines
    # The stub's model has no outfall that reads the record's series, so no role carries the level row; the level
    # file's row is the fill's, written only into a role that reads it (see the test of the matched outfall).
    assert 'hhemt_water_level_0 FILE "/r/level.dat"' not in lines


def test_the_hydraulics_role_has_no_gauges_and_no_rain_row(tmp_path, model):
    dest = tmp_path / "hydraulics.inp"
    write_swmm_role_model(_stub(tmp_path, model), "hydraulics", dest)
    text = dest.read_text()
    assert "[RAINGAGES]" not in text and "rg1" not in text


def test_a_converted_outfall_is_a_coupling_candidate_and_no_dummy_node_is(tmp_path, model):
    from hhemt.scenario import coupling_candidates

    cfg = _stub(tmp_path, model, outfalls=["dummy_outfall"])._system.cfg_system
    cfg.swmm_coupling_mode = "all_junctions"
    names = coupling_candidates(cfg)
    assert "dummy_outfall" in names
    assert not [n for n in names if n.startswith("hhemt_dummy_")]
    # The unlisted case: the outfall stays an outfall, so it is no candidate.
    cfg.swmm_outfalls_to_couple = []
    assert "dummy_outfall" not in coupling_candidates(cfg)


def test_the_node_grouper_skips_the_dummy_chain_that_has_no_coordinates(tmp_path, model):
    from hhemt.scenario import return_df_of_nodes_grouped_by_DEM_gridcell
    from tests.test_scenario_boundary_guards import _dem

    cfg = _stub(tmp_path, model, outfalls=["dummy_outfall"])._system.cfg_system
    inp = tmp_path / "hydrology.inp"
    inp.write_text(swmm_role_text(cfg, "hydrology"))
    assert "hhemt_dummy_j" in inp.read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(inp, _dem(tmp_path / "dem.tif"))
    assert not [n for n in df["node_key"] if str(n).startswith("hhemt_dummy_")]
    assert "hhemt_dummy_of" in outfalls


def test_a_listed_name_that_is_not_an_outfall_is_refused_when_the_role_is_derived(tmp_path, model):
    cfg = _stub(tmp_path, model, outfalls=["J1"])._system.cfg_system
    with pytest.raises(ConfigurationError, match="J1"):
        swmm_role_text(cfg, "hydraulics")


def test_a_mapping_lacking_the_gage_id_column_is_one_named_problem_and_the_column_present_is_none(tmp_path, model):
    from hhemt.swmm_utils import swmm_model_problems

    def cfg(mapping_csv: Path):
        return SimpleNamespace(
            SWMM_model=model,
            swmm_outfalls_to_couple=[],
            subcatchment_raingage_mapping=mapping_csv,
            subcatchment_raingage_mapping_gage_id_colname="raingage_id",
        )

    lacking = tmp_path / "lacking.csv"
    lacking.write_text("subcatchment,gauge\nS1,rg1\n")
    problems = swmm_model_problems(cfg(lacking))
    assert [p[0] for p in problems] == ["system.subcatchment_raingage_mapping_gage_id_colname"]
    assert "raingage_id" in problems[0][1] and "gauge" in problems[0][1]
    present = tmp_path / "present.csv"
    present.write_text("raingage_id\n" + "\n".join(["rg1", "RG1"]) + "\n")
    assert not [p for p in swmm_model_problems(cfg(present)) if p[0].endswith("gage_id_colname")]


def _threads_line(path: Path) -> str:
    return next(ln.split()[1] for ln in path.read_text().splitlines() if ln.startswith("THREADS"))


def test_the_thread_count_is_written_into_the_hydraulics_and_full_roles_only_and_recorded(tmp_path, model):
    """NEW CAPABILITY. Wrong implementation: write n_omp_threads, write the count into the hydrology role (inert), or
    write it without recording it (DR-36)."""
    scenario = _stub(tmp_path, model)
    scenario._analysis.cfg_analysis.swmm_threads = 3
    recorded: dict = {}
    scenario.log.swmm_threads_written = SimpleNamespace(set=lambda written: recorded.update(written))
    paths = {role: tmp_path / "out" / f"{role}.inp" for role in ("full", "hydraulics", "hydrology")}
    for role, dest in paths.items():
        write_swmm_role_model(scenario, role, dest)
    assert (_threads_line(paths["full"]), _threads_line(paths["hydraulics"])) == ("3", "3")
    assert _threads_line(paths["hydrology"]) == "1"  # the model's own line, untouched
    assert recorded == {"full": 3, "hydraulics": 3}


def test_no_role_the_toolkit_writes_holds_a_series_row_and_a_relative_file_series_is_made_absolute(tmp_path):
    """NEW CAPABILITY. Wrong implementation: copy the base model's inline rows into a role, or leave a relative FILE
    name that SWMM resolves against the scenario directory."""
    from hhemt.swmm_runnable import _node_rows, _parse

    text = runnable_source_text() + '\n[TIMESERIES]\nrel_series FILE "rel.dat"\n'
    (tmp_path / "rel.dat").write_text("01/01/2020 00:00 1.0\n")
    model = tmp_path / "base.inp"
    model.write_text(text)
    scenario = _stub(tmp_path, model)
    for role in ("full", "hydraulics", "hydrology"):
        dest = tmp_path / "o" / f"{role}.inp"
        write_swmm_role_model(scenario, role, dest)
        rows = _node_rows(_parse(dest.read_text()), "TIMESERIES")
        assert rows and all(r[1] == "FILE" and len(r) == 3 for r in rows), (role, rows)
        pointer = next(r[2] for r in rows if r[0] == "rel_series")
        assert Path(pointer.strip('"')) == tmp_path / "rel.dat"
    extracted = sorted((tmp_path / "system" / "swmm_model_series").rglob("*.dat"))
    stamps = {p: p.stat().st_mtime_ns for p in extracted}
    write_swmm_role_model(scenario, "full", tmp_path / "o" / "again.inp")
    assert {p: p.stat().st_mtime_ns for p in extracted} == stamps  # a second preparation writes no new extraction file
