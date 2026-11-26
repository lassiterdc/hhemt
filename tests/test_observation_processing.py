"""Observation-point series: reading TRITON's text files, the per-scenario variables, and the consolidated groups.

TRITON writes one global text file per printed variable under ``{out_dir}/series/`` (header line, then ``simtime`` and
one value per point). The processing step reads them into the EXISTING per-scenario gridded store of the same model as
``point_flood_depth``, ``point_unit_discharge_x`` and ``point_unit_discharge_y`` on ``(point_elapsed_s, obs_point)``,
and the analysis consolidation appends them to ``{family}/triton_observation/timeseries`` and ``/summary`` of the one
``analysis_datatree.zarr``. No zarr store other than the existing ones is created.

Every file here is hand-written and the tree is the committed multi-sim golden, unzipped once per module. Laptop tier:
no solver, no compile, no Snakemake. The series files carry, on purpose, a resume duplication (rows between the
checkpoint and the interruption appear twice) and periodic rows at an off-nominal ``simtime``, which TRITON records as
the actual step time of the first step at or after each period.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.exceptions import ProcessingError
from hhemt.process_simulation import (
    OBSERVATION_SERIES_FILES,
    TRITONSWMM_sim_post_processing,
    build_observation_dataset,
    nominal_observation_times,
    read_observation_series_file,
)
from hhemt.triton_cfg import ObservationPoint
from tests.fixtures.golden_trees import construct_golden_analysis, extract_golden

#: The golden's DEM is 6 rows by 4 columns of 10 m cells with its lower-left corner at the origin, so point A falls on
#: cell (row 0, col 0) and point B on cell (row 5, col 3).
POINT_CSV = "id,x,y\nA,5.0,55.0\nB,35.0,5.0\n"
INTERVAL_S = 50.0
DURATION_S = 120.0  # the golden's simulated duration: 12 reporting steps of 10 s
STORE_ATTR = {"tritonswmm": "output_tritonswmm_triton_timeseries", "triton": "output_triton_only_timeseries"}
OUT_DIR_ATTR = {"tritonswmm": "out_tritonswmm", "triton": "out_triton"}
FAMILY = {"tritonswmm": "tritonswmm", "triton": "triton_only"}
#: The two coupled and TRITON-only arms get different values so a swapped read of one for the other shows.
MODEL_FACTOR = {"tritonswmm": 1.0, "triton": 3.0}
EVENTS = (0, 1)
OBSERVATION_VARIABLES = (
    "point_flood_depth",
    "point_unit_discharge_x",
    "point_unit_discharge_y",
    "max_point_flood_depth",
    "time_of_max_point_flood_depth",
)


def _series_text(prefix: str, rows: list[tuple[float, float, float]], *, n_points: int = 2) -> str:
    header = "Time(s)," + ",".join(f"{prefix}_at_Point_{i}" for i in range(1, n_points + 1))
    return header + "\n" + "".join(f"{t},{a},{b}\n" for t, a, b in rows)


def _rows(scale: float) -> list[tuple[float, float, float]]:
    """The rows of the depth file of one event: initial, two off-nominal periodic rows, then a resume, then the end.

    The first two periodic rows are the pre-interruption copies (value +9) and are rewritten identically in time by the
    resumed run (the true values), so the LAST occurrence of each ``simtime`` is the one that must survive.
    """
    return [
        (0.0, 0.0, 0.0),
        (50.4, 0.5 * scale + 9, 0.2 * scale + 9),
        (100.7, 1.5 * scale + 9, 0.1 * scale + 9),
        (50.4, 0.5 * scale, 0.2 * scale),
        (100.7, 1.5 * scale, 0.1 * scale),
        (120.0, 0.4 * scale, 0.0),
    ]


def _write_series_files(out_dir: Path, scale: float) -> None:
    series = out_dir / "series"
    series.mkdir(parents=True, exist_ok=True)
    for _variable, (file_name, prefix) in OBSERVATION_SERIES_FILES.items():
        factor = {"H": 1.0, "QX": 0.1, "QY": -0.1}[prefix]
        rows = [(t, a * factor, b * factor) for t, a, b in _rows(scale)]
        (series / file_name).write_text(_series_text(prefix, rows))


def _extract_with_points(tmp_path_factory, name: str, *, analysis_extra: str = ""):
    """The multi-sim golden with the two-point file in its system config and the 50 s interval in its analysis one."""
    root = extract_golden("multisim", tmp_path_factory.mktemp(name) / "golden")
    points = root / "inputs" / "observation_points.csv"
    points.write_text(POINT_CSV)
    system_cfg = root / "system_config.yaml"
    system_cfg.write_text(system_cfg.read_text() + f'observation_points: "{points}"\n')
    analysis_cfg = root / "analysis_config.yaml"
    analysis_cfg.write_text(
        analysis_cfg.read_text() + f"TRITON_observation_interval_s: {INTERVAL_S}\n" + analysis_extra
    )
    return root


def _tiny_tree(root: Path):
    """The golden's analysis cut to two events, with series files written for both events and both arms."""
    from hhemt.scenario import TRITONSWMM_scenario

    analysis = construct_golden_analysis(root)
    analysis.all_sims_for_test = analysis.df_sims
    analysis.df_sims = analysis.df_sims.iloc[:2]
    analysis.nsims = len(analysis.df_sims)
    for event_iloc in EVENTS:
        scen = TRITONSWMM_scenario(event_iloc, analysis)
        for model in ("tritonswmm", "triton"):
            _write_series_files(
                getattr(scen.scen_paths, OUT_DIR_ATTR[model]), scale=(event_iloc + 1) * MODEL_FACTOR[model]
            )
    return analysis


@pytest.fixture(scope="module")
def _identity_patch():
    """The analysis constructor refuses a dirty checkout; a tree under edit is dirty, and this is not about that."""
    import hhemt.analysis as analysis_module

    patch = pytest.MonkeyPatch()
    real = analysis_module.running_identity
    patch.setattr(analysis_module, "running_identity", lambda: dataclasses.replace(real(), dirty=False))
    yield
    patch.undo()


@pytest.fixture(scope="module")
def processed(tmp_path_factory, _identity_patch):
    """The shared tiny tree after the processing step has written the series into every scenario's gridded store.

    Built once per module (the developer's G14 ruling), so every per-event call of the writer, with its first-use
    imports and its zarr writes, is paid here and not by the test that reads the result.
    """
    root = _extract_with_points(
        tmp_path_factory, "observation", analysis_extra="toggle_consolidate_timeseries: false\n"
    )
    analysis = _tiny_tree(root)
    before = _zarr_paths(analysis)
    for event_iloc in EVENTS:
        for model in ("tritonswmm", "triton"):
            _process(analysis, event_iloc, model)
    return SimpleNamespace(analysis=analysis, zarr_before=before)


def _open_groups(analysis) -> dict[str, xr.Dataset]:
    """The four observation groups of the tree, loaded, keyed ``{family}/{branch}``."""
    groups = {}
    for family in FAMILY.values():
        for branch in ("timeseries", "summary"):
            groups[f"{family}/{branch}"] = _group(analysis, f"{family}/triton_observation/{branch}").load()
    return groups


@pytest.fixture(scope="module")
def consolidated(processed):
    """``consolidate_to_datatree`` over the tiny tree with ``toggle_consolidate_timeseries`` OFF in its config."""
    processed.analysis.process.consolidate_to_datatree()
    processed.groups = _open_groups(processed.analysis)
    return processed


def _process(analysis, event_iloc: int, model: str) -> Path:
    from hhemt.scenario import TRITONSWMM_scenario

    scen = TRITONSWMM_scenario(event_iloc, analysis)
    store = getattr(scen.scen_paths, STORE_ATTR[model])
    TRITONSWMM_sim_post_processing(scen.run, model_log=scen.get_log(model))._export_observation_series(model, store)
    return Path(store)


def _zarr_paths(analysis) -> set[str]:
    """Every zarr store directory under the analysis directory, found without descending into a store."""
    root = Path(analysis.analysis_paths.analysis_dir)
    found: set[str] = set()
    for dirpath, dirnames, __ in os.walk(root):
        for name in [d for d in dirnames if d.endswith(".zarr")]:
            found.add(str((Path(dirpath) / name).relative_to(root)))
            dirnames.remove(name)
    return found


def _group(analysis, path: str) -> xr.Dataset:
    return xr.open_zarr(analysis.analysis_paths.analysis_datatree_zarr, group=path, consolidated=True)


# ---------------------------------------------------------------------------------------------------------------
# Reading the text files (no tree)
# ---------------------------------------------------------------------------------------------------------------


def test_the_header_must_name_the_scenarios_point_count(tmp_path):
    """A file written for another point set is refused naming the file and both counts, never read as fewer points."""
    path = tmp_path / "H_at_Xsec.txt"
    path.write_text(_series_text("H", [(0.0, 0.0, 0.0)], n_points=2))
    with pytest.raises(ProcessingError, match=r"H_at_Xsec.txt.*2 point column.*lists 3"):
        read_observation_series_file(path, "H", 3)


def test_a_file_with_a_header_and_no_data_row_is_refused(tmp_path):
    path = tmp_path / "H_at_Xsec.txt"
    path.write_text(_series_text("H", []))
    simtimes, values = read_observation_series_file(path, "H", 2)
    assert simtimes.size == 0 and values.shape == (0, 2)
    with pytest.raises(ProcessingError, match="header and no data row"):
        nominal_observation_times(simtimes, INTERVAL_S, DURATION_S, "H_at_Xsec.txt")


def test_resume_duplicates_keep_the_last_row_and_periodic_rows_take_their_nominal_time():
    """Rows at 50.4 and 100.7 repeat after a resume; the last of each stays, and the periodic rows read 50 and 100."""
    simtimes = np.array([r[0] for r in _rows(1.0)])
    kept, nominal = nominal_observation_times(simtimes, INTERVAL_S, DURATION_S, "H")
    assert nominal.tolist() == [0.0, 50.0, 100.0, 120.0]
    assert kept.tolist() == [0, 3, 4, 5]


def test_a_missing_period_is_refused_with_the_counts_named():
    simtimes = np.array([0.0, 100.7, 120.0])  # the first period's row is missing
    with pytest.raises(ProcessingError, match=r"holds 3 distinct row.*holds 4"):
        nominal_observation_times(simtimes, INTERVAL_S, DURATION_S, "H")


def test_a_final_write_that_falls_on_the_last_period_is_one_row():
    simtimes = np.array([0.0, 40.0, 80.0, 120.0])
    __, nominal = nominal_observation_times(simtimes, 40.0, 120.0, "H")
    assert nominal.tolist() == [0.0, 40.0, 80.0, 120.0]


def test_the_dataset_carries_the_point_identity_in_file_order(tmp_path):
    for __, (file_name, prefix) in OBSERVATION_SERIES_FILES.items():
        (tmp_path / file_name).write_text(_series_text(prefix, _rows(1.0)))
    points = [ObservationPoint("A", 5.0, 55.0), ObservationPoint("B", 35.0, 5.0)]
    ds = build_observation_dataset(tmp_path, points, [(0, 0), (5, 3)], interval_s=INTERVAL_S, duration_s=DURATION_S)
    assert ds["point_flood_depth"].dims == ("point_elapsed_s", "obs_point")
    assert ds["obs_point_id"].values.tolist() == ["A", "B"]
    assert ds["obs_row"].values.tolist() == [0, 5] and ds["obs_col"].values.tolist() == [0, 3]
    assert ds["point_flood_depth"].isel(obs_point=0).values.tolist() == [0.0, 0.5, 1.5, 0.4]


# ---------------------------------------------------------------------------------------------------------------
# The per-scenario store
# ---------------------------------------------------------------------------------------------------------------


def test_the_variables_land_in_the_existing_gridded_store_with_nominal_times(processed):
    from hhemt.scenario import TRITONSWMM_scenario

    store = getattr(TRITONSWMM_scenario(0, processed.analysis).scen_paths, STORE_ATTR["tritonswmm"])
    ds = xr.open_zarr(store, consolidated=True)
    assert ds["point_flood_depth"].dims == ("point_elapsed_s", "obs_point")
    assert ds["point_elapsed_s"].values.tolist() == [0.0, 50.0, 100.0, 120.0]
    assert ds["point_flood_depth"].isel(obs_point=0).values.tolist() == [0.0, 0.5, 1.5, 0.4]
    assert ds["point_unit_discharge_y"].isel(obs_point=0).values.tolist() == pytest.approx([0.0, -0.05, -0.15, -0.04])
    assert ds["obs_point_id"].values.tolist() == ["A", "B"]
    assert ds["obs_row"].values.tolist() == [0, 5] and ds["obs_col"].values.tolist() == [0, 3]
    assert "timestep" in ds["flood_depth"].dims, "the gridded variables are untouched"
    assert ds["point_flood_depth"].attrs["units"] == "m" and ds["point_elapsed_s"].attrs["units"] == "s"


def test_the_series_went_into_the_existing_stores_and_no_zarr_store_was_created(processed):
    assert _zarr_paths(processed.analysis) == processed.zarr_before


def test_the_prune_of_the_gridded_keep_list_does_not_reach_the_point_variables(processed):
    """The keep-list prune acts on variables that carry ``timestep``; the point variables carry ``point_elapsed_s``."""
    from hhemt.scenario import TRITONSWMM_scenario
    from hhemt.timeseries_selection import prune_timeseries_store

    store = getattr(TRITONSWMM_scenario(1, processed.analysis).scen_paths, STORE_ATTR["triton"])
    removed = prune_timeseries_store(Path(store), "triton_gridded", "triton_only", ())
    assert "flood_depth" in removed and not any(v.startswith("point_") for v in removed)
    assert "point_flood_depth" in xr.open_zarr(store, consolidated=True).data_vars


def test_processing_twice_leaves_the_store_unchanged(processed):
    """Arm (a): a second pass over one scenario changes no size and creates no store."""
    store = _process(processed.analysis, 0, "tritonswmm")
    first = dict(xr.open_zarr(store, consolidated=True).sizes)
    before = _zarr_paths(processed.analysis)
    _process(processed.analysis, 0, "tritonswmm")
    assert dict(xr.open_zarr(store, consolidated=True).sizes) == first
    assert _zarr_paths(processed.analysis) == before


def test_a_store_without_the_series_and_without_the_text_files_is_refused(processed, tmp_path):
    """A scenario run with no points set carries neither; the system config now names points, so it is refused."""
    import shutil

    from hhemt.scenario import TRITONSWMM_scenario

    scen = TRITONSWMM_scenario(0, processed.analysis)
    series = Path(scen.scen_paths.out_triton) / "series"
    moved = tmp_path / "series_moved"
    shutil.move(series, moved)
    store = Path(getattr(scen.scen_paths, STORE_ATTR["triton"]))
    kept = tmp_path / "store_moved"
    shutil.move(store, kept)
    # A twin store with no observation variables: the golden's gridded variables only.
    xr.open_zarr(kept, consolidated=True)[["flood_depth"]].load().to_zarr(store, mode="w")
    try:
        with pytest.raises(ProcessingError, match="did not print observations"):
            _process(processed.analysis, 0, "triton")
    finally:
        shutil.rmtree(store)
        shutil.move(kept, store)
        shutil.move(moved, series)


# ---------------------------------------------------------------------------------------------------------------
# The consolidated groups
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["tritonswmm", "triton"])
def test_consolidation_writes_the_two_groups_with_the_dims_and_coords(consolidated, model):
    """With the toggle OFF the groups are still written: the ruling is that the points end up in the analysis zarr."""
    family = FAMILY[model]
    series = consolidated.groups[f"{family}/timeseries"]
    assert series["point_flood_depth"].dims == ("event_iloc", "elapsed_s", "obs_point")
    assert series["elapsed_s"].values.tolist() == [0, 50, 100, 120]
    assert series["event_iloc"].values.tolist() == [0, 1]
    assert series["obs_point_id"].values.tolist() == ["A", "B"]
    assert series["obs_x"].values.tolist() == [5.0, 35.0] and series["obs_row"].values.tolist() == [0, 5]
    # event 1 is the second event, scale 2: the arm's factor and the event's scale both reach the tree.
    expect = 2 * 1.5 * MODEL_FACTOR[model]
    assert float(series["point_flood_depth"].sel(event_iloc=1, elapsed_s=100).isel(obs_point=0)) == pytest.approx(
        expect
    )
    summary = consolidated.groups[f"{family}/summary"]
    assert summary["max_point_flood_depth"].dims == ("event_iloc", "obs_point")
    assert summary["max_point_flood_depth"].isel(event_iloc=1).values.tolist() == pytest.approx(
        [expect, 0.2 * 2 * MODEL_FACTOR[model]]
    )
    assert summary["time_of_max_point_flood_depth"].isel(event_iloc=1).values.tolist() == [100.0, 50.0]


def test_consolidation_stamps_the_cf_attributes_and_creates_no_store(consolidated):
    series = consolidated.groups["tritonswmm/timeseries"]
    assert series["point_flood_depth"].attrs["units"] == "m"
    assert series["point_unit_discharge_x"].attrs["units"] == "m2 s-1"
    assert series["elapsed_s"].attrs["units"] == "s" and series["obs_x"].attrs["units"] == "m"
    summary = consolidated.groups["tritonswmm/summary"]
    assert summary["max_point_flood_depth"].attrs["cell_methods"] == "time: maximum"
    assert _zarr_paths(consolidated.analysis) == consolidated.zarr_before


def test_the_consolidation_version_is_six():
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing

    assert TRITONSWMM_analysis_post_processing.CONSOLIDATION_VERSION == 6


def test_the_zarr_path_set_check_fires_when_a_store_is_added(processed):
    """Control for the no-extra-store checks: the matcher must see a zarr directory that is not in the baseline."""
    stray = Path(processed.analysis.analysis_paths.analysis_dir) / "stray_observation.zarr"
    stray.mkdir()
    try:
        assert _zarr_paths(processed.analysis) - processed.zarr_before == {"stray_observation.zarr"}
    finally:
        stray.rmdir()


def test_a_scenario_with_no_series_variables_is_refused_by_name_at_consolidation(processed):
    """Event 2 of the golden was never processed with points, so its gridded store carries no ``point_*`` variable."""
    from hhemt.scenario import TRITONSWMM_scenario

    analysis = processed.analysis
    analysis.df_sims = analysis.all_sims_for_test  # the golden's third event, which the tiny tree leaves out
    try:
        scen = TRITONSWMM_scenario(2, analysis)
        start, __ = scen._resolve_event_window()
        with pytest.raises(ProcessingError, match=r"event_index\.2.*has no \[.*point_flood_depth"):
            analysis.process._observation_timeseries_slice(
                scen, "tritonswmm_observation_ts", 2, start, INTERVAL_S, np.array([0, 50, 100, 120])
            )
    finally:
        analysis.df_sims = analysis.all_sims_for_test.iloc[:2]


def test_a_row_time_off_the_groups_axis_is_refused_by_name(processed):
    from hhemt.scenario import TRITONSWMM_scenario

    scen = TRITONSWMM_scenario(1, processed.analysis)
    start, __ = scen._resolve_event_window()
    with pytest.raises(ProcessingError, match=r"event_index\.1.*\[100.0, 120.0\] s.*not on the group's axis"):
        processed.analysis.process._observation_timeseries_slice(
            scen, "tritonswmm_observation_ts", 1, start, INTERVAL_S, np.array([0, 50])
        )


def test_clear_raw_reclaims_the_series_text_directory(consolidated):
    """``series/`` is in the allowlist of ``_clear_raw_outputs``, which removes it with ``H/`` and keeps the rest."""
    from hhemt.scenario import TRITONSWMM_scenario

    scen = TRITONSWMM_scenario(0, consolidated.analysis)
    out_dir = Path(scen.scen_paths.out_tritonswmm)
    assert (out_dir / "series").is_dir() and (out_dir / "performance.txt").is_file()
    proc = TRITONSWMM_sim_post_processing(scen.run, model_log=scen.get_log("tritonswmm"))
    proc._clear_raw_outputs("tritonswmm")
    assert not (out_dir / "series").exists() and not (out_dir / "bin").exists()
    assert (out_dir / "performance.txt").is_file(), "the completion predicate file survives the reclaim"


def test_the_ro_crate_names_the_observation_variables(consolidated):
    """The crate describes the five observation variables, and the matcher is shown to find a known variable.

    Class: NEW CAPABILITY. The crate advertises only the variables of the in-memory tree, and the observation groups
    are appended after it is assembled, so a crate built from the tree alone names none of them (measured: 0 matches
    before the stand-in dataset was added). Second correct implementation: building the crate from the stored groups.
    """
    import json

    crate = Path(consolidated.analysis.analysis_paths.analysis_dir) / "ro-crate-metadata.json"
    text = crate.read_text()
    json.loads(text)
    assert text.count("max_flood_depth") > 0, "control: the matcher finds a variable the crate has always named"
    for name in OBSERVATION_VARIABLES:
        assert f'"{name}"' in text, f"{name} is not described by the crate"


def test_an_uninterrupted_file_keeps_its_period_labels():
    """Property: with no restart the labels are 0, k * interval and the final row's actual time, as before.

    Class: REGRESSION (the labels the first version of the reader gave). Reachability: a file whose simtime never goes
    backwards. Killed by: a reader that labels from the first row's simtime. Second correct implementation: labelling
    by period index ``round(simtime / interval)``.
    """
    simtimes = np.array([0.0, 50.1, 100.2, 150.3, 200.4, 210.0])
    kept, nominal = nominal_observation_times(simtimes, 50.0, 210.0, "H")
    assert kept.tolist() == [0, 1, 2, 3, 4, 5]
    assert nominal.tolist() == [0.0, 50.0, 100.0, 150.0, 200.0, 210.0]


def test_a_resume_off_the_observation_grid_labels_the_rows_from_the_restart_time():
    """Property: a resumed file whose checkpoint is not one of its observation rows labels the restart row with its
    actual time and the rows after it ``restart + j * interval`` (print interval 30 s, observation interval 50 s).

    Class: NEW CAPABILITY. The file: rows at 0, 50, 100 and 150 before a kill, then a resume from the checkpoint at
    100.2 s that writes 100.2, 150.2, 200.2 and the end at 210. The superseded row is the one at 150 only; the row at
    100 is before the restart and stays. Killed by: labelling the restart row 100 (on the grid) or the later rows
    150, 200. Second correct implementation: computing labels from the restart row's position in the file.
    """
    simtimes = np.array([0.0, 50.0, 100.0, 150.0, 100.2, 150.2, 200.2, 210.0])
    kept, nominal = nominal_observation_times(simtimes, 50.0, 210.0, "H")
    assert kept.tolist() == [0, 1, 2, 4, 5, 6, 7]
    assert nominal.tolist() == pytest.approx([0.0, 50.0, 100.0, 100.2, 150.2, 200.2, 210.0])


def test_the_axis_holds_the_resume_labels_and_the_uninterrupted_scenario_keeps_the_shared_grid():
    """Property: one axis serves a resumed and an uninterrupted scenario, and the uninterrupted one lands on its grid.

    Class: NEW CAPABILITY. Plausible wrong implementation: an axis built from the grid alone, on which the resumed
    scenario's 100.2 and 150.2 have no place. Second correct implementation: an axis built from every scenario's labels.
    """
    from hhemt.processing_analysis import observation_axis_values

    plain = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 210.0])
    resumed = np.array([0.0, 50.0, 100.0, 100.2, 150.2, 200.2, 210.0])
    grid_only = observation_axis_values(50.0, [210.0, 210.0], [plain])
    axis = observation_axis_values(50.0, [210.0, 210.0], [plain, resumed])
    assert grid_only.tolist() == plain.tolist(), "an uninterrupted-only analysis keeps exactly the shared grid"
    for label in resumed:
        assert np.isclose(axis, label, rtol=0, atol=1e-6).any(), f"{label} has no place on the axis"
    assert set(plain.tolist()) <= set(axis.tolist()), "the uninterrupted scenario's labels are all on the shared axis"
