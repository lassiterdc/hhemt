"""Synthetic-model single-sim tier. Mirror of test_PC_01 using synth fixtures.

The fixture ``synth_all_models_analysis_cached`` carries the TRITON inputs of hhemt 1.2 (see ``tests/conftest.py``): a
Courant number of 0.45, the Kokkos configuration print, 2 observation points read every 600 s, and an initial-depth
raster of zeros. The tests below follow one pipeline (prepare, run, process, consolidate) and each stage asserts the
inputs where that stage can see them, so the inputs cost no solver run of their own. A failure in a stage that touches
the inputs is therefore not always a failure of the stage's own subject.
"""

import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import tests.conftest as suite_conftest
import tests.utils_for_testing as tst_ut

pytestmark = [
    # Every test here drives a real single-sim pipeline against the compiled
    # TRITON binaries. `tritonswmm_cpu_compiled` skips when cmake+mpic++ are
    # absent and HARD-FAILS under HHEMT_REQUIRE_COMPILE_TIER=1.
    pytest.mark.usefixtures("tritonswmm_cpu_compiled"),
]

# Per model: the family group of the consolidated tree, the per-scenario gridded series store, the model output
# folder TRITON writes under, and the cfg file the toolkit renders. SWMM-only has no TRITON input and no row here.
_TRITON_ARMS = {
    "tritonswmm": {
        "family": "tritonswmm",
        "store": "output_tritonswmm_triton_timeseries",
        "out": "out_tritonswmm",
        "cfg": "triton_swmm_cfg",
    },
    "triton": {
        "family": "triton_only",
        "store": "output_triton_only_timeseries",
        "out": "out_triton",
        "cfg": "triton_cfg",
    },
}


def _triton_arms(analysis):
    """The ``(model_type, arm)`` pairs of the TRITON models this analysis enables."""
    enabled = tst_ut.get_enabled_model_types(analysis)
    return [(model_type, arm) for model_type, arm in _TRITON_ARMS.items() if model_type in enabled]


def _cfg_values(path) -> dict[str, str]:
    """The active ``key=value`` lines of a TRITON cfg, comment lines dropped and one pair of quotes removed."""
    values = {}
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, __, value = text.partition("=")
        values[key.strip()] = value.strip().strip('"')
    return values


def _zarr_store_paths(analysis) -> set[str]:
    """Every zarr store directory under the analysis directory, found without descending into a store."""
    root = Path(analysis.analysis_paths.analysis_dir)
    found: set[str] = set()
    for dirpath, dirnames, __ in os.walk(root):
        for name in [d for d in dirnames if d.endswith(".zarr")]:
            found.add(str((Path(dirpath) / name).relative_to(root)))
            dirnames.remove(name)
    return found


def _remove_entries_added_since(analysis_dir: Path, names_before: set[str]) -> None:
    """Delete the top-level entries of ``analysis_dir`` that ``names_before`` does not list.

    The analysis directory is shared by every test that takes ``synth_all_models_analysis_cached``. A consolidated
    tree left in it with no layout record makes the version-state tests refuse it, since a tree that holds results
    and carries no record reads as one of unknown version; a test that consolidates removes what it added.
    """
    for entry in Path(analysis_dir).iterdir():
        if entry.name in names_before:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _open_store(path):
    """A per-scenario series store, zarr or netcdf, opened without relying on a consolidated index."""
    path = Path(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path, consolidated=False)
    return xr.open_dataset(path, engine="h5netcdf")


def _assert_folded_inputs_rendered(analysis) -> None:
    """Each TRITON cfg the preparation wrote carries the folded non-default inputs, and the points file is in order.

    Class: NEW CAPABILITY. Property: the cfg of every enabled TRITON model holds ``courant`` at the folded value,
    ``time_series_flag=1``, ``print_observation`` at the folded interval, an ``observation_loc_file`` that holds the
    CSV's ``x,y`` rows in file order, and an ``h_infile`` that names the zero raster. Why required: the arrival of a
    value in the cfg is the first of the two places a setting can be lost, and the second, the solver's read, is
    asserted after the run. Killed by: a renderer that drops the setting, a points file written in another order, or
    an overlay that never reached the scenario. A second correct implementation compares whole cfg text to a golden.
    Not established: that TRITON parses these lines; the cfg dump asserted after the run does that.
    """
    expected_rows = [f"{x:g},{y:g}" for __, x, y in suite_conftest.FOLDED_OBSERVATION_POINTS]
    arms = _triton_arms(analysis)
    checked = 0
    for event_iloc in analysis.df_sims.index:
        paths = analysis._retrieve_sim_runs(event_iloc)._scenario.scen_paths
        for model_type, arm in arms:
            cfg_path = Path(getattr(paths, arm["cfg"]))
            values = _cfg_values(cfg_path)
            where = f"{model_type} cfg {cfg_path}"
            assert float(values["courant"]) == suite_conftest.FOLDED_COURANT_NUMBER, where
            assert values["time_series_flag"] == "1", where
            assert float(values["print_observation"]) == suite_conftest.FOLDED_OBSERVATION_INTERVAL_S, where
            points_file = Path(values["observation_loc_file"])
            assert points_file == cfg_path.parent / "observation_points.txt", where
            assert points_file.read_text().split() == expected_rows, f"{where}: the points file rows"
            assert Path(values["h_infile"]).is_file(), f"{where}: h_infile names no file"
            checked += 1
    # Must-fire control: at least one cfg was read, and one for every event and enabled TRITON model.
    assert checked == len(analysis.df_sims) * len(arms) > 0


def _assert_solver_read_the_folded_inputs(analysis) -> None:
    """TRITON's own cfg dump carries the folded inputs, and the run log carries the Kokkos configuration print.

    Class: NEW CAPABILITY. Property: every enabled TRITON model's simulation completed, as ``model_run_completed``
    reads it (the ``Simulation ends`` marker TRITON prints at the end of a run, which a run that exits at the raster
    read never prints); and when the solver ran in this entry, it wrote at least one ``cfg/config_*.cfg``
    under each enabled TRITON model's output folder, every one holds ``courant`` at the folded value and the
    observation and initial-condition keys, and the model's run log holds ``Kokkos Version``, which only
    ``KOKKOS_PRINT_CONFIGURATION=1`` prints. Those facts are then written to a sidecar under the system directory's
    ``folded_inputs/``, which ``clear_raw`` does not reach. When the solver did not run in this entry (the tree is
    warm: ``test_process_sim`` cleared ``cfg/`` and the runner skips a completed model), the same facts are read from
    the sidecar written by the entry that did run it. Why required: a setting that reaches the cfg and not the
    process (an environment variable lost between the launcher and the solver, a cfg key the solver ignores) is
    invisible to the cfg assertion, and a stage that reds on every second entry reports a solver fault that is not
    there. Killed by: an environment export dropped from the launch, a rendered key that TRITON does not read, a
    re-entry that finds no evidence, an initial-condition raster that TRITON cannot read. A second correct
    implementation forces the solver to run in every entry. Not established: the text of the Kokkos print on
    another Kokkos version; a warm tree with no sidecar (one built before this evidence existed) fails here by
    design, because nothing then shows the solver read the inputs. The completion check is the proof that the run
    got past its raster reads, and the cfg dump is not: TRITON writes its dump from ``write_output``, after those
    reads, but a dump found under ``cfg/`` can come from an earlier run of the same tree.
    """
    import json

    from hhemt.run_simulation import model_logfile_for

    sidecar_dir = Path(analysis._system.cfg_system.system_directory) / suite_conftest._FOLDED_INPUTS_DIR
    arms = _triton_arms(analysis)
    checked = 0
    for event_iloc in analysis.df_sims.index:
        run = analysis._retrieve_sim_runs(event_iloc)
        paths = run._scenario.scen_paths
        for model_type, arm in arms:
            assert run.model_run_completed(model_type), (
                f"{model_type} event {event_iloc}: the simulation did not complete, so TRITON did not get past "
                "reading its inputs"
            )
            sidecar = sidecar_dir / f"solver_read_{event_iloc}_{model_type}.json"
            dump_dir = Path(getattr(paths, arm["out"])) / "cfg"
            dumps = sorted(dump_dir.glob("config_*.cfg"))
            if dumps:
                facts = {"dumps": [], "kokkos_print": False}
                for dump in dumps:
                    values = _cfg_values(dump)
                    facts["dumps"].append(
                        {
                            "name": dump.name,
                            "courant": float(values["courant"]),
                            "time_series_flag": values["time_series_flag"],
                            "print_observation": float(values["print_observation"]),
                            "h_infile": "h_infile" in values,
                        }
                    )
                log = model_logfile_for(analysis, event_iloc, model_type)
                facts["kokkos_print"] = "Kokkos Version" in log.read_text(errors="replace")
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(json.dumps(facts))
            else:
                assert sidecar.is_file(), (
                    f"{model_type}: no cfg dump under {dump_dir} and no sidecar {sidecar}; the solver never ran "
                    "with the folded inputs on this tree, so nothing shows it read them"
                )
                facts = json.loads(sidecar.read_text())
            assert facts["dumps"], f"{model_type}: the evidence holds no cfg dump"
            for dump in facts["dumps"]:
                where = f"{model_type} {dump['name']}"
                assert dump["courant"] == suite_conftest.FOLDED_COURANT_NUMBER, where
                assert dump["time_series_flag"] == "1", where
                assert dump["print_observation"] == suite_conftest.FOLDED_OBSERVATION_INTERVAL_S, where
                assert dump["h_infile"], where
            assert facts["kokkos_print"], f"{model_type}: the run log held no Kokkos configuration print"
            checked += 1
    assert checked == len(analysis.df_sims) * len(arms) > 0


def _assert_observation_series_processed(analysis) -> None:
    """Each scenario's gridded store holds the 3 point variables, labelled ``k * interval``.

    Class: NEW CAPABILITY. Property: for every enabled TRITON model the store carries ``point_flood_depth``,
    ``point_unit_discharge_x`` and ``point_unit_discharge_y`` on the two named dimensions with the folded points in file
    order, their ids and their DEM cells; the labels start at 0 and an uninterrupted run keeps ``k * interval`` for the
    first ``floor(duration / interval) + 1`` rows; the values are finite and the depth is not negative; and the raw
    ``series/`` text was reclaimed after the store took the rows. Why required: this is the only end-to-end proof that
    TRITON wrote the files the reader expects, with the header the reader checks and the row count it demands. Killed
    by: a reader that mislabels periods, a run that wrote no series, a clear that ran before the export. A second
    correct implementation checks the tree only. Not established: that the depths are plausible, since a point may
    stay dry.
    """
    interval = suite_conftest.FOLDED_OBSERVATION_INTERVAL_S
    ids = [pid for pid, __, __ in suite_conftest.FOLDED_OBSERVATION_POINTS]
    cells = suite_conftest.FOLDED_OBSERVATION_CELLS
    arms = _triton_arms(analysis)
    checked = 0
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        start, end = proc._scenario._resolve_event_window()
        duration_s = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
        periods = math.floor(duration_s / interval + 1e-9)
        assert periods >= 2, f"the simulated {duration_s} s holds only {periods} period(s) of {interval} s"
        for model_type, arm in arms:
            store = Path(getattr(proc.scen_paths, arm["store"]))
            where = f"{model_type} store {store}"
            with _open_store(store) as ds:
                for name in ("point_flood_depth", "point_unit_discharge_x", "point_unit_discharge_y"):
                    assert ds[name].dims == ("point_elapsed_s", "obs_point"), f"{where}: {name} dims"
                    assert np.isfinite(ds[name].values).all(), f"{where}: {name} holds a non-finite value"
                assert (ds["point_flood_depth"].values >= 0).all(), f"{where}: a negative depth"
                labels = ds["point_elapsed_s"].values
                assert labels[0] == 0.0, where
                assert np.array_equal(labels[: periods + 1], np.arange(periods + 1) * interval), f"{where}: {labels}"
                assert ds.sizes["point_elapsed_s"] in (periods + 1, periods + 2), f"{where}: {labels}"
                assert ds["obs_point_id"].values.tolist() == ids, where
                assert list(zip(ds["obs_row"].values.tolist(), ds["obs_col"].values.tolist(), strict=True)) == list(
                    cells
                ), f"{where}: the DEM cells"
            series_dir = Path(getattr(proc.scen_paths, arm["out"])) / "series"
            assert not series_dir.exists(), f"{where}: the raw text {series_dir} was not reclaimed"
            checked += 1
    assert checked == len(analysis.df_sims) * len(arms) > 0


def _assert_matched_outfall_coupled_through_one_water_level_row(analysis) -> None:
    """The coupling list holds the synthetic matched outfall, and the external-boundary file carries one type-1 row.

    Class: NEW CAPABILITY (package 1h, DR-25). Property: for every event the TRITON coupling-node file written beside
    ``hydraulics.inp`` names the synthetic matched outfall, which the system converted into a junction because the
    boundary layer's water-level record names it; the location file ``extbc/loc.extbc`` holds one data row, of type 1
    (a water level that varies with time), and that row names a series file that exists. Why required: this is the only
    end-to-end proof that a record-converted outfall reaches the solver as a coupled node and that the layer's record
    reaches the solver as one boundary row, and the laptop tests read text only. Killed by: a conversion that leaves
    the outfall out of the list, a record written with another type code, a layer that writes no row or two rows.
    A second correct implementation reads the cfg dump the solver wrote. Not established: that the solver exchanged
    water at the node; the run's completion and the processed outputs show the run got through its inputs.
    """
    from hhemt.scenario import coupling_nodes_file_for, read_coupling_node_file
    from hhemt.synthetic_model.swmm_model import MATCHED_OUTFALL

    checked = 0
    for event_iloc in analysis.df_sims.index:
        paths = analysis._retrieve_sim_runs(event_iloc)._scenario.scen_paths
        nodes = read_coupling_node_file(coupling_nodes_file_for(paths.swmm_hydraulics_inp))
        assert MATCHED_OUTFALL in nodes, f"event {event_iloc}: the coupling list {nodes} omits {MATCHED_OUTFALL}"
        rows = [
            line.strip()
            for line in Path(paths.extbc_loc).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("%")
        ]
        assert len(rows) == 1, f"event {event_iloc}: {len(rows)} external-boundary rows {rows}"
        fields = [field.strip() for field in rows[0].split(",")]
        assert fields[0] == "1", f"event {event_iloc}: the boundary row {rows[0]} is not a water-level (type 1) row"
        assert (Path(paths.extbc_loc).parent / fields[5].strip('"')).is_file(), f"event {event_iloc}: no series file"
        checked += 1
    # Must-fire control: at least one event was read.
    assert checked == len(analysis.df_sims) > 0


def _assert_coupled_swmm_ran_on_one_thread(analysis) -> None:
    """The coupled SWMM report carries ``Number of Threads`` 1, the default of ``swmm_threads`` (DR-36).

    Class: NEW CAPABILITY. Property: for every event the coupled model's ``hydraulics.rpt`` (the file SWMM writes
    for the coupled run) holds a ``Number of Threads`` line, and the count is 1. Why required: the thread line of the
    written model is the one input of the run that reaches SWMM from ``swmm_threads`` and not from TRITON, and a line
    that is missing or carries another count would show a model that never took the setting. Killed by: a model
    written with a thread count taken from the OpenMP count of the run mode, or with no ``THREADS`` line. A second
    correct implementation reads the line from the scenario's model log. Not established: that a count above 1 is
    honoured, because ``serial`` admits only 1 and this fixture runs ``serial``.
    """
    import re

    line = re.compile(r"^\s*Number of Threads\s*\.*\s*(\d+)\s*$", re.MULTILINE)
    checked = 0
    for event_iloc in analysis.df_sims.index:
        rpt = Path(analysis._retrieve_sim_runs(event_iloc)._scenario.scen_paths.swmm_hydraulics_rpt)
        assert rpt.is_file(), f"event {event_iloc}: no coupled SWMM report {rpt}"
        match = line.search(rpt.read_text(errors="replace"))
        assert match is not None, f"event {event_iloc}: {rpt} holds no Number of Threads line"
        assert int(match.group(1)) == 1, f"event {event_iloc}: SWMM ran on {match.group(1)} threads"
        checked += 1
    assert checked == len(analysis.df_sims) > 0


def test_load_system_and_analysis(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    tst_ut.assert_file_exists(analysis.analysis_paths.simulation_directory, "simulation directory")


def test_prepare_all_scenarios(synth_all_models_analysis_cached):
    analysis = synth_all_models_analysis_cached
    analysis.run_prepare_scenarios_serially(
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
    )
    tst_ut.assert_scenarios_setup(analysis)
    _assert_folded_inputs_rendered(analysis)
    _assert_matched_outfall_coupled_through_one_water_level_row(analysis)


def test_run_sim(synth_all_models_analysis_cached):
    analysis = synth_all_models_analysis_cached
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)
    tst_ut.assert_scenarios_run(analysis)

    for model_type in tst_ut.get_enabled_model_types(analysis):
        failed_scenarios = []
        for event_iloc in analysis.df_sims.index:
            run = analysis._retrieve_sim_runs(event_iloc)
            scen = run._scenario
            if not scen.model_run_completed(model_type):
                failed_scenarios.append(str(scen.log.logfile.parent))

        if failed_scenarios:
            pytest.fail(
                f"{len(failed_scenarios)} {model_type} simulation(s) failed to complete:\n"
                + "\n".join(f"  - {d}" for d in failed_scenarios[:5])
                + (f"\n  ... and {len(failed_scenarios) - 5} more" if len(failed_scenarios) > 5 else "")
            )

    _assert_solver_read_the_folded_inputs(analysis)
    _assert_coupled_swmm_ran_on_one_thread(analysis)


def test_process_sim(synth_all_models_analysis_cached):
    analysis = synth_all_models_analysis_cached
    enabled_models = tst_ut.get_enabled_model_types(analysis)

    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_timeseries_outputs(which="both", model_type=model_type, override_clear_raw="all")
            elif model_type == "triton":
                proc.write_timeseries_outputs(which="TRITON", model_type=model_type, override_clear_raw="all")
            elif model_type == "swmm":
                proc.write_timeseries_outputs(which="SWMM", model_type=model_type, override_clear_raw="all")

    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)

        for model_type in enabled_models:
            if model_type == "tritonswmm":
                proc.write_summary_outputs(
                    which="both",
                    model_type=model_type,
                )
            elif model_type == "triton":
                proc.write_summary_outputs(
                    which="TRITON",
                    model_type=model_type,
                )
            elif model_type == "swmm":
                proc.write_summary_outputs(
                    which="SWMM",
                    model_type=model_type,
                )

    analysis._update_log()

    tst_ut.assert_timeseries_processed(analysis)
    for model_type in enabled_models:
        tst_ut.assert_model_outputs_processed(analysis, model_type)

    tst_ut.assert_triton_compiled(analysis)
    tst_ut.assert_tritonswmm_compiled(analysis)
    tst_ut.assert_swmm_compiled(analysis)

    success_clearing = analysis._all_raw_TRITON_outputs_cleared and analysis._all_raw_SWMM_outputs_cleared
    if not success_clearing:
        analysis.print_logfile_for_scenario(0)
        pytest.fail("Clearing raw outputs failed.")

    tst_ut.assert_hydraulic_components_exercised(analysis)
    _assert_observation_series_processed(analysis)


def test_observation_series_consolidate_into_the_one_analysis_store(synth_all_models_analysis_cached):
    """The observation groups land in ``analysis_datatree.zarr`` and the consolidation adds no other zarr store.

    Class: NEW CAPABILITY. Property: after ``consolidate_to_datatree`` the zarr stores under the analysis directory
    are the ones that existed before, plus at most the tree itself; each enabled TRITON model has the
    ``triton_observation`` series group on ``(event_iloc, elapsed_s, obs_point)`` and the summary group on
    ``(event_iloc, obs_point)`` with the folded point ids; and no store is named for observations. Why required: the
    developer's ruling is that observation points end up in the one analysis-level zarr and in no extra zarr, and
    ``toggle_consolidate_timeseries`` is off in this fixture, so the groups must appear without it. Killed by: a
    consolidation that writes a store of its own, one that appends only under the toggle, a group on other dimensions.
    A second correct implementation compares to the set of stores of a points-free run, as the laptop file does.
    Not established: the group content across events, which the laptop file covers on a golden tree.

    The analysis directory is shared, so the test removes the top-level entries it added; the consolidated tree it
    leaves carries no layout record, and the version-state tests refuse such a tree.
    """
    analysis = synth_all_models_analysis_cached
    ids = [pid for pid, __, __ in suite_conftest.FOLDED_OBSERVATION_POINTS]
    arms = _triton_arms(analysis)
    before = _zarr_store_paths(analysis)
    assert before, "control: the per-scenario stores written by the processing step are found"

    analysis_dir = Path(analysis.analysis_paths.analysis_dir)
    names_before = {entry.name for entry in analysis_dir.iterdir()}
    try:
        _assert_consolidation_lands_observations_in_the_one_store(analysis, ids, arms, before)
    finally:
        _remove_entries_added_since(analysis_dir, names_before)


def _assert_consolidation_lands_observations_in_the_one_store(analysis, ids, arms, before) -> None:
    analysis.process.consolidate_to_datatree()

    tree = Path(analysis.analysis_paths.analysis_datatree_zarr)
    tree_rel = str(tree.relative_to(analysis.analysis_paths.analysis_dir))
    after = _zarr_store_paths(analysis)
    assert tree_rel in after, "control: the matcher finds the consolidated tree"
    assert after - before <= {tree_rel}, f"consolidation added stores beyond the tree: {sorted(after - before)}"
    assert not [p for p in after if "observ" in Path(p).name], f"a store is named for observations: {sorted(after)}"
    for model_type, arm in arms:
        group = f"{arm['family']}/triton_observation"
        series = xr.open_zarr(tree, group=f"{group}/timeseries", consolidated=True)
        assert series["point_flood_depth"].dims == ("event_iloc", "elapsed_s", "obs_point"), model_type
        assert series.sizes["event_iloc"] == len(analysis.df_sims), model_type
        assert float(series["elapsed_s"].values[0]) == 0.0, model_type
        assert series["obs_point_id"].values.tolist() == ids, model_type
        summary = xr.open_zarr(tree, group=f"{group}/summary", consolidated=True)
        for name in ("max_point_flood_depth", "time_of_max_point_flood_depth"):
            assert summary[name].dims == ("event_iloc", "obs_point"), f"{model_type} {name}"
        assert summary["obs_point_id"].values.tolist() == ids, model_type


def test_swmm_cross_model_consistency(synth_all_models_analysis_cached):
    analysis = synth_all_models_analysis_cached

    if "swmm" not in tst_ut.get_enabled_model_types(analysis):
        pytest.skip("SWMM-only model not enabled")
    if "tritonswmm" not in tst_ut.get_enabled_model_types(analysis):
        pytest.skip("TRITON-SWMM model not enabled")

    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        paths = proc.scen_paths

        swmm_node_ts = paths.output_swmm_only_node_time_series
        swmm_link_ts = paths.output_swmm_only_link_time_series
        tritonswmm_node_ts = paths.output_tritonswmm_node_time_series
        tritonswmm_link_ts = paths.output_tritonswmm_link_time_series

        tst_ut.assert_file_exists(swmm_node_ts, "SWMM-only node timeseries")
        tst_ut.assert_file_exists(swmm_link_ts, "SWMM-only link timeseries")
        tst_ut.assert_file_exists(tritonswmm_node_ts, "TRITON-SWMM node timeseries")
        tst_ut.assert_file_exists(tritonswmm_link_ts, "TRITON-SWMM link timeseries")

        ds_swmm_nodes = xr.open_dataset(swmm_node_ts)
        ds_swmm_links = xr.open_dataset(swmm_link_ts)
        ds_tritonswmm_nodes = xr.open_dataset(tritonswmm_node_ts)
        ds_tritonswmm_links = xr.open_dataset(tritonswmm_link_ts)

        swmm_node_ids = set(ds_swmm_nodes["node_id"].values.tolist())
        swmm_link_ids = set(ds_swmm_links["link_id"].values.tolist())
        tritonswmm_node_ids = set(ds_tritonswmm_nodes["node_id"].values.tolist())
        tritonswmm_link_ids = set(ds_tritonswmm_links["link_id"].values.tolist())

        missing_nodes = swmm_node_ids - tritonswmm_node_ids
        missing_links = swmm_link_ids - tritonswmm_link_ids

        if missing_nodes:
            pytest.fail(f"TRITON-SWMM node_ids missing {len(missing_nodes)} SWMM-only nodes.")
        if missing_links:
            pytest.fail(f"TRITON-SWMM link_ids missing {len(missing_links)} SWMM-only links.")

        # A coupled run's SWMM output carries 0 or 1 FEWER report periods than a
        # standalone run of the same window -- NEVER more, and never a gap in the
        # interior. saveResults() (external/swmm/src/solver/swmm5.c:588) emits a
        # period on `NewRoutingTime >= ReportTime`, and NewRoutingTime is a pure
        # accumulation (routing.c:324) that is never re-anchored to a boundary.
        # Interior periods are safe because the step that crosses a boundary
        # overshoots it and `>=` catches the overshoot. Only the TERMINAL period
        # needs an exact landing on RoutingDuration, because no later step exists
        # to overshoot with. The standalone driver reaches that landing
        # structurally: `do { swmm_step } while (elapsedTime > 0.0)` keeps stepping
        # until SWMM reports the duration consumed, and the overshoot step takes
        # execRouting's clamp branch (swmm5.c:550-555), which ASSIGNS
        # nextRoutingTime = RoutingDuration. TRITON's loop
        # (`while (simtime < sim_duration)`, triton.h:2152) exits on its own clock
        # and never consults elapsedTime, so the clamp is never entered and the
        # terminal period is dropped whenever accumulation lands short of the
        # 1e-8 msec tolerance. Measured on this fixture: coupled hydraulics.out
        # NUM_PERIODS=1079 vs standalone hydro.out NUM_PERIODS=1080, identical
        # report_step and start_date.
        #
        # So equality is NOT the invariant. What IS guaranteed -- and what this
        # asserts -- is strictly stronger on everything except that terminal
        # element: same start, exact elementwise prefix (hence identical cadence
        # and identical shared timestamps), and a deficit of at most one period.
        # The span itself is guarded earlier and fail-closed by
        # scenario.py::_assert_scenario_forcing_window_agreement, which raises
        # when TRITON's sim_duration disagrees with the SWMM .inp window.
        for kind, ds_swmm, ds_coupled in (
            ("Node", ds_swmm_nodes, ds_tritonswmm_nodes),
            ("Link", ds_swmm_links, ds_tritonswmm_links),
        ):
            t_swmm = ds_swmm["date_time"].values.tolist()
            t_coupled = ds_coupled["date_time"].values.tolist()
            # A prefix relation is VACUOUSLY satisfiable on an empty axis:
            # [] == [][:0] is True, and an empty coupled axis against a
            # length-1 standalone axis even produces a legitimate-looking
            # deficit of 1. Both would report green on a state that should
            # fail, so emptiness is rejected explicitly before either check
            # below is allowed to speak. Measured, not assumed.
            if not t_swmm or not t_coupled:
                pytest.fail(
                    f"{kind} time series: an axis is EMPTY (standalone="
                    f"{len(t_swmm)}, coupled={len(t_coupled)}). The prefix and "
                    "deficit checks below are vacuously satisfiable on an empty "
                    "axis, so this is failed explicitly rather than passed "
                    "silently."
                )
            deficit = len(t_swmm) - len(t_coupled)
            if deficit < 0 or deficit > 1:
                pytest.fail(
                    f"{kind} time series: coupled run has {len(t_coupled)} report "
                    f"periods against standalone's {len(t_swmm)} (deficit "
                    f"{deficit}). Only a deficit of 0 or 1 is structurally "
                    "possible; see the comment above."
                )
            if t_coupled != t_swmm[: len(t_coupled)]:
                pytest.fail(
                    f"{kind} time series: the coupled date_time axis is not an "
                    "exact prefix of the standalone axis -- the two runs disagree "
                    "on start time or reporting cadence, which the shared SWMM "
                    ".inp window makes impossible unless something upstream "
                    "diverged."
                )

        tst_ut.assert_node_vars_consistent(ds_swmm_nodes, ds_tritonswmm_nodes, synth_all_models_analysis_cached)
        swmm_link_vars = {str(v) for v in ds_swmm_links.data_vars}
        tritonswmm_link_vars = {str(v) for v in ds_tritonswmm_links.data_vars}

        swmm_link_vars = tst_ut.normalize_swmm_link_vars(swmm_link_vars)
        tritonswmm_link_vars = tst_ut.normalize_swmm_link_vars(tritonswmm_link_vars)

        if swmm_link_vars != tritonswmm_link_vars:
            pytest.fail("Link time series data variables do not match")


def _assert_every_series_holds_every_period(series_by_node, n_periods, n_times):
    """Assert each node series, and the time axis, hold exactly ``n_periods`` report periods, over more than one.

    ``n_periods`` is the output's own ``NUM_PERIODS``. Must-fire control: ``n_periods > 1`` and a non-empty node
    set, so a reader that returns nothing, or a one-period output, fails here instead of passing vacuously.

    Class: NEW CAPABILITY. Property: every node series and the time axis of the hydrology ``.out`` carry exactly
    ``n_periods`` values. Why required: the invariant test's other clauses compare nodes to one another, so a
    read that drops the final report period from every series left them all green. Killed by: a reader that
    returns ``n_periods - 1`` values (pyswmm 2.1.0's ``Output.node_series``); measured red on a Rivanna compute
    node with a plugin that drops the final period. A second correct reader (any read over ``(0, NUM_PERIODS)``)
    still passes. Not established: that every possible accessor defect is a one-period drop.
    """
    assert n_periods > 1, f"the hydrology .out reports {n_periods} period(s); the count check is vacuous below 2"
    assert series_by_node, "no node series were read, so the period count was not checked"
    assert n_times == n_periods, f"the time axis has {n_times} steps for {n_periods} report periods"
    short = {k: len(v) for k, v in series_by_node.items() if len(v) != n_periods}
    assert not short, (
        f"{len(short)} node series do not hold all {n_periods} report periods (final period dropped?): "
        f"{dict(sorted(short.items()))}"
    )


def test_hydrology_variant_local_runoff_invariant(synth_all_models_analysis_cached):
    """Local-runoff-invariant guard — synth-swmm-hydrology-continuity Phase 1 (R7/R8).

    Pins the three interlocking clauses of the stipulation ``synthetic hydrology
    variant is conduit-free with junction runoff nodes and kinwave routing``
    against the hydrology variant's real ``.out``/``.rpt``. VALUE-FREE by
    construction (no hardcoded discharge magnitude), so it survives a fixture
    retune, and it FAILS under each of the three refuted regressions the plan
    exists to prevent (V4 negative controls in the master Validation Plan):

    * outfall promotion -> the runoff nodes get skipped by
      ``write_hydrograph_files`` (``if key not in lst_outfalls``), so the
      hydrograph count drops below 11 (measured 0/11);
    * added conduits -> downstream nodes fold in upstream runoff and
      double-count, breaking per-node uniformity (measured 1.00x-10.73x);
    * KINWAVE->DYNWAVE reversion -> per-node ``TOTAL_INFLOW`` is byte-identical
      across DW/KW (master R4/A1), so assertions (1)-(3) are structurally BLIND;
      the secondary continuity assertion (4) is that regression's ONLY coverage.

    A fourth, count-bearing clause (0) asserts every node series and the time axis hold the output's FULL
    ``NUM_PERIODS`` report periods, through the direct toolkit read and through the toolkit's own reader, so a
    read that drops the final period (pyswmm 2.1.0's ``Output.node_series``) fails it; the clauses above are
    value-free and node-relative and cannot tell 11 periods from 12.

    Mirrors ``swmm_runoff_modeling.write_hydrograph_files``'s two filters: the
    outfall-skip and the ``d_inflow.sum() > 0`` positive-inflow filter that
    selects the nodes actually delivering a hydrograph to TRITON.
    """
    import pandas as pd
    from pyswmm import Output
    from swmm.toolkit import output as swmm_output
    from swmm.toolkit.shared_enum import NodeAttribute, Time

    from hhemt.scenario import return_df_of_nodes_grouped_by_DEM_gridcell
    from hhemt.swmm_output_parser import return_swmm_system_outputs

    analysis = synth_all_models_analysis_cached
    # System-level preprocessing produces the processed DEM
    # (`elevation_<res>m.dem`) that write_hydrograph_files and
    # return_df_of_nodes_grouped_by_DEM_gridcell read. run_prepare_scenarios_serially
    # is scenario-level and does NOT produce it. Since the R12 fix the cached builder
    # (tests/fixtures/test_case_builder.py, the `not start_from_scratch and not
    # skip_run` branch) already seeds it on every construction; this call is kept as
    # an idempotent restatement of the guard's own precondition so the test stands
    # alone even if that seed is ever re-gated. Idempotent: skips when the DEM
    # exists and passes integrity (system.py::create_dem_for_TRITON).
    analysis._system.process_system_level_inputs()
    analysis.run_prepare_scenarios_serially(
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
    )

    scen = analysis._retrieve_sim_runs(0)._scenario
    hydro_inp = scen.scen_paths.swmm_hydro_inp
    dem_processed = scen._system.sys_paths.dem_processed
    hydro_out = str(hydro_inp).replace(".inp", ".out")
    hydro_rpt = str(hydro_inp).replace(".inp", ".rpt")

    # Reproduce write_hydrograph_files' outfall-skip partition via the SAME
    # function it calls: every node NOT in [OUTFALLS] is a hydrograph candidate.
    df_node_locs, lst_outfalls = return_df_of_nodes_grouped_by_DEM_gridcell(hydro_inp, dem_processed)
    non_outfall = [k for k in df_node_locs["node_key"] if k not in lst_outfalls]

    # Summed TOTAL_INFLOW per non-outfall node, from the same .out
    # write_hydrograph_files consumes.
    # Read over the window (0, NUM_PERIODS) through the toolkit, and not Output.node_series: pyswmm 2.1.0 drops
    # the final report period from that accessor's default window, which would make the sum depend on the
    # installed pyswmm version.
    with Output(hydro_out) as out:
        n_periods = int(swmm_output.get_times(out.handle, Time.NUM_PERIODS))
        series_by_node = {
            key: pd.Series(
                swmm_output.get_node_series(out.handle, out.nodes[key], NodeAttribute.TOTAL_INFLOW, 0, n_periods)
            )
            for key in non_outfall
        }
        n_times = len(out.times)

    # (0) the FULL report-period count of the output read. The value-free uniformity and zero-set assertions
    #     below compare nodes to one another, so a read that drops the final report period from EVERY series
    #     leaves them all green; this is the assertion that tells 11 periods from 12. The count is anchored to
    #     the file (NUM_PERIODS and the time axis), never to a literal, so a fixture retune does not touch it.
    _assert_every_series_holds_every_period(series_by_node, n_periods, n_times)
    inflow_sum = {key: float(series.sum()) for key, series in series_by_node.items()}

    # The reader the toolkit itself calls (write_hydrograph_files and the capture writer) returns the same
    # full count for the same file: a reader that regressed to a pyswmm accessor dropping the final period
    # fails here even when the direct read above is whole. Looked up on the module at call time.
    from hhemt import swmm_runoff_modeling as srm

    node_inflow = srm.read_hydrology_node_inflow(hydro_out, df_node_locs, lst_outfalls)
    _assert_every_series_holds_every_period(
        {k: pd.Series(v) for k, v in node_inflow.inflow_by_node.items()}, n_periods, len(node_inflow.times_hr)
    )

    # write_hydrograph_files' second filter (`if d_inflow.sum() > 0`): only
    # positive-inflow nodes produce a TRITON hydrograph. That is the
    # coupling-meaningful count.
    runoff_nodes = {k: v for k, v in inflow_sum.items() if v > 0.0}
    zero_nodes = {k: v for k, v in inflow_sum.items() if v == 0.0}

    # (1) exactly 11 runoff nodes deliver a hydrograph to TRITON. Outfall
    #     promotion drops this below 11; a spurious inflow at a BC node raises it.
    assert len(runoff_nodes) == 11, (
        f"expected 11 hydrograph-producing runoff nodes, got {len(runoff_nodes)}: "
        f"{sorted(runoff_nodes)} (non-outfall nodes: {sorted(non_outfall)}; "
        f"outfalls skipped: {sorted(lst_outfalls)})"
    )

    # (2) the BC-side junctions carry no local runoff -> exactly 0.0. A conduit would fold upstream runoff into
    #     them (non-zero). The matched outfall is one of them: the hydrology role converts it to a junction, so it
    #     is not in `lst_outfalls`, and it books zero like `collector` and `sewer_outflow`.
    from hhemt.synthetic_model.swmm_model import MATCHED_OUTFALL

    expected_zero = {"collector", "sewer_outflow", MATCHED_OUTFALL}
    assert set(zero_nodes) == expected_zero, (
        f"expected exactly {sorted(expected_zero)} to book zero TOTAL_INFLOW, got {sorted(zero_nodes)}"
    )

    # (3) every runoff node books IDENTICAL local runoff (uniform subcatchments +
    #     conduit-free -> each node sees only its own DEM cell). VALUE-FREE:
    #     compares nodes to one another, never to a magnitude. Added conduits
    #     double-count downstream and break this.
    reference = next(iter(runoff_nodes.values()))
    for name, val in runoff_nodes.items():
        assert val == reference, (
            f"runoff node {name} TOTAL_INFLOW sum {val} != reference {reference}; "
            f"local-runoff uniformity broken (a conduit folds upstream runoff "
            f"into downstream nodes)"
        )

    # (4) SECONDARY — pins the routing choice against a silent KINWAVE->DYNWAVE
    #     reversion. NOT primary: per master R4/A1 the per-node TOTAL_INFLOW above
    #     is byte-identical across DW/KW, so (1)-(3) cannot see a reversion; this
    #     is its only coverage. Continuity is NOT in the .out binary -> read .rpt.
    with open(hydro_rpt) as f:
        rpt_lines = f.readlines()
    sys_out = return_swmm_system_outputs(rpt_lines)
    assert abs(sys_out["flow_continuity_error"]) < 5.0, (
        f"hydrology-variant flow-routing continuity error "
        f"{sys_out['flow_continuity_error']}% >= 5.0% -- routing likely "
        f"reverted to DYNWAVE on a conduit-free network (see "
        f"swmm_model._options_df)"
    )
