# tests/test_TRITON_SWMM_toolkit.py
import pytest
from TRITON_SWMM_toolkit.examples import retrieve_norfolk_testcase


def test_load_system_and_experiment():
    single_sim_single_core = retrieve_norfolk_testcase()
    assert (
        single_sim_single_core.system.experiment.exp_paths.simulation_directory.exists()
    )


# SYSTEM TESTS
def test_create_dem_for_TRITON():
    single_sim_single_core = single_sim_single_core = retrieve_norfolk_testcase()
    single_sim_single_core.system.create_dem_for_TRITON()
    rds = single_sim_single_core.system.open_processed_dem_as_rds()
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_create_mannings_file_for_TRITON():
    single_sim_single_core = single_sim_single_core = retrieve_norfolk_testcase()
    single_sim_single_core.system.create_mannings_file_for_TRITON()
    rds = single_sim_single_core.system.open_processed_mannings_as_rds()
    assert rds.shape == (1, 537, 551)  # type: ignore


# COMPILING TRITON-SWMM
def test_compile_TRITONSWMM_for_cpu_sims():
    single_sim_single_core = single_sim_single_core = retrieve_norfolk_testcase()
    single_sim_single_core.system.experiment.compile_TRITON_SWMM()
    assert single_sim_single_core.system.experiment._validate_compilation()


# SCENARIO SET UP
def test_prepare_all_scenarios():
    single_sim_single_core = single_sim_single_core = retrieve_norfolk_testcase()
    single_sim_single_core.system.experiment.prepare_all_scenarios(
        overwrite_sims=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    if not single_sim_single_core.system.experiment.scenarios[
        0
    ].log.scenario_creation_complete:
        single_sim_single_core.system.experiment.print_logfile_for_scenario(0)
        pytest.fail(f"Scenario not succesfully set up.")


def test_run_sim():
    single_sim_single_core = single_sim_single_core = retrieve_norfolk_testcase()
    single_sim_single_core.system.experiment.run_all_sims_in_series(
        mode=single_sim_single_core.system.experiment.run_modes.SINGLE_CORE,
        pickup_where_leftoff=False,
    )
    status = single_sim_single_core.system.experiment.scenarios[0].latest_sim_status()
    if status != "simulation completed":
        single_sim_single_core.system.experiment.print_logfile_for_scenario(0)
        pytest.fail(f"Simulation did not run successfully.")
