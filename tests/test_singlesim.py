# tests/test_TRITON_SWMM_toolkit.py
import pytest
from TRITON_SWMM_toolkit.examples import TRITON_SWMM_testcases as tst


def test_load_system_and_analysis():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case()
    assert (
        single_sim_single_core.system.analysis.analysis_paths.simulation_directory.exists()
    )


# SYSTEM TESTS
def test_create_dem_for_TRITON():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case()
    single_sim_single_core.system.create_dem_for_TRITON()
    rds = single_sim_single_core.system.processed_dem_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_create_mannings_file_for_TRITON():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case()
    single_sim_single_core.system.create_mannings_file_for_TRITON()
    rds = single_sim_single_core.system.open_processed_mannings_as_rds()
    assert rds.shape == (1, 537, 551)  # type: ignore


# COMPILING TRITON-SWMM
def test_compile_TRITONSWMM_for_cpu_sims():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case()
    single_sim_single_core.system.analysis.compile_TRITON_SWMM()
    assert single_sim_single_core.system.analysis._validate_compilation()


# SCENARIO SET UP
def test_prepare_all_scenarios():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case(
        start_from_scratch=True
    )
    single_sim_single_core.system.analysis.compile_TRITON_SWMM()
    single_sim_single_core.system.analysis.prepare_all_scenarios(
        overwrite_sims=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    if not single_sim_single_core.system.analysis.scenarios[
        0
    ].log.scenario_creation_complete.get():
        single_sim_single_core.system.analysis.print_logfile_for_scenario(0)
        pytest.fail(f"Scenario not succesfully set up.")


def test_run_sim():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case()
    single_sim_single_core.system.analysis.run_all_sims_in_series(
        mode=single_sim_single_core.system.analysis.run_modes.SINGLE_CORE,
        pickup_where_leftoff=False,
    )
    success = single_sim_single_core.system.analysis.scenarios[0].sim_run_completed
    if not success:
        single_sim_single_core.system.analysis.print_logfile_for_scenario(0)
        pytest.fail(f"Simulation did not run successfully.")


def test_process_sim():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case()
    exp = single_sim_single_core.system.analysis
    exp.process_all_sim_outputs()
    success_processing = (
        exp.log.all_TRITON_timeseries_processed.get()
        and exp.log.all_SWMM_timeseries_processed.get()
    )
    if not success_processing:
        exp.print_logfile_for_scenario(0)
        pytest.fail(f"Processing TRITON and SWMM time series failed.")
    success_clearing = (
        exp.log.all_raw_TRITON_outputs_cleared.get()
        and exp.log.all_raw_SWMM_outputs_cleared.get()
    )
    if not success_clearing:
        exp.print_logfile_for_scenario(0)
        pytest.fail(f"Clearning raw outputs failed.")
