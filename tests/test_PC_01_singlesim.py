# tests/test_TRITON_SWMM_toolkit.py
import pytest
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_load_system_and_analysis(norfolk_single_sim_analysis):
    analysis = norfolk_single_sim_analysis
    assert analysis.analysis_paths.simulation_directory.exists()


# SYSTEM TESTS
def test_create_dem_for_TRITON(norfolk_single_sim_analysis):
    analysis = norfolk_single_sim_analysis
    analysis._system.create_dem_for_TRITON()
    rds = analysis._system.processed_dem_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_create_mannings_file_for_TRITON(norfolk_single_sim_analysis):
    analysis = norfolk_single_sim_analysis
    analysis._system.create_mannings_file_for_TRITON()
    rds = analysis._system.mannings_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


# COMPILING TRITON-SWMM
def test_compile_TRITONSWMM_for_cpu_sims(norfolk_single_sim_analysis):
    analysis = norfolk_single_sim_analysis
    analysis._system.compile_TRITON_SWMM(
        recompile_if_already_done_successfully=True,
        redownload_triton_swmm_if_exists=True,
    )
    assert analysis._system.compilation_successful


# SCENARIO SET UP
def test_prepare_all_scenarios(norfolk_single_sim_analysis_cached):
    analysis = norfolk_single_sim_analysis_cached
    analysis.run_prepare_scenarios_serially(
        overwrite_scenarios=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    tst_ut.assert_scenarios_setup(analysis)


def test_run_sim(norfolk_single_sim_analysis_cached):
    analysis = norfolk_single_sim_analysis_cached
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)
    tst_ut.assert_scenarios_run(analysis)


def test_process_sim(norfolk_single_sim_analysis_cached):
    analysis = norfolk_single_sim_analysis_cached
    analysis.process_all_sim_timeseries_serially()
    tst_ut.assert_timeseries_processed(analysis)
    success_clearing = (
        analysis.log.all_raw_TRITON_outputs_cleared.get()
        and analysis.log.all_raw_SWMM_outputs_cleared.get()
    )
    if not success_clearing:
        analysis.print_logfile_for_scenario(0)
        pytest.fail(f"Clearning raw outputs failed.")
