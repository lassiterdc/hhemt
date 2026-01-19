# tests/test_TRITON_SWMM_toolkit.py
import pytest
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils import is_scheduler_context
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario

pytestmark = pytest.mark.skipif(
    is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_load_system_and_analysis():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case(
        start_from_scratch=True
    )
    analysis = single_sim_single_core.system.analysis
    assert analysis.analysis_paths.simulation_directory.exists()


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
    analysis = single_sim_single_core.system.analysis
    analysis.compile_TRITON_SWMM()
    assert analysis.compilation_successful


# SCENARIO SET UP
def test_prepare_all_scenarios():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case(
        start_from_scratch=False
    )
    analysis = single_sim_single_core.system.analysis
    analysis.run_prepare_scenarios_serially(
        overwrite_scenarios=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    scen = TRITONSWMM_scenario(0, analysis)
    if not scen.log.scenario_creation_complete.get():
        analysis.print_logfile_for_scenario(0)
        pytest.fail(f"Scenario not succesfully set up.")


def test_run_sim():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case()
    analysis = single_sim_single_core.system.analysis
    analysis.run_sims_in_sequence(pickup_where_leftoff=False)

    scen = TRITONSWMM_scenario(0, analysis)
    success = scen.sim_run_completed

    if not success:
        analysis.print_logfile_for_scenario(0)
        pytest.fail(f"Simulation did not run successfully.")


def test_process_sim():
    single_sim_single_core = tst.retreive_norfolk_single_sim_test_case()
    analysis = single_sim_single_core.system.analysis
    analysis.process_all_sim_timeseries_serially()
    success_processing = (
        analysis.log.all_TRITON_timeseries_processed.get()
        and analysis.log.all_SWMM_timeseries_processed.get()
    )
    if not success_processing:
        analysis.print_logfile_for_scenario(0)
        pytest.fail(f"Processing TRITON and SWMM time series failed.")
    success_clearing = (
        analysis.log.all_raw_TRITON_outputs_cleared.get()
        and analysis.log.all_raw_SWMM_outputs_cleared.get()
    )
    if not success_clearing:
        analysis.print_logfile_for_scenario(0)
        pytest.fail(f"Clearning raw outputs failed.")
