import pytest
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from tests.utils_for_testing import is_scheduler_context

pytestmark = pytest.mark.skipif(
    is_scheduler_context(), reason="Only runs on non-HPC systems."
)

pytestmark = pytest.mark.skipif(False, reason="Skipping for now...")


def test_run_multisim_concurrently():
    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    analysis._system.compile_TRITON_SWMM(recompile_if_already_done_successfully=True)
    assert analysis._system.compilation_successful
    prepare_scenario_launchers = analysis.retrieve_prepare_scenario_launchers(
        overwrite_scenario=True, verbose=True
    )
    analysis.run_python_functions_concurrently(prepare_scenario_launchers)
    assert analysis.log.all_scenarios_created.get()
    launch_functions = analysis._create_launchable_sims(
        pickup_where_leftoff=False, verbose=True
    )
    analysis.run_simulations_concurrently(launch_functions, verbose=True)
    assert analysis.log.all_sims_run.get() == True


def test_concurrently_process_scenario_timeseries():
    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    scenario_timeseries_processing_launchers = (
        analysis.retreive_scenario_timeseries_processing_launchers()
    )
    analysis.run_python_functions_concurrently(scenario_timeseries_processing_launchers)
    # verify that time series outputs processed
    success_processing = (
        analysis.log.all_TRITON_timeseries_processed.get()
        and analysis.log.all_SWMM_timeseries_processed.get()
    )
    if not success_processing:
        analysis.print_logfile_for_scenario(0)
        pytest.fail(f"Processing TRITON and SWMM time series failed.")

    analysis.consolidate_TRITON_and_SWMM_simulation_summaries(overwrite_if_exist=True)
    assert analysis.TRITON_analysis_summary_created
    assert analysis.SWMM_node_analysis_summary_created
    assert analysis.SWMM_link_analysis_summary_created
