import pytest


import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_run_multisim_concurrently(norfolk_multi_sim_analysis):
    analysis = norfolk_multi_sim_analysis
    analysis._system.compile_TRITON_SWMM(recompile_if_already_done_successfully=True)
    prepare_scenario_launchers = analysis.retrieve_prepare_scenario_launchers(
        overwrite_scenario=True, verbose=True
    )
    analysis.run_python_functions_concurrently(prepare_scenario_launchers)
    launch_functions = analysis._create_launchable_sims(
        pickup_where_leftoff=False, verbose=True
    )
    analysis.run_simulations_concurrently(launch_functions, verbose=True)

    tst_ut.assert_system_setup(analysis)
    tst_ut.assert_scenarios_setup(analysis)
    tst_ut.assert_scenarios_run(analysis)


def test_concurrently_process_scenario_timeseries(norfolk_multi_sim_analysis_cached):
    analysis = norfolk_multi_sim_analysis_cached
    scenario_timeseries_processing_launchers = (
        analysis.retrieve_scenario_timeseries_processing_launchers(
            clear_raw_outputs=False
        )
    )
    analysis.run_python_functions_concurrently(scenario_timeseries_processing_launchers)
    analysis.consolidate_TRITON_and_SWMM_simulation_summaries(
        overwrite_if_exist=True,
    )
    tst_ut.assert_timeseries_processed(analysis)
    tst_ut.assert_analysis_summaries_created(analysis)
