import pytest
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst


def test_run_multiple_sims_in_sequence():
    multi_sim = tst.retreive_norfolk_multi_sim_test_case(start_from_scratch=True)
    analysis = multi_sim.system.analysis
    analysis.compile_TRITON_SWMM()
    assert analysis.log.TRITONSWMM_compiled_successfully.get()
    analysis.run_prepare_scenarios_serially(
        overwrite_scenarios=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    assert analysis.log.all_scenarios_created.get()
    analysis.run_all_sims_in_serially(
        pickup_where_leftoff=False,
        process_outputs_after_sim_completion=True,
    )
    # verify that models ran
    success = analysis.scenarios[0].sim_run_completed
    if not success:
        analysis.print_logfile_for_scenario(0)
        pytest.fail(f"Multi simulation did not run successfully.")
    # verify that time series outputs processed
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


def test_consolidate_multisim_TRITON_outputs():
    multi_sim = tst.retreive_norfolk_multi_sim_test_case(start_from_scratch=False)
    multi_sim.system.analysis.consolidate_TRITON_simulation_summaries(
        overwrite_if_exist=True
    )
    assert multi_sim.system.analysis.TRITON_analysis_summary_created


def test_consolidate_multisim_SWMM_outputs():
    multi_sim = tst.retreive_norfolk_multi_sim_test_case(start_from_scratch=False)
    multi_sim.system.analysis.consolidate_SWMM_simulation_summaries(
        overwrite_if_exist=True
    )
    assert multi_sim.system.analysis.SWMM_node_analysis_summary_created
    assert multi_sim.system.analysis.SWMM_link_analysis_summary_created


def test_run_multisim_concurrently():
    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    analysis.compile_TRITON_SWMM(recompile_if_already_done_successfully=True)
    assert analysis.log.TRITONSWMM_compiled_successfully.get()
    prepare_scenario_launchers = analysis.retrieve_prepare_scenario_launchers(
        overwrite_scenario=True, verbose=True
    )
    analysis.run_python_functions_concurrently(prepare_scenario_launchers)
    assert analysis.log.all_scenarios_created.get()
    launch_functions = analysis._create_launchable_sims(
        pickup_where_leftoff=False, verbose=True
    )
    analysis.run_simulations_concurrently_on_desktop(launch_functions, verbose=True)
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
