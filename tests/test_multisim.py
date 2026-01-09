import pytest
from TRITON_SWMM_toolkit.examples import TRITON_SWMM_testcases as tst


def test_run_multiple_sims_in_sequence():
    multi_sim = tst.retreive_norfolk_multi_sim_test_case(start_from_scratch=True)
    exp = multi_sim.system.analysis
    exp.compile_TRITON_SWMM()
    exp.prepare_all_scenarios(
        overwrite_sims=True, rerun_swmm_hydro_if_outputs_exist=True
    )
    exp.run_all_sims_in_series(
        mode=exp.run_modes.SINGLE_CORE,
        pickup_where_leftoff=False,
        process_outputs_after_sim_completion=True,
    )
    # verify that models ran
    success = exp.scenarios[0].sim_run_completed
    if not success:
        exp.print_logfile_for_scenario(0)
        pytest.fail(f"Multi simulation did not run successfully.")
    # verify that time series outputs processed
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
