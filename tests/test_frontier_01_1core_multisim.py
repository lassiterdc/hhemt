import os
import pytest
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    not tst_ut.on_frontier(), reason="Only runs on Frontier HPC"
)

# cd /lustre/orion/***REMOVED***/proj-shared/***REMOVED***/TRITON-SWMM_toolkit
# salloc -A ***REMOVED*** -p batch -t 0-02:00:00 -N 2 --cpus-per-task=1 --ntasks-per-node=32 --gres=gpu:2 -q debug --mem=0
# conda activate triton_swmm_toolkit

# bash commands
# pgrep -l srun # lists all srun processes
# ps -o pid= --ppid $$ | xargs kill -9 # kills all srun processes


def test_load_system_and_analysis():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=True
    )
    assert (
        nrflk_multisim_ensemble.system.analysis.analysis_paths.simulation_directory.exists()
    )


def test_create_dem_for_TRITON():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=False
    )
    nrflk_multisim_ensemble.system.create_dem_for_TRITON()
    rds = nrflk_multisim_ensemble.system.processed_dem_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_create_mannings_file_for_TRITON():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=False
    )
    nrflk_multisim_ensemble.system.create_mannings_file_for_TRITON()
    rds = nrflk_multisim_ensemble.system.mannings_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_compile_TRITONSWMM_for_cpu_sims():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=False
    )
    nrflk_multisim_ensemble.system.compile_TRITON_SWMM()
    assert nrflk_multisim_ensemble.system.compilation_successful


def test_prepare_scenarios():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=False
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    prepare_scenario_launchers = analysis.retrieve_prepare_scenario_launchers(
        overwrite_scenario=True, verbose=True
    )
    analysis.run_python_functions_concurrently(prepare_scenario_launchers, verbose=True)
    if analysis.log.all_scenarios_created.get() != True:
        scens_not_created = "\n".join(analysis.scenarios_not_created)
        pytest.fail(
            f"Processing TRITON and SWMM time series failed.Scenarios not created: \n{scens_not_created}"
        )


def test_run_sims():
    from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst

    nrflk_multisim_ensemble = tst.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=False
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    launch_functions = analysis._create_launchable_sims(
        pickup_where_leftoff=True, verbose=True
    )

    analysis.run_simulations_concurrently(launch_functions, verbose=True)

    if analysis.log.all_sims_run.get() != True:
        sims_not_run = "\n".join(analysis.scenarios_not_run)
        pytest.fail(
            f"Running TRITONSWMM ensemble failed. Scenarios not run: \n{sims_not_run}"
        )


def test_concurrently_process_scenario_timeseries():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=False
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    scenario_timeseries_processing_launchers = (
        analysis.retrieve_scenario_timeseries_processing_launchers(which="TRITON")
    )
    analysis.run_python_functions_concurrently(
        scenario_timeseries_processing_launchers, verbose=True
    )
    # verify that time series outputs processed
    success_processing = (
        analysis.log.all_TRITON_timeseries_processed.get()
        # and analysis.log.all_SWMM_timeseries_processed.get()
    )
    if not success_processing:
        analysis._update_log()
        analysis.log.print()
        pytest.fail(f"Processing TRITON and SWMM time series failed.")

    analysis.consolidate_TRITON_simulation_summaries(overwrite_if_exist=True)
    assert analysis.TRITON_analysis_summary_created
    # assert analysis.SWMM_node_analysis_summary_created
    # assert analysis.SWMM_link_analysis_summary_created
