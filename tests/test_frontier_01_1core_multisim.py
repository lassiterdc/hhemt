import pytest
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


def test_load_system_and_analysis(norfolk_frontier_multisim_analysis):
    analysis = norfolk_frontier_multisim_analysis
    tst_ut.assert_file_exists(
        analysis.analysis_paths.simulation_directory, "simulation directory"
    )


def test_create_dem_for_TRITON(norfolk_frontier_multisim_analysis_cached):
    analysis = norfolk_frontier_multisim_analysis_cached
    analysis._system.create_dem_for_TRITON()
    rds = analysis._system.processed_dem_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_create_mannings_file_for_TRITON(norfolk_frontier_multisim_analysis_cached):
    analysis = norfolk_frontier_multisim_analysis_cached
    analysis._system.create_mannings_file_for_TRITON()
    rds = analysis._system.mannings_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_compile_TRITONSWMM_for_cpu_sims(norfolk_frontier_multisim_analysis_cached):
    analysis = norfolk_frontier_multisim_analysis_cached
    analysis._system.compile_TRITON_SWMM()
    assert analysis._system.compilation_successful


def test_prepare_scenarios(norfolk_frontier_multisim_analysis_cached):
    analysis = norfolk_frontier_multisim_analysis_cached
    prepare_scenario_launchers = analysis.retrieve_prepare_scenario_launchers(
        overwrite_scenario_if_already_set_up=True, verbose=True
    )
    analysis.run_python_functions_concurrently(prepare_scenario_launchers, verbose=True)
    tst_ut.assert_scenarios_setup(analysis)


def test_run_sims(norfolk_frontier_multisim_analysis_cached):
    analysis = norfolk_frontier_multisim_analysis_cached
    launch_functions = analysis._create_launchable_sims(
        pickup_where_leftoff=True, verbose=True
    )

    analysis.run_simulations_concurrently(launch_functions, verbose=True)
    tst_ut.assert_scenarios_run(analysis)


def test_concurrently_process_scenario_timeseries(
    norfolk_frontier_multisim_analysis_cached,
):
    analysis = norfolk_frontier_multisim_analysis_cached
    scenario_timeseries_processing_launchers = (
        analysis.retrieve_scenario_timeseries_processing_launchers(which="both")
    )
    analysis.run_python_functions_concurrently(
        scenario_timeseries_processing_launchers, verbose=True
    )
    tst_ut.assert_timeseries_processed(analysis, which="both")
