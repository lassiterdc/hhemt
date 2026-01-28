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


def test_retrieve_test(norfolk_frontier_all_configs_analysis):
    analysis = norfolk_frontier_all_configs_analysis
    assert analysis.analysis_paths.simulation_directory.exists()


def test_compile(norfolk_frontier_all_configs_analysis_cached):
    analysis = norfolk_frontier_all_configs_analysis_cached
    analysis.sensitivity.compile_TRITON_SWMM_for_sensitivity_analysis(verbose=True)
    assert analysis._system.compilation_successful, "TRITON-SWMM not compiled"


def test_prepare_scenarios(norfolk_frontier_all_configs_analysis_cached):
    analysis = norfolk_frontier_all_configs_analysis_cached
    analysis.sensitivity.prepare_scenarios_in_each_subanalysis(
        concurrent=True, verbose=True
    )
    assert analysis.sensitivity.all_scenarios_created


def test_run_all_sims(norfolk_frontier_all_configs_analysis_cached):
    analysis = norfolk_frontier_all_configs_analysis_cached
    analysis.sensitivity.run_all_sims(
        pickup_where_leftoff=False, concurrent=False, verbose=True, which="TRITON"
    )
    assert analysis.sensitivity.all_sims_run == True
    if not (
        analysis.sensitivity.all_TRITON_timeseries_processed
        and analysis.sensitivity.all_SWMM_timeseries_processed
    ):
        unprcsd_triton = "\n".join(
            analysis.sensitivity.TRITON_time_series_not_processed
        )
        unprcsd_swmm = "\n".join(analysis.sensitivity.SWMM_time_series_not_processed)
        pytest.fail(
            "Processing TRITON and SWMM time series failed.\n"
            f"Unprocessed TRITON: {unprcsd_triton}\n"
            f"Unprocessed SWMM: {unprcsd_swmm}"
        )


def test_consolidate_outputs(norfolk_frontier_all_configs_analysis_cached):
    analysis = norfolk_frontier_all_configs_analysis_cached
    analysis.sensitivity.consolidate_outputs(which="TRITON")
    assert analysis.log.TRITON_analysis_summary_created.get() == True
    # analysis.sensitivity.consolidate_SWMM_outputs_for_analysis()
    # assert analysis.log.SWMM_node_analysis_summary_created.get() == True
    # assert analysis.log.SWMM_link_analysis_summary_created.get() == True
