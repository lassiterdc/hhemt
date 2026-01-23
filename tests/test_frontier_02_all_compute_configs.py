import pytest
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils_for_testing import on_frontier

pytestmark = pytest.mark.skipif(not on_frontier(), reason="Only runs on Frontier HPC")
# cd /lustre/orion/***REMOVED***/proj-shared/***REMOVED***/TRITON-SWMM_toolkit
# salloc -A ***REMOVED*** -p batch -t 0-02:00:00 -N 2 --cpus-per-task=1 --ntasks-per-node=32 --gres=gpu:2 -q debug --mem=0
# conda activate triton_swmm_toolkit

# bash commands
# pgrep -l srun # lists all srun processes
# ps -o pid= --ppid $$ | xargs kill -9 # kills all srun processes


def test_retrieve_test():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=True
    )


def test_compile():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=False
    )
    analysis = nrflk_multiconfig.system.analysis
    analysis.sensitivity.compile_TRITON_SWMM_for_sensitivity_analysis(verbose=True)
    assert analysis._system.compilation_successful, "TRITON-SWMM not compiled"


def test_prepare_scenarios():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=False
    )
    analysis = nrflk_multiconfig.system.analysis
    analysis.sensitivity.prepare_scenarios_in_each_subanalysis(
        concurrent=True, verbose=True
    )
    assert analysis.sensitivity.all_scenarios_created


def test_run_all_sims():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=False
    )
    analysis = nrflk_multiconfig.system.analysis
    analysis.sensitivity.run_all_sims(
        pickup_where_leftoff=False, concurrent=False, verbose=True, which="TRITON"
    )
    assert analysis.sensitivity.all_sims_run == True
    success_processing = (
        analysis.sensitivity.all_TRITON_timeseries_processed
        and analysis.sensitivity.all_SWMM_timeseries_processed
    )
    if not success_processing:
        unprcsd_triton = "\n".join(
            analysis.sensitivity.TRITON_time_series_not_processed
        )
        unprcsd_swmm = "\n".join(analysis.sensitivity.SWMM_time_series_not_processed)
        pytest.fail(
            f"Processing TRITON and SWMM time series failed.\nUnprocessed TRITON: {unprcsd_triton}\nUnprocessed SWMM: {unprcsd_swmm}"
        )


def test_consolidate_outputs():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=False
    )
    analysis = nrflk_multiconfig.system.analysis
    analysis.sensitivity.consolidate_outputs(which="TRITON")
    assert analysis.log.TRITON_analysis_summary_created.get() == True
    # analysis.sensitivity.consolidate_SWMM_outputs_for_analysis()
    # assert analysis.log.SWMM_node_analysis_summary_created.get() == True
    # assert analysis.log.SWMM_link_analysis_summary_created.get() == True
