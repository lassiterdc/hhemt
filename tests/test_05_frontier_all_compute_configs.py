import os
import pytest
import socket
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst

pytestmark = pytest.mark.skipif(
    "frontier" not in socket.gethostname(), reason="Only runs on Frontier HPC"
)


def test_retrieve_test():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=True
    )


def test_compile():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=False
    )
    analysis = nrflk_multiconfig.system.analysis
    analysis.compile_TRITON_SWMM(recompile_if_already_done_successfully=True)
    assert analysis.compilation_successful == True


def test_prepare_scenarios():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=False
    )
    analysis = nrflk_multiconfig.system.analysis
    analysis.sensitivity.prepare_scenarios_in_each_subanalysis(concurrent=True)
    assert analysis.log.all_scenarios_created.get() == True


def test_run_all_sims():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=False
    )
    analysis = nrflk_multiconfig.system.analysis
    analysis.sensitivity.run_all_sims(pickup_where_leftoff=False, concurrent=False)
    assert analysis.log.all_sims_run.get() == True
    success_processing = (
        analysis.log.all_TRITON_timeseries_processed.get()
        and analysis.log.all_SWMM_timeseries_processed.get()
    )
    if not success_processing:
        analysis.log.print()
        pytest.fail(f"Processing TRITON and SWMM time series failed.")


def test_consolidate_outputs():
    nrflk_multiconfig = tst.retreive_norfolk_frontier_all_configs(
        start_from_scratch=False
    )
    analysis = nrflk_multiconfig.system.analysis
    analysis.sensitivity.consolidate_TRITON_outputs_for_analysis()
    assert analysis.log.TRITON_analysis_summary_created.get() == True
    analysis.sensitivity.consolidate_SWMM_outputs_for_analysis()
    assert analysis.log.SWMM_node_analysis_summary_created.get() == True
    assert analysis.log.SWMM_link_analysis_summary_created.get() == True
