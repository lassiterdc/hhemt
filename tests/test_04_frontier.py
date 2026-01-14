import os
import pytest
import socket
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst

pytestmark = pytest.mark.skipif(
    "frontier" not in socket.gethostname(), reason="Only runs on Frontier HPC"
)


def test_run_multisim_concurrently():
    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    analysis.compile_TRITON_SWMM(recompile_if_already_done_successfully=True)
    analysis.prepare_all_scenarios()
    launch_functions = analysis._create_launchable_sims(
        pickup_where_leftoff=False, verbose=True
    )
    analysis.run_simulations_concurrently_on_desktop(
        launch_functions, use_gpu=False, total_gpus_available=0, verbose=True
    )
    assert analysis.log.all_sims_run.get() == True
