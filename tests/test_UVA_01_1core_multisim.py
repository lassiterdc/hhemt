import os
import pytest
import socket
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils_for_testing import on_UVA_HPC

pytestmark = pytest.mark.skipif(not on_UVA_HPC(), reason="Only runs on UVA HPC")

# ijob \
#   -A ***REMOVED*** \
#   -p standard \
#   --time=08:00:00 \
#   -N 1 \
#  --cpus-per-task=1 \
#  --ntasks-per-node=96

# ijob \
#   -A ***REMOVED*** \
#   -p interactive \
#   --time=08:00:00 \
#   -N 1 \
#  --cpus-per-task=1 \
#  --ntasks-per-node=24

#   --gres=gpu:1 \

# module purge
# module load gompi/14.2.0_5.0.7 miniforge
# conda activate triton_swmm_toolkit
# export PYTHONNOUSERSITE=1


def test_load_system_and_analysis():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=True
    )
    assert (
        nrflk_multisim_ensemble.system.analysis.analysis_paths.simulation_directory.exists()
    )


def test_create_dem_for_TRITON():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=False
    )
    nrflk_multisim_ensemble.system.create_dem_for_TRITON()
    rds = nrflk_multisim_ensemble.system.processed_dem_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_create_mannings_file_for_TRITON():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=False
    )
    nrflk_multisim_ensemble.system.create_mannings_file_for_TRITON()
    rds = nrflk_multisim_ensemble.system.open_processed_mannings_as_rds()
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_compile_TRITONSWMM_for_cpu_sims():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=False
    )
    nrflk_multisim_ensemble.system.compile_TRITON_SWMM(
        verbose=True,
        redownload_triton_swmm_if_exists=True,
        recompile_if_already_done_successfully=True,
    )
    assert nrflk_multisim_ensemble.system.compilation_successful


def test_prepare_scenarios():
    nrflk_multisim_ensemble = tst.retrieve_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=False
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    prepare_scenario_launchers = analysis.retrieve_prepare_scenario_launchers(
        overwrite_scenario=True, verbose=True
    )
    analysis.run_python_functions_concurrently(
        prepare_scenario_launchers,
        verbose=True,
    )
    if analysis.log.all_scenarios_created.get() != True:
        scens_not_created = "\n".join(analysis.scenarios_not_created)
        pytest.fail(
            f"Processing TRITON and SWMM time series failed.Scenarios not created: \n{scens_not_created}"
        )
