import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


# SYSTEM TESTS
def test_create_dem_for_TRITON(norfolk_single_sim_analysis):
    analysis = norfolk_single_sim_analysis
    analysis._system.create_dem_for_TRITON()
    rds = analysis._system.processed_dem_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_create_mannings_file_for_TRITON(norfolk_single_sim_analysis):
    analysis = norfolk_single_sim_analysis
    analysis._system.create_mannings_file_for_TRITON()
    rds = analysis._system.mannings_rds
    assert rds.shape == (1, 537, 551)  # type: ignore


def test_compile_swmm(norfolk_single_sim_analysis):
    analysis = norfolk_single_sim_analysis
    analysis._system.compile_SWMM(
        recompile_if_already_done_successfully=True,
        redownload_swmm_if_exists=True,
    )
    tst_ut.assert_swmm_compiled(analysis)


def test_compile_tritonswmm(norfolk_single_sim_analysis):
    analysis = norfolk_single_sim_analysis
    analysis._system.compile_TRITON_SWMM(
        recompile_if_already_done_successfully=True,
        redownload_triton_swmm_if_exists=True,
    )
    tst_ut.assert_tritonswmm_compiled(analysis)


def test_compile_triton_only(norfolk_single_sim_analysis):
    analysis = norfolk_single_sim_analysis
    analysis._system.compile_TRITON_only(
        recompile_if_already_done_successfully=True,
    )
    tst_ut.assert_triton_compiled(analysis)
