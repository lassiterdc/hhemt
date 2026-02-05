import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


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
