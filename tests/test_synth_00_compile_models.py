"""Synthetic-model compile tier. Mirror of test_PC_00 using synth fixtures."""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_create_dem_for_TRITON(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    analysis._system.create_dem_for_TRITON()
    rds = analysis._system.processed_dem_rds
    # synth fixture iter-8 narrowed n_cols 20→16 (cache.py:43); test assertion
    # was missed in that change and the failure was masked by the pre-Phase-5.5
    # CompilationError fired earlier in prepare_scenario.
    assert rds.shape == (1, 30, 16)  # type: ignore


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_create_mannings_file_for_TRITON(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    analysis._system.create_mannings_file_for_TRITON()
    rds = analysis._system.mannings_rds
    assert rds.shape == (1, 30, 16)  # type: ignore


def test_compile_swmm(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    analysis._system.compile_SWMM(
        recompile_if_already_done_successfully=True,
        redownload_swmm_if_exists=True,
    )
    tst_ut.assert_swmm_compiled(analysis)


def test_compile_tritonswmm(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    analysis._system.compile_TRITON_SWMM(
        recompile_if_already_done_successfully=True,
        redownload_triton_swmm_if_exists=True,
        verbose=True,
    )
    tst_ut.assert_tritonswmm_compiled(analysis)


def test_compile_triton_only(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    analysis._system.compile_TRITON_only(
        recompile_if_already_done_successfully=True, verbose=True
    )
    tst_ut.assert_triton_compiled(analysis)
