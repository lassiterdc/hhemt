"""Synthetic-model compile tier. Mirror of test_PC_00 using synth fixtures."""

import pytest

import tests.utils_for_testing as tst_ut

# NO module-level scheduler gate, deliberately. This module carried
# `skipif(is_scheduler_context(), reason="Only runs on non-HPC systems.")` from its
# first commit (4e4da31d), inherited wholesale from the `test_PC_*` convention this
# module's docstring names. That convention was one half of a two-sided routing
# scheme whose other half -- the `test_UVA_*` / `test_frontier_*` / `test_PILOT_*`
# platform tier -- has since been deleted, so the gate was routing away from a tier
# that no longer exists.
#
# It was not free: it skips the WHOLE module, and because a module-level skipif is
# evaluated BEFORE fixture setup, it also pre-empted `tritonswmm_cpu_compiled`'s own
# `require_compile_tier()` hard-fail. A warm step running inside a SLURM allocation
# therefore compiled nothing and reported `5 skipped` at exit 0.
#
# What protects this module is CAPABILITY gating, which is purpose-built for the
# real hazard and is live: `tritonswmm_cpu_compiled` skips when cmake+mpic++ are
# absent and hard-fails under HHEMT_REQUIRE_COMPILE_TIER=1, and `test_compile_swmm`
# is deliberately ungated because SWMM needs only cmake and a C compiler (see the
# comment above the TRITON compiles below). Nothing here reads the environment:
# `system.py` carries zero `os.environ` / `SLURM_JOB_ID` / `in_slurm` references, so
# these compiles produce the same artifact inside an allocation as outside one.
#
# Scoped to THIS module at the time of writing; the suite-wide removal superseded
# that scoping. The dead-rationale finding was established for this module's gate
# first, and was subsequently found to hold for the other tier-routing gates too,
# which were removed with it. The safety guards on the two run-proof modules are a
# DIFFERENT population and survive as `tst_ut.on_scheduler_node()`.


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


@pytest.mark.compile_tier
def test_compile_swmm(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    analysis._system.compile_SWMM(
        recompile_if_already_done_successfully=True,
        redownload_swmm_if_exists=True,
    )
    tst_ut.assert_swmm_compiled(analysis)


# The two TRITON compiles hard-require cmake + mpic++ (TRITON's machine.cmake
# FORCE-sets mpic++ and main.cpp #includes mpi.h). `tritonswmm_cpu_compiled` is
# the capability gate: it SKIPS when the toolchain is absent (bare ubuntu-latest
# CI runner) and HARD-FAILS under HHEMT_REQUIRE_COMPILE_TIER=1 (compile-tests.yml),
# so a real compile regression cannot hide behind the skip. `test_compile_swmm`
# below is deliberately NOT gated -- SWMM needs only cmake + a C compiler and
# compiles fine on the bare runner, so that coverage is preserved.
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_compile_tritonswmm(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    analysis._system.compile_TRITON_SWMM(
        recompile_if_already_done_successfully=True,
        redownload_triton_swmm_if_exists=True,
        verbose=True,
    )
    tst_ut.assert_tritonswmm_compiled(analysis)


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_compile_triton_only(synth_all_models_analysis):
    analysis = synth_all_models_analysis
    analysis._system.compile_TRITON_only(recompile_if_already_done_successfully=True, verbose=True)
    tst_ut.assert_triton_compiled(analysis)
