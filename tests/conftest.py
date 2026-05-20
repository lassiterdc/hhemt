import pytest

import tests.fixtures.test_case_catalog as cases

# import tests.fixtures.test_case_catalog as cases


@pytest.fixture
def norfolk_single_sim_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_single_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_multi_sim_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_multi_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_triton_only():
    """TRITON-only sensitivity analysis varying n_omp_threads."""
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_triton_only(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_swmm_only():
    """SWMM-only sensitivity analysis varying n_omp_threads."""
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_swmm_only(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_frontier_multisim_analysis():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_cpu_serial_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_frontier_multisim_analysis_cached():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_cpu_serial_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_frontier_multisim_gpu_analysis():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_gpu_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_frontier_sensitivity_analysis():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_sensitivity_minimal(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_frontier_sensitivity_analysis_cached():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_sensitivity_minimal(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_uva_multisim_analysis():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_multisim_1cpu_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_uva_multisim_analysis_cached():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_multisim_1cpu_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_uva_sensitivity_analysis():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_sensitivity_minimal(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_uva_sensitivity_analysis_cached():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_sensitivity_minimal(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_uva_sensitivity_full_ensemble_analysis():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_sensitivity_CPU_full_ensemble_short_sims(start_from_scratch=True)
    return case.analysis


# ========== Multi-Model Test Fixtures ==========


@pytest.fixture
def norfolk_triton_only_analysis():
    """TRITON-only analysis (no SWMM coupling)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_triton_only_analysis_cached():
    """TRITON-only analysis (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_swmm_only_analysis():
    """SWMM-only analysis (standalone EPA SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_swmm_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_swmm_only_analysis_cached():
    """SWMM-only analysis (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_swmm_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_all_models_analysis():
    """Analysis with all models enabled (TRITON, TRITON-SWMM, SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_all_models_analysis_cached():
    """Analysis with all models (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def norfolk_triton_and_tritonswmm_analysis():
    """Analysis with TRITON and TRITON-SWMM (no standalone SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_and_tritonswmm_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def norfolk_triton_and_tritonswmm_analysis_cached():
    """Analysis with TRITON and TRITON-SWMM (cached)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_and_tritonswmm_test_case(start_from_scratch=False)
    return case.analysis


# ========== Phase 6b.2.1: Unified Fixture API (Pilot - Local Only) ==========
# Proof-of-concept for parametrized fixtures before full platform expansion.
# Keeps all existing fixtures untouched during validation.


@pytest.fixture(params=[pytest.param("local", id="local")])
def platform_pilot(request):
    """Platform selection for parametrized fixtures (pilot: local only).

    This is a pilot implementation to validate the parametrization pattern
    before expanding to UVA and Frontier platforms.

    Future expansion (Phase 6b.2.2):
        @pytest.fixture(params=[
            pytest.param("local", id="local"),
            pytest.param("uva", marks=pytest.mark.skipif(...), id="uva"),
            pytest.param("frontier", marks=pytest.mark.skipif(...), id="frontier"),
        ])
    """
    return request.param


@pytest.fixture
def norfolk_multi_sim_unified(platform_pilot):
    """Multi-simulation analysis (unified API, pilot: local only).

    This fixture demonstrates the unified API pattern that will replace
    platform-specific fixtures once validated.

    Replaces (in future):
        - norfolk_multi_sim_analysis (local)
        - norfolk_uva_multisim_analysis (UVA)
        - norfolk_frontier_multisim_analysis (Frontier)

    Usage:
        def test_workflow(norfolk_multi_sim_unified):
            analysis = norfolk_multi_sim_unified
            # Test logic runs once per platform param
    """
    # Currently only supports local platform (pilot phase)
    if platform_pilot == "local":
        case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(start_from_scratch=True)
    else:
        pytest.fail(f"Unsupported platform in pilot: {platform_pilot}")

    return case.analysis


@pytest.fixture
def norfolk_multi_sim_unified_cached(platform_pilot):
    """Multi-simulation analysis (unified API, cached, pilot: local only).

    Cached variant of norfolk_multi_sim_unified for tests that don't need
    fresh setup each time.
    """
    if platform_pilot == "local":
        case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(start_from_scratch=False)
    else:
        pytest.fail(f"Unsupported platform in pilot: {platform_pilot}")

    return case.analysis


# ========== Synthetic Test Fixtures ==========


@pytest.fixture
def synth_all_models_analysis():
    case = cases.Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_all_models_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_all_models_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_multi_sim_analysis():
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_multi_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synthetic_multisim_builder():
    """Synthetic-tier SnakemakeWorkflowBuilder for at-most-once-guard unit tests.

    Yields ``analysis._workflow_builder`` from a fresh synth multisim
    analysis (start_from_scratch=True) so ``_status/`` starts empty and
    ``_status/_submitted/`` is writable. No simulations are executed —
    Phase 1 unit tests construct sentinel files directly and exercise the
    reconciliation guard's classification logic in isolation.

    The builder exposes ``analysis_paths``, ``_reconcile_inflight_submissions``,
    and ``_recover_inflight_via_comment`` — the exact surface the guard's
    test cases monkeypatch against.
    """
    case = cases.Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=True)
    return case.analysis._workflow_builder


@pytest.fixture
def synth_triton_only_analysis():
    case = cases.Local_TestCases.retrieve_synth_triton_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_triton_only_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_triton_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_swmm_only_analysis():
    case = cases.Local_TestCases.retrieve_synth_swmm_only_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_swmm_only_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_swmm_only_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_triton_and_tritonswmm_analysis():
    case = cases.Local_TestCases.retrieve_synth_triton_and_tritonswmm_test_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_triton_and_tritonswmm_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_triton_and_tritonswmm_test_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_sensitivity_analysis():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_analysis_cached():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(start_from_scratch=False)
    return case.analysis


@pytest.fixture
def synth_sensitivity_triton_only():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_triton_only(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_swmm_only():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_swmm_only(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_with_system_overlay():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_with_system_overlay(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_mutex_violation():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mutex_violation(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_invalid_overlay():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_invalid_overlay(start_from_scratch=True)
    return case.analysis


@pytest.fixture
def synth_sensitivity_legacy_gpu_hardware_override():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_legacy_gpu_hardware_override(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_typo_in_prefixed_column():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_typo_in_prefixed_column(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_with_system_gpu_hardware_override():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_with_system_gpu_hardware_override(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_all_analysis_prefixed():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_all_analysis_prefixed(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def synth_sensitivity_mixed_prefixed_columns():
    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mixed_prefixed_columns(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture(scope="session")
def synthetic_multisim_completed(tritonswmm_cpu_compiled):
    """Yield a synth multisim TRITONSWMM_analysis with sim outputs produced.

    Used by Phase 2 reprocess tests. Session-scoped: the first invocation
    in a pytest session runs the synth multisim through
    ``analysis.run(from_scratch=False, ...)`` if the analysis is not
    already at the ``e_consolidate_complete.flag`` state; subsequent
    invocations reuse the materialized analysis from the test-case cache.

    Stale ``.snakemake/locks/`` / ``.snakemake/incomplete/`` (and the
    reprocess-side ``.snakemake_reprocess/.snakemake/locks/`` /
    ``.snakemake_reprocess/.snakemake/incomplete/``) directories from
    prior interrupted runs are silently cleared before yielding.
    Follow-up: integrate this clear into the broader
    ``_pytest_uses_non_interactive_snakemake_lock_clear`` autouse fixture
    described by stipulation
    ``library/docs/stipulations/TRITON-SWMM_toolkit/snakemake fixture setup clears locks and incomplete.md``
    (cross-plan synth-test-isolation work; not yet implemented).
    """
    import shutil
    from pathlib import Path as _Path

    from tests.fixtures.test_case_catalog import Local_TestCases

    case = Local_TestCases.retrieve_synth_multi_sim_test_case(start_from_scratch=False)
    analysis = case.analysis
    analysis_dir = analysis.analysis_paths.analysis_dir

    # Clear stale lock + incomplete subtrees on both the run and reprocess
    # working dirs so a leftover state from an interrupted prior run does
    # not interfere with the fixture's analysis.run() (if it fires) or
    # with the test body's analysis.reprocess() invocations.
    for sub_root_name in (".snakemake", ".snakemake_reprocess"):
        sub_root = analysis_dir / sub_root_name
        if sub_root.exists():
            shutil.rmtree(sub_root / "locks", ignore_errors=True)
            shutil.rmtree(sub_root / "incomplete", ignore_errors=True)
            (sub_root / "log").mkdir(parents=True, exist_ok=True)

    # Ensure the analysis is in a "post-consolidate" state. If
    # ``e_consolidate_complete.flag`` is absent, run() once to materialize.
    consolidate_flag = analysis_dir / "_status" / "e_consolidate_complete.flag"
    if not consolidate_flag.exists():
        report_config = (
            _Path(__file__).resolve().parents[0].parent
            / "tests" / "configs" / "reports" / "synth_multisim_report_config.yaml"
        )
        analysis.run(
            from_scratch=False,
            report_config=report_config if report_config.exists() else None,
        )

    return analysis


@pytest.fixture(scope="session")
def tritonswmm_cpu_compiled():
    """Pre-compile TRITON-SWMM CPU once per test session for each test-case
    family used by the coupled-mode tests.

    Required by coupled-mode tests whose `prepare_scenario` gate at
    scenario.py:800-811 checks `_system.compilation_cpu_successful`.
    The property reads the in-memory system log; a fresh system init
    shows compile-not-yet-done even when a valid build artifact exists
    on disk. Each test-case family has its own `_software_root`, so the
    fixture iterates over the 3 families touched by the marked tests
    (synth_all_models, synth_multi_sim, norfolk_multi_sim).

    Process-safety note: this fixture writes to ~/.cache/.../_software/
    or test_data/.../triton/ and assumes no concurrent test sessions are
    running an actual compile against the same cache dir.
    """
    from tests.fixtures.test_case_catalog import Local_TestCases

    for retrieve in (
        Local_TestCases.retrieve_synth_all_models_test_case,
        Local_TestCases.retrieve_synth_multi_sim_test_case,
        Local_TestCases.retrieve_norfolk_multi_sim_test_case,
        Local_TestCases.retrieve_norfolk_single_sim_test_case,
    ):
        case = retrieve(start_from_scratch=False)
        case.analysis._system.compile_TRITON_SWMM(
            backends=["cpu"],
            recompile_if_already_done_successfully=False,
        )
        if case.analysis._system.cfg_system.toggle_swmm_model:
            case.analysis._system.compile_SWMM(
                recompile_if_already_done_successfully=False,
            )
