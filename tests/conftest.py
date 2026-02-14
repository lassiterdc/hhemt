import pytest

import tests.fixtures.test_case_catalog as cases

# import tests.fixtures.test_case_catalog as cases


@pytest.fixture
def norfolk_single_sim_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_single_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_multi_sim_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_multi_sim_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_analysis():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_analysis_cached():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_triton_only():
    """TRITON-only sensitivity analysis varying n_omp_threads."""
    case = (
        cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_triton_only(
            start_from_scratch=True
        )
    )
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_swmm_only():
    """SWMM-only sensitivity analysis varying n_omp_threads."""
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_swmm_only(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_multisim_analysis():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_multisim_analysis_cached():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_cpu_serial_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_multisim_gpu_analysis():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_gpu_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_sensitivity_analysis():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_sensitivity_minimal(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_frontier_sensitivity_analysis_cached():
    case = cases.Frontier_TestCases.retrieve_norfolk_frontier_sensitivity_minimal(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_multisim_analysis():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_multisim_analysis_cached():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_sensitivity_analysis():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_sensitivity_minimal(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_sensitivity_analysis_cached():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_sensitivity_minimal(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_uva_sensitivity_full_ensemble_analysis():
    case = cases.UVA_TestCases.retrieve_norfolk_UVA_sensitivity_CPU_full_ensemble_short_sims(
        start_from_scratch=True
    )
    return case.analysis


# ========== Multi-Model Test Fixtures ==========


@pytest.fixture
def norfolk_triton_only_analysis():
    """TRITON-only analysis (no SWMM coupling)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_only_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_triton_only_analysis_cached():
    """TRITON-only analysis (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_only_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_swmm_only_analysis():
    """SWMM-only analysis (standalone EPA SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_swmm_only_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_swmm_only_analysis_cached():
    """SWMM-only analysis (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_swmm_only_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_all_models_analysis():
    """Analysis with all models enabled (TRITON, TRITON-SWMM, SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_all_models_analysis_cached():
    """Analysis with all models (cached - for faster iteration)."""
    case = cases.Local_TestCases.retrieve_norfolk_all_models_test_case(
        start_from_scratch=False
    )
    return case.analysis


@pytest.fixture
def norfolk_triton_and_tritonswmm_analysis():
    """Analysis with TRITON and TRITON-SWMM (no standalone SWMM)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_and_tritonswmm_test_case(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_triton_and_tritonswmm_analysis_cached():
    """Analysis with TRITON and TRITON-SWMM (cached)."""
    case = cases.Local_TestCases.retrieve_norfolk_triton_and_tritonswmm_test_case(
        start_from_scratch=False
    )
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
        case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
            start_from_scratch=True
        )
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
        case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
            start_from_scratch=False
        )
    else:
        pytest.fail(f"Unsupported platform in pilot: {platform_pilot}")

    return case.analysis
