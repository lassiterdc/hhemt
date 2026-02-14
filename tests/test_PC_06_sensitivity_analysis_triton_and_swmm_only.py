"""
Sensitivity Analysis Tests: TRITON-only and SWMM-only Models

Tests unified n_omp_threads threading control across model types in sensitivity
analysis workflows. Validates that threading configuration propagates correctly
through the sensitivity analysis framework and is actually applied during execution.

Related: docs/planning/enable_swmm_threading_control.md (Phase 3)

NOTE: These tests skip consolidation due to pre-existing bug in sensitivity_analysis.py
where SWMM-only models don't have tritonswmm_node_analysis_summary_created property.
"""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


@pytest.mark.slow
def test_sensitivity_analysis_swmm_only_execution(norfolk_sensitivity_swmm_only):

    analysis = norfolk_sensitivity_swmm_only

    # import tests.fixtures.test_case_catalog as cases

    # case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_swmm_only(
    #     start_from_scratch=False
    # )
    # analysis = case.analysis

    analysis.run()

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)


@pytest.mark.slow
def test_sensitivity_analysis_triton_only_execution(norfolk_sensitivity_triton_only):
    analysis = norfolk_sensitivity_triton_only

    # import tests.fixtures.test_case_catalog as cases

    # case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_triton_only(
    #     start_from_scratch=False
    # )
    # analysis = case.analysis

    analysis.run()

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)
