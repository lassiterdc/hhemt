"""Regression smoke against real Norfolk data.

Detailed assertions live in test_synth_06_sensitivity_analysis_triton_and_swmm_only.py.
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
