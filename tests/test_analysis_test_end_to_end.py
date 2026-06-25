"""End-to-end smoke of ``analysis.test()`` against the real Norfolk case study.

This file is the replacement for the retired ``test_PC_*`` tier. Instead of the
old staged per-phase real-data tests, each function below builds the real
Norfolk analysis (via the surviving Local ``norfolk_*`` fixtures) and asserts
``analysis.test()`` runs the strict ``_test/`` subset end to end
(compile -> run -> process -> consolidate -> report) for every representative
(model-toggle x compilation-backend x partition x compute-config) group.

Detailed per-stage assertions live in the synthetic-model developer-CI tier
(``test_synth_*``); these real-data smokes only confirm the same design works on
a real case study. They are ``@pytest.mark.slow`` and skipped on HPC scheduler
contexts (the platform-gated tier they replace is gone).
"""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
    pytest.mark.slow,
]


def _assert_analysis_test_completes(analysis):
    """Run ``analysis.test()`` and assert the ``_test/`` subtree was produced and
    every representative sub-analysis completed its workflow successfully."""
    result = analysis.test()
    assert (analysis.analysis_paths.analysis_dir / "_test").exists()
    assert result.subanalyses, "analysis.test() produced no _test sub-analyses"
    for _analysis_id, sub in result.subanalyses:
        tst_ut.assert_analysis_workflow_completed_successfully(sub)


def test_analysis_test_single_sim(norfolk_single_sim_analysis):
    _assert_analysis_test_completes(norfolk_single_sim_analysis)


def test_analysis_test_multi_sim(norfolk_multi_sim_analysis):
    _assert_analysis_test_completes(norfolk_multi_sim_analysis)


def test_analysis_test_sensitivity(norfolk_sensitivity_analysis):
    _assert_analysis_test_completes(norfolk_sensitivity_analysis)


def test_analysis_test_sensitivity_triton_only(norfolk_sensitivity_triton_only):
    _assert_analysis_test_completes(norfolk_sensitivity_triton_only)


def test_analysis_test_sensitivity_swmm_only(norfolk_sensitivity_swmm_only):
    _assert_analysis_test_completes(norfolk_sensitivity_swmm_only)
