"""Synthetic-tier replacement for the RETIRED ``tests/test_analysis_test_end_to_end.py``.

Covers the ``analysis.test()`` properties the real-Norfolk tier was the sole
evaluator of: the multi-candidate branch of ``_select_test_representatives``
(reachable only from a sensitivity master), the ``_capture_binary_provenance``
no-``triton.exe`` early return (reachable only from a SWMM-only representative),
and end-to-end ``.test()`` across the model-toggle axis.

Deliberately carries NO ``skipif(on_scheduler_node())``. Its sibling run-proof
``test_synth_09_from_doi_run_proof.py`` does, which is why that module cannot
substitute for this one on a SLURM harness: ``on_scheduler_node()`` is True
whenever ``SLURM_JOB_ID`` is set, so the sibling never runs there.
"""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    pytest.mark.slow,
]


def _assert_analysis_test_completes(analysis, *, min_groups: int = 1):
    """Assertion set of the real-data smoke, plus the two selection properties
    that smoke left implicit.

    ``min_groups`` is a floor, not an equality: the count is derived from the
    sensitivity CSV's distinct compute-config rows, and a floor of 2 pins the
    property that matters (the multi-candidate branch was entered) without
    asserting a row count this file does not own.
    """
    result = analysis.test(execution_mode="local", verbose=False)
    assert (analysis.analysis_paths.analysis_dir / "_test").exists()
    assert result.analyses, "analysis.test() produced no _test sub-analyses"
    assert len(result.representatives) >= min_groups, (
        f"expected >= {min_groups} representative group(s), got "
        f"{len(result.representatives)}; a grouping-key regression collapses "
        "distinct compute-configs into one group"
    )
    keys = [r.key for r in result.representatives]
    assert len(set(keys)) == len(keys), f"duplicate representative keys: {keys}"
    for sub_result in result.analyses:
        tst_ut.assert_analysis_workflow_completed_successfully(sub_result.analysis)


def test_synth_analysis_test_all_models(synth_all_models_analysis):
    _assert_analysis_test_completes(synth_all_models_analysis)


def test_synth_analysis_test_multi_sim(synth_multi_sim_analysis):
    _assert_analysis_test_completes(synth_multi_sim_analysis)


def test_synth_analysis_test_sensitivity(synth_sensitivity_analysis):
    _assert_analysis_test_completes(synth_sensitivity_analysis, min_groups=2)


def test_synth_analysis_test_sensitivity_triton_only(synth_sensitivity_triton_only):
    _assert_analysis_test_completes(synth_sensitivity_triton_only, min_groups=2)


def test_synth_analysis_test_sensitivity_swmm_only(synth_sensitivity_swmm_only):
    _assert_analysis_test_completes(synth_sensitivity_swmm_only, min_groups=2)
