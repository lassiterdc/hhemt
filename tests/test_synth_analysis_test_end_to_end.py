"""Synthetic-tier replacement for the RETIRED ``tests/test_analysis_test_end_to_end.py``.

Runs ``analysis.test()`` end to end on the synthetic sensitivity case, which covers the multi-candidate branch of
``_select_test_representatives`` (reachable only from a sensitivity master) through a real run. The selection per
model-toggle combination, the ``_test/`` subset construction and both arms of ``_capture_binary_provenance`` (the
no-``triton.exe`` early return included) are checked without a run in
``tests/test_analysis_test_representatives_battery.py``; the four per-combination run tests that used to sit here
(all models, multi-sim, TRITON-only and SWMM-only sensitivity) were replaced by that battery.

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
    assert result.members, "analysis.test() produced no _test members"
    assert len(result.representatives) >= min_groups, (
        f"expected >= {min_groups} representative group(s), got "
        f"{len(result.representatives)}; a grouping-key regression collapses "
        "distinct compute-configs into one group"
    )
    keys = [r.key for r in result.representatives]
    assert len(set(keys)) == len(keys), f"duplicate representative keys: {keys}"
    for sub_result in result.members:
        tst_ut.assert_analysis_workflow_completed_successfully(sub_result.analysis)


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_synth_analysis_test_sensitivity(synth_sensitivity_analysis):
    _assert_analysis_test_completes(synth_sensitivity_analysis, min_groups=2)
