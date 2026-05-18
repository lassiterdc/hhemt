"""
Sensitivity Analysis Tests: TRITON-only and SWMM-only Models (synthetic tier).

Mirror of test_PC_06 using synth fixtures. Tests unified n_omp_threads threading
control across model types in sensitivity analysis workflows.
"""

from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
]

_SYNTH_SENSITIVITY_REPORT_CONFIG = (
    Path(__file__).parent / "fixtures" / "synthetic_model" / "report_config_synth_sensitivity.yaml"
)


@pytest.mark.slow
def test_sensitivity_analysis_swmm_only_execution(synth_sensitivity_swmm_only):
    analysis = synth_sensitivity_swmm_only

    analysis.run(report_config=_SYNTH_SENSITIVITY_REPORT_CONFIG)

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)


@pytest.mark.slow
def test_sensitivity_analysis_triton_only_execution(synth_sensitivity_triton_only):
    analysis = synth_sensitivity_triton_only

    analysis.run(report_config=_SYNTH_SENSITIVITY_REPORT_CONFIG)

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)
