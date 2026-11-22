"""
Sensitivity Analysis Tests: TRITON-only and SWMM-only Models (synthetic tier).

Mirror of test_PC_06 using synth fixtures. Tests unified n_omp_threads threading
control across model types in sensitivity analysis workflows.
"""

from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut
from hhemt.consolidate_workflow import validate_resource_usage

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    # Both tests execute real triton-only / swmm-only sensitivity workflows.
    # Skips without cmake+mpic++; HARD-FAILS under HHEMT_REQUIRE_COMPILE_TIER=1.
    pytest.mark.usefixtures("tritonswmm_cpu_compiled"),
]

_SYNTH_SENSITIVITY_REPORT_CONFIG = (
    Path(__file__).parent / "fixtures" / "synthetic_model" / "report_config_synth_sensitivity.yaml"
)


@pytest.mark.slow
def test_sensitivity_analysis_swmm_only_execution(synth_sensitivity_swmm_only):
    analysis = synth_sensitivity_swmm_only

    analysis.run(report_config=_SYNTH_SENSITIVITY_REPORT_CONFIG)

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)

    # The resource check is part of the assertion above; this names its OpenMP-thread arm over the SWMM rows. A SWMM
    # row measures `swmm_threads` bounded by the member's OpenMP count (default 1), never the OpenMP count itself, so
    # a member of n_omp_threads 3 or 2 reports one thread. The rows are required to exist and to have been measured,
    # because an empty population would pass the mismatch assertion without examining anything.
    df_status = analysis.df_status
    swmm_rows = df_status[df_status["model_type"] == "swmm"]
    assert len(swmm_rows) > 0, "no SWMM rows in the sensitivity df_status"
    assert swmm_rows["actual_omp_threads"].notna().all(), "a SWMM row has no measured thread count"
    _passed, issues = validate_resource_usage(analysis, df_status=df_status)
    omp_mismatches = [i for i in issues if i["resource"] == "OMP threads"]
    assert omp_mismatches == [], f"SWMM rows whose thread count differs from the one the run sets: {omp_mismatches}"
