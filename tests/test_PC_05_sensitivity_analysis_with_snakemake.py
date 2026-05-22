"""Regression smoke against real Norfolk data.

Detailed assertions live in test_synth_05_sensitivity_analysis_with_snakemake.py.
"""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
]


@pytest.mark.slow
def test_snakemake_sensitivity_workflow_execution(norfolk_sensitivity_analysis):
    """
    Test Snakemake sensitivity analysis workflow execution.

    Validates that:
    1. submit_workflow() returns success
    2. Setup phase completes for each sub-analysis
    3. All simulations execute without errors
    4. Sub-analysis consolidation completes
    5. Master consolidation completes
    6. Final sensitivity analysis summaries are generated
    """
    analysis = norfolk_sensitivity_analysis

    which = "both"

    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=True,
        prepare_scenarios=True,
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which=which,
        override_clear_raw="all",
        compression_level=5,
        pickup_where_leftoff=False,
        verbose=True,
    )

    assert result["success"], f"Workflow submission failed: {result.get('message', '')}"

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)
