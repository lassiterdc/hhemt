import pytest
import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(not tst_ut.on_UVA_HPC(), reason="Only runs on UVA HPC")

# ijob \
#   -A ***REMOVED*** \
#   -p standard \
#   --time=08:00:00 \
#   -N 1 \
#  --cpus-per-task=1 \
#  --ntasks-per-node=96

# ijob \
#   -A ***REMOVED*** \
#   -p interactive \
#   --time=08:00:00 \
#   -N 1 \
#  --cpus-per-task=1 \
#  --ntasks-per-node=24

#   --gres=gpu:1 \

# module purge
# module load gompi/14.2.0_5.0.7 miniforge
# source activate triton_swmm_toolkit
# export PYTHONNOUSERSITE=1


@pytest.mark.slow
def test_snakemake_sensitivity_workflow_execution(
    norfolk_uva_sensitivity_full_ensemble_analysis,
):
    """
    Test Snakemake sensitivity analysis workflow execution on UVA HPC with SLURM.

    Validates that:
    1. submit_workflow() returns success
    2. Setup phase completes for each sub-analysis
    3. All simulations execute without errors
    4. Sub-analysis consolidation completes
    5. Master consolidation completes
    6. Final sensitivity analysis summaries are generated
    """
    analysis = norfolk_uva_sensitivity_full_ensemble_analysis
    which = "both"

    # Submit the workflow using submit_workflow (not the old batch job method)
    result = analysis.submit_workflow(
        mode="slurm",  # Explicitly use SLURM mode
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=True,
        prepare_scenarios=True,
        overwrite_scenario=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which=which,
        clear_raw_outputs=True,
        overwrite_if_exist=True,
        compression_level=5,
        pickup_where_leftoff=False,
        verbose=True,
        wait_for_completion=True,
    )

    # Verify workflow submission was successful
    assert result["success"], f"Workflow submission failed: {result.get('message', '')}"
    assert result["mode"] == "slurm", "Should be running in SLURM mode"

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)
