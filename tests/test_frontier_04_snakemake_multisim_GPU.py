import pytest
import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    pytest.mark.skipif(
        not tst_ut.on_frontier(), reason="Only runs on Frontier HPC"
    ),
]

# cd /lustre/orion/***REMOVED***/proj-shared/***REMOVED***/TRITON-SWMM_toolkit
# salloc -A ***REMOVED*** -p batch -t 0-02:00:00 -N 2 --cpus-per-task=1 --ntasks-per-node=32 --gres=gpu:2 -q debug --mem=0
# conda activate triton_swmm_toolkit


@pytest.mark.slow
def test_snakemake_workflow_execution(norfolk_frontier_multisim_gpu_analysis):
    """
    Test Snakemake workflow execution on Frontier HPC with SLURM (2 simulations).

    Validates that:
    1. submit_workflow() returns success
    2. Setup phase completes
    3. Simulations execute without errors
    4. Scenarios are prepared correctly
    5. Simulations run successfully
    6. Analysis summaries are generated
    """
    analysis = norfolk_frontier_multisim_gpu_analysis
    which = "both"

    # Submit the workflow using submit_workflow
    # Note: On Frontier with 1_job_many_srun_tasks mode, this will use single_job submission
    result = analysis.submit_workflow(
        mode="auto",  # Auto-detect mode (will use single_job if configured)
        process_system_level_inputs=True,
        overwrite_system_inputs=False,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=False,
        prepare_scenarios=True,
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which=which,
        clear_raw_outputs=True,
        overwrite_outputs_if_already_created=True,
        compression_level=5,
        pickup_where_leftoff=False,
        wait_for_completion=True,
        verbose=True,
    )

    # Verify workflow submission was successful
    assert result["success"], f"Workflow submission failed: {result.get('message', '')}"
    assert result["mode"] in [
        "slurm",
        "single_job",
    ], f"Expected slurm or single_job mode, got {result['mode']}"

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)
