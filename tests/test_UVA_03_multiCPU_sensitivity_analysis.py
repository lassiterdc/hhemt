import os
import pytest
import socket
import subprocess
import time
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils import on_UVA_HPC

pytestmark = pytest.mark.skipif(not on_UVA_HPC(), reason="Only runs on UVA HPC")

# ijob \
#   -A ***REMOVED*** \
#   -p interactive \
#   --time=08:00:00 \
#   -N 1 \
#  --cpus-per-task=1 \
#  --ntasks-per-node=24

# module purge
# module load gompi/14.2.0_5.0.7 miniforge
# source activate triton_swmm_toolkit
# export PYTHONNOUSERSITE=1

# /home/***REMOVED***/.conda/envs/triton_swmm_toolkit/bin/python


def test_consolidated_workflow_with_system_inputs_and_compilation():
    nrflk_multisim_ensemble = tst.retreive_norfolk_UVA_sensitivtiy_CPU(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    system = nrflk_multisim_ensemble.system
    sensitivity = analysis.sensitivity
    # ===== Phase 2: Submit the three-phase workflow with dependencies =====
    final_job_id = sensitivity.run_sensitivity_analysis_as_batch_job(
        process_system_level_inputs=True,
        overwrite_system_inputs=False,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=False,
        # ensemble run stuff
        prepare_scenarios=True,
        overwrite_scenario=False,
        rerun_swmm_hydro_if_outputs_exist=False,
        process_timeseries=True,
        which="TRITON",
        clear_raw_outputs=True,
        overwrite_if_exist=False,
        compression_level=5,
        pickup_where_leftoff=True,
        # other
        verbose=True,
    )

    # Verify jobs were submitted
    assert final_job_id is not None, "Final job ID should be returned from submission"
    assert final_job_id.isdigit(), "Job ID should be numeric"

    # ===== Phase 3: Wait for final job completion =====
    # Poll squeue until the final job (consolidation) completes
    max_wait_time = 3600  # 1 hour timeout
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            pytest.fail(
                f"Job {final_job_id} did not complete within {max_wait_time} seconds"
            )

        result = subprocess.run(
            ["squeue", "-j", final_job_id], capture_output=True, text=True
        )
        # Check if job_id appears in the output (job still running)
        if final_job_id not in result.stdout:
            break
        time.sleep(5)  # Check every 5 seconds

    # ===== Phase 4: Verify all phases completed successfully =====
    # Update analysis log to reflect current state
    sensitivity._update_master_analysis_log()
    analysis._update_log()

    # Verify Phase 1 outputs (system inputs and compilation)
    dem_file = system.sys_paths.dem_processed
    assert dem_file.exists(), "DEM file should be created"

    mannings_file = system.sys_paths.mannings_processed
    assert mannings_file.exists(), "Mannings file should be created"

    assert sensitivity.compilation_successful, "TRITON-SWMM should be compiled"

    # Verify Phase 2 outputs (simulations ran)
    assert sensitivity.all_scenarios_created, "All scenarios should be created"
    assert sensitivity.all_sims_run, "All simulations should run"

    if sensitivity.all_sims_run != True:
        sims_not_run = "\n".join(sensitivity.scenarios_not_run)
        pytest.fail(
            f"Running TRITONSWMM ensemble failed. Scenarios not run: \n{sims_not_run}"
        )

    assert (
        sensitivity.all_TRITON_timeseries_processed
    ), "All TRITON timeseries should be processed"

    assert (
        analysis.TRITON_analysis_summary_created
    ), "TRITON analysis summary should be created"

    triton_output = analysis.analysis_paths.output_triton_summary
    assert triton_output.exists(), "TRITON consolidated output should exist"
