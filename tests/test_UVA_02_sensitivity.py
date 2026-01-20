import os
import pytest
import socket
import subprocess
import time
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils import on_UVA_HPC

pytestmark = pytest.mark.skipif(not on_UVA_HPC(), reason="Only runs on UVA HPC")


def test_consolidated_workflow_with_system_inputs_and_compilation():
    """
    Test the consolidated SLURM workflow that handles:
    1. Process system-level inputs (DEM, Mannings)
    2. Compile TRITON-SWMM
    3. Run ensemble simulations
    4. Consolidate outputs

    This test verifies the complete three-phase workflow in a single batch job.
    """
    nrflk_multisim_ensemble = tst.retreive_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    system = nrflk_multisim_ensemble.system

    # ===== Phase 1: Generate consolidated workflow script =====
    # This should create a heterogeneous SLURM job with three phases
    script_path = analysis.generate_consolidated_SLURM_workflow_script(
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=False,
        prepare_scenarios=True,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=True,
        overwrite_if_exist=False,
        compression_level=5,
        pickup_where_leftoff=False,
        overwrite_scenario=False,
        rerun_swmm_hydro_if_outputs_exist=False,
        consolidate_outputs=True,
        verbose=True,
    )

    # Verify script was created
    assert script_path.exists(), "Consolidated workflow script was not created"
    assert script_path.suffix == ".sh", "Script should be a shell script"

    # Verify script contains heterogeneous job directives
    script_content = script_path.read_text()
    assert (
        "#SBATCH --heterogeneous" in script_content
    ), "Script should contain heterogeneous job directive"
    assert (
        "#SBATCH hetjob" in script_content
    ), "Script should contain hetjob separators between phases"

    # Verify Phase 1 (Setup) is in the script
    assert (
        "setup_workflow" in script_content
    ), "Script should contain setup_workflow command"
    assert (
        "--process-system-inputs" in script_content
    ), "Script should process system inputs"
    assert (
        "--compile-triton-swmm" in script_content
    ), "Script should compile TRITON-SWMM"

    # Verify Phase 2 (Ensemble) is in the script
    assert (
        "run_single_simulation" in script_content
    ), "Script should contain run_single_simulation command"
    assert "--array=" in script_content, "Script should contain job array directive"
    assert "--prepare-scenario" in script_content, "Script should prepare scenarios"
    assert "--process-timeseries" in script_content, "Script should process timeseries"

    # Verify Phase 3 (Consolidation) is in the script
    assert (
        "consolidate_workflow" in script_content
    ), "Script should contain consolidate_workflow command"
    assert (
        "--consolidate-outputs" in script_content
    ), "Script should consolidate outputs"

    # ===== Phase 2: Submit the consolidated workflow =====
    script_path, job_id = analysis.submit_SLURM_job_array(
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=False,
        prepare_scenarios=True,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=True,
        overwrite_if_exist=False,
        compression_level=5,
        pickup_where_leftoff=False,
        overwrite_scenario=False,
        rerun_swmm_hydro_if_outputs_exist=False,
        consolidate_outputs=True,
        verbose=True,
    )

    # Verify job was submitted
    assert script_path.exists(), "Job script should exist after submission"
    assert job_id is not None, "Job ID should be returned from submission"
    assert job_id.isdigit(), "Job ID should be numeric"

    # ===== Phase 3: Wait for job completion =====
    # Poll squeue until job completes
    max_wait_time = 3600  # 1 hour timeout
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            pytest.fail(f"Job {job_id} did not complete within {max_wait_time} seconds")

        result = subprocess.run(
            ["squeue", "-j", job_id], capture_output=True, text=True
        )
        # Check if job_id appears in the output (job still running)
        if job_id not in result.stdout:
            break
        time.sleep(5)  # Check every 5 seconds

    # ===== Phase 4: Verify all phases completed successfully =====
    # Update analysis log to reflect current state
    analysis._update_log()

    # Verify Phase 1 outputs (system inputs and compilation)
    dem_file = system.sys_paths.dem_processed
    assert dem_file.exists(), "DEM file should be created in Phase 1"

    mannings_file = system.sys_paths.mannings_processed
    assert mannings_file.exists(), "Mannings file should be created in Phase 1"

    assert (
        analysis.compilation_successful
    ), "TRITON-SWMM should be compiled successfully in Phase 1"

    # Verify Phase 2 outputs (simulations ran)
    assert (
        analysis.log.all_scenarios_created.get() == True
    ), "All scenarios should be created in Phase 2"
    assert (
        analysis.log.all_sims_run.get() == True
    ), "All simulations should run in Phase 2"

    if analysis.log.all_sims_run.get() != True:
        sims_not_run = "\n".join(analysis.scenarios_not_run)
        pytest.fail(
            f"Running TRITONSWMM ensemble failed. Scenarios not run: \n{sims_not_run}"
        )

    # Verify timeseries processing in Phase 2
    assert (
        analysis.log.all_TRITON_timeseries_processed.get() == True
    ), "All TRITON timeseries should be processed in Phase 2"
    assert (
        analysis.log.all_SWMM_timeseries_processed.get() == True
    ), "All SWMM timeseries should be processed in Phase 2"

    # Verify Phase 3 outputs (consolidation)
    assert (
        analysis.TRITON_analysis_summary_created
    ), "TRITON analysis summary should be created in Phase 3"
    assert (
        analysis.SWMM_node_analysis_summary_created
    ), "SWMM node analysis summary should be created in Phase 3"
    assert (
        analysis.SWMM_link_analysis_summary_created
    ), "SWMM link analysis summary should be created in Phase 3"

    # Verify consolidated output files exist
    triton_output = analysis.analysis_paths.output_triton_summary
    assert triton_output.exists(), "TRITON consolidated output should exist"

    swmm_node_output = analysis.analysis_paths.output_swmm_node_summary
    assert swmm_node_output.exists(), "SWMM node consolidated output should exist"

    swmm_link_output = analysis.analysis_paths.output_swmm_links_summary
    assert swmm_link_output.exists(), "SWMM link consolidated output should exist"
