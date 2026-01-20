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
    Test the consolidated SLURM workflow with three separate scripts and dependencies:
    1. Phase 1: Process system-level inputs (DEM, Mannings) and compile TRITON-SWMM
    2. Phase 2: Run ensemble simulations (depends on Phase 1)
    3. Phase 3: Consolidate outputs (depends on Phase 2)

    This test verifies the complete three-phase workflow with SLURM job dependencies.
    """
    nrflk_multisim_ensemble = tst.retreive_norfolk_UVA_multisim_1cpu_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis
    system = nrflk_multisim_ensemble.system

    # ===== Phase 1: Generate three separate workflow scripts =====
    # Generate Phase 1 (Setup) script
    setup_script = analysis.generate_setup_workflow_script(
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=False,
        verbose=True,
    )
    assert setup_script.exists(), "Setup workflow script was not created"
    assert setup_script.suffix == ".sh", "Setup script should be a shell script"

    # Generate Phase 2 (Ensemble) script
    ensemble_script = analysis.generate_ensemble_simulations_script(
        prepare_scenarios=True,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=True,
        overwrite_if_exist=False,
        compression_level=5,
        pickup_where_leftoff=False,
        overwrite_scenario=False,
        rerun_swmm_hydro_if_outputs_exist=False,
        verbose=True,
    )
    assert ensemble_script.exists(), "Ensemble simulations script was not created"
    assert ensemble_script.suffix == ".sh", "Ensemble script should be a shell script"

    # Generate Phase 3 (Consolidation) script
    consolidation_script = analysis.generate_consolidation_workflow_script(
        consolidate_outputs=True,
        overwrite_if_exist=False,
        compression_level=5,
        verbose=True,
    )
    assert (
        consolidation_script.exists()
    ), "Consolidation workflow script was not created"
    assert (
        consolidation_script.suffix == ".sh"
    ), "Consolidation script should be a shell script"

    # Verify scripts contain correct commands
    setup_content = setup_script.read_text()
    assert (
        "setup_workflow" in setup_content
    ), "Setup script should contain setup_workflow command"
    assert (
        "--process-system-inputs" in setup_content
    ), "Setup script should process system inputs"
    assert (
        "--compile-triton-swmm" in setup_content
    ), "Setup script should compile TRITON-SWMM"

    ensemble_content = ensemble_script.read_text()
    assert (
        "run_single_simulation" in ensemble_content
    ), "Ensemble script should contain run_single_simulation"
    assert (
        "--array=" in ensemble_content
    ), "Ensemble script should contain job array directive"

    consolidation_content = consolidation_script.read_text()
    assert (
        "consolidate_workflow" in consolidation_content
    ), "Consolidation script should contain consolidate_workflow"
    assert (
        "--consolidate-outputs" in consolidation_content
    ), "Consolidation script should consolidate outputs"

    # ===== Phase 2: Submit the three-phase workflow with dependencies =====
    ensemble_script_path, final_job_id = analysis.submit_SLURM_job_array(
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

    # Verify jobs were submitted
    assert (
        ensemble_script_path.exists()
    ), "Ensemble script should exist after submission"
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
