import pytest
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils_for_testing import on_UVA_HPC

pytestmark = pytest.mark.skipif(not on_UVA_HPC(), reason="Only runs on UVA HPC")


@pytest.mark.slow
def test_snakemake_sensitivity_workflow_execution():
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
    from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst

    nrflk_cpu_sensitivity = (
        tst.retreive_norfolk_UVA_sensitivity_CPU_full_ensemble_short_sims(
            start_from_scratch=True
        )
    )
    system = nrflk_cpu_sensitivity.system
    analysis = system.analysis
    sensitivity = analysis.sensitivity

    # Submit the workflow using submit_workflow (not the old batch job method)
    result = sensitivity.submit_workflow(
        mode="slurm",  # Explicitly use SLURM mode
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=True,
        prepare_scenarios=True,
        overwrite_scenario=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which="TRITON",
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

    # Note: On SLURM, the workflow is submitted but may not complete immediately
    # We would need to poll for completion or check logs
    # For now, we just verify successful submission

    # Verify Phase 1 outputs (system inputs and compilation)
    dem_file = system.sys_paths.dem_processed
    assert dem_file.exists(), "DEM file should be created"

    mannings_file = system.sys_paths.mannings_processed
    assert mannings_file.exists(), "Mannings file should be created"

    assert system.compilation_successful, "TRITON-SWMM should be compiled"

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
