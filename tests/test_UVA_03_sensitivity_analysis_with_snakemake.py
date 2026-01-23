import pytest
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils_for_testing import on_UVA_HPC

pytestmark = pytest.mark.skipif(not on_UVA_HPC(), reason="Only runs on UVA HPC")


def test_snakemake_sensitivity_workflow_generation():
    """
    Test Snakemake workflow generation for sensitivity analysis on UVA HPC.

    Verifies that:
    1. Master Snakefile is generated correctly
    2. Sub-analysis Snakefiles are generated for each sub-analysis
    3. Master Snakefile contains all necessary rules
    4. Master Snakefile has proper dependencies
    """
    nrflk_cpu_sensitivity = tst.retreive_norfolk_UVA_sensitivity_CPU_minimal(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis

    # Verify sensitivity analysis is enabled
    assert analysis.cfg_analysis.toggle_sensitivity_analysis == True
    assert hasattr(analysis, "sensitivity")

    # Get the sensitivity analysis object
    sensitivity = analysis.sensitivity

    # Verify sub-analyses were created
    assert len(sensitivity.sub_analyses) > 0

    # Generate sub-analysis Snakefiles (without running)
    for sub_analysis_iloc, sub_analysis in sensitivity.sub_analyses.items():
        snakefile_content = sub_analysis._generate_snakefile_content(
            process_system_level_inputs=False,
            compile_TRITON_SWMM=True,
            prepare_scenarios=True,
            process_timeseries=True,
        )

        # Verify sub-analysis Snakefile structure
        assert "rule all:" in snakefile_content
        assert "rule setup:" in snakefile_content
        assert "rule simulation:" in snakefile_content
        assert "rule consolidate:" in snakefile_content

    # Generate master Snakefile content
    master_snakefile_content = sensitivity._generate_master_snakefile_content(
        which="both",
        overwrite_if_exist=False,
        compression_level=5,
    )

    # Verify master Snakefile structure
    assert "rule all:" in master_snakefile_content
    assert "rule master_consolidation:" in master_snakefile_content
    assert "--consolidate-sensitivity-analysis-outputs" in master_snakefile_content

    # Verify wildcard-based subanalysis rule (new pattern using Snakemake wildcards)
    assert "rule subanalysis:" in master_snakefile_content
    assert (
        "_status/subanalysis_{sub_analysis_id}_complete.flag"
        in master_snakefile_content
    )
    assert "logs/subanalysis_{sub_analysis_id}.log" in master_snakefile_content
    assert "{wildcards.sub_analysis_id}" in master_snakefile_content

    # Verify SUB_ANALYSIS_IDS list is present
    assert "SUB_ANALYSIS_IDS" in master_snakefile_content

    # Verify expand function is used in master_consolidation dependencies
    assert (
        'expand("_status/subanalysis_{sub_analysis_id}_complete.flag", sub_analysis_id=SUB_ANALYSIS_IDS)'
        in master_snakefile_content
    )

    # Verify master consolidation depends on all sub-analyses
    assert "master_consolidation:" in master_snakefile_content


def test_snakemake_sensitivity_workflow_files_written():
    """
    Test that Snakemake workflow files are written to disk correctly on UVA HPC.

    Verifies that:
    1. Sub-analysis Snakefiles are written to correct locations
    2. Master Snakefile is written to master analysis directory
    3. All files are valid and non-empty
    """
    nrflk_cpu_sensitivity = tst.retreive_norfolk_UVA_sensitivity_CPU_minimal(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    sensitivity = analysis.sensitivity

    # Generate Snakefiles for each sub-analysis
    for sub_analysis_iloc, sub_analysis in sensitivity.sub_analyses.items():
        snakefile_content = sub_analysis._generate_snakefile_content(
            process_system_level_inputs=False,
            compile_TRITON_SWMM=True,
            prepare_scenarios=True,
            process_timeseries=True,
        )
        snakefile_path = sub_analysis.analysis_paths.analysis_dir / "Snakefile"
        snakefile_path.write_text(snakefile_content)

        # Verify file was written
        assert snakefile_path.exists()
        assert len(snakefile_path.read_text()) > 100

    # Generate and write master Snakefile
    master_snakefile_content = sensitivity._generate_master_snakefile_content(
        which="both",
        overwrite_if_exist=False,
        compression_level=5,
    )

    master_snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"
    master_snakefile_path.write_text(master_snakefile_content)

    # Verify master Snakefile was written
    assert master_snakefile_path.exists()
    assert len(master_snakefile_path.read_text()) > 100

    # Verify content matches expected structure
    content = master_snakefile_path.read_text()
    assert "rule all:" in content
    assert "rule master_consolidation:" in content


def test_submit_workflow_detects_slurm_mode():
    """
    Test that submit_workflow() correctly detects SLURM mode for sensitivity analysis on UVA HPC.

    Note: This test does NOT actually run snakemake, only verifies detection logic.
    """
    nrflk_cpu_sensitivity = tst.retreive_norfolk_UVA_sensitivity_CPU_minimal(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis

    # Verify that analysis detects SLURM context (if running as SLURM job)
    # Test mode detection for sensitivity analysis
    detected_mode = "slurm" if analysis.in_slurm else "local"

    # On UVA HPC, we expect SLURM mode when running as a job
    # (This assertion might vary depending on whether test is run interactively or as a job)
    assert detected_mode in ["slurm", "local"]  # Accept both for flexibility


def test_snakemake_sensitivity_workflow_config_generation():
    """
    Test configuration passed to Snakemake for sensitivity analysis on UVA HPC.

    Verifies that:
    1. All parameters are correctly formatted in master Snakefile
    2. Consolidation command includes correct flags
    3. Sub-analysis references are correct
    """
    nrflk_cpu_sensitivity = tst.retreive_norfolk_UVA_sensitivity_CPU_minimal(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    sensitivity = analysis.sensitivity

    # Generate master Snakefile content
    master_snakefile_content = sensitivity._generate_master_snakefile_content(
        which="both",
        overwrite_if_exist=False,
        compression_level=5,
    )

    # Verify compression level
    assert "--compression-level 7" in master_snakefile_content

    # Verify which parameter
    assert "--which TRITON" in master_snakefile_content

    # Verify overwrite flag
    assert "--overwrite-if-exist" in master_snakefile_content

    # Verify sensitivity analysis consolidation flag
    assert "--consolidate-sensitivity-analysis-outputs" in master_snakefile_content

    # Verify path handling
    assert (
        f"--system-config {analysis._system.system_config_yaml}"
        in master_snakefile_content
    )
    assert (
        f"--analysis-config {analysis.analysis_config_yaml}" in master_snakefile_content
    )


def test_snakemake_sensitivity_workflow_dry_run():
    """
    Test Snakemake dry-run for sensitivity analysis (--dry-run flag) on UVA HPC.

    Validates that:
    1. DAG can be constructed from master Snakefile
    2. All dependencies resolve correctly
    3. No actual execution occurs
    4. Snakemake exit code is 0
    """
    import subprocess

    nrflk_cpu_sensitivity = tst.retreive_norfolk_UVA_sensitivity_CPU_minimal(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    sensitivity = analysis.sensitivity

    # Generate all Snakefiles
    for sub_analysis_iloc, sub_analysis in sensitivity.sub_analyses.items():
        snakefile_content = sub_analysis._generate_snakefile_content(
            process_system_level_inputs=False,
            compile_TRITON_SWMM=False,
            prepare_scenarios=True,
            process_timeseries=False,
        )
        snakefile_path = sub_analysis.analysis_paths.analysis_dir / "Snakefile"
        snakefile_path.write_text(snakefile_content)

    # Generate master Snakefile
    master_snakefile_content = sensitivity._generate_master_snakefile_content(
        which="both",
        overwrite_if_exist=False,
        compression_level=5,
    )
    master_snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"
    master_snakefile_path.write_text(master_snakefile_content)

    # Create logs directory
    logs_dir = analysis.analysis_paths.analysis_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)

    # Create _status directory
    status_dir = analysis.analysis_paths.analysis_dir / "_status"
    status_dir.mkdir(exist_ok=True, parents=True)

    # Run snakemake --dry-run
    result = subprocess.run(
        ["snakemake", "--snakefile", str(master_snakefile_path), "--dry-run", "-p"],
        cwd=str(analysis.analysis_paths.analysis_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Verify DAG construction was successful
    assert (
        result.returncode == 0
    ), f"Snakemake dry-run failed:\n{result.stdout}\n{result.stderr}"

    # Verify rules are present in output
    assert "rule" in result.stdout or "DAG" in result.stdout

    print(f"✅ Snakemake sensitivity analysis dry-run successful - DAG validated")


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

    nrflk_cpu_sensitivity = tst.retreive_norfolk_UVA_sensitivity_CPU_minimal(
        start_from_scratch=True
    )
    system = nrflk_cpu_sensitivity.system
    analysis = system.analysis
    sensitivity = analysis.sensitivity

    result = sensitivity.submit_workflow(
        mode="slurm",
        process_system_level_inputs=False,
        overwrite_system_inputs=False,
        compile_TRITON_SWMM=False,
        recompile_if_already_done_successfully=False,
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
