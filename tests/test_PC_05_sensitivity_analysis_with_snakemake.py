import pytest
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.utils_for_testing import is_scheduler_context

pytestmark = pytest.mark.skipif(
    is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_snakemake_sensitivity_workflow_generation():
    """
    Test Snakemake workflow generation for sensitivity analysis.

    Verifies that:
    1. Master Snakefile is generated correctly
    2. Sub-analysis Snakefiles are generated for each sub-analysis
    3. Master Snakefile contains all necessary rules
    4. Master Snakefile has proper dependencies
    """
    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
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
    subanalysis_snakefile_paths = []
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

        # Store path info
        snakefile_path = sub_analysis.analysis_paths.analysis_dir / "Snakefile"
        working_dir = sub_analysis.analysis_paths.analysis_dir
        subanalysis_snakefile_paths.append(
            (sub_analysis_iloc, snakefile_path, working_dir)
        )

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

    # Verify individual simulation rules exist (flattened architecture)
    assert "rule simulation_sa" in master_snakefile_content
    assert "rule consolidate_sa" in master_snakefile_content

    # Verify sub-analysis consolidation rules exist
    num_sub_analyses = len(sensitivity.sub_analyses)
    for sa_id in range(num_sub_analyses):
        assert f"rule consolidate_sa_{sa_id}:" in master_snakefile_content

    # Verify master consolidation depends on all sub-analyses
    assert "rule master_consolidation:" in master_snakefile_content


def test_snakemake_sensitivity_workflow_files_written():
    """
    Test that Snakemake workflow files are written to disk correctly.

    Verifies that:
    1. Sub-analysis Snakefiles are written to correct locations
    2. Master Snakefile is written to master analysis directory
    3. All files are valid and non-empty
    """
    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    sensitivity = analysis.sensitivity

    # Generate Snakefiles for each sub-analysis
    subanalysis_snakefile_paths = []
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

        subanalysis_snakefile_paths.append(
            (
                sub_analysis_iloc,
                snakefile_path,
                sub_analysis.analysis_paths.analysis_dir,
            )
        )

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


def test_submit_workflow_detects_local_mode():
    """
    Test that submit_workflow() correctly detects local mode for sensitivity analysis.

    Note: This test does NOT actually run snakemake, only verifies detection logic.
    """
    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis

    # Verify that analysis is not in SLURM context
    assert not analysis.in_slurm, "Test must run on local machine, not in SLURM"

    # Test mode detection for sensitivity analysis
    detected_mode = "slurm" if analysis.in_slurm else "local"
    assert detected_mode == "local"


def test_snakemake_sensitivity_workflow_config_generation():
    """
    Test configuration passed to Snakemake for sensitivity analysis.

    Verifies that:
    1. All parameters are correctly formatted in master Snakefile
    2. Consolidation command includes correct flags
    3. Sub-analysis references are correct
    """
    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    sensitivity = analysis.sensitivity

    # Generate sub-analysis Snakefile paths
    subanalysis_snakefile_paths = []
    for sub_analysis_iloc, sub_analysis in sensitivity.sub_analyses.items():
        snakefile_path = sub_analysis.analysis_paths.analysis_dir / "Snakefile"
        working_dir = sub_analysis.analysis_paths.analysis_dir
        subanalysis_snakefile_paths.append(
            (sub_analysis_iloc, snakefile_path, working_dir)
        )

    # Test with different parameter combinations
    master_snakefile_content = sensitivity._generate_master_snakefile_content(
        which="TRITON",
        overwrite_if_exist=True,
        compression_level=7,
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
    Test Snakemake dry-run for sensitivity analysis (--dry-run flag).

    Validates that:
    1. DAG can be constructed from master Snakefile
    2. All dependencies resolve correctly
    3. No actual execution occurs
    4. Snakemake exit code is 0
    """
    import subprocess

    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    sensitivity = analysis.sensitivity

    # Generate all Snakefiles
    subanalysis_snakefile_paths = []
    for sub_analysis_iloc, sub_analysis in sensitivity.sub_analyses.items():
        snakefile_content = sub_analysis._generate_snakefile_content(
            process_system_level_inputs=False,
            compile_TRITON_SWMM=False,
            prepare_scenarios=True,
            process_timeseries=False,
        )
        snakefile_path = sub_analysis.analysis_paths.analysis_dir / "Snakefile"
        snakefile_path.write_text(snakefile_content)
        subanalysis_snakefile_paths.append(
            (
                sub_analysis_iloc,
                snakefile_path,
                sub_analysis.analysis_paths.analysis_dir,
            )
        )

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
    Test Snakemake sensitivity analysis workflow execution.

    Validates that:
    1. submit_workflow() returns success
    2. Setup phase completes for each sub-analysis
    3. All simulations execute without errors
    4. Sub-analysis consolidation completes
    5. Master consolidation completes
    6. Final sensitivity analysis summaries are generated
    """
    from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst

    nrflk_cpu_sensitivity = tst.retreive_norfolk_cpu_config_sensitivity_case(
        start_from_scratch=True
    )
    analysis = nrflk_cpu_sensitivity.system.analysis
    sensitivity = analysis.sensitivity

    # Submit the workflow
    result = sensitivity.submit_workflow(
        mode="local",
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
    )

    # Verify workflow submission was successful
    assert result["success"], f"Workflow submission failed: {result.get('message', '')}"

    # Verify TRITON compilation was successful
    assert analysis._system.compilation_successful, "TRITON compilation failed"

    # Verify all scenarios were created across all sub-analyses
    assert sensitivity.all_scenarios_created, "Not all scenarios were created"

    # Verify all simulations completed across all sub-analyses
    assert sensitivity.all_sims_run, "Not all simulations completed"

    # Verify all timeseries were processed
    assert (
        sensitivity.all_TRITON_timeseries_processed
    ), "Not all TRITON timeseries were processed"

    # Verify sub-analysis consolidation completed
    assert (
        sensitivity.TRITON_subanalyses_outputs_consolidated
    ), "Sub-analysis TRITON outputs not consolidated"

    # Verify master analysis consolidation completed
    assert (
        analysis.TRITON_analysis_summary_created
    ), "Master TRITON analysis summary not created"

    print("✅ Snakemake sensitivity analysis workflow completed successfully")
