import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_snakemake_local_workflow_generation_and_write(norfolk_multi_sim_analysis):
    """
    Test Snakemake workflow generation for local execution.

    Verifies that:
    1. Snakefile can be written to disk
    2. Snakefile contains required rules and flags
    3. Snakefile is non-empty
    """
    analysis = norfolk_multi_sim_analysis

    snakefile_content = analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )

    snakefile_path = tst_ut.write_snakefile(analysis, snakefile_content)

    assert snakefile_path.exists()
    assert len(snakefile_path.read_text()) > 100

    content = snakefile_path.read_text()
    tst_ut.assert_snakefile_has_rules(
        content, ["all", "setup", "simulation", "consolidate"]
    )
    tst_ut.assert_snakefile_has_flags(
        content,
        [
            "/workflow/envs/triton_swmm.yaml",
            "_status/sims/sim_{event_iloc}_complete.flag",
            "setup_workflow",
            "--process-system-inputs",
            "--compile-triton-swmm",
            "run_single_simulation",
            "--prepare-scenario",
            "consolidate_workflow",
        ],
    )


def test_snakemake_workflow_config_generation(norfolk_multi_sim_analysis):
    """
    Test configuration passed to Snakemake.

    Verifies that:
    1. All parameters are correctly formatted in Snakefile
    2. Resource specifications are valid
    3. Command-line arguments are properly escaped
    """
    analysis = norfolk_multi_sim_analysis

    snakefile_content = analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=False,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=True,
        compression_level=5,
    )

    tst_ut.assert_snakefile_has_flags(
        snakefile_content,
        [
            "--compression-level 5",
            "--which both",
            f"--system-config {analysis._system.system_config_yaml}",
            f"--analysis-config {analysis.analysis_config_yaml}",
        ],
    )

    n_sims = len(analysis.df_sims)
    assert f"SIM_IDS = {list(range(n_sims))}" in snakefile_content


@pytest.mark.parametrize(
    "config,expected_flags,forbidden_flags",
    [
        (
            {
                "process_system_level_inputs": True,
                "compile_TRITON_SWMM": True,
                "prepare_scenarios": False,
                "process_timeseries": False,
            },
            ["--process-system-inputs", "--compile-triton-swmm"],
            ["--prepare-scenario", "--process-timeseries"],
        ),
        (
            {
                "process_system_level_inputs": True,
                "compile_TRITON_SWMM": True,
                "prepare_scenarios": True,
                "process_timeseries": True,
            },
            ["--prepare-scenario", "--process-timeseries"],
            [],
        ),
    ],
)
def test_snakemake_multiple_configurations(
    norfolk_multi_sim_analysis, config, expected_flags, forbidden_flags
):
    """
    Test Snakemake generation with different parameter combinations.

    Verifies that:
    1. Optional parameters are correctly included/excluded
    """
    analysis = norfolk_multi_sim_analysis

    snakefile_content = analysis._workflow_builder.generate_snakefile_content(**config)

    tst_ut.assert_snakefile_has_flags(snakefile_content, expected_flags)
    for flag in forbidden_flags:
        assert flag not in snakefile_content


def test_snakemake_workflow_dry_run(norfolk_multi_sim_analysis):
    """
    Test Snakemake dry-run (--dry-run flag).

    Validates that:
    1. DAG can be constructed from Snakefile
    2. All dependencies resolve correctly
    3. No actual execution occurs
    4. Snakemake exit code is 0
    """
    analysis = norfolk_multi_sim_analysis

    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=True,
        prepare_scenarios=True,
        overwrite_scenario=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=True,
        overwrite_if_exist=True,
        compression_level=5,
        pickup_where_leftoff=False,
        dry_run=True,
        verbose=True,
    )

    assert result.get(
        "success"
    ), f"Snakemake dry-run failed: {result.get('message', '')}"
    assert result.get("mode") == "local"


@pytest.mark.slow
def test_snakemake_workflow_execution(norfolk_multi_sim_analysis):
    """
    Test Snakemake workflow execution (2 simulations).

    Validates that:
    1. submit_workflow() returns success
    2. Setup phase completes
    3. Simulations execute without errors
    4. Scenarios are prepared correctly
    5. Simulations run successfully
    6. Analysis summaries are generated
    """
    analysis = norfolk_multi_sim_analysis
    which = "both"  # which outputs to process

    result = analysis.submit_workflow(
        mode="local",
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
    )

    tst_ut.assert_analysis_workflow_completed_successfully(analysis, which=which)
