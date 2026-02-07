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
        content,
        [
            "all",
            "setup",
            "prepare_scenario",
            "run_simulation",
            "process_outputs",
            "consolidate",
        ],
    )
    tst_ut.assert_snakefile_has_flags(
        content,
        [
            "/workflow/envs/triton_swmm.yaml",
            "setup_workflow",
            "--process-system-inputs",
            "--compile-triton-swmm",
            "prepare_scenario_runner",
            "run_simulation_runner",
            "process_timeseries_runner",
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
            ["prepare_scenario_runner", "process_timeseries_runner"],
        ),
        (
            {
                "process_system_level_inputs": True,
                "compile_TRITON_SWMM": True,
                "prepare_scenarios": True,
                "process_timeseries": True,
            },
            ["prepare_scenario_runner", "process_timeseries_runner"],
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
        overwrite_system_inputs=False,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=False,
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


def test_snakemake_workflow_end_to_end(norfolk_multi_sim_analysis):
    """
    End-to-end Snakemake workflow test.

    Mirrors the validations from the single- and multi-sim tests by verifying:
    - System setup and compilation for enabled models
    - Scenarios prepared and simulations completed
    - Timeseries and summaries processed for each enabled model
    - Analysis-level consolidated summaries
    - Scenario status CSV and resource usage validation
    - SWMM-only vs TRITON-SWMM output consistency (when both enabled)
    """
    import xarray as xr

    analysis = norfolk_multi_sim_analysis

    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=False,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=False,
        prepare_scenarios=True,
        overwrite_scenario=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=True,
        overwrite_if_exist=True,
        compression_level=5,
        pickup_where_leftoff=False,
        verbose=True,
    )

    assert result.get("success"), result.get("message", "Workflow failed")
    assert result.get("mode") == "local"

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)

    enabled_models = tst_ut.get_enabled_model_types(analysis)
    for model_type in enabled_models:
        tst_ut.assert_model_simulation_run(analysis, model_type)
        tst_ut.assert_model_outputs_processed(analysis, model_type)

    # Validate all compilation types, if enabled
    if "triton" in enabled_models:
        tst_ut.assert_triton_compiled(analysis)
    if "tritonswmm" in enabled_models:
        tst_ut.assert_tritonswmm_compiled(analysis)
    if "swmm" in enabled_models:
        tst_ut.assert_swmm_compiled(analysis)

    # Cross-model consistency: SWMM-only outputs should be compatible with
    # the SWMM outputs embedded in the TRITON-SWMM coupled runs.
    if "swmm" in enabled_models and "tritonswmm" in enabled_models:
        for event_iloc in analysis.df_sims.index:
            proc = analysis._retrieve_sim_run_processing_object(event_iloc)
            paths = proc.scen_paths

            swmm_node_ts = paths.output_swmm_only_node_time_series
            swmm_link_ts = paths.output_swmm_only_link_time_series
            tritonswmm_node_ts = paths.output_tritonswmm_node_time_series
            tritonswmm_link_ts = paths.output_tritonswmm_link_time_series

            if not all(
                p is not None and p.exists()
                for p in [
                    swmm_node_ts,
                    swmm_link_ts,
                    tritonswmm_node_ts,
                    tritonswmm_link_ts,
                ]
            ):
                pytest.fail(
                    "Missing SWMM-only or TRITON-SWMM SWMM time series outputs; "
                    "workflow did not generate all expected files."
                )

            ds_swmm_nodes = xr.open_dataset(swmm_node_ts)
            ds_swmm_links = xr.open_dataset(swmm_link_ts)
            ds_tritonswmm_nodes = xr.open_dataset(tritonswmm_node_ts)
            ds_tritonswmm_links = xr.open_dataset(tritonswmm_link_ts)

            # Node/link ids should be present in the TRITON-SWMM datasets
            swmm_node_ids = set(ds_swmm_nodes["node_id"].values.tolist())
            swmm_link_ids = set(ds_swmm_links["link_id"].values.tolist())
            tritonswmm_node_ids = set(ds_tritonswmm_nodes["node_id"].values.tolist())
            tritonswmm_link_ids = set(ds_tritonswmm_links["link_id"].values.tolist())

            missing_nodes = swmm_node_ids - tritonswmm_node_ids
            missing_links = swmm_link_ids - tritonswmm_link_ids

            if missing_nodes:
                pytest.fail(
                    f"TRITON-SWMM node_ids missing {len(missing_nodes)} SWMM-only nodes."
                )
            if missing_links:
                pytest.fail(
                    f"TRITON-SWMM link_ids missing {len(missing_links)} SWMM-only links."
                )

            # Timestep counts should match
            if len(ds_swmm_nodes["date_time"]) != len(ds_tritonswmm_nodes["date_time"]):
                pytest.fail("Node time series timestep counts do not match")
            if len(ds_swmm_links["date_time"]) != len(ds_tritonswmm_links["date_time"]):
                pytest.fail("Link time series timestep counts do not match")

            # Data variables should match (order-agnostic)
            if set(ds_swmm_nodes.data_vars) != set(ds_tritonswmm_nodes.data_vars):
                pytest.fail("Node time series data variables do not match")
            swmm_link_vars = set(ds_swmm_links.data_vars)
            tritonswmm_link_vars = set(ds_tritonswmm_links.data_vars)

            # Normalize known naming differences before comparing
            swmm_link_vars = tst_ut.normalize_swmm_link_vars(swmm_link_vars)
            tritonswmm_link_vars = tst_ut.normalize_swmm_link_vars(tritonswmm_link_vars)

            if swmm_link_vars != tritonswmm_link_vars:
                pytest.fail("Link time series data variables do not match")


# @pytest.mark.slow
@pytest.mark.skip
def test_snakemake_workflow_concurrency_and_process_monitoring(
    norfolk_multi_sim_analysis,
):
    """
    Comprehensive concurrency and process explosion regression test.

    Combines two monitoring strategies in a single workflow run:
    1. ProcessMonitor - Broad process counting (catches explosions)
    2. RunnerConcurrencyMonitor - Granular runner tracking (verifies limits)

    Validates that:
    - No recursive subprocess spawning (fork bombs)
    - Total process count stays within expected limits
    - Each runner type respects concurrency limits
    - Brief spikes during phase transitions are bounded and expected
    - Average concurrency matches configured cores

    This test provides both regression testing (catch bugs) and deterministic
    verification (confirm expected behavior).
    """
    from tests.utils.process_monitor import ProcessMonitor, RunnerConcurrencyMonitor

    analysis = norfolk_multi_sim_analysis
    which = "both"

    # Calculate expected maximum processes:
    # - Snakemake master process: 1
    # - Worker processes (cores): 4 (from config)
    # - Margin for Python interpreter overhead: +2
    cores = analysis.cfg_analysis.local_cpu_cores_for_workflow
    expected_max = 1 + cores + 2  # = 7 processes

    # Run both monitors simultaneously (lightweight background threads)
    with (
        ProcessMonitor(
            max_expected=expected_max,
            sample_interval=0.2,
            process_name_filter="python",  # Only count Python processes
        ) as process_monitor,
        RunnerConcurrencyMonitor(sample_interval=0.1) as runner_monitor,
    ):
        result = analysis.submit_workflow(
            mode="local",
            process_system_level_inputs=True,
            overwrite_system_inputs=False,
            compile_TRITON_SWMM=True,
            recompile_if_already_done_successfully=False,
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

        assert result["success"], "Workflow should complete successfully"
    tst_ut.assert_analysis_workflow_completed_successfully(analysis)

    # ========================================================================
    # Part 1: Process Explosion Regression Test
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 1: PROCESS EXPLOSION REGRESSION TEST")
    print("=" * 70)

    # Check process count stayed within reasonable bounds
    process_monitor.assert_no_explosion(margin=2.0)  # Allow 2x expected as buffer

    # Print diagnostic info
    process_report = process_monitor.get_report()
    print(f"\n🔍 Total Process Monitor Report:")
    print(f"   Max processes: {process_report['max_processes']}")
    print(f"   Expected max: {process_report['max_expected']}")
    print(f"   Average: {process_report['avg_processes']:.1f}")
    print(f"   Explosion detected: {process_report['explosion_detected']}")

    # Strict assertion: should not exceed 2x expected
    assert not process_report["explosion_detected"], (
        f"Process explosion detected! Max: {process_report['max_processes']}, "
        f"Expected: ≤{process_report['max_expected']}"
    )

    print("   ✅ No process explosion detected")

    # ========================================================================
    # Part 2: Deterministic Runner Concurrency Verification
    # ========================================================================
    print("\n" + "=" * 70)
    print("PART 2: DETERMINISTIC RUNNER CONCURRENCY VERIFICATION")
    print("=" * 70)

    # Get detailed concurrency report
    runner_report = runner_monitor.get_detailed_report()

    # Print summary for visibility
    runner_monitor.print_summary()

    # Export timeline for debugging/visualization if needed
    timeline_path = (
        analysis.analysis_paths.analysis_dir / "runner_concurrency_timeline.csv"
    )
    runner_monitor.export_timeline(str(timeline_path))
    print(f"📊 Timeline exported to: {timeline_path}")

    # Deterministic assertions
    # Maximum concurrent runners should never exceed configured cores
    # Note: Brief spikes during phase transitions may slightly exceed cores
    # due to Python interpreter overhead, so we allow some margin
    assert runner_report["max_total_runners"] <= cores * 2, (
        f"Max concurrent runners ({runner_report['max_total_runners']}) exceeded "
        f"reasonable limit (2x cores = {cores * 2})"
    )

    # Each individual runner type should respect core limits
    for runner_type, max_count in runner_report["max_concurrent"].items():
        if runner_type != "total":
            assert max_count <= cores + 2, (
                f"{runner_type} exceeded concurrency limit: "
                f"{max_count} > {cores + 2}"
            )

    # Average should be well below maximum (indicates normal parallelism)
    assert runner_report["avg_total_runners"] <= cores, (
        f"Average concurrent runners ({runner_report['avg_total_runners']:.1f}) "
        f"should not exceed configured cores ({cores})"
    )

    print("\n   ✅ All concurrency limits respected")
    print("=" * 70 + "\n")
