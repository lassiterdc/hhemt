"""Regression smoke against real Norfolk data. Detailed assertions live in test_synth_04_multisim_with_snakemake.py."""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


@pytest.mark.slow
def test_snakemake_workflow_dry_run(norfolk_multi_sim_analysis):
    """End-to-end Snakemake dry-run smoke: DAG constructs, dependencies resolve, no execution."""
    analysis = norfolk_multi_sim_analysis

    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=False,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=False,
        prepare_scenarios=True,
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=True,
        overwrite_outputs_if_already_created=True,
        compression_level=5,
        pickup_where_leftoff=False,
        dry_run=True,
        verbose=True,
    )

    assert result.get(
        "success"
    ), f"Snakemake dry-run failed: {result.get('message', '')}"
    assert result.get("mode") == "local"


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
    """
    from tests.utils.process_monitor import ProcessMonitor, RunnerConcurrencyMonitor

    analysis = norfolk_multi_sim_analysis
    which = "both"

    cores = analysis.cfg_analysis.local_cpu_cores_for_workflow
    expected_max = 1 + cores + 2

    with (
        ProcessMonitor(
            max_expected=expected_max,
            sample_interval=0.2,
            process_name_filter="python",
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
            overwrite_scenario_if_already_set_up=True,
            rerun_swmm_hydro_if_outputs_exist=True,
            process_timeseries=True,
            which=which,
            clear_raw_outputs=True,
            overwrite_outputs_if_already_created=True,
            compression_level=5,
            pickup_where_leftoff=False,
            verbose=True,
        )

        assert result["success"], "Workflow should complete successfully"
    tst_ut.assert_analysis_workflow_completed_successfully(analysis)

    process_monitor.assert_no_explosion(margin=2.0)
    process_report = process_monitor.get_report()
    assert not process_report["explosion_detected"], (
        f"Process explosion detected! Max: {process_report['max_processes']}, "
        f"Expected: ≤{process_report['max_expected']}"
    )

    runner_report = runner_monitor.get_detailed_report()
    timeline_path = (
        analysis.analysis_paths.analysis_dir / "runner_concurrency_timeline.csv"
    )
    runner_monitor.export_timeline(str(timeline_path))

    assert runner_report["max_total_runners"] <= cores * 2, (
        f"Max concurrent runners ({runner_report['max_total_runners']}) exceeded "
        f"reasonable limit (2x cores = {cores * 2})"
    )

    for runner_type, max_count in runner_report["max_concurrent"].items():
        if runner_type != "total":
            assert max_count <= cores + 2, (
                f"{runner_type} exceeded concurrency limit: "
                f"{max_count} > {cores + 2}"
            )

    assert runner_report["avg_total_runners"] <= cores, (
        f"Average concurrent runners ({runner_report['avg_total_runners']:.1f}) "
        f"should not exceed configured cores ({cores})"
    )
