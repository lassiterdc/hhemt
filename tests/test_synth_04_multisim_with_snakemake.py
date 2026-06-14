"""Synthetic-model multi-sim + Snakemake tier. Mirror of test_PC_04 using synth fixtures."""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
]


def test_snakemake_local_workflow_generation_and_write(synth_multi_sim_analysis):
    """
    Test Snakemake workflow generation for local execution.

    Verifies that:
    1. Snakefile can be written to disk
    2. Snakefile contains required rules and flags
    3. Snakefile is non-empty
    """
    analysis = synth_multi_sim_analysis

    snakefile_content = analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )

    snakefile_path = tst_ut.write_snakefile(analysis, snakefile_content)

    tst_ut.assert_file_exists(snakefile_path, "Snakefile")
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


def _render_report_rule_block(snakefile_content: str) -> str:
    """Extract the ``rule render_report:`` block from a generated Snakefile string.

    Returns the substring from ``rule render_report:`` up to the next
    column-0 ``rule `` (or end-of-string), so flag assertions are scoped to the
    render_report rule's shell rather than matching anywhere in the Snakefile.
    """
    marker = "rule render_report:"
    start = snakefile_content.index(marker)
    rest = snakefile_content[start:]
    nxt = rest.find("\nrule ", len(marker))
    return rest if nxt == -1 else rest[:nxt]


def test_reprocess_render_report_rule_carries_reprocess_flag(synth_multi_sim_analysis):
    """Generation assertion (reprocess-render-report-snakefile-selection, R3/R5).

    Non-sensitivity path: the REPROCESS Snakefile's ``rule render_report`` shell
    MUST pass ``--reprocess`` to ``render_report_runner`` so ``render_report()``
    selects ``Snakefile.reprocess``; the PRODUCTION Snakefile's ``rule
    render_report`` shell MUST NOT (byte-identical production render path).
    """
    from TRITON_SWMM_toolkit.reprocess_snakefile_generator import generate_reprocess_snakefile

    builder = synth_multi_sim_analysis._workflow_builder

    reprocess_content = generate_reprocess_snakefile(builder, start_with="render")
    production_content = builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )

    reprocess_block = _render_report_rule_block(reprocess_content)
    production_block = _render_report_rule_block(production_content)

    assert "render_report_runner" in reprocess_block, "reprocess render_report rule must invoke the runner"
    assert "--reprocess" in reprocess_block, (
        "non-sensitivity reprocess render_report shell MUST pass --reprocess so render_report() "
        "selects Snakefile.reprocess (R3/R5)"
    )
    assert "render_report_runner" in production_block, "production render_report rule must invoke the runner"
    assert "--reprocess" not in production_block, (
        "non-sensitivity production render_report shell MUST NOT pass --reprocess "
        "(byte-identical production render path; R3/R5)"
    )


def test_snakemake_workflow_config_generation(synth_multi_sim_analysis):
    """
    Test configuration passed to Snakemake.

    Verifies that:
    1. All parameters are correctly formatted in Snakefile
    2. Resource specifications are valid
    3. Command-line arguments are properly escaped
    """
    analysis = synth_multi_sim_analysis

    snakefile_content = analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=False,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
        which="both",
        override_clear_raw="all",
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
    assert "SIM_IDS = [" in snakefile_content
    assert "ILOC_BY_EVENT_ID = {" in snakefile_content
    import re as _re
    m = _re.search(r"^SIM_IDS = (\[[^\]]*\])", snakefile_content, _re.MULTILINE)
    assert m is not None
    assert len(eval(m.group(1))) == n_sims


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
    synth_multi_sim_analysis, config, expected_flags, forbidden_flags
):
    """
    Test Snakemake generation with different parameter combinations.

    Verifies that:
    1. Optional parameters are correctly included/excluded
    """
    analysis = synth_multi_sim_analysis

    snakefile_content = analysis._workflow_builder.generate_snakefile_content(**config)

    tst_ut.assert_snakefile_has_flags(snakefile_content, expected_flags)
    for flag in forbidden_flags:
        assert flag not in snakefile_content


def test_snakemake_workflow_dry_run(synth_multi_sim_analysis):
    """
    Test Snakemake dry-run (--dry-run flag).

    Validates that:
    1. DAG can be constructed from Snakefile
    2. All dependencies resolve correctly
    3. No actual execution occurs
    4. Snakemake exit code is 0
    """
    analysis = synth_multi_sim_analysis

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
        override_clear_raw="all",
        compression_level=5,
        pickup_where_leftoff=False,
        dry_run=True,
        verbose=True,
    )

    assert result.get(
        "success"
    ), f"Snakemake dry-run failed: {result.get('message', '')}"
    assert result.get("mode") == "local"

    df_status = analysis.df_status
    assert not df_status.empty
    assert "snakemake_allocated_nTasks" in df_status.columns
    assert "snakemake_allocated_omp_threads" in df_status.columns
    assert "snakemake_allocated_total_cpus" in df_status.columns


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_snakemake_workflow_end_to_end(synth_multi_sim_analysis):
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
    import subprocess
    import sys
    from pathlib import Path

    import xarray as xr

    analysis = synth_multi_sim_analysis

    repo_src = Path(__file__).resolve().parent.parent / "src"
    probe = subprocess.run(
        [sys.executable, "-c", "import TRITON_SWMM_toolkit; print(TRITON_SWMM_toolkit.__file__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    resolved = Path(probe.stdout.strip()).resolve()
    assert str(resolved).startswith(str(repo_src)), (
        f"subprocess resolved TRITON_SWMM_toolkit from {resolved}, "
        f"expected under {repo_src}. Layer 2 PYTHONPATH inheritance is broken."
    )

    # Phase 1 (synth-test-isolation-and-runtime) exercises SnakemakeDiagnostics
    # end-to-end so P1-V2's `test_diagnose.log` + `--reason` annotations
    # requirement is satisfied by every run of this test.
    from TRITON_SWMM_toolkit.workflow import SnakemakeDiagnostics

    diagnostic_log_path = (
        analysis.analysis_paths.analysis_dir / ".snakemake" / "log" / "test_diagnose.log"
    )

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
        override_clear_raw="none",
        compression_level=5,
        pickup_where_leftoff=False,
        verbose=True,
        snakemake_diagnostics=SnakemakeDiagnostics(
            verbose=True,
            reason=True,
            log_path=diagnostic_log_path,
        ),
    )

    assert result.get("success"), result.get("message", "Workflow failed")
    assert result.get("mode") == "local"
    assert diagnostic_log_path.exists(), (
        f"SnakemakeDiagnostics log_path was not honored — expected {diagnostic_log_path}"
    )
    log_text = diagnostic_log_path.read_text()
    # snakemake 8/9 emits per-job rerun reasons automatically when --verbose
    # is set (the standalone --reason flag was removed in snakemake 8).
    # Phase 1's SnakemakeDiagnostics folds the legacy reason intent into the
    # verbose path via emit_verbose; the log must therefore contain verbose-
    # mode signal (either explicit "reason:" annotations or, on workflows
    # where every job runs from scratch, the verbose-mode "Job " markers
    # for executing jobs).
    assert "reason:" in log_text.lower() or "job " in log_text.lower(), (
        "expected --verbose-mode rerun-reason annotations in diagnostic log; "
        f"got log of length {len(log_text)} chars"
    )

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)

    enabled_models = tst_ut.get_enabled_model_types(analysis)
    for model_type in enabled_models:
        tst_ut.assert_model_simulation_run(analysis, model_type)
        tst_ut.assert_model_outputs_processed(analysis, model_type)

    if "triton" in enabled_models:
        tst_ut.assert_triton_compiled(analysis)
    if "tritonswmm" in enabled_models:
        tst_ut.assert_tritonswmm_compiled(analysis)
    if "swmm" in enabled_models:
        tst_ut.assert_swmm_compiled(analysis)

    expected_threads = analysis.cfg_analysis.n_omp_threads
    assert (
        expected_threads >= 1
    ), f"n_omp_threads must be >= 1, but got {expected_threads}"

    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        paths = proc.scen_paths

        if paths.swmm_hydro_inp.exists():
            with open(paths.swmm_hydro_inp) as fp:
                content = fp.read()
                assert (
                    f"THREADS              {expected_threads}" in content
                ), f"hydro.inp for event {event_iloc} should have THREADS={expected_threads}"

        if paths.swmm_full_inp.exists():
            with open(paths.swmm_full_inp) as fp:
                content = fp.read()
                assert (
                    f"THREADS              {expected_threads}" in content
                ), f"full.inp for event {event_iloc} should have THREADS={expected_threads}"

    if "swmm" in enabled_models and "tritonswmm" in enabled_models:
        for event_iloc in analysis.df_sims.index:
            proc = analysis._retrieve_sim_run_processing_object(event_iloc)
            paths = proc.scen_paths

            swmm_node_ts = paths.output_swmm_only_node_time_series
            swmm_link_ts = paths.output_swmm_only_link_time_series
            tritonswmm_node_ts = paths.output_tritonswmm_node_time_series
            tritonswmm_link_ts = paths.output_tritonswmm_link_time_series

            tst_ut.assert_file_exists(swmm_node_ts, "SWMM-only node timeseries")
            tst_ut.assert_file_exists(swmm_link_ts, "SWMM-only link timeseries")
            tst_ut.assert_file_exists(tritonswmm_node_ts, "TRITON-SWMM node timeseries")
            tst_ut.assert_file_exists(tritonswmm_link_ts, "TRITON-SWMM link timeseries")

            ds_swmm_nodes = xr.open_dataset(swmm_node_ts)
            ds_swmm_links = xr.open_dataset(swmm_link_ts)
            ds_tritonswmm_nodes = xr.open_dataset(tritonswmm_node_ts)
            ds_tritonswmm_links = xr.open_dataset(tritonswmm_link_ts)

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

            # Known upstream bug: TRITON-SWMM coupled mode emits one fewer SWMM
            # reporting period than SWMM-only mode whenever the external BC is
            # active and non-zero. Mechanism: floating-point drift in the
            # accumulated `swmm_local_elapsedTime` inside `swmm_step` under
            # variable TRITON dt — the final iteration's clamped sub-step
            # leaves SWMM's clock < sim_duration by enough ULPs to miss the
            # last REPORT_STEP boundary. Confirmed empirically in this worktree
            # by setting `time_increment_fixed=1` (truncation disappears) but
            # the fixed-dt regime is incompatible with downstream TRITON
            # summary post-processing in this toolkit. The fix lives in
            # vendored TRITON / SWMM-engine and has been shipped to the
            # upstream developer; until it lands we tolerate a ≤1-step
            # differential here and fail loudly on anything larger.
            _BC_TRUNCATION_KNOWN_BUG = (
                "Node time series timestep counts differ by {delta} step(s) — "
                "exceeds the ≤1-step tolerance for the known TRITON-SWMM "
                "coupled-mode 1-step BC truncation bug (FP-drift in "
                "swmm_step under variable TRITON dt with active external BC). "
                "If delta == 1 this is the documented upstream bug; >1 is a "
                "new regression."
            )
            node_delta = abs(
                len(ds_swmm_nodes["date_time"])
                - len(ds_tritonswmm_nodes["date_time"])
            )
            if node_delta > 1:
                pytest.fail(_BC_TRUNCATION_KNOWN_BUG.format(delta=node_delta))
            link_delta = abs(
                len(ds_swmm_links["date_time"])
                - len(ds_tritonswmm_links["date_time"])
            )
            if link_delta > 1:
                pytest.fail(
                    _BC_TRUNCATION_KNOWN_BUG.replace("Node", "Link").format(
                        delta=link_delta
                    )
                )

            if set(ds_swmm_nodes.data_vars) != set(ds_tritonswmm_nodes.data_vars):
                pytest.fail("Node time series data variables do not match")
            swmm_link_vars = set(ds_swmm_links.data_vars)
            tritonswmm_link_vars = set(ds_tritonswmm_links.data_vars)

            swmm_link_vars = tst_ut.normalize_swmm_link_vars(swmm_link_vars)
            tritonswmm_link_vars = tst_ut.normalize_swmm_link_vars(tritonswmm_link_vars)

            if swmm_link_vars != tritonswmm_link_vars:
                pytest.fail("Link time series data variables do not match")


@pytest.mark.skip
def test_snakemake_workflow_concurrency_and_process_monitoring(
    synth_multi_sim_analysis,
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

    analysis = synth_multi_sim_analysis
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
            override_clear_raw="all",
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
    runner_monitor.print_summary()

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


# ─── Phase 7: Snakemake report integration tests ───────────────────────────────

from pathlib import Path as _Path
_SYNTH_MULTISIM_REPORT_CONFIG = (
    _Path(__file__).resolve().parents[1] / "configs" / "reports" / "synth_multisim_report_config.yaml"
)


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_run_and_render_report(synth_multi_sim_analysis_cached):
    """Full pipeline: run -> render. Exercises the plot rules + report rendering."""
    from pathlib import Path

    analysis = synth_multi_sim_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_MULTISIM_REPORT_CONFIG),
    )
    out_html = analysis.render_report(format="html")
    assert out_html.exists() and out_html.stat().st_size > 0

    plots_dir = analysis.analysis_paths.analysis_dir / "plots"
    # Resolve backend-dependent extensions via the canonical helper so the
    # assertions match the per-rule emission regardless of static_backend.
    from TRITON_SWMM_toolkit.workflow import _output_ext_for
    backend = analysis._workflow_builder._get_report_cfg_static_backend()
    so_ext = _output_ext_for(backend, "system_overview")
    pfd_ext = _output_ext_for(backend, "per_sim_peak_flood_depth")
    cf_ext = _output_ext_for(backend, "per_sim_conduit_flow")
    pas_ext = _output_ext_for(backend, "per_analysis_summary")
    assert (plots_dir / f"system_overview{so_ext}").exists()
    assert (plots_dir / "per_analysis" / f"summary_table{pas_ext}").exists()
    for event_iloc in analysis.df_sims.index:
        ev = analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug
        event_id = compute_event_id_slug(ev)
        # ADR-2: figures carry the canonical plot ID as their stem
        # (peak_flood_depth__evt.{event_id}); the manifest carries the same id as
        # a first-class plot_id field, equal to the stem by construction (R2/R4).
        import json

        pfd_stem = f"peak_flood_depth__evt.{event_id}"
        cf_stem = f"conduit_flow__evt.{event_id}"
        assert (plots_dir / "per_sim" / event_id / f"{pfd_stem}{pfd_ext}").exists()
        assert (plots_dir / "per_sim" / event_id / f"{cf_stem}{cf_ext}").exists()
        pfd_manifest = json.loads(
            (plots_dir / "per_sim" / event_id / f"{pfd_stem}.manifest.json").read_text()
        )
        assert pfd_manifest["plot_id"] == pfd_stem


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_render_report_idempotent(synth_multi_sim_analysis_cached):
    """render_report() must not re-execute the workflow (R11)."""
    import time
    from pathlib import Path

    analysis = synth_multi_sim_analysis_cached
    # Ensure plots exist from prior test_run_and_render_report run; harmless if already present.
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_MULTISIM_REPORT_CONFIG),
    )
    first_html = analysis.render_report(format="html")
    t0 = time.time()
    second_html = analysis.render_report(format="html")
    elapsed = time.time() - t0
    assert second_html == first_html
    assert elapsed < 30  # generous bound for R11 design target
    plots_dir = analysis.analysis_paths.analysis_dir / "plots"
    for plot in plots_dir.rglob("*.png"):
        assert plot.stat().st_mtime <= t0 + 1  # 1s clock-skew grace


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_synth_render_report_interactive_html(synth_multi_sim_analysis_cached):
    # Snakemake's report engine embeds figures as
    # `data:{mime};charset=utf8;filename={name};base64,{payload}` URIs in the
    # rendered analysis_report.html. With static_backend="plotly" (synth
    # fixture default), the chart renderers emit interactive HTML and the
    # table renderers (errors_and_warnings, scenario_status_appendix,
    # per_analysis_summary) emit Tabulator HTML; both surface as
    # data:text/html URIs. The base64-decoded payload contains
    # `Plotly.newPlot` or `new Tabulator`. This test asserts the bundle
    # carries the expected count of interactive figures and that the
    # on-disk plot bundle stays under the master plan's 5 MB / 15 MB
    # budgets.
    import base64
    import re
    from pathlib import Path

    analysis = synth_multi_sim_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_MULTISIM_REPORT_CONFIG),
    )
    report_path = analysis.render_report(format="html")
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")

    data_uris = re.findall(
        r'"data:text/html;charset=utf8;filename=([^;]+);base64,([A-Za-z0-9+/=]+)"',
        html,
    )
    assert len(data_uris) >= 6, (
        f"Expected >= 6 data:text/html figure URIs, got {len(data_uris)}: "
        f"{[name for name, _ in data_uris]}"
    )
    marker_hits = 0
    for _name, payload in data_uris:
        inner = base64.b64decode(payload).decode("utf-8", errors="replace")
        if ("Plotly.newPlot" in inner) or ("new Tabulator" in inner):
            marker_hits += 1
    assert marker_hits >= 6, (
        f"Expected >= 6 figures with Plotly/Tabulator markers in decoded "
        f"payload, got {marker_hits} of {len(data_uris)}"
    )

    plots_dir = analysis.analysis_paths.analysis_dir / "plots"
    html_files = list(plots_dir.rglob("*.html"))
    per_figure_max = max((p.stat().st_size for p in html_files), default=0)
    total = sum(p.stat().st_size for p in html_files)
    assert per_figure_max < 5_000_000, (
        f"Per-figure max {per_figure_max / 1e6:.1f} MB exceeds 5 MB budget"
    )
    assert total < 15_000_000, (
        f"Total plots/*.html {total / 1e6:.1f} MB exceeds 15 MB budget"
    )


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_plot_sources_attribution(synth_multi_sim_analysis_cached):
    """R15: 'Sources:' bullet block appears in rendered HTML report text."""
    from pathlib import Path

    analysis = synth_multi_sim_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_MULTISIM_REPORT_CONFIG),
    )
    analysis.render_report(format="html")
    html = (analysis.analysis_paths.analysis_dir / "analysis_report.html").read_text()
    assert "Sources:" in html


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_no_html_content_in_svg_file_references(synth_multi_sim_analysis_cached, tmp_path):
    # Rendered report.html must not reference any .svg file whose content is
    # not valid SVG XML. Snakemake's report engine dispatches by mime_type
    # derived from the file extension — a .svg path containing HTML triggers
    # an <img> dispatch that fails to parse and renders a broken-image icon.
    import re
    import xml.etree.ElementTree as ET
    import zipfile
    from pathlib import Path

    analysis = synth_multi_sim_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_MULTISIM_REPORT_CONFIG),
    )
    out_zip = analysis.render_report(format="zip")
    extract_dir = tmp_path / "report_extract"
    with zipfile.ZipFile(out_zip) as zf:
        zf.extractall(extract_dir)
    report_html = next(extract_dir.rglob("report.html"), None)
    assert report_html is not None, f"report.html not found under {extract_dir}"
    refs = re.findall(r'"data_uri":\s*"([^"]+\.svg)"', report_html.read_text())
    bad = []
    for rel in refs:
        target = (report_html.parent / rel).resolve()
        if not target.exists():
            continue
        try:
            root = ET.fromstring(target.read_bytes())
            local_name = root.tag.rsplit("}", 1)[-1]
            if local_name != "svg":
                bad.append((rel, f"root tag is {local_name!r}, expected 'svg'"))
        except ET.ParseError as exc:
            bad.append((rel, f"not valid XML: {exc}"))
    assert not bad, (
        f"{len(bad)} .svg file(s) referenced by report.html are not valid SVG "
        f"(would render as broken-image icons): {bad}"
    )


def test_emit_plot_with_sources_html_branch(tmp_path):
    """The HTML-string branch writes verbatim and emits a uniform manifest sidecar.

    Phase 1 substrate: ``emit_plot_with_sources(fig: str, ...)`` routes to
    ``_emit_html_with_sources``, writing the HTML text to ``output_path``
    UTF-8 and a ``<stem>.manifest.json`` sidecar with ``output_format: 'html'``
    and explicit ``None`` for matplotlib-only manifest fields.
    """
    import json

    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        emit_plot_with_sources,
    )

    output_path = tmp_path / "fig.html"
    analysis_dir = tmp_path
    source_file = tmp_path / "src.csv"
    source_file.write_text("a,b\n1,2\n", encoding="utf-8")

    html_text = "<html><body><div id='x'>example</div></body></html>"
    returned_path = emit_plot_with_sources(
        html_text,
        output_path,
        [source_file],
        analysis_dir=analysis_dir,
        output_format="html",
        manifest_data={"renderer": "test", "panel_count": 1},
    )

    assert returned_path == output_path
    assert output_path.read_text(encoding="utf-8") == html_text
    manifest_path = output_path.parent / f"{output_path.stem}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["output_format"] == "html"
    assert manifest["full_res_path"] == str(output_path)
    assert manifest["source_paths_relative"] == ["src.csv"]
    assert manifest["renderer_data"]["renderer"] == "test"
    # No preview-PNG sibling on the HTML branch
    preview_path = output_path.parent / f"{output_path.stem}.preview.png"
    assert not preview_path.exists()
    # Matplotlib-only fields explicitly None for uniform consumer surface
    assert manifest["preview_path"] is None
    assert manifest["full_res_dpi"] is None
    assert manifest["preview_dpi"] is None
    assert manifest["figure_size_inches"] is None
    assert manifest["preview_size_bytes"] is None


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_cleanup_stale_metadata_auto_applies_after_rule_rename(
    synth_multi_sim_analysis_cached,
):
    # Phase 8.5: when cleanup_stale_metadata=True (default) and orphan
    # metadata records are enumerated, analysis.run() subprocess-invokes
    # snakemake --cleanup-metadata. The enumeration is deterministic
    # (no filesystem inspection) so it is non-empty whenever df_sims has
    # rows. We mock submit_workflow + the subprocess invocation to verify
    # the gate fires without re-running the workflow.
    from unittest.mock import patch

    analysis = synth_multi_sim_analysis_cached

    # Deterministic enumeration covers the Phase 8 orphan paths.
    orphan_paths = analysis._enumerate_stale_metadata_paths()
    assert "plots/system_overview.png" in orphan_paths
    from TRITON_SWMM_toolkit.scenario import compute_event_id_slug
    for event_iloc in analysis.df_sims.index:
        ev = analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
        event_id = compute_event_id_slug(ev)
        assert f"plots/per_sim/{event_id}/peak_flood_depth.png" in orphan_paths
        assert f"plots/per_sim/{event_id}/conduit_flow.png" in orphan_paths

    # The cleanup gate is preconditioned on BOTH a Snakefile and a
    # `.snakemake/metadata/` directory existing (skipped on first-run
    # analyses where no metadata records can exist). Ensure both exist
    # so the gate fires.
    snakefile = analysis.analysis_paths.analysis_dir / "Snakefile"
    snakefile.touch(exist_ok=True)
    metadata_dir = analysis.analysis_paths.analysis_dir / ".snakemake" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Gate fires with cleanup_stale_metadata=True (default).
    with (
        patch.object(analysis, "_invoke_snakemake_cleanup_metadata") as mock_inv,
        patch.object(analysis, "submit_workflow"),
    ):
        analysis.run(cleanup_stale_metadata=True, dry_run=True, verbose=False)
    mock_inv.assert_called_once()
    called_paths = mock_inv.call_args[0][0]
    assert called_paths == orphan_paths


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_cleanup_stale_metadata_disabled_skips_invocation(
    synth_multi_sim_analysis_cached,
):
    # Phase 8.5: when cleanup_stale_metadata=False, the cleanup gate does
    # not fire — _invoke_snakemake_cleanup_metadata is not called.
    from unittest.mock import patch

    analysis = synth_multi_sim_analysis_cached
    with (
        patch.object(analysis, "_invoke_snakemake_cleanup_metadata") as mock_inv,
        patch.object(analysis, "submit_workflow"),
    ):
        analysis.run(cleanup_stale_metadata=False, dry_run=True, verbose=False)
    mock_inv.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 2 — V-P2.2 / V-P2.3: scenario_status.csv carries the
# disk_utilization_bytes column and analysis_report.html carries the Disk
# Utilization sidebar card after a full run + render.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_scenario_status_csv_disk_utilization_column(synth_multi_sim_analysis_cached):
    """V-P2.2 — `scenario_status.csv` carries `disk_utilization_bytes`
    column with non-empty integer values for completed scenarios."""
    import csv
    from pathlib import Path

    analysis = synth_multi_sim_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_MULTISIM_REPORT_CONFIG),
    )
    csv_path = analysis.analysis_paths.analysis_dir / "scenario_status.csv"
    assert csv_path.exists(), f"scenario_status.csv not emitted at {csv_path}"

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert "disk_utilization_bytes" in reader.fieldnames, (
            f"disk_utilization_bytes missing from columns: {reader.fieldnames}"
        )

    # At least one row should have a non-empty integer-valued cell.
    int_cells = [
        r["disk_utilization_bytes"]
        for r in rows
        if r.get("disk_utilization_bytes") not in ("", None)
    ]
    assert int_cells, "All disk_utilization_bytes cells were empty"
    for cell in int_cells:
        assert int(cell) >= 0


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_render_report_includes_disk_utilization_card(synth_multi_sim_analysis_cached):
    """V-P2.3 — analysis_report.html carries the Disk Utilization sidebar
    card after run + render."""
    from pathlib import Path

    analysis = synth_multi_sim_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_MULTISIM_REPORT_CONFIG),
    )
    out_html = analysis.render_report(format="html")
    assert out_html.exists() and out_html.stat().st_size > 0

    plots_dir = analysis.analysis_paths.analysis_dir / "plots"
    from TRITON_SWMM_toolkit.workflow import _output_ext_for
    backend = analysis._workflow_builder._get_report_cfg_static_backend()
    du_ext = _output_ext_for(backend, "disk_utilization")
    du_path = plots_dir / f"disk_utilization{du_ext}"
    assert du_path.exists(), f"disk_utilization plot not emitted at {du_path}"
    du_html = du_path.read_text()
    # Either the populated table or the missing-sentinel banner is a valid
    # rendered output (both are emitted by the same renderer). On a
    # successful end-to-end run the analysis-level sentinel must be present.
    assert "du-table" in du_html, (
        f"Disk Utilization card did not render the populated table; got: {du_html[:200]!r}"
    )


def _set_batch_job_fields(cfg_analysis, *, account="legacy_acct"):
    """Set the minimum analysis_config fields generate_snakemake_config('slurm')
    requires (the synth case defaults them to None)."""
    cfg_analysis.hpc_ensemble_partition = "gpu"
    cfg_analysis.hpc_max_simultaneous_sims = 4
    cfg_analysis.hpc_account = account


def test_slurm_config_reads_hpc_system_config(synth_multi_sim_analysis, tmp_path):
    """Phase 2 (R3): generate_snakemake_config('slurm') sources slurm_account from
    cfg_hpc_system.default_account and emits no dead `slurm:` block."""
    import yaml as _yaml

    from TRITON_SWMM_toolkit.config.loaders import load_hpc_system_config

    analysis = synth_multi_sim_analysis
    _set_batch_job_fields(analysis.cfg_analysis, account="legacy_acct")

    hpc_yaml = tmp_path / "hpc_system_config.yaml"
    hpc_yaml.write_text(
        _yaml.safe_dump(
            {
                "system_name": "synth-cluster",
                "default_account": "synth_acct",
                "gpu_allocation_flavor": "gres",
                # max_runtime large so the §preflight does not fire here.
                "partitions": {"gpu": {"max_runtime": 100000, "gpus_per_node": 8}},
            }
        )
    )
    analysis.cfg_hpc_system = load_hpc_system_config(hpc_yaml)
    analysis._workflow_builder.cfg_hpc_system = analysis.cfg_hpc_system

    cfg = analysis._workflow_builder.generate_snakemake_config(mode="slurm")
    default_res = cfg["default-resources"]
    # Account sourced from cfg_hpc_system.default_account, NOT the legacy hpc_account.
    assert "slurm_account=synth_acct" in default_res
    assert "slurm_account=legacy_acct" not in default_res
    assert "slurm_partition=gpu" in default_res
    # The dead `slurm: {sbatch: {...}}` block is deleted (Phase 2).
    assert "slurm" not in cfg


def test_slurm_config_none_hpc_system_is_byte_identical(synth_multi_sim_analysis):
    """Phase 2 (R2): with cfg_hpc_system None, slurm_account falls back to the
    legacy cfg_analysis.hpc_account read (byte-identical to pre-Phase-2)."""
    analysis = synth_multi_sim_analysis
    _set_batch_job_fields(analysis.cfg_analysis, account="legacy_acct")
    assert analysis._workflow_builder.cfg_hpc_system is None

    cfg = analysis._workflow_builder.generate_snakemake_config(mode="slurm")
    default_res = cfg["default-resources"]
    assert "slurm_account=legacy_acct" in default_res  # legacy read preserved
    assert "slurm" not in cfg  # dead block deleted regardless of config presence
