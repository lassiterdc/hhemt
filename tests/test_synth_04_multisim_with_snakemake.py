"""Synthetic-model multi-sim + Snakemake tier. Mirror of test_PC_04 using synth fixtures."""

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


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
        clear_raw_outputs=False,
        overwrite_outputs_if_already_created=True,
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
        assert (plots_dir / "per_sim" / event_id / f"peak_flood_depth{pfd_ext}").exists()
        assert (plots_dir / "per_sim" / event_id / f"conduit_flow{cf_ext}").exists()


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
