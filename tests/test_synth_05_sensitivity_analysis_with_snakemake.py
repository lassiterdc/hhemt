"""Synthetic-model sensitivity + Snakemake tier. Mirror of test_PC_05 using synth fixtures."""

from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)

def test_snakemake_sensitivity_workflow_generation_and_write(
    synth_sensitivity_analysis,
):
    """
    Test Snakemake workflow generation for sensitivity analysis.

    Verifies that:
    1. Sub-analysis Snakefiles are generated and written correctly
    2. Master Snakefile is generated and written correctly
    3. Master Snakefile contains required rules and flags
    """
    analysis = synth_sensitivity_analysis

    assert analysis.cfg_analysis.toggle_sensitivity_analysis is True
    assert hasattr(analysis, "sensitivity")

    sensitivity = analysis.sensitivity

    assert len(sensitivity.sub_analyses) > 0

    for sub_analysis in sensitivity.sub_analyses.values():
        snakefile_content = sub_analysis._workflow_builder.generate_snakefile_content(
            process_system_level_inputs=False,
            compile_TRITON_SWMM=True,
            prepare_scenarios=True,
            process_timeseries=True,
        )

        tst_ut.assert_snakefile_has_rules(
            snakefile_content,
            [
                "all",
                "setup",
                "prepare_scenario",
                "run_simulation",
                "process_outputs",
                "consolidate",
            ],
        )

        sub_snakefile_path = tst_ut.write_snakefile(sub_analysis, snakefile_content)
        tst_ut.assert_file_exists(sub_snakefile_path, "Sub-analysis Snakefile")
        assert len(sub_snakefile_path.read_text()) > 100

    master_snakefile_content = (
        sensitivity._workflow_builder.generate_master_snakefile_content(
            which="both",
            overwrite_outputs_if_already_created=False,
            compression_level=5,
        )
    )

    tst_ut.assert_snakefile_has_rules(
        master_snakefile_content,
        [
            "all",
            "master_consolidation",
            "prepare_sa",
            "simulation_sa",
            "process_sa",
            "consolidate_",
            "plot_sensitivity_benchmarking",
        ],
    )
    assert (
        'expand("plots/sensitivity/benchmarking/{independent_var}_vs_total.html"'
        in master_snakefile_content
    ), "rule all must wildcard-expand benchmarking HTML figures over independent_var (plotly backend default)"
    tst_ut.assert_snakefile_has_flags(
        master_snakefile_content,
        [
            "--consolidate-sensitivity-analysis-outputs",
            "prepare_scenario_runner",
            "run_simulation_runner",
            "process_timeseries_runner",
        ],
    )

    num_sub_analyses = len(sensitivity.sub_analyses)
    for sa_id in range(num_sub_analyses):
        assert f"rule consolidate_sa_{sa_id}:" in master_snakefile_content

    master_snakefile_path = tst_ut.write_snakefile(analysis, master_snakefile_content)
    tst_ut.assert_file_exists(master_snakefile_path, "Master Snakefile")
    assert len(master_snakefile_path.read_text()) > 100


def test_phase3_master_snakefile_emits_per_target_setup_rules(
    synth_sensitivity_analysis,
):
    """Phase 3: master Snakefile emits one `rule setup_target_{N}` per unique system target.

    The synth fixture has no per-SA `system_config_yaml` column, so all sub-analyses
    collapse to a single UniqueSystemTarget. The Snakefile must emit exactly one
    `rule setup_target_0` and zero standalone `rule setup:` blocks.
    """
    analysis = synth_sensitivity_analysis
    sensitivity = analysis.sensitivity
    n_targets = len(sensitivity.unique_system_targets)
    assert n_targets == 1, "synth fixture is single-target (no system_config_yaml column)"

    content = sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both",
        overwrite_outputs_if_already_created=False,
        compression_level=5,
    )

    assert "rule setup_target_0:" in content
    assert "_status/a_setup_target_0_complete.flag" in content
    # The legacy single-target rule name and flag must not appear.
    assert "rule setup:\n" not in content
    assert "_status/a_setup_complete.flag" not in content
    # Per-SA prepare rules must depend on the new flag.
    for sa_id in sensitivity.sub_analyses.keys():
        assert (
            f"rule prepare_sa_{str(sa_id).replace('.', '_').replace('-', '_')}_evt_"
            in content
        )
    # The setup-target flag must surface in rule all so the DAG planner reaches it
    # even for sub-analyses whose df_sims is empty.
    rule_all_block = content.split("rule all:")[1].split("rule setup_target_0:")[0]
    assert "_status/a_setup_target_0_complete.flag" in rule_all_block


@pytest.mark.parametrize(
    "config,expected_flags",
    [
        (
            {
                "which": "TRITON",
                "overwrite_outputs_if_already_created": True,
                "compression_level": 7,
            },
            [
                "--compression-level 7",
                "--which TRITON",
                "--overwrite-outputs-if-already-created",
                "--consolidate-sensitivity-analysis-outputs",
            ],
        ),
        (
            {
                "which": "both",
                "overwrite_outputs_if_already_created": False,
                "compression_level": 5,
            },
            [
                "--compression-level 5",
                "--which both",
                "--consolidate-sensitivity-analysis-outputs",
            ],
        ),
    ],
)
def test_snakemake_sensitivity_workflow_config_generation(
    synth_sensitivity_analysis, config, expected_flags
):
    """
    Test configuration passed to Snakemake for sensitivity analysis.

    Verifies that:
    1. All parameters are correctly formatted in master Snakefile
    2. Consolidation command includes correct flags
    3. Sub-analysis references are correct
    """
    analysis = synth_sensitivity_analysis
    sensitivity = analysis.sensitivity

    master_snakefile_content = (
        sensitivity._workflow_builder.generate_master_snakefile_content(**config)
    )

    tst_ut.assert_snakefile_has_flags(
        master_snakefile_content,
        expected_flags
        + [
            f"--system-config {analysis._system.system_config_yaml}",
            f"--analysis-config {analysis.analysis_config_yaml}",
        ],
    )


def test_snakemake_sensitivity_workflow_dry_run(
    synth_sensitivity_analysis,
):
    """
    Test Snakemake dry-run for sensitivity analysis (--dry-run flag).

    Validates that:
    1. DAG can be constructed from master Snakefile
    2. All dependencies resolve correctly
    3. No actual execution occurs
    4. Snakemake exit code is 0
    """
    analysis = synth_sensitivity_analysis
    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=True,
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
    assert "sa_id" in df_status.columns
    expected_ids = [str(i) for i in range(len(df_status))]
    assert df_status["sa_id"].tolist() == expected_ids


@pytest.mark.slow
def test_snakemake_sensitivity_workflow_execution(synth_sensitivity_analysis):
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
    analysis = synth_sensitivity_analysis

    which = "both"

    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=True,
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

    assert result["success"], f"Workflow submission failed: {result.get('message', '')}"

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)

    analysis_dir = analysis.analysis_paths.analysis_dir
    for indep_var in ("n_devices",):
        figure = analysis_dir / "plots" / "sensitivity" / "benchmarking" / f"{indep_var}_vs_total.html"
        assert figure.exists(), f"Expected benchmarking figure missing: {figure}"


# ─── Phase 7: Snakemake report integration tests (sensitivity master) ──────────

from pathlib import Path as _Path
_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7 = (
    _Path(__file__).resolve().parents[1] / "configs" / "reports" / "synth_sensitivity_report_config.yaml"
)


@pytest.mark.slow
def test_run_and_render_report(synth_sensitivity_analysis_cached):
    """Sensitivity run -> master render. Asserts master report exists; no per-sub-analysis report (R13)."""
    from pathlib import Path

    analysis = synth_sensitivity_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7),
    )
    out_html = analysis.sensitivity.render_report(format="html")
    assert out_html.exists() and out_html.stat().st_size > 0

    master_dir = analysis.sensitivity.master_analysis.analysis_paths.analysis_dir
    bench_dir = master_dir / "plots" / "sensitivity" / "benchmarking"
    assert bench_dir.exists()
    assert any(bench_dir.glob("*_vs_total.html"))

    # R13: no per-sub-analysis report
    for sa_id, sub in analysis.sensitivity.sub_analyses.items():
        sub_html = sub.analysis_paths.analysis_dir / "analysis_report.html"
        assert not sub_html.exists(), (
            f"unexpected per-sub-analysis report at {sub_html} for sa_id={sa_id}"
        )


@pytest.mark.slow
def test_render_report_idempotent(synth_sensitivity_analysis_cached):
    """Sensitivity render_report() is idempotent (R11)."""
    import time
    from pathlib import Path

    analysis = synth_sensitivity_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7),
    )
    first_html = analysis.sensitivity.render_report(format="html")
    t0 = time.time()
    second_html = analysis.sensitivity.render_report(format="html")
    elapsed = time.time() - t0
    assert second_html == first_html
    assert elapsed < 30


@pytest.mark.slow
def test_plot_sources_attribution(synth_sensitivity_analysis_cached):
    """R15: 'Sources:' bullet block appears in master HTML report."""
    from pathlib import Path

    analysis = synth_sensitivity_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7),
    )
    analysis.sensitivity.render_report(format="html")

    master_dir = analysis.sensitivity.master_analysis.analysis_paths.analysis_dir
    html = (master_dir / "analysis_report.html").read_text()
    assert "Sources:" in html


@pytest.mark.slow
def test_no_html_content_in_svg_file_references(synth_sensitivity_analysis_cached, tmp_path):
    # Rendered report.html must not reference any .svg file whose content is
    # not valid SVG XML. See test_synth_04 mirror for the failure-mode rationale.
    import re
    import xml.etree.ElementTree as ET
    import zipfile
    from pathlib import Path

    analysis = synth_sensitivity_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7),
    )
    out_zip = analysis.sensitivity.render_report(format="zip")
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


# ============================================================================
# Phase 1 of prefixed_column_config_variation — tests
# ============================================================================

import yaml as _yaml
from TRITON_SWMM_toolkit.exceptions import ConfigurationError
import tests.fixtures.test_case_catalog as _cases


def test_system_overlay_mutual_exclusion_with_system_config_yaml():
    """Phase 1 R3 — row with both system_config_yaml and system.* raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mutex_violation(
            start_from_scratch=True
        )


def test_system_overlay_validator_re_fire_invalid_value():
    """Phase 1 R4 — invalid overlay value raises ConfigurationError via Pydantic."""
    with pytest.raises(ConfigurationError, match="SystemConfig validation"):
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_invalid_overlay(
            start_from_scratch=True
        )


def test_gpu_hardware_override_column_raises_migration_error():
    """Phase 1 R8, R-X-2 — legacy column raises ConfigurationError naming migration path."""
    with pytest.raises(ConfigurationError, match=r"gpu_hardware_override.*system\.gpu_hardware"):
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_legacy_gpu_hardware_override(
            start_from_scratch=True
        )


def test_unknown_column_rejected_with_nearest_match_hint():
    """Phase 1 R9 — unknown column (typo) raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Unknown sensitivity-CSV columns"):
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_typo_in_prefixed_column(
            start_from_scratch=True
        )


def test_system_dot_prefix_column_recognized_and_synthesizes_yaml(
    synth_sensitivity_with_system_overlay,
):
    """Phase 1 R1, R5, R6 — system.target_dem_resolution synthesizes two per-target YAMLs."""
    analysis = synth_sensitivity_with_system_overlay
    generated_dir = analysis.analysis_paths.analysis_dir / "_generated"
    assert generated_dir.exists()
    target_yamls = sorted(generated_dir.glob("target_*.yaml"))
    assert len(target_yamls) == 2  # two distinct overlay values, two targets
    cfgs = [_yaml.safe_load(p.read_text()) for p in target_yamls]
    assert {c["target_dem_resolution"] for c in cfgs} == {1.0, 2.0}


def test_fingerprint_payload_includes_system_overlay(
    synth_sensitivity_with_system_overlay,
):
    """Phase 1 R7 — fingerprint payload attaches system_overlay key + schema 3."""
    analysis = synth_sensitivity_with_system_overlay
    sub_analysis = analysis.sensitivity.sub_analyses["0"]
    payload = analysis.sensitivity._compute_sa_id_fingerprint_payload(sub_analysis)
    assert payload["__schema_version__"] == 3
    assert "system_overlay" in payload
    assert payload["system_overlay"].get("target_dem_resolution") in {1.0, 2.0}


def test_build_unique_system_targets_skips_purge_in_runner_subprocess(
    synth_sensitivity_with_system_overlay, monkeypatch,
):
    """Phase 1 R-P1-4 — is_main_orchestrator=False skips fast_rmtree of _generated/."""
    from TRITON_SWMM_toolkit import utils as _utils_mod
    analysis = synth_sensitivity_with_system_overlay
    generated_dir = analysis.analysis_paths.analysis_dir / "_generated"
    assert generated_dir.exists()
    called: list[Path] = []
    def fake_rmtree(p, *args, **kwargs):
        called.append(Path(p))
    monkeypatch.setattr(_utils_mod, "fast_rmtree", fake_rmtree)
    analysis.sensitivity._build_unique_system_targets(
        analysis.sensitivity._df_setup_full,
        is_main_orchestrator=False,
    )
    assert generated_dir not in called, (
        f"fast_rmtree was called on _generated/ in runner-subprocess mode: {called}"
    )
