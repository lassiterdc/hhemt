"""Synthetic-model sensitivity + Snakemake tier. Mirror of test_PC_05 using synth fixtures."""

from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut

pytestmark = [
    pytest.mark.requires_snakemake_subprocess,
    pytest.mark.skipif(
        tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
    ),
    pytest.mark.skipif(
        tst_ut.compile_toolchain_unavailable() and not tst_ut.require_compile_tier(),
        reason=(
            "TRITON-SWMM/SWMM compile toolchain (cmake + mpic++) absent; this "
            "module compiles at run time via analysis.run(). Runs under the "
            "hhemt conda env; HHEMT_REQUIRE_COMPILE_TIER=1 turns absence into a "
            "hard failure."
        ),
    ),
]

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
        'expand("plots/sensitivity/benchmarking/benchmarking__{independent_var}.vs.total.html"'
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


def test_reprocess_render_report_rule_carries_reprocess_flag(synth_sensitivity_analysis):
    """Generation assertion (reprocess-render-report-snakefile-selection, R3/R5).

    The REPROCESS master Snakefile's ``rule render_report`` shell MUST pass
    ``--reprocess`` to ``render_report_runner`` so ``render_report()`` selects
    ``Snakefile.reprocess``; the PRODUCTION master Snakefile's ``rule
    render_report`` shell MUST NOT (keeping the production render path
    byte-identical). Guards against the synergy refactor silently dropping the
    flag.
    """
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder

    reprocess_content = builder.generate_reprocess_master_snakefile_content(which="both", start_with="render")
    production_content = builder.generate_master_snakefile_content(which="both", compression_level=5)

    reprocess_block = _render_report_rule_block(reprocess_content)
    production_block = _render_report_rule_block(production_content)

    assert "render_report_runner" in reprocess_block, "reprocess render_report rule must invoke the runner"
    assert "--reprocess" in reprocess_block, (
        "reprocess master render_report shell MUST pass --reprocess so render_report() "
        "selects Snakefile.reprocess (R3/R5)"
    )
    assert "render_report_runner" in production_block, "production render_report rule must invoke the runner"
    assert "--reprocess" not in production_block, (
        "production master render_report shell MUST NOT pass --reprocess "
        "(byte-identical production render path; R3/R5)"
    )


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
                "compression_level": 7,
            },
            [
                "--compression-level 7",
                "--which TRITON",
                "--consolidate-sensitivity-analysis-outputs",
            ],
        ),
        (
            {
                "which": "both",
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
        override_clear_raw="all",
        compression_level=5,
        pickup_where_leftoff=False,
        verbose=True,
    )

    assert result["success"], f"Workflow submission failed: {result.get('message', '')}"

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)

    analysis_dir = analysis.analysis_paths.analysis_dir
    for indep_var in ("n_devices",):
        figure = analysis_dir / "plots" / "sensitivity" / "benchmarking" / f"benchmarking__{indep_var}.vs.total.html"
        assert figure.exists(), f"Expected benchmarking figure missing: {figure}"


@pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)
@pytest.mark.slow
def test_reprocess_process_self_heals_deleted_summary(synth_sensitivity_analysis):
    """Regression (reprocess-rebuild-divergence-fix).

    The divergence state is: a per-scenario summary output was deleted, but its
    ``d_process_*`` completion flag survived. ``reprocess(start_with="process")``
    must self-heal it — reconcile the flag + per-model processing log against
    on-disk summary presence — and rebuild the missing summary, REGARDLESS of
    ``regenerate_existing`` (the May-31 failure ran the default ``False``).

    Asserts:
    - R1/R2/R4/R6 (rebuild): delete one summary zarr while leaving its
      ``d_process`` flag → ``reprocess(start_with="process",
      regenerate_existing=False)`` rebuilds the summary, re-runs consolidate,
      and reproduces the analysis summaries.
    - R3 (healthy no-op): every OTHER ``d_process`` flag retains its
      pre-reprocess mtime (only the divergent pair was unlinked).
    - First-run idempotence: a SECOND ``reprocess(start_with="process",
      dry_run=True)`` emits ZERO ``process_sa_*`` rules (the gate is now closed
      for all events) — asserted via a deterministic Snakefile.reprocess parse.
    """
    import shutil

    import hhemt.analysis as _analysis_mod
    from hhemt.constants import (
        consolidate_subanalysis_flag,
        process_timeseries_flag_per_sa,
    )
    from hhemt.scenario import (
        TRITONSWMM_scenario,
        compute_event_id_slug,
    )

    # Guard: this regression test is meaningless unless it exercises the SOURCE
    # under test (the worktree edits), not a stale installed copy. Assert the
    # loaded analysis module is the worktree's and carries the new self-heal
    # helper before doing any expensive build/run work.
    assert hasattr(
        _analysis_mod.TRITONSWMM_analysis,
        "_reconcile_stale_process_flags_against_summaries",
    ), (
        "self-heal helper missing from the loaded TRITONSWMM_analysis — the test "
        f"is exercising the wrong source copy: {_analysis_mod.__file__}"
    )

    # model_type -> consolidate-consumed summary attrs (mirrors the production
    # self-heal's D3 predicate in analysis.py).
    _SUMMARY_ATTRS_BY_MODEL = {
        "tritonswmm": (
            "output_tritonswmm_triton_summary",
            "output_tritonswmm_node_summary",
            "output_tritonswmm_link_summary",
            "output_tritonswmm_performance_summary",
        ),
        "triton": (
            "output_triton_only_summary",
            "output_triton_only_performance_summary",
        ),
        "swmm": (
            "output_swmm_only_node_summary",
            "output_swmm_only_link_summary",
        ),
    }

    analysis = synth_sensitivity_analysis

    # ── Arrange: run the full sensitivity workflow to completion so real
    # summary zarrs exist on disk (mirrors test_snakemake_sensitivity_workflow_
    # execution). ──────────────────────────────────────────────────────────────
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
        # Preserve raw sim outputs (out_tritonswmm/*): the reprocess rebuild
        # re-reads them to regenerate the deleted summary. Clearing raw here
        # ("all") would delete the rebuild source and make the rebuild assertion
        # unsatisfiable — that is the separate raw-also-gone case the
        # _assert_reprocess_rebuild_sources_present preflight guards, not the
        # divergence-rebuild path this test validates.
        override_clear_raw="none",
        compression_level=5,
        pickup_where_leftoff=False,
        verbose=True,
    )
    assert result["success"], f"Workflow submission failed: {result.get('message', '')}"
    tst_ut.assert_analysis_summaries_created(analysis)

    master_dir = analysis.analysis_paths.analysis_dir
    status_dir = master_dir / "_status"

    # Pick one sub-analysis + its first event + its first enabled model, and
    # resolve a present summary output to delete.
    # Sub-analyses are owned by the sensitivity manager; the top-level
    # TRITONSWMM_analysis delegates reprocess() to self.sensitivity.reprocess().
    # Iterate items() (keys may be int) and string-cast the chosen id for the
    # flag-token helpers, which validate a str sa_id fragment.
    sa_items = list(analysis.sensitivity.sub_analyses.items())
    assert sa_items, "fixture produced no sub-analyses"
    target_sa_key, sub = sa_items[0]
    target_sa = str(target_sa_key)

    scen = TRITONSWMM_scenario(0, sub)
    evt = compute_event_id_slug(
        sub._retrieve_weather_indexer_using_integer_index(0)
    )
    model_type = scen.run.model_types_enabled[0]

    summary_to_delete = None
    for attr in _SUMMARY_ATTRS_BY_MODEL[model_type]:
        p = getattr(scen.scen_paths, attr, None)
        if p is not None and p.exists():
            summary_to_delete = p
            break
    assert summary_to_delete is not None, (
        f"no present summary output found to delete for model_type={model_type}"
    )

    flag_path = master_dir / process_timeseries_flag_per_sa(model_type, target_sa, evt)
    assert flag_path.exists(), f"expected d_process flag present before delete: {flag_path}"

    # Create the divergence: delete the summary, leave the flag.
    if summary_to_delete.is_dir():
        shutil.rmtree(summary_to_delete)
    else:
        summary_to_delete.unlink()
    assert not summary_to_delete.exists(), "summary should be deleted"
    assert flag_path.exists(), (
        "d_process flag must survive summary deletion (this IS the divergence state)"
    )

    # Record all d_process flag mtimes for the no-op (R3) arm.
    pre_mtimes = {
        f: f.stat().st_mtime for f in status_dir.glob("d_process_*_complete.flag")
    }

    # ── Act: reprocess on the process path with the DEFAULT
    # regenerate_existing=False (the configuration that originally failed). ──────
    analysis.reprocess(
        start_with="process", execution_mode="local", regenerate_existing=False
    )

    # ── Assert (rebuild, R1/R2/R4/R6). ─────────────────────────────────────────
    assert summary_to_delete.exists(), (
        "self-heal must rebuild the deleted summary on the process path"
    )
    consolidate_flag = master_dir / consolidate_subanalysis_flag(target_sa)
    assert consolidate_flag.exists(), (
        f"per-sa consolidate flag must be present after reprocess: {consolidate_flag}"
    )
    tst_ut.assert_analysis_summaries_created(analysis)

    # ── Assert (healthy no-op, R3): every OTHER d_process flag keeps its
    # pre-reprocess mtime; only the divergent pair was unlinked + rebuilt. ──────
    for f, mtime in pre_mtimes.items():
        if f == flag_path:
            continue  # the healed flag was unlinked + rebuilt (mtime expected to change)
        assert f.exists(), f"healthy d_process flag unexpectedly missing: {f}"
        assert f.stat().st_mtime == mtime, (
            f"healthy d_process flag mtime changed unexpectedly (should be a no-op): {f}"
        )

    # ── Assert (first-run idempotence): a SECOND process reprocess (dry-run)
    # emits ZERO process_sa_ rules — the gate's `not d_process_path.exists()` is
    # now False for every event. Parse the generated Snakefile.reprocess
    # (deterministic; does not depend on snakemake dry-run stats formatting). ───
    analysis.reprocess(start_with="process", execution_mode="local", dry_run=True)
    snakefile = master_dir / "Snakefile.reprocess"
    assert snakefile.exists(), f"reprocess Snakefile not generated: {snakefile}"
    snakefile_text = snakefile.read_text()
    assert "rule process_sa_" not in snakefile_text, (
        "steady-state second reprocess must emit zero process_sa_ rules "
        "(all d_process flags now present)"
    )


@pytest.mark.skipif(tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems.")
@pytest.mark.slow
def test_master_consolidation_tolerates_incomplete_subanalysis(synth_sensitivity_analysis):
    """Regression (sensitivity-consolidation-tolerate-incomplete).

    Reproduces the FUNCTION-LEVEL crash behind the live uva_sensitivity_suite
    master_consolidation FileNotFoundError: consolidate_sensitivity_datatree
    raising when one sub-analysis is incomplete (its per-scenario summary was
    never produced, its analysis_datatree.zarr was never built, and its
    datatree_consolidation_complete log is False — the real never-consolidated
    state). This test calls consolidate_sensitivity_datatree directly (the
    unit-level gate for the function under change); the consolidate_workflow.py
    call site inherits the True default UNCHANGED under Decision D2, so that
    default-inheritance wiring is covered by the end-to-end synth_05 run (V2),
    not by this unit test. Master consolidation must still assemble
    sensitivity_datatree.zarr over the COMPLETED subset under
    allow_incomplete=True (the default per Decision D2), skipping the incomplete sub.

    Asserts R2/R4 (tolerant regen over the completed subset; experiment definition
    preserved in the root `parameters` node) and R3 (fail-fast when allow_incomplete=False).
    """
    import shutil

    import xarray as xr

    from hhemt.scenario import TRITONSWMM_scenario

    _SUMMARY_ATTRS_BY_MODEL = {
        "tritonswmm": (
            "output_tritonswmm_triton_summary",
            "output_tritonswmm_node_summary",
            "output_tritonswmm_link_summary",
            "output_tritonswmm_performance_summary",
        ),
        "triton": (
            "output_triton_only_summary",
            "output_triton_only_performance_summary",
        ),
        "swmm": (
            "output_swmm_only_node_summary",
            "output_swmm_only_link_summary",
        ),
    }

    analysis = synth_sensitivity_analysis

    # Arrange: full sensitivity run so real per-sub datatrees + master tree exist.
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
        override_clear_raw="all",
        compression_level=5,
        pickup_where_leftoff=False,
        verbose=True,
    )
    assert result["success"], f"Workflow submission failed: {result.get('message', '')}"

    sensitivity = analysis.sensitivity
    sa_items = list(sensitivity.sub_analyses.items())
    assert len(sa_items) >= 2, "test needs >=2 sub-analyses (one incomplete, one complete)"

    master_zarr = sensitivity.analysis_paths.sensitivity_datatree_zarr
    assert master_zarr is not None and master_zarr.exists(), "master tree should exist after full run"

    incomplete_key, incomplete_sub = sa_items[0]
    complete_key, _complete_sub = sa_items[-1]
    incomplete_id = str(incomplete_key)
    complete_id = str(complete_key)
    prefix = sensitivity.sub_analyses_prefix  # e.g. "sa_"

    # Induce the *never-consolidated* state for the incomplete sub:
    #   (a) delete one per-scenario summary  -> _retrieve_combined_output raises FileNotFoundError
    #   (b) delete its analysis_datatree.zarr -> eager pre-build loop attempts a rebuild
    #   (c) clear datatree_consolidation_complete -> build_sensitivity_datatree's open_datatree
    #       raises ValueError (the real "never built" branch), caught by its except ValueError.
    scen = TRITONSWMM_scenario(0, incomplete_sub)
    model_type = scen.run.model_types_enabled[0]
    summary_to_delete = None
    for attr in _SUMMARY_ATTRS_BY_MODEL[model_type]:
        p = getattr(scen.scen_paths, attr, None)
        if p is not None and p.exists():
            summary_to_delete = p
            break
    assert summary_to_delete is not None, f"no present summary for model_type={model_type}"
    if summary_to_delete.is_dir():
        shutil.rmtree(summary_to_delete)
    else:
        summary_to_delete.unlink()

    sub_datatree = incomplete_sub.analysis_paths.analysis_datatree_zarr
    assert sub_datatree is not None and sub_datatree.exists()
    shutil.rmtree(sub_datatree)

    incomplete_sub._refresh_log()
    incomplete_sub.log.datatree_consolidation_complete.set(False)

    # consolidate_sensitivity_datatree early-returns if the master zarr exists.
    shutil.rmtree(master_zarr)

    # Act: tolerant consolidation (default True; pass explicitly for clarity).
    out = sensitivity.consolidate_sensitivity_datatree(allow_incomplete=True)

    # Assert R2/R4.
    assert out.exists(), "master tree must regenerate over the completed subset"
    tree = xr.open_datatree(out, engine="zarr", consolidated=False)
    sa_nodes = {c for c in tree.children if c.startswith(prefix)}
    assert f"{prefix}{incomplete_id}" not in sa_nodes, (
        f"incomplete sub {prefix}{incomplete_id} must be absent from the master tree nodes"
    )
    assert f"{prefix}{complete_id}" in sa_nodes, (
        f"complete sub {prefix}{complete_id} must be present in the master tree nodes"
    )
    assert len(sa_nodes) == len(sa_items) - 1, (
        "exactly one sub-analysis (the incomplete one) must be absent from the tree nodes"
    )
    # The root `parameters` node (experiment definition) still lists every sub-analysis.
    assert "parameters" in tree.children, "root `parameters` node must be present"
    assert len(tree["parameters"].to_dataset().to_dataframe()) == len(sensitivity.df_setup), (
        "root `parameters` Dataset must list every defined sub-analysis"
    )

    # Assert R3: fail-fast preserved when allow_incomplete=False.
    shutil.rmtree(out)
    with pytest.raises((FileNotFoundError, ValueError)):
        sensitivity.consolidate_sensitivity_datatree(allow_incomplete=False)


# ─── Phase 7: Snakemake report integration tests (sensitivity master) ──────────

from pathlib import Path as _Path  # noqa: E402

_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7 = (
    _Path(__file__).resolve().parents[1] / "configs" / "reports" / "synth_sensitivity_report_config.yaml"
)


def test_synth_sensitivity_report_inlined_passes_run_entry_validation():
    """R2 (DECISION 1 = Option A — D2 consolidation): the BARE ``_TestCaseBuilder``
    synth-sensitivity construction (no catalog report injection) must inline
    ``cfg_analysis.report`` so the ``run()``-entry cross-validation passes.

    Red-before-green characterization (SE F-B 2): with the ~13 synth-wrapper
    ``additional_analysis_configs={"report": ...}`` injections removed from
    ``test_case_catalog.py``, this path routes through the bare builder. PRE-D2 the
    builder emitted ``report: {}`` -> ``cfg_analysis.report.sensitivity is None`` ->
    ``validate_sensitivity_independent_vars`` raises ``ConfigurationError`` (verified
    RED before the ``test_case_builder.py`` inline landed). POST-D2 the builder
    inlines ``report_config_synth_sensitivity.yaml``, so ``report.sensitivity`` is
    populated and validation passes. This is the test the consolidation makes
    meaningful — the bare-builder path is now the standard synth construction, so a
    pattern-match against the (previously report-injecting) catalog wrappers would
    have been tautologically green.
    """
    from pathlib import Path

    import tests.fixtures.test_case_catalog as cases
    from hhemt.config.report import validate_sensitivity_independent_vars

    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
        start_from_scratch=False, skip_run=True
    )
    analysis = case.analysis

    # Post-D2: the bare builder inlines the synth sensitivity report block.
    assert analysis.cfg_analysis.report.sensitivity is not None, (
        "D2 builder-inline regressed: bare _TestCaseBuilder emitted an empty "
        "report block (cfg_analysis.report.sensitivity is None)"
    )

    # Mirrors the run()-entry cross-validation: analysis.run() resolves
    # cfg_report = self.cfg_analysis.report and sa_csv = cfg_analysis.sensitivity_analysis.
    sa_csv = Path(analysis.cfg_analysis.sensitivity_analysis)
    validate_sensitivity_independent_vars(analysis.cfg_analysis.report, sa_csv)


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
    assert any(bench_dir.glob("benchmarking__*.vs.total.html"))

    # R13: no per-sub-analysis report
    for sa_id, sub in analysis.sensitivity.sub_analyses.items():
        sub_html = sub.analysis_paths.analysis_dir / "analysis_report.html"
        assert not sub_html.exists(), (
            f"unexpected per-sub-analysis report at {sub_html} for sa_id={sa_id}"
        )

    # OE-3 / R8: master-tree membership parity (silent-drop guard).
    # After a FULL clean consolidation (all subs present), every sub-analysis
    # whose analysis_datatree.zarr exists on disk MUST appear as a node in the
    # master sensitivity_datatree.zarr. Asserting the on-disk-zarr set is a
    # subset of the master-tree-node set catches the silent-drop class (Gotcha 36
    # `allow_incomplete=True` lets build_sensitivity_datatree drop a skipped sub
    # while returning success=True) against on-disk ground truth -- independent
    # of the clobberable/stale in-memory + log completion flags this plan fixed.
    import xarray as xr

    sensitivity = analysis.sensitivity
    prefix = sensitivity.sub_analyses_prefix
    master_zarr = sensitivity.analysis_paths.sensitivity_datatree_zarr
    assert master_zarr is not None and master_zarr.exists(), (
        "master sensitivity_datatree.zarr must exist after a full clean run"
    )
    master_tree = xr.open_datatree(master_zarr, engine="zarr", consolidated=False)
    tree_sa_ids = {
        c.removeprefix(prefix) for c in master_tree.children if c.startswith(prefix)
    }
    on_disk_sa_ids = {
        str(sa_id)
        for sa_id, sub in sensitivity.sub_analyses.items()
        if sub.analysis_paths.analysis_datatree_zarr is not None
        and sub.analysis_paths.analysis_datatree_zarr.exists()
    }
    assert on_disk_sa_ids <= tree_sa_ids, (
        f"silent-drop: subs on disk but missing from master tree: {on_disk_sa_ids - tree_sa_ids}"
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
def test_synth_render_report_interactive_zip(synth_sensitivity_analysis_cached, tmp_path):
    # ZIP-fallback path: render_report(format="zip") writes
    # `analysis_report.zip` containing `report.html` + per-figure
    # `data/*.html` entries. Asserts the bundle carries >= 6 data/*.html
    # entries (one per interactive figure under the master plan's expected
    # shape).
    import zipfile
    from pathlib import Path

    analysis = synth_sensitivity_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7),
    )
    zip_path = analysis.sensitivity.render_report(format="zip")
    assert zip_path.exists() and zip_path.suffix == ".zip"
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    # Snakemake's --report zip lays figures out under
    # `analysis_report/data/raw/<hash>/<figure>.html`. Match anywhere under
    # `data/` to stay robust to root-prefix changes across Snakemake versions.
    html_entries = [n for n in names if "/data/" in n and n.endswith(".html")]
    assert len(html_entries) >= 6, (
        f"Expected >= 6 data/*.html entries in ZIP, got {len(html_entries)}: {html_entries}"
    )


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

import yaml as _yaml  # noqa: E402

import tests.fixtures.test_case_catalog as _cases  # noqa: E402
from hhemt.exceptions import ConfigurationError  # noqa: E402


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
    """Legacy `gpu_hardware_override` column is rejected as unknown.

    Phase 6: the prior migration target `system.gpu_hardware` is itself retired
    (Phase 4, R7 — gpu_hardware is now partition-DERIVED). The column is rejected
    as unknown; the message lists the valid columns (including the `hpc.partition`
    alias that selects GPU hardware via the partition spec)."""
    with pytest.raises(
        ConfigurationError, match="Unknown sensitivity-CSV columns"
    ) as excinfo:
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_legacy_gpu_hardware_override(
            start_from_scratch=True
        )
    assert "gpu_hardware_override" in str(excinfo.value)


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


def test_df_setup_with_system_overlays_carries_prefixed_system_columns(
    synth_sensitivity_with_system_overlay,
):
    """Accessor unions df_setup with system.* overlay columns (prefixed names retained)."""
    sens = synth_sensitivity_with_system_overlay.sensitivity
    # system.* column is dropped from the analysis-filtered df_setup ...
    assert "system.target_dem_resolution" not in sens.df_setup.columns
    # ... but present (prefixed) in the overlay-union accessor ...
    overlay_frame = sens.df_setup_with_system_overlays
    assert "system.target_dem_resolution" in overlay_frame.columns
    # ... and the analysis-config columns are preserved too.
    assert "n_mpi_procs" in overlay_frame.columns
    # Index parity with df_setup (same sa_ids, same order).
    assert list(overlay_frame.index) == list(sens.df_setup.index)


# ============================================================================
# Phase 2 of prefixed_column_config_variation — tests
# ============================================================================


def test_bare_name_analysis_config_column_emits_deprecation_warning():
    """Phase 2 R2 — bare-name analysis-config columns emit DeprecationWarning at construction."""
    # The default synth sensitivity CSV uses bare names (run_mode, n_mpi_procs, etc.),
    # so a fresh load must surface a DeprecationWarning naming a bare analysis-config column.
    with pytest.warns(DeprecationWarning, match=r"Bare-name analysis-config column"):
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(
            start_from_scratch=True
        )


def test_analysis_dot_prefix_column_accepted_silently(synth_sensitivity_all_analysis_prefixed):
    """Phase 2 R2 — `analysis.{field}` columns accepted without DeprecationWarning."""
    import warnings
    analysis = synth_sensitivity_all_analysis_prefixed
    # Construction already completed via the fixture. Re-derive the sub-analyses under
    # an explicit DeprecationWarning-as-error filter to confirm no bare-name warning fires
    # for the all-prefixed fixture.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ = analysis.sensitivity._create_sub_analyses()


def test_attributes_varied_for_analysis_split(synth_sensitivity_mixed_prefixed_columns):
    """Phase 2 R10 — `analysis_independent_vars` and `system_independent_vars` are separate."""
    analysis = synth_sensitivity_mixed_prefixed_columns
    assert "n_omp_threads" in analysis.sensitivity.analysis_independent_vars  # bare-name in fixture
    assert "n_mpi_procs" in analysis.sensitivity.analysis_independent_vars  # analysis.n_mpi_procs in fixture
    assert "target_dem_resolution" in analysis.sensitivity.system_independent_vars
    assert "target_dem_resolution" not in analysis.sensitivity.analysis_independent_vars


def test_build_unique_system_targets_skips_purge_in_runner_subprocess(
    synth_sensitivity_with_system_overlay, monkeypatch,
):
    """Phase 1 R-P1-4 — is_main_orchestrator=False skips fast_rmtree of _generated/."""
    from hhemt import utils as _utils_mod
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


@pytest.mark.skipif(tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems.")
@pytest.mark.slow
def test_reprocess_render_report_over_partial_completion(synth_sensitivity_analysis):
    """Behavioral regression (reprocess-render-report-snakefile-selection, R4).

    Reproduces the live uva_sensitivity_suite failure: ``reprocess(start_with=
    "render")`` over a PARTIAL-completion sensitivity suite hard-failed because
    ``render_report()`` ran ``snakemake --report`` against the PRODUCTION
    Snakefile (which enumerates ``report()`` targets for EVERY defined sub,
    including a never-completed one), so the report engine's existence gate
    (``report/__init__.py``) aborted on the first absent ``report()``-flagged
    figure with a ``WorkflowError``. The fix routes the reprocess render to
    ``Snakefile.reprocess`` (Gotcha 37 filters report targets to the completed
    subset), so the report renders over exactly the completed subset.

    Makes the differential real: deletes one sub's per-scenario summary (so the
    reprocess generator's summary-existence filter EXCLUDES it from
    ``Snakefile.reprocess``) AND its per-sim report figures under the master
    ``plots/`` tree (so the UNFIXED production-Snakefile path would raise
    ``WorkflowError`` on the missing figure). With the fix, ``reprocess(
    start_with="render")`` reads ``Snakefile.reprocess``, never enumerates the
    deleted figures, and re-renders ``analysis_report.zip`` over the completed
    subset with no error.
    """
    import shutil
    from pathlib import Path

    from hhemt.scenario import TRITONSWMM_scenario

    _SUMMARY_ATTRS_BY_MODEL = {
        "tritonswmm": (
            "output_tritonswmm_triton_summary",
            "output_tritonswmm_node_summary",
            "output_tritonswmm_link_summary",
            "output_tritonswmm_performance_summary",
        ),
        "triton": (
            "output_triton_only_summary",
            "output_triton_only_performance_summary",
        ),
        "swmm": (
            "output_swmm_only_node_summary",
            "output_swmm_only_link_summary",
        ),
    }

    analysis = synth_sensitivity_analysis

    # Arrange: full local sensitivity run WITH a report config so per-sim
    # figures + the master report are materialized for every sub.
    analysis.run(
        from_scratch=True,
        execution_mode="local",
        report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7),
        report_formats=["zip"],
        verbose=True,
    )

    sensitivity = analysis.sensitivity
    sa_items = list(sensitivity.sub_analyses.items())
    assert len(sa_items) >= 2, "test needs >=2 sub-analyses (one incomplete, one complete)"

    master_dir = sensitivity.master_analysis.analysis_paths.analysis_dir
    report_zip = master_dir / "analysis_report.zip"
    assert report_zip.exists(), "master analysis_report.zip should exist after the full run"

    incomplete_key, incomplete_sub = sa_items[0]
    incomplete_id = str(incomplete_key)

    # (a) Delete one per-scenario summary so the reprocess generator's
    #     summary-existence filter (Gotcha 37) EXCLUDES this sub from
    #     Snakefile.reprocess's report-target enumeration.
    scen = TRITONSWMM_scenario(0, incomplete_sub)
    model_type = scen.run.model_types_enabled[0]
    summary_to_delete = None
    for attr in _SUMMARY_ATTRS_BY_MODEL[model_type]:
        p = getattr(scen.scen_paths, attr, None)
        if p is not None and p.exists():
            summary_to_delete = p
            break
    assert summary_to_delete is not None, f"no present summary for model_type={model_type}"
    if summary_to_delete.is_dir():
        shutil.rmtree(summary_to_delete)
    else:
        summary_to_delete.unlink()

    # (b) Delete the incomplete sub's per-sim report figures so the UNFIXED
    #     production-Snakefile render path would raise WorkflowError on the
    #     missing report()-flagged figure. Figures live at
    #     plots/sensitivity/per_sim/sa-{id}/{event_id}/*.{html,png}.
    incomplete_fig_dir = master_dir / "plots" / "sensitivity" / "per_sim" / f"sa-{incomplete_id}"
    assert incomplete_fig_dir.exists(), (
        f"precondition: incomplete sub per-sim figures must exist after the full run: {incomplete_fig_dir}"
    )
    shutil.rmtree(incomplete_fig_dir)

    # Act: reprocess the RENDER stage. With the fix this reads
    # Snakefile.reprocess (completed subset only); pre-fix it read the
    # production Snakefile and raised WorkflowError on the deleted figure.
    reprocess_result = analysis.reprocess(
        start_with="render",
        execution_mode="local",
        report_formats=["zip"],
        verbose=True,
    )

    # Assert R4: render succeeds over the completed subset, no WorkflowError.
    assert reprocess_result["success"], (
        "reprocess(start_with='render') over a partial-completion suite must render the "
        f"report over the completed subset with no WorkflowError; got "
        f"{reprocess_result.get('message')!r}. Snakemake log: {reprocess_result.get('snakemake_logfile')}"
    )
    assert report_zip.exists() and report_zip.stat().st_size > 0, (
        "master analysis_report.zip must exist after the reprocess render"
    )


@pytest.mark.skipif(
    not tst_ut.provenance_audit_enabled(),
    reason="renderer-IO provenance audit is opt-in (ADR-18); set "
    "HHEMT_ENABLE_PROVENANCE_AUDIT=1 to enable the audit AND run this test.",
)
@pytest.mark.slow
def test_renderer_provenance_audit_passes_for_all_sensitivity_renderers(synth_sensitivity_analysis_cached):
    """Sensitivity-tier audit-passes guard — exercises sensitivity_benchmarking
    (Plotly branch), the rebased per-sa per-sim renderers, and errors_and_warnings /
    per_analysis_summary reading the persisted validation_report.json (Option D).

    Master plots are deleted before the second run() to force a fresh render through
    the audited _cli subprocess path (the _cached fixture key is the swmm-topology
    SHA, not the renderer source). render_report() requires every report()-flagged
    figure to exist (Gotcha 39), so its success is the audit-passed-for-every-renderer
    signal: a single renderer's audit ProcessingError leaves its figure missing.
    """
    import shutil
    from pathlib import Path

    analysis = synth_sensitivity_analysis_cached
    analysis.run(from_scratch=False, report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7))

    master_dir = analysis.sensitivity.master_analysis.analysis_paths.analysis_dir
    plots_dir = master_dir / "plots"
    shutil.rmtree(plots_dir, ignore_errors=True)
    analysis.run(from_scratch=False, report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7))

    out_html = analysis.sensitivity.render_report(format="html")
    assert out_html.exists() and out_html.stat().st_size > 0, (
        "sensitivity render_report failed after a forced fresh render — likely an "
        f"audit ProcessingError; inspect plot-rule logs under {plots_dir}"
    )
    manifests = list(plots_dir.rglob("*.manifest.json"))
    assert manifests, "no manifest sidecars (sensitivity renderers did not re-run)"
