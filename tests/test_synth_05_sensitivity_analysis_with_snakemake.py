"""Synthetic-model sensitivity + Snakemake tier. Mirror of test_PC_05 using synth fixtures.

Checks, for a sensitivity analysis on the synthetic model and on a committed golden tree: master and member Snakefile
generation, the reprocess self-heal and render paths, report rendering, EDA rule emission for the reporting sets, and
the ``system.`` / ``analysis.`` sensitivity-CSV columns.
"""

import os
import sys
from pathlib import Path

import pytest

import tests.utils_for_testing as tst_ut
from tests.fixtures.test_case_builder import _TEST_RUNS_ROOT_OVERRIDE_ENV

pytestmark = [pytest.mark.requires_snakemake_subprocess]

_TOOLCHAIN_ABSENT_REASON = (
    "TRITON-SWMM/SWMM compile toolchain (cmake + mpic++) absent. Runs under the "
    "hhemt conda env; HHEMT_REQUIRE_COMPILE_TIER=1 turns absence into a "
    "hard failure."
)


# Rows that build a sensitivity configuration, expect a ``ConfigurationError`` while the configuration is built, and
# compile and run nothing. They need no toolchain, so the gate below does not skip them. Exactly these names are
# exempt; any other row of this module that is not a golden row still skips without the toolchain.
_NO_TOOLCHAIN_ROWS = frozenset(
    {
        "test_system_overlay_mutual_exclusion_with_system_config_yaml",
        "test_system_overlay_validator_re_fire_invalid_value",
        "test_gpu_hardware_override_column_raises_migration_error",
        "test_unknown_column_rejected_with_nearest_match_hint",
        "test_bare_name_analysis_config_column_is_refused_naming_the_prefixed_form",
    }
)


@pytest.fixture
def _toolkit_identity_not_dirty(monkeypatch):
    """Let the analysis constructor see the running toolkit identity with ``dirty`` cleared; the sha is unchanged.

    The constructor refuses a checkout with modified tracked files before it reaches the error these rows assert, so
    on a working tree under edit each row would fail on that refusal. The refusal is about which toolkit runs, not
    about the configuration under test, and ``golden_extractor`` neutralises it the same way."""
    import dataclasses

    import hhemt.analysis as analysis_module

    real = analysis_module.running_identity
    monkeypatch.setattr(analysis_module, "running_identity", lambda: dataclasses.replace(real(), dirty=False))


@pytest.fixture(autouse=True)
def _compile_toolchain_gate(request):
    """Skip a test that needs the compile toolchain when it is absent (the module-level ``skipif`` this replaces).

    The rows that run on the committed golden need no toolchain (they compile and run no solver), so they are
    exempt: a module-level skip would silently drop them on any machine without cmake and mpic++. The rows named in
    ``_NO_TOOLCHAIN_ROWS`` are exempt for the same reason."""
    if {"golden_sensitivity_analysis", "golden_sensitivity_analysis_keeping_scenarios"} & set(request.fixturenames):
        return
    if request.node.name in _NO_TOOLCHAIN_ROWS:
        return
    if tst_ut.compile_toolchain_unavailable() and not tst_ut.require_compile_tier():
        pytest.skip(_TOOLCHAIN_ABSENT_REASON)


# The identity shim ``_IDENTITY_SHIM`` and the fixtures ``golden_reprocess_env`` and ``golden_planned`` below supply
# what a reprocess over a committed golden tree needs and a tree built by a real run already has. The shim string is
# copied into other test modules that reprocess a golden, so one grep for ``_IDENTITY_SHIM`` finds every copy.
# Each prerequisite here is needed by the rows that request it, and none is included for a row that does not need
# it. In particular the consolidation-build escape (HHEMT_DECLARE_STALE_BUILD) is absent on purpose: no row below
# asserts the modification time of a consolidated store, which is the only thing that escape protects.
#
# Child interpreters (the reprocess rules) refuse a toolkit checkout with uncommitted tracked changes, because each
# constructs an analysis (``hhemt.validation.running_identity`` -> ``dirty``). The golden fixtures clear ``dirty`` in
# THIS process only, so this ``sitecustomize`` shim, put first on the children's ``PYTHONPATH``, clears it once
# ``hhemt.validation`` is imported. It never touches the sha, and on a clean tree it is the identity.
_IDENTITY_SHIM = """
import dataclasses
import importlib.abc
import importlib.machinery
import sys


class _ClearDirty(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "hhemt.validation":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module

        def patched(module):
            exec_module(module)
            real = module.running_identity
            module.running_identity = lambda: dataclasses.replace(real(), dirty=False)

        loader.exec_module = patched
        return spec


sys.meta_path.insert(0, _ClearDirty())
"""


@pytest.fixture
def golden_reprocess_env(monkeypatch, tmp_path):
    """The child-interpreter environment of a golden reprocess: the identity shim first on ``PYTHONPATH``, the
    interpreter's directory first on ``PATH`` (the sensitivity Snakefile calls a bare ``python``)."""
    shim_dir = tmp_path / "identity_shim"
    shim_dir.mkdir()
    (shim_dir / "sitecustomize.py").write_text(_IDENTITY_SHIM)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(shim_dir), os.environ.get("PYTHONPATH", "")])))
    monkeypatch.setenv("PATH", os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")]))


@pytest.fixture
def golden_planned(golden_sensitivity_analysis_keeping_scenarios, golden_reprocess_env):
    """The sensitivity golden with the Snakefile a golden does not carry (tests/goldens/SPEC.md section 9), written by
    a dry run.

    Needed by ``test_reprocess_render_report_over_partial_completion`` (the report rules read it);
    ``test_reprocess_process_self_heals_deleted_summary`` runs without it."""
    from tests.fixtures.golden_trees import dry_run_plan

    dry_run_plan(golden_sensitivity_analysis_keeping_scenarios)
    return golden_sensitivity_analysis_keeping_scenarios


def test_snakemake_sensitivity_workflow_generation_and_write(
    synth_sensitivity_analysis,
):
    """
    Test Snakemake workflow generation for sensitivity analysis.

    Verifies that:
    1. Member Snakefiles are generated and written correctly
    2. Master Snakefile is generated and written correctly
    3. Master Snakefile contains required rules and flags
    """
    experiment = synth_sensitivity_analysis

    assert experiment.cfg_analysis.toggle_sensitivity_analysis is True
    assert hasattr(experiment, "sensitivity")

    sensitivity = experiment.sensitivity

    assert len(sensitivity.members) > 0

    for analysis in sensitivity.members.values():
        snakefile_content = analysis._workflow_builder.generate_snakefile_content(
            process_system_level_inputs=False,
            run_setup_phase=True,
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

        sub_snakefile_path = tst_ut.write_snakefile(analysis, snakefile_content)
        tst_ut.assert_file_exists(sub_snakefile_path, "Member Snakefile")
        assert len(sub_snakefile_path.read_text()) > 100

    master_snakefile_content = sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both",
        compression_level=5,
    )

    tst_ut.assert_snakefile_has_rules(
        master_snakefile_content,
        [
            "all",
            "experiment_consolidation",
            "prepare_member",
            "simulation_member",
            "process_member",
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

    num_analyses = len(sensitivity.members)
    for member_id in range(num_analyses):
        assert f"rule consolidate_member_{member_id}:" in master_snakefile_content

    master_snakefile_path = tst_ut.write_snakefile(experiment, master_snakefile_content)
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
    """Generation assertion for the ``--reprocess`` flag on ``rule render_report``.

    The REPROCESS master Snakefile's ``rule render_report`` shell MUST pass
    ``--reprocess`` to ``render_report_runner`` so ``render_report()`` selects
    ``Snakefile.reprocess``; the PRODUCTION master Snakefile's ``rule
    render_report`` shell MUST NOT (keeping the production render path
    byte-identical). Guards against the flag being silently dropped from the
    reprocess rule.
    """
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder

    reprocess_content = builder.generate_reprocess_master_snakefile_content(which="both", start_with="render")
    production_content = builder.generate_master_snakefile_content(which="both", compression_level=5)

    reprocess_block = _render_report_rule_block(reprocess_content)
    production_block = _render_report_rule_block(production_content)

    assert "render_report_runner" in reprocess_block, "reprocess render_report rule must invoke the runner"
    assert "--reprocess" in reprocess_block, (
        "reprocess master render_report shell MUST pass --reprocess so render_report() selects Snakefile.reprocess"
    )
    assert "render_report_runner" in production_block, "production render_report rule must invoke the runner"
    assert "--reprocess" not in production_block, (
        "production master render_report shell MUST NOT pass --reprocess (byte-identical production render path)"
    )


def test_master_snakefile_emits_per_target_setup_rules(
    synth_sensitivity_analysis,
):
    """The master Snakefile emits one `rule setup_target_{N}` per unique system target.

    The synth fixture has no per-SA `system_config_yaml` column, so all members
    collapse to a single UniqueSystemTarget. The Snakefile must contain `rule setup_target_0`,
    no standalone `rule setup:` block and no `_status/a_setup_complete.flag`.
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
    # The standalone `rule setup:` and `_status/a_setup_complete.flag` must not appear.
    assert "rule setup:\n" not in content
    assert "_status/a_setup_complete.flag" not in content
    # Every member has a `prepare_member_{id}_evt_` rule.
    for member_id in sensitivity.members.keys():
        assert f"rule prepare_member_{str(member_id).replace('.', '_').replace('-', '_')}_evt_" in content
    # The setup-target flag must surface in rule all.
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
def test_snakemake_sensitivity_workflow_config_generation(synth_sensitivity_analysis, config, expected_flags):
    """
    Test configuration passed to Snakemake for sensitivity analysis.

    Verifies that:
    1. All parameters are correctly formatted in master Snakefile
    2. Consolidation command includes correct flags
    """
    analysis = synth_sensitivity_analysis
    sensitivity = analysis.sensitivity

    master_snakefile_content = sensitivity._workflow_builder.generate_master_snakefile_content(**config)

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
    5. df_status carries the ``snakemake_allocated_*`` columns and a leading ``member_id`` column holding
       ``str(i)`` for each row
    """
    analysis = synth_sensitivity_analysis
    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        run_setup_phase=True,
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

    assert result.get("success"), f"Snakemake dry-run failed: {result.get('message', '')}"
    assert result.get("mode") == "local"

    df_status = analysis.df_status
    assert not df_status.empty
    assert "snakemake_allocated_nTasks" in df_status.columns
    assert "snakemake_allocated_omp_threads" in df_status.columns
    assert "snakemake_allocated_total_cpus" in df_status.columns
    # `sa_id` is a member of sensitivity_analysis._RETIRED_ID_COLUMNS; the master
    # emits `member_id`. Asserting POSITION as well as presence, because the defect
    # this caught was the column surviving but being demoted out of the identity block.
    assert "member_id" in df_status.columns
    assert df_status.columns[0] == "member_id"
    expected_ids = [str(i) for i in range(len(df_status))]
    assert df_status["member_id"].tolist() == expected_ids


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_snakemake_sensitivity_workflow_execution(synth_sensitivity_analysis):
    """
    Test Snakemake sensitivity analysis workflow execution.

    Validates that:
    1. submit_workflow() returns success
    2. ``tst_ut.assert_analysis_workflow_completed_successfully`` passes: it runs ``validate_analysis`` over the
       analysis ("Runs all 7 checks", per that helper's docstring)
    3. The benchmarking HTML figure for ``n_devices`` exists under ``plots/sensitivity/benchmarking``
    """
    analysis = synth_sensitivity_analysis

    which = "both"

    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        run_setup_phase=True,
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


# Property: reprocess(start_with="process") self-heals a summary deleted under a surviving
# d_process flag at the default regenerate_existing=False, leaves healthy flag mtimes untouched,
# and emits zero process_member_* rules on the second pass.
def test_reprocess_process_self_heals_deleted_summary(
    golden_sensitivity_analysis_keeping_scenarios, golden_reprocess_env
):
    """Regression: a deleted summary under a surviving completion flag must be rebuilt by a default reprocess.

    The divergence state is: a per-scenario summary output was deleted, but its
    ``d_process_*`` completion flag survived. ``reprocess(start_with="process")``
    must self-heal it — reconcile the flag + per-model processing log against
    on-disk summary presence — and rebuild the missing summary, REGARDLESS of
    ``regenerate_existing``.

    Asserts:
    - Rebuild: delete one summary zarr while leaving its
      ``d_process`` flag → ``reprocess(start_with="process",
      regenerate_existing=False)`` rebuilds the summary, re-runs consolidate,
      and reproduces the analysis summaries.
    - Healthy no-op: every OTHER ``d_process`` flag retains its
      pre-reprocess mtime (only the divergent pair was unlinked).
    - Steady state: a SECOND ``reprocess(start_with="process",
      dry_run=True)`` emits ZERO ``process_member_*`` rules — asserted via a
      deterministic Snakefile.reprocess parse.
    """
    import shutil

    import hhemt.analysis as _analysis_mod
    from hhemt.constants import (
        consolidate_analysis_flag,
        process_timeseries_flag_per_member,
    )
    from hhemt.scenario import (
        TRITONSWMM_scenario,
        compute_event_id_slug,
    )

    # Guard: assert that the loaded TRITONSWMM_analysis carries
    # `_reconcile_stale_process_flags_against_summaries` before doing any expensive build/run work; the failure
    # message names the loaded module's file, which shows a stale installed copy if that is what was imported.
    assert hasattr(
        _analysis_mod.TRITONSWMM_analysis,
        "_reconcile_stale_process_flags_against_summaries",
    ), (
        "self-heal helper missing from the loaded TRITONSWMM_analysis — the test "
        f"is exercising the wrong source copy: {_analysis_mod.__file__}"
    )

    # model_type -> consolidate-consumed summary attrs (the canonical table in
    # hhemt.summary_paths, shared with the production self-heal's predicate).
    from hhemt.summary_paths import _SUMMARY_ATTRS_BY_MODEL

    analysis = golden_sensitivity_analysis_keeping_scenarios

    # ── Arrange: the committed golden is already a completed sensitivity analysis with the raw layer, so the real
    # summary zarrs exist on disk and need no workflow run. ─────────────────────────────────────────────────────────
    tst_ut.assert_analysis_summaries_created(analysis)

    master_dir = analysis.analysis_paths.analysis_dir
    status_dir = master_dir / "_status"

    # Pick one member + its first event + its first enabled model, and
    # resolve a present summary output to delete.
    # Members are owned by the sensitivity manager; the top-level
    # TRITONSWMM_analysis delegates reprocess() to self.sensitivity.reprocess().
    # Iterate items() (keys may be int) and string-cast the chosen id for the
    # flag-token helpers, which validate a str member_id fragment.
    member_items = list(analysis.sensitivity.members.items())
    assert member_items, "fixture produced no members"
    target_member_key, sub = member_items[0]
    target_member = str(target_member_key)

    scen = TRITONSWMM_scenario(0, sub)
    evt = compute_event_id_slug(sub._retrieve_weather_indexer_using_integer_index(0))
    model_type = scen.run.model_types_enabled[0]

    summary_to_delete = None
    for attr in _SUMMARY_ATTRS_BY_MODEL[model_type]:
        p = getattr(scen.scen_paths, attr, None)
        if p is not None and p.exists():
            summary_to_delete = p
            break
    assert summary_to_delete is not None, f"no present summary output found to delete for model_type={model_type}"

    flag_path = master_dir / process_timeseries_flag_per_member(model_type, target_member, evt)
    assert flag_path.exists(), f"expected d_process flag present before delete: {flag_path}"

    # Create the divergence: delete the summary, leave the flag.
    if summary_to_delete.is_dir():
        shutil.rmtree(summary_to_delete)
    else:
        summary_to_delete.unlink()
    assert not summary_to_delete.exists(), "summary should be deleted"
    assert flag_path.exists(), "d_process flag must survive summary deletion (this IS the divergence state)"

    # Record all d_process flag mtimes for the healthy no-op arm.
    pre_mtimes = {f: f.stat().st_mtime for f in status_dir.glob("d_process_*_complete.flag")}

    # ── Act: reprocess on the process path with the DEFAULT
    # regenerate_existing=False (the configuration that originally failed). ──────
    analysis.reprocess(start_with="process", execution_mode="local", regenerate_existing=False)

    # ── Assert (rebuild). ──────────────────────────────────────────────────────
    assert summary_to_delete.exists(), "self-heal must rebuild the deleted summary on the process path"
    consolidate_flag = master_dir / consolidate_analysis_flag(target_member)
    assert consolidate_flag.exists(), f"per-member consolidate flag must be present after reprocess: {consolidate_flag}"
    tst_ut.assert_analysis_summaries_created(analysis)

    # ── Assert (healthy no-op): every OTHER d_process flag keeps its
    # pre-reprocess mtime; only the divergent pair was unlinked + rebuilt. ──────
    for f, mtime in pre_mtimes.items():
        if f == flag_path:
            continue  # the healed flag was unlinked + rebuilt (mtime expected to change)
        assert f.exists(), f"healthy d_process flag unexpectedly missing: {f}"
        assert f.stat().st_mtime == mtime, f"healthy d_process flag mtime changed unexpectedly (should be a no-op): {f}"

    # ── Assert (steady state): a SECOND process reprocess (dry-run)
    # emits ZERO process_member_ rules — the gate's `not d_process_path.exists()` is
    # False for every event. Parse the generated Snakefile.reprocess
    # (deterministic; does not depend on snakemake dry-run stats formatting). ───
    analysis.reprocess(start_with="process", execution_mode="local", dry_run=True)
    snakefile = master_dir / "Snakefile.reprocess"
    assert snakefile.exists(), f"reprocess Snakefile not generated: {snakefile}"
    snakefile_text = snakefile.read_text()
    assert "rule process_member_" not in snakefile_text, (
        "steady-state second reprocess must emit zero process_member_ rules (all d_process flags now present)"
    )


def test_experiment_consolidation_tolerates_incomplete_analysis(golden_sensitivity_analysis):
    """Regression: master consolidation with one incomplete member must not crash.

    Reproduces the FUNCTION-LEVEL crash behind the live uva_sensitivity_suite
    experiment_consolidation FileNotFoundError: consolidate_sensitivity_datatree
    raising when one member is incomplete (its per-scenario summary was
    never produced, its analysis_datatree.zarr was never built, and its
    datatree_consolidation_complete log is False — the real never-consolidated
    state). This test calls consolidate_sensitivity_datatree directly (the
    unit-level gate for the function under change); the consolidate_workflow.py
    call site passes no ``allow_incomplete`` and so inherits the True default, a
    wiring the end-to-end runs in this module reach (the master Snakefile passes
    ``--consolidate-sensitivity-analysis-outputs``) and this unit test does not.
    Master consolidation must still assemble
    sensitivity_datatree.zarr over the COMPLETED subset under
    allow_incomplete=True (the default), skipping the incomplete sub.

    Asserts the tolerant regeneration over the completed subset (experiment definition
    preserved in the root `parameters` node) and the fail-fast when allow_incomplete=False.
    """
    import shutil

    import xarray as xr

    from hhemt.scenario import TRITONSWMM_scenario
    from hhemt.summary_paths import _SUMMARY_ATTRS_BY_MODEL

    analysis = golden_sensitivity_analysis

    # Arrange: the committed golden already holds real per-sub datatrees and the master tree.
    sensitivity = analysis.sensitivity
    member_items = list(sensitivity.members.items())
    assert len(member_items) >= 2, "test needs >=2 members (one incomplete, one complete)"

    master_zarr = sensitivity.analysis_paths.experiment_datatree_zarr
    assert master_zarr is not None and master_zarr.exists(), "master tree should exist after full run"

    incomplete_key, incomplete_sub = member_items[0]
    complete_key, _complete_sub = member_items[-1]
    incomplete_id = str(incomplete_key)
    complete_id = str(complete_key)
    prefix = sensitivity.member_prefix  # e.g. "member_"

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

    # Assert the tolerant regeneration over the completed subset.
    assert out.exists(), "master tree must regenerate over the completed subset"
    tree = xr.open_datatree(out, engine="zarr", consolidated=False)
    member_nodes = {c for c in tree.children if c.startswith(prefix)}
    assert f"{prefix}{incomplete_id}" not in member_nodes, (
        f"incomplete sub {prefix}{incomplete_id} must be absent from the master tree nodes"
    )
    assert f"{prefix}{complete_id}" in member_nodes, (
        f"complete sub {prefix}{complete_id} must be present in the master tree nodes"
    )
    assert len(member_nodes) == len(member_items) - 1, (
        "exactly one member (the incomplete one) must be absent from the tree nodes"
    )
    # The root `parameters` node (experiment definition) still lists every member.
    assert "parameters" in tree.children, "root `parameters` node must be present"
    assert len(tree["parameters"].to_dataset().to_dataframe()) == len(sensitivity.df_setup), (
        "root `parameters` Dataset must list every defined member"
    )

    # Assert the fail-fast is preserved when allow_incomplete=False.
    shutil.rmtree(out)
    with pytest.raises((FileNotFoundError, ValueError)):
        sensitivity.consolidate_sensitivity_datatree(allow_incomplete=False)


# ─── Sensitivity-master report configuration and rendering tests ───────────────

from pathlib import Path as _Path  # noqa: E402

_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7 = (
    _Path(__file__).resolve().parents[1] / "configs" / "reports" / "synth_sensitivity_report_config.yaml"
)


def test_synth_sensitivity_report_inlined_passes_run_entry_validation():
    """The BARE ``_TestCaseBuilder`` synth-sensitivity construction (no catalog report
    injection) must inline ``cfg_analysis.report`` so the ``run()``-entry
    cross-validation passes.

    The catalog wrappers no longer inject ``additional_analysis_configs={"report": ...}``,
    so this path routes through the bare builder. Before the builder inlined a report
    block it emitted ``report: {}`` -> ``cfg_analysis.report.sensitivity is None`` ->
    ``validate_sensitivity_independent_vars`` raises ``ConfigurationError``. Now the
    builder inlines ``report_config_synth_sensitivity.yaml``, so ``report.sensitivity``
    is populated and validation passes. The bare-builder path is the standard synth
    construction, so a pattern-match against the (previously report-injecting) catalog
    wrappers would have been tautologically green.
    """
    from pathlib import Path

    import tests.fixtures.test_case_catalog as cases
    from hhemt.config.report import validate_sensitivity_independent_vars

    case = cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case(start_from_scratch=False, skip_run=True)
    analysis = case.analysis

    # The bare builder inlines the synth sensitivity report block.
    assert analysis.cfg_analysis.report.sensitivity is not None, (
        "builder-inline regressed: bare _TestCaseBuilder emitted an empty "
        "report block (cfg_analysis.report.sensitivity is None)"
    )

    # Mirrors the run()-entry cross-validation: analysis.run() resolves
    # cfg_report = self.cfg_analysis.report and member_csv = cfg_analysis.sensitivity_analysis.
    member_csv = Path(analysis.cfg_analysis.sensitivity_analysis)
    validate_sensitivity_independent_vars(analysis.cfg_analysis.report, member_csv)


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_run_and_render_report(synth_sensitivity_analysis_cached):
    """Sensitivity run -> master render.

    Asserts the master report exists, the benchmarking figures exist under ``plots/sensitivity/benchmarking``, no
    per-member report exists, and every member whose ``analysis_datatree.zarr`` exists on disk is a node of the
    master ``sensitivity_datatree.zarr`` (silent-drop guard).
    """
    from pathlib import Path

    analysis = synth_sensitivity_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7),
    )
    out_html = analysis.sensitivity.render_report(format="html")
    assert out_html.exists() and out_html.stat().st_size > 0

    master_dir = analysis.sensitivity.experiment.analysis_paths.analysis_dir
    bench_dir = master_dir / "plots" / "sensitivity" / "benchmarking"
    assert bench_dir.exists()
    assert any(bench_dir.glob("benchmarking__*.vs.total.html"))

    for member_id, sub in analysis.sensitivity.members.items():
        sub_html = sub.analysis_paths.analysis_dir / "analysis_report.html"
        assert not sub_html.exists(), f"unexpected per-member report at {sub_html} for member_id={member_id}"

    # Master-tree membership parity (silent-drop guard).
    # After a FULL clean consolidation (all subs present), every member
    # whose analysis_datatree.zarr exists on disk MUST appear as a node in the
    # master sensitivity_datatree.zarr. Asserting the on-disk-zarr set is a
    # subset of the master-tree-node set catches the silent-drop class (where
    # `allow_incomplete=True` lets build_sensitivity_datatree drop a skipped sub
    # while returning success=True) against on-disk ground truth -- independent
    # of the clobberable/stale in-memory + log completion flags.
    import xarray as xr

    sensitivity = analysis.sensitivity
    prefix = sensitivity.member_prefix
    master_zarr = sensitivity.analysis_paths.experiment_datatree_zarr
    assert master_zarr is not None and master_zarr.exists(), (
        "master sensitivity_datatree.zarr must exist after a full clean run"
    )
    master_tree = xr.open_datatree(master_zarr, engine="zarr", consolidated=False)
    tree_member_ids = {c.removeprefix(prefix) for c in master_tree.children if c.startswith(prefix)}
    on_disk_member_ids = {
        str(member_id)
        for member_id, sub in sensitivity.members.items()
        if sub.analysis_paths.analysis_datatree_zarr is not None and sub.analysis_paths.analysis_datatree_zarr.exists()
    }
    assert on_disk_member_ids <= tree_member_ids, (
        f"silent-drop: subs on disk but missing from master tree: {on_disk_member_ids - tree_member_ids}"
    )


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_render_report_idempotent(synth_sensitivity_analysis_cached):
    """Sensitivity render_report() is idempotent."""
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
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_synth_render_report_interactive_zip(synth_sensitivity_analysis_cached, tmp_path):
    # ZIP-fallback path: render_report(format="zip") writes
    # `analysis_report.zip` containing `report.html` + per-figure
    # `data/*.html` entries. Asserts the bundle carries >= 6 data/*.html
    # entries.
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
    assert len(html_entries) >= 6, f"Expected >= 6 data/*.html entries in ZIP, got {len(html_entries)}: {html_entries}"


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_plot_sources_attribution(synth_sensitivity_analysis_cached):
    """The string 'Sources:' appears in the master HTML report."""
    from pathlib import Path

    analysis = synth_sensitivity_analysis_cached
    analysis.run(
        from_scratch=False,
        report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7),
    )
    analysis.sensitivity.render_report(format="html")

    master_dir = analysis.sensitivity.experiment.analysis_paths.analysis_dir
    html = (master_dir / "analysis_report.html").read_text()
    assert "Sources:" in html


@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
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
# Sensitivity-CSV columns prefixed ``system.``: validation errors, synthesis of
# per-target system YAMLs, the fingerprint payload, and the df_setup accessor
# ============================================================================

import yaml as _yaml  # noqa: E402

import tests.fixtures.test_case_catalog as _cases  # noqa: E402
from hhemt.exceptions import ConfigurationError  # noqa: E402


def test_system_overlay_mutual_exclusion_with_system_config_yaml(_toolkit_identity_not_dirty):
    """A row with both system_config_yaml and system.* columns raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_mutex_violation(start_from_scratch=True)


def test_system_overlay_validator_re_fire_invalid_value(_toolkit_identity_not_dirty):
    """An invalid system.* overlay value raises ConfigurationError via Pydantic."""
    with pytest.raises(ConfigurationError, match="SystemConfig validation"):
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_invalid_overlay(start_from_scratch=True)


def test_gpu_hardware_override_column_raises_migration_error(_toolkit_identity_not_dirty):
    """Legacy `gpu_hardware_override` column is rejected as unknown.

    The prior migration target `system.gpu_hardware` is itself retired
    (gpu_hardware is now partition-DERIVED). The column is rejected
    as unknown; the message lists the valid columns (including the `hpc.partition`
    alias that selects GPU hardware via the partition spec)."""
    with pytest.raises(ConfigurationError, match="Unknown sensitivity-CSV columns") as excinfo:
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_legacy_gpu_hardware_override(
            start_from_scratch=True
        )
    assert "gpu_hardware_override" in str(excinfo.value)


def test_unknown_column_rejected_with_nearest_match_hint(_toolkit_identity_not_dirty):
    """An unknown column (a typo) raises ConfigurationError."""
    with pytest.raises(ConfigurationError, match="Unknown sensitivity-CSV columns"):
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_typo_in_prefixed_column(
            start_from_scratch=True
        )


def test_system_dot_prefix_column_recognized_and_synthesizes_yaml(
    synth_sensitivity_with_system_overlay,
):
    """A system.target_dem_resolution column synthesizes two per-target YAMLs."""
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
    """The fingerprint payload attaches a system_overlay key and schema version 3."""
    experiment = synth_sensitivity_with_system_overlay
    analysis = experiment.sensitivity.members["0"]
    payload = experiment.sensitivity._compute_member_id_fingerprint_payload(analysis)
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
    assert "analysis.n_mpi_procs" in overlay_frame.columns
    # Index parity with df_setup (same member_ids, same order).
    assert list(overlay_frame.index) == list(sens.df_setup.index)


# ============================================================================
# Sensitivity-CSV columns prefixed ``analysis.`` or left bare (the bare-name DeprecationWarning and the
# analysis/system split of the independent variables); the fourth test below checks that a runner subprocess does
# not purge ``_generated/``.
# ============================================================================


def test_bare_name_analysis_config_column_is_refused_naming_the_prefixed_form(
    monkeypatch, tmp_path, _toolkit_identity_not_dirty
):
    """A bare-name analysis-config column raises ConfigurationError naming `analysis.{name}`."""
    # This is a start_from_scratch=True retrieve on a slug of its own; the runs_root
    # override redirects the analysis tree to tmp_path so the construction (which is all
    # this test exercises) cannot touch the shared synth_sensitivity cache.
    monkeypatch.setenv(_TEST_RUNS_ROOT_OVERRIDE_ENV, str(tmp_path))
    with pytest.raises(ConfigurationError, match="bare analysis_config field names") as excinfo:
        _cases.Local_TestCases.retrieve_synth_cpu_config_sensitivity_case_bare_name_column(start_from_scratch=True)
    assert "analysis.mem_gb_per_cpu" in str(excinfo.value)


def test_analysis_dot_prefix_column_accepted_silently(synth_sensitivity_all_analysis_prefixed):
    """`analysis.{field}` columns are accepted without DeprecationWarning."""
    import warnings

    analysis = synth_sensitivity_all_analysis_prefixed
    # Construction already completed via the fixture. Re-derive the members under
    # an explicit DeprecationWarning-as-error filter to confirm no bare-name warning fires
    # for the all-prefixed fixture.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _ = analysis.sensitivity._create_members()


def test_attributes_varied_for_analysis_split(synth_sensitivity_mixed_prefixed_columns):
    """`analysis_independent_vars` and `system_independent_vars` are separate."""
    analysis = synth_sensitivity_mixed_prefixed_columns
    assert "n_omp_threads" in analysis.sensitivity.analysis_independent_vars  # bare-name in fixture
    assert "n_mpi_procs" in analysis.sensitivity.analysis_independent_vars  # analysis.n_mpi_procs in fixture
    assert "target_dem_resolution" in analysis.sensitivity.system_independent_vars
    assert "target_dem_resolution" not in analysis.sensitivity.analysis_independent_vars


def test_build_unique_system_targets_skips_purge_in_runner_subprocess(
    synth_sensitivity_with_system_overlay,
    monkeypatch,
):
    """is_main_orchestrator=False skips fast_rmtree of _generated/."""
    from hhemt import utils as _utils_mod

    analysis = synth_sensitivity_with_system_overlay
    generated_dir = analysis.analysis_paths.analysis_dir / "_generated"
    assert generated_dir.exists()
    called: list[Path] = []

    def fake_rmtree(p, *args, **kwargs):
        # Records each path passed to fast_rmtree instead of deleting it, so the assertion below can check whether
        # _generated/ was among them.
        called.append(Path(p))

    monkeypatch.setattr(_utils_mod, "fast_rmtree", fake_rmtree)
    analysis.sensitivity._build_unique_system_targets(
        analysis.sensitivity._df_setup_full,
        is_main_orchestrator=False,
    )
    assert generated_dir not in called, f"fast_rmtree was called on _generated/ in runner-subprocess mode: {called}"


# Property: reprocess(start_with="render") over a partial-completion suite reads
# Snakefile.reprocess, whose report targets are filtered to completed subs, so snakemake --report
# never enumerates a deleted figure.
def test_reprocess_render_report_over_partial_completion(golden_planned):
    """Behavioral regression: a render-stage reprocess over a partial-completion suite must succeed.

    Reproduces the live uva_sensitivity_suite failure: ``reprocess(start_with=
    "render")`` over a PARTIAL-completion sensitivity suite hard-failed because
    ``render_report()`` ran ``snakemake --report`` against the PRODUCTION
    Snakefile (which enumerates ``report()`` targets for EVERY defined sub,
    including a never-completed one), so the report engine's existence gate
    (``report/__init__.py``) aborted on the first absent ``report()``-flagged
    figure with a ``WorkflowError``. The fix routes the reprocess render to
    ``Snakefile.reprocess`` (which filters report targets to the completed
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

    from hhemt.scenario import TRITONSWMM_scenario
    from hhemt.summary_paths import _SUMMARY_ATTRS_BY_MODEL

    analysis = golden_planned

    # Arrange: a golden carries no plots or report (tests/goldens/SPEC.md section 9), so a consolidate-stage reprocess
    # of the complete golden materializes the per-sim figures and the master report for every sub.
    arrange = analysis.reprocess(
        start_with="consolidate", execution_mode="local", override_report_formats=["zip"], verbose=False
    )
    assert arrange["success"], f"arranging the report failed: {arrange.get('message')!r}"

    sensitivity = analysis.sensitivity
    member_items = list(sensitivity.members.items())
    assert len(member_items) >= 2, "test needs >=2 members (one incomplete, one complete)"

    master_dir = sensitivity.experiment.analysis_paths.analysis_dir
    report_zip = master_dir / "analysis_report.zip"
    assert report_zip.exists(), "master analysis_report.zip should exist after the full run"

    incomplete_key, incomplete_sub = member_items[0]
    incomplete_id = str(incomplete_key)

    # (a) Delete one per-scenario summary so the reprocess generator's
    #     summary-existence filter EXCLUDES this sub from
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
    #     plots/sensitivity/per_sim/member-{id}/{event_id}/*.{html,png}.
    incomplete_fig_dir = master_dir / "plots" / "sensitivity" / "per_sim" / f"member-{incomplete_id}"
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
        override_report_formats=["zip"],
        verbose=True,
    )

    # Assert the render succeeds over the completed subset, with no WorkflowError.
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
    reason="renderer-IO provenance audit is opt-in; set "
    "HHEMT_ENABLE_PROVENANCE_AUDIT=1 to enable the audit AND run this test.",
)
@pytest.mark.slow
@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_renderer_provenance_audit_passes_for_all_sensitivity_renderers(synth_sensitivity_analysis_cached):
    """Sensitivity-tier audit-passes guard — exercises sensitivity_benchmarking
    (Plotly branch), the per-member per-sim renderers, and errors_and_warnings /
    per_analysis_summary reading the persisted validation_report.json.

    Master plots are deleted before the second run() to force a fresh render through
    the audited _cli subprocess path (the _cached fixture key is the swmm-topology
    SHA, not the renderer source). render_report() requires every report()-flagged
    figure to exist, so its success is the audit-passed-for-every-renderer
    signal: a single renderer's audit ProcessingError leaves its figure missing.
    """
    import shutil
    from pathlib import Path

    analysis = synth_sensitivity_analysis_cached
    analysis.run(from_scratch=False, report_config=Path(_SYNTH_SENSITIVITY_REPORT_CONFIG_PHASE7))

    master_dir = analysis.sensitivity.experiment.analysis_paths.analysis_dir
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


def _eda_templates_for(set_name):
    """The `eda_compute_sensitivity` selection's rule_spec_template tuple for a set."""
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    for sel in get_reporting_set(set_name).renderer_selection:
        if sel.builder_key == "eda_compute_sensitivity":
            return sel.rule_spec_template
    raise AssertionError(f"set {set_name!r} carries no eda_compute_sensitivity selection")


def _rule_block(content: str, rule_name: str) -> str:
    """The text of `rule {rule_name}:` up to the next column-0 `rule ` (or EOF)."""
    start = content.index(f"rule {rule_name}:")
    nxt = content.find("\nrule ", start + 1)
    return content[start:] if nxt == -1 else content[start:nxt]


from hhemt.report_renderers._reporting_sets import (  # noqa: E402
    compose_reporting_sets,
    eda_rule_spec_templates,
    get_reporting_set,
)


@pytest.mark.parametrize(
    "active_set, expected_template_count",
    [
        pytest.param(get_reporting_set("dem_resolution"), 4, id="registry"),
        pytest.param(compose_reporting_sets(("dem_resolution", "compute_sensitivity")), 5, id="composed"),
    ],
)
def test_dem_resolution_set_emits_one_eda_rule_per_registry_template(
    synth_sensitivity_analysis, active_set, expected_template_count
):
    """The dem-resolution set emits ONE EDA rule per registry template.

    Guards the registry-read generalization end to end: the builder iterates
    ``sel.rule_spec_template`` instead of hardcoding one figure, and BOTH the ``rule
    all`` enumeration and ``render_report``'s DERIVED input follow from the same
    single source. The expected template count is asserted (4 for ``dem_resolution``; 5 for the
    composed set, the 4 plus the single template that
    ``test_compute_sensitivity_set_still_emits_its_single_eda_rule`` asserts for
    ``compute_sensitivity``) and the emitted rule names are compared by membership PARITY
    against the registry templates, so adding, removing or renaming a template without
    updating the emission path fails here.

    Parametrized over (registry object, composed object). The registry arm is the
    differently-positioned satisfying arm and must stay green -- it is what catches a
    composition that changed single-set emission. The composed arm is the one that is
    red before _emit_active_set_plot_rules handles a merged template tuple.
    """
    analysis = synth_sensitivity_analysis
    # Hand-assign the run-entry attr because _resolve_active_reporting_set PREFERS it
    # over the cfg fallback -- that is the reason, not frozenness. report_config is NOT
    # frozen; it also has validate_assignment unset, so assigning the CONFIG field would
    # skip the normalizing field_validator and install a value construction can never
    # produce. The attr is the honest surface.
    analysis._active_reporting_set = active_set
    assert analysis.cfg_analysis.eda.enabled_plots, "the has_eda_artifact gate requires a non-empty eda.enabled_plots"

    content = analysis.sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both", compression_level=5
    )

    # The expectation is CONTRIBUTOR-derived on the composed arm. A registry-derived
    # expectation is correct for the registry arm and WRONG for the composed one, where
    # the union over contributors is the answer -- and using one expectation for both
    # arms is what makes the assertion an identity over the registry.
    templates = tuple(eda_rule_spec_templates(active_set))
    assert len(templates) == expected_template_count

    # (a) one emitted rule per registry template — membership parity, not a count.
    emitted = {t.rule_name for t in templates if f"rule {t.rule_name}:" in content}
    assert emitted == {t.rule_name for t in templates}, (
        "EDA rule emission is not one-per-registry-template.\n  missing from Snakefile: "
        f"{sorted({t.rule_name for t in templates} - emitted)}"
    )

    # (b) rule all lists every template's output path.
    expected_paths = [t.output_path_template.replace("__OUTPUT_EXT__", ".html") for t in templates]
    rule_all = _rule_block(content, "all")
    for path in expected_paths:
        assert f'"{path}"' in rule_all, f"rule all is missing the EDA output {path!r}"

    # (c) render_report's input is DERIVED from rule_all_inputs, not hand-edited:
    # every EDA path appears EXACTLY ONCE (a hand-edit duplicates it), and the
    # `_status/` flags that rule all carries are filtered out of it.
    render_block = _rule_block(content, "render_report")
    for path in expected_paths:
        assert render_block.count(f'"{path}"') == 1, (
            f"render_report input must contain {path!r} exactly once (a hand-edit "
            f"duplicates the derived input); found {render_block.count(chr(34) + path + chr(34))}"
        )
    render_input_block = render_block.split("output:")[0]
    assert '"_status/' not in render_input_block, (
        "render_report input must be the rule_all_inputs subset filtered of _status/ "
        "flags — an unfiltered copy means the derivation at workflow.py was bypassed"
    )


def test_compute_sensitivity_set_still_emits_its_single_eda_rule(
    synth_sensitivity_analysis,
):
    """The SHIPPED compute-sensitivity set is output-preserving.

    The one-rule-per-registry-template generalization tested above changed a code path
    this set runs. Its single EDA rule must still be emitted under its registry rule_name
    and output path, and its log path must now read `_logs/` (a deliberate
    reconciliation: the builder was the lone `logs/` outlier versus the registry template).
    """
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    analysis = synth_sensitivity_analysis
    analysis._active_reporting_set = get_reporting_set("compute_sensitivity")

    content = analysis.sensitivity._workflow_builder.generate_master_snakefile_content(
        which="both", compression_level=5
    )

    templates = _eda_templates_for("compute_sensitivity")
    assert len(templates) == 1, "compute-sensitivity must carry exactly one EDA template"
    (tmpl,) = templates
    assert f"rule {tmpl.rule_name}:" in content
    assert f'"{tmpl.output_path_template.replace("__OUTPUT_EXT__", ".html")}"' in content
    assert "_logs/plots/eda_compute_sensitivity.log" in content
    assert "logs/plots/eda_compute_sensitivity.log" not in content.replace(
        "_logs/plots/eda_compute_sensitivity.log", ""
    ), "the builder's `logs/` log path was not reconciled to `_logs/`"
