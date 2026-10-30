"""Fail when a test would run a simulation workflow without requesting the model build.

The problem: a workflow run (``analysis.run()``, ``submit_workflow()``, ``test()``) needs the
solver build to exist, and the workflow's setup rule refuses when it does not. On a machine
that has built once, the build is already in the cache, so a test that forgot to request it
stays green there and fails only on a fresh checkout. Nothing else in the suite reds on that
omission, so this module does.

How it decides, in seven rules.

1. Collection, not execution. A child pytest process collects the whole ``tests/`` tree with
   ``HHEMT_FORBID_COMPILE=1`` and runs nothing, so no fixture is set up and nothing compiles. The
   child starts without ``PYTEST_ADDOPTS`` and with no ``-m``, ``-k`` or ``--deselect``, and the
   test fails if it reports any deselected item: a deselection narrows the items the child
   records. For each collected item the child reads the function OBJECT (a class method
   included) and ``item.fixturenames``, never ``--setup-plan`` text, which cannot see a class
   method and misparses a node line whose parametrize id contains ``]``.
2. Run-reaching. A node reaches a run when its call graph holds a call to an attribute named in
   ``_RUN_ATTRS`` — ``.run(``, ``.submit_workflow(``, ``.test(``, ``.reprocess(`` or
   ``.submit_reprocess_workflow(`` — whose receiver is not the ``subprocess`` module, or a
   command-line invocation of a command whose callback reaches one: ``invoke(app, [...])`` in
   process, or a child ``hhemt`` or ``python -m hhemt`` call. The call graph is the node's own
   function, the function of every fixture in its closure, and every function or method they
   call that resolves (through the caller's ``__globals__`` or an ``import`` statement inside
   its body) to a file under ``tests/`` or the ``hhemt`` package, each walked once. The walk
   crosses into ``hhemt`` because a run is reached there through entry points named by none of
   the three attributes: ``hhemt run-experiment`` reaches ``tk.run(`` only inside
   ``hhemt.experiment_definition.run_experiment``.
3. Covered. A node is covered when it carries the ``compile_tier`` marker, which
   ``tests/conftest.py::pytest_collection_modifyitems`` derives from ``tritonswmm_cpu_compiled``
   in the fixture closure. That derivation is read, not restated here.
4. Excused. A node is excused when every run-reaching call in its graph passes a literal
   ``dry_run=True`` or ``--dry-run`` or is a CLI call carrying ``--help``; when it carries an
   unconditional ``skip`` marker; or when its node id is in the excuse table below. One real call
   beside dry ones is not excused.
5. What fails. A run-reaching node that is neither covered nor excused, and a table entry that is
   STALE (its node no longer collects, no longer reaches a run, or is now covered), so the table
   shrinks with the suite instead of rotting.
6. Tree-emptiness, the rung above the run itself. Every run-reaching node is classified from the
   SOURCE of its own function and of every in-scope function its fixture closure reaches, as
   ``override-isolated`` (a case factory called with a literal ``start_from_scratch=True``, or a
   per-test clone helper), ``cache-backed`` (a factory called with a literal
   ``start_from_scratch=False``, with ``share_scratch_root=True``, with a non-literal
   ``start_from_scratch``, or with no ``start_from_scratch`` argument at all — the builder's
   default reads the shared cache), or ``empty-tree`` (no case factory anywhere in the closure
   supplies a tree; a factory is a ``retrieve_*`` callable defined in the fixtures' case modules,
   because those modules also hold helpers that supply none). The class is read from CALL SITES,
   never from a fixture's name, and a mixed closure takes its highest class. This dimension is
   what a later package converts, so a false
   ``empty-tree`` member costs an edit to a file that already reads the cache: every
   ``empty-tree`` entry therefore carries the fixture names in its closure, so a tree supplied by
   an unrecognized factory is visible as a name rather than silently absent.
7. The counts are a machine artifact, not a printed line. Every rung carries its ``unit`` beside
   its ``count`` and its site list, at ``.pytest_cache/build_request_census.json`` (gitignored)
   or at the path ``$HHEMT_BUILD_REQUEST_CENSUS_ARTIFACT`` names. A count published without its
   unit is not a measurement: the re-derive population is measured in SITES, while the triage it
   pays for is measured in NODES.

Not established: the walk is static. It does not see a run reached by dynamic dispatch
(``getattr(obj, name)()``, a callable passed as an argument), a run reached through a call that
resolves outside ``tests/`` and ``hhemt``, a CLI call whose command name no literal, fixture
value or parametrize value spells out, an argv sliced or rebuilt from another (``argv = argv[:-1]``,
which the walk reads as a binding carrying no flag, so the call is never excused), or a
``compile_tier`` mark applied by hand to a test that does not compile. The tree dimension is
static in the same way: a tree supplied by a call that does not resolve statically, by a factory
in a file this module does not recognize, or one a test builds inline by constructing an analysis
object directly instead of through a case factory, reads as ``empty-tree``. Each such entry carries
its fixture names so the member is reviewable rather than merely unknown; and the direct-construction
shape is dormant rather than hypothetical — five modules in ``tests/`` construct an analysis inline,
but the one of them that reaches a run is cache-backed, so no ``empty-tree`` member has this cause.
A ``--dry-run`` or ``--help`` flag excuses a command-line call only when every binding of the argv
name, every ``return`` or ``yield`` of an argv helper, and every return path of an argv fixture
carries it; a parametrize id supplies one value, so its flags are certain.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

# The attribute names that start a workflow, and the literal arguments that make a call a no-op.
_RUN_ATTRS = frozenset({"reprocess", "run", "submit_reprocess_workflow", "submit_workflow", "test"})
# The subset whose call sites are the re-derive population: they rebuild a processed tree.
_RE_DERIVE_ATTRS = frozenset({"reprocess", "submit_reprocess_workflow"})
_DRY_FLAG = "--dry-run"
_HELP_FLAG = "--help"
_FLAGS = frozenset({_DRY_FLAG, _HELP_FLAG})

# Tree-emptiness classes, ASCENDING: a closure holding two of them answers with the last one.
_TREE_EMPTY = "empty-tree"
_TREE_CACHE_BACKED = "cache-backed"
_TREE_OVERRIDE_ISOLATED = "override-isolated"
_TREE_CLASSES = (_TREE_EMPTY, _TREE_CACHE_BACKED, _TREE_OVERRIDE_ISOLATED)
_TREE_RANK = {name: rank for rank, name in enumerate(_TREE_CLASSES)}

# Files whose callables construct or retrieve an analysis tree. A call into one of these is what
# makes a closure "supplies a tree" at all; the class then comes from its own literal arguments.
_CASE_FACTORY_MODULES = frozenset({"test_case_builder.py", "test_case_catalog.py"})
# Those two modules also hold helpers that supply no tree, so the callable's own name must match:
# every factory is named `retrieve_*`, and nothing else in either module is one.
_CASE_FACTORY_PREFIX = "retrieve_"
# Per-test clone helpers: they copy a cached tree into a private root rather than share it.
_CLONE_FACTORY_NAMES = frozenset({"clone_analysis_to_tmp", "prepare_clone_dir", "construct_analysis_from_paths"})

# The machine artifact: gitignored under .pytest_cache/, overridable for a consumer that wants it
# elsewhere. A census run must leave no tracked-tree dirt.
_ARTIFACT_ENV = "HHEMT_BUILD_REQUEST_CENSUS_ARTIFACT"
_ARTIFACT_DEFAULT = _REPO_ROOT / ".pytest_cache" / "build_request_census.json"

_STUBBED = "stubbed"
_REFUSES = "refuses before running"
_DRY_NON_LITERAL = "dry run by a non-literal argument"
_CATEGORIES = frozenset({_STUBBED, _REFUSES, _DRY_NON_LITERAL})

# How many helper returns deep an argv expression is traced before the trace gives up.
_DEPTH_LIMIT = 6


@dataclass(frozen=True)
class _Excuse:
    """One adjudicated group of run-reaching nodes that are neither covered nor auto-excused.

    ``category`` is one of ``_CATEGORIES``; ``quote`` is the source line that shows it, copied
    from the test or fixture file; ``node_ids`` are the full pytest node ids it covers.
    """

    category: str
    source: str  # repo-relative path of the file holding the quoted line
    quote: str
    node_ids: tuple[str, ...]


# EXCUSE TABLE: every uncovered run-reaching node that no literal argument or skip marker
# excuses. Adding a row is the one way a new test may reach a run without the build, so each
# row needs a category and the quoted line that shows it.
_EXCUSES: tuple[_Excuse, ...] = (
    _Excuse(
        _REFUSES,
        "tests/test_cli_01_validation.py",
        'analysis_config.write_text("version: 1\\n")',
        (
            "tests/test_cli_01_validation.py::test_from_scratch_and_resume_mutually_exclusive",
            "tests/test_cli_01_validation.py::test_event_ilocs_and_range_mutually_exclusive",
            "tests/test_cli_01_validation.py::test_verbose_and_quiet_mutually_exclusive",
            "tests/test_cli_01_validation.py::test_system_config_is_required",
            "tests/test_cli_01_validation.py::test_analysis_config_is_required",
            "tests/test_cli_01_validation.py::test_invalid_model_value",
            "tests/test_cli_01_validation.py::test_invalid_which_value",
            "tests/test_cli_01_validation.py::test_invalid_redownload_value",
            "tests/test_cli_01_validation.py::test_invalid_log_level_value",
            "tests/test_cli_01_validation.py::test_invalid_walltime_format_missing_leading_zero",
            "tests/test_cli_01_validation.py::test_invalid_walltime_format_wrong_separator",
            "tests/test_cli_01_validation.py::test_valid_walltime_format",
            "tests/test_cli_01_validation.py::test_valid_model_values",
            "tests/test_cli_01_validation.py::test_valid_which_values",
            "tests/test_cli_01_validation.py::test_valid_redownload_values",
            "tests/test_cli_01_validation.py::test_valid_log_level_values",
        ),
    ),
    _Excuse(
        _REFUSES,
        "tests/test_cli_02_exit_codes.py",
        '"/nonexistent/system.yaml",',
        ("tests/test_cli_02_exit_codes.py::test_exit_code_2_configuration_error_nonexistent_file",),
    ),
    _Excuse(
        _REFUSES,
        "tests/test_cli_02_exit_codes.py",
        'analysis_config.write_text("version: 1\\n")',
        (
            "tests/test_cli_02_exit_codes.py::test_exit_code_2_cli_validation_error_mutually_exclusive",
            "tests/test_cli_02_exit_codes.py::test_exit_code_2_cli_validation_error_invalid_enum",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_cli_05_run_experiment.py",
        "tk = Toolkit(SimpleNamespace(analysis=SimpleNamespace(run=_spy)))",
        ("tests/test_cli_05_run_experiment.py::test_toolkit_run_refuses_unknown_mode",),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_cli_05_run_experiment.py",
        'monkeypatch.setattr(eb, "run_experiment", fn)',
        (
            "tests/test_cli_05_run_experiment.py::test_mode_invalid_value_is_a_usage_error",
            "tests/test_cli_05_run_experiment.py::test_mode_overwrite_is_refused",
            "tests/test_cli_05_run_experiment.py::test_mode_valid_values_reach_run_experiment[fresh]",
            "tests/test_cli_05_run_experiment.py::test_mode_valid_values_reach_run_experiment[resume]",
            "tests/test_cli_05_run_experiment.py::test_mode_default_is_resume",
            "tests/test_cli_05_run_experiment.py::test_failure_result_exits_3_and_prints_no_green",
            "tests/test_cli_05_run_experiment.py::test_partial_failure_result_names_failed_rules",
            "tests/test_cli_05_run_experiment.py::test_reporter_prints_producer_text_literally",
            "tests/test_cli_05_run_experiment.py::test_success_result_prints_detail_lines",
            "tests/test_cli_05_run_experiment.py::test_run_and_run_experiment_share_the_result_reporter",
            "tests/test_cli_05_run_experiment.py::test_override_force_rerun_reaches_run_experiment",
            "tests/test_cli_05_run_experiment.py::test_exception_class_maps_to_documented_code[CLIValidationError]",
            "tests/test_cli_05_run_experiment.py::test_exception_class_maps_to_documented_code[ConfigurationError]",
            "tests/test_cli_05_run_experiment.py::test_exception_class_maps_to_documented_code[WorkflowPlanningError]",
            "tests/test_cli_05_run_experiment.py::test_exception_class_maps_to_documented_code[WorkflowError]",
            "tests/test_cli_05_run_experiment.py::test_exception_class_maps_to_documented_code[CompilationError]",
            "tests/test_cli_05_run_experiment.py::test_exception_class_maps_to_documented_code[SimulationError]",
            "tests/test_cli_05_run_experiment.py::test_exception_class_maps_to_documented_code[ProcessingError]",
            "tests/test_cli_05_run_experiment.py::test_exception_class_maps_to_documented_code[pydantic.ValidationError]",
            "tests/test_cli_05_run_experiment.py::test_exception_class_maps_to_documented_code[RuntimeError]",
            "tests/test_cli_05_run_experiment.py::test_schema_invalid_config_exits_2",
            "tests/test_cli_05_run_experiment.py::test_exit_1_is_not_a_run_experiment_code",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_cli_06_run_mode_flags.py",
        'monkeypatch.setattr(hhemt.analysis, "TRITONSWMM_analysis", _Analysis)',
        (
            "tests/test_cli_06_run_mode_flags.py::test_overwrite_forwards_a_different_run_argument_than_resume",
            "tests/test_cli_06_run_mode_flags.py::test_overwrite_forwards_force_rerun_all",
            "tests/test_cli_06_run_mode_flags.py::test_overwrite_contradicted_by_another_force_rerun_value_is_refused",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_cli_07_run_flag_combinations.py",
        'monkeypatch.setattr(hhemt.analysis, "TRITONSWMM_analysis", _Analysis)',
        (
            "tests/test_cli_07_run_flag_combinations.py::test_control_resume_off_route_reaches_the_fresh_branch",
            "tests/test_cli_07_run_flag_combinations.py::test_overwrite_never_cancels_from_scratch",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_cli_08_experiment_cluster_resolution.py",
        'monkeypatch.setattr(eb, "run_experiment", _fake)',
        (
            "tests/test_cli_08_experiment_cluster_resolution.py::test_experiment_verbs_resolve_the_cluster_one_way[run-experiment-one-declared-no-flag]",
            "tests/test_cli_08_experiment_cluster_resolution.py::test_experiment_verbs_resolve_the_cluster_one_way[run-experiment-two-declared-no-flag]",
            "tests/test_cli_08_experiment_cluster_resolution.py::test_experiment_verbs_resolve_the_cluster_one_way[run-experiment-zero-declared-no-flag]",
            "tests/test_cli_08_experiment_cluster_resolution.py::test_experiment_verbs_resolve_the_cluster_one_way[run-experiment-two-declared-flag-names-one]",
            "tests/test_cli_08_experiment_cluster_resolution.py::test_experiment_verbs_resolve_the_cluster_one_way[run-experiment-control-one-declared-matching-flag]",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_cli_10_experiment_verbs.py",
        'monkeypatch.setattr(hhemt.analysis, "TRITONSWMM_analysis", _Analysis)',
        ("tests/test_cli_10_experiment_verbs.py::test_run_status_prints_the_status_report_verbatim",),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_cli_10_experiment_verbs.py",
        'monkeypatch.setattr(eb, "build_case_from_definition", _fake_build)',
        (
            "tests/test_cli_10_experiment_verbs.py::test_test_analysis_calls_analysis_test_and_exits_on_its_outcome[all-ok]",
            "tests/test_cli_10_experiment_verbs.py::test_test_analysis_calls_analysis_test_and_exits_on_its_outcome[one-fails]",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_cli_11_experiment_verb_output.py",
        'monkeypatch.setattr(eb, "build_case_from_definition", _fake_build)',
        (
            "tests/test_cli_11_experiment_verb_output.py::test_printed_directory_is_the_one_returned[test-analysis-test-root]",
            "tests/test_cli_11_experiment_verb_output.py::test_printed_directory_survives_wrapping_and_emoji_codes[test-analysis-longer-than-80-columns]",
            "tests/test_cli_11_experiment_verb_output.py::test_printed_directory_survives_wrapping_and_emoji_codes[test-analysis-colon-word-component]",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_cli_12_run_from_scratch.py",
        'monkeypatch.setattr(hhemt.analysis, "TRITONSWMM_analysis", _Analysis)',
        (
            "tests/test_cli_12_run_from_scratch.py::test_from_scratch_alone_selects_fresh_and_the_guard_still_refuses_both",
        ),
    ),
    _Excuse(
        _REFUSES,
        "tests/test_experiment_definition_runner.py",
        'pytest.fail("gate should refuse before build")',
        ("tests/test_experiment_definition_runner.py::test_run_experiment_non_tty_refuses_override",),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_experiment_definition_runner.py",
        'monkeypatch.setattr(eb, "build_case_from_definition", lambda *a, **k: _FakeTk())',
        (
            "tests/test_experiment_definition_runner.py::test_run_experiment_assume_yes_proceeds",
            "tests/test_experiment_definition_runner.py::test_run_experiment_wait_autodetect[None-False-12345-True]",
            "tests/test_experiment_definition_runner.py::test_run_experiment_wait_autodetect[None-False-None-False]",
            "tests/test_experiment_definition_runner.py::test_run_experiment_wait_autodetect[None-True-12345-False]",
            "tests/test_experiment_definition_runner.py::test_run_experiment_wait_autodetect[True-False-None-True]",
            "tests/test_experiment_definition_runner.py::test_run_experiment_wait_autodetect[False-False-12345-False]",
            "tests/test_experiment_definition_runner.py::test_run_experiment_threads_override_force_rerun_to_tk_run",
            "tests/test_experiment_definition_runner.py::test_run_experiment_forwards_mode_to_tk_run",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_reprocess_delete_r8.py",
        "analysis._workflow_builder.submit_reprocess_delete_workflow = _fake",
        (
            "tests/test_reprocess_delete_r8.py::test_nonsensitivity_explicit_local_execution_mode_is_not_overridden",
            "tests/test_reprocess_delete_r8.py::test_nonsensitivity_failed_scoped_delete_raises_instead_of_consolidating_stale",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_reprocess_rebuild_invalidation.py",
        '"submit_reprocess_delete_workflow",',
        (
            "tests/test_reprocess_rebuild_invalidation.py::test_reprocess_regenerate_slurm_route_clears_flags_and_logs",
            "tests/test_reprocess_rebuild_invalidation.py::test_reprocess_regenerate_slurm_route_clears_log_nonsensitivity",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_reprocess_rebuild_invalidation.py",
        '"submit_reprocess_workflow",',
        ("tests/test_reprocess_rebuild_invalidation.py::test_reprocess_consolidate_inprocess_preserves_processed",),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_sensitivity_force_rerun_applied_once.py",
        "submit_workflow=_stop,",
        (
            "tests/test_sensitivity_force_rerun_applied_once.py::test_sensitivity_submit_applies_force_rerun_once[True]",
            "tests/test_sensitivity_force_rerun_applied_once.py::test_sensitivity_submit_applies_force_rerun_once[False]",
            "tests/test_sensitivity_force_rerun_applied_once.py::test_plain_submit_still_applies_force_rerun[True]",
            "tests/test_sensitivity_force_rerun_applied_once.py::test_plain_submit_still_applies_force_rerun[False]",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_version_migration_state.py",
        'return_value={"success": True, "mode": "local", "snakefile_path": None, "message": "ok"},',
        ("tests/test_version_migration_state.py::test_analysis_run_stamps_version_file",),
    ),
    _Excuse(
        _REFUSES,
        "tests/test_workflow_1job_sbatch_generation.py",
        'with pytest.raises(ConfigurationError, match="override_hpc_total_nodes"):',
        ("tests/test_workflow_1job_sbatch_generation.py::test_override_hpc_total_nodes_wrong_mode",),
    ),
    _Excuse(
        _REFUSES,
        "tests/test_workflow_1job_sbatch_generation.py",
        'with pytest.raises(ConfigurationError, match="extra_sbatch_args is only valid"):',
        ("tests/test_workflow_1job_sbatch_generation.py::test_extra_sbatch_args_wrong_mode",),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_workflow_queued_sentinel_adoption.py",
        'monkeypatch.setattr(SnakemakeWorkflowBuilder, "_submit_tmux_workflow", _fake_submit)',
        (
            "tests/test_workflow_queued_sentinel_adoption.py::test_sensitivity_batch_job_call_site_passes_the_adopted_token_count",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_workflow_queued_sentinel_adoption.py",
        'monkeypatch.setattr(type(builder), "_submit_tmux_workflow", _fake_submit)',
        (
            "tests/test_workflow_queued_sentinel_adoption.py::test_multisim_batch_job_call_site_passes_the_adopted_token_count",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_workflow_retry_policy.py",
        'patch.object(SnakemakeWorkflowBuilder, "_submit_single_job_workflow", return_value=submitted),',
        (
            "tests/test_workflow_retry_policy.py::test_waited_single_job_run_is_swept",
            "tests/test_workflow_retry_policy.py::test_waited_single_job_run_reads_every_error_path_spelling[i-two-timestamps]",
            "tests/test_workflow_retry_policy.py::test_waited_single_job_run_reads_every_error_path_spelling[ii-user-error-line]",
            "tests/test_workflow_retry_policy.py::test_waited_single_job_run_reads_every_error_path_spelling[iii-short-spelling]",
            "tests/test_workflow_retry_policy.py::test_waited_single_job_run_reads_every_error_path_spelling[iv-long-prefix]",
        ),
    ),
    _Excuse(
        _STUBBED,
        "tests/test_workflow_single_job_wait.py",
        'patch.object(builder, "run_snakemake_local", return_value={"success": True}),',
        (
            "tests/test_workflow_single_job_wait.py::TestSubmitWorkflowIntegration::test_submit_workflow_passes_wait_parameter",
            "tests/test_workflow_single_job_wait.py::TestSubmitWorkflowIntegration::test_submit_workflow_default_no_wait",
        ),
    ),
)


@dataclass
class _Site:
    """One run-reaching call: where it is, whether it is a no-op by its own literals."""

    kind: str  # "attr" for a .run/.submit_workflow/.test call, "cli" for a command-line call
    where: str  # repo-relative "path:line"
    text: str  # the source line, stripped
    dry: bool
    help: bool
    attr: str | None = None  # the attribute name an "attr" site called; None for a "cli" site


@dataclass
class _Pending:
    """A command-line call whose command name is resolved per node, once fixtures are known."""

    app: object  # the click group the call dispatches into
    tokens: list[str]
    params: set[str]  # the caller's own parameters the argv expression mentions
    flags: frozenset[str]  # the flags every path to this argv carries
    where: str
    text: str


@dataclass
class _Facts:
    """What one function contributes to a call graph on its own, before any callee is followed."""

    sites: list[_Site] = field(default_factory=list)
    pending: list[_Pending] = field(default_factory=list)
    callees: list[object] = field(default_factory=list)


@dataclass
class _Reach:
    """The union of ``_Facts`` over every function reachable from one root."""

    sites: list[_Site] = field(default_factory=list)
    pending: list[_Pending] = field(default_factory=list)


class _Context:
    """The names a function's body can see: its globals, its own imports, its locals and its class."""

    def __init__(self, func, node: ast.AST, owner):
        """Index ``node``'s imports, assignments and parameters; ``owner`` is the class ``self`` means."""
        self.func = func
        self.node = node
        self.globals = func.__globals__
        self.owner = owner
        self.imports: dict[str, tuple[str | None, str | None, int, str | None]] = {}
        self.assignments: dict[str, list[ast.expr]] = {}
        args = node.args
        self.params = {a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                self.params.add(extra.arg)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                for alias in sub.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    target = alias.name if alias.asname else alias.name.split(".")[0]
                    self.imports[bound] = (target, None, 0, None)
            elif isinstance(sub, ast.ImportFrom):
                for alias in sub.names:
                    self.imports[alias.asname or alias.name] = (sub.module, alias.name, sub.level, None)
            elif isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if isinstance(target, ast.Name):
                        self.assignments.setdefault(target.id, []).append(sub.value)
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(sub.target, ast.Name) and sub.value is not None:
                    self.assignments.setdefault(sub.target.id, []).append(sub.value)


class _CallGraph:
    """Builds call graphs from live function objects and answers "what runs does this reach?"."""

    def __init__(self):
        """Resolve the scope roots; importing ``hhemt`` here is what makes its files followable."""
        import hhemt

        self._package_dir = Path(hhemt.__file__).resolve().parent
        self._root_conftest = _REPO_ROOT / "conftest.py"
        self._trees: dict[str, tuple[dict[int, ast.AST], list[str]]] = {}
        self._facts: dict[object, _Facts] = {}
        self._reach: dict[object, _Reach] = {}
        self._return_tokens_cache: dict[object, tuple[list[str], set[str], frozenset[str]]] = {}
        self._command_sites: dict[int, list[_Site]] = {}
        self._app_cache: dict[int, object] = {}
        self._reach_functions: dict[object, frozenset] = {}
        self._tree_cache: dict[object, tuple[str, str | None]] = {}

    # -- source access ------------------------------------------------------------------------

    def in_scope(self, path: str) -> bool:
        """True when ``path`` is a file whose calls the walk follows: ``tests/``, ``hhemt``, root conftest."""
        try:
            resolved = Path(path).resolve()
        except OSError:
            return False
        if resolved == self._root_conftest:
            return True
        return any(base in resolved.parents for base in (_TESTS_DIR, self._package_dir))

    def _parsed(self, path: str) -> tuple[dict[int, ast.AST], list[str]]:
        """Parse ``path`` once; return its function defs keyed by first source line, and its lines."""
        if path not in self._trees:
            source = Path(path).read_text(encoding="utf-8")
            defs: dict[int, ast.AST] = {}
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
                    defs[first] = node
            self._trees[path] = (defs, source.splitlines())
        return self._trees[path]

    def _line(self, path: str, lineno: int) -> str:
        """The stripped source line ``lineno`` of ``path``."""
        lines = self._parsed(path)[1]
        return lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""

    def _where(self, path: str, lineno: int) -> str:
        """``path:lineno`` with ``path`` relative to the repository root when it lies under it."""
        try:
            rel = Path(path).resolve().relative_to(_REPO_ROOT)
        except ValueError:
            rel = Path(path)
        return f"{rel}:{lineno}"

    # -- resolving names to objects -----------------------------------------------------------

    def _owner_class(self, func):
        """The class a method was defined in, found from ``__qualname__`` through its module globals."""
        parts = func.__qualname__.split(".")
        if len(parts) < 2 or parts[-2] == "<locals>":
            return None
        obj = func.__globals__.get(parts[0])
        for part in parts[1:-1]:
            obj = getattr(obj, part, None)
        return obj if isinstance(obj, type) else None

    def _import_object(self, ctx: _Context, name: str):
        """The object an ``import`` statement inside ``ctx``'s function binds to ``name``, or None."""
        module, attr, level, _ = ctx.imports[name]
        try:
            if level:
                package = ctx.globals.get("__package__") or ""
                module = importlib.util.resolve_name("." * level + (module or ""), package)
            obj = importlib.import_module(module)
            if attr is None:
                return obj
            if hasattr(obj, attr):
                return getattr(obj, attr)
            return importlib.import_module(f"{module}.{attr}")
        except Exception:  # noqa: BLE001 -- an unimportable name is simply not followed
            return None

    def resolve(self, expr: ast.AST, ctx: _Context):
        """The module, class or function ``expr`` names statically, or None.

        Only names (through imports, globals and the owner class for ``self``/``cls``) and
        attribute chains over modules and classes resolve. A method called on an instance does
        not: its receiver's type is not knowable without running the code.
        """
        if isinstance(expr, ast.Name):
            if expr.id in ("self", "cls"):
                return ctx.owner
            if expr.id in ctx.params:
                return None
            if expr.id in ctx.imports:
                return self._import_object(ctx, expr.id)
            return ctx.globals.get(expr.id)
        if isinstance(expr, ast.Attribute):
            base = self.resolve(expr.value, ctx)
            if inspect.ismodule(base):
                if hasattr(base, expr.attr):
                    return getattr(base, expr.attr)
                try:
                    return importlib.import_module(f"{base.__name__}.{expr.attr}")
                except Exception:  # noqa: BLE001 -- see _import_object
                    return None
            if isinstance(base, type):
                return getattr(base, expr.attr, None)
        return None

    def _followable(self, obj):
        """The plain function behind ``obj`` when it is a function or method in scope, else None."""
        if inspect.ismethod(obj):
            obj = obj.__func__
        if not inspect.isfunction(obj):
            return None
        obj = inspect.unwrap(obj)
        code = getattr(obj, "__code__", None)
        if code is None or not self.in_scope(code.co_filename):
            return None
        return obj

    # -- argv tokens --------------------------------------------------------------------------

    def _collect(
        self, expr: ast.AST, ctx: _Context, tokens: list[str], params: set[str], depth: int, seen: set
    ) -> frozenset[str]:
        """Append to ``tokens`` every string ``expr`` spells out; add the caller's parameters it uses to ``params``.

        Returns the flags (``--dry-run``, ``--help``) the argv CERTAINLY carries: those on every
        path that can produce it. A name bound more than once, or a helper with more than one
        ``return``, contributes only the flags every binding or return carries, so a sibling
        binding that is dry never makes a real call beside it a no-op. ``tokens`` stays a union,
        because the command name is resolved from every spelling.

        Literals, f-string literal parts, list/tuple/``+`` composition, locals assigned in the
        function, module-level constants and the returned strings of an in-scope helper resolve.
        Anything else contributes nothing; a parameter it mentions is recorded so the node can
        supply its fixture or parametrize value later.
        """
        if depth > _DEPTH_LIMIT:
            return frozenset()
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):
                new = expr.value.split() if expr.value.startswith("hhemt ") else [expr.value]
                tokens.extend(new)
                return frozenset(t for t in new if t in _FLAGS)
            return frozenset()
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            return frozenset().union(*(self._collect(e, ctx, tokens, params, depth + 1, seen) for e in expr.elts))
        if isinstance(expr, ast.Starred):
            return self._collect(expr.value, ctx, tokens, params, depth + 1, seen)
        if isinstance(expr, ast.BinOp):
            left = self._collect(expr.left, ctx, tokens, params, depth + 1, seen)
            return left | self._collect(expr.right, ctx, tokens, params, depth + 1, seen)
        if isinstance(expr, ast.JoinedStr):
            return frozenset().union(*(self._collect(v, ctx, tokens, params, depth + 1, seen) for v in expr.values))
        if isinstance(expr, ast.FormattedValue):
            return self._collect(expr.value, ctx, tokens, params, depth + 1, seen)
        if isinstance(expr, ast.Name):
            return self._collect_name(expr.id, ctx, tokens, params, depth, seen)
        if isinstance(expr, ast.Call):
            return self._collect_call(expr, ctx, tokens, params, depth, seen)
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name) and sub.id in ctx.params:
                params.add(sub.id)
        return frozenset()

    def _collect_name(self, name: str, ctx: _Context, tokens, params, depth, seen) -> frozenset[str]:
        """Resolve a variable: its local assignments, else a parameter, else a module-level string list.

        The certain flags of a name bound more than once are the INTERSECTION over its bindings.
        """
        if name in ctx.assignments and (id(ctx.node), name) not in seen:
            seen.add((id(ctx.node), name))
            per_binding = [self._collect(v, ctx, tokens, params, depth + 1, seen) for v in ctx.assignments[name]]
            return frozenset.intersection(*per_binding) if per_binding else frozenset()
        if name in ctx.params:
            params.add(name)
            return frozenset()
        live = ctx.globals.get(name)
        found = (
            [live]
            if isinstance(live, str)
            else [i for i in live if isinstance(i, str)]
            if isinstance(live, (list, tuple))
            else []
        )
        tokens.extend(found)
        return frozenset(t for t in found if t in _FLAGS)

    def _collect_call(self, call: ast.Call, ctx: _Context, tokens, params, depth, seen) -> frozenset[str]:
        """Resolve ``list(x)``/``.split()``/``shlex.split(x)``-style wrappers and in-scope helper returns."""
        func = call.func
        wrapper = isinstance(func, ast.Name) and func.id in ("list", "tuple", "sorted", "set", "str")
        wrapper = wrapper or (isinstance(func, ast.Attribute) and func.attr in ("split", "copy"))
        if wrapper:
            certain: frozenset[str] = frozenset()
            if isinstance(func, ast.Attribute):
                certain |= self._collect(func.value, ctx, tokens, params, depth + 1, seen)
            for arg in call.args:
                certain |= self._collect(arg, ctx, tokens, params, depth + 1, seen)
            return certain
        helper = self._followable(self.resolve(func, ctx))
        if helper is not None:
            found, _, certain = self._return_tokens(helper, depth + 1)
            tokens.extend(found)
            return certain
        for sub in ast.walk(call):
            if isinstance(sub, ast.Name) and sub.id in ctx.params:
                params.add(sub.id)
        return frozenset()

    def _return_tokens(self, func, depth: int = 0) -> tuple[list[str], set[str], frozenset[str]]:
        """What a function's ``return``/``yield`` expressions spell out: strings, parameters used, certain flags.

        The certain flags are those carried by EVERY return path.
        """
        if func in self._return_tokens_cache:
            return self._return_tokens_cache[func]
        self._return_tokens_cache[func] = ([], set(), frozenset())
        ctx = self._context(func)
        tokens: list[str] = []
        params: set[str] = set()
        per_return: list[frozenset[str]] = []
        if ctx is not None:
            for sub in ast.walk(ctx.node):
                if isinstance(sub, (ast.Return, ast.Yield)) and sub.value is not None:
                    per_return.append(self._collect(sub.value, ctx, tokens, params, depth, set()))
        certain = frozenset.intersection(*per_return) if per_return else frozenset()
        self._return_tokens_cache[func] = (tokens, params, certain)
        return tokens, params, certain

    # -- per-function facts -------------------------------------------------------------------

    def _context(self, func) -> _Context | None:
        """The ``_Context`` of ``func``, or None when its definition cannot be found in its file."""
        code = getattr(func, "__code__", None)
        if code is None or not self.in_scope(code.co_filename):
            return None
        node = self._parsed(code.co_filename)[0].get(code.co_firstlineno)
        if node is None:
            return None
        return _Context(func, node, self._owner_class(func))

    def _is_subprocess(self, receiver: ast.AST, ctx: _Context) -> bool:
        """True when ``receiver`` is the ``subprocess`` module, whose ``run`` starts no workflow."""
        return self.resolve(receiver, ctx) is importlib.import_module("subprocess")

    def _is_process_launch(self, callee: ast.AST, ctx: _Context) -> bool:
        """True when ``callee`` is a ``subprocess`` function or ``os.system``/``os.popen``, which start a child."""
        if not isinstance(callee, ast.Attribute):
            return False
        base = self.resolve(callee.value, ctx)
        return base is importlib.import_module("subprocess") or (
            base is importlib.import_module("os") and callee.attr in ("system", "popen")
        )

    def _app_for(self, expr: ast.AST, ctx: _Context):
        """The click group a ``invoke(app, ...)`` first argument names, or None when it is not an app."""
        obj = self.resolve(expr, ctx)
        if obj is None:
            return None
        key = id(obj)
        if key not in self._app_cache:
            try:
                import typer.main

                group = typer.main.get_command(obj) if hasattr(obj, "registered_commands") else obj
                # Duck-typed: typer vendors its own click, so `isinstance(group, click.Command)` is False.
                self._app_cache[key] = group if hasattr(group, "callback") else None
            except Exception:  # noqa: BLE001 -- a non-app first argument is simply not a CLI call
                self._app_cache[key] = None
        return self._app_cache[key]

    def _hhemt_app(self):
        """The click group of the package's own CLI, the target of every child ``hhemt`` call."""
        import typer.main

        from hhemt.cli import app

        return typer.main.get_command(app)

    def facts(self, func) -> _Facts:
        """The sites, pending CLI calls and followable callees found in ``func``'s own body."""
        if func in self._facts:
            return self._facts[func]
        facts = _Facts()
        self._facts[func] = facts
        ctx = self._context(func)
        if ctx is None:
            return facts
        path = func.__code__.co_filename
        for call in (n for n in ast.walk(ctx.node) if isinstance(n, ast.Call)):
            where = self._where(path, call.lineno)
            text = self._line(path, call.lineno)
            callee = call.func
            if isinstance(callee, ast.Attribute) and callee.attr in _RUN_ATTRS:
                if not self._is_subprocess(callee.value, ctx):
                    dry = any(
                        kw.arg == "dry_run" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                        for kw in call.keywords
                    )
                    facts.sites.append(_Site("attr", where, text, dry, False, callee.attr))
                    continue
            if self._cli_call(call, ctx, facts, where, text):
                continue
            target = self._followable(self.resolve(callee, ctx))
            if target is not None:
                facts.callees.append(target)
        return facts

    def _cli_call(self, call: ast.Call, ctx: _Context, facts: _Facts, where: str, text: str) -> bool:
        """Record ``call`` as a pending CLI call when it is ``invoke(app, argv)`` or a child ``hhemt`` call."""
        callee = call.func
        name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
        if name == "invoke" and len(call.args) >= 2:
            app = self._app_for(call.args[0], ctx)
            if app is None:
                return False
            tokens: list[str] = []
            params: set[str] = set()
            flags = self._collect(call.args[1], ctx, tokens, params, 0, set())
            facts.pending.append(_Pending(app, tokens, params, flags, where, text))
            return True
        if not self._is_process_launch(callee, ctx):
            return False
        tokens, params = [], set()
        flags: frozenset[str] = frozenset()
        for arg in [*call.args, *(kw.value for kw in call.keywords)]:
            flags |= self._collect(arg, ctx, tokens, params, 0, set())
        if any(tok == "hhemt" or tok.endswith("/hhemt") for tok in tokens):
            facts.pending.append(_Pending(self._hhemt_app(), tokens, params, flags, where, text))
            return True
        return False

    def reach(self, root) -> _Reach:
        """Everything reachable from ``root``: each function in scope walked once, breadth first."""
        if root in self._reach:
            return self._reach[root]
        out = _Reach()
        seen = {root}
        queue = [root]
        while queue:
            facts = self.facts(queue.pop())
            out.sites.extend(facts.sites)
            out.pending.extend(facts.pending)
            for callee in facts.callees:
                if callee not in seen:
                    seen.add(callee)
                    queue.append(callee)
        self._reach[root] = out
        return out

    # -- command-line calls -------------------------------------------------------------------

    def _commands_named(self, group, tokens: list[str]) -> list:
        """The click commands ``tokens`` name inside ``group``, descending through nested groups.

        A named group contributes its own callback, because click runs a group's callback
        before its subcommand's (``hhemt test analysis`` is itself a group whose callback calls
        ``.test()``). A group named with no callback and no subcommand after it contributes
        nothing: click prints its command list and no workflow is reached.
        """
        found = []
        for tok in tokens:
            cmd = group.commands.get(tok) if hasattr(group, "commands") else None
            if cmd is None:
                continue
            found.append(cmd)
            if hasattr(cmd, "commands"):
                found.extend(self._commands_named(cmd, [t for t in tokens if t != tok]))
        return found

    def _callback_sites(self, cmd, depth: int) -> list[_Site]:
        """The run sites reachable from a click command's callback, memoized per command."""
        key = id(cmd)
        if key not in self._command_sites:
            self._command_sites[key] = []
            callback = self._followable(cmd.callback) if cmd.callback else None
            if callback is not None:
                self._command_sites[key] = self.finalize(self.reach(callback), {}, {}, depth + 1)
        return self._command_sites[key]

    def finalize(
        self,
        reach: _Reach,
        extra: dict[str, list[str]],
        extra_flags: dict[str, frozenset[str]],
        depth: int = 0,
    ) -> list[_Site]:
        """All run sites in ``reach``, with each pending CLI call resolved to a site or dropped.

        ``extra`` maps a parameter name to the strings its fixture or parametrize value spells
        out for this node; ``extra_flags`` maps it to the flags that value CERTAINLY carries
        (those on every return path of a fixture; those of the one value of a parametrize id).
        A pending call becomes a site only when a command it names has a
        callback that reaches a run; one that names no registered command is dropped.
        """
        sites = list(reach.sites)
        if depth > 3:
            return sites
        for pend in reach.pending:
            tokens = list(pend.tokens)
            for param in pend.params:
                tokens.extend(extra.get(param, []))
            if not any(self._callback_sites(cmd, depth) for cmd in self._commands_named(pend.app, tokens)):
                continue
            flags = pend.flags.union(*(extra_flags.get(p, frozenset()) for p in pend.params))
            sites.append(_Site("cli", pend.where, pend.text, _DRY_FLAG in flags, _HELP_FLAG in flags))
        return sites

    # -- the tree-emptiness dimension ---------------------------------------------------------

    def _defining_file(self, obj) -> str | None:
        """The basename of the file ``obj`` is defined in, or None when it has no readable source."""
        if isinstance(obj, type):
            try:
                return Path(inspect.getfile(obj)).name
            except (TypeError, OSError):
                return None
        code = getattr(obj, "__code__", None)
        return Path(code.co_filename).name if code is not None else None

    @staticmethod
    def _literal_bool(call: ast.Call, name: str) -> bool | None:
        """The boolean ``call`` passes for keyword ``name`` as a literal, or None when it passes none."""
        for keyword in call.keywords:
            value = keyword.value
            if keyword.arg == name and isinstance(value, ast.Constant) and isinstance(value.value, bool):
                return value.value
        return None

    def _reachable_functions(self, root) -> frozenset:
        """Every in-scope function reachable from ``root``, ``root`` included, each followed once."""
        if root in self._reach_functions:
            return self._reach_functions[root]
        seen = {root}
        queue = [root]
        while queue:
            for callee in self.facts(queue.pop()).callees:
                if callee not in seen:
                    seen.add(callee)
                    queue.append(callee)
        self._reach_functions[root] = frozenset(seen)
        return self._reach_functions[root]

    def _call_tree_class(self, call: ast.Call, ctx: _Context, path: str) -> tuple[str, str | None] | None:
        """The tree class ONE call supplies, or None when the call names no case factory.

        ``share_scratch_root=True`` is tested before ``start_from_scratch=True`` because it is the
        builder's own precedence: a build that opts back into the shared root is cache-backed even
        though it also wipes. A literal False, a non-literal argument and an absent argument all
        read the shared cache, since the builder's default is False.

        The factory test is the resolved callable's own ``__name__`` prefix, not its module alone:
        those two modules also hold helpers that supply no tree, and treating a helper as a factory
        would report a cache-backed closure where there is none.
        """
        callee = call.func
        name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
        evidence = f"{self._where(path, call.lineno)} {self._line(path, call.lineno)}"
        if name in _CLONE_FACTORY_NAMES:
            return _TREE_OVERRIDE_ISOLATED, evidence
        target = self.resolve(callee, ctx)
        resolved_name = getattr(target, "__name__", name)
        if self._defining_file(target) not in _CASE_FACTORY_MODULES:
            return None
        if not resolved_name.startswith(_CASE_FACTORY_PREFIX):
            return None
        if self._literal_bool(call, "share_scratch_root") is True:
            return _TREE_CACHE_BACKED, evidence
        if self._literal_bool(call, "start_from_scratch") is True:
            return _TREE_OVERRIDE_ISOLATED, evidence
        return _TREE_CACHE_BACKED, evidence

    def _function_tree_class(self, func) -> tuple[str, str | None]:
        """The highest tree class ``func``'s OWN body supplies, with the evidence that decided it."""
        if func in self._tree_cache:
            return self._tree_cache[func]
        self._tree_cache[func] = (_TREE_EMPTY, None)
        best: tuple[str, str | None] = (_TREE_EMPTY, None)
        ctx = self._context(func)
        if ctx is not None:
            path = func.__code__.co_filename
            for call in (node for node in ast.walk(ctx.node) if isinstance(node, ast.Call)):
                found = self._call_tree_class(call, ctx, path)
                if found is not None and _TREE_RANK[found[0]] > _TREE_RANK[best[0]]:
                    best = found
        self._tree_cache[func] = best
        return best

    def tree_class_of(self, roots) -> tuple[str, str | None]:
        """The highest tree class across everything ``roots`` reach, with the evidence for it.

        Candidates are ordered by (root index, file, first line) before the maximum is taken, so a
        tie between two equal-ranked functions resolves deterministically and toward the node's OWN
        function: the artifact's evidence string must not vary between two runs over one tree.
        """
        best: tuple[str, str | None] = (_TREE_EMPTY, None)
        candidates = [
            (index, func.__code__.co_filename, func.__code__.co_firstlineno, func)
            for index, root in enumerate(roots)
            for func in self._reachable_functions(root)
        ]
        for _, _, _, func in sorted(candidates, key=lambda candidate: candidate[:3]):
            found = self._function_tree_class(func)
            if _TREE_RANK[found[0]] > _TREE_RANK[best[0]]:
                best = found
        return best

    # -- one collected item -------------------------------------------------------------------

    def describe(self, item) -> dict:
        """One collected item as plain data: its node id, coverage, run sites and tree class."""
        func = getattr(item, "function", None)
        roots = []
        extra: dict[str, list[str]] = {}
        extra_flags: dict[str, frozenset[str]] = {}
        if func is not None:
            top = self._followable(func)
            if top is not None:
                roots.append(top)
            info = getattr(item, "_fixtureinfo", None)
            defs = getattr(info, "name2fixturedefs", {}) if info is not None else {}
            for name in item.fixturenames:
                for fixturedef in defs.get(name, ()):
                    fn = self._followable(fixturedef.func)
                    if fn is not None:
                        roots.append(fn)
                        found, _, certain = self._return_tokens(fn)
                        if found:
                            extra.setdefault(name, []).extend(found)
                        # An override chain defines the name more than once: only a flag every
                        # definition carries on every return path is certain.
                        extra_flags[name] = extra_flags[name] & certain if name in extra_flags else certain
            callspec = getattr(item, "callspec", None)
            for name, value in (getattr(callspec, "params", None) or {}).items():
                values = value if isinstance(value, (list, tuple)) else [value]
                strings = [v for v in values if isinstance(v, str)]
                extra.setdefault(name, []).extend(strings)
                extra_flags[name] = frozenset(v for v in strings if v in _FLAGS)
        sites: list[_Site] = []
        for root in roots:
            sites.extend(self.finalize(self.reach(root), extra, extra_flags))
        unique = {(s.kind, s.where): s for s in sites}
        tree, evidence = self.tree_class_of(roots) if unique else (None, None)
        described = {
            "id": item.nodeid,
            "covered": item.get_closest_marker("compile_tier") is not None,
            "skip": any(mark.name == "skip" for mark in item.iter_markers()),
            "sites": [vars(s) for s in unique.values()],
            "tree": tree,
            "tree_evidence": evidence,
            "completed_fixture": next((name for name in item.fixturenames if _is_completed_fixture(name)), None),
        }
        if tree == _TREE_EMPTY:
            described["fixtures"] = sorted(item.fixturenames)
        return described


class _Recorder:
    """A pytest plugin for the child process: records every collected item and any deselection."""

    def __init__(self, out_path: str):
        """Remember where to write the report."""
        self.out_path = out_path
        self.deselected = 0

    def pytest_deselected(self, items):
        """Count items the child's own options or ini dropped, so the parent can refuse a narrowed run."""
        self.deselected += len(items)

    def pytest_collection_finish(self, session):
        """Describe every item after all `pytest_collection_modifyitems` hooks (the `compile_tier` mark) ran."""
        graph = _CallGraph()
        nodes = [graph.describe(item) for item in session.items]
        Path(self.out_path).write_text(json.dumps({"nodes": nodes, "deselected": self.deselected}), encoding="utf-8")


def run_child(out_path: str) -> int:
    """Collect ``tests/`` in this process with `_Recorder` attached; return pytest's exit code."""
    return int(
        pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", str(_TESTS_DIR)], plugins=[_Recorder(out_path)])
    )


_CHILD_SNIPPET = "import sys; from tests.test_build_request_census import run_child; sys.exit(run_child(sys.argv[1]))"


def _collect_in_child(out_path: Path) -> dict:
    """Run the collecting child and return its report, failing with its output when it did not collect."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST") and not key.startswith("PYTEST_XDIST")
    }
    env["HHEMT_FORBID_COMPILE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SNIPPET, str(out_path)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0 and out_path.exists(), (
        f"the collecting child exited {proc.returncode} and wrote {'a' if out_path.exists() else 'no'} report; "
        f"the census cannot run on a tree that does not collect.\n{(proc.stdout + proc.stderr)[-3000:]}"
    )
    return json.loads(out_path.read_text(encoding="utf-8"))


def _all_sites_are_noops(node: dict) -> bool:
    """True when every run-reaching call of ``node`` is a literal dry run or a `--help` call."""
    return all(site["dry"] or site["help"] for site in node["sites"])


def _is_completed_fixture(name: str) -> bool:
    """True when ``name`` is a fixture supplying an already-completed synthetic analysis tree.

    ``synthetic_multisim_completed`` and ``synthetic_sensitivity_completed`` are the session trees,
    and so are their ``_isolated`` clones, so one rule recognises both arms of the closure.
    """
    return name.startswith("synthetic_") and "_completed" in name


def _artifact_path() -> Path:
    """Where the machine artifact goes: the env override when set, else the gitignored pytest cache."""
    override = os.environ.get(_ARTIFACT_ENV)
    return Path(override) if override else _ARTIFACT_DEFAULT


def _unique_sites(nodes: list[dict]) -> list[dict]:
    """Every distinct run site across ``nodes``, keyed by kind and location, in a stable order.

    A fixture's site appears in the closure of every node consuming it, so counting closures
    would multiply one call by its consumers; the population is deduplicated here.
    """
    unique = {(site["kind"], site["where"]): site for node in nodes for site in node["sites"]}
    return [unique[key] for key in sorted(unique)]


def _disposition(node: dict, table: dict[str, _Excuse]) -> str:
    """How one run-reaching node is accounted for: coverage, an auto-excuse, a table row, or nothing."""
    if node["covered"]:
        return "covered"
    if node["skip"] or _all_sites_are_noops(node):
        return "auto-excused"
    return "table-excused" if node["id"] in table else "undispositioned"


def _rung_artifact(nodes: list[dict]) -> dict:
    """The ladder's rung 1 — every distinct ATTRIBUTE site, the files holding one, and the node count.

    The ladder row this block is published beside is "any actuator attribute, any receiver", so the
    population is attribute call sites. The walk also records command-line invocations, which reach a
    run only through a command whose callback does; those are the other site kind and counting them
    here would publish a population no ladder row defines, incomparable with the figure beside it.
    They stay in each node's own ``sites``, every site carrying its ``kind``.
    """
    sites = [site for site in _unique_sites(nodes) if site["kind"] == "attr"]
    files = sorted({site["where"].rsplit(":", 1)[0] for site in sites})
    attr_nodes = [{"id": n["id"]} for n in nodes if any(s["kind"] == "attr" for s in n["sites"])]
    return {
        "unit": "sites",
        "count": len(sites),
        "file_count": len(files),
        "node_count": len(attr_nodes),
        "nodes": attr_nodes,
        "files": files,
        "sites": sites,
    }


def _re_derive_artifact(nodes: list[dict], table: dict[str, _Excuse]) -> dict:
    """The re-derive population, and the triage delta the attribute widening creates.

    ``count``/``count_live`` are SITES, the unit the shape's figures use. ``delta`` is NODES,
    because a node is the unit a person adjudicates: one table row covers a whole node.
    """
    sites = [site for site in _unique_sites(nodes) if site["attr"] in _RE_DERIVE_ATTRS]
    live = [site for site in sites if not (site["dry"] or site["help"])]
    delta = [n for n in nodes if n["sites"] and all(s["attr"] in _RE_DERIVE_ATTRS for s in n["sites"])]
    return {
        "unit": "sites",
        "count": len(sites),
        "count_live": len(live),
        "live_predicate": "not (a literal dry_run=True, or a --help call)",
        "node_count": len([n for n in nodes if any(s["attr"] in _RE_DERIVE_ATTRS for s in n["sites"])]),
        "nodes": [
            {"id": n["id"], "completed_fixture": n["completed_fixture"], "disposition": _disposition(n, table)}
            for n in nodes
            if any(s["attr"] in _RE_DERIVE_ATTRS for s in n["sites"])
        ],
        "sites": sites,
        "delta": {
            "unit": "nodes",
            "count": len(delta),
            "undispositioned": len([n for n in delta if _disposition(n, table) == "undispositioned"]),
            "nodes": [{"id": n["id"], "disposition": _disposition(n, table)} for n in delta],
        },
    }


def _empty_tree_artifact(nodes: list[dict], table: dict[str, _Excuse]) -> dict:
    """The nodes whose closure supplies no analysis tree, each carrying its four excuse terms.

    ``count_actionable`` is the subset carrying NONE of them; every other member is already
    covered, skip-marked, a literal no-op, or in the excuse table, and so needs no conversion.
    """
    entries = [
        {
            "id": node["id"],
            "covered": node["covered"],
            "skip": node["skip"],
            "noop": _all_sites_are_noops(node),
            "table_excused": node["id"] in table,
            "sites": node["sites"],
            "fixtures": node.get("fixtures", []),
        }
        for node in nodes
        if node["tree"] == _TREE_EMPTY
    ]
    actionable = [e for e in entries if not (e["covered"] or e["skip"] or e["noop"] or e["table_excused"])]
    return {"unit": "nodes", "count": len(entries), "count_actionable": len(actionable), "nodes": entries}


def test_every_run_reaching_test_requests_the_build(tmp_path, capsys):
    """Property: a test whose call graph reaches a workflow run requests the build, or is excused.

    Class: every collected test node in `tests/`, of any model, in any module, class method or
    parametrize id. Why it is required: on a warm cache a test that reaches a run without the
    build passes, so only a fresh checkout reveals the omission, and no other check in the suite
    reds on it. What kills it: a test that calls `.run()` and requests no build fixture, a
    stale excuse-table row, or a child collection narrowed by `-m`/`-k`/`PYTEST_ADDOPTS`. A
    second correct implementation under which it still passes: one that decides run-reaching
    from `--setup-plan` output or from a runtime guard on `.run()`, since only the failing and
    passing node sets are asserted, not how reachability is computed.
    """
    report = _collect_in_child(tmp_path / "census.json")
    nodes = {node["id"]: node for node in report["nodes"]}
    assert _RE_DERIVE_ATTRS <= _RUN_ATTRS, "the re-derive attributes must be run-reaching attributes"
    bad_categories = [e.category for e in _EXCUSES if e.category not in _CATEGORIES]
    assert not bad_categories, f"excuse-table categories outside {sorted(_CATEGORIES)}: {bad_categories}"
    lost_quotes = [
        f"{e.source}: {e.quote}" for e in _EXCUSES if e.quote not in (_REPO_ROOT / e.source).read_text(encoding="utf-8")
    ]
    assert not lost_quotes, "excuse-table quotes no longer found in their source file:\n  " + "\n  ".join(lost_quotes)
    table = {node_id: entry for entry in _EXCUSES for node_id in entry.node_ids}

    reaching = [n for n in nodes.values() if n["sites"]]
    uncovered = [n for n in reaching if not n["covered"]]
    auto_excused = [n for n in uncovered if n["skip"] or _all_sites_are_noops(n)]
    unexcused = [n for n in uncovered if n not in auto_excused and n["id"] not in table]

    stale = []
    for node_id in table:
        node = nodes.get(node_id)
        if node is None:
            stale.append(f"{node_id}: no longer collects")
        elif not node["sites"]:
            stale.append(f"{node_id}: no longer reaches a run")
        elif node["covered"]:
            stale.append(f"{node_id}: is now covered by the compile_tier marker")

    artifact = {
        "schema": 1,
        "run_attributes": sorted(_RUN_ATTRS),
        "re_derive_attributes": sorted(_RE_DERIVE_ATTRS),
        "collected": len(nodes),
        "run_reaching": len(reaching),
        "rung_1": _rung_artifact(reaching),
        "re_derive": _re_derive_artifact(reaching, table),
        "empty_tree": _empty_tree_artifact(reaching, table),
        "tree_classes": {name: len([n for n in reaching if n["tree"] == name]) for name in _TREE_CLASSES},
        "tree_nodes": [
            {
                "id": n["id"],
                "tree": n["tree"],
                "completed_fixture": n["completed_fixture"],
                "evidence": n["tree_evidence"],
            }
            for n in reaching
        ],
    }
    artifact_path = _artifact_path()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=1), encoding="utf-8")

    with capsys.disabled():
        print(
            f"\nbuild-request census: collected={len(nodes)} run_reaching={len(reaching)} "
            f"covered={len(reaching) - len(uncovered)} auto_excused={len(auto_excused)} "
            f"table_excused={len([n for n in uncovered if n['id'] in table and n not in auto_excused])} "
            f"table_size={len(table)} unexcused={len(unexcused)} stale={len(stale)} "
            f"rung_1={artifact['rung_1']['count']}sites/{artifact['rung_1']['file_count']}files "
            f"re_derive={artifact['re_derive']['count']}/{artifact['re_derive']['count_live']}live "
            f"empty_tree={artifact['empty_tree']['count']}/{artifact['empty_tree']['count_actionable']}actionable "
            f"tree_classes={artifact['tree_classes']} artifact={artifact_path}"
        )

    missing = [key for key in ("rung_1", "re_derive", "empty_tree") if key not in artifact]
    assert not missing, f"the census artifact carries no {missing} block(s)"
    assert artifact["rung_1"]["count"] == len(artifact["rung_1"]["sites"])
    assert artifact["re_derive"]["count"] == len(artifact["re_derive"]["sites"])
    assert artifact["re_derive"]["count_live"] <= artifact["re_derive"]["count"]
    assert artifact["empty_tree"]["count"] == len(artifact["empty_tree"]["nodes"])
    assert sum(artifact["tree_classes"].values()) == len(reaching), (
        "a run-reaching node was classified outside the tree dimension"
    )
    assert len(artifact["tree_nodes"]) == len(reaching)
    undelta = [n["id"] for n in artifact["re_derive"]["delta"]["nodes"] if n["disposition"] == "undispositioned"]
    assert artifact["re_derive"]["delta"]["undispositioned"] == 0, (
        "nodes that reach a run only through the re-derive attributes and carry no excuse:\n  " + "\n  ".join(undelta)
    )

    def describe(node: dict) -> str:
        """One failing node with the run sites that make it run-reaching."""
        return node["id"] + "".join(f"\n      {s['where']}  {s['text']}" for s in node["sites"][:3])

    assert report["deselected"] == 0, (
        f"the collecting child deselected {report['deselected']} item(s); a deselection narrows the items it "
        f"records, so the census would be blind to them. Look for -m/-k/--deselect in pytest addopts or in "
        f"conftest hooks."
    )
    assert not unexcused, (
        f"{len(unexcused)} test(s) reach a simulation workflow without requesting the build "
        f"(no compile_tier marker; add `usefixtures('tritonswmm_cpu_compiled')` or an excuse-table row):\n  "
        + "\n  ".join(describe(n) for n in unexcused)
    )
    assert not stale, "stale excuse-table rows (remove them):\n  " + "\n  ".join(stale)
