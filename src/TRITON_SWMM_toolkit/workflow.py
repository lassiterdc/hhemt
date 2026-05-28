"""
Snakemake Workflow Generation Module

This module handles the generation and execution of Snakemake workflows for
TRITON-SWMM simulations. It provides a clean interface for creating workflow
files and submitting them to either local or SLURM execution environments.

Key Components:
- SnakemakeWorkflowBuilder: Main class for workflow generation and submission
- SensitivityAnalysisWorkflowBuilder: Specialized builder for sensitivity analysis workflows
"""

import datetime
import json
import math
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import yaml  # type: ignore

from TRITON_SWMM_toolkit.config.analysis import ClearRawValue
from TRITON_SWMM_toolkit.exceptions import ConfigurationError, WorkflowError
from TRITON_SWMM_toolkit.report_renderers._figure_emission import format_sources_rst
from TRITON_SWMM_toolkit.utils import fast_rmtree

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis
    from .sensitivity_analysis import TRITONSWMM_sensitivity_analysis


from dataclasses import dataclass

# Sentinel env var: when set to "1", _check_and_clear_snakemake_lock silently
# rmtrees .snakemake/locks/ and .snakemake/incomplete/ before every snakemake
# invocation instead of prompting the user. The tests/conftest.py session
# fixture sets this for the duration of every pytest run; production / CLI
# invocations leave it unset so the existing interactive prompt remains the
# default path. Centralized as a constant so the fixture, the helper, and the
# unit test all agree without a hand-synced string literal.
_NON_INTERACTIVE_LOCK_CLEAR_ENV = "TRITON_SWMM_TEST_NON_INTERACTIVE_LOCK_CLEAR"


@dataclass(frozen=True)
class ResolvedForceRerunSpec:
    """Post-resolution force_rerun target set, ready for filesystem globbing.

    The orchestrator resolves event_iloc integers to event_id slugs BEFORE
    constructing this spec (per V0001's stable event-slug invariant); the
    builder helper consumes only slugs/sa_ids. The ``scope`` field names which
    axis the ``tokens`` index into, eliminating the same-type-different-content
    failure mode where ints could be either user-supplied event_ilocs or
    already-resolved slug strings.

    Per cleanup-rerun-delete-redesign Phase 4.
    """

    scope: Literal["all", "none", "sa", "event"]
    tokens: tuple[str, ...]  # () for "all"/"none"; sa_id strings for "sa"; event_id slugs for "event"


@dataclass(frozen=True)
class SnakemakeDiagnostics:
    """Diagnostic flags that travel together to a snakemake CLI call.

    Grouped so each facade layer adds one kwarg rather than several, keeping
    the 19-kwarg submit_workflow signature manageable. Default-constructed
    instance matches prior behavior: no ``--verbose``, no log capture
    (``--printshellcmds`` remains unconditional outside this dataclass).

    The ``reason`` field is retained for API compatibility with the Phase 1
    design surface (sources: synth-test-isolation-and-runtime master plan
    Decision D1) but is a no-op against snakemake 8/9 — the standalone
    ``--reason`` CLI flag was removed in snakemake 8; per-job rerun reasons
    are now emitted automatically when ``--verbose`` is set. Setting
    ``reason=True`` therefore implies ``verbose=True`` to preserve intent
    without invoking the removed flag.
    """

    verbose: bool = False
    reason: bool = False
    log_path: Path | None = None

    @property
    def emit_verbose(self) -> bool:
        """True iff a --verbose flag should be appended (folds the reason
        intent into verbose since standalone --reason was removed in
        snakemake 8)."""
        return self.verbose or self.reason


@dataclass(frozen=True)
class RuleEmissionContext:
    """Per-Snakefile-emission state shared across plot-rule, render-report, and
    rule-all helpers.

    Constructed by ``SnakemakeWorkflowBuilder._make_rule_emission_context()``
    (source-side) and ``bundle.snakefile_generator._make_rule_emission_context()``
    (consume-side). The two sites differ only in which path prefixes they
    supply: source-side passes absolute HPC paths; consume-side passes
    bundle-root-relative paths.

    Attributes
    ----------
    python_executable : str
        Interpreter for shell-block commands.
    log_dir_rel : str
        Snakefile-relative log dir (e.g. ``"logs/plots"`` source-side,
        ``"_logs/plots"`` consume-side).
    conda_env_path : str
        Path to the conda env yaml referenced by rules' ``conda:`` directive.
        Source-side: absolute path under the toolkit checkout. Consume-side:
        empty string (consume-side does not invoke ``--use-conda``).
    config_args_str : str
        Pre-formatted ``--system-config X --analysis-config Y`` shell-arg
        substring. Source-side absolute; consume-side bundle-relative
        (``--system-config cfg_system.yaml --analysis-config cfg_analysis.yaml``).
    is_sensitivity : bool
        Selects multi-sim vs. sensitivity-master rule-set shape.
    static_backend : Literal["matplotlib", "plotly"]
        Per D3 — selects ``output_ext`` per renderer (".png" for matplotlib,
        ".html" for plotly chart figures so Snakemake's report engine
        dispatches via iframe instead of <img>). Required; no in-code default.
    """

    python_executable: str
    log_dir_rel: str
    conda_env_path: str
    config_args_str: str
    is_sensitivity: bool
    static_backend: Literal["matplotlib", "plotly"]


@dataclass(frozen=True)
class RuleSpec:
    """Per-figure rule descriptor consumed by ``_emit_plot_rule``.

    Source-side ``_build_plot_rule_block_*`` methods construct this from
    analysis-instance state. Consume-side ``bundle.snakefile_generator``
    constructs it from bundle data (manifest sidecars + cfg files).

    Attributes
    ----------
    rule_name : str
        Snakemake rule name (e.g. ``"plot_system_overview"``).
    renderer_module : str
        First positional arg to ``python -m TRITON_SWMM_toolkit.report_renderers._cli``
        (e.g. ``"system_overview"``, ``"per_sim_peak_flood_depth"``).
    input_flags : tuple[str, ...]
        Snakefile ``input:`` entries (e.g. ``("_status/e_consolidate_complete.flag",)``).
    output_path_template : str
        Snakefile-relative output path with optional ``{output_ext}`` placeholder
        (resolved by ``_emit_plot_rule`` against ``ctx.static_backend``).
    source_paths : tuple[dict | str, ...]
        Rich form (``{"path": str, "variables": list[str]}``) when available;
        bare ``str`` accepted (``format_sources_rst`` degrades gracefully). The
        consume-side bundle generator passes ``list[str]`` from sidecar
        ``source_paths_relative``.
    wildcards : tuple[str, ...]
        Snakemake wildcards on the output path (e.g. ``("event_id",)`` for
        per-sim rules). Empty tuple for fixed-output rules.
    extra_cli_flags : tuple[str, ...]
        Extra shell-arg substrings (e.g. ``("--event-iloc {params.event_iloc}",)``).
    extra_params : tuple[tuple[str, str], ...]
        Extra ``params:`` block entries as ``(name, value_expr)`` pairs (e.g.
        ``(("event_iloc", "lambda wildcards: ILOC_BY_EVENT_ID[wildcards.event_id]"),)``).
    report_kwargs : dict[str, str] | None
        When non-None, wraps the first output in ``report(...)`` with these
        kwargs (``caption``, ``category``, ``labels``). None for rules that
        do not contribute to Snakemake's report engine.
    resources_yaml : str
        Pre-formatted YAML block for ``resources:`` (output of
        ``_build_resource_block`` on the source side; a small fixed block on
        the consume side).
    log_path_template : str
        Snakefile-relative log path with optional wildcards (e.g.
        ``"logs/plots/per_sim_{event_id}_peak_flood_depth.log"``).
    """

    rule_name: str
    renderer_module: str
    input_flags: tuple[str, ...]
    output_path_template: str
    source_paths: tuple
    wildcards: tuple[str, ...]
    extra_cli_flags: tuple[str, ...]
    extra_params: tuple[tuple[str, str], ...]
    report_kwargs: dict[str, str] | None
    resources_yaml: str
    log_path_template: str
    source_paths_fn_name: str | None = None
    input_label: str = "consolidated"
    # Additional positional inputs emitted alongside the labeled `input_flags`.
    # Positional entries are emitted BEFORE labeled entries because Snakemake
    # parses the input: block as a Python function-call argument list and
    # "positional after keyword" is a SyntaxError.
    additional_inputs: tuple[str, ...] = ()


_OUTPUT_EXT_BY_RENDERER: dict[str, dict[str, str]] = {
    # Plotly chart-renderer outputs are interactive HTML emitted via pio.to_html;
    # extension must be .html so Snakemake's report engine sets mime_type=text/html
    # and dispatches each figure via <iframe> (which loads HTML correctly under
    # both HTTP and file:// double-click). A .svg extension here triggers
    # mime_type=image/svg+xml and an <img> dispatch that fails to parse HTML.
    "system_overview": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_peak_flood_depth": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_conduit_flow": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_per_sa_peak_flood_depth": {"matplotlib": ".png", "plotly": ".html"},
    "per_sim_per_sa_conduit_flow": {"matplotlib": ".png", "plotly": ".html"},
    "sensitivity_benchmarking": {"matplotlib": ".png", "plotly": ".html"},
    "per_analysis_summary": {"matplotlib": ".html", "plotly": ".html"},
    "scenario_status_appendix": {"matplotlib": ".html", "plotly": ".html"},
    "errors_and_warnings": {"matplotlib": ".html", "plotly": ".html"},
    # Disk utilization is a table renderer — emits HTML unconditionally
    # (no matplotlib raster branch). Matches per_analysis_summary /
    # scenario_status_appendix / errors_and_warnings.
    "disk_utilization": {"matplotlib": ".html", "plotly": ".html"},
}


def _output_ext_for(static_backend: Literal["matplotlib", "plotly"], renderer_module: str) -> str:
    """Return the output extension for a renderer under the given static backend.

    See VMS-5 in the Phase 2 plan doc. Three-place output_ext coupling:
    rule output path, rule report() first arg, and rule_all / render_report
    input lists must all use this same extension.
    """
    return _OUTPUT_EXT_BY_RENDERER[renderer_module][static_backend]


def _resolve_rule_all_extensions(
    static_backend: Literal["matplotlib", "plotly"],
) -> dict[str, str]:
    """Return per-renderer extensions for rule_all / render_report inputs.

    `_emit_plot_rule` already resolves ext per-rule via `__OUTPUT_EXT__`
    substitution. The rule_all + render_report `input:` declarations must
    list the same backend-resolved extensions so Snakemake's DAG planner
    can match rule outputs to their consumers. Hardcoding `.png` here
    (the legacy state) breaks DAG-build when static_backend != matplotlib.
    """
    return {key: _output_ext_for(static_backend, key) for key in _OUTPUT_EXT_BY_RENDERER}


def _emit_plot_rule(spec: RuleSpec, ctx: RuleEmissionContext) -> str:
    """Emit a single Snakemake plot rule as a string.

    Single-f-string-template assembly per VMS-5. Per-spec variability
    (literal vs function source_paths, optional event_iloc param,
    optional --event-iloc shell flag, report-kwargs shape) is encoded
    as pre-computed sub-strings consumed by the template. No
    per-renderer branching.
    """
    output_ext = _output_ext_for(ctx.static_backend, spec.renderer_module)
    # NB: use plain string replacement (not .format) so Snakemake's
    # {wildcard} braces in the output_path_template survive unescaped.
    output_path = spec.output_path_template.replace("__OUTPUT_EXT__", output_ext)

    # input: block — `{label} = "<flag>",` for each non-empty input_flag.
    # Label is "consolidated" for multi-sim and sensitivity-sub-analysis plot
    # rules, "master" for sensitivity-master rules. Empty / missing flags emit
    # `input: []` (regen-only bundle Snakefile: plot files exist on disk via
    # the bundle, rules need no upstream status-flag inputs).
    nonempty_flags = tuple(f for f in spec.input_flags if f)
    additional = tuple(p for p in spec.additional_inputs if p)
    if nonempty_flags or additional:
        # Snakemake parses input: as a Python argument list — positional
        # entries must precede keyword entries.
        positional_lines = [f'"{p}",' for p in additional]
        labeled_lines = [f'{spec.input_label} = "{f}",' for f in nonempty_flags]
        input_block = "input:\n        " + "\n        ".join(positional_lines + labeled_lines)
    else:
        input_block = "input: []"

    # output: report(...) wrapper. report_kwargs carries caption,
    # category, optional subcategory, and labels as pre-formatted dict.
    rk = spec.report_kwargs or {}
    labels_str = rk.get("labels", '{"figure": "?"}')
    subcategory_line = f'\n            subcategory="{rk["subcategory"]}",' if "subcategory" in rk else ""
    output_block = (
        f"report(\n"
        f'            "{output_path}",\n'
        f'            caption="{rk.get("caption", "")}",\n'
        f'            category="{rk.get("category", "")}",'
        f"{subcategory_line}\n"
        f"            labels={labels_str},\n"
        f"        )"
    )

    # params: block — either literal source_paths or function reference,
    # plus any extra_params entries (e.g. event_iloc lambdas).
    if spec.source_paths_fn_name is not None:
        fn = spec.source_paths_fn_name
        params_lines = [
            f"        source_paths = {fn},",
            f"        source_paths_rst = lambda w: _fmt_sources_rst({fn}(w)),",
        ]
    else:
        sp = list(spec.source_paths)
        params_lines = [
            f"        source_paths = {sp!r},",
            f"        source_paths_rst = {format_sources_rst(sp)!r},",
        ]
    for name, value in spec.extra_params:
        params_lines.append(f"        {name} = {value},")
    params_block = "\n".join(params_lines)

    # shell: block — optional extra CLI flags appear between config_args
    # and --output. Each emitted on its own indented continuation line.
    extra_flags_block = "".join(f"            {flag} \\\n" for flag in spec.extra_cli_flags)

    # Plot-rule shell uses literal "python" (the rule's conda: env
    # provides the interpreter); only setup / run / process / consolidate
    # / render_report rules use ctx.python_executable's full path.
    return f'''
rule {spec.rule_name}:
    {input_block}
    output:
        {output_block}
    params:
{params_block}
    log: "{spec.log_path_template}"
    conda: "{ctx.conda_env_path}"
    resources: {spec.resources_yaml}
    shell:
        """
        python -m TRITON_SWMM_toolkit.report_renderers._cli {spec.renderer_module} \\
            {ctx.config_args_str} \\
{extra_flags_block}            --output {{output}} \\
            > {{log}} 2>&1
        """
'''


def _emit_render_report_rule(
    plot_output_paths: tuple[str, ...],
    ctx: RuleEmissionContext,
    resources_yaml: str,
) -> str:
    """Emit the ``rule render_report`` block.

    ``plot_output_paths`` is the list of (output_ext-resolved) plot
    paths this rule depends on. Each entry is either a literal path or
    a bare `expand("...", ...)` invocation string (for wildcarded
    per-sim/per-sa rules); the helper inserts each verbatim into the
    input: block.
    """
    input_lines = "\n        ".join(f"{p}," for p in plot_output_paths)
    return f'''
rule render_report:
    input:
        {input_lines}
    output:
        "analysis_report.{{format}}"
    wildcard_constraints:
        format="zip|html"
    log: "{ctx.log_dir_rel}/render_report_{{format}}.log"
    resources:
{resources_yaml}
    shell:
        """
        {ctx.python_executable} -m TRITON_SWMM_toolkit.render_report_runner \\
            {ctx.config_args_str} \\
            --format {{wildcards.format}} \\
            > {{log}} 2>&1
        """
'''


def _emit_rule_all(
    *,
    status_flags: tuple[str, ...],
    plot_output_paths: tuple[str, ...],
    render_report_targets: tuple[str, ...],
    ctx: RuleEmissionContext,
) -> str:
    """Emit the ``rule all`` block.

    Each tuple's entries are written verbatim into ``input:`` (already
    quote-wrapped or `expand(...)`-wrapped by caller, per the
    source-side emission conventions).
    """
    lines = list(status_flags) + list(plot_output_paths) + list(render_report_targets)
    input_block = "\n        ".join(f"{e}," for e in lines)
    return f"""
rule all:
    input:
        {input_block}
"""


def _emit_report_artifacts(dest_root: Path) -> None:
    """Copy report_templates/ -> {dest_root}/report/.

    Uses importlib.resources for package-resource resolution (robust across
    editable and site-packages installs). Falls back to Path(__file__) arithmetic
    only when importlib.resources is unavailable. Requires report_templates/
    to ship as package data under src/TRITON_SWMM_toolkit/ via pyproject.toml's
    [tool.setuptools.package-data] entry.

    The Jinja2 workflow_description.rst.j2 template is renamed to
    workflow_description.rst on copy because Snakemake's report engine
    renders all .rst files through Jinja2 — the .j2 extension is a
    repo-side convention, not a Snakemake one.

    Lifted to module scope at Plan Phase 2 (per VMS-1 + F-I3 resolution) so
    the bundle-side consume path (Bundle.regenerate_report) can call it
    without an analysis-instance dependency.
    """
    try:
        from importlib.resources import files as _resource_files

        src_templates = Path(str(_resource_files("TRITON_SWMM_toolkit") / "report_templates"))
    except (ImportError, ModuleNotFoundError):
        src_templates = Path(__file__).parent / "report_templates"

    dst_report = dest_root / "report"
    dst_report.mkdir(parents=True, exist_ok=True)
    (dst_report / "report.css").write_text((src_templates / "report.css").read_text())
    captions_dst = dst_report / "captions"
    captions_dst.mkdir(exist_ok=True)
    for cap in (src_templates / "captions").glob("*.rst"):
        (captions_dst / cap.name).write_text(cap.read_text())
    (dst_report / "workflow_description.rst").write_text((src_templates / "workflow_description.rst.j2").read_text())


def _sanitize_rule_name(token: str) -> str:
    """Coerce a sentinel rule_token into a valid Snakemake/Python identifier.

    Snakemake rule names must be valid Python identifiers (no ``-``, ``.``).
    Sentinel rule_tokens written by :mod:`run_simulation_runner` follow the
    pattern ``run_{model_type}_evt-{event_id}`` (multisim) or
    ``simulation_sa_{sa_id}_evt-{event_id}`` (sensitivity); both contain
    literal hyphens that must be replaced for use as a Snakemake rule name.

    Per sentinel-system-v2 Phase 2.
    """
    return token.replace("-", "_").replace(".", "_")


# Pinned to the format emitted by snakemake-executor-plugin-slurm/submit_string.py:29
# (--comment rule_{job.name}_wildcards_{...}). Bump on plugin version changes.
_COMMENT_RULE_PREFIXES: tuple[str, ...] = ("rule_run_", "rule_simulation_sa_")


def _slurm_job_is_live(job_id: str, *, timeout_s: float = 10.0) -> bool:
    """True if ``job_id`` is PENDING/RUNNING/etc. in squeue.

    squeue is authoritative for live jobs (sacct gaps cannot hide a live job).
    Absent from squeue → treated as not-live (stale). Callers using this for
    the at-most-once-execution guard should reclaim the sentinel when this
    returns False.

    On ``subprocess.TimeoutExpired`` (controller unresponsive) returns False
    and emits a stderr warning — the caller's at-most-once guard treats
    unknown state as not-live, which is the safe direction (a real live job
    that's skipped here would simply be queued and Snakemake would re-detect
    it on the next submit; the alternative — blocking ``submit_workflow()``
    indefinitely on a hung controller — is strictly worse).
    """
    _LIVE = {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "REQUEUED",
        "RESIZING",
        "SUSPENDED",
    }
    try:
        r = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[reconcile] WARNING: squeue {job_id} timed out after {timeout_s}s — treating as not-live",
            file=sys.stderr,
            flush=True,
        )
        return False
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip().split()[0] in _LIVE
    return False


# Terminal sacct State codes that mean the job is gone. Anything NOT in this
# set AND present in sacct is treated as alive (still in the scheduler).
_SACCT_DEAD_STATES: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
    }
)


def _sacct_states_batched(job_ids: list[str], *, timeout_s: float = 30.0) -> dict[str, tuple[str, str, str]]:
    """Batched sacct probe: ONE call for N job-ids (R5 scheduler-politeness).

    Returns ``{job_id: (state, exit_code, reason)}`` for every job-id that
    sacct returned a row for. Job-ids ABSENT from the result map are UNKNOWN
    (purged from the accounting DB, or job-id reuse gap) and must be handled
    by the caller's mtime-age fallback — never silently treated as alive.

    ``CANCELLED by <uid>`` is normalized to ``CANCELLED``. On subprocess
    timeout or non-zero return, returns an EMPTY map — every job-id then falls
    to UNKNOWN, which the caller's mtime-age tiebreak resolves safely.
    """
    if not job_ids:
        return {}
    try:
        r = subprocess.run(
            ["sacct", "-j", ",".join(job_ids), "-n", "-P", "-X", "-o", "JobIDRaw,State,ExitCode,Reason"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[reconcile] WARNING: sacct batched probe ({len(job_ids)} jobs) "
            f"timed out after {timeout_s}s — all unresolved job-ids fall to "
            f"UNKNOWN (mtime-age tiebreak applies)",
            file=sys.stderr,
            flush=True,
        )
        return {}
    except FileNotFoundError:
        # sacct not on PATH (local mode, or a misconfigured HPC env). Same
        # "probe could not resolve state" outcome as a timeout: return an empty
        # map so every job-id falls to UNKNOWN and the mtime-age tiebreak
        # resolves it safely. Surfaced once per reconcile so a genuinely-missing
        # sacct on HPC stays visible.
        print(
            f"[reconcile] WARNING: sacct not found on PATH — batched probe "
            f"({len(job_ids)} jobs) skipped; all unresolved job-ids fall to "
            f"UNKNOWN (mtime-age tiebreak applies)",
            file=sys.stderr,
            flush=True,
        )
        return {}
    if r.returncode != 0:
        return {}
    out: dict[str, tuple[str, str, str]] = {}
    for line in r.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid, state, exit_code, reason = parts[0], parts[1], parts[2], parts[3]
        out[jid] = (state.split()[0], exit_code, reason)
    return out


def _max_plausible_job_lifetime_min(cfg_analysis, *, slack_min: int = 30) -> int:
    """Upper bound on how long a sim job could plausibly run: its own SLURM
    walltime + slack (queue/startup/accounting lag). Single source of truth for
    BOTH the wait-rule poll cap (R-WAITCAP) and the R-STALE UNKNOWN-bucket
    mtime-age fail-safe. Falls back to hpc_max_wait_for_inflight_min only when
    hpc_total_job_duration_min is unset (e.g. local mode)."""
    base = cfg_analysis.hpc_total_job_duration_min
    if base is None:
        return cfg_analysis.hpc_max_wait_for_inflight_min
    return base + slack_min


class _ClearedToken(NamedTuple):
    """Self-describing record for an R-STALE-reclaimed stale token (replaces the
    positional 4-tuple so the reconcile surface and tests read field names)."""

    rule_token: str
    job_id: str
    state: str
    reason: str


class SnakemakeWorkflowBuilder:
    """
    Builder class for generating and executing Snakemake workflows.

    This class encapsulates all Snakemake-related functionality including:
    - Snakefile content generation
    - Dynamic configuration generation
    - Local execution
    - SLURM/HPC execution

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The parent analysis object containing configuration and paths
    """

    def __init__(self, analysis: "TRITONSWMM_analysis"):
        """
        Initialize the workflow builder.

        Parameters
        ----------
        analysis : TRITONSWMM_analysis
            The parent analysis object containing configuration and paths
        """
        self.analysis = analysis
        self.cfg_analysis = analysis.cfg_analysis
        self.system = analysis._system
        self.analysis_paths = analysis.analysis_paths
        # Prefer an explicit interpreter path for generated shell commands.
        # If analysis stores a generic command ("python"/"python3"), use
        # the current interpreter running this process to avoid PATH issues.
        configured_python = str(analysis._python_executable)
        if configured_python in {"python", "python3"}:
            self.python_executable = sys.executable
        else:
            self.python_executable = configured_python

    def _get_conda_env_path(self) -> Path:
        """Get absolute path to conda environment file.

        The path is embedded in generated Snakefiles via the 'conda:' directive, but
        --use-conda is not currently passed to Snakemake, so the directive is inert.
        The two-environment split is aspirational; this file is currently the single
        working environment for all toolkit work.
        """
        triton_toolkit_root = Path(__file__).parent.parent.parent
        return triton_toolkit_root / "workflow" / "envs" / "triton_swmm.yaml"

    def _get_snakemake_base_cmd(self) -> list[str]:
        """Return command prefix for invoking Snakemake.

        Prefer `python -m snakemake` so execution works even when the
        `snakemake` console script is not on PATH.
        """
        return [sys.executable, "-m", "snakemake"]

    def _check_and_clear_snakemake_lock(
        self,
        snakefile_path: Path,
        dry_run: bool,
        verbose: bool = True,
        working_dir: Path | None = None,
    ) -> None:
        """Check for a stale Snakemake lock and prompt the user to clear it.

        Snakemake leaves lock files in .snakemake/locks/ when a workflow is
        killed (e.g. SLURM time limit). If not cleared before the next run,
        Snakemake exits immediately with LockException, wasting any queued
        compute allocation.

        Skipped when dry_run=True — dry runs don't submit anything, so a lock
        is not dangerous, and the real submission call will check again.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile (used to build the --unlock command).
        dry_run : bool
            If True, skip the lock check entirely.
        verbose : bool
            If True, print status messages.
        working_dir : Path | None, default None
            Snakemake working directory whose ``.snakemake/locks/`` subtree
            should be cleared. When ``None`` (the run/submit default), falls
            through to ``self.analysis_paths.analysis_dir`` — the existing
            behavior, unchanged. The reprocess path passes
            ``analysis_dir / ".snakemake_reprocess"`` so the reprocess driver
            clears its own lock subtree rather than the main ``.snakemake/``.

        Raises
        ------
        WorkflowError
            If lock files are found and the user declines to unlock, or if
            snakemake --unlock itself fails.
        """
        if dry_run:
            return
        wd = working_dir if working_dir is not None else self.analysis_paths.analysis_dir
        snakemake_state = wd / ".snakemake"
        # Non-interactive test path: silently rmtree locks/ AND incomplete/ and
        # re-create log/ for the tee target. Production / CLI invocations
        # leave the env var unset and fall through to the interactive prompt.
        # Phase 1 of synth-test-isolation-and-runtime — Decision D1-Option-D
        # routes the unconditional pre-snakemake clear through this site so
        # fixtures don't need to clear separately. metadata/ is deliberately
        # untouched: start_from_scratch=True wipes the entire analysis_dir on
        # the from-scratch path, and on the cached path metadata must persist
        # for rerun-triggers to work.
        import os

        if os.environ.get(_NON_INTERACTIVE_LOCK_CLEAR_ENV) == "1":
            for sub in ("locks", "incomplete"):
                target = snakemake_state / sub
                if target.exists():
                    fast_rmtree(target)
            (snakemake_state / "log").mkdir(parents=True, exist_ok=True)
            return
        locks_dir = snakemake_state / "locks"
        lock_files = list(locks_dir.glob("*.lock")) if locks_dir.exists() else []
        if not lock_files:
            return

        lock_names = ", ".join(f.name for f in lock_files)
        print(
            f"[Snakemake] WARNING: Stale lock files detected in {locks_dir}:",
            flush=True,
        )
        print(f"[Snakemake]   {lock_names}", flush=True)
        print(
            "[Snakemake] This usually means a previous job was killed before Snakemake "
            "could clean up.\n"
            "[Snakemake] Only unlock if no other Snakemake process is currently running "
            "in this directory.",
            flush=True,
        )

        response = input("[Snakemake] Run snakemake --unlock and proceed? [y/N]: ").strip()
        if response.lower() != "y":
            manual_cmd = f"{sys.executable} -m snakemake --unlock --snakefile {snakefile_path}"
            raise WorkflowError(
                phase="pre-submission lock check",
                return_code=1,  # sentinel: user aborted (WorkflowError requires int)
                stderr=(
                    "Workflow submission aborted. If no other Snakemake process is "
                    f"running, unlock manually and retry:\n  {manual_cmd}"
                ),
            )

        unlock_cmd = self._get_snakemake_base_cmd() + [
            "--unlock",
            "--snakefile",
            str(snakefile_path),
        ]
        if verbose:
            print(f"[Snakemake] Running: {' '.join(unlock_cmd)}", flush=True)

        result = subprocess.run(
            unlock_cmd,
            cwd=str(wd),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise WorkflowError(
                phase="snakemake --unlock",
                return_code=result.returncode,
                stderr=result.stderr,
            )
        if verbose:
            print("[Snakemake] Unlock successful. Proceeding.", flush=True)

    def _reconcile_inflight_submissions(self, analysis_dir: Path | None = None) -> list[tuple[str, str]]:
        """At-most-once-execution guard: reconcile prior-driver simulation submissions.

        Sweeps the ``_status/_submitted/`` sentinel directory and classifies
        each sentinel via :meth:`_classify_via_state_markers` (sentinel-only
        liveness — no SLURM CLI). Completed/failed sentinels are reclaimed
        (deleted) by the classifier; sentinels with neither marker are returned
        as the alive set so the caller can substitute wait-rules for run-rules
        at Snakefile-build time (v2 graceful-rerun).

        Fast path: returns immediately with zero SLURM calls when no
        sentinels exist (the common case for fresh analyses).

        Parameters
        ----------
        analysis_dir : Path | None, default None
            Analysis directory whose ``_status/`` subtree is swept. ``None``
            uses ``self.analysis_paths.analysis_dir`` (multisim / master). The
            sensitivity path passes each per-sub-analysis dir
            (``master/subanalyses/{analysis_id}``) because sensitivity markers
            live under the sub-analysis dir, not the master dir (Spec D).

        Returns
        -------
        list of (rule_token, slurm_jobid) tuples
            The alive set: submitted-sentinels for which neither a completed
            nor a failed marker has appeared. Empty when no in-flight work
            remains.
        """
        base_dir = analysis_dir if analysis_dir is not None else self.analysis_paths.analysis_dir
        submitted_dir = base_dir / "_status" / "_submitted"
        sentinels = sorted(submitted_dir.glob("*.json")) if submitted_dir.exists() else []
        if not sentinels:
            return []  # fast path: no alive set

        # v2 graceful-rerun: classify via sentinel state markers (no SLURM CLI).
        # The wait-rule emission layer consumes the alive set to substitute
        # wait-on-sentinel rules for the in-flight set.
        marker_less = self._classify_via_state_markers(sentinels, reclaim_completed=True, analysis_dir=base_dir)
        # R-STALE: authoritatively classify the marker-less set via ONE batched
        # sacct call so a dead-without-marker token is never held alive (and
        # never blocks the 8h wait-rule cap). DEAD tokens are reclaimed here.
        alive, cleared = self._classify_stale_via_sacct(marker_less, analysis_dir=base_dir)
        if cleared:
            print(
                f"[reconcile] R-STALE: cleared {len(cleared)} stale SLURM token(s) — "
                f"these jobs terminated WITHOUT emitting a success/failure marker and "
                f"INDICATE A BUG TO INVESTIGATE:",
                flush=True,
            )
            for rule_token, jid, state, reason in cleared:
                print(
                    f"[reconcile]   {rule_token} (job {jid}): {state}"
                    f"{f' ({reason})' if reason and reason != 'None' else ''}",
                    flush=True,
                )
        if alive:
            print(
                f"[reconcile] v2 graceful-rerun: {len(alive)} in-flight rule(s) "
                f"detected via sentinel state markers; emitting wait-rules in lieu "
                f"of resubmit. Rule tokens: {sorted(t for t, _ in alive)}",
                flush=True,
            )
        return alive

    def _recover_inflight_via_comment(self, known_jobids: set[str]) -> list[tuple[str, str]]:
        """Recover in-flight simulation jobs missed by the sentinel sweep.

        Catches the lost-sentinel window: a driver that died post-``sbatch``
        but before the worker wrote its sentinel. Queries ``sacct`` for the
        current user's recent jobs and matches the SLURM ``--comment`` field
        the executor plugin sets (``rule_{job.name}_wildcards_{...}``). Live
        jobs whose ids are not already in ``known_jobids`` are returned.

        Empirical-tripwire (Phase 1 DoD note): the assumption that
        ``sacct -o Comment`` returns the executor's
        ``rule_{job.name}_wildcards_{...}`` string on this cluster is
        unconfirmed for UVA Rivanna / Frontier at Phase 1 close. To make the
        assumption grep-detectable in HPC stderr logs rather than silently
        elided, this method emits a one-line summary of every invocation
        when sacct returned any rows: how many rows scanned, how many had
        comments matching ``_COMMENT_RULE_PREFIXES``, and how many of those
        were live. A persistent ``0 prefix-matched`` line in production logs
        is the signal that the comment-format assumption has drifted and
        the recovery branch has degenerated to a no-op (which falls back
        cleanly to the sentinel-only guard — degradation, not incorrectness).

        Parameters
        ----------
        known_jobids
            Job-ids already accounted for by the sentinel sweep; excluded
            from the recovery result to avoid double-counting.

        Returns
        -------
        list of (label, job_id) tuples
            Labels are prefixed with ``comment:`` so the caller can
            distinguish recovery-path hits from sentinel hits in the
            ``WorkflowError`` listing.
        """
        import getpass

        out = subprocess.run(
            [
                "sacct",
                "-u",
                getpass.getuser(),
                "-n",
                "-P",
                "-o",
                "JobIDRaw,State,Comment",
            ],
            capture_output=True,
            text=True,
        )
        found: list[tuple[str, str]] = []
        rows_scanned = 0
        prefix_matched = 0
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                rows_scanned += 1
                jid, _state, comment = parts[0], parts[1], parts[2]
                if jid in known_jobids:
                    continue
                if comment.startswith(_COMMENT_RULE_PREFIXES):
                    prefix_matched += 1
                    if _slurm_job_is_live(jid):
                        found.append((f"comment:{comment}", jid))
        # Empirical-tripwire stderr line — only when sacct returned data, to
        # avoid noise on fresh clusters with no recent user jobs. The line
        # is the falsifiable artifact for the Phase 1 DoD's pending empirical
        # confirmation step.
        if rows_scanned > 0:
            print(
                f"[reconcile] sacct-comment recovery: {rows_scanned} rows scanned, "
                f"{prefix_matched} prefix-matched, {len(found)} live",
                file=sys.stderr,
                flush=True,
            )
        return found

    def _pre_snakemake_invocation_guards(
        self,
        snakefile_path: Path,
        dry_run: bool,
        verbose: bool,
        working_dir: Path | None = None,
    ) -> None:
        """Shared pre-Snakemake-invocation guard sequence.

        Threaded into every submit-path call site (local, single-job, tmux)
        so the at-most-once reconciliation runs on every SLURM-submitting
        code path and the dry-run skip stays uniform across all three modes.
        Order matters: the lock check runs first (its prompt is the
        highest-friction interactive step and a stale lock would defeat any
        downstream reconciliation), then the reconciliation runs only when
        ``dry_run`` is False (a dry run plans without submitting, so an
        in-flight duplicate does not yet matter).

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile (passed to the lock check for --unlock).
        dry_run : bool
            If True, skip the lock check and reconciliation entirely.
        verbose : bool
            If True, print status messages from the lock check.
        working_dir : Path | None, default None
            Snakemake working directory whose ``.snakemake/locks/`` subtree
            should be cleared by the lock check. When ``None`` (the
            run/submit default), the lock check falls through to
            ``self.analysis_paths.analysis_dir`` — the existing behavior on
            every Phase-1 caller, unchanged. The reprocess path threads
            ``analysis_dir / ".snakemake_reprocess"`` so the reprocess
            driver clears its own lock subtree. The reconciliation guard is
            unaffected by ``working_dir`` — it sweeps the analysis-level
            ``_status/_submitted/`` sentinel directory which is shared
            across run and reprocess paths.
        """
        self._check_and_clear_snakemake_lock(
            snakefile_path,
            dry_run=dry_run,
            verbose=verbose,
            working_dir=working_dir,
        )
        if not dry_run:
            # v2: reconcile here (return discarded) preserves reclaim+detect on
            # every guard call site — including the reprocess path and the
            # single-job/tmux submit methods that do NOT thread an alive-set
            # into Snakefile build. The Snakefile-build callers
            # (generate_snakefile_content / generate_master_snakefile_content)
            # additionally call _reconcile_inflight_submissions to CAPTURE the
            # alive set for wait-rule emission. The double call is idempotent:
            # reconcile only reclaims completed/failed markers and returns a
            # stable alive set. Per sentinel-system-v2 Phase 2 (Spec C — guard
            # call-site fan-out).
            self._reconcile_inflight_submissions()

    def _get_config_args(
        self,
        analysis_config_yaml: Path | None = None,
        system_config_yaml: Path | None = None,
    ) -> str:
        """
        Generate common config path arguments.

        Parameters
        ----------
        analysis_config_yaml : Path | None
            If provided, use this analysis config instead of self.analysis.analysis_config_yaml
        system_config_yaml : Path | None
            If provided, use this system config instead of self.system.system_config_yaml.
            Threaded by SensitivityAnalysisWorkflowBuilder so each sub-analysis rule
            invokes its runner script with the correct per-SA system config (Phase 3).

        Returns
        -------
        str
            Config arguments string
        """
        analysis_cfg = analysis_config_yaml or self.analysis.analysis_config_yaml
        system_cfg = system_config_yaml or self.system.system_config_yaml
        return f"--system-config {system_cfg} \\\n            --analysis-config {analysis_cfg}"

    def _delete_flags_for_force_rerun(
        self,
        spec: ResolvedForceRerunSpec,
    ) -> None:
        """Pre-delete `_status/*.flag` markers so Snakemake's MTIME trigger
        re-fires the dependent rules on the next workflow invocation.

        Per cleanup-rerun-delete-redesign Phase 4 + R10. Snakemake's DAG
        re-planning cascades downstream invalidation automatically once an
        upstream input flag is deleted — the helper itself only deletes the
        directly-matched flags + their `.flag.json` sidecars.

        Glob anchors use delimiter-anchored separators per the FQ3 canonical
        flag-name table — ``*sa-{v}_*.flag`` (non-terminal) AND
        ``*sa-{v}.flag`` (terminal). Substring-only ``*sa-{v}*.flag`` is
        NOT used because it false-matches `sa-1` against `sa-10`, `sa-11`,
        `sa-100`.

        Parameters
        ----------
        spec : ResolvedForceRerunSpec
            Pre-resolved (scope, tokens) target set. ``scope == "none"``
            short-circuits with no filesystem touch.
        """
        if spec.scope == "none":
            return
        status_dir = self.analysis_paths.analysis_dir / "_status"
        if not status_dir.exists():
            return

        matched_flags: set[Path] = set()
        if spec.scope == "all":
            matched_flags.update(status_dir.glob("*.flag"))
        elif spec.scope == "sa":
            for v in spec.tokens:
                matched_flags.update(status_dir.glob(f"*sa-{v}_*.flag"))
                matched_flags.update(status_dir.glob(f"*sa-{v}.flag"))
        elif spec.scope == "event":
            for v in spec.tokens:
                matched_flags.update(status_dir.glob(f"*evt-{v}_*.flag"))
                matched_flags.update(status_dir.glob(f"*evt-{v}.flag"))
        else:
            raise ValueError(f"Unrecognized spec.scope: {spec.scope!r}")

        for flag_path in matched_flags:
            flag_path.unlink(missing_ok=True)
            sidecar = flag_path.with_suffix(flag_path.suffix + ".json")
            sidecar.unlink(missing_ok=True)

    def _build_resource_block(
        self,
        partition: str | None,
        runtime_min: int,
        mem_mb: int,
        nodes: int,
        tasks: int,
        cpus_per_task: int,
        gpus_total: int = 0,
        gpus_per_node_config: int = 0,
        gpu_hardware: str | None = None,
        gpu_alloc_mode: Literal["gres", "gpus"] = "gres",
        mpi: bool = False,
    ) -> str:
        """
        Build a Snakemake resources block.

        Parameters
        ----------
        partition : str | None
            SLURM partition name (defaults to "standard" if None)
        runtime_min : int
            Runtime limit in minutes
        mem_mb : int
            Memory in MB
        nodes : int
            Number of nodes
        tasks : int
            Number of MPI tasks
        cpus_per_task : int
            CPUs per task (OpenMP threads)
        gpus_total : int
            Total GPUs per job (0 if no GPUs)
        gpus_per_node_config : int
            GPUs per node configured for the cluster (0 if no GPUs)
        gpu_hardware : str | None
            GPU model name for SLURM gres/gpus specification
        gpu_alloc_mode : Literal["gres", "gpus"]
            Which SLURM GPU directive to emit in resources
        mpi : bool
            If True, adds mpi=True to resources (required for SLURM executor to set --ntasks > 1)

        Returns
        -------
        str
            Formatted resources block
        """
        if partition is None and (self.cfg_analysis.multi_sim_run_method != "local"):
            raise ValueError("hpc partition must be set when generating SLURM resources")
        partition_name = partition
        if gpus_total > 0 and gpus_per_node_config < 1:
            raise ValueError("hpc_gpus_per_node must be set when requesting GPUs")

        nodes_from_gpu = self._calculate_nodes_for_gpus(gpus_total, gpus_per_node_config)
        sim_nodes = max(nodes, nodes_from_gpu)
        gpus_per_node = math.ceil(gpus_total / sim_nodes) if gpus_total > 0 else 0

        block = f"""        slurm_partition=\"{partition_name}\",
        runtime={runtime_min},"""

        # GPU-job task emission, branched on the duplication hazard.
        #
        # gres-mode multi-GPU (UVA, gpus_total>=2): the snakemake-executor-plugin-
        # slurm jobstep runs the rule command via `srun -n1` (anti-dup), but a
        # parent sbatch carrying --ntasks-per-gpu=1 exports SLURM_NTASKS_PER_GPU=1,
        # which that inner `srun -n1` inherits and SLURM re-expands to N tasks
        # (_handle_ntasks_per_tres_step) -> N duplicate run_simulation_runner copies
        # race on shared _status/hotstart/output. The fix routes this case through
        # the executor's `mpi` + `tasks` path instead: tasks_per_gpu=0 SUPPRESSES
        # --ntasks-per-gpu (submit_string.py:90 `if ntasks_per_gpu >= 1`, issue #316);
        # mpi=True + tasks=N emits a bare --ntasks=N (submit_string.py:97-121) and
        # flips the jobstep to its no-srun branch (jobstep __init__.py:97 — command
        # runs ONCE). set_gres_string still emits --gres=gpu:hw:N. Net sbatch:
        # `--gres=gpu:hw:N --ntasks=N` with NO --ntasks-per-gpu poison var. Per-rank
        # GPU binding is re-established by run_simulation.py's inner gres srun
        # (--ntasks-per-gpu=1 -> tres_bind=single:1), unchanged and proven (P1-b).
        #
        # Single-GPU gres (gpus_total==1), Frontier gpus-mode, and CPU jobs are
        # IMMUNE and keep their existing emission byte-identically.
        gres_multi_gpu = gpus_total >= 2 and gpu_alloc_mode == "gres"

        # For GPU jobs: set tasks=1 (1 task per GPU, SLURM executor uses --ntasks-per-gpu)
        # For non-GPU jobs: set tasks=<actual MPI rank count>
        if gres_multi_gpu:
            block += f"\n        tasks={gpus_total},"  # one task per requested GPU
        elif gpus_total > 0:
            block += "\n        tasks=1,"  # 1:1 GPU-to-task mapping
        else:
            block += f"\n        tasks={tasks},"  # Use actual task count

        block += f"""
        cpus_per_task={cpus_per_task},
        mem_mb={mem_mb},
        nodes={sim_nodes}"""

        # Emit mpi=True for non-GPU MPI jobs AND for gres-mode multi-GPU jobs.
        # For gres-multi-GPU it drives the executor's --ntasks=N path (see block
        # comment above); for non-GPU MPI it has its historical meaning. Frontier
        # gpus-mode and single-GPU gres never set it.
        if (mpi and gpus_total == 0) or gres_multi_gpu:
            block += ",\n        mpi=True"

        if gres_multi_gpu:
            # tasks_per_gpu=0 suppresses --ntasks-per-gpu in the executor's gpu_job
            # branch so the mpi/--ntasks path is the sole task-count driver.
            block += ",\n        tasks_per_gpu=0"

        if gpus_total > 0:
            if gpu_alloc_mode == "gpus":
                block += f",\n        gpu={gpus_total}"
                if gpu_hardware:
                    block += f',\n        gpu_model="{gpu_hardware}"'
            else:
                if gpu_hardware:
                    block += f',\n        gres="gpu:{gpu_hardware}:{gpus_per_node}"'
                else:
                    block += f',\n        gres="gpu:{gpus_per_node}"'
        return block

    @staticmethod
    def _calculate_nodes_for_gpus(total_gpus: int, gpus_per_node: int) -> int:
        if total_gpus <= 0:
            return 1
        return max(1, math.ceil(total_gpus / gpus_per_node))

    def _get_report_cfg_static_backend(self) -> Literal["matplotlib", "plotly"]:
        # Post-F2 (R1): cfg_analysis.report is required by analysis_config
        # Pydantic schema; sourcing static_backend inline removes the need
        # for the legacy peer-file path resolution.
        return self.cfg_analysis.report.interactive.static_backend

    def _make_rule_emission_context(self, *, static_backend: Literal["matplotlib", "plotly"]) -> RuleEmissionContext:
        """Build the shared per-Snakefile-emission context the new
        module-level rule helpers (_emit_plot_rule, _emit_render_report_rule,
        _emit_rule_all) consume. Post-F2: all renderer + runner rules read
        their report cfg inline from cfg_analysis.report, so a single
        config-args string suffices.
        """
        log_dir_rel = str(self.analysis_paths.analysis_log_directory.relative_to(self.analysis_paths.analysis_dir))
        return RuleEmissionContext(
            python_executable=self.python_executable,
            log_dir_rel=log_dir_rel,
            conda_env_path=str(self._get_conda_env_path()),
            config_args_str=self._get_config_args(),
            is_sensitivity=bool(getattr(self.cfg_analysis, "toggle_sensitivity_analysis", False)),
            static_backend=static_backend,
        )

    def _build_plot_rule_block_system_overview(
        self,
        input_flag: str = "_status/e_consolidate_complete.flag",
        *,
        ctx: RuleEmissionContext | None = None,
    ) -> str:
        """Generate the Snakemake rule for the 2-panel system-overview plot.

        Left panel is the SWMM model elements view (R5); right panel is the
        DEM elevation raster. Combined into one figure per iteration-4
        feedback from the Phase 2 STOP gate.

        ``input_flag`` defaults to the regular multisim consolidation flag
        (`e_consolidate_complete`); the sensitivity master Snakefile passes
        `f_consolidate_master_complete.flag` instead.
        """
        import os as _os

        if ctx is None:
            ctx = self._make_rule_emission_context(static_backend=self._get_report_cfg_static_backend())

        analysis_dir = self.analysis.analysis_paths.analysis_dir
        analysis_root = str(analysis_dir.resolve())
        cfg_ana = self.analysis.cfg_analysis
        if getattr(cfg_ana, "toggle_sensitivity_analysis", False):
            subs = self.analysis.sensitivity.sub_analyses
            first_sub = subs[next(iter(subs))]
            repr_scen_paths = first_sub._retrieve_sim_runs(0)._scenario.scen_paths
        else:
            repr_scen_paths = self.analysis._retrieve_sim_runs(0)._scenario.scen_paths
        source_paths: list[dict] = [
            {
                "path": _os.path.relpath(str(self.system.sys_paths.dem_processed.resolve()), analysis_root),
                "variables": [],
            },
            {
                "path": _os.path.relpath(str(Path(repr_scen_paths.swmm_hydro_inp).resolve()), analysis_root),
                "variables": ["[SUBCATCHMENTS]", "[JUNCTIONS]", "[OUTFALLS]"],
            },
            {
                "path": _os.path.relpath(str(Path(repr_scen_paths.swmm_hydraulics_inp).resolve()), analysis_root),
                "variables": ["[CONDUITS]", "[JUNCTIONS]", "[POLYGONS]"],
            },
        ]
        if cfg_ana.toggle_storm_tide_boundary and cfg_ana.storm_tide_boundary_line_gis:
            source_paths.append(
                {
                    "path": _os.path.relpath(str(Path(cfg_ana.storm_tide_boundary_line_gis).resolve()), analysis_root),
                    "variables": [],
                }
            )

        spec = RuleSpec(
            rule_name="plot_system_overview",
            renderer_module="system_overview",
            input_flags=(input_flag,),
            output_path_template="plots/system_overview__OUTPUT_EXT__",
            source_paths=tuple(source_paths),
            wildcards=(),
            extra_cli_flags=(),
            extra_params=(),
            report_kwargs={
                "caption": "report/captions/system_map.rst",
                "category": "System Information",
                "labels": '{"figure": "System map"}',
            },
            resources_yaml="mem_mb=2000, time_min=10",
            log_path_template="logs/plots/system_overview.log",
        )
        return _emit_plot_rule(spec, ctx)

    def _build_plot_rule_block_disk_utilization(
        self,
        input_flag: str = "_status/e_consolidate_complete.flag",
        *,
        ctx: RuleEmissionContext | None = None,
    ) -> str:
        """Emit the Snakemake rule for the Disk Utilization sidebar card.

        Reads the analysis-level `_status/_du.json` written by Phase 1's
        analysis-scope consolidate path; renders a compact HTML table.
        """
        if ctx is None:
            ctx = self._make_rule_emission_context(
                static_backend=self._get_report_cfg_static_backend()
            )

        spec = RuleSpec(
            rule_name="plot_disk_utilization",
            renderer_module="disk_utilization",
            input_flags=(input_flag,),
            output_path_template="plots/disk_utilization__OUTPUT_EXT__",
            source_paths=(
                {
                    "path": "_status/_du.json",
                    "variables": ["disk_utilization_bytes", "sub_path_breakdown"],
                },
            ),
            wildcards=(),
            extra_cli_flags=(),
            extra_params=(),
            report_kwargs={
                "caption": "report/captions/disk_utilization.rst",
                "category": "System Information",
                "labels": '{"figure": "Disk Utilization"}',
            },
            resources_yaml="mem_mb=1000, time_min=5",
            log_path_template="logs/plots/disk_utilization.log",
        )
        return _emit_plot_rule(spec, ctx)

    def _build_process_rule_block(
        self,
        model_type: str,
        *,
        which_arg: str,
        config_args: str,
        log_dir_str: str,
        conda_env_path: str,
        process_resources: str,
        compression_level: int,
        override_clear_raw: ClearRawValue | None,
    ) -> str:
        """Emit a single ``rule process_{model_type}`` block.

        Extracted from ``generate_snakefile_content`` so the same template
        is reused by the reprocess generator
        (``reprocess_snakefile_generator.generate_reprocess_snakefile``).
        Clearing is driven by ``cfg_analysis.clear_raw`` + the
        ``--override-clear-raw`` runtime override (this method's
        ``override_clear_raw`` parameter); force-rerun is handled by
        ``--override-force-rerun`` via login-side flag pre-deletion (per
        cleanup-rerun-delete-redesign Phase 4).
        """
        override_clear_raw_arg = (
            f"--override-clear-raw '{json.dumps(override_clear_raw)}' " if override_clear_raw is not None else ""
        )
        return f'''
rule process_{model_type}:
    input: "_status/c_run_{model_type}_evt-{{event_id}}_complete.flag"
    output: "_status/d_process_{model_type}_evt-{{event_id}}_complete.flag"
    log: "{log_dir_str}/sims/process_{model_type}_evt-{{event_id}}.log"
    group: "process_evt_{{event_id}}"
    conda: "{conda_env_path}"
    params:
        event_iloc=lambda wildcards: ILOC_BY_EVENT_ID[wildcards.event_id],
    resources:
{process_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.process_timeseries_runner \\
            --event-iloc {{params.event_iloc}} \\
            {config_args} \\
            --model-type {model_type} \\
            --which {which_arg} \\
            {override_clear_raw_arg}\\
            --compression-level {compression_level} \\
            --flag-output {{output}} \\
            --rule-name process_{model_type} \\
            --event-id {{wildcards.event_id}} \\
            > {{log}} 2>&1
        """
'''

    def _build_consolidate_rule_block(
        self,
        *,
        consolidate_input_str: str,
        which: str,
        config_args: str,
        log_dir_str: str,
        conda_env_path: str,
        consolidate_resources: str,
        compression_level: int,
        allow_incomplete: bool = False,
    ) -> str:
        """Emit the ``rule consolidate`` block.

        Extracted from ``generate_snakefile_content`` so the same template
        is reused by the reprocess generator
        (``reprocess_snakefile_generator.generate_reprocess_snakefile``)
        which can supply a different ``consolidate_input_str`` (referencing
        existing ``c_run_*`` sim flags directly when reprocess starts at
        ``consolidate`` and the process stage is skipped).

        ``allow_incomplete`` (default False) opts into the consolidate runner's
        ``--allow-incomplete`` mode, which demotes the runner's
        ``all_sims_run`` hard fail to a warning so reprocess against a
        partially-complete analysis dir can proceed against the
        Snakefile-DAG-scoped subset of completed scenarios. The canonical
        workflow path (``generate_snakefile_content``) leaves this False so
        unexpected sim absence still fails fast.

        Force-rerun is handled by ``--override-force-rerun`` via login-side
        flag pre-deletion (per cleanup-rerun-delete-redesign Phase 4).
        """
        return f'''
rule consolidate:
    input: {consolidate_input_str}
    output: "_status/e_consolidate_complete.flag"
    log: "{log_dir_str}/consolidate.log"
    conda: "{conda_env_path}"
    resources:
{consolidate_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {config_args} \\
            --compression-level {compression_level} \\
            {"--allow-incomplete " if allow_incomplete else ""}\\
            --which {which} \\
            --flag-output {{output}} \\
            --rule-name consolidate \\
            > {{log}} 2>&1
        """
'''

    def _build_consolidate_scenario_rule_block(
        self,
        *,
        enabled_models: list[str],
        config_args: str,
        log_dir_str: str,
        conda_env_path: str,
        consolidate_scenario_resources: str,
        compression_level: int,
    ) -> str:
        """Emit a single ``rule consolidate_scenario`` block (wildcarded on event_id).

        Fans-in on every enabled model-type's ``d_process_{model_type}_evt-{event_id}_complete.flag``
        for a given event_id and runs ``consolidate_workflow --event-id {event_id}`` which writes
        the per-scenario DU sentinel at ``{scenario_dir}/_status/_du.json`` via
        ``du_sentinels.compute_and_write_scope_sentinel`` (compare-and-write semantics; mtime
        preserved when payload bytes are unchanged). The rule joins the existing
        ``process_evt_{event_id}`` Snakemake group (see ``_build_process_rule_block:1116``) so it
        co-schedules into the per-event SLURM allocation rather than submitting a separate sbatch.

        The output declaration ``_status/f_consolidate_scenario_evt-{event_id}_complete.flag``
        matches the existing letter-prefixed flag convention (``c_run_*``, ``d_process_*``,
        ``e_consolidate_complete``). The DU sentinel itself is a secondary output so consumer
        rules that declare it as ``input:`` get Snakemake-tracked mtime semantics.
        """
        input_flags = ", ".join(
            f'"_status/d_process_{model_type}_evt-{{event_id}}_complete.flag"'
            for model_type in enabled_models
        )
        return f'''
rule consolidate_scenario:
    input: {input_flags}
    output:
        flag="_status/f_consolidate_scenario_evt-{{event_id}}_complete.flag",
        du_sentinel="sims/{{event_id}}/_status/_du.json",
    log: "{log_dir_str}/sims/consolidate_scenario_evt-{{event_id}}.log"
    group: "process_evt_{{event_id}}"
    conda: "{conda_env_path}"
    resources:
{consolidate_scenario_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {config_args} \\
            --compression-level {compression_level} \\
            --flag-output {{output.flag}} \\
            --rule-name consolidate_scenario \\
            --event-id {{wildcards.event_id}} \\
            > {{log}} 2>&1
        """
'''

    def generate_snakefile_content(
        self,
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: str = "TRITON",
        override_clear_raw: ClearRawValue | None = None,
        compression_level: int = 5,
        pickup_where_leftoff: bool = False,
        report_formats: list[str] | None = None,
        alive_by_token: dict[str, str] | None = None,
    ) -> str:
        """
        Generate Snakefile content with separate rules for prep, simulation, and processing.

        This creates a five-phase workflow:
        1. Setup: System inputs processing and compilation
        2. Scenario preparation: SWMM model generation (lightweight, 1 CPU)
        3. Simulation execution: TRITON-SWMM runs (resource-intensive, GPUs/CPUs)
        4. Output processing: Timeseries extraction and compression (I/O bound, 1-2 CPUs)
        5. Consolidation: Analysis-level output aggregation

        Parameters
        ----------
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings) in Phase 1
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM in Phase 1
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, each simulation will prepare its scenario before running
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after each simulation
        which : str
            Which outputs to process: "TRITON", "SWMM", or "both"
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw`` threaded into the
            emitted ``--override-clear-raw <json>`` rule-shell arg. ``None``
            reads the YAML at runner-time.
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint

        Returns
        -------
        str
            Complete Snakefile content as a string
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        # Emit report templates (CSS, captions, Jinja2 workflow description) into
        # {analysis_dir}/report/ so Snakemake's --report engine can resolve the
        # caption= / report: paths inside generated rules at report-render time.
        _emit_report_artifacts(self.analysis_paths.analysis_dir)

        n_sims = len(self.analysis.df_sims)
        event_ids = [
            compute_event_id_slug(self.analysis._retrieve_weather_indexer_using_integer_index(i)) for i in range(n_sims)
        ]
        iloc_by_event_id = {event_ids[i]: i for i in range(n_sims)}
        hpc_time_min = self.cfg_analysis.hpc_time_min_per_sim or 30

        mpi_ranks = self.cfg_analysis.n_mpi_procs or 1
        omp_threads = self.cfg_analysis.n_omp_threads or 1
        n_gpus = self.cfg_analysis.n_gpus or 0
        cpus_per_sim = mpi_ranks * omp_threads

        # The SLURM executor maps the `tasks` resource to --ntasks (non-GPU) or
        # --ntasks-per-gpu (gres-GPU), and `threads` to --cpus-per-task only (a fallback
        # used when no cpus_per_task resource is set). `threads` never governs --ntasks.
        # Verified against snakemake-executor-plugin-slurm v2.0.3 submit_string.py:79-128.
        # snakemake_threads below drives the Snakemake scheduler's local concurrency
        # accounting and the --cpus-per-task fallback — not --ntasks.
        snakemake_threads = cpus_per_sim

        # Conservative estimate: 2GB per CPU (can be made configurable later)
        mem_mb_per_sim = self.cfg_analysis.mem_gb_per_cpu * cpus_per_sim * 1000
        n_nodes = self.cfg_analysis.n_nodes or 1
        gpus_per_node_config = self.cfg_analysis.hpc_gpus_per_node or 0
        gpu_alloc_mode = self.system.cfg_system.preferred_slurm_option_for_allocating_gpus or "gpus"

        # Get absolute path to conda environment file using helper
        conda_env_path = self._get_conda_env_path()
        config_args = self._get_config_args()
        skip_setup = not (process_system_level_inputs or compile_TRITON_SWMM)

        # Make log dirs
        analysis_dir = self.analysis_paths.analysis_dir
        log_dir = self.analysis_paths.analysis_log_directory
        (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "sims").mkdir(parents=True, exist_ok=True)

        if skip_setup:
            setup_shell = '''"""
        touch {output}
        """
        '''
        else:
            tritonswmm_model = self.system.cfg_system.toggle_tritonswmm_model
            setup_shell = f'''"""
        {self.python_executable} -m TRITON_SWMM_toolkit.setup_workflow \\
            {config_args} \\
            {"--process-system-inputs " if process_system_level_inputs else ""}\\
            {"--overwrite-system-inputs " if overwrite_system_inputs else ""}\\
            {"--compile-triton-swmm " if compile_TRITON_SWMM and tritonswmm_model else ""}\\
            {"--compile-triton-only " if compile_TRITON_SWMM and self.system.cfg_system.toggle_triton_model else ""}\\
            {"--compile-swmm " if compile_TRITON_SWMM and self.system.cfg_system.toggle_swmm_model else ""}\\
            {"--recompile-if-already-done " if recompile_if_already_done_successfully else ""}\\
            --flag-output {{output}} \\
            --rule-name setup \\
            > {{log}} 2>&1
        """'''

        # Build resource blocks using helper
        setup_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
            runtime_min=self.cfg_analysis.hpc_runtime_min_for_setup,
            mem_mb=self.cfg_analysis.hpc_mem_allocation_for_setup_mb,
            nodes=1,
            tasks=1,
            cpus_per_task=1,
        )

        # Scenario preparation: lightweight (1 CPU, minimal memory)
        prep_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
            runtime_min=30,
            mem_mb=self.cfg_analysis.mem_gb_per_cpu * 1000,
            nodes=1,
            tasks=1,
            cpus_per_task=1,
        )

        # Simulation: resource-intensive (multi-CPU, GPUs, high memory)
        sim_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_ensemble_partition,
            runtime_min=hpc_time_min,
            mem_mb=mem_mb_per_sim,
            nodes=n_nodes,
            tasks=mpi_ranks,
            cpus_per_task=omp_threads,
            gpus_total=n_gpus,
            gpus_per_node_config=gpus_per_node_config,
            gpu_hardware=self.system.cfg_system.gpu_hardware,
            gpu_alloc_mode=gpu_alloc_mode,
            mpi=(self.cfg_analysis.run_mode in ["hybrid", "mpi"]),
        )

        # Output processing: I/O bound (1-2 CPUs for compression)
        process_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
            runtime_min=120,
            mem_mb=self.cfg_analysis.hpc_mem_allocation_for_sim_output_processing_mb,
            nodes=1,
            tasks=1,
            cpus_per_task=2,  # Parallel compression
        )

        # Consolidation resources
        consolidate_resources = self._build_resource_block(
            partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
            runtime_min=30,
            mem_mb=self.cfg_analysis.hpc_mem_allocation_for_analysis_output_consolidation_mb,
            nodes=1,
            tasks=1,
            cpus_per_task=2,
        )

        log_dir_str = str(log_dir)
        analysis_id_str = str(self.cfg_analysis.analysis_id)
        # Auto-render — modeled as an explicit Snakemake rule (NOT an `onsuccess:`
        # hook). `onsuccess:` only fires when rules execute; on a fully up-to-date
        # workflow Snakemake exits with `Nothing to be done` and skips the hook,
        # so the report would never get rendered on resume runs. The render_report
        # rule (appended below) consumes plot outputs and emits
        # `analysis_report.{fmt}`; rule all lists those outputs as targets so the
        # DAG planner always considers them.
        # Backend-aware extension resolution. The per-rule emission already uses
        # _output_ext_for via __OUTPUT_EXT__ substitution; rule_all + render_report
        # inputs MUST match or Snakemake's DAG planner cannot resolve the wildcards.
        _ext = _resolve_rule_all_extensions(self._get_report_cfg_static_backend())

        _formats = report_formats if report_formats is not None else ["zip"]
        render_targets_in_rule_all = "".join(f',\n        "analysis_report.{fmt}"' for fmt in _formats)
        snakefile_content = f'''# Auto-generated by TRITONSWMM_analysis

import os
import glob
import subprocess
from datetime import datetime as _dt
from TRITON_SWMM_toolkit.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("TRITON_SWMM_toolkit")
except Exception:
    _toolkit_version = "unknown"

# Config dict consumed by report_templates/workflow_description.rst.j2
config["analysis_id"] = {analysis_id_str!r}
config["toolkit_version"] = _toolkit_version
config["n_sims"] = {n_sims}
config["is_sensitivity"] = False
config["report"] = {{"generated_at": _dt.now().isoformat(timespec="seconds")}}

report: "report/workflow_description.rst"

SIM_IDS = {event_ids!r}
ILOC_BY_EVENT_ID = {iloc_by_event_id!r}

rule all:
    input:
        "_status/e_consolidate_complete.flag",
        "scenario_status.csv",
        "workflow_summary.md",
        "plots/system_overview{_ext["system_overview"]}",
        expand("plots/per_sim/{{event_id}}/peak_flood_depth{_ext["per_sim_peak_flood_depth"]}", event_id=SIM_IDS),
        expand("plots/per_sim/{{event_id}}/conduit_flow{_ext["per_sim_conduit_flow"]}",     event_id=SIM_IDS),
        "plots/per_analysis/summary_table{_ext["per_analysis_summary"]}",
        "plots/appendix/scenario_status{_ext["scenario_status_appendix"]}",
        "plots/errors_and_warnings/validation_report{_ext["errors_and_warnings"]}",
        "plots/disk_utilization{_ext["disk_utilization"]}"{render_targets_in_rule_all},

# onsuccess: removed — `rule export_scenario_status` (added below) now produces
# scenario_status.csv and workflow_summary.md on the success path via the
# Snakemake DAG. The previous `onsuccess:` hook fired AFTER all rules
# completed, which is too late for the renderer rules that consume the CSV.

onerror:
    # Partial-run debugging fallback: when a rule earlier in the DAG fails,
    # the export rule never fires (its input flag is absent), so this hook
    # is the only path to a diagnostic CSV. Snakemake treats `onerror:` exit
    # codes as informational — the workflow has already failed.
    shell("""
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            {config_args} \\
            > {log_dir_str}/export_scenario_status.log 2>&1
    """)

rule setup:
    output: "_status/a_setup_complete.flag"
    log: "{log_dir_str}/setup.log"
    conda: "{conda_env_path}"
    resources:
{setup_resources}
    shell:
        {setup_shell}
'''

        # Add scenario preparation rule if requested
        if prepare_scenarios:
            snakefile_content += f'''
rule prepare_scenario:
    input: "_status/a_setup_complete.flag"
    output: "_status/b_prepare_evt-{{event_id}}_complete.flag"
    log: "{log_dir_str}/sims/prepare_evt-{{event_id}}.log"
    conda: "{conda_env_path}"
    params:
        event_iloc=lambda wildcards: ILOC_BY_EVENT_ID[wildcards.event_id],
    resources:
{prep_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.prepare_scenario_runner \\
            --event-iloc {{params.event_iloc}} \\
            {config_args} \\
            {"--overwrite-scenario-if-already-set-up " if overwrite_scenario_if_already_set_up else ""}\\
            {"--rerun-swmm-hydro " if rerun_swmm_hydro_if_outputs_exist else ""}\\
            --flag-output {{output}} \\
            --rule-name prepare_scenario \\
            --event-id {{wildcards.event_id}} \\
            > {{log}} 2>&1
        """
'''

        # Add simulation rules (separate rules per model type)
        sim_input = (
            "_status/b_prepare_evt-{event_id}_complete.flag" if prepare_scenarios else "_status/a_setup_complete.flag"
        )

        # Determine which model types are enabled
        enabled_models = []
        if self.system.cfg_system.toggle_triton_model:
            enabled_models.append("triton")
        if self.system.cfg_system.toggle_tritonswmm_model:
            enabled_models.append("tritonswmm")
        if self.system.cfg_system.toggle_swmm_model:
            enabled_models.append("swmm")

        if not enabled_models:
            raise ValueError(
                "No model types enabled! Enable at least one of: toggle_triton_model, toggle_tritonswmm_model, toggle_swmm_model"  # noqa: E501
            )

        # Generate separate simulation rule for each enabled model type
        for model_type in enabled_models:
            # For SWMM, use fixed CPU-only resources (no GPU, limited threads)
            if model_type == "swmm":
                swmm_cpus = self.cfg_analysis.n_omp_threads or 1
                swmm_resources = self._build_resource_block(
                    partition=self.cfg_analysis.hpc_ensemble_partition,
                    runtime_min=hpc_time_min,
                    mem_mb=self.cfg_analysis.mem_gb_per_cpu * swmm_cpus * 1000,
                    nodes=1,
                    tasks=1,
                    cpus_per_task=swmm_cpus,
                    gpus_total=0,  # SWMM has no GPU support
                    gpus_per_node_config=0,
                )
                model_resources = swmm_resources
                model_threads = swmm_cpus
            else:
                # TRITON and TRITON-SWMM use configured resources
                model_resources = sim_resources
                model_threads = snakemake_threads

            snakefile_content += f'''
rule run_{model_type}:
    input: "{sim_input}"
    output: "_status/c_run_{model_type}_evt-{{event_id}}_complete.flag"
    log: "{log_dir_str}/sims/{model_type}_evt-{{event_id}}.log"
    conda: "{conda_env_path}"
    threads: {model_threads}
    params:
        event_iloc=lambda wildcards: ILOC_BY_EVENT_ID[wildcards.event_id],
    resources:
{model_resources}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.run_simulation_runner \\
            --event-iloc {{params.event_iloc}} \\
            {config_args} \\
            --model-type {model_type} \\
            {"--pickup-where-leftoff " if pickup_where_leftoff else ""}\\
            --flag-output {{output}} \\
            --rule-name run_{model_type} \\
            --event-id {{wildcards.event_id}} \\
            > {{log}} 2>&1
        """
'''

            # v2 graceful-rerun: per alive (model_type, event_id) tuple matching
            # this model's rule_token prefix, append a concrete-output wait-rule
            # and a ``ruleorder`` directive that prefers it over the wildcard
            # ``rule run_{model_type}`` for that specific event_id. Snakemake
            # raises AmbiguousRuleException for two-rules-one-output without an
            # explicit ruleorder; the wildcard-vs-concrete output overlap IS
            # that ambiguity, and the ruleorder line is the documented
            # resolution (Snakemake docs § Handling Ambiguous Rules).
            # R-WAITCAP: cap the wait-rule poll at the waited-on sim's own
            # walltime+slack, bounded above by the optional override ceiling.
            wait_walltime_cap_min = min(
                _max_plausible_job_lifetime_min(self.cfg_analysis),
                self.cfg_analysis.hpc_max_wait_for_inflight_min,
            )
            _prefix = f"run_{model_type}_evt-"
            for rule_token in sorted(alive_by_token or {}):
                if not rule_token.startswith(_prefix):
                    continue
                event_id = rule_token[len(_prefix):]
                flag_output_path = f"_status/c_run_{model_type}_evt-{event_id}_complete.flag"
                run_rule_inputs = (
                    [f"_status/b_prepare_evt-{event_id}_complete.flag"]
                    if prepare_scenarios
                    else ["_status/a_setup_complete.flag"]
                )
                sanitized = _sanitize_rule_name(rule_token)
                snakefile_content += f"\nruleorder: wait_for_{sanitized} > run_{model_type}\n\n"
                snakefile_content += self._emit_wait_for_sim_rule_block(
                    rule_token=rule_token,
                    flag_output_path=flag_output_path,
                    run_rule_inputs=run_rule_inputs,
                    wait_walltime_cap_min=wait_walltime_cap_min,
                )

        # Add output processing rules (one per model type) if requested
        if process_timeseries:
            for model_type in enabled_models:
                # Determine --which flag based on model type
                if model_type == "triton":
                    which_arg = "TRITON"
                elif model_type == "tritonswmm":
                    which_arg = "both"
                elif model_type == "swmm":
                    which_arg = "SWMM"
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                snakefile_content += self._build_process_rule_block(
                    model_type,
                    which_arg=which_arg,
                    config_args=config_args,
                    log_dir_str=log_dir_str,
                    conda_env_path=str(conda_env_path),
                    process_resources=process_resources,
                    compression_level=compression_level,
                    override_clear_raw=override_clear_raw,
                )

            # Per-scenario consolidate rule: fans-in on all enabled model_types' process flags
            # for a given event_id and writes the per-scenario _du.json sentinel exactly once.
            # Resource block is intentionally lightweight; SLURM grouping absorbs the wall-clock
            # cost into the per-event process-rule allocation (see _build_process_rule_block).
            consolidate_scenario_resources = self._build_resource_block(
                partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=10,
                mem_mb=2048,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )
            snakefile_content += self._build_consolidate_scenario_rule_block(
                enabled_models=enabled_models,
                config_args=config_args,
                log_dir_str=log_dir_str,
                conda_env_path=str(conda_env_path),
                consolidate_scenario_resources=consolidate_scenario_resources,
                compression_level=compression_level,
            )

        # Consolidation rule depends on final output of each model type
        # Build list of all output flags from all enabled models
        consolidate_inputs = []
        if process_timeseries:
            # Per-scenario consolidate rules (one per event_id) fan-in on the N
            # per-model-type process flags AND write the per-scenario DU sentinel.
            # The analysis-level consolidate waits for every per-scenario consolidate
            # to finish — a K-input dependency instead of the prior N×K.
            consolidate_inputs.append(
                'expand("_status/f_consolidate_scenario_evt-{event_id}_complete.flag", event_id=SIM_IDS)'
            )
        else:
            # No per-scenario consolidate exists when process_timeseries is False;
            # fall back to depending directly on the per-(event, model_type) run flags.
            for model_type in enabled_models:
                flag_pattern = f"c_run_{model_type}_evt-{{event_id}}_complete.flag"
                consolidate_inputs.append(f'expand("_status/{flag_pattern}", event_id=SIM_IDS)')

        # Join all input patterns
        consolidate_input_str = " + ".join(consolidate_inputs)

        snakefile_content += self._build_consolidate_rule_block(
            consolidate_input_str=consolidate_input_str,
            which=which,
            config_args=config_args,
            log_dir_str=log_dir_str,
            conda_env_path=str(conda_env_path),
            consolidate_resources=consolidate_resources,
            compression_level=compression_level,
        )
        snakefile_content += self._build_plot_rule_block_system_overview()
        snakefile_content += self._build_plot_rule_block_per_sim()
        snakefile_content += self._build_plot_rule_block_per_analysis_summary()
        snakefile_content += self._build_plot_rule_block_scenario_status_appendix()
        snakefile_content += self._build_plot_rule_block_errors_and_warnings()
        snakefile_content += self._build_plot_rule_block_disk_utilization()
        snakefile_content += self._build_export_scenario_status_rule(
            input_flag="_status/e_consolidate_complete.flag",
        )

        # Render-report rule (replaces the broken onsuccess auto-render).
        # Wildcarded on `format` — fires once per `analysis_report.{fmt}` target
        # listed in `rule all`. Inputs mirror the plot outputs so Snakemake's
        # DAG planner re-fires the render whenever any plot is newer than the
        # existing report, and skips it when the report is current.
        snakefile_content += f'''
rule render_report:
    input:
        "plots/system_overview{_ext["system_overview"]}",
        expand("plots/per_sim/{{event_id}}/peak_flood_depth{_ext["per_sim_peak_flood_depth"]}", event_id=SIM_IDS),
        expand("plots/per_sim/{{event_id}}/conduit_flow{_ext["per_sim_conduit_flow"]}",     event_id=SIM_IDS),
        "plots/per_analysis/summary_table{_ext["per_analysis_summary"]}",
        "plots/appendix/scenario_status{_ext["scenario_status_appendix"]}",
        "plots/errors_and_warnings/validation_report{_ext["errors_and_warnings"]}",
        "plots/disk_utilization{_ext["disk_utilization"]}",
        "scenario_status.csv",
    output:
        "analysis_report.{{format}}"
    wildcard_constraints:
        format="zip|html"
    log: "{log_dir_str}/render_report_{{format}}.log"
    resources:
{
            self._build_resource_block(
                partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=30,
                mem_mb=self.cfg_analysis.mem_gb_per_cpu * 1000,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )
        }
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.render_report_runner \\
            {config_args} \\
            --format {{wildcards.format}} \\
            > {{log}} 2>&1
        """
'''
        return snakefile_content

    def _collect_per_analysis_summary_source_paths(self) -> list[dict]:
        """Return analysis-dir-relative .rpt + TRITON-log descriptors the renderer reads.

        Per Gotcha 5: dispatch on enabled model types — `swmm_hydraulics_rpt`
        for TRITON-SWMM coupled mode, `swmm_full_rpt_file` for SWMM-only;
        `log_run_tritonswmm` / `log_run_triton` for TRITON-side logs.

        Each returned dict has the schema ``{"path": str, "variables": list[str]}``
        — the variable list names which fields the renderer parses from each
        source (e.g., "Flow Routing Continuity error" from SWMM .rpt). Caption
        RSTs render the dict as a path bullet with variable sub-bullets, with a
        backward-compat shim for callers still returning ``list[str]``.

        Sensitivity-master detection: if the analysis is a sensitivity master,
        iterate per-sub-analysis scenarios so the master per_analysis_summary
        table has provenance for every sub-analysis's status counts (per
        Iteration 6 "show all sub-analyses" scope).
        """
        import os as _os

        analysis_dir = self.analysis.analysis_paths.analysis_dir
        analysis_root = str(Path(analysis_dir).resolve())
        enabled = self.analysis._get_enabled_model_types()
        sources: list[dict] = []
        # Sensitivity-master scope: iterate every sub-analysis's scenarios.
        is_sensitivity_master = (
            getattr(self.analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
            and getattr(self.analysis, "sensitivity", None) is not None
        )
        if is_sensitivity_master:
            scenario_objs = []
            for sub in self.analysis.sensitivity.sub_analyses.values():
                for event_iloc in sub.df_sims.index:
                    try:
                        scenario_objs.append(sub._retrieve_sim_run_processing_object(event_iloc).scen_paths)
                    except Exception:
                        continue
        else:
            scenario_objs = []
            for event_iloc in self.analysis.df_sims.index:
                scenario_objs.append(self.analysis._retrieve_sim_run_processing_object(event_iloc).scen_paths)
        for scen_paths in scenario_objs:
            if "tritonswmm" in enabled and scen_paths.swmm_hydraulics_rpt:
                sources.append(
                    {
                        "path": _os.path.relpath(
                            str(Path(scen_paths.swmm_hydraulics_rpt).resolve()),
                            analysis_root,
                        ),
                        "variables": ["Flow Routing Continuity error (%)"],
                    }
                )
            elif "swmm" in enabled and scen_paths.swmm_full_rpt_file:
                sources.append(
                    {
                        "path": _os.path.relpath(
                            str(Path(scen_paths.swmm_full_rpt_file).resolve()),
                            analysis_root,
                        ),
                        "variables": ["Flow Routing Continuity error (%)"],
                    }
                )
            # Per-model-type model-state JSON logs (sim_folder/log_{mt}.json) are
            # what the renderer's _is_scenario_successful / _is_scenario_pending
            # actually read for status counts — NOT the simulation execution
            # logs (log_run_*) that the renderer never opens. Enumerate one entry
            # per enabled model type per scenario.
            sim_folder = getattr(scen_paths, "sim_folder", None)
            if sim_folder is not None:
                for mt in enabled:
                    log_file = Path(sim_folder) / f"log_{mt}.json"
                    if log_file.exists():
                        sources.append(
                            {
                                "path": _os.path.relpath(str(log_file.resolve()), analysis_root),
                                "variables": [
                                    f"model_run_completed[{mt}] (status flag for n_successful / n_pending counts)",
                                ],
                            }
                        )
        return sources

    def _build_plot_rule_block_per_analysis_summary(
        self,
        input_flag: str = "_status/e_consolidate_complete.flag",
        *,
        ctx: RuleEmissionContext | None = None,
    ) -> str:
        """Generate the Snakemake rule for the per-analysis summary table (R7).

        Produces `plots/per_analysis/summary_table.html` — a Tabulator data
        grid (n sims, expected total, n successful/pending/failed, enabled
        model types, sensitivity-analysis mode). HTML emit replaced the prior
        matplotlib SVG path at Phase 6.

        ``input_flag`` defaults to the regular multisim consolidation flag
        (`e_consolidate_complete`); the sensitivity master Snakefile passes
        `f_consolidate_master_complete.flag` instead.
        """
        if ctx is None:
            ctx = self._make_rule_emission_context(static_backend=self._get_report_cfg_static_backend())
        source_paths = self._collect_per_analysis_summary_source_paths()
        spec = RuleSpec(
            rule_name="plot_per_analysis_summary_table",
            renderer_module="per_analysis_summary",
            input_flags=(input_flag,),
            output_path_template="plots/per_analysis/summary_table__OUTPUT_EXT__",
            source_paths=tuple(source_paths),
            wildcards=(),
            extra_cli_flags=(),
            extra_params=(),
            report_kwargs={
                "caption": "report/captions/per_analysis_summary_table.rst",
                "category": "Workflow Status",
                "subcategory": "Workflow Health Summary",
                "labels": '{"figure": "Summary table"}',
            },
            resources_yaml="mem_mb=2000, time_min=5",
            log_path_template="logs/plots/per_analysis_summary_table.log",
        )
        return _emit_plot_rule(spec, ctx)

    def _build_plot_rule_block_scenario_status_appendix(
        self,
        input_flag: str = "_status/e_consolidate_complete.flag",
        *,
        ctx: RuleEmissionContext | None = None,
    ) -> str:
        """Generate the Snakemake rule for the scenario_status.csv Appendix table.

        Iter 8 agenda item 3: produces `plots/appendix/scenario_status.html` —
        an inline-styled HTML table rendered from `analysis_dir / scenario_status.csv`
        (written by the `export_scenario_status` Snakemake rule on the success
        path; the `onerror:` hook is retained as a partial-run debugging
        fallback). Sidebar category is "Appendix"; the comparator-fallback in
        the category-order post-process places it after all known categories
        alphabetically.

        ``input_flag`` defaults to the regular multisim consolidation flag;
        the sensitivity master Snakefile passes the master flag instead.
        """
        import os as _os

        if ctx is None:
            ctx = self._make_rule_emission_context(static_backend=self._get_report_cfg_static_backend())
        analysis_dir = self.analysis.analysis_paths.analysis_dir
        analysis_root = str(analysis_dir.resolve())
        csv_rel = _os.path.relpath(str((analysis_dir / "scenario_status.csv").resolve()), analysis_root)
        source_paths = [
            {
                "path": csv_rel,
                "variables": ["event_id", "model_type", "status", "runtime_s", "continuity_error_pct", "notes"],
            }
        ]
        spec = RuleSpec(
            rule_name="plot_scenario_status_appendix",
            renderer_module="scenario_status_appendix",
            input_flags=(input_flag,),
            output_path_template="plots/appendix/scenario_status__OUTPUT_EXT__",
            source_paths=tuple(source_paths),
            wildcards=(),
            extra_cli_flags=(),
            extra_params=(),
            report_kwargs={
                "caption": "report/captions/scenario_status_appendix.rst",
                "category": "Appendix",
                "subcategory": "Scenario Status",
                "labels": '{"figure": "Per-scenario status table"}',
            },
            resources_yaml="mem_mb=1000, time_min=5",
            log_path_template="logs/plots/scenario_status_appendix.log",
            additional_inputs=("scenario_status.csv",),
        )
        return _emit_plot_rule(spec, ctx)

    def _build_plot_rule_block_errors_and_warnings(
        self,
        input_flag: str = "_status/e_consolidate_complete.flag",
        *,
        ctx: RuleEmissionContext | None = None,
    ) -> str:
        """Generate the Snakemake rule for the Errors and Warnings validation report.

        Iter 9 agenda: produces `plots/errors_and_warnings/validation_report.html`
        — calls `analysis_validation.validate_analysis(analysis)` and renders
        a structured pass/fail report organized into 4 sections (system-level
        checks; aggregate per-scenario; granular per-scenario failures;
        resource-utilization mismatches). Replaces the placeholder injection
        for "Errors and Warnings" added in Subiteration 8.1.

        ``input_flag`` defaults to the regular multisim consolidation flag;
        the sensitivity master Snakefile passes the master flag instead.
        """
        import os as _os

        if ctx is None:
            ctx = self._make_rule_emission_context(static_backend=self._get_report_cfg_static_backend())
        analysis_dir = self.analysis.analysis_paths.analysis_dir
        analysis_root = str(analysis_dir.resolve())
        csv_rel = _os.path.relpath(str((analysis_dir / "scenario_status.csv").resolve()), analysis_root)
        source_paths = [
            {
                "path": csv_rel,
                "variables": [
                    "scenario_setup",
                    "run_completed",
                    "actual_nTasks",
                    "actual_omp_threads",
                    "actual_total_gpus",
                    "actual_gpu_backend",
                ],
            },
            {
                "path": "sims/<event_id>/log_<model_type>.json",
                "variables": ["simulation_completed (per scenario × model_type)"],
            },
            {
                "path": "../system_log.json",
                "variables": [
                    "compilation_successful",
                    "compilation_triton_only_successful",
                    "compilation_swmm_successful",
                ],
            },
        ]
        spec = RuleSpec(
            rule_name="plot_errors_and_warnings",
            renderer_module="errors_and_warnings",
            input_flags=(input_flag,),
            output_path_template="plots/errors_and_warnings/validation_report__OUTPUT_EXT__",
            source_paths=tuple(source_paths),
            wildcards=(),
            extra_cli_flags=(),
            extra_params=(),
            report_kwargs={
                "caption": "report/captions/errors_and_warnings.rst",
                "category": "Errors and Warnings",
                "subcategory": "Validation Report",
                "labels": '{"figure": "Validation report"}',
            },
            resources_yaml="mem_mb=1000, time_min=5",
            log_path_template="logs/plots/errors_and_warnings.log",
            additional_inputs=("scenario_status.csv",),
        )
        return _emit_plot_rule(spec, ctx)

    def _build_export_scenario_status_rule(
        self,
        input_flag: str = "_status/e_consolidate_complete.flag",
    ) -> str:
        """Emit a Snakemake rule that writes scenario_status.csv and workflow_summary.md.

        Replaces the previous onsuccess/onerror hook mechanism on the success
        path. The hook fired AFTER all rules — including the rules that read
        the CSV — completed, producing a false-positive 'CSV missing' warning
        in the rendered Errors and Warnings page. Promoting the export to a
        regular rule lets Snakemake's DAG scheduler order the CSV write before
        any rule that consumes it. The onerror: hook is retained separately as
        a partial-run debugging fallback.

        Listed under `localrules:` so it runs on the local executor — the rule
        is a sub-second CSV write against in-memory log JSON; dispatching it
        as a separate sbatch job under multi_sim_run_method='batch_job' would
        add ~10-30s of queue latency for zero work.

        ``input_flag`` defaults to the regular multisim consolidation flag;
        the sensitivity-master Snakefile passes the master flag instead.
        """
        log_dir = self.analysis.analysis_paths.analysis_dir / "logs"
        log_dir_str = str(log_dir)
        config_args = self._get_config_args()
        conda_env_path = str(self._get_conda_env_path())
        resources_block = self._build_resource_block(
            partition=self.cfg_analysis.hpc_setup_and_analysis_processing_partition,
            runtime_min=10,
            mem_mb=1000,
            nodes=1,
            tasks=1,
            cpus_per_task=1,
        )
        return f'''
localrules: export_scenario_status

rule export_scenario_status:
    input: "{input_flag}"
    output:
        csv = "scenario_status.csv",
        md  = "workflow_summary.md",
    log: "{log_dir_str}/export_scenario_status.log"
    conda: "{conda_env_path}"
    resources:
{resources_block}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            {config_args} \\
            > {{log}} 2>&1
        """
'''

    def _build_plot_rule_block_per_sim(self, *, ctx: RuleEmissionContext | None = None) -> str:
        """Generate two per-sim plot rules wildcarded over event_id (Phase 3, R6).

        `params.source_paths` for each rule is a function-based lookup
        (`_per_sim_*_sources`) that reads event-scoped paths at rule-schedule
        time (not Snakefile-emit time) — keeps the generated Snakefile
        readable even for large scenario counts. The wildcards' `event_id`
        is mapped to the integer `event_iloc` argument expected by the
        renderer CLI via the Snakefile-level `ILOC_BY_EVENT_ID` dict (set
        alongside `SIM_IDS` in `generate_snakefile_content`).

        System-level paths (DEM, watershed shapefile) used by the
        peak_flood_depth renderer are computed at emit time and baked into
        the closure as default kwargs — they are constant per analysis
        regardless of event wildcard.
        """
        import os as _os

        if ctx is None:
            ctx = self._make_rule_emission_context(static_backend=self._get_report_cfg_static_backend())
        analysis_root = str(self.analysis.analysis_paths.analysis_dir.resolve())
        dem_rel = _os.path.relpath(str(self.system.sys_paths.dem_processed.resolve()), analysis_root)
        watershed_path = self.system.cfg_system.watershed_gis_polygon
        watershed_rel = _os.path.relpath(str(Path(watershed_path).resolve()), analysis_root) if watershed_path else None
        rainfall_datavar = self.analysis.cfg_analysis.weather_time_series_spatial_mean_rainfall_datavar
        storm_tide_datavar = self.analysis.cfg_analysis.weather_time_series_storm_tide_datavar

        helpers = f"""
def _per_sim_flood_depth_sources(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "peak_flood_depth",
        wildcards.event_id,
        rainfall_datavar={rainfall_datavar!r},
        storm_tide_datavar={storm_tide_datavar!r},
        dem_rel_path={dem_rel!r},
        watershed_rel_path={watershed_rel!r},
    )

def _per_sim_conduit_flow_sources(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "conduit_flow",
        wildcards.event_id,
        rainfall_datavar={rainfall_datavar!r},
        storm_tide_datavar={storm_tide_datavar!r},
        dem_rel_path={dem_rel!r},
        watershed_rel_path={watershed_rel!r},
    )
"""

        flood_spec = RuleSpec(
            rule_name="plot_per_sim_peak_flood_depth",
            renderer_module="per_sim_peak_flood_depth",
            input_flags=("_status/e_consolidate_complete.flag",),
            output_path_template="plots/per_sim/{event_id}/peak_flood_depth__OUTPUT_EXT__",
            source_paths=(),
            wildcards=("event_id",),
            extra_cli_flags=("--event-iloc {params.event_iloc}",),
            extra_params=(("event_iloc", "lambda w: ILOC_BY_EVENT_ID[w.event_id]"),),
            report_kwargs={
                "caption": "report/captions/per_sim_peak_flood_depth.rst",
                "category": "Per Simulation Results",
                "labels": '{"event_id": "{event_id}", "figure": "Peak flood depth"}',
            },
            resources_yaml="mem_mb=4000, time_min=15",
            log_path_template="logs/plots/per_sim_peak_flood_depth_{event_id}.log",
            source_paths_fn_name="_per_sim_flood_depth_sources",
        )
        conduit_spec = RuleSpec(
            rule_name="plot_per_sim_conduit_flow",
            renderer_module="per_sim_conduit_flow",
            input_flags=("_status/e_consolidate_complete.flag",),
            output_path_template="plots/per_sim/{event_id}/conduit_flow__OUTPUT_EXT__",
            source_paths=(),
            wildcards=("event_id",),
            extra_cli_flags=("--event-iloc {params.event_iloc}",),
            extra_params=(("event_iloc", "lambda w: ILOC_BY_EVENT_ID[w.event_id]"),),
            report_kwargs={
                "caption": "report/captions/per_sim_conduit_flow.rst",
                "category": "Per Simulation Results",
                "labels": '{"event_id": "{event_id}", "figure": "Conduit flow"}',
            },
            resources_yaml="mem_mb=4000, time_min=15",
            log_path_template="logs/plots/per_sim_conduit_flow_{event_id}.log",
            source_paths_fn_name="_per_sim_conduit_flow_sources",
        )
        return helpers + _emit_plot_rule(flood_spec, ctx) + _emit_plot_rule(conduit_spec, ctx)

    def generate_snakemake_config(self, mode: Literal["local", "slurm", "single_job"]) -> dict:
        """
        Generate dynamic snakemake config based on analysis_config and system_config.

        Supports three execution modes:
        - local: Uses cores based on system capabilities
        - slurm: Uses 'executor: slurm' with job steps (many SLURM jobs)
        - single_job: Behaves like local execution but respects SLURM allocation
          (one SLURM job with many srun tasks inside)

        Parameters
        ----------
        mode : Literal["local", "slurm", "single_job"]
            Execution mode (local, slurm, or single_job)

        Returns
        -------
        dict
            Snakemake configuration dictionary
        """
        # Base config shared by all modes
        config = {
            "use-conda": False,
            "conda-frontend": "mamba",
            "printshellcmds": True,
            "rerun-incomplete": True,
            "keep-going": True,
            # Phase 1 (v2 post-death-recovery hardening): narrow to mtime only so a
            # resume after driver/orchestrator death cannot re-fire COMPLETED sims via
            # the `input` trigger (which would waste GPU-days). The mtime trigger still
            # covers every legitimate rerun the toolkit relies on — including the
            # per-sa_id fingerprint mechanism (sa-{id}_inputs.json bumps mtime on
            # content change; see Gotcha 17 and the comment near the fingerprint write
            # site) and missing-output reruns after Phase 2 DEAD-token reclaim. The
            # delete (:5045 → mtime,input) and reprocess (:4546/:7044 → mtime) paths
            # set --rerun-triggers explicitly on the CLI, which overrides this profile
            # default under configargparse precedence, so they are unaffected.
            "rerun-triggers": ["mtime"],
        }
        assert isinstance(self.cfg_analysis.local_cpu_cores_for_workflow, int), (
            "local_cpu_cores_for_workflow must be specified for local runs"
        )
        if mode == "local":
            config.update(
                {
                    "cores": self.cfg_analysis.local_cpu_cores_for_workflow,
                    "keep-going": True,
                }
            )
        elif mode == "single_job":
            # Single-job mode: cores and GPU resources set dynamically via CLI in SBATCH script
            # Don't set cores or resources here - will be passed via CLI args in SBATCH script
            config.update(
                {
                    "keep-going": True,  # Continue other sims if one fails
                    "latency-wait": 60,
                }
            )
        else:  # slurm
            # SLURM mode: support both modern executor and legacy cluster modes
            slurm_partition = self.cfg_analysis.hpc_ensemble_partition
            max_concurrent = self.cfg_analysis.hpc_max_simultaneous_sims
            assert isinstance(max_concurrent, int), (
                "hpc_max_simultaneous_sims is required for generate_snakemake_config"
            )
            # Modern executor mode: uses 'executor: slurm' with job steps
            config.update(
                {
                    "executor": "slurm",
                    "jobs": max_concurrent,
                    "latency-wait": 60,
                    "max-jobs-per-second": 5,
                    "max-status-checks-per-second": 10,
                    "default-resources": [
                        "nodes=1",
                        "mem_mb=2000",
                        "runtime=30",
                        f"slurm_partition={slurm_partition}",
                        f"slurm_account={self.cfg_analysis.hpc_account}",
                    ],
                    "slurm": {
                        "sbatch": {
                            "partition": "{resources.slurm_partition}",
                            "account": "{resources.slurm_account}",
                        }
                    },
                }
            )

        return config

    def write_snakemake_config(self, config: dict, mode: Literal["local", "slurm", "single_job"]) -> Path:
        """
        Write snakemake config to analysis directory.

        Parameters
        ----------
        config : dict
            Snakemake configuration dictionary
        mode : Literal["local", "slurm", "single_job"]
            Execution mode (local, slurm, or single_job)

        Returns
        -------
        Path
            Path to the written config directory
        """
        config_dir = self.analysis_paths.analysis_dir / ".snakemake_profile" / mode
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"

        with open(config_path, "w") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                sort_keys=False,
                width=float("inf"),  # Prevent YAML from breaking long lines
            )

        return config_dir

    def _generate_single_job_submission_script(
        self,
        snakefile_path: Path,
        config_dir: Path,
        override_hpc_total_nodes: int | None = None,
        extra_sbatch_args: list[str] | None = None,
    ) -> Path:
        """
        Generate SLURM batch script that runs Snakemake.

        For 1_job_many_srun_tasks mode, this requests exclusive access to nodes
        specified by hpc_total_nodes. Concurrency is determined dynamically from
        the SLURM allocation rather than being pre-calculated.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        config_dir : Path
            Path to the Snakemake profile config directory
        override_hpc_total_nodes : int | None
            If provided, overrides cfg_analysis.hpc_total_nodes for this submission
            without mutating the config object. Only valid for 1_job_many_srun_tasks mode.

        Returns
        -------
        Path
            Path to the generated batch script
        """
        import TRITON_SWMM_toolkit.utils as ut

        batch_log_path = self.analysis.analysis_paths.analysis_log_directory / "_slurm_logs"
        batch_log_path.mkdir(exist_ok=True, parents=True)
        # Get per-simulation resource requirements (without requiring totals)
        sim_resources = self.analysis._resource_manager._get_simulation_resource_requirements()

        # Get total nodes — use override if provided, otherwise fall back to config
        total_nodes = (
            override_hpc_total_nodes if override_hpc_total_nodes is not None else self.cfg_analysis.hpc_total_nodes
        )  # noqa: E501
        assert isinstance(total_nodes, int), "hpc_total_nodes required for 1_job_many_srun_tasks mode"

        # Get job duration
        job_time = self.cfg_analysis.hpc_total_job_duration_min
        assert isinstance(job_time, int), "hpc_total_job_duration_min required"

        assert self.analysis.in_slurm, "_generate_submission_script only makes sense to run in a SLURM environment."

        # Convert to HH:MM:SS format
        hours = job_time // 60
        minutes = job_time % 60
        estimated_time = f"{hours:02d}:{minutes:02d}:00"

        # additional_sbatch_args is computed below after gpu_directive is set
        # (it needs gpu_directive for the override-detection map).

        modules = self.analysis._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
        module_load_cmd = ""
        if modules:
            module_load_cmd = f"module load {modules}"

        # Conda initialization for non-interactive shells
        # In SLURM batch scripts, conda's shell integration is not automatically available
        # Strategy: After module load sets CONDA_EXE, use conda's shell hook to initialize
        conda_init_cmd = """
# Initialize conda for non-interactive shell (required in SLURM batch scripts)
# After 'module load miniforge3', CONDA_EXE is set by the module system
# Use conda's shell hook for robust initialization
if [ -n "${CONDA_EXE}" ]; then
    eval "$(${CONDA_EXE} shell.bash hook)"
elif [ -f "${CONDA_PREFIX}/../etc/profile.d/conda.sh" ]; then
    source "${CONDA_PREFIX}/../etc/profile.d/conda.sh"
else
    echo "ERROR: Cannot find conda initialization. CONDA_EXE and CONDA_PREFIX are both unset."
    echo "  CONDA_EXE=${CONDA_EXE:-<not set>}"
    echo "  CONDA_PREFIX=${CONDA_PREFIX:-<not set>}"
    exit 1
fi

conda activate triton_swmm_toolkit

# Fix for Frontier: conda activate in SLURM batch scripts doesn't add lib to LD_LIBRARY_PATH
# Explicitly add conda lib directory to ensure shared libraries (like libproj.so.25) are found
if [ -n "${CONDA_PREFIX}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    echo "Added ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH"
else
    echo "WARNING: CONDA_PREFIX not set after conda activate"
fi

# Fix for Frontier: GPFS /ccs/home compute node mounts do not support HDF5 POSIX byte-range
# locking. Without this, xr.open_dataset() on NetCDF-4 files fails with errno 524 on compute
# nodes. This env var is inherited by all srun child steps via SLURM's default --export=ALL.
export HDF5_USE_FILE_LOCKING=FALSE

# ===================================================================
# DIAGNOSTIC OUTPUT - Environment state after LD_LIBRARY_PATH fix
# ===================================================================
echo "=========================================="
echo "DIAGNOSTICS: Environment after LD_LIBRARY_PATH fix"
echo "=========================================="
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<not set>}"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-<not set>}"
echo ""
echo "LD_LIBRARY_PATH (line-by-line):"
echo "${LD_LIBRARY_PATH:-<not set>}" | tr ':' '\n' | sed 's/^/  /'
echo ""
echo "Python executable:"
which python
echo ""
echo "Checking for libproj.so.25 in conda env:"
if [ -n "${CONDA_PREFIX}" ]; then
    ls -la ${CONDA_PREFIX}/lib/libproj.so* 2>&1 || echo "  libproj.so* not found"
else
    echo "  CONDA_PREFIX not set, cannot check"
fi
echo ""
echo "Verification: Is conda lib in LD_LIBRARY_PATH?"
if [[ "${LD_LIBRARY_PATH}" == *"${CONDA_PREFIX}/lib"* ]]; then
    echo "  ✓ YES - ${CONDA_PREFIX}/lib is in LD_LIBRARY_PATH"
else
    echo "  ✗ NO - ${CONDA_PREFIX}/lib is NOT in LD_LIBRARY_PATH"
fi
echo "=========================================="
echo ""
"""

        # Build GPU directive if needed
        # Check if any simulation uses GPUs (handles sensitivity analysis)
        n_gpus_per_sim = sim_resources["n_gpus"]
        gpu_directive = ""
        gpu_calculation = ""
        gpu_cli_arg = ""

        if n_gpus_per_sim > 0:
            gpus_per_node = self.cfg_analysis.hpc_gpus_per_node
            assert isinstance(gpus_per_node, int), (
                "hpc_gpus_per_node required when using GPUs in 1_job_many_srun_tasks mode"
            )
            # --gres/--gpus-per-node are per-node, SLURM will multiply by --nodes automatically
            gpu_hardware = self.system.cfg_system.gpu_hardware
            if gpu_hardware:
                gpu_directive = f"#SBATCH --gres=gpu:{gpu_hardware}:{gpus_per_node}\n"
            else:
                gpu_directive = f"#SBATCH --gres=gpu:{gpus_per_node}\n"
            # Calculate total GPUs dynamically in bash script
            gpu_calculation = f"\n# Calculate total GPUs from SLURM allocation\nTOTAL_GPUS=$((SLURM_JOB_NUM_NODES * {gpus_per_node}))\n"  # noqa: E501
            gpu_cli_arg = " --resources gpu=$TOTAL_GPUS"

        # Combine cfg_analysis.additional_SBATCH_params (config-level baseline)
        # with extra_sbatch_args (runtime override). Runtime args are appended
        # AFTER config args so SLURM's last-directive-wins semantics let
        # runtime values shadow config values per flag without parser logic
        # for the actual override mechanism.
        #
        # Override detection (transparency side-channel): when extra_sbatch_args
        # contains a flag that matches a config-derived directive — either an
        # always-emitted directive (--partition / --account / --nodes / --gres /
        # --time / --output / --error / --job-name / --exclusive) emitted in
        # the script template below, or a flag in
        # cfg_analysis.additional_SBATCH_params — an INFO
        # "[extra_sbatch_args] OVERRIDE" message is printed naming the flag,
        # the origin of the original value, and the new runtime value. The
        # merge itself remains a plain append; the print is a side-channel
        # surfacing that lets users confirm what their runtime override
        # actually does. Emitted only when `extra_sbatch_args` is non-empty.
        combined_sbatch_params: list[str] = []
        if self.cfg_analysis.additional_SBATCH_params:
            combined_sbatch_params.extend(self.cfg_analysis.additional_SBATCH_params)
        if extra_sbatch_args:
            config_emitted_directives: dict[str, tuple[str, str]] = {
                "--job-name": ("hardcoded in script template", "triton_workflow"),
                "--partition": (
                    "cfg_analysis.hpc_ensemble_partition",
                    str(self.cfg_analysis.hpc_ensemble_partition),
                ),
                "--account": (
                    "cfg_analysis.hpc_account",
                    str(self.cfg_analysis.hpc_account),
                ),
                "--nodes": (
                    "cfg_analysis.hpc_total_nodes (or override_hpc_total_nodes runtime kwarg)",
                    str(total_nodes),
                ),
                "--exclusive": ("hardcoded in script template", ""),
                "--time": (
                    "cfg_analysis.hpc_total_job_duration_min",
                    estimated_time,
                ),
                "--output": (
                    "computed from analysis_paths.analysis_log_directory",
                    f"{batch_log_path}/workflow_*_%j.out",
                ),
                "--error": (
                    "computed from analysis_paths.analysis_log_directory",
                    f"{batch_log_path}/workflow_*_%j.out",
                ),
            }
            if gpu_directive:
                gres_value = gpu_directive.replace("#SBATCH ", "").strip()
                config_emitted_directives["--gres"] = (
                    "cfg_analysis.hpc_gpus_per_node + cfg_system.gpu_hardware",
                    gres_value,
                )
            for cfg_arg in self.cfg_analysis.additional_SBATCH_params or []:
                cfg_flag = cfg_arg.split("=", 1)[0].split(" ", 1)[0].strip()
                config_emitted_directives[cfg_flag] = (
                    "cfg_analysis.additional_SBATCH_params",
                    cfg_arg,
                )
            for runtime_arg in extra_sbatch_args:
                runtime_flag = runtime_arg.split("=", 1)[0].split(" ", 1)[0].strip()
                if runtime_flag in config_emitted_directives:
                    origin, original = config_emitted_directives[runtime_flag]
                    print(
                        f"[extra_sbatch_args] OVERRIDE: '{runtime_flag}' "
                        f"was '{original}' (from {origin}), "
                        f"now '{runtime_arg}' (from extra_sbatch_args runtime kwarg)",
                        flush=True,
                    )
            combined_sbatch_params.extend(extra_sbatch_args)
        additional_sbatch_args = ""
        if combined_sbatch_params:
            additional_sbatch_args = "#SBATCH "
            additional_sbatch_args += "\n#SBATCH ".join(combined_sbatch_params)

        script_content = f"""#!/bin/bash
#SBATCH --job-name=triton_workflow
#SBATCH --partition={self.cfg_analysis.hpc_ensemble_partition}
#SBATCH --account={self.cfg_analysis.hpc_account}
#SBATCH --nodes={total_nodes}
#SBATCH --exclusive
{gpu_directive}#SBATCH --time={estimated_time}
#SBATCH --output={str(batch_log_path)}/workflow_{ut.current_datetime_string(filepath_friendly=True)}_%j.out
#SBATCH --error={str(batch_log_path)}/workflow_{ut.current_datetime_string(filepath_friendly=True)}_%j.out
{additional_sbatch_args}

module purge

# Load required modules
{module_load_cmd}

{conda_init_cmd}

# Calculate total CPUs dynamically from SLURM allocation
if [ -z "$SLURM_CPUS_ON_NODE" ]; then
    echo "ERROR: SLURM_CPUS_ON_NODE not set. Cannot determine CPU allocation."
    exit 1
fi
TOTAL_CPUS=$((SLURM_CPUS_ON_NODE * SLURM_JOB_NUM_NODES))
{gpu_calculation}
# Run Snakemake with dynamic resource limits
${{CONDA_PREFIX}}/bin/python -m snakemake \\
    --profile {config_dir} --snakefile {snakefile_path} \\
    --cores $TOTAL_CPUS{gpu_cli_arg}
"""

        script_path = self.analysis_paths.analysis_dir / "run_workflow_1job.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        return script_path

    def run_snakemake_local(
        self,
        snakefile_path: Path,
        verbose: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """
        Run Snakemake workflow on local machine.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        verbose : bool
            If True, print progress messages
        dry_run : bool
            If True, perform a Snakemake dry run only

        Returns
        -------
        dict
            Status dictionary
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Running workflow locally with Snakemake",
                    flush=True,
                )
                if dry_run:
                    print(
                        "[Snakemake] DRY RUN",
                        flush=True,
                    )

            # Generate and write dynamic config
            config = self.generate_snakemake_config(mode="local")
            config_dir = self.write_snakemake_config(config, mode="local")

            if verbose:
                print(f"[Snakemake] Using dynamic config from: {config_dir}", flush=True)

            # Create log directory and file for Snakemake output
            logs_dir = self.analysis_paths.analysis_log_directory
            logs_dir.mkdir(parents=True, exist_ok=True)
            logfile_name = "snakemake_master_dry_run.log" if dry_run else "snakemake_master.log"
            snakemake_logfile = logs_dir / logfile_name

            if verbose:
                print(
                    f"[Snakemake] Snakemake output will be logged to: {snakemake_logfile}",
                    flush=True,
                )

            cmd_args = self._get_snakemake_base_cmd() + [
                "--profile",
                str(config_dir),
                "--snakefile",
                str(snakefile_path),
            ]

            # Explicitly pass --cores for multicore local runs
            # (ensures CLI-level cores setting when profile behavior varies)
            local_cores = self.cfg_analysis.local_cpu_cores_for_workflow
            assert isinstance(local_cores, int), "local_cpu_cores_for_workflow must be specified for local runs"
            if local_cores > 1:
                cmd_args.extend(["--cores", str(local_cores)])

            # Add dry-run flag last
            if dry_run:
                cmd_args.append("--dry-run")

            # Diagnostic flags from SnakemakeDiagnostics (Phase 1, synth-test-
            # isolation-and-runtime): --verbose is opt-in per call. The reason
            # intent folds into --verbose because snakemake 8+ removed the
            # standalone --reason flag and now auto-emits per-job rerun
            # reasons whenever --verbose is set.
            diag = getattr(self, "_active_snakemake_diagnostics", SnakemakeDiagnostics())
            if diag.emit_verbose:
                cmd_args.append("--verbose")
            if diag.log_path is not None:
                snakemake_logfile = Path(diag.log_path)
                snakemake_logfile.parent.mkdir(parents=True, exist_ok=True)

            # Pre-Snakemake guards: lock check + at-most-once reconciliation
            # (reconciliation skipped on dry runs; see helper docstring).
            self._pre_snakemake_invocation_guards(snakefile_path, dry_run=dry_run, verbose=verbose)

            with open(snakemake_logfile, "w") as log_f:
                result = subprocess.run(
                    cmd_args,
                    cwd=str(self.analysis_paths.analysis_dir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if verbose:
                cmd = " ".join(cmd_args)
                print(f"[Snakemake] command: \n     {cmd}")

            if result.returncode != 0:
                error_msg = f"Snakemake workflow failed.\nSee logs for {snakefile_path.parent}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "local",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": error_msg,
                    "snakemake_logfile": snakemake_logfile,
                }

            if verbose:
                print("[Snakemake] Workflow completed successfully", flush=True)

            return {
                "success": True,
                "mode": "local",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Workflow completed successfully",
                "snakemake_logfile": snakemake_logfile,
            }

        except Exception as e:
            error_msg = f"Failed to run Snakemake: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "local",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": error_msg,
                "snakemake_logfile": snakemake_logfile,
            }

    def _validate_single_job_dry_run(
        self,
        snakefile_path: Path,
        analysis: "TRITONSWMM_analysis",
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
    ) -> dict:
        """
        Perform dry-run validation for 1_job_many_srun_tasks mode.

        Computes expected resource allocations and validates the workflow DAG
        using the same CLI arguments that will be used in the SBATCH script.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        analysis : TRITONSWMM_analysis
            The analysis object (regular or master sensitivity analysis)
        verbose : bool
            If True, print progress messages
        override_hpc_total_nodes : int | None
            If provided, overrides cfg_analysis.hpc_total_nodes for the CPU budget
            calculation. Must match the value passed to _generate_single_job_submission_script.

        Returns
        -------
        dict
            Status dictionary with 'success' and 'mode' keys
        """
        # Compute expected resources to match SBATCH script (--cores $TOTAL_CPUS)
        hpc_cpus_per_node = getattr(analysis.cfg_analysis, "hpc_cpus_per_node", None)
        hpc_total_nodes = (
            override_hpc_total_nodes
            if override_hpc_total_nodes is not None
            else getattr(analysis.cfg_analysis, "hpc_total_nodes", None)
        )
        if not isinstance(hpc_cpus_per_node, int) or not isinstance(hpc_total_nodes, int):
            if verbose:
                print(
                    "[Snakemake] Skipping single-job dry-run validation: "
                    "hpc_cpus_per_node or hpc_total_nodes missing in config",
                    flush=True,
                )
            return {
                "success": True,
                "mode": "single_job",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Dry run skipped (missing hpc_cpus_per_node or hpc_total_nodes)",
            }

        expected_total_cpus = hpc_cpus_per_node * hpc_total_nodes

        # Temporarily align local dry-run cores with expected SLURM allocation.
        # This keeps run_snakemake_local config-driven while validating the DAG
        # under expected single-job CPU availability.
        original_local_cores = analysis.cfg_analysis.local_cpu_cores_for_workflow
        analysis.cfg_analysis.local_cpu_cores_for_workflow = expected_total_cpus
        try:
            dry_run_result = self.run_snakemake_local(
                snakefile_path=snakefile_path,
                verbose=verbose,
                dry_run=True,
            )
        finally:
            analysis.cfg_analysis.local_cpu_cores_for_workflow = original_local_cores

        if not dry_run_result.get("success"):
            raise RuntimeError("Dry run failed; workflow submission aborted.")

        # Override mode to indicate intended execution context
        dry_run_result["mode"] = "single_job"
        return dry_run_result

    # TODO - since we are unlikely to run models as detached processes, this and all calls to it can probably be deleted
    def _run_snakemake_slurm_detached(
        self,
        snakefile_path: Path,
        verbose: bool = True,
        wait_for_completion: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """
        Run Snakemake workflow on SLURM HPC system.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        verbose : bool
            If True, print progress messages
        wait_for_completion : bool
            If True, block and wait for workflow completion. If False (default),
            return immediately after submission (non-blocking).
        dry_run : bool
            If True, perform a Snakemake dry run only

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool - Did submission succeed?
            - mode: str - "slurm"
            - snakefile_path: Path - Path to Snakefile
            - job_id: str | None - Always None (job ID not extracted)
            - message: str - Status message
            - process: Popen - Process object
            - wait_for_completion: bool - Whether we waited
            - completed: bool - True only if wait_for_completion=True and job finished
            - completion_status: str | None - "success"/"failed" (only if waited)
            - snakemake_logfile: Path - Path to snakemake output log
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Running workflow on SLURM with Snakemake",
                    flush=True,
                )
                if dry_run:
                    print(
                        "[Snakemake] DRY RUN",
                        flush=True,
                    )

            # Generate and write dynamic config
            config = self.generate_snakemake_config(mode="slurm")
            config_dir = self.write_snakemake_config(config, mode="slurm")

            if verbose:
                print(f"[Snakemake] Using config from: {config_dir}", flush=True)

            # Create log directory and file for Snakemake output
            logs_dir = self.analysis_paths.analysis_log_directory
            logs_dir.mkdir(parents=True, exist_ok=True)
            logfile_name = "snakemake_master_dry_run.log" if dry_run else "snakemake_master.log"
            snakemake_logfile = logs_dir / logfile_name

            # Create SLURM efficiency report directory and set timestamped filename
            import TRITON_SWMM_toolkit.utils as ut

            efficiency_report_dir = logs_dir / "slurm_efficiency_report"
            efficiency_report_dir.mkdir(parents=True, exist_ok=True)
            efficiency_report_filename = (
                f"slurm_efficiency_report_{ut.current_datetime_string(filepath_friendly=True)}.csv"
            )
            efficiency_report_path = efficiency_report_dir / efficiency_report_filename

            if verbose:
                print(
                    f"[Snakemake] Snakemake output will be logged to: {snakemake_logfile}",
                    flush=True,
                )
                print(
                    f"[Snakemake] SLURM efficiency report will be written to: {efficiency_report_path}",
                    flush=True,
                )

            cmd_args = self._get_snakemake_base_cmd() + [
                "--profile",
                str(config_dir),
                "--snakefile",
                str(snakefile_path),
                "--executor",
                "slurm",
                "--printshellcmds",
                "--slurm-efficiency-report",
                "--slurm-efficiency-report-path",
                str(efficiency_report_path),
            ]
            if dry_run:
                cmd_args.append("--dry-run")
            if verbose:
                cmd_args.append("--verbose")

            # Diagnostic flags from SnakemakeDiagnostics. --verbose may
            # already be appended above when the function's `verbose` kwarg
            # is True; snakemake accepts duplicate flags idempotently. The
            # standalone --reason flag was removed in snakemake 8; the reason
            # intent folds into --verbose via SnakemakeDiagnostics.emit_verbose.
            diag = getattr(self, "_active_snakemake_diagnostics", SnakemakeDiagnostics())
            if diag.emit_verbose and not verbose:
                cmd_args.append("--verbose")
            if diag.log_path is not None:
                snakemake_logfile = Path(diag.log_path)
                snakemake_logfile.parent.mkdir(parents=True, exist_ok=True)

            with open(snakemake_logfile, "w") as log_f:
                proc = subprocess.Popen(
                    cmd_args,
                    cwd=str(self.analysis_paths.analysis_dir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            if not wait_for_completion:
                if verbose:
                    print(
                        f"[Snakemake] Workflow submitted to background (PID: {proc.pid})",
                        flush=True,
                    )
                    print(
                        f"[Snakemake] Monitor progress with: tail -f {snakemake_logfile}",
                        flush=True,
                    )
                return {
                    "success": True,
                    "mode": "slurm",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": "Workflow submitted to background",
                    "process": proc,
                    "wait_for_completion": False,
                    "completed": False,
                    "completion_status": None,
                    "snakemake_logfile": snakemake_logfile,
                }

            if verbose:
                print("[Snakemake] Waiting for workflow completion...", flush=True)
            proc.wait()
            success = proc.returncode == 0
            completion_status = "success" if success else "failed"

            if verbose:
                print(
                    f"[Snakemake] Workflow completed with status: {completion_status}",
                    flush=True,
                )
                print(
                    f"[Snakemake] Full output available in: {snakemake_logfile}",
                    flush=True,
                )

            return {
                "success": success,
                "mode": "slurm",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": f"Workflow completed with status: {completion_status}",
                "process": proc,
                "wait_for_completion": True,
                "completed": True,
                "completion_status": completion_status,
                "snakemake_logfile": snakemake_logfile,
            }

        except Exception as e:
            error_msg = f"Failed to submit workflow: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "slurm",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": error_msg,
                "process": None,
                "wait_for_completion": wait_for_completion,
                "completed": False,
                "completion_status": None,
                "snakemake_logfile": None,
            }

    def _validate_batch_job_dry_run(
        self,
        snakefile_path: Path,
        verbose: bool = True,
    ) -> dict:
        """
        Perform a dry-run validation for batch_job mode using the SLURM profile.

        This validates the Snakemake DAG/resources before submitting the
        orchestration SBATCH job.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        verbose : bool
            If True, print progress messages

        Returns
        -------
        dict
            Dry-run status dictionary
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Running batch_job dry-run validation",
                    flush=True,
                )

            config = self.generate_snakemake_config(mode="slurm")
            config_dir = self.write_snakemake_config(config, mode="slurm")

            logs_dir = self.analysis_paths.analysis_log_directory
            logs_dir.mkdir(parents=True, exist_ok=True)
            snakemake_logfile = logs_dir / "snakemake_master_dry_run.log"

            # Create SLURM efficiency report directory for dry run validation
            import TRITON_SWMM_toolkit.utils as ut

            efficiency_report_dir = logs_dir / "slurm_efficiency_report"
            efficiency_report_dir.mkdir(parents=True, exist_ok=True)
            efficiency_report_filename = (
                f"slurm_efficiency_report_dry_run_{ut.current_datetime_string(filepath_friendly=True)}.csv"
            )
            efficiency_report_path = efficiency_report_dir / efficiency_report_filename

            cmd_args = self._get_snakemake_base_cmd() + [
                "--profile",
                str(config_dir),
                "--snakefile",
                str(snakefile_path),
                "--executor",
                "slurm",
                "--printshellcmds",
                "--slurm-efficiency-report",
                "--slurm-efficiency-report-path",
                str(efficiency_report_path),
                "--dry-run",
            ]
            if verbose:
                cmd_args.append("--verbose")

            diag = getattr(self, "_active_snakemake_diagnostics", SnakemakeDiagnostics())
            if diag.emit_verbose and not verbose:
                cmd_args.append("--verbose")
            if diag.log_path is not None:
                snakemake_logfile = Path(diag.log_path)
                snakemake_logfile.parent.mkdir(parents=True, exist_ok=True)

            with open(snakemake_logfile, "w") as log_f:
                result = subprocess.run(
                    cmd_args,
                    cwd=str(self.analysis_paths.analysis_dir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

            if result.returncode != 0:
                error_msg = f"Snakemake batch_job dry run failed. See logs for {snakefile_path.parent}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "batch_job",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": error_msg,
                    "snakemake_logfile": snakemake_logfile,
                }

            if verbose:
                print("[Snakemake] Batch-job dry run completed successfully", flush=True)

            return {
                "success": True,
                "mode": "batch_job",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Batch-job dry run completed successfully",
                "snakemake_logfile": snakemake_logfile,
            }

        except Exception as e:
            error_msg = f"Failed to run batch-job dry run: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "batch_job",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": error_msg,
                "snakemake_logfile": None,
            }

    def _wait_for_slurm_job_completion(
        self,
        job_id: str,
        poll_interval: int = 1,
        timeout: int | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Wait for SLURM job to complete by polling job status.

        Uses squeue for active jobs and sacct for completed jobs.

        Parameters
        ----------
        job_id : str
            SLURM job ID to monitor
        poll_interval : int, default=1
            Seconds between status checks
        timeout : int | None, default=None
            Maximum seconds to wait (None = indefinite)
        verbose : bool, default=True
            Print status updates

        Returns
        -------
        dict
            Job completion info:
            - completed: bool - True if job finished successfully
            - state: str - SLURM job state (COMPLETED, FAILED, etc.)
            - exit_code: int | None - Job exit code
            - message: str - Human-readable status
        """
        import time

        start_time = time.time()
        last_state = None

        if verbose:
            print(f"[Snakemake] Waiting for SLURM job {job_id} to complete...", flush=True)

        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                msg = f"Job {job_id} timed out after {timeout}s"
                if verbose:
                    print(f"[Snakemake] ERROR: {msg}", flush=True)
                return {
                    "completed": False,
                    "state": "TIMEOUT",
                    "exit_code": None,
                    "message": msg,
                }

            # Query squeue for running/pending jobs
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                state = result.stdout.strip()

                # Print status update if changed
                if verbose and state != last_state:
                    elapsed = int(time.time() - start_time)
                    print(
                        f"[Snakemake] [{elapsed}s] Job {job_id}: {state}",
                        flush=True,
                    )
                    last_state = state

                if state in ["PENDING", "RUNNING", "CONFIGURING", "COMPLETING"]:
                    time.sleep(poll_interval)
                    continue

            # Job not in squeue - check sacct for completion
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-X", "-o", "State,ExitCode"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split()
                state = parts[0]
                exit_code_str = parts[1] if len(parts) > 1 else "0:0"
                exit_code = int(exit_code_str.split(":")[0])

                completed = state == "COMPLETED" and exit_code == 0

                if verbose:
                    elapsed = int(time.time() - start_time)
                    status = "✓" if completed else "✗"
                    print(
                        f"[Snakemake] [{elapsed}s] Job {job_id}: {state} {status}",
                        flush=True,
                    )

                return {
                    "completed": completed,
                    "state": state,
                    "exit_code": exit_code,
                    "message": f"Job {job_id} {state} (exit {exit_code})",
                }

            # Job not found yet - might be starting up
            time.sleep(poll_interval)

    def _submit_single_job_workflow(
        self,
        snakefile_path: Path,
        wait_for_completion: bool = False,
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
        extra_sbatch_args: list[str] | None = None,
    ) -> dict:
        """
        Submit workflow as a single SLURM batch job.

        This method generates a batch script that submits a single SLURM job
        which runs Snakemake inside the allocation using the single_job profile.
        Each simulation is then launched via srun within that allocation.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        wait_for_completion : bool, default=False
            If True, wait for job completion
        verbose : bool, default=True
            If True, print progress messages
        override_hpc_total_nodes : int | None
            If provided, overrides cfg_analysis.hpc_total_nodes for this submission
            without mutating the config object. Only valid for 1_job_many_srun_tasks mode.
        extra_sbatch_args : list[str] | None
            Optional list of additional SBATCH directive strings appended to the
            generated run_workflow_1job.sh script after every other source of
            #SBATCH directives — both the always-emitted directives derived from
            cfg_analysis fields (--partition, --account, --nodes, --gres, --time,
            --output, --error) and the directives in
            cfg_analysis.additional_SBATCH_params. Any flag in extra_sbatch_args
            that matches a flag emitted earlier in the script WILL OVERRIDE the
            config-derived value via SLURM's last-directive-wins parser
            semantics. When such an override is detected, an informational
            "[extra_sbatch_args] OVERRIDE: ..." message is printed naming the
            flag, the origin of the original value (e.g.
            cfg_analysis.hpc_ensemble_partition), and the new runtime value.

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool
            - mode: str ("single_job")
            - job_id: str | None
            - script_path: Path
            - message: str
            - completed: bool (only if wait_for_completion=True)
            - state: str (only if wait_for_completion=True)
            - exit_code: int | None (only if wait_for_completion=True)
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Preparing single-job workflow submission",
                    flush=True,
                )

            # Pre-Snakemake guards: lock check + at-most-once reconciliation
            self._pre_snakemake_invocation_guards(snakefile_path, dry_run=False, verbose=verbose)

            # Generate single_job profile
            config = self.generate_snakemake_config(mode="single_job")
            config_dir = self.write_snakemake_config(config, mode="single_job")

            # Generate submission script
            script_path = self._generate_single_job_submission_script(
                snakefile_path,
                config_dir,
                override_hpc_total_nodes=override_hpc_total_nodes,
                extra_sbatch_args=extra_sbatch_args,
            )

            if verbose:
                print(
                    f"[Snakemake] Generated submission script: {script_path}",
                    flush=True,
                )

            # Submit with sbatch
            if verbose:
                print(f"[Snakemake] Submitting with sbatch: {script_path}", flush=True)

            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(self.analysis_paths.analysis_dir),
            )

            # Parse job ID from sbatch output
            job_id = None
            if result.returncode == 0 and result.stdout:
                # sbatch output typically: "Submitted batch job 12345"
                parts = result.stdout.strip().split()
                if len(parts) >= 4 and parts[0] == "Submitted":
                    job_id = parts[-1]

            if result.returncode != 0:
                error_msg = f"sbatch submission failed: {result.stderr}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "single_job",
                    "job_id": None,
                    "script_path": script_path,
                    "message": error_msg,
                }

            if verbose:
                print(
                    f"[Snakemake] Single-job workflow submitted successfully (Job ID: {job_id})",
                    flush=True,
                )

            # Base result
            result_dict = {
                "success": True,
                "mode": "single_job",
                "job_id": job_id,
                "script_path": script_path,
                "message": f"Single-job workflow submitted (Job ID: {job_id})",
            }

            # E2: persist orchestrator identity so a live single-job driver is
            # detectable by the reprocess orchestration-liveness gate (Phase 2).
            # Mirrors the tmux path's persistence at ~workflow.py:4071-4076.
            if job_id:
                self.analysis.log.orchestrator_slurm_jobid.set(job_id)
                self.analysis.log.workflow_submission_mode.set("1_job_many_srun_tasks")
                self.analysis.log.workflow_submission_time.set(datetime.datetime.now().isoformat())
                self.analysis.log.workflow_submission_node.set(socket.gethostname())

            # Wait for completion if requested
            if wait_for_completion:
                if job_id:
                    completion_info = self._wait_for_slurm_job_completion(
                        job_id=job_id,
                        timeout=None,
                        verbose=verbose,
                    )

                    result_dict.update(completion_info)
                    result_dict["success"] = completion_info["completed"]
                else:
                    if verbose:
                        print(
                            "[Snakemake] ERROR: Failed to parse job ID for wait",
                            flush=True,
                        )
                    result_dict["success"] = False
                    result_dict["completed"] = False
                    result_dict["message"] = "Failed to parse job ID"

            return result_dict

        except Exception as e:
            error_msg = f"Failed to submit single-job workflow: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "single_job",
                "job_id": None,
                "script_path": None,
                "message": error_msg,
            }

    def _deprecated_submit_batch_job_workflow(
        self,
        snakefile_path: Path,
        wait_for_completion: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        DEPRECATED: Submit Snakemake workflow as an SLURM sbatch orchestration job.

        **WARNING**: This method runs Snakemake inside an sbatch job, which causes
        orphaned worker jobs when the orchestrator is canceled. This approach is
        deprecated in favor of tmux-based orchestration.

        This method is kept for backward compatibility but should not be used.
        The batch_job mode now uses _submit_tmux_workflow() instead.

        Parameters
        ----------
        snakefile_path : Path
            Path to the generated Snakefile
        wait_for_completion : bool, default=False
            If True, block until orchestration job completes
        verbose : bool, default=True
            Print progress messages

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool
            - mode: str ("batch_job")
            - job_id: str | None
            - script_path: Path | None
            - message: str
            - completed/state/exit_code when wait_for_completion=True
        """
        import warnings

        warnings.warn(
            "The sbatch orchestrator approach is deprecated due to orphaned job issues. "
            "This method should not be called directly. batch_job mode now uses tmux orchestration.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            if verbose:
                print(
                    "[Snakemake] Preparing batch_job orchestration submission",
                    flush=True,
                )

            # Build and write slurm profile used by the orchestration job
            config = self.generate_snakemake_config(mode="slurm")
            config_dir = self.write_snakemake_config(config, mode="slurm")

            # Long-duration walltime for orchestration job
            job_time = self.cfg_analysis.hpc_total_job_duration_min
            assert isinstance(job_time, int), "hpc_total_job_duration_min required for multi_sim_run_method='batch_job'"

            hours = job_time // 60
            minutes = job_time % 60
            estimated_time = f"{hours:02d}:{minutes:02d}:00"

            # Lightweight orchestration resources (single-core process)
            mem_mb = self.cfg_analysis.mem_gb_per_cpu * 1000
            orchestration_partition = (
                self.cfg_analysis.hpc_setup_and_analysis_processing_partition
                or self.cfg_analysis.hpc_ensemble_partition
            )

            if orchestration_partition is None:
                raise ValueError(
                    "Either hpc_setup_and_analysis_processing_partition or "
                    "hpc_ensemble_partition must be set for batch_job orchestration"
                )

            # Logs for sbatch script stdout/stderr
            import TRITON_SWMM_toolkit.utils as ut

            batch_log_path = self.analysis.analysis_paths.analysis_log_directory / "_slurm_logs"
            batch_log_path.mkdir(exist_ok=True, parents=True)

            additional_sbatch_args = ""
            if self.cfg_analysis.additional_SBATCH_params:
                additional_sbatch_args = "#SBATCH "
                additional_sbatch_args += "\n#SBATCH ".join(self.cfg_analysis.additional_SBATCH_params)

            modules = self.analysis._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
            module_load_cmd = ""
            if modules:
                module_load_cmd = f"module load {modules}"

            # Conda initialization for non-interactive SLURM shell
            conda_init_cmd = """
# Initialize conda for non-interactive shell
if [ -n "${CONDA_EXE}" ]; then
    eval "$(${CONDA_EXE} shell.bash hook)"
elif [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/../etc/profile.d/conda.sh" ]; then
    source "${CONDA_PREFIX}/../etc/profile.d/conda.sh"
else
    echo "ERROR: Cannot find conda initialization. CONDA_EXE and CONDA_PREFIX are both unset."
    exit 1
fi

conda activate triton_swmm_toolkit

# Ensure conda libs are discoverable (important on some HPC systems)
if [ -n "${CONDA_PREFIX}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# Diagnostics: confirm activation and Snakemake availability
echo "=========================================="
echo "DIAGNOSTICS: Conda activation + Snakemake"
echo "=========================================="
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<not set>}"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-<not set>}"
echo "Python (PATH): $(which python)"
echo "PATH (head):"
echo "${PATH}" | tr ':' '\n' | head -n 10 | sed 's/^/  /'
echo "=========================================="
"""

            account_directive = ""
            if self.cfg_analysis.hpc_account:
                account_directive = f"#SBATCH --account={self.cfg_analysis.hpc_account}\n"

            # Create SLURM efficiency report directory and set timestamped filename
            efficiency_report_dir = self.analysis.analysis_paths.analysis_log_directory / "slurm_efficiency_report"
            efficiency_report_dir.mkdir(parents=True, exist_ok=True)
            efficiency_report_filename = (
                f"slurm_efficiency_report_{ut.current_datetime_string(filepath_friendly=True)}.csv"
            )
            efficiency_report_path = efficiency_report_dir / efficiency_report_filename

            # The orchestration job runs snakemake; snakemake then submits worker jobs via executor=slurm
            script_content = f"""#!/bin/bash
#SBATCH --job-name={self.cfg_analysis.analysis_id}_orchestrator
#SBATCH --partition={orchestration_partition}
{account_directive}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem_mb}
#SBATCH --time={estimated_time}
#SBATCH --output={str(batch_log_path)}/workflow_batch_{ut.current_datetime_string(filepath_friendly=True)}_%j.out
#SBATCH --error={str(batch_log_path)}/workflow_batch_{ut.current_datetime_string(filepath_friendly=True)}_%j.out
{additional_sbatch_args}


module purge
{module_load_cmd}

{conda_init_cmd}

${{CONDA_PREFIX}}/bin/python -V
${{CONDA_PREFIX}}/bin/python -m snakemake --version

# Capture Snakemake plugin stack versions and environment size for debugging.
mkdir -p {str(self.analysis_paths.analysis_log_directory)}
{{
    echo "captured: $(date -Iseconds)"
    echo "env_size_bytes: $(env | wc -c)"
    echo "path_length_chars: ${{#PATH}}"
    ${{CONDA_PREFIX}}/bin/python -m snakemake --version 2>/dev/null | sed 's/^/snakemake: /'
    ${{CONDA_PREFIX}}/bin/pip show \\
        snakemake-executor-plugin-slurm \\
        snakemake-executor-plugin-slurm-jobstep \\
        snakemake-interface-executor-plugins \\
        snakemake-interface-common \\
        2>/dev/null | grep -E "^(Name|Version):"
    ${{CONDA_PREFIX}}/bin/python --version 2>&1 | sed 's/^/python: /'
}} > {str(self.analysis_paths.analysis_log_directory)}/snakemake_versions.txt

# Trim PATH before launching Snakemake to prevent ARG_MAX overflow in scontrol calls.
SLURM_BIN=$(dirname "$(command -v scontrol 2>/dev/null || echo "/opt/slurm/current/bin/scontrol")")
env PATH="${{CONDA_PREFIX}}/bin:${{SLURM_BIN}}:/usr/local/bin:/usr/bin:/usr/sbin:/bin" \\
    LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib" \\
    ${{CONDA_PREFIX}}/bin/python -m snakemake \\
    --profile {config_dir} \\
    --snakefile {snakefile_path} \\
    --executor slurm \\
    --printshellcmds \\
    --slurm-efficiency-report \\
    --slurm-efficiency-report-path {efficiency_report_path}
"""

            script_path = self.analysis_paths.analysis_dir / "run_workflow_batch_job.sh"
            script_path.write_text(script_content)
            script_path.chmod(0o755)

            if verbose:
                print(
                    f"[Snakemake] Generated batch orchestration script: {script_path}",
                    flush=True,
                )
                print(f"[Snakemake] Submitting with sbatch: {script_path}", flush=True)

            submit_result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                cwd=str(self.analysis_paths.analysis_dir),
            )

            # Parse job id from: "Submitted batch job 12345"
            job_id = None
            if submit_result.returncode == 0 and submit_result.stdout:
                parts = submit_result.stdout.strip().split()
                if len(parts) >= 4 and parts[0] == "Submitted":
                    job_id = parts[-1]

                    # Persist job ID to analysis log (batch_job mode - deprecated)
                    import datetime

                    # Note: batch_job mode is deprecated; use tmux mode instead
                    self.analysis.log.workflow_submission_time.set(datetime.datetime.now().isoformat())
                    self.analysis.log.workflow_submission_mode.set("batch_job")

            if submit_result.returncode != 0:
                error_msg = f"sbatch submission failed: {submit_result.stderr}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "batch_job",
                    "job_id": None,
                    "script_path": script_path,
                    "message": error_msg,
                }

            if verbose:
                print(
                    f"[Snakemake] Batch orchestration job submitted successfully (Job ID: {job_id})",
                    flush=True,
                )

            result_dict = {
                "success": True,
                "mode": "batch_job",
                "job_id": job_id,
                "script_path": script_path,
                "message": f"Batch orchestration workflow submitted (Job ID: {job_id})",
            }

            if wait_for_completion:
                if job_id:
                    completion_info = self._wait_for_slurm_job_completion(
                        job_id=job_id,
                        timeout=None,
                        verbose=verbose,
                    )
                    result_dict.update(completion_info)
                    result_dict["success"] = completion_info["completed"]
                else:
                    if verbose:
                        print(
                            "[Snakemake] ERROR: Failed to parse job ID for wait",
                            flush=True,
                        )
                    result_dict["success"] = False
                    result_dict["completed"] = False
                    result_dict["message"] = "Failed to parse job ID"

            return result_dict

        except Exception as e:
            error_msg = f"Failed to submit batch-job workflow: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "batch_job",
                "job_id": None,
                "script_path": None,
                "message": error_msg,
            }

    def _submit_tmux_workflow(
        self,
        snakefile_path: Path,
        wait_for_completion: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        Submit Snakemake workflow in detached tmux session.

        This approach runs Snakemake on the login node in a persistent tmux session,
        avoiding the orphaned jobs problem with sbatch orchestration. Snakemake's
        SIGINT handler properly cancels all worker jobs when the session receives SIGINT.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        wait_for_completion : bool
            If True, block until tmux session exits
        verbose : bool
            Print status messages

        Returns
        -------
        dict
            - success: bool
            - mode: str ("tmux")
            - session_name: str
            - snakemake_pid: int
            - message: str
        """
        try:
            # Build module load prefix for HPC systems
            module_load_prefix = self._get_module_load_prefix()

            # Check if tmux is available (with module load on HPC)
            tmux_check_cmd = f"{module_load_prefix}which tmux" if module_load_prefix else "which tmux"
            tmux_check = subprocess.run(
                ["bash", "-c", tmux_check_cmd],
                capture_output=True,
                text=True,
            )
            if tmux_check.returncode != 0:
                raise OSError(
                    "tmux is required for tmux workflow mode but not found in PATH. "
                    "Please install tmux or use multi_sim_run_method='local'."
                )

            # Pre-Snakemake guards: lock check + at-most-once reconciliation
            self._pre_snakemake_invocation_guards(snakefile_path, dry_run=False, verbose=verbose)

            # Generate unique session name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"triton_swmm_{self.cfg_analysis.analysis_id}_{timestamp}"

            # Check if session already exists (with module load on HPC)
            has_session_cmd = f"{module_load_prefix}tmux has-session -t {session_name}"
            session_check = subprocess.run(
                ["bash", "-c", has_session_cmd],
                capture_output=True,
                text=True,
            )
            if session_check.returncode == 0:
                raise RuntimeError(
                    f"Tmux session '{session_name}' already exists. "
                    "Please check if another workflow is running or kill the session manually."
                )

            # Build Snakemake command with absolute paths
            config_dir = self.analysis_paths.analysis_dir / ".snakemake_profile" / "slurm"

            # Create SLURM efficiency report directory and set timestamped filename
            from TRITON_SWMM_toolkit import utils as ut

            efficiency_report_dir = self.analysis.analysis_paths.analysis_log_directory / "slurm_efficiency_report"
            efficiency_report_dir.mkdir(parents=True, exist_ok=True)
            efficiency_report_filename = (
                f"slurm_efficiency_report_{ut.current_datetime_string(filepath_friendly=True)}.csv"
            )
            efficiency_report_path = efficiency_report_dir / efficiency_report_filename

            # Build module load commands for inside tmux session (reuse the same modules)
            # This ensures the Snakemake process has access to required modules
            module_load_cmd = module_load_prefix.removesuffix(" && ") if module_load_prefix else ""

            # Build the full command that will run inside tmux
            # Write output to a timestamped log file for debugging
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tmux_log = self.analysis_paths.analysis_log_directory / f"tmux_session_{timestamp}.log"

            # Create THE workflow script - this is what actually gets executed
            workflow_script = self.analysis_paths.analysis_dir / "run_workflow_tmux.sh"
            workflow_content = f"""#!/bin/bash
# TRITON-SWMM Tmux Workflow Script
# This is the ACTUAL script executed inside the tmux session
# Generated by TRITON-SWMM toolkit

{{
set -e  # Exit on error

echo "=== Tmux session started at $(date) ==="

# Load required modules (including tmux if needed)
{module_load_cmd}

echo "=== Modules loaded ==="

# Initialize conda
if [ -n "${{CONDA_EXE}}" ]; then
    eval "$(${{CONDA_EXE}} shell.bash hook)"
elif [ -n "${{CONDA_PREFIX}}" ] && [ -f "${{CONDA_PREFIX}}/../etc/profile.d/conda.sh" ]; then
    source "${{CONDA_PREFIX}}/../etc/profile.d/conda.sh"
else
    echo "ERROR: Cannot find conda initialization"
    exit 1
fi

echo "=== Conda initialized ==="

# Activate environment
conda activate triton_swmm_toolkit

echo "=== Environment activated ==="

# Ensure conda libs are discoverable
if [ -n "${{CONDA_PREFIX}}" ]; then
    export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH}}"
fi

# Diagnostics: confirm activation and Snakemake availability
echo "=========================================="
echo "DIAGNOSTICS: Conda activation + Snakemake"
echo "=========================================="
echo "CONDA_PREFIX: ${{CONDA_PREFIX:-<not set>}}"
echo "CONDA_DEFAULT_ENV: ${{CONDA_DEFAULT_ENV:-<not set>}}"
echo "Python (PATH): $(which python)"
echo "Python (conda): ${{CONDA_PREFIX}}/bin/python"
echo "PATH (head):"
echo "${{PATH}}" | tr ':' '\\n' | head -n 10 | sed 's/^/  /'
echo "=========================================="

${{CONDA_PREFIX}}/bin/python -V
${{CONDA_PREFIX}}/bin/python -m snakemake --version

# Capture Snakemake plugin stack versions and environment size for debugging.
# Written before the PATH trim so env_size_bytes reflects the pre-trim state.
mkdir -p {self.analysis_paths.analysis_log_directory}
{{
    echo "captured: $(date -Iseconds)"
    echo "env_size_bytes: $(env | wc -c)"
    echo "path_length_chars: ${{#PATH}}"
    ${{CONDA_PREFIX}}/bin/python -m snakemake --version 2>/dev/null | sed 's/^/snakemake: /'
    ${{CONDA_PREFIX}}/bin/pip show \\
        snakemake-executor-plugin-slurm \\
        snakemake-executor-plugin-slurm-jobstep \\
        snakemake-interface-executor-plugins \\
        snakemake-interface-common \\
        2>/dev/null | grep -E "^(Name|Version):"
    ${{CONDA_PREFIX}}/bin/python --version 2>&1 | sed 's/^/python: /'
}} > {self.analysis_paths.analysis_log_directory}/snakemake_versions.txt

# Trim PATH and LD_LIBRARY_PATH before launching Snakemake.
# After module load and conda activate, PATH can exceed Linux ARG_MAX limits.
# The snakemake-executor-plugin-slurm calls scontrol inheriting the full
# environment; if the env is too large, it crashes with OSError: [Errno 7]
# Argument list too long. We scope the trim to just the Snakemake process
# using `env` so the surrounding tmux script is unaffected.
SLURM_BIN=$(dirname "$(command -v scontrol 2>/dev/null || echo "/opt/slurm/current/bin/scontrol")")

# Run Snakemake
cd {self.analysis_paths.analysis_dir}
echo "=== Starting Snakemake ==="
set +e
env PATH="${{CONDA_PREFIX}}/bin:${{SLURM_BIN}}:/usr/local/bin:/usr/bin:/usr/sbin:/bin" \\
    LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib" \\
    ${{CONDA_PREFIX}}/bin/python -m snakemake \\
    --profile {config_dir} \\
    --snakefile {snakefile_path} \\
    --executor slurm \\
    --printshellcmds \\
    --slurm-efficiency-report \\
    --slurm-efficiency-report-path {efficiency_report_path}
snakemake_status=$?
echo "=== Snakemake completed at $(date) (exit: $snakemake_status) ==="
tmux kill-session -t {session_name}
exit $snakemake_status
}} >> {tmux_log} 2>&1
"""
            workflow_script.write_text(workflow_content)
            workflow_script.chmod(0o755)

            # Create detached tmux session (with module load on HPC)
            new_session_cmd = f"{module_load_prefix}tmux new-session -d -s {session_name} bash"

            tmux_result = subprocess.run(
                ["bash", "-c", new_session_cmd],
                capture_output=True,
                text=True,
                cwd=str(self.analysis_paths.analysis_dir),
            )

            if tmux_result.returncode != 0:
                error_msg = f"Failed to create tmux session: {tmux_result.stderr}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "tmux",
                    "session_name": None,
                    "snakemake_pid": None,
                    "message": error_msg,
                }

            # Execute THE workflow script in the tmux session
            exec_cmd = f"bash {workflow_script}"
            send_keys_cmd = f"{module_load_prefix}tmux send-keys -t {session_name} {shlex.quote(exec_cmd)} Enter"

            send_cmd_result = subprocess.run(
                ["bash", "-c", send_keys_cmd],
                capture_output=True,
                text=True,
            )

            if send_cmd_result.returncode != 0:
                # Clean up the session (with module load on HPC)
                kill_session_cmd = f"{module_load_prefix}tmux kill-session -t {session_name}"
                subprocess.run(["bash", "-c", kill_session_cmd], capture_output=True)
                error_msg = f"Failed to send command to tmux session: {send_cmd_result.stderr}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "tmux",
                    "session_name": None,
                    "snakemake_pid": None,
                    "message": error_msg,
                }

            # Wait a moment for process to start
            time.sleep(2)

            # Extract Snakemake PID from tmux session
            snakemake_pid = self._get_snakemake_pid_from_tmux(session_name)

            # Note: snakemake_pid may be None if Snakemake hasn't started yet

            # Capture the login node hostname for node-pinned reattach commands
            submission_node = socket.gethostname()

            # Persist session info to analysis log
            self.analysis.log.tmux_session_name.set(session_name)
            if snakemake_pid:
                self.analysis.log.snakemake_pid.set(snakemake_pid)
            self.analysis.log.workflow_submission_time.set(datetime.datetime.now().isoformat())
            self.analysis.log.workflow_submission_mode.set("tmux")
            self.analysis.log.workflow_submission_node.set(submission_node)

            # Determine the node to use in reattach commands:
            # prefer explicit config value, fall back to auto-detected hostname
            reattach_node = self.cfg_analysis.hpc_login_node or submission_node

            # Build node-pinned reattach commands (required when cluster uses
            # round-robin login load balancers, e.g. login.hpc.virginia.edu)
            module_load_cmd = self._get_module_load_prefix()
            if module_load_cmd:
                # On HPC: include module load tmux so a fresh SSH session can attach
                reattach_cmd = f"ssh {reattach_node} -t 'module load tmux && tmux attach -t {session_name}'"
                kill_cmd = f"ssh {reattach_node} -t 'module load tmux && tmux kill-session -t {session_name}'"
                list_cmd = f"ssh {reattach_node} -t 'module load tmux && tmux list-sessions'"
            else:
                reattach_cmd = f"tmux attach -t {session_name}"
                kill_cmd = f"tmux kill-session -t {session_name}"
                list_cmd = "tmux list-sessions"

            if verbose:
                print(
                    "[Snakemake] Tmux workflow submitted successfully",
                    flush=True,
                )
                print(f"[Snakemake] Session name: {session_name}", flush=True)
                print(f"[Snakemake] Submission node: {submission_node}", flush=True)
                if snakemake_pid:
                    print(f"[Snakemake] Snakemake PID: {snakemake_pid}", flush=True)
                print(f"[Snakemake] Log file: {tmux_log}", flush=True)
                print("", flush=True)
                print("[Snakemake] Useful commands:", flush=True)
                print(f"[Snakemake]   Monitor log:      tail -f {tmux_log}", flush=True)
                print(
                    f"[Snakemake]   Attach to session: {reattach_cmd}",
                    flush=True,
                )
                print("[Snakemake]   Detach from session: Ctrl+B, then D", flush=True)
                print(
                    f"[Snakemake]   Kill this session: {kill_cmd}",
                    flush=True,
                )
                print(
                    f"[Snakemake]   List all sessions: {list_cmd}",
                    flush=True,
                )

            result_dict = {
                "success": True,
                "mode": "tmux",
                "session_name": session_name,
                "snakemake_pid": snakemake_pid,
                "message": f"Tmux workflow submitted (session: {session_name})",
            }

            if wait_for_completion:
                if verbose:
                    print("[Snakemake] Waiting for workflow completion...", flush=True)
                completion_info = self._wait_for_tmux_session_completion(
                    session_name=session_name,
                    verbose=verbose,
                )
                result_dict.update(completion_info)
                result_dict["success"] = completion_info["completed"]

            return result_dict

        except Exception as e:
            error_msg = f"Failed to submit tmux workflow: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "tmux",
                "session_name": None,
                "snakemake_pid": None,
                "message": error_msg,
            }

    def _get_module_load_prefix(self) -> str:
        """
        Build module load prefix for HPC tmux commands.

        Returns
        -------
        str
            Shell command prefix to load modules, or empty string if not on HPC
        """
        modules_str = self.system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc

        # If we're in SLURM or using batch_job mode, always try to load tmux
        # Even if no other modules are specified, tmux might not be in default PATH
        if self.analysis.in_slurm or self.cfg_analysis.multi_sim_run_method == "batch_job":
            if modules_str:
                # modules_str is a space-separated string, e.g., "gcc/11.2.0 openmpi/4.1.1"
                return f"module purge && module load tmux {modules_str} && "
            else:
                # No other modules, but still load tmux on HPC
                return "module load tmux && "

        return ""

    def _get_snakemake_pid_from_tmux(self, session_name: str) -> int | None:
        """
        Extract Snakemake process ID from tmux session.

        Parameters
        ----------
        session_name : str
            Name of the tmux session

        Returns
        -------
        int | None
            Snakemake PID if found, None otherwise
        """
        module_load_prefix = self._get_module_load_prefix()
        try:
            # Get the shell PID in the tmux pane (with module load on HPC)
            list_panes_cmd = f"{module_load_prefix}tmux list-panes -t {session_name} -F '#{{pane_pid}}'"
            pane_pid_result = subprocess.run(
                ["bash", "-c", list_panes_cmd],
                capture_output=True,
                text=True,
            )

            if pane_pid_result.returncode != 0:
                return None

            shell_pid = int(pane_pid_result.stdout.strip())

            # Recursively search for Snakemake process in descendant tree
            # ps --ppid only shows direct children, so we need to recurse manually
            def find_snakemake_in_descendants(parent_pid: int) -> int | None:
                # Get direct children of this parent
                children_result = subprocess.run(
                    ["ps", "-o", "pid", "--ppid", str(parent_pid), "--no-headers"],
                    capture_output=True,
                    text=True,
                )

                if children_result.returncode != 0:
                    return None

                child_pids = [
                    int(pid.strip()) for pid in children_result.stdout.strip().split("\n") if pid.strip().isdigit()
                ]

                # Check each child process
                for child_pid in child_pids:
                    # Get the command line for this child
                    cmd_result = subprocess.run(
                        ["ps", "-o", "cmd", "-p", str(child_pid), "--no-headers"],
                        capture_output=True,
                        text=True,
                    )

                    if cmd_result.returncode == 0:
                        cmd = cmd_result.stdout.strip()
                        # Check if this is the Snakemake process
                        if "snakemake" in cmd and "python" in cmd:
                            return child_pid

                    # Recurse into this child's descendants
                    found_pid = find_snakemake_in_descendants(child_pid)
                    if found_pid:
                        return found_pid

                return None

            return find_snakemake_in_descendants(shell_pid)

        except Exception:
            return None

    def _wait_for_tmux_session_completion(
        self,
        session_name: str,
        verbose: bool = True,
    ) -> dict:
        """
        Wait for tmux session to exit.

        Parameters
        ----------
        session_name : str
            Name of the tmux session
        verbose : bool
            Print status messages

        Returns
        -------
        dict
            - completed: bool
            - message: str
        """
        module_load_prefix = self._get_module_load_prefix()
        try:
            while True:
                # Check if session still exists (with module load on HPC)
                has_session_cmd = f"{module_load_prefix}tmux has-session -t {session_name}"
                check_result = subprocess.run(
                    ["bash", "-c", has_session_cmd],
                    capture_output=True,
                    text=True,
                )

                if check_result.returncode != 0:
                    # Session no longer exists - workflow completed
                    if verbose:
                        print(
                            "[Snakemake] Tmux session exited - workflow complete",
                            flush=True,
                        )
                    return {
                        "completed": True,
                        "message": "Workflow completed successfully",
                    }

                # Session still exists, wait and check again
                time.sleep(5)

        except KeyboardInterrupt:
            if verbose:
                print("\n[Snakemake] Wait interrupted by user", flush=True)
            return {
                "completed": False,
                "message": "Wait interrupted by user",
            }
        except Exception as e:
            return {
                "completed": False,
                "message": f"Error while waiting: {str(e)}",
            }

    def submit_workflow(
        self,
        mode: Literal["local", "slurm", "auto"] = "auto",
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        override_clear_raw: ClearRawValue | None = None,
        compression_level: int = 5,
        pickup_where_leftoff: bool = False,
        wait_for_completion: bool = False,
        dry_run: bool = False,
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
        report_formats: list[str] | None = None,
        extra_sbatch_args: list[str] | None = None,
        snakemake_diagnostics: SnakemakeDiagnostics | None = None,
    ) -> dict:
        """
        Submit workflow using Snakemake.

        Automatically detects execution context (local vs. HPC) and submits accordingly.
        If multi_sim_run_method is "1_job_many_srun_tasks", submits as a single SLURM
        job with multiple srun tasks inside.

        Parameters
        ----------
        mode : Literal["local", "slurm", "auto"]
            Execution mode. If "auto", detects based on SLURM environment variables.
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings) in Phase 1
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM in Phase 1
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, each simulation will prepare its scenario before running
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after each simulation
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process (only used if process_timeseries=True)
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw`` threaded through to
            the emitted Snakefile rule shells.
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        wait_for_completion : bool
            If True, wait for workflow completion (relevant for slurm jobs only)
        dry_run : bool
            If True, only perform a dry run and return that result
        verbose : bool
            If True, print progress messages
        override_hpc_total_nodes : int | None
            If provided, overrides cfg_analysis.hpc_total_nodes for SLURM script generation
            without mutating the config object. Only valid when multi_sim_run_method is
            "1_job_many_srun_tasks"; raises ConfigurationError otherwise.

        Returns
        -------
        dict
            Status dictionary with keys defined by run_snakemake_local or run_snakemake_slurm
        """
        # Stash the diagnostics dataclass so the snakemake cmd-arg construction
        # sites (run_snakemake_local, _run_snakemake_slurm_detached,
        # _validate_batch_job_dry_run) can pick up the verbose/reason/log_path
        # flags without threading the kwarg through every internal signature.
        # A default-constructed instance is equivalent to passing None.
        self._active_snakemake_diagnostics = snakemake_diagnostics or SnakemakeDiagnostics()

        # Check if we should use 1-job mode based on config
        multi_sim_method = self.cfg_analysis.multi_sim_run_method

        if override_hpc_total_nodes is not None and multi_sim_method != "1_job_many_srun_tasks":
            raise ConfigurationError(
                field="override_hpc_total_nodes",
                message=(
                    f"override_hpc_total_nodes is only valid when multi_sim_run_method='1_job_many_srun_tasks', "
                    f"but current method is '{multi_sim_method}'."
                ),
                config_path=None,
            )

        if extra_sbatch_args is not None and multi_sim_method != "1_job_many_srun_tasks":
            raise ConfigurationError(
                field="extra_sbatch_args",
                message=(
                    f"extra_sbatch_args is only valid when multi_sim_run_method='1_job_many_srun_tasks' "
                    f"(it appends #SBATCH lines to the generated run_workflow_1job.sh), "
                    f"but current method is '{multi_sim_method}'."
                ),
                config_path=None,
            )

        if multi_sim_method == "1_job_many_srun_tasks":
            # Always submit a batch job for 1-job mode
            if verbose:
                print(
                    "[Snakemake] Using 1-job many-srun-tasks mode",
                    flush=True,
                )

            # v2 graceful-rerun: reconcile before Snakefile build so the alive
            # set substitutes wait-rules for run-rules at emit time (Phase 2).
            alive_by_token = dict(self._reconcile_inflight_submissions())
            # Generate Snakefile content
            snakefile_content = self.generate_snakefile_content(
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                compile_TRITON_SWMM=compile_TRITON_SWMM,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                which=which,
                override_clear_raw=override_clear_raw,
                compression_level=compression_level,
                pickup_where_leftoff=pickup_where_leftoff,
                report_formats=report_formats,
                alive_by_token=alive_by_token,
            )

            # Write Snakefile to disk
            snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
            snakefile_path.write_text(snakefile_content)

            if verbose:
                print(f"[Snakemake] Snakefile generated: {snakefile_path}", flush=True)

            # Always perform a dry run validation first
            dry_run_result = self._validate_single_job_dry_run(
                snakefile_path=snakefile_path,
                analysis=self.analysis,
                verbose=verbose,
                override_hpc_total_nodes=override_hpc_total_nodes,
            )

            if dry_run:
                # Override mode to indicate intended execution context
                dry_run_result["mode"] = "single_job"
                self.analysis._refresh_log()
                return dry_run_result

            result = self._submit_single_job_workflow(
                snakefile_path=snakefile_path,
                wait_for_completion=wait_for_completion,
                verbose=verbose,
                override_hpc_total_nodes=override_hpc_total_nodes,
                extra_sbatch_args=extra_sbatch_args,
            )

            self.analysis._refresh_log()
            return result

        if multi_sim_method == "batch_job":
            if verbose:
                print(
                    "[Snakemake] Using batch_job mode (tmux orchestration)",
                    flush=True,
                )

            # v2 graceful-rerun: reconcile before Snakefile build (Phase 2).
            alive_by_token = dict(self._reconcile_inflight_submissions())
            snakefile_content = self.generate_snakefile_content(
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                compile_TRITON_SWMM=compile_TRITON_SWMM,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                which=which,
                override_clear_raw=override_clear_raw,
                compression_level=compression_level,
                pickup_where_leftoff=pickup_where_leftoff,
                report_formats=report_formats,
                alive_by_token=alive_by_token,
            )

            snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
            snakefile_path.write_text(snakefile_content)

            if verbose:
                print(f"[Snakemake] Snakefile generated: {snakefile_path}", flush=True)

            # Use batch_job dry run validation (same SLURM profile)
            dry_run_result = self._validate_batch_job_dry_run(
                snakefile_path=snakefile_path,
                verbose=verbose,
            )

            if not dry_run_result.get("success"):
                raise RuntimeError("Dry run failed; workflow submission aborted.")

            if dry_run:
                self.analysis._refresh_log()
                return dry_run_result

            result = self._submit_tmux_workflow(
                snakefile_path=snakefile_path,
                wait_for_completion=wait_for_completion,
                verbose=verbose,
            )

            self.analysis._refresh_log()
            return result

        # Standard workflow submission (existing logic)
        if mode == "auto":
            mode = "slurm" if self.analysis.in_slurm else "local"

        if verbose:
            print(f"[Snakemake] Submitting workflow in {mode} mode", flush=True)

        # v2 graceful-rerun: reconcile before Snakefile build (Phase 2).
        alive_by_token = dict(self._reconcile_inflight_submissions())
        # Generate Snakefile content
        snakefile_content = self.generate_snakefile_content(
            process_system_level_inputs=process_system_level_inputs,
            overwrite_system_inputs=overwrite_system_inputs,
            compile_TRITON_SWMM=compile_TRITON_SWMM,
            recompile_if_already_done_successfully=recompile_if_already_done_successfully,
            prepare_scenarios=prepare_scenarios,
            overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            process_timeseries=process_timeseries,
            which=which,
            override_clear_raw=override_clear_raw,
            compression_level=compression_level,
            pickup_where_leftoff=pickup_where_leftoff,
            report_formats=report_formats,
            alive_by_token=alive_by_token,
        )

        # Write Snakefile to disk
        snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
        snakefile_path.write_text(snakefile_content)

        if verbose:
            print(f"[Snakemake] Snakefile generated: {snakefile_path}", flush=True)

        # Always perform a dry run first
        if mode == "local":
            dry_run_result = self.run_snakemake_local(
                snakefile_path=snakefile_path,
                verbose=verbose,
                dry_run=True,
            )
        else:  # slurm
            dry_run_result = self._run_snakemake_slurm_detached(
                snakefile_path=snakefile_path,
                wait_for_completion=True,
                verbose=verbose,
                dry_run=True,
            )

        if not dry_run_result.get("success"):
            raise RuntimeError("Dry run failed; workflow submission aborted.")

        if dry_run:
            self.analysis._refresh_log()
            return dry_run_result

        # Submit workflow based on mode
        if mode == "local":
            result = self.run_snakemake_local(
                snakefile_path=snakefile_path,
                verbose=verbose,
            )
        else:  # slurm
            result = self._run_snakemake_slurm_detached(
                snakefile_path=snakefile_path,
                wait_for_completion=wait_for_completion,
                verbose=verbose,
            )

        self.analysis._refresh_log()
        return result

    def submit_reprocess_workflow(
        self,
        *,
        start_with: Literal["process", "consolidate", "render"],
        execution_mode: Literal["auto", "local", "slurm"] = "auto",
        multi_sim_run_method_override: str | None = None,
        dry_run: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Submit a reprocess-scoped workflow against existing sim outputs.

        Re-runs downstream stages (process / consolidate / plot / render)
        without re-running simulations. Uses a separate
        ``{analysis_dir}/.snakemake_reprocess/`` working directory so it can
        coexist with a live simulation driver and clears that subtree's
        ``.snakemake/locks/`` rather than the main ``.snakemake/locks/``.

        Parameters
        ----------
        start_with
            Downstream stage to re-fire from. See
            :func:`TRITON_SWMM_toolkit.reprocess_snakefile_generator.generate_reprocess_snakefile`
            for the stage → re-emitted rule mapping.
        execution_mode
            ``"auto"`` detects SLURM context; ``"local"`` forces local
            subprocess execution; ``"slurm"`` forces SLURM submission.
        multi_sim_run_method_override
            When set, takes precedence over the analysis's configured
            ``multi_sim_run_method`` for execution dispatch. Used by
            :meth:`TRITONSWMM_analysis.reprocess` to force the
            ``1_job_many_srun_tasks`` method to ``batch_job`` semantics on
            reprocess paths (the original allocation contract does not
            apply to reprocess).
        dry_run
            If True, runs ``snakemake --dry-run`` only and returns.
        verbose
            If True, print progress messages.

        Returns
        -------
        dict
            Status dictionary matching the run-path's shape (``success``,
            ``mode``, ``snakefile_path``, ``job_id``, ``message``,
            ``snakemake_logfile``).
        """
        from TRITON_SWMM_toolkit.reprocess_snakefile_generator import (
            write_reprocess_snakefile,
        )

        # Effective execution dispatch — reprocess overrides take precedence.
        effective_method = (
            multi_sim_run_method_override
            if multi_sim_run_method_override is not None
            else self.cfg_analysis.multi_sim_run_method
        )
        if execution_mode == "auto":
            mode: Literal["local", "slurm"] = "slurm" if self.analysis.in_slurm else "local"
        else:
            mode = execution_mode  # type: ignore[assignment]

        if verbose:
            print(
                f"[Snakemake] Submitting reprocess workflow (start_with={start_with!r}, "
                f"mode={mode}, method={effective_method})",
                flush=True,
            )

        # Write the reprocess-scoped Snakefile.
        snakefile_path = write_reprocess_snakefile(self, start_with=start_with)
        if verbose:
            print(
                f"[Snakemake] Reprocess Snakefile generated: {snakefile_path}",
                flush=True,
            )

        # Logs.
        logs_dir = self.analysis_paths.analysis_log_directory
        logs_dir.mkdir(parents=True, exist_ok=True)
        logfile_name = "snakemake_reprocess_dry_run.log" if dry_run else "snakemake_reprocess.log"
        snakemake_logfile = logs_dir / logfile_name

        # Snakemake working directory: ``analysis_dir`` itself. The plan's
        # original design used ``--directory analysis_dir/.snakemake_reprocess``
        # to isolate the reprocess driver's ``.snakemake/`` state from a
        # parallel live sim driver, but Snakemake's ``--directory`` flag also
        # re-roots every relative path in the Snakefile (rule inputs/outputs)
        # against the working dir — which breaks resolution of
        # ``_status/c_run_*.flag`` and every other relative artifact path
        # because those live in ``analysis_dir/_status/``, not
        # ``.snakemake_reprocess/_status/``. Sharing ``analysis_dir/.snakemake/``
        # is safe for Phase 2's local tests and CLI smoke because the
        # reprocess Snakefile lives at a distinct path
        # (``Snakefile.reprocess``) and Snakemake locks are keyed per
        # Snakefile. **Follow-up**: true coexistence with a concurrent live
        # ``rule run_*`` driver requires either rewriting reprocess Snakefile
        # paths to absolute form or adopting a future ``--lock-dir``-style
        # mechanism. See ``# Follow-up Ideas`` in the in-flight sidecar.
        reprocess_working_dir = self.analysis_paths.analysis_dir

        # Build the snakemake command. Reuses the run/submit base command and
        # adds ``--rerun-triggers mtime`` so downstream rules only re-fire
        # when outputs are missing or older — the surgical reprocess intent.
        cmd_args = self._get_snakemake_base_cmd() + [
            "--snakefile",
            str(snakefile_path),
            "--rerun-triggers",
            "mtime",
        ]

        if mode == "local":
            local_cores = self.cfg_analysis.local_cpu_cores_for_workflow
            assert isinstance(local_cores, int), "local_cpu_cores_for_workflow must be specified for local runs"
            if local_cores > 1:
                cmd_args.extend(["--cores", str(local_cores)])
            else:
                cmd_args.extend(["--cores", "1"])
        else:  # slurm
            # Build slurm profile (same as the run path) — reprocess inherits
            # the analysis's slurm submission contract for downstream rules.
            config = self.generate_snakemake_config(mode="slurm")
            config_dir = self.write_snakemake_config(config, mode="slurm")
            cmd_args.extend(
                [
                    "--profile",
                    str(config_dir),
                    "--executor",
                    "slurm",
                    "--printshellcmds",
                ]
            )

        if dry_run:
            cmd_args.append("--dry-run")

        # Facade — lock check (against the shared analysis_dir/.snakemake/
        # per the working-dir explanation above) + reconciliation against
        # analysis_dir/_status/_submitted/. Phase 1's at-most-once guard
        # protects reprocess from a parallel live sim driver
        # double-submitting; the lock-check working_dir matches the
        # subprocess cwd below.
        self._pre_snakemake_invocation_guards(
            snakefile_path,
            dry_run=dry_run,
            verbose=verbose,
            working_dir=reprocess_working_dir,
        )

        # Subprocess invocation. Local runs block; slurm runs detach (the run
        # path's distinction is preserved here).
        if mode == "local":
            with open(snakemake_logfile, "w") as log_f:
                result = subprocess.run(
                    cmd_args,
                    cwd=str(self.analysis_paths.analysis_dir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if verbose:
                print(f"[Snakemake] command: \n     {' '.join(cmd_args)}")
            if result.returncode != 0:
                return {
                    "success": False,
                    "mode": "local",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": (f"Snakemake reprocess failed. See {snakemake_logfile} for details."),
                    "snakemake_logfile": snakemake_logfile,
                }
            if verbose:
                print("[Snakemake] Reprocess completed successfully", flush=True)
            self.analysis._refresh_log()
            return {
                "success": True,
                "mode": "local",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Reprocess completed successfully",
                "snakemake_logfile": snakemake_logfile,
            }

        # slurm path
        with open(snakemake_logfile, "w") as log_f:
            proc = subprocess.Popen(
                cmd_args,
                cwd=str(self.analysis_paths.analysis_dir),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        if verbose:
            print(
                f"[Snakemake] Reprocess submitted to background (PID: {proc.pid})",
                flush=True,
            )
        self.analysis._refresh_log()
        return {
            "success": True,
            "mode": "slurm",
            "snakefile_path": snakefile_path,
            "job_id": None,
            "message": "Reprocess submitted to SLURM (detached)",
            "process": proc,
            "snakemake_logfile": snakemake_logfile,
        }

    def _classify_live_sentinels(
        self,
        sentinel_paths: list[Path],
        *,
        reclaim_dead: bool = True,
    ) -> list[tuple[str, str]]:
        """Classify submitted-sentinel files by SLURM job liveness.

        Returns the list of ``(sentinel_stem, job_id)`` tuples for sentinels
        whose recorded SLURM job is still live per the module-level
        :func:`_slurm_job_is_live` helper. When ``reclaim_dead=True`` (the
        default), sentinels whose recorded job is dead or whose JSON payload
        is corrupt are unlinked in-place — matching the original behavior of
        :meth:`_reconcile_inflight_submissions`. Pass ``reclaim_dead=False``
        from the delete path so the destructive sentinel sweep is owned by
        the delete workflow itself (not by the preflight guard).

        Per cleanup-rerun-delete-redesign Phase 2 (extracted from the body of
        :meth:`_reconcile_inflight_submissions` so the delete-path guard can
        share the classification primitive).
        """
        import json

        alive: list[tuple[str, str]] = []
        for s in sentinel_paths:
            try:
                jid = str(json.loads(s.read_text()).get("slurm_jobid") or "")
            except json.JSONDecodeError:
                print(
                    f"[reconcile] WARNING: corrupt sentinel {s.name}; "
                    f"{'deleting and skipping' if reclaim_dead else 'skipping (not reclaimed)'}",
                    file=sys.stderr,
                    flush=True,
                )
                if reclaim_dead:
                    s.unlink(missing_ok=True)
                continue
            except OSError as _e:
                raise WorkflowError(
                    phase="preflight-reconciliation",
                    return_code=1,
                    stderr=(f"Failed to read sentinel {s}: {_e}. Retry, or inspect _status/_submitted/ manually."),
                ) from _e
            if jid and _slurm_job_is_live(jid):
                alive.append((s.stem, jid))
            elif reclaim_dead:
                s.unlink(missing_ok=True)  # DEAD/stale → reclaim
        return alive

    def _classify_via_state_markers(
        self,
        sentinel_paths: list[Path],
        *,
        reclaim_completed: bool = True,
        analysis_dir: Path | None = None,
    ) -> list[tuple[str, str]]:
        """Classify submitted-sentinel files by v2 state-marker presence.

        Sentinel-only liveness primitive (no SLURM CLI probes). For each
        submitted-sentinel under ``_status/_submitted/``, check for the
        existence of a sibling marker under
        ``_status/_completed/{rule_token}.json`` or
        ``_status/_failed/{rule_token}.json``. Classification:

        - Marker present (completed or failed) → not alive; the runner has
          finished and the submitted-sentinel will be cleaned up by the
          runner's finally (or has been racy-deleted already). When
          ``reclaim_completed=True`` (the default), the submitted-sentinel
          is unlinked here as a safety net.
        - No marker present → alive (the original SLURM job is still
          running, or has died at the OS level before the finally fired).

        The returned alive list carries ``(sentinel_stem, slurm_jobid)``
        tuples — same shape as :meth:`_classify_live_sentinels` so callers
        can treat them interchangeably from the alive-set side. The
        dead-side behavior differs: :meth:`_classify_live_sentinels`
        returns from squeue state; this helper returns from marker
        presence.

        Per cleanup-rerun-delete-redesign + sentinel-system-v2 Phase A.
        """
        import json

        alive: list[tuple[str, str]] = []
        base_dir = analysis_dir if analysis_dir is not None else self.analysis_paths.analysis_dir
        completed_dir = base_dir / "_status" / "_completed"
        failed_dir = base_dir / "_status" / "_failed"
        for s in sentinel_paths:
            rule_token = s.stem
            completed = completed_dir / f"{rule_token}.json"
            failed = failed_dir / f"{rule_token}.json"
            if completed.exists() or failed.exists():
                if reclaim_completed:
                    s.unlink(missing_ok=True)
                continue
            try:
                jid = str(json.loads(s.read_text()).get("slurm_jobid") or "")
            except (json.JSONDecodeError, OSError):
                # Corrupt sentinel; classify as alive conservatively (a
                # subsequent wait-rule will time out via walltime cap).
                jid = ""
            alive.append((rule_token, jid))
        return alive

    def _classify_stale_via_sacct(
        self,
        marker_less_alive: list[tuple[str, str]],
        *,
        analysis_dir: Path | None = None,
        walltime_slack_min: int = 30,
    ) -> tuple[list[tuple[str, str]], list[_ClearedToken]]:
        """R-STALE second pass: authoritatively classify the marker-less set.

        Three-state classification per the FQ1 table:
        - sacct State in _SACCT_DEAD_STATES        -> DEAD-stale (reclaim + surface)
        - sacct State present, not in dead set     -> ALIVE (keep as wait-rule)
        - job-id absent from sacct OR blank job-id -> UNKNOWN, resolved by the
          submitted-sentinel mtime-age tiebreak.

        DEAD-classified submitted-sentinels are unlinked in-place. Returns
        ``(still_alive, cleared)`` where ``cleared`` is
        ``(rule_token, job_id, state, reason)`` for the R-STALE surface.
        ONE sacct call regardless of |marker_less_alive| (R5); zero-call no-op
        when the input is empty (the common all-markers case).
        """
        if not marker_less_alive:
            return [], []

        base_dir = analysis_dir if analysis_dir is not None else self.analysis_paths.analysis_dir
        submitted_dir = base_dir / "_status" / "_submitted"
        # Single-slack site (resolves the double-slack hazard): the helper already
        # adds the slack, so pass walltime_slack_min through it ONCE and do NOT add
        # slack again here.
        cap_min = _max_plausible_job_lifetime_min(self.cfg_analysis, slack_min=walltime_slack_min)
        max_plausible_s = cap_min * 60

        job_ids = [j for _, j in marker_less_alive if j]
        states = _sacct_states_batched(job_ids)

        still_alive: list[tuple[str, str]] = []
        cleared: list[_ClearedToken] = []
        for rule_token, jid in marker_less_alive:
            sentinel = submitted_dir / f"{rule_token}.json"
            row = states.get(jid) if jid else None
            if row is not None:
                state, _exit, reason = row
                if state in _SACCT_DEAD_STATES:
                    sentinel.unlink(missing_ok=True)
                    cleared.append(_ClearedToken(rule_token, jid, state, reason))
                else:
                    still_alive.append((rule_token, jid))
                continue
            try:
                age_s = time.time() - sentinel.stat().st_mtime
            except OSError:
                age_s = max_plausible_s + 1
            if age_s >= max_plausible_s:
                sentinel.unlink(missing_ok=True)
                cleared.append(_ClearedToken(rule_token, jid or "(no jobid)", "UNKNOWN", "purged/age-exceeded"))
            else:
                still_alive.append((rule_token, jid))
        return still_alive, cleared

    def _emit_wait_for_sim_rule_block(
        self,
        *,
        rule_token: str,
        flag_output_path: str,
        run_rule_inputs: list[str],
        wait_walltime_cap_min: int,
        analysis_dir_override: str | None = None,
    ) -> str:
        """Emit a Snakemake rule body that waits on the original SLURM job's
        completion-marker write, in place of a normal ``rule run_*`` block.

        The wait-rule:

        - Is declared local via the Snakefile's ``localrules: wait_for_*``
          preamble entry so it runs as a Snakemake-local task (no sbatch).
        - Declares ``output:`` matching the run-rule's flag path exactly.
        - Declares ``input:`` as the run-rule's input set (R7 — a superset is
          permitted; here the equal set satisfies the superset constraint),
          ensuring downstream rules' INPUT trigger does not fire spuriously
          after wait -> run transitions on subsequent driver invocations.
        - Polls ``_status/_completed/{rule_token}.json`` and
          ``_status/_failed/{rule_token}.json`` from its ``shell:`` body — NOT
          from ``input:`` — so the INPUT trigger does not see markers as
          load-bearing dependencies.
        - Exits 0 on ``_completed/`` marker presence. The original SLURM
          job's runner already wrote ``flag_output_path`` before its
          ``try/finally`` wrote the ``_completed/`` marker, so by the time the
          wait-rule observes the marker the flag is already on disk and
          Snakemake's output check passes. The wait-rule does NOT write the
          flag itself (preserves the v1 "completion flag is written exactly
          once, by the original worker" contract).
        - Exits 1 on ``_failed/`` marker presence (propagating failure to
          Snakemake's standard error handling).
        - Exits 1 after ``wait_walltime_cap_min`` minutes if neither marker
          appears.

        Deliberately emits no ``conda:`` directive — the shell uses an
        absolute ``{python_exe} -m TRITON_SWMM_toolkit.wait_for_sentinel_runner``
        so it runs in the driver's interpreter (the poll loop has no
        analysis-env dependency).

        ``analysis_dir_override`` sets the ``--analysis-dir`` the wait-runner
        polls. The multisim path leaves it ``None`` (markers live under the
        single master ``analysis_dir``). The sensitivity path MUST pass the
        per-sub-analysis dir (``master/subanalyses/{analysis_id}``) because
        ``run_simulation_runner`` writes sensitivity markers under the sub
        analysis's own ``analysis_dir`` (sensitivity_analysis.py:1485), not the
        master dir — without the override the wait-runner would poll the wrong
        directory and always hit the walltime cap.

        Per sentinel-system-v2 Phase 2 (resolves D-Q5=A, D-Q2=A, R7, Spec D).
        """
        sanitized = _sanitize_rule_name(rule_token)
        inputs_block = "\n        ".join(f'"{p}",' for p in sorted(set(run_rule_inputs)))
        analysis_dir = (
            analysis_dir_override if analysis_dir_override is not None else str(self.analysis_paths.analysis_dir)
        )
        python_exe = self.python_executable
        # Snakemake `localrules:` takes literal rule names (not globs) and is
        # additive across statements (docs § Local Rules). Emit a concrete
        # per-rule localrules line so the wait-rule runs on the host node
        # rather than being submitted to SLURM. A `wait_for_*` glob does NOT
        # work — it would be read as a literal (nonexistent) rule name.
        return (
            f"localrules: wait_for_{sanitized}\n\n"
            f"rule wait_for_{sanitized}:\n"
            f"    input:\n"
            f"        {inputs_block}\n"
            f"    output:\n"
            f'        "{flag_output_path}"\n'
            f"    resources: cpus_per_task=1, mem_mb=100, runtime={wait_walltime_cap_min}\n"
            f"    shell:\n"
            f'        "{python_exe} -m TRITON_SWMM_toolkit.wait_for_sentinel_runner "\n'
            f'        "--rule-token {rule_token} "\n'
            f'        "--flag-output {{output}} "\n'
            f'        "--analysis-dir {analysis_dir} "\n'
            f'        "--max-wait-minutes {wait_walltime_cap_min}"\n\n'
        )

    def _pre_delete_guards(self, *, override_in_flight: bool) -> None:
        """Entry guard for the distributed delete workflow.

        Mirrors :meth:`_pre_snakemake_invocation_guards` but for the delete
        path: (1) lock-check scoped to ``analysis_dir/.snakemake_delete/`` so
        a stale Snakemake lock from a prior aborted delete attempt is
        surfaced before resubmit; (2) sentinel classification via
        :meth:`_classify_live_sentinels` (no reclaim — the delete workflow
        owns sentinel cleanup); (3) ``--comment``-based sacct recovery via
        :meth:`_recover_inflight_via_comment` so the lost-sentinel window
        does not silently admit a delete against a live analysis. Refuses
        with :class:`ConfigurationError` when any live job is detected and
        ``override_in_flight`` is False.

        Per cleanup-rerun-delete-redesign Phase 2 (D-DeleteSentinelInteraction
        resolution + design recommendations C.4 and C.5).
        """
        analysis_dir = self.analysis_paths.analysis_dir

        # (1) Lock-check scoped to .snakemake_delete/ (C.5).
        snakefile_delete = analysis_dir / "Snakefile.delete"
        if snakefile_delete.exists():
            self._check_and_clear_snakemake_lock(
                snakefile_delete,
                dry_run=False,
                verbose=True,
                working_dir=analysis_dir / ".snakemake_delete",
            )

        # (2) Sentinel classification (no reclaim — destructive sentinel
        # cleanup belongs to the delete-consolidation runner, not the
        # preflight guard).
        submitted_dir = analysis_dir / "_status" / "_submitted"
        sentinels = sorted(submitted_dir.glob("*.json")) if submitted_dir.exists() else []
        alive = self._classify_live_sentinels(sentinels, reclaim_dead=False)

        # (3) Comment-recovery for the lost-sentinel window (C.4).
        alive += self._recover_inflight_via_comment(known_jobids={j for _, j in alive})

        if alive and not override_in_flight:
            live_jids = sorted({j for _, j in alive})
            raise ConfigurationError(
                field="analysis.delete()",
                message=(
                    f"Refusing to delete analysis_dir while {len(live_jids)} "
                    f"simulation jobs are still live in SLURM: {live_jids}. "
                    f"Cancel via `scancel {' '.join(str(j) for j in live_jids)}` "
                    f"and retry, or pass `override_in_flight=True` (Python) "
                    f"or `--override-in-flight` (CLI) to proceed."
                ),
                config_path=str(submitted_dir),
            )
        if alive and override_in_flight:
            live_jids = sorted({j for _, j in alive})
            print(
                f"[delete] override_in_flight=True — proceeding despite {len(live_jids)} live SLURM jobs: {live_jids}",
                flush=True,
            )

    def _build_delete_snakefile_content(self) -> str:
        """Build the non-sensitivity delete Snakefile content.

        Per cleanup-rerun-delete-redesign Phase 2 + D-DeleteBoundary Option 1.
        Uses absolute paths in rule outputs so resolution is robust regardless
        of Snakemake's ``--directory`` flag.
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        analysis_dir = str(self.analysis_paths.analysis_dir)
        python_exe = self.python_executable

        # Per-scenario delete rules. Snakemake rule names must be valid Python
        # identifiers (no dots / hyphens / spaces), so the event_id slug is
        # sanitized for the rule name only; the file-path interpolations keep
        # the original event_id so flag paths match what
        # `_enumerate_expected_delete_sentinels` produces.
        rules = []
        per_scenario_flags = []
        for i in range(len(self.analysis.df_sims)):
            event_id = compute_event_id_slug(self.analysis._retrieve_weather_indexer_using_integer_index(i))
            rule_name_slug = event_id.replace(".", "_").replace("-", "_")
            flag = f"{analysis_dir}/_status/_deleting/scenario_evt-{event_id}.flag"
            per_scenario_flags.append(flag)
            rules.append(
                f"rule delete_scenario_{rule_name_slug}:\n"
                f"    output:\n"
                f'        "{flag}"\n'
                f'    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n'
                f'    shell:\n'
                f'        "{python_exe} -m TRITON_SWMM_toolkit.delete_scenario_runner "\n'
                f'        "--event-id {event_id} "\n'
                f'        "--analysis-dir {analysis_dir}"\n\n'
            )

        consolidation_flag = f"{analysis_dir}/_status/_deleting/analysis_consolidation.flag"
        consolidation_inputs = ",\n        ".join(f'"{f}"' for f in per_scenario_flags)
        rules.append(
            f"rule delete_analysis_consolidation:\n"
            f"    input:\n"
            f"        {consolidation_inputs if per_scenario_flags else ''}\n"
            f"    output:\n"
            f'        "{consolidation_flag}"\n'
            f'    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n'
            f'    shell:\n'
            f'        "{python_exe} -m TRITON_SWMM_toolkit.delete_consolidation_runner "\n'
            f'        "--analysis-dir {analysis_dir}"\n\n'
        )

        rule_all = f'rule all:\n    input:\n        "{consolidation_flag}"\n\n'
        return rule_all + "".join(rules)

    def _build_delete_sensitivity_snakefile_content(self, sa_ids: list[str]) -> str:
        """Build the sensitivity-master delete Snakefile content.

        Per cleanup-rerun-delete-redesign Phase 2 + D-DeleteBoundary Option 1.
        Per-sub-analysis delete rules + analysis-level consolidation rule.
        """
        analysis_dir = str(self.analysis_paths.analysis_dir)
        python_exe = self.python_executable

        rules = []
        per_sa_flags = []
        for sa_id in sa_ids:
            # Snakemake rule names must be valid Python identifiers; sanitize
            # the sa_id for the rule name only, keep flag-path interpolation
            # using the original sa_id so paths match `_enumerate_expected_*`.
            rule_name_slug = sa_id.replace(".", "_").replace("-", "_")
            flag = f"{analysis_dir}/_status/_deleting/subanalysis_sa-{sa_id}.flag"
            per_sa_flags.append(flag)
            rules.append(
                f"rule delete_subanalysis_{rule_name_slug}:\n"
                f"    output:\n"
                f'        "{flag}"\n'
                f'    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n'
                f'    shell:\n'
                f'        "{python_exe} -m TRITON_SWMM_toolkit.delete_subanalysis_runner "\n'
                f'        "--sa-id {sa_id} "\n'
                f'        "--analysis-dir {analysis_dir}"\n\n'
            )

        consolidation_flag = f"{analysis_dir}/_status/_deleting/analysis_consolidation.flag"
        consolidation_inputs = ",\n        ".join(f'"{f}"' for f in per_sa_flags)
        rules.append(
            f"rule delete_analysis_consolidation:\n"
            f"    input:\n"
            f"        {consolidation_inputs if per_sa_flags else ''}\n"
            f"    output:\n"
            f'        "{consolidation_flag}"\n'
            f'    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n'
            f'    shell:\n'
            f'        "{python_exe} -m TRITON_SWMM_toolkit.delete_consolidation_runner "\n'
            f'        "--analysis-dir {analysis_dir}"\n\n'
        )

        rule_all = f'rule all:\n    input:\n        "{consolidation_flag}"\n\n'
        return rule_all + "".join(rules)

    def _resolve_delete_mode_from_method(
        self,
        method: Literal["local", "batch_job", "1_job_many_srun_tasks"] | None,
    ) -> Literal["local", "slurm"]:
        """Map ``analysis_config.multi_sim_run_method`` (or None) to delete-executor mode.

        ``None`` or ``"local"`` → ``"local"``; ``"batch_job"`` or
        ``"1_job_many_srun_tasks"`` → ``"slurm"``. The ``None`` branch covers
        analyses whose ``cfg_analysis`` was loaded from a YAML that did not
        explicitly set ``multi_sim_run_method`` — these are treated as
        ``"local"`` by default, matching the pre-Phase-3 ``--cores 1`` behavior.
        """
        if method is None or method == "local":
            return "local"
        if method in ("batch_job", "1_job_many_srun_tasks"):
            return "slurm"
        raise ConfigurationError(
            field="multi_sim_run_method",
            message=f"Unrecognized multi_sim_run_method={method!r} for delete-executor resolution",
        )

    def _submit_delete_snakemake(
        self,
        snakefile_path: Path,
        *,
        override_multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks"] | None = None,
        verbose: bool = True,
    ) -> dict:
        """Invoke the delete Snakefile via subprocess.

        Per cleanup-rerun-delete-redesign Phase 2 + distributed-delete-and-du-
        recording Phase 3: supports both local (``--cores N``) and slurm
        (``--executor slurm --profile <config_dir>``) execution. Mode is
        resolved from ``analysis_config.multi_sim_run_method`` unless an
        explicit ``override_multi_sim_run_method`` is supplied (read-config-
        when-None per the override-prefix convention stipulation).
        """
        analysis_dir = self.analysis_paths.analysis_dir
        snakemake_dir = analysis_dir / ".snakemake_delete"
        snakemake_dir.mkdir(exist_ok=True)
        logs_dir = self.analysis_paths.analysis_log_directory
        logs_dir.mkdir(parents=True, exist_ok=True)
        logfile = logs_dir / "snakemake_delete.log"

        resolved_method = (
            override_multi_sim_run_method
            if override_multi_sim_run_method is not None
            else self.cfg_analysis.multi_sim_run_method
        )
        executor_mode = self._resolve_delete_mode_from_method(resolved_method)

        cmd_args = self._get_snakemake_base_cmd() + [
            "--snakefile",
            str(snakefile_path),
            "--directory",
            str(snakemake_dir),
            # NOTE: the delete path deliberately RETAINS the `input` rerun-trigger
            # (unlike the run path, narrowed to mtime-only in generate_snakemake_config
            # for post-death-recovery). The DU-sentinel delete rules declare `input:`
            # deps whose SET changes legitimately require the `input` trigger to refire.
            # Do not "consistency"-narrow this to mtime-only.
            "--rerun-triggers",
            "mtime",
            "input",
        ]
        if executor_mode == "slurm":
            config = self.generate_snakemake_config(mode="slurm")
            config_dir = self.write_snakemake_config(config, mode="slurm")
            cmd_args += ["--executor", "slurm", "--profile", str(config_dir)]
        else:
            cmd_args += ["--cores", str(self.cfg_analysis.hpc_max_simultaneous_sims or 1)]
        if verbose:
            print(f"[Snakemake] Delete command: {' '.join(cmd_args)}", flush=True)
        with open(logfile, "w") as log_f:
            result = subprocess.run(
                cmd_args,
                cwd=str(analysis_dir),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        return {
            "success": result.returncode == 0,
            "snakefile_path": snakefile_path,
            "snakemake_logfile": logfile,
            "returncode": result.returncode,
        }

    def submit_delete_workflow(
        self,
        *,
        override_in_flight: bool = False,
        override_multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks"] | None = None,
    ) -> dict:
        """Generate and submit the distributed delete Snakefile (non-sensitivity).

        Per cleanup-rerun-delete-redesign Phase 2 + D-DeleteBoundary Option 1.
        Runs :meth:`_pre_delete_guards` first so the live-sentinel refusal
        fires before any Snakefile is written.
        """
        self._pre_delete_guards(override_in_flight=override_in_flight)

        snakefile_path = self.analysis_paths.analysis_dir / "Snakefile.delete"
        snakefile_path.write_text(self._build_delete_snakefile_content())
        return self._submit_delete_snakemake(
            snakefile_path,
            override_multi_sim_run_method=override_multi_sim_run_method,
        )

    def submit_delete_workflow_sensitivity(
        self,
        *,
        sa_ids: list[str],
        override_in_flight: bool = False,
        override_multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks"] | None = None,
    ) -> dict:
        """Generate and submit the sensitivity-master delete Snakefile.

        Per cleanup-rerun-delete-redesign Phase 2 + D-DeleteBoundary Option 1.
        Per-sub-analysis fan-out + analysis-level consolidation.
        """
        self._pre_delete_guards(override_in_flight=override_in_flight)

        snakefile_path = self.analysis_paths.analysis_dir / "Snakefile.delete"
        snakefile_path.write_text(self._build_delete_sensitivity_snakefile_content(sa_ids))
        return self._submit_delete_snakemake(
            snakefile_path,
            override_multi_sim_run_method=override_multi_sim_run_method,
        )


class SensitivityAnalysisWorkflowBuilder:
    """
    Builder class for generating and executing Snakemake workflows for sensitivity analysis.

    This class handles the unique requirements of sensitivity analysis workflows,
    which involve a hierarchical structure (master analysis → sub-analyses → simulations)
    with multiple consolidation steps. It composes SnakemakeWorkflowBuilder to reuse
    common workflow patterns while adding sensitivity-specific logic.

    Key Features:
    - Generates flattened master Snakefile with all simulation rules
    - Handles dynamic resource allocation per sub-analysis
    - Supports multiple consolidation levels (per-subanalysis + master)
    - Delegates workflow submission to base SnakemakeWorkflowBuilder

    Parameters
    ----------
    sensitivity_analysis : TRITONSWMM_sensitivity_analysis
        The parent sensitivity analysis object containing configuration and sub-analyses
    """

    def __init__(self, sensitivity_analysis: "TRITONSWMM_sensitivity_analysis"):
        """
        Initialize the sensitivity analysis workflow builder.

        Parameters
        ----------
        sensitivity_analysis : TRITONSWMM_sensitivity_analysis
            The parent sensitivity analysis object containing configuration and sub-analyses
        """
        self.sensitivity_analysis = sensitivity_analysis
        self.master_analysis = sensitivity_analysis.master_analysis
        self.system = self.master_analysis._system
        self.analysis_paths = self.master_analysis.analysis_paths
        self.python_executable = self.master_analysis._python_executable
        # Phase 3: unique compile targets (deduplicated by compile-relevant tuple
        # in Phase 1). One Snakemake `rule setup_target_{N}` is emitted per entry
        # so a sensitivity study spanning different gpu_hardware / DEM resolution
        # values compiles once per target rather than once per sub-analysis.
        self.unique_system_targets = sensitivity_analysis.unique_system_targets

        # Compose base workflow builder for common patterns
        self._base_builder = SnakemakeWorkflowBuilder(self.master_analysis)

    def submit_delete_workflow_sensitivity(
        self,
        *,
        override_in_flight: bool = False,
        override_multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks"] | None = None,
    ) -> dict:
        """Submit the distributed sensitivity-master delete workflow.

        Thin facade: derives the ``sa_ids`` list from
        ``self.sensitivity_analysis.df_setup.index`` and delegates to
        :meth:`SnakemakeWorkflowBuilder.submit_delete_workflow_sensitivity`
        on the composed base builder. Pre-delete guards (live-sentinel
        refusal, scoped lock-check, comment-recovery) fire inside the base
        builder's method.

        Per cleanup-rerun-delete-redesign Phase 2.
        """
        sa_ids = self.sensitivity_analysis.df_setup.index.astype(str).tolist()
        return self._base_builder.submit_delete_workflow_sensitivity(
            sa_ids=sa_ids,
            override_in_flight=override_in_flight,
            override_multi_sim_run_method=override_multi_sim_run_method,
        )

    def generate_master_snakefile_content(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        override_clear_raw: ClearRawValue | None = None,
        pickup_where_leftoff: bool = True,
        report_formats: list[str] | None = None,
        alive_by_token: dict[str, str] | None = None,
        alive_token_to_dir: dict[str, str] | None = None,
    ) -> str:
        """
        For sensitivity analyses.

        Generate flattened master Snakefile with individual simulation rules.

        This method generates a single Snakefile with all simulation rules
        flattened directly into it (no nested Snakemake calls). Each simulation
        gets its own rule with exact resource requirements from its sub-analysis config.

        This avoids resource contention issues where sub-analyses with different
        CPU/GPU requirements would fail due to incorrect resource allocation.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process
        compression_level : int
            Compression level for output files (0-9)
        process_system_level_inputs : bool
            If True, process system-level inputs in master setup rule
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM in master setup rule
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, prepare scenarios before running
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after simulations
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw`` (None reads YAML).
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint

        Returns
        -------
        str
            Master Snakefile content
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        # Emit report templates into the master analysis_dir/report/ so the
        # snakemake --report engine can resolve caption= paths.
        _emit_report_artifacts(self.master_analysis.analysis_paths.analysis_dir)

        # Get absolute path to conda environment file using helper
        conda_env_path = self._base_builder._get_conda_env_path()
        master_config_args = self._base_builder._get_config_args(
            analysis_config_yaml=self.master_analysis.analysis_config_yaml
        )

        # Post-F2 (R1): report cfg is inline on cfg_analysis; source the
        # sensitivity benchmarking independent_vars directly from there so
        # the master Snakefile can wildcard the plot rule per independent_var.
        _report_cfg = self.master_analysis.cfg_analysis.report
        _independent_vars: list[str] = (
            list(_report_cfg.sensitivity.independent_vars) if _report_cfg.sensitivity is not None else []
        )
        _group_by_var: str | None = (
            _report_cfg.sensitivity.group_by_var if _report_cfg.sensitivity is not None else None
        )

        # Determine the single enabled model type for sensitivity analysis
        # Sensitivity analysis doesn't support multi-model (would explode parameter space)
        enabled_models = []
        if self.system.cfg_system.toggle_triton_model:
            enabled_models.append("triton")
        if self.system.cfg_system.toggle_tritonswmm_model:
            enabled_models.append("tritonswmm")
        if self.system.cfg_system.toggle_swmm_model:
            enabled_models.append("swmm")

        if len(enabled_models) == 0:
            raise ValueError("No model types enabled in system configuration")
        if len(enabled_models) > 1:
            raise ValueError(
                f"Sensitivity analysis does not support multi-model execution. "
                f"Enabled models: {enabled_models}. Please enable only one model type."
            )

        model_type = enabled_models[0]

        log_dir_str = str(self.master_analysis.analysis_paths.analysis_log_directory)
        master_analysis_id = str(self.master_analysis.cfg_analysis.analysis_id)
        n_sub_analyses = len(self.sensitivity_analysis.sub_analyses)
        # Total scenarios across all sub-analyses (best-effort; matches per-sub-analysis n_sims sum)
        try:
            total_n_sims = sum(len(sub.df_sims) for sub in self.sensitivity_analysis.sub_analyses.values())
        except Exception:
            total_n_sims = n_sub_analyses

        # Compute paired (sa_id, event_id) lists for per-sa per-event plot rules.
        # Used by `_build_plot_rule_block_per_sim_per_sa` and the master `rule all`
        # via `expand(..., zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)`.
        # Event IDs derived via the canonical slug helper used elsewhere in workflow.py.
        sa_event_pairs_sa: list[str] = []
        sa_event_pairs_evt: list[str] = []
        try:
            for sa_id_pair, sub_pair in self.sensitivity_analysis.sub_analyses.items():
                for event_iloc in sub_pair.df_sims.index:
                    ev = sub_pair._retrieve_weather_indexer_using_integer_index(event_iloc)
                    sa_event_pairs_sa.append(str(sa_id_pair))
                    sa_event_pairs_evt.append(compute_event_id_slug(ev))
        except Exception:
            # Best-effort: if any sub-analysis can't materialize event ids, leave the
            # paired lists empty — per-sa per-event plot rules will simply not emit
            # any wildcarded outputs and the master report will skip Per-Simulation panels.
            sa_event_pairs_sa = []
            sa_event_pairs_evt = []

        # Auto-render — modeled as an explicit Snakemake rule (not an `onsuccess:`
        # hook). `onsuccess:` only fires when rules execute; on a fully up-to-date
        # workflow Snakemake exits with `Nothing to be done` and skips the hook,
        # so the report would never get rendered on resume runs. As a rule with
        # `output: analysis_report.{fmt}` and that output added to `rule all`
        # inputs, Snakemake's DAG planner always considers it: skipped when the
        # report is newer than its plot inputs, fired when stale or missing.
        _formats = report_formats if report_formats is not None else ["zip"]

        # Start building the Snakefile
        snakefile_content = f'''# Auto-generated flattened master Snakefile for sensitivity analysis
# Each sub-analysis simulation phase gets its own rule with appropriate resources

import os
from datetime import datetime as _dt
from TRITON_SWMM_toolkit.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("TRITON_SWMM_toolkit")
except Exception:
    _toolkit_version = "unknown"

# Config dict consumed by report_templates/workflow_description.rst.j2
config["analysis_id"] = {master_analysis_id!r}
config["toolkit_version"] = _toolkit_version
config["n_sims"] = {total_n_sims}
config["is_sensitivity"] = True
config["n_sub_analyses"] = {n_sub_analyses}
config["independent_vars"] = {_independent_vars!r}
config["group_by_var"] = {_group_by_var!r}
config["report"] = {{"generated_at": _dt.now().isoformat(timespec="seconds")}}

# Paired (sa_id, event_id) lists for per-sa per-event plot rules.
# Used by `expand(..., zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)`
# in the master `rule all` and in the per-sa per-event plot rule definitions.
SA_EVENT_PAIRS_SA = {sa_event_pairs_sa!r}
SA_EVENT_PAIRS_EVT = {sa_event_pairs_evt!r}

report: "report/workflow_description.rst"

onstart:
    shell("mkdir -p _status {log_dir_str}/sims {log_dir_str}")

# onsuccess: removed — `rule export_scenario_status` (added below) now produces
# scenario_status.csv and workflow_summary.md on the success path via the
# Snakemake DAG.

onerror:
    shell("""
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            {master_config_args} \\
            > {log_dir_str}/export_scenario_status.log 2>&1
    """)


'''

        # Backend-aware extension resolution. The per-rule emission already uses
        # _output_ext_for via __OUTPUT_EXT__ substitution; rule_all (and the
        # render_report subset derived from it below) MUST match or the DAG
        # planner cannot resolve the wildcards. `_ext` is in scope for every
        # subsequent `rule_all_inputs.append(...)` site below, including those
        # nested inside the `if sa_event_pairs_sa:` and `if _independent_vars:`
        # blocks (Python's nested-block scoping rules).
        _ext = _resolve_rule_all_extensions(self._base_builder._get_report_cfg_static_backend())

        # Build the rule all with all dependencies
        consolidation_flags = []
        for sa_id in self.sensitivity_analysis.sub_analyses.keys():  # type: ignore
            consolidation_flags.append(
                f"_status/e_consolidate_sa-{sa_id}_complete.flag"  # type: ignore
            )

        # Phase 3: per-target setup flags. Listed explicitly in rule_all_inputs so
        # the DAG planner can reach setup_target rules even for sub-analyses whose
        # df_sims is empty (otherwise only reachable via transitive deps through
        # prepare_sa rules).
        setup_target_flags = [
            f"_status/a_setup_target_{target.target_id}_complete.flag" for target in self.unique_system_targets
        ]
        # sa_id (str) → target_id reverse lookup used when emitting per-SA rule
        # dependencies. Built once per Snakefile generation. Keys are coerced to
        # str to match sub_analyses dict iteration keys regardless of source type.
        sa_id_to_target_id: dict[str, int] = {
            str(sa_id): target.target_id for target in self.unique_system_targets for sa_id in target.sub_analysis_ids
        }

        rule_all_inputs = [f'"{flag}"' for flag in setup_target_flags]
        rule_all_inputs.extend(f'"{flag}"' for flag in consolidation_flags)
        rule_all_inputs.append('"_status/f_consolidate_master_complete.flag"')
        # System-overview at master scope: the DEM and SWMM topology are shared
        # across sub-analyses, so a single system_overview.png in the master
        # report is the natural place to surface them. Per-analysis summary at
        # master scope renders one row per sub-analysis (Iteration 6 "show all
        # sub-analyses" scope). Per-sim plots wildcarded over (sa_id, event_id)
        # pairs (Iteration 7 Change 3b — "show all" panel parity per the user's
        # scope expansion: identical-looking panels across sub-analyses are a QC
        # signal; expected variation is also visible).
        rule_all_inputs.append(f'"plots/system_overview{_ext["system_overview"]}"')
        rule_all_inputs.append(f'"plots/per_analysis/summary_table{_ext["per_analysis_summary"]}"')
        rule_all_inputs.append(f'"plots/appendix/scenario_status{_ext["scenario_status_appendix"]}"')
        rule_all_inputs.append(f'"plots/errors_and_warnings/validation_report{_ext["errors_and_warnings"]}"')
        rule_all_inputs.append(f'"plots/disk_utilization{_ext["disk_utilization"]}"')
        rule_all_inputs.append('"scenario_status.csv"')
        rule_all_inputs.append('"workflow_summary.md"')

        if sa_event_pairs_sa:
            _e_pfd = _ext["per_sim_per_sa_peak_flood_depth"]
            _e_cf = _ext["per_sim_per_sa_conduit_flow"]
            rule_all_inputs.append(
                f'expand("plots/sensitivity/per_sim/sa-{{sa_id}}/{{event_id}}/peak_flood_depth{_e_pfd}", '
                "zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)"
            )
            rule_all_inputs.append(
                f'expand("plots/sensitivity/per_sim/sa-{{sa_id}}/{{event_id}}/conduit_flow{_e_cf}", '
                "zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)"
            )
        if _independent_vars:
            _e_bench = _ext["sensitivity_benchmarking"]
            rule_all_inputs.append(
                f'expand("plots/sensitivity/benchmarking/{{independent_var}}_vs_total{_e_bench}", '
                f"independent_var={_independent_vars!r})"
            )
        # Snapshot the plot-input list before appending the report targets — the
        # render rule uses this same set as its `input:` so Snakemake's DAG
        # planner re-fires the render whenever any plot output is newer than
        # the existing report.
        render_rule_input_items = [item for item in rule_all_inputs if not item.startswith('"_status/')]
        # Auto-render: every requested format is a top-level target. Snakemake's
        # DAG planning then always considers each report output and skips iff
        # the output is current relative to its plot inputs.
        for _fmt in _formats:
            rule_all_inputs.append(f'"analysis_report.{_fmt}"')

        snakefile_content += f"""rule all:
    input:
        {", ".join(rule_all_inputs)}

"""

        # Phase 3: emit one setup rule per unique compile target. For a sensitivity
        # study that varies gpu_hardware or target_dem_resolution across sub-analyses,
        # this materializes the per-target compile DAG without redundant compilation.
        # Backward-compat: a study with no `system_config_yaml` column (or all rows
        # collapsing to one target) yields exactly one rule (`setup_target_0`).
        for target in self.unique_system_targets:
            target_config_args = self._base_builder._get_config_args(
                analysis_config_yaml=self.master_analysis.analysis_config_yaml,
                system_config_yaml=target.system_config_yaml,
            )
            target_cfg_system = target.system.cfg_system
            snakefile_content += f'''rule setup_target_{target.target_id}:
    output: "_status/a_setup_target_{target.target_id}_complete.flag"
    log: "{log_dir_str}/setup_target_{target.target_id}.log"
    conda: "{conda_env_path}"
    resources:
{
                self._base_builder._build_resource_block(
                    partition=self.master_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                    runtime_min=self.master_analysis.cfg_analysis.hpc_runtime_min_for_setup,
                    mem_mb=self.master_analysis.cfg_analysis.hpc_mem_allocation_for_setup_mb,
                    nodes=1,
                    tasks=1,
                    cpus_per_task=1,
                )
            }
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.setup_workflow \\
            {target_config_args} \\
            {"--process-system-inputs " if process_system_level_inputs else ""}\\
            {"--overwrite-system-inputs " if overwrite_system_inputs else ""}\\
            {"--compile-triton-swmm " if compile_TRITON_SWMM and target_cfg_system.toggle_tritonswmm_model else ""}\\
            {"--compile-triton-only " if compile_TRITON_SWMM and target_cfg_system.toggle_triton_model else ""}\\
            {"--compile-swmm " if compile_TRITON_SWMM and target_cfg_system.toggle_swmm_model else ""}\\
            {"--recompile-if-already-done " if recompile_if_already_done_successfully else ""}\\
            --flag-output {{output}} \\
            --rule-name setup_target_{target.target_id} \\
            --target-id {target.target_id} \\
            > {{log}} 2>&1
        """

'''

        # Write per-sa_id input fingerprint files (compare-and-write).
        # Per-sa_id input fingerprint files declared as `input:` of every
        # sub-analysis rule. When a row in the parent sensitivity CSV/Excel
        # changes (any independent_vars column value), the corresponding
        # fingerprint file's content changes → its mtime bumps → Snakemake's
        # mtime trigger re-runs only that sa_id's rule chain.
        status_dir = self.master_analysis.analysis_paths.analysis_dir / "_status"
        status_dir.mkdir(parents=True, exist_ok=True)
        for sa_id, sub_analysis in self.sensitivity_analysis.sub_analyses.items():  # type: ignore
            fingerprint_path = status_dir / f"sa-{sa_id}_inputs.json"
            self.sensitivity_analysis._write_sa_id_fingerprint(sub_analysis, fingerprint_path)
        if len(self.sensitivity_analysis.independent_vars) == 0:
            print(
                "[Sensitivity] WARNING: independent_vars is empty; sensitivity-row-edit rerun trigger is a no-op",
                flush=True,
            )

        # Generate simulation rules for each sub-analysis
        subanalysis_flags = []
        for sa_id, sub_analysis in self.sensitivity_analysis.sub_analyses.items():  # type: ignore
            # Extract resource requirements from sub-analysis config
            n_mpi = sub_analysis.cfg_analysis.n_mpi_procs or 1
            n_omp = sub_analysis.cfg_analysis.n_omp_threads or 1
            n_gpus = sub_analysis.cfg_analysis.n_gpus or 0
            n_nodes = sub_analysis.cfg_analysis.n_nodes or 1
            hpc_time = sub_analysis.cfg_analysis.hpc_time_min_per_sim or 30
            mem_per_cpu = sub_analysis.cfg_analysis.mem_gb_per_cpu or 2
            gpus_per_node_config = sub_analysis.cfg_analysis.hpc_gpus_per_node or 0
            cpus_per_sim = n_mpi * n_omp
            run_mode = sub_analysis.cfg_analysis.run_mode

            sub_config_args = self._base_builder._get_config_args(
                analysis_config_yaml=sub_analysis.analysis_config_yaml,
                system_config_yaml=sub_analysis._system.system_config_yaml,
            )

            # Phase 3: per-SA system config sources gpu_alloc_mode + gpu_hw so a
            # sensitivity study spanning UVA (gres) and Frontier (gpus) emits the
            # correct SLURM directive per sub-analysis.
            gpu_alloc_mode = sub_analysis._system.cfg_system.preferred_slurm_option_for_allocating_gpus or "gpus"

            # Setup-target flag this sub-analysis depends on (Phase 3). Keyed by
            # str(sa_id) to match the map built before the rule_all section.
            setup_target_flag = f"_status/a_setup_target_{sa_id_to_target_id[str(sa_id)]}_complete.flag"

            # Build resource blocks for this sub-analysis
            prep_resources_sa = self._base_builder._build_resource_block(
                partition=sub_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=30,
                mem_mb=mem_per_cpu * 1000,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )

            # The SLURM executor maps the `tasks` resource to --ntasks (non-GPU) or
            # --ntasks-per-gpu (gres-GPU), and `threads` to --cpus-per-task only. `threads`
            # never governs --ntasks. Verified against snakemake-executor-plugin-slurm
            # v2.0.3 submit_string.py:79-128. snakemake_threads drives the Snakemake
            # scheduler's local concurrency accounting and the --cpus-per-task fallback.
            snakemake_threads = cpus_per_sim

            # gpu_hardware comes directly from the per-target cfg_system. Under the
            # prefixed-column overlay mechanism, `system.gpu_hardware` overlay values
            # already populated this field via the synthesized per-target YAML.
            gpu_hw = sub_analysis._system.cfg_system.gpu_hardware
            sim_resources_sa = self._base_builder._build_resource_block(
                partition=sub_analysis.cfg_analysis.hpc_ensemble_partition,
                runtime_min=hpc_time,
                mem_mb=int(mem_per_cpu * n_mpi * n_omp * 1000),
                nodes=n_nodes,
                tasks=n_mpi,
                cpus_per_task=n_omp,
                gpus_total=n_gpus,
                gpus_per_node_config=gpus_per_node_config,
                gpu_hardware=gpu_hw,
                gpu_alloc_mode=gpu_alloc_mode,
                mpi=(run_mode in ["hybrid", "mpi"]),
            )

            process_resources_sa = self._base_builder._build_resource_block(
                partition=sub_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=120,
                mem_mb=sub_analysis.cfg_analysis.hpc_mem_allocation_for_sim_output_processing_mb,
                nodes=1,
                tasks=1,
                cpus_per_task=2,
            )

            # For each simulation in this sub-analysis
            sub_analysis_sim_flags = []
            for event_iloc in sub_analysis.df_sims.index:
                event_id = compute_event_id_slug(sub_analysis._retrieve_weather_indexer_using_integer_index(event_iloc))
                # Rule names must be valid Python identifiers (no `.`, `-`).
                # Flag paths keep the hyphen-delimited format for wildcard parsing.
                sa_id_rule = str(sa_id).replace(".", "_").replace("-", "_")
                event_id_rule = event_id.replace(".", "_").replace("-", "_")
                # Phase 1: Scenario preparation (if enabled)
                if prepare_scenarios:
                    prep_rule_name = f"prepare_sa_{sa_id_rule}_evt_{event_id_rule}"
                    prep_outflag = f"_status/b_prepare_sa-{sa_id}_evt-{event_id}_complete.flag"

                    snakefile_content += f'''rule {prep_rule_name}:
    input:
        "{setup_target_flag}",
        "_status/sa-{sa_id}_inputs.json"
    output: "{prep_outflag}"
    log: "{log_dir_str}/sims/{prep_rule_name}.log"
    conda: "{conda_env_path}"
    resources:
{prep_resources_sa}
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.prepare_scenario_runner \\
            --event-iloc {event_iloc} \\
            {sub_config_args} \\
            {"--overwrite-scenario-if-already-set-up " if overwrite_scenario_if_already_set_up else ""}\\
            {"--rerun-swmm-hydro " if rerun_swmm_hydro_if_outputs_exist else ""}\\
            --flag-output {{output}} \\
            --rule-name {prep_rule_name} \\
            --sa-id {sa_id} \\
            --event-id {event_id} \\
            > {{log}} 2>&1
        """

'''

                # Phase 2: Simulation execution
                sim_rule_name = f"simulation_sa_{sa_id_rule}_evt_{event_id_rule}"
                sim_outflag = f"_status/c_run_{model_type}_sa-{sa_id}_evt-{event_id}_complete.flag"
                upstream_flag = prep_outflag if prepare_scenarios else setup_target_flag

                # v2 graceful-rerun: if this scenario is in the alive set, emit a
                # wait-rule instead of the run-rule. The sensitivity sentinel
                # rule_token convention (run_simulation_runner.py) is
                # ``simulation_sa_{sa_id}_evt-{event_id}`` (literal hyphen). The
                # wait-rule's ``--analysis-dir`` MUST be the per-sub-analysis dir
                # (markers live under master/subanalyses/{id}/_status/, NOT the
                # master dir — Spec D). Distinct concrete rule names mean no
                # ``ruleorder`` is needed (unlike the multisim wildcard path).
                _sentinel_token = f"simulation_sa_{sa_id}_evt-{event_id}"
                if _sentinel_token in (alive_by_token or {}):
                    snakefile_content += self._base_builder._emit_wait_for_sim_rule_block(
                        rule_token=_sentinel_token,
                        flag_output_path=sim_outflag,
                        run_rule_inputs=[upstream_flag, f"_status/sa-{sa_id}_inputs.json"],
                        wait_walltime_cap_min=min(
                            _max_plausible_job_lifetime_min(sub_analysis.cfg_analysis),
                            sub_analysis.cfg_analysis.hpc_max_wait_for_inflight_min,
                        ),
                        analysis_dir_override=(alive_token_to_dir or {}).get(
                            _sentinel_token, str(sub_analysis.analysis_paths.analysis_dir)
                        ),
                    )
                else:
                    snakefile_content += f'''rule {sim_rule_name}:
    input:
        "{upstream_flag}",
        "_status/sa-{sa_id}_inputs.json"
    output: "{sim_outflag}"
    log: "{log_dir_str}/sims/{sim_rule_name}.log"
    conda: "{conda_env_path}"
    threads: {snakemake_threads}
    resources:
{sim_resources_sa}
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.run_simulation_runner \\
            --event-iloc {event_iloc} \\
            {sub_config_args} \\
            --model-type {model_type} \\
            --sa-id {sa_id} \\
            {"--pickup-where-leftoff " if pickup_where_leftoff else ""}\\
            --flag-output {{output}} \\
            --rule-name {sim_rule_name} \\
            --event-id {event_id} \\
            > {{log}} 2>&1
        """

'''

                # Phase 3: Output processing (if enabled)
                if process_timeseries:
                    process_rule_name = f"process_sa_{sa_id_rule}_evt_{event_id_rule}"
                    process_outflag = f"_status/d_process_{model_type}_sa-{sa_id}_evt-{event_id}_complete.flag"

                    snakefile_content += f'''rule {process_rule_name}:
    input:
        "{sim_outflag}",
        "_status/sa-{sa_id}_inputs.json"
    output: "{process_outflag}"
    log: "{log_dir_str}/sims/{process_rule_name}.log"
    conda: "{conda_env_path}"
    resources:
{process_resources_sa}
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.process_timeseries_runner \\
            --event-iloc {event_iloc} \\
            {sub_config_args} \\
            --model-type {model_type} \\
            --which {which} \\
            {f"--override-clear-raw '{json.dumps(override_clear_raw)}' " if override_clear_raw is not None else ""}\\
            --compression-level {compression_level} \\
            --flag-output {{output}} \\
            --rule-name {process_rule_name} \\
            --sa-id {sa_id} \\
            --event-id {event_id} \\
            > {{log}} 2>&1
        """

'''
                    final_flag = process_outflag
                else:
                    final_flag = sim_outflag

                sub_analysis_sim_flags.append(final_flag)

            subanalysis_flag = f"_status/e_consolidate_sa-{sa_id}_complete.flag"
            subanalysis_flags.append(subanalysis_flag)

            # Consolidate outputs after all sims have been run. Sanitize for
            # use as a Snakemake rule identifier.
            prefix = self.sensitivity_analysis.sub_analyses_prefix  # type: ignore
            consolidate_inputs = [f'"{flag}"' for flag in sub_analysis_sim_flags]
            consolidate_inputs.append(f'"_status/sa-{sa_id}_inputs.json"')
            snakefile_content += f'''rule consolidate_{prefix}{sa_id_rule}:
    input: {", ".join(consolidate_inputs)}
    output: "{subanalysis_flag}"
    log: "{log_dir_str}/sims/consolidate_{prefix}{sa_id}.log"
    conda: "{conda_env_path}"
    resources:
{
                self._base_builder._build_resource_block(
                    partition=sub_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                    runtime_min=30,
                    mem_mb=sub_analysis.cfg_analysis.hpc_mem_allocation_for_sim_output_processing_mb,
                    nodes=1,
                    tasks=1,
                    cpus_per_task=1,
                )
            }
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {sub_config_args} \\
            --which {which} \\
            --compression-level {compression_level} \\
            --flag-output {{output}} \\
            --rule-name consolidate_{prefix}{sa_id_rule} \\
            --sa-id {sa_id} \\
            > {{log}} 2>&1
        """

'''

        # Generate master consolidation rule
        snakefile_content += f'''rule master_consolidation:
    input: {", ".join([f'"{flag}"' for flag in subanalysis_flags])}
    output: "_status/f_consolidate_master_complete.flag"
    log: "{log_dir_str}/master_consolidation.log"
    conda: "{conda_env_path}"
    resources:
{
            self._base_builder._build_resource_block(
                partition=self.master_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=30,
                mem_mb=sub_analysis.cfg_analysis.hpc_mem_allocation_for_analysis_output_consolidation_mb,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )
        }
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {master_config_args} \\
            --consolidate-sensitivity-analysis-outputs \\
            --which {which} \\
            --compression-level {compression_level} \\
            --flag-output {{output}} \\
            --rule-name master_consolidation \\
            > {{log}} 2>&1
        """
'''

        # Append system_overview + per_analysis_summary rules at master scope (match rule_all above).
        # Master uses f_consolidate_master_complete.flag (NOT the multisim e_consolidate_complete flag).
        snakefile_content += self._base_builder._build_plot_rule_block_system_overview(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_per_analysis_summary(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_scenario_status_appendix(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_errors_and_warnings(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_disk_utilization(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_export_scenario_status_rule(
            input_flag="_status/f_consolidate_master_complete.flag",
        )
        # Per-sa per-event plot rules (Iteration 7 Change 3b — "show all" parity).
        # Only emit when sa_event_pairs are populated (best-effort guarded above).
        if sa_event_pairs_sa:
            snakefile_content += self._build_plot_rule_block_per_sim_per_sa()

        if _independent_vars:
            snakefile_content += self._build_plot_rule_block_sensitivity_benchmarking(_independent_vars)

        # Render-report rule (replaces the broken onsuccess auto-render approach).
        # Single rule wildcarded on `format` — Snakemake fires it once per
        # `analysis_report.{fmt}` target listed in `rule all`.
        render_inputs_str = ",\n        ".join(render_rule_input_items)
        snakefile_content += f'''
rule render_report:
    input:
        {render_inputs_str}
    output:
        "analysis_report.{{format}}"
    wildcard_constraints:
        format="zip|html"
    log: "{log_dir_str}/render_report_{{format}}.log"
    resources:
{
            self._base_builder._build_resource_block(
                partition=self.master_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=30,
                mem_mb=self.master_analysis.cfg_analysis.mem_gb_per_cpu * 1000,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )
        }
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.render_report_runner \\
            {master_config_args} \\
            --format {{wildcards.format}} \\
            > {{log}} 2>&1
        """
'''

        return snakefile_content

    def generate_reprocess_master_snakefile_content(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        report_formats: list[str] | None = None,
        start_with: Literal["process", "consolidate", "render"] = "consolidate",
    ) -> str:
        """Generate a reprocess-scoped master Snakefile for the sensitivity master.

        Sibling to :meth:`generate_master_snakefile_content` — keeps the full
        master generator stable while emitting only the downstream rules needed
        for a master-level reprocess:

        - ``rule consolidate_{prefix}{sa_id}`` for each sub-analysis WITH AT
          LEAST ONE COMPLETED SIM (filtered by ``c_run_*`` flag existence on
          disk). Sub-analyses with zero completed sims are silently excluded
          from the DAG — no per-sa consolidate rule emitted, no fingerprint
          fallback input. Per-sub-analysis resources are sourced from the
          sub-analysis's own ``cfg_analysis`` (honoring per-row overrides for
          ``hpc_mem_allocation_for_sim_output_processing_mb`` and partition).
        - Under ``start_with='process'``: for each (sa_id, event_id) whose
          ``c_run_*`` flag exists AND whose ``d_process_*`` flag does NOT
          exist on disk, emit a per-sa per-event ``rule process_{model_type}_sa_{sa_id}_evt_{event_id}``
          mirroring the canonical sensitivity workflow generator's process
          emit (workflow.py:4920-4954). The per-sa consolidate's input list
          dynamically mixes ``d_process_*`` flags (newly-emitted process rules)
          and ``c_run_*`` flags (events already processed). The per-sa
          consolidate shell appends ``--allow-incomplete`` ONLY when at least
          one event in this sub-analysis went through the conditional process
          emit (truly mixed state); fully-complete sub-analyses retain the
          consolidate runner's existing fail-fast behavior. Under
          ``start_with='consolidate'`` or ``'render'``: no process rules
          emitted; consolidate consumes ``c_run_*`` flags directly and
          ``--allow-incomplete`` is not added (fail-fast preserved).
        - ``rule master_consolidation`` aggregating the per-sa flags into
          ``f_consolidate_master_complete.flag`` (overwrite + allow-incomplete
          baked).
        - The full plot + ``export_scenario_status`` + ``render_report`` rules,
          reusing the same helpers as the production master Snakefile so the
          rendered report is identical.

        The reprocess driver invokes this Snakefile via
        ``submit_reprocess_workflow``-equivalent dispatch in
        :meth:`TRITONSWMM_sensitivity_analysis.reprocess` against the same
        ``analysis_dir/.snakemake/`` lock dir as the full master Snakefile.
        Per-Snakefile locking (distinct ``Snakefile`` vs ``Snakefile.reprocess``
        paths) preserves coexistence safety for local tests + CLI smoke; true
        coexistence with a concurrent live ``rule simulation_*`` driver is
        tracked as a Phase 2 follow-up (see ``submit_reprocess_workflow``'s
        body comment).

        Parameters
        ----------
        which
            ``"both"`` / ``"TRITON"`` / ``"SWMM"`` — threaded into the
            consolidate rule shells' ``--which`` flag.
        compression_level
            Compression level (0-9) for the consolidate rule shells.
        report_formats
            List of report formats to render; defaults to ``["zip"]``.
        start_with
            Stage to re-fire from. ``"process"`` enables conditional per-sa
            per-event process_timeseries emission (Option C: only emit for
            (sa_id, event_id) pairs whose d_process flag is missing).
            ``"consolidate"`` and ``"render"`` skip process emission entirely
            and the per-sa consolidate consumes c_run flags directly.
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        # Emit report templates into master analysis_dir/report/ so the
        # snakemake --report engine can resolve caption= paths. Mirrors the
        # full master generator.
        _emit_report_artifacts(self.master_analysis.analysis_paths.analysis_dir)

        conda_env_path = self._base_builder._get_conda_env_path()
        master_config_args = self._base_builder._get_config_args(
            analysis_config_yaml=self.master_analysis.analysis_config_yaml
        )

        # Sensitivity benchmarking + per-sim plotting context (sourced same as
        # the full master generator so render targets match).
        _report_cfg = self.master_analysis.cfg_analysis.report
        _independent_vars: list[str] = (
            list(_report_cfg.sensitivity.independent_vars) if _report_cfg.sensitivity is not None else []
        )
        _group_by_var: str | None = (
            _report_cfg.sensitivity.group_by_var if _report_cfg.sensitivity is not None else None
        )

        # Single-enabled-model contract (sensitivity does not support multi-model).
        enabled_models = []
        if self.system.cfg_system.toggle_triton_model:
            enabled_models.append("triton")
        if self.system.cfg_system.toggle_tritonswmm_model:
            enabled_models.append("tritonswmm")
        if self.system.cfg_system.toggle_swmm_model:
            enabled_models.append("swmm")
        if len(enabled_models) == 0:
            raise ValueError("No model types enabled in system configuration")
        if len(enabled_models) > 1:
            raise ValueError(
                f"Sensitivity analysis does not support multi-model execution. "
                f"Enabled models: {enabled_models}. Please enable only one model type."
            )
        model_type = enabled_models[0]

        log_dir_str = str(self.master_analysis.analysis_paths.analysis_log_directory)
        master_analysis_id = str(self.master_analysis.cfg_analysis.analysis_id)
        n_sub_analyses = len(self.sensitivity_analysis.sub_analyses)
        try:
            total_n_sims = sum(len(sub.df_sims) for sub in self.sensitivity_analysis.sub_analyses.values())
        except Exception:
            total_n_sims = n_sub_analyses

        # Paired (sa_id, event_id) lists for per-sa per-event plot rules.
        # Option C invariant: only include (sa_id, event_id) pairs whose
        # c_run_* flag exists on disk. Excluded sub-analyses' events would
        # otherwise trigger per-sim plot rules whose renderer reads
        # non-existent per-sa scenario data (the plot rule's only input is
        # the master consolidation flag, so Snakemake would schedule them
        # unconditionally and they would fail at render time).
        from TRITON_SWMM_toolkit.constants import sim_run_flag_per_sa

        sa_event_pairs_sa: list[str] = []
        sa_event_pairs_evt: list[str] = []
        analysis_dir_for_pairs = self.master_analysis.analysis_paths.analysis_dir
        try:
            for sa_id_pair, sub_pair in self.sensitivity_analysis.sub_analyses.items():
                for event_iloc in sub_pair.df_sims.index:
                    ev = sub_pair._retrieve_weather_indexer_using_integer_index(event_iloc)
                    event_id_pair = compute_event_id_slug(ev)
                    c_run_flag_path = analysis_dir_for_pairs / sim_run_flag_per_sa(
                        model_type, str(sa_id_pair), event_id_pair
                    )
                    if not c_run_flag_path.exists():
                        continue
                    sa_event_pairs_sa.append(str(sa_id_pair))
                    sa_event_pairs_evt.append(event_id_pair)
        except Exception:
            sa_event_pairs_sa = []
            sa_event_pairs_evt = []

        _formats = report_formats if report_formats is not None else ["zip"]

        # Snakefile preamble — config dict + onstart/onerror hooks identical to
        # the production master generator so the rendered report's metadata
        # surface is unchanged on reprocess.
        snakefile_content = f'''# Auto-generated reprocess-scoped master Snakefile for sensitivity analysis
# Re-runs downstream stages (per-sa consolidate + master consolidation +
# plots + render) against existing per-sa sim completion flags. No
# simulation or scenario-preparation rules are emitted.

import os
from datetime import datetime as _dt
from TRITON_SWMM_toolkit.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("TRITON_SWMM_toolkit")
except Exception:
    _toolkit_version = "unknown"

config["analysis_id"] = {master_analysis_id!r}
config["toolkit_version"] = _toolkit_version
config["n_sims"] = {total_n_sims}
config["is_sensitivity"] = True
config["n_sub_analyses"] = {n_sub_analyses}
config["independent_vars"] = {_independent_vars!r}
config["group_by_var"] = {_group_by_var!r}
config["report"] = {{"generated_at": _dt.now().isoformat(timespec="seconds")}}

SA_EVENT_PAIRS_SA = {sa_event_pairs_sa!r}
SA_EVENT_PAIRS_EVT = {sa_event_pairs_evt!r}

report: "report/workflow_description.rst"

onstart:
    shell("mkdir -p _status {log_dir_str}/sims {log_dir_str}")

onerror:
    shell("""
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            {master_config_args} \\
            > {log_dir_str}/export_scenario_status.log 2>&1
    """)


'''

        _ext = _resolve_rule_all_extensions(self._base_builder._get_report_cfg_static_backend())

        # rule all — mirrors the production master generator's render-target
        # set (system_overview, per_analysis_summary, scenario_status_appendix,
        # errors_and_warnings, scenario_status.csv, workflow_summary.md, per-sa
        # per-event plots, sensitivity-benchmarking plots, analysis_report.*).
        # Determine the set of sub-analyses that will actually emit per-sa
        # consolidate rules — Option C invariant: only sub-analyses with at
        # least one c_run_* flag on disk get a per-sa consolidate rule (the
        # per-sa loop further down enforces this with the same predicate).
        # rule_all's consolidation_flags list initializes from THIS filtered
        # set, not the full sensitivity-definition set. Snakemake's DAG
        # planner would otherwise complain about missing inputs for the
        # consolidate rules we never emit for un-completed sub-analyses.
        from TRITON_SWMM_toolkit.constants import (
            consolidate_subanalysis_flag,
        )
        from TRITON_SWMM_toolkit.constants import (
            sim_run_flag_per_sa as _sim_run_flag_per_sa,
        )

        _analysis_dir_for_consolidation_flags = self.master_analysis.analysis_paths.analysis_dir
        completed_sa_ids: list[str] = []
        for sa_id_check, sub_check in self.sensitivity_analysis.sub_analyses.items():
            for event_iloc_check in sub_check.df_sims.index:
                event_id_check = compute_event_id_slug(
                    sub_check._retrieve_weather_indexer_using_integer_index(event_iloc_check)
                )
                c_run_flag_check = _sim_run_flag_per_sa(model_type, str(sa_id_check), event_id_check)
                if (_analysis_dir_for_consolidation_flags / c_run_flag_check).exists():
                    completed_sa_ids.append(str(sa_id_check))
                    break
        consolidation_flags = [consolidate_subanalysis_flag(sa_id) for sa_id in completed_sa_ids]
        rule_all_inputs = [f'"{flag}"' for flag in consolidation_flags]
        rule_all_inputs.append('"_status/f_consolidate_master_complete.flag"')
        rule_all_inputs.append(f'"plots/system_overview{_ext["system_overview"]}"')
        rule_all_inputs.append(f'"plots/per_analysis/summary_table{_ext["per_analysis_summary"]}"')
        rule_all_inputs.append(f'"plots/appendix/scenario_status{_ext["scenario_status_appendix"]}"')
        rule_all_inputs.append(f'"plots/errors_and_warnings/validation_report{_ext["errors_and_warnings"]}"')
        rule_all_inputs.append(f'"plots/disk_utilization{_ext["disk_utilization"]}"')
        rule_all_inputs.append('"scenario_status.csv"')
        rule_all_inputs.append('"workflow_summary.md"')
        if sa_event_pairs_sa:
            _e_pfd = _ext["per_sim_per_sa_peak_flood_depth"]
            _e_cf = _ext["per_sim_per_sa_conduit_flow"]
            rule_all_inputs.append(
                f'expand("plots/sensitivity/per_sim/sa-{{sa_id}}/{{event_id}}/peak_flood_depth{_e_pfd}", '
                "zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)"
            )
            rule_all_inputs.append(
                f'expand("plots/sensitivity/per_sim/sa-{{sa_id}}/{{event_id}}/conduit_flow{_e_cf}", '
                "zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)"
            )
        if _independent_vars:
            _e_bench = _ext["sensitivity_benchmarking"]
            rule_all_inputs.append(
                f'expand("plots/sensitivity/benchmarking/{{independent_var}}_vs_total{_e_bench}", '
                f"independent_var={_independent_vars!r})"
            )
        render_rule_input_items = [item for item in rule_all_inputs if not item.startswith('"_status/')]
        for _fmt in _formats:
            rule_all_inputs.append(f'"analysis_report.{_fmt}"')

        snakefile_content += f"""rule all:
    input:
        {", ".join(rule_all_inputs)}

"""

        # Per-sub-analysis emit block — Option C (2026-05-21):
        #   1. Filter sub-analyses to those with at least one c_run_* flag on
        #      disk (Issue A fix: skip emit entirely for un-completed
        #      sub-analyses, no fingerprint fallback).
        #   2. Under start_with='process': for each (sa_id, event_id) with
        #      c_run flag present but d_process flag missing, emit a per-sa
        #      per-event process_timeseries rule mirroring the canonical
        #      sensitivity workflow generator (workflow.py:4920-4954).
        #   3. Per-sa consolidate input list mixes d_process flags (newly-
        #      emitted process rules) and c_run flags (already-processed
        #      events). Per-sa consolidate shell appends --allow-incomplete
        #      ONLY when this sub-analysis has at least one event that went
        #      through the conditional process emit (truly mixed state);
        #      fully-complete sub-analyses retain fail-fast behavior.
        #   4. Per-sub-analysis process_resources_sa is computed INSIDE the
        #      loop so per-row cfg overrides (e.g., sa-33's bumped
        #      hpc_mem_allocation_for_sim_output_processing_mb for 1.1m DEM)
        #      are honored — mirrors canonical generator at workflow.py:4850.
        # Flag-name builders live in TRITON_SWMM_toolkit.constants (single
        # source of truth for new code; existing hardcoded sites are
        # tracked as a follow-up refactor).
        from TRITON_SWMM_toolkit.constants import (
            consolidate_subanalysis_flag,
            process_timeseries_flag_per_sa,
            sa_inputs_fingerprint_flag,
            sim_run_flag_per_sa,
        )

        analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        subanalysis_flags: list[str] = []

        for sa_id, sub_analysis in self.sensitivity_analysis.sub_analyses.items():
            sa_id_rule = str(sa_id).replace(".", "_").replace("-", "_")
            sub_config_args = self._base_builder._get_config_args(
                analysis_config_yaml=sub_analysis.analysis_config_yaml,
                system_config_yaml=sub_analysis._system.system_config_yaml,
            )
            # Per-sub-analysis process resources sourced from THIS sub-analysis's
            # cfg (mirrors canonical sensitivity workflow at workflow.py:4850);
            # honors per-sub-analysis overrides via the `analysis.*` overlay
            # column convention.
            process_resources_sa = self._base_builder._build_resource_block(
                partition=sub_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=120,
                mem_mb=sub_analysis.cfg_analysis.hpc_mem_allocation_for_sim_output_processing_mb,
                nodes=1,
                tasks=1,
                cpus_per_task=2,
            )
            # Walk this sub-analysis's events, classifying each by on-disk
            # c_run / d_process flag presence.
            consolidate_inputs: list[str] = []
            per_event_process_rules: list[str] = []
            had_conditional_process_emit = False
            for event_iloc in sub_analysis.df_sims.index:
                event_id = compute_event_id_slug(sub_analysis._retrieve_weather_indexer_using_integer_index(event_iloc))
                c_run_flag = sim_run_flag_per_sa(model_type, str(sa_id), event_id)
                d_process_flag = process_timeseries_flag_per_sa(model_type, str(sa_id), event_id)
                c_run_path = analysis_dir / c_run_flag
                d_process_path = analysis_dir / d_process_flag
                if not c_run_path.exists():
                    # Sim never completed → exclude from DAG for this reprocess.
                    continue
                if start_with == "process" and not d_process_path.exists():
                    # Conditional process emit: this (sa_id, event_id) needs
                    # process_timeseries re-fired. Build the rule and route
                    # consolidate's input through the d_process flag it will
                    # produce.
                    had_conditional_process_emit = True
                    event_id_rule = event_id.replace(".", "_").replace("-", "_")
                    process_rule_name = f"process_sa_{sa_id_rule}_evt_{event_id_rule}"
                    fingerprint_flag = sa_inputs_fingerprint_flag(str(sa_id))
                    per_event_process_rules.append(f'''rule {process_rule_name}:
    input:
        "{c_run_flag}",
        "{fingerprint_flag}"
    output: "{d_process_flag}"
    log: "{log_dir_str}/sims/{process_rule_name}.log"
    conda: "{conda_env_path}"
    resources:
{process_resources_sa}
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.process_timeseries_runner \\
            --event-iloc {event_iloc} \\
            {sub_config_args} \\
            --model-type {model_type} \\
            --which {which} \\
            --compression-level {compression_level} \\
            --flag-output {{output}} \\
            --rule-name {process_rule_name} \\
            --sa-id {sa_id} \\
            --event-id {event_id} \\
            > {{log}} 2>&1
        """

''')
                    consolidate_inputs.append(f'"{d_process_flag}"')
                else:
                    # Either start_with is not 'process', or d_process flag
                    # already exists — consolidate consumes the c_run flag
                    # directly.
                    consolidate_inputs.append(f'"{c_run_flag}"')

            if not consolidate_inputs:
                # Issue A fix: this sub-analysis has zero completed sims;
                # skip emitting the per-sa consolidate rule entirely. No
                # fingerprint fallback. Spec 5's rule_all reconciliation
                # excludes this sa_id from rule_all's input list.
                continue

            # Emit any per-(sa, event) process rules first so they sit above
            # the per-sa consolidate that depends on them (Snakemake rule
            # order does not affect DAG resolution but readability puts
            # producers above consumers).
            for process_rule_block in per_event_process_rules:
                snakefile_content += process_rule_block

            # Conditional --allow-incomplete: only emit when truly mixed
            # state (option (a) from the SE specialist's plan review).
            # Fully-complete sub-analyses retain consolidate's fail-fast.
            allow_incomplete_line = "            --allow-incomplete \\\n" if had_conditional_process_emit else ""
            subanalysis_flag = consolidate_subanalysis_flag(str(sa_id))
            subanalysis_flags.append(subanalysis_flag)
            prefix = self.sensitivity_analysis.sub_analyses_prefix
            snakefile_content += f'''rule consolidate_{prefix}{sa_id_rule}:
    input: {", ".join(consolidate_inputs)}
    output: "{subanalysis_flag}"
    log: "{log_dir_str}/sims/consolidate_{prefix}{sa_id}.log"
    conda: "{conda_env_path}"
    resources:
{
                self._base_builder._build_resource_block(
                    partition=sub_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                    runtime_min=30,
                    mem_mb=sub_analysis.cfg_analysis.hpc_mem_allocation_for_sim_output_processing_mb,
                    nodes=1,
                    tasks=1,
                    cpus_per_task=1,
                )
            }
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {sub_config_args} \\
            --which {which} \\
{allow_incomplete_line}            --compression-level {compression_level} \\
            --flag-output {{output}} \\
            --rule-name consolidate_{prefix}{sa_id_rule} \\
            --sa-id {sa_id} \\
            > {{log}} 2>&1
        """

'''

        # Sanity assertion: the per-sa loop's subanalysis_flags must match
        # the up-front completed_sa_ids list — both filter on the same
        # c_run_* flag predicate. If they diverge, the up-front filter and
        # the per-sa loop's filter disagree, which is a generator bug.
        from TRITON_SWMM_toolkit.constants import consolidate_subanalysis_flag as _cons_flag

        _expected_subanalysis_flags = [_cons_flag(sa_id) for sa_id in completed_sa_ids]
        if subanalysis_flags != _expected_subanalysis_flags:
            raise RuntimeError(
                "generate_reprocess_master_snakefile_content: per-sa loop's "
                f"subanalysis_flags={subanalysis_flags!r} does not match the "
                f"up-front completed_sa_ids derivation={_expected_subanalysis_flags!r}; "
                "Option C generator invariant violated."
            )

        # Master consolidation — aggregates the EMITTED per-sa flags into the
        # master flag + sensitivity_datatree.zarr; overwrite + allow-incomplete
        # baked. Uses the central flag-name builder for the master flag (Spec 1).
        from TRITON_SWMM_toolkit.constants import consolidate_master_flag

        snakefile_content += f'''rule master_consolidation:
    input: {", ".join([f'"{flag}"' for flag in subanalysis_flags])}
    output: "{consolidate_master_flag()}"
    log: "{log_dir_str}/master_consolidation.log"
    conda: "{conda_env_path}"
    resources:
{
            self._base_builder._build_resource_block(
                partition=self.master_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=30,
                mem_mb=self.master_analysis.cfg_analysis.hpc_mem_allocation_for_analysis_output_consolidation_mb,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )
        }
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            {master_config_args} \\
            --consolidate-sensitivity-analysis-outputs \\
            --allow-incomplete \\
            --which {which} \\
            --compression-level {compression_level} \\
            --flag-output {{output}} \\
            --rule-name master_consolidation \\
            > {{log}} 2>&1
        """
'''

        # Plot + export + render rules — reuse the same helper methods as the
        # production master so the rendered report set is byte-equivalent.
        snakefile_content += self._base_builder._build_plot_rule_block_system_overview(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_per_analysis_summary(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_scenario_status_appendix(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_errors_and_warnings(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_plot_rule_block_disk_utilization(
            input_flag="_status/f_consolidate_master_complete.flag"
        )
        snakefile_content += self._base_builder._build_export_scenario_status_rule(
            input_flag="_status/f_consolidate_master_complete.flag",
        )
        if sa_event_pairs_sa:
            snakefile_content += self._build_plot_rule_block_per_sim_per_sa()
        if _independent_vars:
            snakefile_content += self._build_plot_rule_block_sensitivity_benchmarking(_independent_vars)

        # Render-report rule (wildcarded over format=zip|html), matching the
        # production master generator.
        render_inputs_str = ",\n        ".join(render_rule_input_items)
        snakefile_content += f'''
rule render_report:
    input:
        {render_inputs_str}
    output:
        "analysis_report.{{format}}"
    wildcard_constraints:
        format="zip|html"
    log: "{log_dir_str}/render_report_{{format}}.log"
    resources:
{
            self._base_builder._build_resource_block(
                partition=self.master_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition,
                runtime_min=30,
                mem_mb=self.master_analysis.cfg_analysis.mem_gb_per_cpu * 1000,
                nodes=1,
                tasks=1,
                cpus_per_task=1,
            )
        }
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.render_report_runner \\
            {master_config_args} \\
            --format {{wildcards.format}} \\
            > {{log}} 2>&1
        """
'''

        return snakefile_content

    def _build_plot_rule_block_sensitivity_benchmarking(
        self, independent_vars: list[str], *, ctx: RuleEmissionContext | None = None
    ) -> str:
        """Generate the sensitivity benchmarking plot rule, wildcarded over independent_var.

        Charset validation for independent_var names is upstream, at Phase 1's
        ``validate_sensitivity_independent_vars()``; names reaching here are guaranteed
        Snakemake-safe.

        SWMM-only sub-analyses' .rpt paths are computed at emit time and baked
        into the closure as a list, so the collector can declare them as
        provenance even though they are conditional on enabled-model-types.
        """
        import os as _os

        if ctx is None:
            ctx = self._base_builder._make_rule_emission_context(
                static_backend=self._base_builder._get_report_cfg_static_backend()
            )
        master_root = str(self.master_analysis.analysis_paths.analysis_dir.resolve())
        swmm_only_rpt_rels: list[str] = []
        for sub in self.sensitivity_analysis.sub_analyses.values():
            sub_enabled = sub._get_enabled_model_types()
            if sub_enabled == ["swmm"] or sub_enabled == ("swmm",):
                for event_iloc in sub.df_sims.index:
                    try:
                        scen_paths = sub._retrieve_sim_run_processing_object(event_iloc).scen_paths
                        rpt = getattr(scen_paths, "swmm_full_rpt_file", None)
                        if rpt:
                            swmm_only_rpt_rels.append(_os.path.relpath(str(Path(rpt).resolve()), master_root))
                    except Exception:
                        continue
        helpers = f"""
INDEPENDENT_VARS = {independent_vars!r}

def _sensitivity_source_paths(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_sensitivity_source_paths,
    )
    return collect_sensitivity_source_paths(
        wildcards.independent_var,
        swmm_only_rpt_rel_paths={swmm_only_rpt_rels!r},
    )
"""
        spec = RuleSpec(
            rule_name="plot_sensitivity_benchmarking",
            renderer_module="sensitivity_benchmarking",
            input_flags=("_status/f_consolidate_master_complete.flag",),
            output_path_template="plots/sensitivity/benchmarking/{independent_var}_vs_total__OUTPUT_EXT__",
            source_paths=(),
            wildcards=("independent_var",),
            extra_cli_flags=("--independent-var {wildcards.independent_var}",),
            extra_params=(),
            report_kwargs={
                "caption": "report/captions/sensitivity_benchmarking.rst",
                "category": "Key Results",
                "subcategory": "Benchmarking",
                "labels": '{"independent_var": "{independent_var}", "figure": "vs Total runtime"}',
            },
            resources_yaml="mem_mb=4000, time_min=10",
            log_path_template="logs/plots/sensitivity_benchmarking_{independent_var}.log",
            source_paths_fn_name="_sensitivity_source_paths",
        )
        # plot_sensitivity_benchmarking honors the backend-resolved extension via
        # __OUTPUT_EXT__ substitution in _emit_plot_rule, matching the pattern
        # used by every other chart-renderer rule.
        spec_with_label = RuleSpec(**{**spec.__dict__, "input_label": "master"})
        return helpers + _emit_plot_rule(spec_with_label, ctx)

    def _build_plot_rule_block_per_sim_per_sa(self, *, ctx: RuleEmissionContext | None = None) -> str:
        """Generate per-sa per-event plot rules for the sensitivity master Snakefile.

        Realizes Iteration 7 Change 3b ("show all sub-analyses" panel parity per
        the user's scope expansion). For each (sa_id, event_id) pair in
        SA_EVENT_PAIRS, emits two plot rules: peak_flood_depth + conduit_flow.
        Both rules dispatch the per-sim renderer with `--sa-id {wildcards.sa_id}`
        + `--event-iloc {params.event_iloc}` so the renderer (via _cli.py
        sub-analysis routing) resolves the sub-analysis from the master and
        operates on per-sa-scoped scenario data.

        ILOC_BY_EVENT_ID_BY_SA is emitted as a master-Snakefile global to map
        (sa_id, event_id) -> event_iloc for the renderer dispatch.
        """
        from TRITON_SWMM_toolkit.scenario import compute_event_id_slug

        if ctx is None:
            ctx = self._base_builder._make_rule_emission_context(
                static_backend=self._base_builder._get_report_cfg_static_backend()
            )

        iloc_by_event_id_by_sa: dict[str, dict[str, int]] = {}
        for sa_id, sub in self.sensitivity_analysis.sub_analyses.items():
            iloc_by_event_id_by_sa[str(sa_id)] = {}
            for event_iloc in sub.df_sims.index:
                ev = sub._retrieve_weather_indexer_using_integer_index(event_iloc)
                event_id = compute_event_id_slug(ev)
                iloc_by_event_id_by_sa[str(sa_id)][event_id] = int(event_iloc)

        rainfall_datavar = self.master_analysis.cfg_analysis.weather_time_series_spatial_mean_rainfall_datavar
        storm_tide_datavar = self.master_analysis.cfg_analysis.weather_time_series_storm_tide_datavar

        helpers = f"""
ILOC_BY_EVENT_ID_BY_SA = {iloc_by_event_id_by_sa!r}

def _per_sim_per_sa_flood_depth_sources(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "peak_flood_depth",
        wildcards.event_id,
        rainfall_datavar={rainfall_datavar!r},
        storm_tide_datavar={storm_tide_datavar!r},
        sa_id=wildcards.sa_id,
    )

def _per_sim_per_sa_conduit_flow_sources(wildcards):
    from TRITON_SWMM_toolkit.report_renderers._figure_emission import (
        collect_per_sim_source_paths,
    )
    return collect_per_sim_source_paths(
        "conduit_flow",
        wildcards.event_id,
        rainfall_datavar={rainfall_datavar!r},
        storm_tide_datavar={storm_tide_datavar!r},
        sa_id=wildcards.sa_id,
    )
"""
        flood_spec = RuleSpec(
            rule_name="plot_per_sim_per_sa_peak_flood_depth",
            renderer_module="per_sim_per_sa_peak_flood_depth",
            input_flags=("_status/f_consolidate_master_complete.flag",),
            output_path_template="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}/peak_flood_depth__OUTPUT_EXT__",
            source_paths=(),
            wildcards=("sa_id", "event_id"),
            extra_cli_flags=(
                "--sa-id {wildcards.sa_id}",
                "--event-iloc {params.event_iloc}",
            ),
            extra_params=(("event_iloc", "lambda w: ILOC_BY_EVENT_ID_BY_SA[w.sa_id][w.event_id]"),),
            report_kwargs={
                "caption": "report/captions/per_sim_peak_flood_depth.rst",
                "category": "Per Simulation Results",
                "labels": '{"sa_id": "{sa_id}", "event_id": "{event_id}", "figure": "Peak flood depth"}',
            },
            resources_yaml="mem_mb=4000, time_min=15",
            log_path_template="logs/plots/per_sim_per_sa_peak_flood_depth_sa-{sa_id}_{event_id}.log",
            source_paths_fn_name="_per_sim_per_sa_flood_depth_sources",
            input_label="master",
        )
        # NB: the underlying renderer module is `per_sim_peak_flood_depth`
        # (not `per_sim_per_sa_peak_flood_depth`) — sa-routing is via the
        # --sa-id flag. The renderer_module field is also used by
        # _output_ext_for; emit the correct shell-side module name via a
        # local RuleSpec that overrides renderer_module just for emission.
        flood_emit = RuleSpec(**{**flood_spec.__dict__, "renderer_module": "per_sim_peak_flood_depth"})
        conduit_spec = RuleSpec(
            rule_name="plot_per_sim_per_sa_conduit_flow",
            renderer_module="per_sim_per_sa_conduit_flow",
            input_flags=("_status/f_consolidate_master_complete.flag",),
            output_path_template="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}/conduit_flow__OUTPUT_EXT__",
            source_paths=(),
            wildcards=("sa_id", "event_id"),
            extra_cli_flags=(
                "--sa-id {wildcards.sa_id}",
                "--event-iloc {params.event_iloc}",
            ),
            extra_params=(("event_iloc", "lambda w: ILOC_BY_EVENT_ID_BY_SA[w.sa_id][w.event_id]"),),
            report_kwargs={
                "caption": "report/captions/per_sim_conduit_flow.rst",
                "category": "Per Simulation Results",
                "labels": '{"sa_id": "{sa_id}", "event_id": "{event_id}", "figure": "Conduit flow"}',
            },
            resources_yaml="mem_mb=4000, time_min=15",
            log_path_template="logs/plots/per_sim_per_sa_conduit_flow_sa-{sa_id}_{event_id}.log",
            source_paths_fn_name="_per_sim_per_sa_conduit_flow_sources",
            input_label="master",
        )
        conduit_emit = RuleSpec(**{**conduit_spec.__dict__, "renderer_module": "per_sim_conduit_flow"})
        return helpers + _emit_plot_rule(flood_emit, ctx) + _emit_plot_rule(conduit_emit, ctx)

    def _reconcile_sensitivity_alive(self) -> tuple[dict[str, str], dict[str, str]]:
        """Reconcile in-flight sensitivity sims across all sub-analysis dirs.

        Sensitivity markers live under each sub-analysis's own ``analysis_dir``
        (``master/subanalyses/{analysis_id}``), not the master dir — so the
        master-dir reconcile would miss them (Spec D). This sweeps every
        sub-analysis dir and returns two maps keyed by sentinel rule_token:
        ``alive_by_token`` (token -> slurm_jobid) for the run-vs-wait branch in
        :meth:`generate_master_snakefile_content`, and ``alive_token_to_dir``
        (token -> sub-analysis dir str) so each wait-rule polls the directory
        where its markers actually land.

        Per sentinel-system-v2 Phase 2 (Spec 9 + Spec D).
        """
        alive_by_token: dict[str, str] = {}
        alive_token_to_dir: dict[str, str] = {}
        for _sub in self.sensitivity_analysis.sub_analyses.values():  # type: ignore[attr-defined]
            _sub_dir = _sub.analysis_paths.analysis_dir
            for _tok, _jid in self._base_builder._reconcile_inflight_submissions(analysis_dir=_sub_dir):
                alive_by_token[_tok] = _jid
                alive_token_to_dir[_tok] = str(_sub_dir)
        return alive_by_token, alive_token_to_dir

    def submit_workflow(
        self,
        mode: Literal["local", "slurm", "auto"] = "auto",
        # setup stuff
        process_system_level_inputs: bool = True,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        # ensemble run stuff
        prepare_scenarios: bool = True,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        override_clear_raw: ClearRawValue | None = None,
        compression_level: int = 5,
        pickup_where_leftoff: bool = True,
        wait_for_completion: bool = False,  # relevant for slurm jobs only
        dry_run: bool = False,
        verbose: bool = True,
        override_hpc_total_nodes: int | None = None,
        report_formats: list[str] | None = None,
        extra_sbatch_args: list[str] | None = None,
        snakemake_diagnostics: SnakemakeDiagnostics | None = None,
    ) -> dict:
        """
        Submit sensitivity analysis workflow using Snakemake.

        This orchestrates multiple sub-analysis workflows and a final master
        consolidation step that combines all sub-analysis outputs.
        If multi_sim_run_method is "1_job_many_srun_tasks", submits as a single SLURM
        job with multiple srun tasks inside.

        Parameters
        ----------
        mode : Literal["local", "slurm", "auto"]
            Execution mode. If "auto", detects based on SLURM environment variables.
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings)
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, prepare scenarios before running
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after simulations
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process
        override_clear_raw : ClearRawValue | None
            Runtime override for ``cfg_analysis.clear_raw`` (None reads YAML).
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        dry_run : bool
            If True, only perform a dry run and return that result
        verbose : bool
            If True, print progress messages
        override_hpc_total_nodes : int | None
            If provided, overrides cfg_analysis.hpc_total_nodes for SLURM script generation
            without mutating the config object. Only valid when multi_sim_run_method is
            "1_job_many_srun_tasks"; raises ConfigurationError otherwise.

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool
            - mode: str
            - snakefile_path: Path
            - message: str
        """
        # Stash diagnostics on the base builder so the cmd-arg sites
        # (run_snakemake_local, _run_snakemake_slurm_detached,
        # _validate_batch_job_dry_run on _base_builder) pick them up.
        self._base_builder._active_snakemake_diagnostics = snakemake_diagnostics or SnakemakeDiagnostics()

        # Check if we should use 1-job mode based on config
        multi_sim_method = self.master_analysis.cfg_analysis.multi_sim_run_method

        sim_resources = self.master_analysis._resource_manager._get_simulation_resource_requirements()
        n_gpus_per_sim = sim_resources["n_gpus"]
        if n_gpus_per_sim > 0 and not self.system.cfg_system.gpu_compilation_backend:
            raise ConfigurationError(
                field="gpu_compilation_backend",
                message=(
                    "Sensitivity analysis requests GPUs (n_gpus > 0) but system config "
                    "has gpu_compilation_backend unset. Set gpu_compilation_backend to "
                    "CUDA/HIP or set n_gpus: 0 in sub-analyses."
                ),
                config_path=self.system.system_config_yaml,
            )

        if override_hpc_total_nodes is not None and multi_sim_method != "1_job_many_srun_tasks":
            raise ConfigurationError(
                field="override_hpc_total_nodes",
                message=(
                    f"override_hpc_total_nodes is only valid when multi_sim_run_method='1_job_many_srun_tasks', "
                    f"but current method is '{multi_sim_method}'."
                ),
                config_path=None,
            )

        if extra_sbatch_args is not None and multi_sim_method != "1_job_many_srun_tasks":
            raise ConfigurationError(
                field="extra_sbatch_args",
                message=(
                    f"extra_sbatch_args is only valid when multi_sim_run_method='1_job_many_srun_tasks' "
                    f"(it appends #SBATCH lines to the generated run_workflow_1job.sh), "
                    f"but current method is '{multi_sim_method}'."
                ),
                config_path=None,
            )

        if multi_sim_method == "1_job_many_srun_tasks":
            # Always submit a batch job for 1-job mode
            if verbose:
                print(
                    "[Snakemake] Using 1-job many-srun-tasks mode for sensitivity analysis",
                    flush=True,
                )

            # v2 graceful-rerun: reconcile per sub-analysis before build (Phase 2, Spec 9/D).
            alive_by_token, alive_token_to_dir = self._reconcile_sensitivity_alive()
            # Generate master Snakefile
            master_snakefile_content = self.generate_master_snakefile_content(
                which=which,
                compression_level=compression_level,
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                compile_TRITON_SWMM=compile_TRITON_SWMM,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                override_clear_raw=override_clear_raw,
                pickup_where_leftoff=pickup_where_leftoff,
                report_formats=report_formats,
                alive_by_token=alive_by_token,
                alive_token_to_dir=alive_token_to_dir,
            )

            master_snakefile_path = self.master_analysis.analysis_paths.analysis_dir / "Snakefile"
            master_snakefile_path.write_text(master_snakefile_content)

            if verbose:
                print(
                    f"[Snakemake] Generated master Snakefile: {master_snakefile_path}",
                    flush=True,
                )

            # Create required directories
            analysis_dir = self.master_analysis.analysis_paths.analysis_dir
            (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
            self.analysis_paths.analysis_log_directory.mkdir(parents=True, exist_ok=True)
            (self.analysis_paths.analysis_log_directory / "sims").mkdir(parents=True, exist_ok=True)

            # Always perform a dry run validation first
            dry_run_result = self._base_builder._validate_single_job_dry_run(
                snakefile_path=master_snakefile_path,
                analysis=self.master_analysis,
                verbose=verbose,
                override_hpc_total_nodes=override_hpc_total_nodes,
            )

            if dry_run:
                # Override mode to indicate intended execution context
                dry_run_result["mode"] = "single_job"
                self.sensitivity_analysis._update_master_analysis_log()
                return dry_run_result

            result = self._base_builder._submit_single_job_workflow(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                wait_for_completion=wait_for_completion,
                override_hpc_total_nodes=override_hpc_total_nodes,
                extra_sbatch_args=extra_sbatch_args,
            )

            self.sensitivity_analysis._update_master_analysis_log()
            return result

        if multi_sim_method == "batch_job":
            if verbose:
                print(
                    "[Snakemake] Using batch_job orchestration mode for sensitivity analysis",
                    flush=True,
                )

            # v2 graceful-rerun: reconcile per sub-analysis before build (Phase 2, Spec 9/D).
            alive_by_token, alive_token_to_dir = self._reconcile_sensitivity_alive()
            master_snakefile_content = self.generate_master_snakefile_content(
                which=which,
                compression_level=compression_level,
                process_system_level_inputs=process_system_level_inputs,
                overwrite_system_inputs=overwrite_system_inputs,
                compile_TRITON_SWMM=compile_TRITON_SWMM,
                recompile_if_already_done_successfully=recompile_if_already_done_successfully,
                prepare_scenarios=prepare_scenarios,
                overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                process_timeseries=process_timeseries,
                override_clear_raw=override_clear_raw,
                pickup_where_leftoff=pickup_where_leftoff,
                report_formats=report_formats,
                alive_by_token=alive_by_token,
                alive_token_to_dir=alive_token_to_dir,
            )

            master_snakefile_path = self.master_analysis.analysis_paths.analysis_dir / "Snakefile"
            master_snakefile_path.write_text(master_snakefile_content)

            if verbose:
                print(
                    f"[Snakemake] Generated master Snakefile: {master_snakefile_path}",
                    flush=True,
                )

            analysis_dir = self.master_analysis.analysis_paths.analysis_dir
            (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
            self.analysis_paths.analysis_log_directory.mkdir(parents=True, exist_ok=True)
            (self.analysis_paths.analysis_log_directory / "sims").mkdir(parents=True, exist_ok=True)

            dry_run_result = self._base_builder._validate_batch_job_dry_run(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
            )

            if not dry_run_result.get("success"):
                raise RuntimeError("Dry run failed; workflow submission aborted.")

            if dry_run:
                self.sensitivity_analysis._update_master_analysis_log()
                return dry_run_result

            result = self._base_builder._submit_tmux_workflow(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                wait_for_completion=wait_for_completion,
            )

            self.sensitivity_analysis._update_master_analysis_log()
            return result

        # Standard workflow submission (existing logic)
        # Detect execution mode
        if mode == "auto":
            mode = "slurm" if self.master_analysis.in_slurm else "local"

        if verbose:
            print(
                f"[Snakemake] Submitting sensitivity analysis workflow in {mode} mode",
                flush=True,
            )

        # v2 graceful-rerun: reconcile per sub-analysis before build (Phase 2, Spec 9/D).
        alive_by_token, alive_token_to_dir = self._reconcile_sensitivity_alive()
        # Generate master Snakefile with flattened hierarchy
        # (no nested Snakemake calls - all rules in one file)
        master_snakefile_content = self.generate_master_snakefile_content(
            which=which,
            compression_level=compression_level,
            process_system_level_inputs=process_system_level_inputs,
            overwrite_system_inputs=overwrite_system_inputs,
            compile_TRITON_SWMM=compile_TRITON_SWMM,
            recompile_if_already_done_successfully=recompile_if_already_done_successfully,
            prepare_scenarios=prepare_scenarios,
            overwrite_scenario_if_already_set_up=overwrite_scenario_if_already_set_up,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            process_timeseries=process_timeseries,
            override_clear_raw=override_clear_raw,
            pickup_where_leftoff=pickup_where_leftoff,
            report_formats=report_formats,
            alive_by_token=alive_by_token,
            alive_token_to_dir=alive_token_to_dir,
        )

        master_snakefile_path = self.master_analysis.analysis_paths.analysis_dir / "Snakefile"
        master_snakefile_path.write_text(master_snakefile_content)

        if verbose:
            print(
                f"[Snakemake] Generated master Snakefile: {master_snakefile_path}",
                flush=True,
            )

        # Create required directories BEFORE Snakemake DAG construction
        # (onstart: in Snakefile runs AFTER DAG parsing, too late for file validation)
        analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
        self.master_analysis.analysis_paths.simlog_directory.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(
                f"[Snakemake] Created required directories (_status, {self.master_analysis.analysis_paths.simlog_directory})",  # noqa: E501
                flush=True,
            )

        # Always perform a dry run first
        if mode == "local":
            dry_run_result = self._base_builder.run_snakemake_local(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                dry_run=True,
            )
        else:  # slurm
            dry_run_result = self._base_builder._run_snakemake_slurm_detached(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                wait_for_completion=True,
                dry_run=True,
            )

        if not dry_run_result.get("success"):
            raise RuntimeError("Dry run failed; workflow submission aborted.")

        if dry_run:
            self.sensitivity_analysis._update_master_analysis_log()
            return dry_run_result

        # Submit workflow based on mode
        if mode == "local":
            result = self._base_builder.run_snakemake_local(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                dry_run=False,
            )
        else:  # slurm
            result = self._base_builder._run_snakemake_slurm_detached(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                wait_for_completion=wait_for_completion,
                dry_run=False,
            )

        # Print snakemake log file location if available
        if verbose and result.get("snakemake_logfile") is not None and not wait_for_completion:
            print(
                "[Snakemake] Sensitivity analysis workflow submitted in background.",
                flush=True,
            )
            print(
                f"[Snakemake] Monitor progress with: tail -f {result.get('snakemake_logfile')}",
                flush=True,
            )

        self.sensitivity_analysis._update_master_analysis_log()
        return result

    def submit_reprocess_workflow(
        self,
        *,
        start_with: Literal["process", "consolidate", "render"] = "consolidate",
        execution_mode: Literal["auto", "local", "slurm"] = "auto",
        which: Literal["TRITON", "SWMM", "both"] = "both",
        compression_level: int = 5,
        dry_run: bool = False,
        verbose: bool = True,
        report_formats: list[str] | None = None,
    ) -> dict:
        """Submit a reprocess-scoped master Snakefile for the sensitivity master.

        Sibling to :meth:`submit_workflow`. Writes the scoped master Snakefile
        to ``{analysis_dir}/Snakefile.reprocess`` via
        :meth:`generate_reprocess_master_snakefile_content` (overwrite baked),
        runs the Phase-1 reconciliation guard, and invokes Snakemake with
        ``--rerun-triggers mtime`` so downstream rules only re-fire when
        their outputs are missing or older than their inputs.

        Per the Phase 2 reprocess deviation (in-flight sidecar comment):
        the reprocess driver shares ``analysis_dir/.snakemake/`` with the run
        path. Per-Snakefile locking via the distinct ``Snakefile.reprocess``
        path provides coexistence safety for local tests + CLI smoke;
        true coexistence with a concurrent live ``rule simulation_*`` driver
        is tracked as a Phase 2 follow-up.

        Parameters
        ----------
        start_with
            Stage to re-fire from. ``"consolidate"`` (default) emits per-sa
            consolidate rules consuming ``c_run_*`` flags directly. ``"process"``
            additionally emits per-(sa_id, event_id) process_timeseries rules
            for events whose ``d_process_*`` flag is missing on disk, so partial
            process-step state (sim ran, summary zarrs missing) can be recovered
            without invoking the normal workflow's ``run``. The per-sa
            consolidate shell adds ``--allow-incomplete`` only for sub-analyses
            in truly-mixed state; fully-complete sub-analyses retain
            fail-fast behavior. ``"render"`` skips per-sa consolidate emission
            entirely (handled by the upstream invalidation in
            :meth:`TRITONSWMM_sensitivity_analysis.reprocess`).
        execution_mode
            ``"auto"`` (default) detects SLURM context; ``"local"`` / ``"slurm"``
            force the mode.
        which
            ``"both"`` / ``"TRITON"`` / ``"SWMM"`` — threaded into the
            consolidate rule shells' ``--which`` flag.
        compression_level
            Compression level (0-9) for the consolidate rule shells.
        dry_run
            If True, runs ``snakemake --dry-run`` only.
        verbose
            If True, print progress messages.
        report_formats
            Optional list of report formats to render; defaults to ``["zip"]``.

        Returns
        -------
        dict
            Status dictionary matching the shape of
            :meth:`SnakemakeWorkflowBuilder.submit_reprocess_workflow`.
        """
        # Effective execution mode dispatch — mirror the analysis-level
        # reprocess auto-detect.
        if execution_mode == "auto":
            mode: Literal["local", "slurm"] = "slurm" if self.master_analysis.in_slurm else "local"
        else:
            mode = execution_mode  # type: ignore[assignment]

        if verbose:
            print(
                f"[Snakemake] Submitting sensitivity master reprocess workflow "
                f"(start_with={start_with!r}, mode={mode})",
                flush=True,
            )

        # Emit the scoped master Snakefile.
        snakefile_content = self.generate_reprocess_master_snakefile_content(
            which=which,
            compression_level=compression_level,
            report_formats=report_formats,
            start_with=start_with,
        )
        analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        snakefile_path = analysis_dir / "Snakefile.reprocess"
        (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
        self.analysis_paths.analysis_log_directory.mkdir(parents=True, exist_ok=True)
        (self.analysis_paths.analysis_log_directory / "sims").mkdir(parents=True, exist_ok=True)
        snakefile_path.write_text(snakefile_content)
        if verbose:
            print(
                f"[Snakemake] Reprocess master Snakefile generated: {snakefile_path}",
                flush=True,
            )

        # Logs.
        logs_dir = self.analysis_paths.analysis_log_directory
        logfile_name = (
            "snakemake_sensitivity_reprocess_dry_run.log" if dry_run else "snakemake_sensitivity_reprocess.log"
        )
        snakemake_logfile = logs_dir / logfile_name

        # Build the snakemake command. Reuses the base builder's snakemake
        # command builder + --rerun-triggers mtime.
        cmd_args = self._base_builder._get_snakemake_base_cmd() + [
            "--snakefile",
            str(snakefile_path),
            "--rerun-triggers",
            "mtime",
        ]

        if mode == "local":
            local_cores = self.master_analysis.cfg_analysis.local_cpu_cores_for_workflow
            assert isinstance(local_cores, int), "local_cpu_cores_for_workflow must be specified for local runs"
            cmd_args.extend(["--cores", str(local_cores) if local_cores > 1 else "1"])
        else:  # slurm
            config = self._base_builder.generate_snakemake_config(mode="slurm")
            config_dir = self._base_builder.write_snakemake_config(config, mode="slurm")
            cmd_args.extend(
                [
                    "--profile",
                    str(config_dir),
                    "--executor",
                    "slurm",
                    "--printshellcmds",
                ]
            )

        if dry_run:
            cmd_args.append("--dry-run")

        # Facade — lock check (shared analysis_dir/.snakemake/ per Phase 2's
        # deviation) + reconciliation against analysis_dir/_status/_submitted/.
        # Phase 1's at-most-once guard protects this reprocess from a parallel
        # live sim driver double-submitting.
        self._base_builder._pre_snakemake_invocation_guards(
            snakefile_path,
            dry_run=dry_run,
            verbose=verbose,
            working_dir=analysis_dir,
        )

        # Subprocess invocation. Local runs block; slurm runs detach.
        if mode == "local":
            with open(snakemake_logfile, "w") as log_f:
                result = subprocess.run(
                    cmd_args,
                    cwd=str(analysis_dir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if verbose:
                print(f"[Snakemake] command: \n     {' '.join(cmd_args)}")
            if result.returncode != 0:
                return {
                    "success": False,
                    "mode": "local",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": (f"Snakemake sensitivity reprocess failed. See {snakemake_logfile} for details."),
                    "snakemake_logfile": snakemake_logfile,
                }
            if verbose:
                print("[Snakemake] Sensitivity reprocess completed successfully", flush=True)
            self.sensitivity_analysis._update_master_analysis_log()
            return {
                "success": True,
                "mode": "local",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Sensitivity reprocess completed successfully",
                "snakemake_logfile": snakemake_logfile,
            }

        # slurm path
        with open(snakemake_logfile, "w") as log_f:
            proc = subprocess.Popen(
                cmd_args,
                cwd=str(analysis_dir),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        if verbose:
            print(
                f"[Snakemake] Sensitivity reprocess submitted to background (PID: {proc.pid})",
                flush=True,
            )
        self.sensitivity_analysis._update_master_analysis_log()
        return {
            "success": True,
            "mode": "slurm",
            "snakefile_path": snakefile_path,
            "job_id": None,
            "message": "Sensitivity reprocess submitted to SLURM (detached)",
            "process": proc,
            "snakemake_logfile": snakemake_logfile,
        }
