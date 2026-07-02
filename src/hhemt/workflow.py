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
import re
import shlex
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import yaml  # type: ignore

from hhemt.config.analysis import ClearRawValue
from hhemt.config.hpc_system import (
    resolve_additional_modules,
    resolve_container_spec,
    resolve_gpu_target,
    resolve_gpus_per_node,
)
from hhemt.exceptions import ConfigurationError, WorkflowError
from hhemt.report_plot_ids import (
    _OUTPUT_EXT_BY_RENDERER,
)
from hhemt.report_plot_ids import (
    output_ext_for as _output_ext_for,
)
from hhemt.report_plot_ids import (
    plot_output_template as _plot_output_template,
)
from hhemt.report_renderers._figure_emission import format_sources_rst

# SLURM-liveness primitives live in the leaf module so wait_for_sentinel_runner
# can import them without importing this Snakemake-builder surface. Re-exported
# here for backward compatibility — every existing _slurm_job_is_live /
# _sacct_states_batched / _SACCT_DEAD_STATES reference resolves unchanged.
from hhemt.slurm_liveness import (  # noqa: F401
    _SACCT_DEAD_STATES,
    _sacct_states_batched,
    _slurm_job_is_live,
)
from hhemt.summary_paths import (  # noqa: F401  (re-export shims under the historical private names)
    _SUMMARY_STEMS_BY_MODEL,
)
from hhemt.summary_paths import (  # noqa: F401  (re-export shim under the historical private name)
    scenario_summaries_present as _scenario_summaries_present,
)
from hhemt.summary_paths import (
    sub_analysis_summaries_complete as _sub_analysis_summaries_complete,
)
from hhemt.utils import fast_rmtree

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
_NON_INTERACTIVE_LOCK_CLEAR_ENV = "HHEMT_TEST_NON_INTERACTIVE_LOCK_CLEAR"


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
        First positional arg to ``python -m hhemt.report_renderers._cli``
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


# ADR-2 (reporting-system_canonical-plot-id): _OUTPUT_EXT_BY_RENDERER, the
# canonical-plot-ID grammar (_plot_output_template), and _output_ext_for are the
# single source of truth for figure stems, lifted into the dedicated
# `report_plot_ids` module (the layout-relevant artifact; see
# _layout_relevant_files.yaml) and imported at module top. They remain
# accessible as workflow.* names so the bundle/reprocess generators' existing
# `from hhemt.workflow import _output_ext_for` keeps working.


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
    # report_kwargs is None for static-plot / non-report rules: emit a bare
    # output path with no report() wrapper (R3). The dict branch below is
    # byte-identical to the prior unconditional form, so existing report-rule
    # callers are unchanged.
    if spec.report_kwargs is None:
        output_block = f'"{output_path}"'
    else:
        rk = spec.report_kwargs
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

    # ADR-2 OE-2: constrain each output wildcard to the enforced charset
    # ^[A-Za-z0-9_.]+$ so the "." within-segment separator of a canonical plot
    # ID can never be mis-inferred as a greedy multi-segment wildcard match if
    # two figures ever share a directory. Derived from spec.wildcards (empty for
    # the singleton no-wildcard rules). Mirrors render_report's
    # `wildcard_constraints: format="zip|html"`.
    if spec.wildcards:
        _wc_lines = "\n".join(f'        {w}="[A-Za-z0-9_.]+",' for w in spec.wildcards)
        wildcard_constraints_block = f"    wildcard_constraints:\n{_wc_lines}\n"
    else:
        wildcard_constraints_block = ""

    # Plot-rule shell uses literal "python" (the rule's conda: env
    # provides the interpreter); only setup / run / process / consolidate
    # / render_report rules use ctx.python_executable's full path.
    return f'''
rule {spec.rule_name}:
    {input_block}
    output:
        {output_block}
{wildcard_constraints_block}    params:
{params_block}
    log: "{spec.log_path_template}"
    conda: "{ctx.conda_env_path}"
    resources: {spec.resources_yaml}
    shell:
        """
        python -m hhemt.report_renderers._cli {spec.renderer_module} \\
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
        {ctx.python_executable} -m hhemt.render_report_runner \\
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


def _brand_theme_css_map(theme) -> "dict[str, str]":
    """Map a resolved brand_theme model -> report.css.j2 placeholder names."""
    return {
        "uva_blue": theme.primary_color,
        "uva_orange": theme.accent_color,
        "uva_light_gray": theme.neutral_light,
        "uva_medium_gray": theme.neutral_medium,
        "uva_text_gray": theme.text_muted,
        "uva_link_blue": theme.link_color,
    }


_SNAKEFILE_RUNNER_RE = re.compile(r"-m\s+([A-Za-z_][A-Za-z0-9_]*)\.\w+(?:_runner|_workflow)\b")


def _assert_snakefile_package_current(snakefile_path: Path) -> None:
    """Fail fast at the login node if an on-disk Snakefile bakes a runner module
    path for a distribution other than the installed package.

    A Snakefile written by a prior ``run()`` before a package rename (e.g.
    ``TRITON_SWMM_toolkit`` -> ``hhemt``) keeps the old ``-m {dist}.{runner}``
    token; re-running ``snakemake --report`` (or any read-existing-Snakefile path)
    against it would surface an hours-deep mid-SLURM ``ModuleNotFoundError``. This
    converts that into a sub-second ``ConfigurationError`` (CLI exit 2). No-ops on
    a current or absent Snakefile.
    """
    if not snakefile_path.exists():
        return
    baked = {m.group(1) for m in _SNAKEFILE_RUNNER_RE.finditer(snakefile_path.read_text())}
    current = __name__.split(".", 1)[0]  # "hhemt"
    stale = baked - {current}
    if stale:
        from hhemt.exceptions import ConfigurationError

        raise ConfigurationError(
            field="Snakefile",
            config_path=snakefile_path,
            message=(
                f"On-disk Snakefile bakes runner module path(s) for {sorted(stale)}, "
                f"but the installed package is '{current}'. Regenerate it by re-running "
                f"analysis.run()/submit_workflow() before invoking render/reprocess."
            ),
        )


def _emit_report_artifacts(
    dest_root: Path,
    brand_theme: "Mapping[str, str] | None" = None,
) -> None:
    """Copy report_templates/ -> {dest_root}/report/.

    Uses importlib.resources for package-resource resolution (robust across
    editable and site-packages installs). Falls back to Path(__file__) arithmetic
    only when importlib.resources is unavailable. Requires report_templates/
    to ship as package data under src/hhemt/ via pyproject.toml's
    [tool.setuptools.package-data] entry.

    ``brand_theme`` is a mapping of report.css.j2 placeholder name (uva_blue,
    uva_orange, uva_light_gray, uva_medium_gray, uva_text_gray, uva_link_blue)
    -> hex string. When ``None``, the code-frozen UVA default reproduces the
    pre-branding output byte-for-byte. report.css.j2 is rendered with
    ``string.Template.safe_substitute`` (NOT jinja2) so CSS rule braces and the
    inline data-URI SVG masks pass through untouched.

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

        src_templates = Path(str(_resource_files("hhemt") / "report_templates"))
    except (ImportError, ModuleNotFoundError):
        src_templates = Path(__file__).parent / "report_templates"

    dst_report = dest_root / "report"
    dst_report.mkdir(parents=True, exist_ok=True)
    from string import Template as _StrTemplate

    if brand_theme is None:
        brand_theme = {
            "uva_blue": "#232D4B",
            "uva_orange": "#E57200",
            "uva_light_gray": "#F1F1EF",
            "uva_medium_gray": "#DADADA",
            "uva_text_gray": "#666666",
            "uva_link_blue": "#495E9D",
        }
    _css_src = (src_templates / "report.css.j2").read_text()
    _css_rendered = _StrTemplate(_css_src).safe_substitute(brand_theme)
    (dst_report / "report.css").write_text(_css_rendered)
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


# SLURM-liveness primitives (_slurm_job_is_live, _SACCT_DEAD_STATES,
# _sacct_states_batched) now live in the leaf module slurm_liveness.py and are
# re-exported at the top of this file (see the import near the module header),
# so every reference below resolves unchanged. They were extracted so the
# lightweight wait_for_sentinel_runner.py subprocess can import them without
# importing this Snakemake-builder surface.


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


class _ReportingSetDispatchMixin:
    """Registry-driven plot-rule dispatcher shared by the multisim and
    sensitivity-master/reprocess generators (P1b / TO-8).

    Replaces the hardcoded `_build_plot_rule_block_*` call lists (duplicated
    across `generate_snakefile_content`, `generate_master_snakefile_content`, and
    `generate_reprocess_master_snakefile_content`) with one dispatcher that
    iterates the active reporting set's `renderer_selection`. `SnakemakeWorkflowBuilder`
    and `SensitivityAnalysisWorkflowBuilder` are composition-related (the
    sensitivity builder holds a `_base_builder`), not inheritance-related, so the
    dispatcher lives in a mixin both inherit: `getattr(self, "_base_builder", self)`
    resolves the five common builders to the base builder when `self` is the
    sensitivity builder and to `self` when `self` is the base builder; the two
    conditional builders (per_sim_per_sa, sensitivity_benchmarking) live only on
    the sensitivity builder and resolve via getattr.
    """

    # Predicate-key -> predicate over the per-call predicate_inputs dict. The
    # values (independent_vars, sa_event_pairs_sa) are METHOD-LOCALs inside the
    # master/reprocess generators (NOT instance attributes), threaded in via
    # predicate_inputs; getattr-on-self would always be None. (F-I-2)
    _RENDERER_PREDICATES = {
        "has_independent_vars": lambda inp: bool(inp.get("independent_vars")),
        "has_sa_event_pairs": lambda inp: bool(inp.get("sa_event_pairs_sa")),
    }

    def _resolve_active_reporting_set(self, analysis):
        """Resolve ``analysis``'s active ReportingSet (TO-8 dispatch source).

        `_active_reporting_set` is set only at analysis.run() entry (F-B-1), so a
        generate-without-run() path (e.g. the byte-identity test, which configures
        but never runs the analysis; render_report_runner on a fresh instance)
        must fall back to the CSV-free name resolver — mirroring the established
        getattr-fallback in `analysis.render_report`'s category-order block. The
        fallback resolves to the SAME set the run-entry attr would hold (default
        for multisim, benchmarking for sensitivity), preserving byte-identity (R6).
        """
        active = getattr(analysis, "_active_reporting_set", None)
        if active is not None:
            return active
        from hhemt.config.report import resolve_active_reporting_set_name
        from hhemt.report_renderers._reporting_sets import get_reporting_set

        cfg_report = getattr(analysis, "_cfg_report", None)
        if cfg_report is None:
            cfg_report = analysis.cfg_analysis.report
        name = resolve_active_reporting_set_name(
            cfg_report,
            is_sensitivity=analysis.cfg_analysis.toggle_sensitivity_analysis,
        )
        return get_reporting_set(name)

    def _builder_call_kwargs(self, builder_key, input_flag, ctx, inputs):
        """Resolve the exact kwargs for one builder_key (Option A — the 8
        `_build_plot_rule_block_*` builders are LEFT UNCHANGED, so their
        heterogeneous signatures are honored here rather than normalized). Keeps
        R6 byte-identity free: no builder body is touched, so the proof reduces to
        "the resolver yields the historical call args."
        """
        if builder_key in (
            "system_overview",
            "per_analysis_summary",
            "scenario_status_appendix",
            "errors_and_warnings",
            "disk_utilization",
        ):
            return {"input_flag": input_flag, "ctx": ctx}
        if builder_key in ("per_sim", "per_sim_per_sa"):
            return {"ctx": ctx}  # wildcard over event_id; no input_flag in signature
        if builder_key == "sensitivity_benchmarking":
            # positional independent_vars, forwarded from predicate_inputs (the
            # value the master/reprocess call sites already thread); NO input_flag.
            return {"independent_vars": inputs["independent_vars"], "ctx": ctx}
        raise KeyError(f"unknown builder_key for call-kwargs resolution: {builder_key!r}")

    def _emit_active_set_plot_rules(
        self,
        reporting_set,
        *,
        input_flag: str,
        ctx: "RuleEmissionContext | None" = None,
        predicate_inputs: dict | None = None,
        interleave_after_unconditional=None,
    ) -> str:
        """Emit every plot rule for the active reporting set, in set order (TO-8).

        The set's `renderer_selection` is the single source of which renderers fire
        and in what order; each entry names a builder method (by key) and an
        optional predicate_key gating conditional renderers (benchmarking, per_sa).
        `input_flag` is threaded per generator (multisim: e_consolidate_complete;
        master/reprocess: f_consolidate_master_complete).

        `interleave_after_unconditional` (B-i hook) is a zero-arg callable flushed
        ONCE immediately before the first predicate-keyed entry — so the export
        rule lands BETWEEN the unconditional and conditional renderers at
        master/reprocess, byte-matching the pre-refactor emission order. Multisim
        (no conditional entries) passes None and keeps a trailing export sibling.
        """
        base = getattr(self, "_base_builder", self)
        builders = {
            "system_overview": base._build_plot_rule_block_system_overview,
            "per_sim": base._build_plot_rule_block_per_sim,
            "per_analysis_summary": base._build_plot_rule_block_per_analysis_summary,
            "scenario_status_appendix": base._build_plot_rule_block_scenario_status_appendix,
            "errors_and_warnings": base._build_plot_rule_block_errors_and_warnings,
            "disk_utilization": base._build_plot_rule_block_disk_utilization,
            "per_sim_per_sa": getattr(self, "_build_plot_rule_block_per_sim_per_sa", None),
            "sensitivity_benchmarking": getattr(self, "_build_plot_rule_block_sensitivity_benchmarking", None),
        }
        out = ""
        inputs = predicate_inputs or {}
        interleaved = False
        for sel in reporting_set.renderer_selection:
            if sel.predicate_key is not None:
                # First conditional entry: flush the interleave hook (export rule)
                # so it lands BETWEEN the unconditional and conditional renderers.
                # Flushed BEFORE the predicate check — matching today's
                # unconditional export emission (the export fires regardless of
                # whether the first conditional renderer's predicate passes).
                if interleave_after_unconditional is not None and not interleaved:
                    out += interleave_after_unconditional()
                    interleaved = True
                if not self._RENDERER_PREDICATES[sel.predicate_key](inputs):
                    continue
            builder = builders[sel.builder_key]
            if builder is None:  # conditional builder absent on the base builder
                continue
            out += builder(**self._builder_call_kwargs(sel.builder_key, input_flag, ctx, inputs))
        return out


class SnakemakeWorkflowBuilder(_ReportingSetDispatchMixin):
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
        # Phase 2 (R3): the per-HPC-system config (None when no
        # --hpc-system-config was supplied). The batch_job emitter resolution
        # helpers prefer it when present and fall back to the legacy
        # cfg_analysis/cfg_system reads otherwise (byte-identical when None).
        self.cfg_hpc_system = analysis.cfg_hpc_system
        self.analysis_paths = analysis.analysis_paths
        # Prefer an explicit interpreter path for generated shell commands.
        # If analysis stores a generic command ("python"/"python3"), use
        # the current interpreter running this process to avoid PATH issues.
        configured_python = str(analysis._python_executable)
        if configured_python in {"python", "python3"}:
            self.python_executable = sys.executable
        else:
            self.python_executable = configured_python
        # Runtime retry overrides (resume-retry-resilience P1, Decision 4). Set by
        # submit_workflow before generate_snakemake_config runs; None means "read the
        # config knob". Stored on the builder rather than threaded through
        # generate_snakemake_config's ~5 call sites (FQ3 SITE 5). _simulate resolves at
        # the per-rule retries: on the simulate rules; _other at the global restart-times
        # baseline that directive-less rules inherit.
        self._override_hpc_restart_times_simulate: int | None = None
        self._override_hpc_restart_times_other: int | None = None

        # ADR-1: the container prefix for the PROCESS rungs only (sim rungs wrap
        # inside run_simulation.py; plot/consolidate/render stay native). Empty in
        # native mode (keeps generated Snakefiles byte-identical, R2/TO-1).
        _cspec = resolve_container_spec(self.cfg_hpc_system)
        if getattr(self.cfg_analysis, "execution_environment", "native") == "container" and _cspec is not None:
            # FB2: the process rung MUST bind analysis_dir (+ cluster binds) — the
            # in-container process runner loads configs (firing _check_paths_exist on
            # host paths) and reads out_*/writes processed/+_status under analysis_dir,
            # which on Frontier ($MEMBERWORK) / Rivanna (/scratch) is OUTSIDE Apptainer's
            # default $HOME/CWD binds. This mirrors the sim rung's bind (R7); without it
            # container-mode processing fails at config-load or write on every real cluster.
            _adir = self.analysis_paths.analysis_dir
            _proc_binds = ",".join([*_cspec.binds, f"{_adir}:{_adir}"])
            self._container_process_prefix = f'export APPTAINER_BIND="{_proc_binds}"; apptainer exec {_cspec.sif_path} '
        else:
            self._container_process_prefix = ""

    def _resolved_simulate_retries(self) -> int:
        """Per-rule ``retries:`` for the simulate rules: override-or-config (P1 Decision 4).

        Mirrors the ``override_hpc_total_nodes`` consume-site idiom. Returns the runtime
        override when set, else the ``hpc_restart_times_simulate`` config knob. Emitted as
        a per-rule ``retries:`` directive on the simulate rule blocks, which OVERRIDES the
        global ``restart-times`` baseline (= ``hpc_restart_times_other``) under snakemake
        9.15.0 precedence (Rule.restart_times, rules.py:158-165).
        """
        return (
            self._override_hpc_restart_times_simulate
            if self._override_hpc_restart_times_simulate is not None
            else self.cfg_analysis.hpc_restart_times_simulate
        )

    def _sweep_failed_rules(self, analysis_dir: Path, snakemake_stderr: str = "") -> list[dict]:
        """Enumerate permanently-failed rules after a --keep-going run (FQ3).

        Layered source: (1) _status/_failed/*.json markers (precise rule_token +
        jobid + reason; written by the v2 sim/wait/delete runners' try/finally,
        Gotcha 30/43) for the sim class; (2) a Snakemake-stderr scan for
        ``Error in rule (\\w+):`` to catch the process/consolidate/plot/render
        classes that emit NO _failed/ marker (their shells do not run the v2
        submission-sentinel try/finally). Union the two.
        """
        records: list[dict] = []
        seen: set[str] = set()
        failed_dir = analysis_dir / "_status" / "_failed"
        if failed_dir.is_dir():
            for marker in sorted(failed_dir.glob("*.json")):
                try:
                    rec = json.loads(marker.read_text())
                except (json.JSONDecodeError, OSError):
                    rec = {"rule_token": marker.stem, "reason": "unparseable marker"}
                records.append(rec)
                seen.add(rec.get("rule_token", marker.stem))
        for m in re.finditer(r"Error in rule (\w+):", snakemake_stderr):
            name = m.group(1)
            if name not in seen:
                records.append({"rule_token": name, "reason": "snakemake-log failure (no _failed marker)"})
                seen.add(name)
        return records

    def _augment_result_with_partial_failures(self, result: dict) -> dict:
        """Sweep permanently-failed rules into a blocking-mode submit result (FQ3).

        Reads the merged stdout+stderr Snakemake log (``result["snakemake_logfile"]``,
        written with ``stderr=subprocess.STDOUT``) and unions it with the
        ``_status/_failed/`` markers via :meth:`_sweep_failed_rules`. Sets
        ``result["partial_failures"]`` and forces ``success=False`` when non-empty so a
        single non-retryable failure no longer passes silently after --keep-going let the
        rest of the DAG complete.

        Call ONLY on code paths that block on the Snakemake subprocess return (``local``;
        ``1_job_many_srun_tasks`` with ``wait_for_completion=True``). A detached run
        (``batch_job`` tmux, or 1-job sbatch-and-return) has no in-process completion
        point — sweeping there reads an in-progress/empty ``_failed/`` dir and reports a
        false-clean (captured follow-up: post-hoc get_status sweep).
        """
        if not isinstance(result, dict):
            return result
        log_text = ""
        logfile = result.get("snakemake_logfile")
        if logfile:
            try:
                log_text = Path(logfile).read_text()
            except OSError:
                log_text = ""
        partial_failures = self._sweep_failed_rules(
            self.analysis_paths.analysis_dir,
            snakemake_stderr=log_text,
        )
        result["partial_failures"] = partial_failures
        if partial_failures:
            result["success"] = False
            print(
                f"[Workflow] {len(partial_failures)} rule(s) permanently failed "
                f"(--keep-going let the rest complete): "
                + ", ".join(r.get("rule_token", "?") for r in partial_failures),
                flush=True,
            )
        return result

    def _get_conda_env_path(self) -> Path:
        """Get absolute path to conda environment file.

        The path is embedded in generated Snakefiles via the 'conda:' directive, but
        --use-conda is not currently passed to Snakemake, so the directive is inert.
        The two-environment split is aspirational; this file is currently the single
        working environment for all toolkit work.
        """
        triton_toolkit_root = Path(__file__).parent.parent.parent
        return triton_toolkit_root / "workflow" / "envs" / "hhemt.yaml"

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
        skip_lock_check: bool = False,
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
            behavior, unchanged. The reprocess path shares the main
            ``analysis_dir/.snakemake/`` and instead passes
            ``skip_lock_check=True`` (the lock check is bypassed for reprocess;
            the orchestrator-liveness gate is the concurrency authority).
        skip_lock_check : bool, default False
            When True, return before the lock check (and its interactive
            ``input()`` prompt) entirely. The reprocess path passes True
            because ``--nolock`` governs only the Snakemake subprocess and does
            not shadow this toolkit-side prompt; the ``_status/_orchestrator/``
            liveness gate is the reprocess concurrency authority. The run/submit
            path passes False (the default), so its behavior is byte-identical.

        Raises
        ------
        WorkflowError
            If lock files are found and the user declines to unlock, or if
            snakemake --unlock itself fails.
        """
        if dry_run:
            return
        if skip_lock_check:
            # Reprocess path: the orchestration-liveness sentinel — not the
            # Snakemake lock — is the concurrency authority, and --nolock is on
            # the reprocess subprocess. Skip the toolkit-side lock check so the
            # interactive input() prompt below is never reached in a non-TTY.
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
                    # EXEMPT-DU: lock-file-cleanup
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

        # mechanism (b) PENDING-recovery: discover SLURM-accepted-but-still-PENDING
        # sims. The worker writes _status/_submitted/ only at process START, so a job
        # SLURM accepted but has not yet started leaves NO _submitted/ sentinel — a
        # driver-death-then-rerun would otherwise DOUBLE-submit it. The toolkit wrote
        # _status/_queued/{rule_token}.json for the planned sim-token set at submit; a
        # token still in _queued/ NOT yet superseded by _submitted/_completed/_failed
        # is a still-queued sim to hold (wait-rule), not resubmit.
        queued_dir = base_dir / "_status" / "_queued"
        queued_tokens = sorted(p.stem for p in queued_dir.glob("*.json")) if queued_dir.is_dir() else []

        def _still_queued(tok: str) -> bool:
            # existence-semantic predicate: the _status/{_submitted,_completed,_failed,
            # _queued}/ sentinel classes record that a state-transition occurred
            # (presence == transition), distinct from the _already_written()/
            # processing-log CONTENT-success gate used for processed outputs — raw
            # .exists() is the correct predicate here (F-I Flag 8).
            return not any(
                (base_dir / "_status" / d / f"{tok}.json").exists() for d in ("_submitted", "_completed", "_failed")
            )

        pending_tokens = [t for t in queued_tokens if _still_queued(t)]

        # Fast path: a genuinely-fresh analysis (no _submitted/ sentinels AND no
        # still-_queued/ tokens) returns [] with ZERO sacct calls (R6). A previously-
        # submitted analysis with an empty _submitted/ but live _queued/ tokens must
        # NOT fast-return — it falls through to the PENDING-recovery sweep below.
        if not sentinels and not pending_tokens:
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

        # mechanism (b): merge PENDING-recovered tokens into the alive set so the
        # emission layer substitutes wait_for_{token} rules for still-queued sims
        # (R1, R5 — applied in the SHARED reconcile so the multisim and sensitivity
        # emission sites both inherit it). Each recovered token is a canonical
        # rule-token (the toolkit wrote _queued/{rule_token}.json), so it flows
        # through _emit_wait_for_sim_rule_block unchanged — no comment:-token
        # normalization (mechanism (a) eliminated). Token-keyed dedup (SE Flag 2):
        # a _submitted/-derived (token, jobid) entry ALWAYS wins over a _queued/-
        # derived (token, "") for the same logical token — a hard-kill between the
        # runner's os.replace and its _queued/ unlink can transiently leave BOTH —
        # so the emission layer never sees one token twice with divergent jobids.
        recovered = self._recover_pending_from_queued(pending_tokens, base_dir)
        _alive_tokens = {t for t, _ in alive}
        alive = alive + [r for r in recovered if r[0] not in _alive_tokens]

        if alive:
            print(
                f"[reconcile] v2 graceful-rerun: {len(alive)} in-flight rule(s) "
                f"detected via sentinel state markers; emitting wait-rules in lieu "
                f"of resubmit. Rule tokens: {sorted(t for t, _ in alive)}",
                flush=True,
            )
        return alive

    def _recover_pending_from_queued(self, pending_tokens: list[str], base_dir: Path) -> list[tuple[str, str]]:
        """mechanism (b) PENDING-recovery: classify still-_queued/ tokens (a sim
        SLURM accepted but whose worker has not written _submitted/ yet) by
        WHO-OWNS-THE-SBATCH, returning the alive subset as (rule_token, slurm_jobid)
        tuples for merge into the reconcile alive set.

        - Toolkit-owns-sbatch (the _queued/ payload carries an allocation jobid, e.g.
          Frontier 1_job_many_srun_tasks): classify via _sacct_states_batched with the
          SAME F2 srun-step aliasing guard as _classify_stale_via_sacct (a shared
          allocation jobid is aliasing-unsafe). DEAD -> drop the _queued/ sentinel
          (re-runs); ALIVE or sacct-UNKNOWN -> hold, returning (token, jid).
        - Executor-owns-sbatch (payload jobid null, e.g. UVA & all --executor slurm
          batch_job — the executor assigns per-rule ids the toolkit never sees): F1-O3
          hold-on-PRESENCE, returning (token, ""), bounded by the mtime-age fail-safe
          (R12) — a _queued/ older than _max_plausible_job_lifetime_min is a stale
          orphan, dropped (re-runs).

        NO sacct --name= run-UUID cross-check ships (mechanism (c) struck — the
        executor's run-UUID JobName is not toolkit-controllable; F1-O1 toolkit-minted
        identity is the named enabled future). Returns [] for an empty input with no
        sacct call (R6 fast-path politeness)."""
        if not pending_tokens:
            return []
        queued_dir = base_dir / "_status" / "_queued"
        cap_min = _max_plausible_job_lifetime_min(self.cfg_analysis)
        max_plausible_s = cap_min * 60

        # Read each payload's allocation jobid (None = executor-owns / unreadable).
        jobid_by_token: dict[str, str | None] = {}
        for tok in pending_tokens:
            try:
                payload = json.loads((queued_dir / f"{tok}.json").read_text())
                jobid_by_token[tok] = str(payload.get("slurm_jobid") or "") or None
            except (json.JSONDecodeError, OSError):
                jobid_by_token[tok] = None

        # Toolkit-owns set: ONE batched sacct call (R6 — only when jobid-bearing
        # pending tokens exist; an executor-owns-only / fresh set makes no call).
        toolkit_owned_jids = [j for j in jobid_by_token.values() if j]
        states = _sacct_states_batched(toolkit_owned_jids) if toolkit_owned_jids else {}
        # F2 srun-step aliasing guard for the _queued/ set too: a jobid shared by
        # >=2 pending tokens is not per-token authoritative -> mtime fail-safe.
        _jid_counts = Counter(toolkit_owned_jids)
        _aliased_jids = {j for j, c in _jid_counts.items() if c >= 2}

        recovered: list[tuple[str, str]] = []
        for tok in pending_tokens:
            jid = jobid_by_token[tok]
            qpath = queued_dir / f"{tok}.json"
            if jid and jid not in _aliased_jids:
                row = states.get(jid)
                if row is not None:
                    state, _exit, _reason = row
                    if state in _SACCT_DEAD_STATES:
                        # EXEMPT-DU: status-dir-cleanup
                        qpath.unlink(missing_ok=True)  # dead -> re-run
                        continue
                    recovered.append((tok, jid))  # alive (toolkit-owns)
                    continue
                # jid present but sacct-UNKNOWN -> fall through to mtime fail-safe.
            # executor-owns (jid None) OR aliased OR sacct-UNKNOWN: presence + mtime.
            try:
                age_s = time.time() - qpath.stat().st_mtime
            except OSError:
                age_s = max_plausible_s + 1
            if age_s >= max_plausible_s:
                # EXEMPT-DU: status-dir-cleanup
                qpath.unlink(missing_ok=True)  # stale orphan -> re-run (R12)
                continue
            recovered.append((tok, jid or ""))  # held on presence (F1-O3)
        return recovered

    def _planned_sim_tokens(self) -> list[str]:
        """Enumerate the canonical multisim sim rule-tokens the Snakefile emits
        run-rules for (the mechanism (b) _queued/ writer source). Recomputes the SAME
        (model_type x event_id) cross-product generate_snakefile_content uses:
        event_ids via compute_event_id_slug over the integer weather indexer, enabled
        models via the system toggles. The token form run_{model_type}_evt-{event_id}
        is byte-identical to run_simulation_runner.py's _rule_token (literal evt-)."""
        from hhemt.scenario import compute_event_id_slug

        n_sims = len(self.analysis.df_sims)
        event_ids = [
            compute_event_id_slug(self.analysis._retrieve_weather_indexer_using_integer_index(i)) for i in range(n_sims)
        ]
        enabled_models: list[str] = []
        if self.system.cfg_system.toggle_triton_model:
            enabled_models.append("triton")
        if self.system.cfg_system.toggle_tritonswmm_model:
            enabled_models.append("tritonswmm")
        if self.system.cfg_system.toggle_swmm_model:
            enabled_models.append("swmm")
        return [f"run_{model_type}_evt-{event_id}" for model_type in enabled_models for event_id in event_ids]

    def _write_queued_sentinels(self, planned_tokens: list[str], alloc_jobid: str | None, analysis_dir: Path) -> None:
        """mechanism (b) PENDING-recovery writer: at submit, AFTER the snakemake
        submission call returns success (F2-OC write-after-launch — a pre-launch
        exception leaves no orphan), write _status/_queued/{rule_token}.json for every
        planned sim rule-token so a still-PENDING sim is discoverable at the next
        _reconcile_inflight_submissions BEFORE its worker writes _submitted/.

        Compare-and-write (mtime-preserving): the payload carries NO timestamp, so a
        re-submit of the same (token, jobid) is byte-identical and PRESERVES mtime —
        the R12 mtime-age fail-safe therefore measures age-since-first-submit and ages
        out genuinely-stale orphans even across driver re-invocations (a written_at
        field would bump mtime on every submit and defeat the age-out). Toolkit-owns-
        sbatch (1_job_many_srun_tasks) records the allocation jobid (→ wait-runner
        in-loop probe, R8); executor-owns-sbatch (batch_job) records null. The worker
        unlinks its own _queued/{token}.json at the _submitted/ write (queued→submitted
        handoff). The top-level slurm_jobid key is byte-key-equal to the _submitted/
        payload so wait_for_sentinel_runner._read_submitted_jobid reads it unchanged on
        the _queued/ fallback (SE Flag 3)."""
        qdir = Path(analysis_dir) / "_status" / "_queued"
        qdir.mkdir(parents=True, exist_ok=True)
        for tok in planned_tokens:
            payload = json.dumps({"rule_token": tok, "slurm_jobid": alloc_jobid}, sort_keys=True)
            qpath = qdir / f"{tok}.json"
            try:
                if qpath.read_text() == payload:
                    continue  # byte-identical → preserve mtime (compare-and-write)
            except OSError:
                pass
            tmp = qpath.with_suffix(".json.tmp")
            tmp.write_text(payload)
            tmp.replace(qpath)  # atomic

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

        # NOTE: `sacct -u` is hidden-partition-safe. The slurmdbd accounting DB has no
        # partition-visibility filter, so `sacct -u $USER` returns jobs in SLURM Hidden
        # partitions (e.g. UVA shen GPU partitions) where `squeue -u $USER` is blind.
        # Do NOT "fix" this to squeue. (Empirically: sacct returns hidden-partition
        # PENDING rows; see library/knowledge/slurm/hidden_partition_makes_per_user_squeue_blind.md.)
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
        skip_lock_check: bool = False,
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
            every Phase-1 caller, unchanged. The reprocess path shares the
            main ``analysis_dir/.snakemake/`` and passes ``skip_lock_check=True``
            instead (the lock check is bypassed for reprocess). The
            reconciliation guard is unaffected by ``working_dir`` — it sweeps
            the analysis-level ``_status/_submitted/`` sentinel directory which
            is shared across run and reprocess paths.
        skip_lock_check : bool, default False
            Threaded into :meth:`_check_and_clear_snakemake_lock`. When True
            (reprocess path only), the lock check returns before its
            interactive ``input()`` prompt — the ``_status/_orchestrator/``
            liveness gate is the reprocess concurrency authority. The run/submit
            path passes False (the default), so its behavior is byte-identical.
        """
        self._check_and_clear_snakemake_lock(
            snakefile_path,
            dry_run=dry_run,
            verbose=verbose,
            working_dir=working_dir,
            skip_lock_check=skip_lock_check,
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
        hpc_system_config_yaml: Path | None = None,
        target_partition: str | None = None,
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
        hpc_system_config_yaml : Path | None
            If provided, use this HPC-system config instead of
            self.analysis.hpc_system_config_yaml. Emitted as ``--hpc-system-config``
            only when an HPC config is present (the analysis attribute is None when
            no ``hpc_system_config.yaml`` was supplied) — so the emitted string is
            byte-identical to today when the third config is absent.
        target_partition : str | None
            Phase-4 (4c): the partition whose PartitionSpec GPU hardware/backend the
            GPU-compile runners (setup_workflow, run_simulation_runner) resolve +
            inject into ``TRITONSWMM_system``. Emitted as ``--target-partition`` ONLY
            when provided — the shared/non-GPU-compile rule emissions pass None and
            stay byte-identical. The GPU-compile target is the ENSEMBLE (sim)
            partition for BOTH the setup rule (which compiles the binary that runs on
            the sim partition) and the sim rule (which runs it) — NOT the processing
            partition (which is CPU post-processing and carries no GPU hardware).

        Returns
        -------
        str
            Config arguments string
        """
        analysis_cfg = analysis_config_yaml or self.analysis.analysis_config_yaml
        system_cfg = system_config_yaml or self.system.system_config_yaml
        hpc_cfg = hpc_system_config_yaml or self.analysis.hpc_system_config_yaml
        base = f"--system-config {system_cfg} \\\n            --analysis-config {analysis_cfg}"
        if hpc_cfg is not None:
            base += f" \\\n            --hpc-system-config {hpc_cfg}"
        if target_partition is not None:
            base += f" \\\n            --target-partition {target_partition}"
        return base

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
            # EXEMPT-DU: status-flag
            flag_path.unlink(missing_ok=True)
            sidecar = flag_path.with_suffix(flag_path.suffix + ".json")
            # EXEMPT-DU: status-flag
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
        # --exclusive (whole-node hold) is correct ONLY when the sim wants every GPU
        # on the node. For a strict subset (2 <= n_gpus < gpus_per_node) we allocate
        # exactly n_gpus GPUs + cpus_per_task=1 (no carve), which binds correctly on
        # both UVA gpu-a6000 and gpu-a100-80 (empirically confirmed 2026-06-10 — see
        # knowledge doc single_vs_per_task_gres_binding_on_shared_affinity_topology.md
        # Appendix B.10). gpus_total >= gpus_per_node_config also covers multi-node
        # full-GPU sims (e.g. n_gpus=16 on 8-GPU nodes).
        full_node_gpu = gres_multi_gpu and gpus_per_node_config >= 1 and gpus_total >= gpus_per_node_config

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
            # --exclusive (whole-node hold) is emitted ONLY for a full-node GPU sim
            # (n_gpus >= gpus_per_node — single full node or multi-node full-GPU). It
            # upgrades the grant to whole-node so every GPU's gres.conf affinity cores
            # are held -> all N bind (the sa_36 fix; Matrix C/D + knowledge doc
            # single_vs_per_task_gres_binding_on_shared_affinity_topology.md).
            # For a STRICT SUBSET (2 <= n_gpus < gpus_per_node) --exclusive is OMITTED:
            # the partial-node grant (gres/gpu=N, cpus_per_task=1) binds all N GPUs on
            # both a6000 and a100 (Appendix B.10, 2026-06-10), and holding the whole
            # node would strand the unused GPUs and trip the RC 0%-util auto-cancel.
            # Do NOT carve cpus_per_task for subsets — the per-GPU carve half-binds
            # a100 (B.10/B.3). slurm_extra is the executor passthrough; the bare
            # exclusive=True resource key is NOT recognized.
            if full_node_gpu:
                block += ',\n        slurm_extra="--exclusive"'

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

    def _resolve_account(self) -> str | None:
        """Phase-2 (R3): SLURM account default-resource.

        Returns cfg_hpc_system.default_account when the per-HPC-system config is
        present; else None (Phase-4 4d: the legacy cfg_analysis.hpc_account is
        retired — a LOCAL analysis has no account and emits none, byte-identical
        to the prior null read).
        """
        if self.cfg_hpc_system is not None:
            return self.cfg_hpc_system.default_account
        return None

    def _resolve_gpu_alloc_mode(self) -> Literal["gres", "gpus"]:
        """Phase-2 (R3): gres-vs-gpus GPU allocation channel.

        Prefer cfg_hpc_system.gpu_allocation_flavor; default to "gpus" when no
        flavor is declared (Phase-4 4c: the legacy
        cfg_system.preferred_slurm_option_for_allocating_gpus is retired; the
        "gpus" default preserves the legacy ``or "gpus"`` behavior).
        """
        if self.cfg_hpc_system is not None and self.cfg_hpc_system.gpu_allocation_flavor is not None:
            return self.cfg_hpc_system.gpu_allocation_flavor
        # Phase-4 (4c): legacy cfg_system.preferred_slurm_option_for_allocating_gpus
        # retired; default to "gpus" when no flavor is declared (byte-identical to
        # the legacy ``or "gpus"`` default).
        return "gpus"

    def _resolve_gpus_per_node(self, partition_name: str | None) -> int:
        """Phase-2 (R3): per-node GPU topology for the given (sim) partition.

        Prefer the PartitionSpec.gpus_per_node of the named partition when
        cfg_hpc_system is present AND the partition is declared AND it carries a
        gpus_per_node value; else 0 (Phase-4 4d: the legacy
        cfg_analysis.hpc_gpus_per_node is retired; 0 == no GPU topology per the
        _build_resource_block contract).
        """
        if self.cfg_hpc_system is not None and partition_name is not None:
            spec = self.cfg_hpc_system.partitions.get(partition_name)
            if spec is not None and spec.gpus_per_node is not None:
                return spec.gpus_per_node
        return 0

    def _resolve_cpus_per_node(self, partition_name: str | None) -> int | None:
        """Phase-4 (4a, unconsumed): per-node CPU topology for the given partition.

        Prefer the PartitionSpec.cpus_per_node of the named partition when
        cfg_hpc_system is present AND the partition is declared AND it carries a
        cpus_per_node value; else None (Phase-4 4d: the legacy
        cfg_analysis.hpc_cpus_per_node is retired; the `not isinstance(..., int)`
        guard at the one-big-job dry-run preserves graceful-skip on None).
        """
        if self.cfg_hpc_system is not None and partition_name is not None:
            spec = self.cfg_hpc_system.partitions.get(partition_name)
            if spec is not None and spec.cpus_per_node is not None:
                return spec.cpus_per_node
        return None

    def _resolve_gpu_hardware(self, partition_name: str | None) -> str | None:
        """Phase-4 (4a, unconsumed): the GPU arch string for the given partition.

        Prefer the PartitionSpec.gpu_hardware of the named partition (D1 Option-A:
        gpu_hardware lives on the partition spec) when cfg_hpc_system is present
        AND the partition is declared AND it carries a gpu_hardware value; else
        None (Phase-4 4c: the legacy cfg_system.gpu_hardware is retired). The
        system.py reads switched to the DI'd attribute in 4c; this builder helper
        resolves the per-partition value to inject + to emit in SLURM directives.
        """
        if self.cfg_hpc_system is not None and partition_name is not None:
            spec = self.cfg_hpc_system.partitions.get(partition_name)
            if spec is not None and spec.gpu_hardware is not None:
                return spec.gpu_hardware
        # Phase-4 (4c): legacy cfg_system.gpu_hardware retired; None when the
        # partition declares no GPU hardware / no hpc_system_config is present.
        return None

    def _resolve_additional_modules(self) -> str | None:
        """Phase-4 (4a, unconsumed): the `module load` argument string.

        Prefer cfg_hpc_system.additional_modules (a ``list[str]``) joined on a
        single space into the ``str`` the `module load {modules}` emitters expect
        (module names are space-free, so the join is lossless — D1 str↔list
        bridge); else the legacy cfg_system field (already a space-joined str).
        Returns None when neither source supplies modules so the existing
        ``if modules:`` guards stay byte-identical. Added additive/unconsumed in
        4a; consumers wire when the legacy fields are removed in 4c.
        """
        # Phase-4 (4c): single join site is the config.hpc_system free function
        # (also used by the GPU-compile runners, which have no builder instance);
        # the legacy cfg_system fallback is retired.
        return resolve_additional_modules(self.cfg_hpc_system)

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
            ctx = self._make_rule_emission_context(static_backend=self._get_report_cfg_static_backend())

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
        {self._container_process_prefix}{self.python_executable} -m hhemt.process_timeseries_runner \\
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
        {self.python_executable} -m hhemt.consolidate_workflow \\
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
            f'"_status/d_process_{model_type}_evt-{{event_id}}_complete.flag"' for model_type in enabled_models
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
        {self.python_executable} -m hhemt.consolidate_workflow \\
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
        from hhemt.scenario import compute_event_id_slug

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
        # Phase 2 (R3): prefer the partition-sourced GPU topology + allocation
        # flavor from cfg_hpc_system when present; byte-identical legacy reads
        # when absent. The sim rules target hpc_ensemble_partition (1591/1772),
        # so the topology is keyed on that partition's PartitionSpec.
        gpus_per_node_config = self._resolve_gpus_per_node(self.cfg_analysis.hpc_ensemble_partition)
        gpu_alloc_mode = self._resolve_gpu_alloc_mode()

        # Get absolute path to conda environment file using helper
        conda_env_path = self._get_conda_env_path()
        config_args = self._get_config_args()
        # Phase-4 (4c): the SETUP rule (compiles the GPU binary) and the SIM rule
        # (runs it) resolve GPU hardware/backend from the ENSEMBLE (sim) partition's
        # PartitionSpec — the compile/run target — via --target-partition. NB: the
        # ensemble partition (NOT the processing partition) is the GPU-compile source
        # for the setup rule too, because the binary it builds runs on the sim
        # partition. Other rules keep the shared config_args (no --target-partition).
        gpu_compile_config_args = self._get_config_args(target_partition=self.cfg_analysis.hpc_ensemble_partition)
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
        {self.python_executable} -m hhemt.setup_workflow \\
            {gpu_compile_config_args} \\
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
            gpu_hardware=self._resolve_gpu_hardware(self.cfg_analysis.hpc_ensemble_partition),
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

        # ADR-2 (OE-1 anti-drift): the rule_all + render_report input stems for
        # the per-sim figures derive from the SAME single-source helper as the
        # rule OUTPUTS, so a future stem-grammar change in report_plot_ids cannot
        # desync inputs from outputs (-> MissingInputException / Gotcha 37/39).
        # {event_id} stays a literal Snakemake wildcard for the emitted expand().
        _pfd_per_sim = _plot_output_template(
            renderer_kind="peak_flood_depth",
            subdir="plots/per_sim/{event_id}",
            event_id="{event_id}",
        ).replace("__OUTPUT_EXT__", _ext["per_sim_peak_flood_depth"])
        _cf_per_sim = _plot_output_template(
            renderer_kind="conduit_flow",
            subdir="plots/per_sim/{event_id}",
            event_id="{event_id}",
        ).replace("__OUTPUT_EXT__", _ext["per_sim_conduit_flow"])

        _formats = report_formats if report_formats is not None else ["zip"]
        render_targets_in_rule_all = "".join(f',\n        "analysis_report.{fmt}"' for fmt in _formats)
        snakefile_content = f'''# Auto-generated by TRITONSWMM_analysis

import os
import glob
import subprocess
from datetime import datetime as _dt
from hhemt.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("hhemt")
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
        expand("{_pfd_per_sim}", event_id=SIM_IDS),
        expand("{_cf_per_sim}", event_id=SIM_IDS),
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
        {self.python_executable} -m hhemt.export_scenario_status \\
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
        {self.python_executable} -m hhemt.prepare_scenario_runner \\
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
    retries: {self._resolved_simulate_retries()}
    log: "{log_dir_str}/sims/{model_type}_evt-{{event_id}}.log"
    conda: "{conda_env_path}"
    threads: {model_threads}
    params:
        event_iloc=lambda wildcards: ILOC_BY_EVENT_ID[wildcards.event_id],
    resources:
{model_resources}
    shell:
        """
        {self.python_executable} -m hhemt.run_simulation_runner \\
            --event-iloc {{params.event_iloc}} \\
            {gpu_compile_config_args} \\
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
            # The wait-rule now detects job death in-loop (SLURM-liveness probe
            # in wait_for_sentinel_runner.py), so the cap is a pure backstop —
            # the operator's hpc_max_wait_for_inflight_min ceiling directly, not
            # the old walltime-derived min(). Queue-time is no longer truncated.
            wait_walltime_cap_min = self.cfg_analysis.hpc_max_wait_for_inflight_min
            _prefix = f"run_{model_type}_evt-"
            for rule_token in sorted(alive_by_token or {}):
                if not rule_token.startswith(_prefix):
                    continue
                event_id = rule_token[len(_prefix) :]
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
        # Registry-driven plot-rule dispatch (P1b / TO-8): the active set's
        # renderer_selection (default → the six common renderers in order) is the
        # single source of which renderers fire. Replaces the hardcoded call list.
        snakefile_content += self._emit_active_set_plot_rules(
            self._resolve_active_reporting_set(self.analysis),
            input_flag="_status/e_consolidate_complete.flag",
        )
        # export_scenario_status: set-invariant, non-figure workflow rule
        # (scenario_status.csv + a top-level localrules:); intentionally NOT a
        # renderer_selection entry (Option B); emitted as a dispatcher sibling in
        # its historical trailing position (multisim has no conditional renderers,
        # so no interleave hook).
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
        expand("{_pfd_per_sim}", event_id=SIM_IDS),
        expand("{_cf_per_sim}", event_id=SIM_IDS),
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
        {self.python_executable} -m hhemt.render_report_runner \\
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
            if "tritonswmm" in enabled and scen_paths.swmm_hydraulics_rpt is not None:
                sources.append(
                    {
                        "path": _os.path.relpath(
                            str(Path(scen_paths.swmm_hydraulics_rpt).resolve()),
                            analysis_root,
                        ),
                        "variables": ["Flow Routing Continuity error (%)"],
                    }
                )
            elif "swmm" in enabled and scen_paths.swmm_full_rpt_file is not None:
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
        {self.python_executable} -m hhemt.export_scenario_status \\
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
    from hhemt.report_renderers._figure_emission import (
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
    from hhemt.report_renderers._figure_emission import (
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
            output_path_template=_plot_output_template(
                renderer_kind="peak_flood_depth",
                subdir="plots/per_sim/{event_id}",
                event_id="{event_id}",
            ),
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
            output_path_template=_plot_output_template(
                renderer_kind="conduit_flow",
                subdir="plots/per_sim/{event_id}",
                event_id="{event_id}",
            ),
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
        assert isinstance(
            self.cfg_analysis.local_cpu_cores_for_workflow, int
        ), "local_cpu_cores_for_workflow must be specified for local runs"
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
            # Phase-4 (4d): concurrency cap moved to hpc_system_config.max_concurrent_jobs.
            max_concurrent = self.cfg_hpc_system.max_concurrent_jobs if self.cfg_hpc_system else None
            assert isinstance(
                max_concurrent, int
            ), "hpc_system_config.max_concurrent_jobs is required for generate_snakemake_config (slurm mode)"
            # Modern executor mode: uses 'executor: slurm' with job steps
            config.update(
                {
                    "executor": "slurm",
                    "jobs": max_concurrent,
                    "latency-wait": 60,
                    "max-jobs-per-second": 5,
                    "max-status-checks-per-second": 10,
                    # Auto-retry jobs that SLURM marks FAILED (e.g. transient
                    # `srun` step glitches: "Unable to confirm allocation ...
                    # Invalid job id"). NOTE: this does NOT rescue a job that
                    # hangs in SLURM state RUNNING after its inner command
                    # exits — Snakemake waits for that job until its own
                    # --time wall-limit regardless of restart-times. The
                    # hung-RUNNING case is a SLURM-infra transient not
                    # fixable toolkit-side. All reprocess-path rules
                    # (plot/consolidate/render) are idempotent re-derivations,
                    # so a retried payload re-run is safe. A walltime kill is a
                    # SLURM TIMEOUT (a terminal FAILED state, not hung-RUNNING),
                    # so it IS rescued by restart-times — this is what drives the
                    # hotstart-resume sweep's automatic completion.
                    #
                    # Global baseline for directive-less rules (process/consolidate/
                    # plot/render carry no per-rule retries:, so they inherit this).
                    # Simulate rules carry an explicit retries: {simulate} directive
                    # that OVERRIDES this (snakemake 9.15.0 Rule.restart_times,
                    # rules.py:158-165). Override-resolved per the FQ3 consume-site.
                    "restart-times": (
                        self._override_hpc_restart_times_other
                        if self._override_hpc_restart_times_other is not None
                        else self.cfg_analysis.hpc_restart_times_other
                    ),
                    "default-resources": [
                        "nodes=1",
                        "mem_mb=2000",
                        "runtime=30",
                        f"slurm_partition={slurm_partition}",
                        f"slurm_account={self._resolve_account()}",
                    ],
                    # NOTE: the legacy `slurm: {sbatch: {...}}` block was deleted
                    # in Phase 2 — it is a `--cluster`-generic-executor key shape
                    # the modern `executor: slurm` plugin ignores (slurm_partition/
                    # slurm_account come from default-resources above); snakemake 9's
                    # profile parser silently drops it. (snakemake FQ2 open-finding 2.)
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
        import hhemt.utils as ut

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

        modules = self._resolve_additional_modules()
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

conda activate hhemt

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
            # Phase-3 (R4): per-node GPU topology for the sbatch --gres header.
            # Prefer cfg_hpc_system.partitions[hpc_ensemble_partition].gpus_per_node;
            # else the legacy cfg_analysis.hpc_gpus_per_node. The `> 0` guard
            # preserves the original assert's intent (GPUs requested ⇒ a positive
            # per-node count is required) — _resolve_gpus_per_node resolves an
            # absent value to 0, which is a misconfiguration in the GPU branch.
            gpus_per_node = self._resolve_gpus_per_node(self.cfg_analysis.hpc_ensemble_partition)
            assert (
                isinstance(gpus_per_node, int) and gpus_per_node > 0
            ), "hpc_gpus_per_node required when using GPUs in 1_job_many_srun_tasks mode"
            # --gres/--gpus-per-node are per-node, SLURM will multiply by --nodes automatically
            gpu_hardware = self._resolve_gpu_hardware(self.cfg_analysis.hpc_ensemble_partition)
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
                    "cfg_hpc_system.default_account (or cfg_analysis.hpc_account)",
                    str(self._resolve_account()),
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
                    "cfg_hpc_system.partitions[hpc_ensemble_partition].gpus_per_node "
                    "(or cfg_analysis.hpc_gpus_per_node) + cfg_system.gpu_hardware",
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
#SBATCH --account={self._resolve_account()}
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
            import hhemt.utils as ut

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
            import hhemt.utils as ut

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
            import hhemt.utils as ut

            batch_log_path = self.analysis.analysis_paths.analysis_log_directory / "_slurm_logs"
            batch_log_path.mkdir(exist_ok=True, parents=True)

            additional_sbatch_args = ""
            if self.cfg_analysis.additional_SBATCH_params:
                additional_sbatch_args = "#SBATCH "
                additional_sbatch_args += "\n#SBATCH ".join(self.cfg_analysis.additional_SBATCH_params)

            modules = self._resolve_additional_modules()
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

conda activate hhemt

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
            _account = self._resolve_account()
            if _account:
                account_directive = f"#SBATCH --account={_account}\n"

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
            from hhemt import utils as ut

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
conda activate hhemt

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
            # Phase-4 (4d): login_node moved to hpc_system_config.login_node.
            _login_node = self.cfg_hpc_system.login_node if self.cfg_hpc_system else None
            reattach_node = _login_node or submission_node

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
        modules_str = self._resolve_additional_modules()

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
        override_hpc_restart_times_simulate: int | None = None,
        override_hpc_restart_times_other: int | None = None,
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

        # Stash the retry overrides on the builder (P1 Decision 4, FQ3 SITE 5) BEFORE
        # any generate_snakefile_content/generate_snakemake_config call below, so the
        # global-baseline (_other) and per-rule simulate (_simulate) emission sites read
        # them. None means "use the config knob". Stored here rather than threaded
        # through generate_snakemake_config's ~5 call sites.
        self._override_hpc_restart_times_simulate = override_hpc_restart_times_simulate
        self._override_hpc_restart_times_other = override_hpc_restart_times_other

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

            # Sweep permanently-failed rules ONLY when we actually waited on the
            # allocation. _submit_single_job_workflow sbatch-submits and returns the
            # job_id without blocking unless wait_for_completion=True; sweeping a
            # still-running allocation would read an empty _status/_failed/ and report a
            # false-clean (the detached-mode hazard — captured follow-up otherwise).
            if wait_for_completion:
                result = self._augment_result_with_partial_failures(result)

            # mechanism (b): record the planned sim-token set under _status/_queued/
            # AFTER the submit returned success (write-after-launch). 1_job_many_srun_tasks
            # is toolkit-owns-sbatch — the allocation jobid enables the wait-runner in-loop
            # liveness probe (R8) for a PENDING-recovered wait-rule.
            if isinstance(result, dict) and result.get("success", True):
                self._write_queued_sentinels(
                    self._planned_sim_tokens(), result.get("job_id"), self.analysis_paths.analysis_dir
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

            # mechanism (b): record the planned sim-token set under _status/_queued/
            # AFTER submit-success. batch_job is executor-owns-sbatch — the executor
            # assigns per-rule ids the toolkit never sees, so jobid is null and the
            # token is held on PRESENCE bounded by the mtime fail-safe (F1-O3, R12).
            if isinstance(result, dict) and result.get("success", True):
                self._write_queued_sentinels(self._planned_sim_tokens(), None, self.analysis_paths.analysis_dir)

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
            # Blocking path: sweep permanently-failed rules into the result so a single
            # non-retryable failure surfaces even though --keep-going completed the rest.
            result = self._augment_result_with_partial_failures(result)
        else:  # slurm
            # Detached (_run_snakemake_slurm_detached returns before the run finishes):
            # no in-process completion point to sweep — captured follow-up.
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
        without re-running simulations. Shares ``{analysis_dir}/.snakemake/``
        with the run path and uses ``--nolock`` so it coexists with queued/
        running SLURM sim workers without touching the shared lock; the
        ``_status/_orchestrator/`` liveness gate refuses fast only when a live
        orchestration driver for the same analysis exists.

        Parameters
        ----------
        start_with
            Downstream stage to re-fire from. See
            :func:`hhemt.reprocess_snakefile_generator.generate_reprocess_snakefile`
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
        from hhemt.reprocess_snakefile_generator import (
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
        # ``.snakemake_reprocess/_status/``. The reprocess Snakefile lives at a
        # distinct path (``Snakefile.reprocess``). NOTE: Snakemake locks are
        # keyed on the working-directory input/output file SET (a file-set
        # intersection in persistence.py), NOT the Snakefile path — so a
        # distinct Snakefile provides NO lock isolation. Coexistence with a
        # live ``rule run_*`` driver is therefore handled by ``--nolock`` on
        # the reprocess invocation plus the ``_status/_orchestrator/`` liveness
        # gate, which is the concurrency authority (see the "reprocess uses
        # --nolock + orchestrator-liveness sentinel" decision doc).
        reprocess_working_dir = self.analysis_paths.analysis_dir

        # Build the snakemake command. Reuses the run/submit base command and
        # adds ``--rerun-triggers mtime`` so downstream rules only re-fire
        # when outputs are missing or older — the surgical reprocess intent.
        cmd_args = self._get_snakemake_base_cmd() + [
            "--snakefile",
            str(snakefile_path),
            "--rerun-triggers",
            "mtime",
            "--nolock",
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

        # Reprocess concurrency gate (R3): refuse fast with a WorkflowError —
        # never input() — when a live orchestration DRIVER for this analysis
        # exists. Default-safe when no _orchestrator/ sentinel is present (R6),
        # and coexists with queued/running _submitted/ sim WORKERS (R2). Skipped
        # for dry runs (planning only — nothing is submitted, no zarr is written).
        import os

        from hhemt import orchestrator_sentinels as _osent

        driver_id = _osent.new_driver_id()
        remove_self_sentinel = False
        if not dry_run:
            gate_err = self._orchestrator_liveness_gate(
                analysis_dir=self.analysis_paths.analysis_dir,
                exclude_driver_id=driver_id,
            )
            if gate_err is not None:
                raise gate_err
            # Reprocess self-sentinel (R5): write-own-then-scan-others mutual
            # exclusion vs a second reprocess. Always mode="local"/pid=os.getpid()
            # — reprocess overrides 1_job_many_srun_tasks -> batch_job but never
            # allocates its own tmux/sbatch driver, so it never produces a
            # tmux/sbatch self-sentinel. A blocking-local reprocess removes it in
            # the finally; a detached Popen reprocess leaves it (reclaimed by the
            # gate's ps -p {pid} arm once this login-node process exits).
            _osent.write_orchestrator_sentinel(
                self.analysis_paths.analysis_dir,
                driver_id=driver_id,
                workflow_submission_mode="local",
                pid=os.getpid(),
            )
            remove_self_sentinel = True

        try:
            # Facade — reconciliation against analysis_dir/_status/_submitted/.
            # skip_lock_check=True bypasses the toolkit-side input() prompt; the
            # orchestrator-liveness gate above is the concurrency authority and
            # --nolock is on the subprocess. Phase 1's at-most-once guard still
            # protects reprocess from a parallel live sim driver double-submitting.
            self._pre_snakemake_invocation_guards(
                snakefile_path,
                dry_run=dry_run,
                verbose=verbose,
                working_dir=reprocess_working_dir,
                skip_lock_check=True,
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
            # Detached driver: leave the self-sentinel for the gate's liveness
            # reclaim — do NOT remove it in the finally.
            remove_self_sentinel = False
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
        finally:
            if remove_self_sentinel:
                _osent.remove_orchestrator_sentinel(self.analysis_paths.analysis_dir, driver_id)

    def submit_static_plots_workflow(
        self,
        *,
        resolved_static_plot_configs: "list[Path]",
        static_config_ids: "list[str] | None" = None,
        execution_mode: Literal["auto", "local", "slurm"] = "auto",
        dry_run: bool = False,
        verbose: bool = True,
    ) -> dict:
        """Submit the publication static-plots workflow (ADR-8).

        Writes ``{analysis_dir}/Snakefile.static`` (one bare-output rule per
        static-plot ID + ``rule all``; NO ``render_report``, NO ``report()``
        wrapper) and dispatches it via the SAME execution plumbing as
        run()/reprocess(). Modeled verbatim on :meth:`submit_reprocess_workflow`
        with three differences: (a) the static Snakefile generator replaces the
        reprocess one; (b) NO ``--rerun-triggers mtime`` — the static path uses
        Snakemake's DEFAULT rerun triggers (a first-run static render has no
        prior outputs, so the surgical mtime-only intent of reprocess does not
        apply); (c) ``static_config_ids`` filters the harvested rule set to the
        named subset. ``static_plots/{plot_id}.{ext}`` outputs are disjoint from
        the run/reprocess output set, so ``--nolock`` + ``skip_lock_check=True``
        plus the orchestrator-liveness gate are the concurrency contract (the
        Snakemake working-dir lock is keyed on the input/output file SET, so a
        distinct ``Snakefile.static`` gives NO lock isolation).

        Parameters
        ----------
        resolved_static_plot_configs
            The override-resolved list of per-plot config YAML paths, computed
            at the :meth:`TRITONSWMM_analysis.static_plots` facade and threaded
            DOWN (NOT re-read from ``cfg_analysis`` here) so a passed override is
            honored — the anti-facade-drift contract (master Decision-threading).
        static_config_ids
            When non-None, restrict the emitted rules to configs whose
            ``plot_id`` is in this set; None emits a rule for every resolved
            config.
        execution_mode
            ``"auto"`` detects SLURM context; ``"local"`` / ``"slurm"`` force it.
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
        from hhemt.static_snakefile_generator import write_static_snakefile

        effective_method = self.cfg_analysis.multi_sim_run_method
        if execution_mode == "auto":
            mode: Literal["local", "slurm"] = "slurm" if self.analysis.in_slurm else "local"
        else:
            mode = execution_mode  # type: ignore[assignment]

        if verbose:
            print(
                f"[Snakemake] Submitting static-plots workflow (mode={mode}, method={effective_method})",
                flush=True,
            )

        # Write the static-plots Snakefile. The resolved config list is threaded
        # DOWN from the facade (anti-drift); static_config_ids subset-filtering is
        # applied at the harvest site inside the generator.
        snakefile_path = write_static_snakefile(
            self.analysis,
            static_plot_configs=resolved_static_plot_configs,
            config_args_str=self._get_config_args(),
            static_backend=self._get_report_cfg_static_backend(),
            static_config_ids=static_config_ids,
        )
        if verbose:
            print(f"[Snakemake] Static-plots Snakefile generated: {snakefile_path}", flush=True)

        # Logs.
        logs_dir = self.analysis_paths.analysis_log_directory
        logs_dir.mkdir(parents=True, exist_ok=True)
        logfile_name = "snakemake_static_plots_dry_run.log" if dry_run else "snakemake_static_plots.log"
        snakemake_logfile = logs_dir / logfile_name

        # Working dir is analysis_dir itself (relative artifact paths resolve
        # there). Snakemake locks are keyed on the working-dir input/output file
        # SET, NOT the Snakefile path, so a distinct Snakefile.static gives NO
        # lock isolation — coexistence is handled by --nolock + the
        # orchestrator-liveness gate (see submit_reprocess_workflow).
        static_working_dir = self.analysis_paths.analysis_dir

        # Build the snakemake command. Reuses the run/submit base command. NO
        # ``--rerun-triggers mtime`` — the static path uses Snakemake's DEFAULT
        # rerun triggers (a first-run static render has no prior outputs).
        cmd_args = self._get_snakemake_base_cmd() + [
            "--snakefile",
            str(snakefile_path),
            "--nolock",
        ]

        if mode == "local":
            local_cores = self.cfg_analysis.local_cpu_cores_for_workflow
            assert isinstance(local_cores, int), "local_cpu_cores_for_workflow must be specified for local runs"
            if local_cores > 1:
                cmd_args.extend(["--cores", str(local_cores)])
            else:
                cmd_args.extend(["--cores", "1"])
        else:  # slurm
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

        # Static-render concurrency gate: refuse fast with a WorkflowError — never
        # input() — when a live orchestration DRIVER for this analysis exists.
        # Default-safe when no _orchestrator/ sentinel is present, and coexists
        # with queued/running _submitted/ sim WORKERS. Skipped for dry runs.
        import os

        from hhemt import orchestrator_sentinels as _osent

        driver_id = _osent.new_driver_id()
        remove_self_sentinel = False
        if not dry_run:
            gate_err = self._orchestrator_liveness_gate(
                analysis_dir=self.analysis_paths.analysis_dir,
                exclude_driver_id=driver_id,
            )
            if gate_err is not None:
                raise gate_err
            _osent.write_orchestrator_sentinel(
                self.analysis_paths.analysis_dir,
                driver_id=driver_id,
                workflow_submission_mode="local",
                pid=os.getpid(),
            )
            remove_self_sentinel = True

        try:
            # skip_lock_check=True bypasses the toolkit-side input() prompt; the
            # orchestrator-liveness gate above is the concurrency authority and
            # --nolock is on the subprocess.
            self._pre_snakemake_invocation_guards(
                snakefile_path,
                dry_run=dry_run,
                verbose=verbose,
                working_dir=static_working_dir,
                skip_lock_check=True,
            )

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
                        "message": (f"Snakemake static-plots failed. See {snakemake_logfile} for details."),
                        "snakemake_logfile": snakemake_logfile,
                    }
                if verbose:
                    print("[Snakemake] Static plots completed successfully", flush=True)
                self.analysis._refresh_log()
                return {
                    "success": True,
                    "mode": "local",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": "Static plots completed successfully",
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
                    f"[Snakemake] Static plots submitted to background (PID: {proc.pid})",
                    flush=True,
                )
            # Detached driver: leave the self-sentinel for the gate's liveness reclaim.
            remove_self_sentinel = False
            self.analysis._refresh_log()
            return {
                "success": True,
                "mode": "slurm",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Static plots submitted to SLURM (detached)",
                "process": proc,
                "snakemake_logfile": snakemake_logfile,
            }
        finally:
            if remove_self_sentinel:
                _osent.remove_orchestrator_sentinel(self.analysis_paths.analysis_dir, driver_id)

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
                    # EXEMPT-DU: status-flag
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
                # EXEMPT-DU: status-flag
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
                    # EXEMPT-DU: status-flag
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

        # F2 srun-step job-id-aliasing guard: under 1_job_many_srun_tasks every
        # concurrent srun-step sim records the SAME allocation $SLURM_JOB_ID, so
        # `sacct -X` returns ONE allocation-summary row for N tokens. A terminal
        # allocation State would then mass-classify all N still-running tokens
        # DEAD (empirically confirmed 2026-06-13: a TIMEOUT allocation row is read
        # by every aliased token). For any jobid shared by >=2 tokens, refuse to
        # classify from the shared allocation row — fall through to the mtime-age
        # fail-safe instead. (F1 step-level `sacct -j` per-token classification is
        # the optional precision upgrade; F2 guard is the minimum-correctness fix.)
        _jid_counts = Counter(j for _, j in marker_less_alive if j)
        _aliased_jids = {j for j, c in _jid_counts.items() if c >= 2}

        still_alive: list[tuple[str, str]] = []
        cleared: list[_ClearedToken] = []
        for rule_token, jid in marker_less_alive:
            sentinel = submitted_dir / f"{rule_token}.json"
            row = states.get(jid) if jid else None
            if jid in _aliased_jids:
                # aliased: the single allocation row is not per-token authoritative
                row = None
            if row is not None:
                state, _exit, reason = row
                if state in _SACCT_DEAD_STATES:
                    # EXEMPT-DU: status-flag
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
                # EXEMPT-DU: status-flag
                sentinel.unlink(missing_ok=True)
                cleared.append(_ClearedToken(rule_token, jid or "(no jobid)", "UNKNOWN", "purged/age-exceeded"))
            else:
                still_alive.append((rule_token, jid))
        return still_alive, cleared

    def _orchestrator_liveness_gate(
        self,
        analysis_dir: Path | None = None,
        *,
        exclude_driver_id: str | None = None,
    ) -> "WorkflowError | None":
        """Return a WorkflowError if a LIVE orchestration driver exists for this
        analysis, else None. Reclaims dead/stale sentinels in passing.

        Default-safe: no ``_orchestrator/`` sentinels ⇒ returns None (proceed).
        ``exclude_driver_id`` skips the caller's own self-sentinel (reprocess).
        Reuses ``_classify_stale_via_sacct`` (jobid arm) and the cancel_workflow
        ps/tmux probes — no new liveness probe is authored (R10).
        """
        import subprocess

        from hhemt.orchestrator_sentinels import (
            read_orchestrator_sentinels,
        )

        base = analysis_dir if analysis_dir is not None else self.analysis_paths.analysis_dir
        sentinels = [s for s in read_orchestrator_sentinels(base) if s.get("driver_id") != exclude_driver_id]
        if not sentinels:
            return None  # default-safe: no sentinel ⇒ proceed (R6)

        live: list[str] = []
        for s in sentinels:
            mode = s.get("workflow_submission_mode")
            path = Path(s["_path"])
            alive = False
            if mode == "local":
                pid = s.get("pid")
                alive = bool(pid) and subprocess.run(["ps", "-p", str(pid)], capture_output=True).returncode == 0
            elif mode in ("tmux", "batch_job"):
                session = s.get("tmux_session_name")
                alive = bool(session) and (
                    subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0
                )
            elif mode == "1_job_many_srun_tasks":
                jobid = s.get("slurm_jobid") or ""
                # Use the direct liveness primitive — NOT _classify_stale_via_sacct,
                # which resolves its sentinel path under _status/_submitted/{token}.json
                # and, on an UNKNOWN (sacct-lagging) jobid, mtime-tiebreaks against that
                # NON-EXISTENT _orchestrator path -> OSError -> always-DEAD, a
                # false-proceed against a live single-job driver (the E3 hazard the gate
                # exists to prevent). _slurm_job_is_live has no _submitted/-path coupling.
                alive = bool(jobid) and _slurm_job_is_live(jobid)
            else:
                # Unknown mode: classify alive conservatively (do not silently
                # proceed against an unrecognized driver record).
                alive = True

            if alive:
                live.append(f"{s.get('driver_id')} (mode={mode})")
            else:
                # EXEMPT-DU: status-flag
                path.unlink(missing_ok=True)  # reclaim dead/stale (R4)
                print(
                    f"[orchestrator-gate] reclaimed stale sentinel {s.get('driver_id')} (mode={mode}) at {path}",
                    file=sys.stderr,
                    flush=True,
                )

        if live:
            return WorkflowError(
                phase="reprocess pre-submission orchestrator-liveness gate",
                return_code=1,
                stderr=(
                    "Refusing reprocess: a live orchestration driver for this analysis "
                    f"exists ({'; '.join(live)}). reprocess coexists with queued/running "
                    "SLURM sim workers but must not run concurrently with a live run()/"
                    "reprocess DRIVER (unarbitrated concurrent consolidate-zarr write). "
                    "Cancel the live driver (or wait for it to finish) and retry."
                ),
            )
        return None

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
        absolute ``{python_exe} -m hhemt.wait_for_sentinel_runner``
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
            # Fail-fast: a wait-rule observing a _failed/ marker means the original
            # sim died; re-polling cannot change that. retries: 0 keeps the wait-rule
            # from inheriting the global restart-times baseline (= hpc_restart_times_other)
            # and re-dispatching the poll-runner (FQ2). Correctness, not just cost.
            f"    retries: 0\n"
            f"    resources: cpus_per_task=1, mem_mb=100\n"
            f"    shell:\n"
            f'        "{python_exe} -m hhemt.wait_for_sentinel_runner "\n'
            f'        "--rule-token {rule_token} "\n'
            f'        "--flag-output {{output}} "\n'
            f'        "--analysis-dir {analysis_dir} "\n'
            f'        "--max-wait-minutes {wait_walltime_cap_min}"\n\n'
        )

    def _pre_delete_guards(
        self,
        *,
        override_in_flight: bool,
        snakefile_name: str = "Snakefile.delete",
        working_subdir: str = ".snakemake_delete",
    ) -> None:
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

        # (1) Lock-check scoped to the (parametrized) delete namespace (C.5).
        # Defaults preserve analysis.delete()'s Snakefile.delete + .snakemake_delete/;
        # the scoped reprocess-delete path passes Snakefile.reprocess_delete +
        # .snakemake_reprocess_delete/ so the guard inspects the correct lock dir
        # (F-I Flag 2 — without this the guard would gate on the wrong namespace).
        snakefile_delete = analysis_dir / snakefile_name
        if snakefile_delete.exists():
            self._check_and_clear_snakemake_lock(
                snakefile_delete,
                dry_run=False,
                verbose=True,
                working_dir=analysis_dir / working_subdir,
            )

        # (2) Sentinel classification (no reclaim — destructive sentinel
        # cleanup belongs to the delete-consolidation runner, not the
        # preflight guard).
        submitted_dir = analysis_dir / "_status" / "_submitted"
        sentinels = sorted(submitted_dir.glob("*.json")) if submitted_dir.exists() else []
        alive = self._classify_live_sentinels(sentinels, reclaim_dead=False)

        # (2b) Sweep _status/_queued/ too (mechanism b): a still-PENDING sim
        # (SLURM-accepted, worker not yet started, so no _submitted/ sentinel) is
        # in-flight and must block a delete exactly like a running sim. Presence-only
        # (no reclaim — consistent with the no-reclaim stance of (2); a stale orphan
        # _queued/ is aged out by the run-path reconcile's mtime fail-safe, or the
        # operator passes override_in_flight — never destructive cleanup here).
        queued_dir = analysis_dir / "_status" / "_queued"
        if queued_dir.is_dir():
            for qpath in sorted(queued_dir.glob("*.json")):
                tok = qpath.stem
                if any(
                    (analysis_dir / "_status" / d / f"{tok}.json").exists()
                    for d in ("_submitted", "_completed", "_failed")
                ):
                    continue  # superseded — already covered by the _submitted/ sweep
                try:
                    jid = str(json.loads(qpath.read_text()).get("slurm_jobid") or "")
                except (json.JSONDecodeError, OSError):
                    jid = ""
                alive.append((tok, jid))

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
        from hhemt.scenario import compute_event_id_slug

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
                f"    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n"
                f"    shell:\n"
                f'        "{python_exe} -m hhemt.delete_scenario_runner "\n'
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
            f"    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n"
            f"    shell:\n"
            f'        "{python_exe} -m hhemt.delete_consolidation_runner "\n'
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
                f"    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n"
                f"    shell:\n"
                f'        "{python_exe} -m hhemt.delete_subanalysis_runner "\n'
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
            f"    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n"
            f"    shell:\n"
            f'        "{python_exe} -m hhemt.delete_consolidation_runner "\n'
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
        working_subdir: str = ".snakemake_delete",
        logfile_name: str = "snakemake_delete.log",
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
        snakemake_dir = analysis_dir / working_subdir
        snakemake_dir.mkdir(exist_ok=True)
        logs_dir = self.analysis_paths.analysis_log_directory
        logs_dir.mkdir(parents=True, exist_ok=True)
        logfile = logs_dir / logfile_name

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
            _max_concurrent = self.cfg_hpc_system.max_concurrent_jobs if self.cfg_hpc_system else None
            cmd_args += ["--cores", str(_max_concurrent or 1)]
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

    def _build_reprocess_delete_snakefile_content(self, *, start_with: str) -> str:
        """Build the SCOPED reprocess-delete Snakefile (R8). Non-sensitivity:
        per-scenario delete_processed_{slug} (processed/-only) when
        start_with=='process' + one delete_reprocess_consolidation (master zarr).
        Sensitivity (D-scope Option C — per-sub fan-out mirroring the existing
        delete_subanalysis_{sa} granularity): one delete_subanalysis_reprocess_{sa}
        per sub (deletes that sub's processed/ across all events when
        start_with=='process' + that sub's analysis_datatree.zarr) + one master
        delete_reprocess_consolidation (sensitivity_datatree.zarr) fanning in on
        the per-sub flags. Flags → _status/_deleting_reprocess/."""
        from hhemt.scenario import compute_event_id_slug

        master_dir = str(self.analysis_paths.analysis_dir)
        python_exe = self.python_executable
        rules: list[str] = []
        leaf_flags: list[str] = []

        def _proc_rule(rule_suffix: str, event_id: str, target_analysis_dir: str, flag: str) -> str:
            return (
                f"rule delete_processed_{rule_suffix}:\n"
                f'    output:\n        "{flag}"\n'
                f"    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n"
                f"    shell:\n"
                f'        "{python_exe} -m hhemt.delete_processed_runner "\n'
                f'        "--event-id {event_id} --analysis-dir {target_analysis_dir}"\n\n'
            )

        def _zarr_rule(rule_suffix: str, target_analysis_dir: str, flag: str, inputs: list[str]) -> str:
            inp = ",\n        ".join(f'"{f}"' for f in inputs)
            return (
                f"rule delete_reprocess_zarr_{rule_suffix}:\n"
                f"    input:\n        {inp if inputs else ''}\n"
                f'    output:\n        "{flag}"\n'
                f"    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n"
                f"    shell:\n"
                f'        "{python_exe} -m hhemt.delete_reprocess_zarr_runner "\n'
                f'        "--analysis-dir {target_analysis_dir}"\n\n'
            )

        def _subanalysis_rule(rule_suffix: str, sa_id: str, sub_dir: str, flag: str, start_with: str) -> str:
            # D-scope Option C: ONE rule per sub-analysis (mirrors the existing
            # delete_subanalysis_{sa} granularity). The runner deletes the sub's
            # processed/ across ALL its events (only when start_with=='process')
            # + the sub's analysis_datatree.zarr.
            proc_arg = " --delete-processed" if start_with == "process" else ""
            return (
                f"rule delete_subanalysis_reprocess_{rule_suffix}:\n"
                f'    output:\n        "{flag}"\n'
                f"    resources: cpus_per_task=1, mem_mb=4096, runtime=120\n"
                f"    shell:\n"
                f'        "{python_exe} -m hhemt.delete_subanalysis_reprocess_runner "\n'
                f'        "--sa-id {sa_id} --analysis-dir {sub_dir}{proc_arg}"\n\n'
            )

        is_sensitivity = getattr(self.analysis.cfg_analysis, "toggle_sensitivity_analysis", False)
        if not is_sensitivity:
            if start_with == "process":
                for i in range(len(self.analysis.df_sims)):
                    eid = compute_event_id_slug(self.analysis._retrieve_weather_indexer_using_integer_index(i))
                    slug = eid.replace(".", "_").replace("-", "_")
                    flag = f"{master_dir}/_status/_deleting_reprocess/processed_evt-{eid}.flag"
                    leaf_flags.append(flag)
                    rules.append(_proc_rule(slug, eid, master_dir, flag))
            cons_flag = f"{master_dir}/_status/_deleting_reprocess/reprocess_consolidation.flag"
            rules.append(_zarr_rule("consolidation", master_dir, cons_flag, leaf_flags))
        else:
            # D-scope → Option C: PER-SUB-ANALYSIS fan-out mirroring the EXISTING
            # _build_delete_sensitivity_snakefile_content granularity
            # (delete_subanalysis_{sa}, workflow.py:5316) — ONE rule per sub, NOT
            # per-(sa, event). Each per-sub rule deletes that sub's processed/
            # (across all its events, only when start_with=='process') + that
            # sub's analysis_datatree.zarr via delete_subanalysis_reprocess_runner
            # (item 5b). A master rule then deletes sensitivity_datatree.zarr,
            # fanning in on the per-sub flags. Rationale (vs the rejected
            # per-(sa,evt) shape): ~n_sa jobs instead of ~n_sa*n_evt tiny SLURM
            # jobs (avoids scheduler flood on large suites), and reuses the
            # validated per-sub rule granularity rather than a novel shape.
            sub_flags: list[str] = []
            for sa_id, sub in self.analysis.sensitivity.sub_analyses.items():
                sub_dir = str(sub.analysis_paths.analysis_dir)
                sa_slug = str(sa_id).replace(".", "_").replace("-", "_")
                sub_flag = f"{sub_dir}/_status/_deleting_reprocess/subanalysis_reprocess.flag"
                sub_flags.append(sub_flag)
                rules.append(_subanalysis_rule(sa_slug, sa_id, sub_dir, sub_flag, start_with))
            # master zarr rule (deletes sensitivity_datatree.zarr); fans in on all per-sub flags.
            master_flag = f"{master_dir}/_status/_deleting_reprocess/reprocess_consolidation.flag"
            rules.append(_zarr_rule("consolidation", master_dir, master_flag, sub_flags))
            cons_flag = master_flag

        rule_all = f'rule all:\n    input:\n        "{cons_flag}"\n\n'
        return rule_all + "".join(rules)

    def submit_reprocess_delete_workflow(
        self,
        *,
        start_with: str,
        override_in_flight: bool = False,
        override_multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks"] | None = None,
    ) -> dict:
        """Submit the SCOPED reprocess-delete workflow (R8). Routes the opt-in
        consolidated-zarr (+ processed/ when start_with=='process') deletion
        through the EXISTING delete-executor dispatch so the heavy GPFS deletion
        offloads to SLURM exactly as analysis.delete() does. Runs
        _pre_delete_guards first (parametrized to the scoped namespace). Isolated
        Snakefile.reprocess_delete + .snakemake_reprocess_delete/."""
        self._pre_delete_guards(
            override_in_flight=override_in_flight,
            snakefile_name="Snakefile.reprocess_delete",
            working_subdir=".snakemake_reprocess_delete",
        )
        stale = self.analysis_paths.analysis_dir / "_status" / "_deleting_reprocess"
        if stale.exists():
            # EXEMPT-DU: status-dir-cleanup
            fast_rmtree(stale)
        snakefile = self.analysis_paths.analysis_dir / "Snakefile.reprocess_delete"
        snakefile.write_text(self._build_reprocess_delete_snakefile_content(start_with=start_with))
        return self._submit_delete_snakemake(
            snakefile,
            override_multi_sim_run_method=override_multi_sim_run_method,
            working_subdir=".snakemake_reprocess_delete",
            logfile_name="snakemake_reprocess_delete.log",
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


class SensitivityAnalysisWorkflowBuilder(_ReportingSetDispatchMixin):
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
        # ADR-1: the sensitivity-master + reprocess-master process_{model} shells
        # reference self._container_process_prefix; delegate to the base builder's
        # resolved token so all three process-rung sites carry the identical prefix
        # (empty in native mode → generated Snakefiles byte-identical, R2/TO-1).
        self._container_process_prefix = self._base_builder._container_process_prefix

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
        from hhemt.scenario import compute_event_id_slug

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
from hhemt.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("hhemt")
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
        {self.python_executable} -m hhemt.export_scenario_status \\
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
            # ADR-2 OE-1: per-sa input stems derive from the single-source helper
            # so a future stem-grammar change cannot desync them from the outputs.
            _pfd_sa = _plot_output_template(
                renderer_kind="peak_flood_depth",
                subdir="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}",
                sa_id="{sa_id}",
                event_id="{event_id}",
            ).replace("__OUTPUT_EXT__", _e_pfd)
            _cf_sa = _plot_output_template(
                renderer_kind="conduit_flow",
                subdir="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}",
                sa_id="{sa_id}",
                event_id="{event_id}",
            ).replace("__OUTPUT_EXT__", _e_cf)
            rule_all_inputs.append(
                f'expand("{_pfd_sa}", zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)'
            )
            rule_all_inputs.append(
                f'expand("{_cf_sa}", zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)'
            )
        if _independent_vars:
            _e_bench = _ext["sensitivity_benchmarking"]
            # ADR-2 OE-1 + D3: benchmarking input stem derives from the helper
            # (renderer_kind=benchmarking, descriptor={independent_var}.vs.total).
            _bench_path = _plot_output_template(
                renderer_kind="benchmarking",
                subdir="plots/sensitivity/benchmarking",
                descriptor="{independent_var}.vs.total",
            ).replace("__OUTPUT_EXT__", _e_bench)
            rule_all_inputs.append(f'expand("{_bench_path}", independent_var={_independent_vars!r})')
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
                target_partition=target.target_partition,
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
        {self.python_executable} -m hhemt.setup_workflow \\
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
            gpus_per_node_config = (
                resolve_gpus_per_node(sub_analysis.cfg_hpc_system, sub_analysis.cfg_analysis.hpc_ensemble_partition)
                or 0
            )
            cpus_per_sim = n_mpi * n_omp
            run_mode = sub_analysis.cfg_analysis.run_mode

            sub_config_args = self._base_builder._get_config_args(
                analysis_config_yaml=sub_analysis.analysis_config_yaml,
                system_config_yaml=sub_analysis._system.system_config_yaml,
            )
            # Phase 6 (DQ7b): the run_simulation_runner resolves + injects the GPU
            # hardware from --target-partition. Pass the per-sub ensemble partition
            # so the sim runs (and any per-sim GPU compile) target the right hardware.
            sub_gpu_compile_config_args = self._base_builder._get_config_args(
                analysis_config_yaml=sub_analysis.analysis_config_yaml,
                system_config_yaml=sub_analysis._system.system_config_yaml,
                target_partition=sub_analysis.cfg_analysis.hpc_ensemble_partition,
            )

            # Phase 3: per-SA system config sources gpu_alloc_mode + gpu_hw so a
            # sensitivity study spanning UVA (gres) and Frontier (gpus) emits the
            # correct SLURM directive per sub-analysis.
            gpu_alloc_mode = self._base_builder._resolve_gpu_alloc_mode()

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
            gpu_hw = resolve_gpu_target(sub_analysis.cfg_hpc_system, sub_analysis.cfg_analysis.hpc_ensemble_partition)[
                0
            ]
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
                runtime_min=240,
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
        {self.python_executable} -m hhemt.prepare_scenario_runner \\
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
                        wait_walltime_cap_min=sub_analysis.cfg_analysis.hpc_max_wait_for_inflight_min,
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
    retries: {self._base_builder._resolved_simulate_retries()}
    log: "{log_dir_str}/sims/{sim_rule_name}.log"
    conda: "{conda_env_path}"
    threads: {snakemake_threads}
    resources:
{sim_resources_sa}
    shell:
        """
        mkdir -p {log_dir_str}/sims {log_dir_str} _status
        {self.python_executable} -m hhemt.run_simulation_runner \\
            --event-iloc {event_iloc} \\
            {sub_gpu_compile_config_args} \\
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
        {self._container_process_prefix}{self.python_executable} -m hhemt.process_timeseries_runner \\
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
        {self.python_executable} -m hhemt.consolidate_workflow \\
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
        {self.python_executable} -m hhemt.consolidate_workflow \\
            {master_config_args} \\
            --consolidate-sensitivity-analysis-outputs \\
            --which {which} \\
            --compression-level {compression_level} \\
            --flag-output {{output}} \\
            --rule-name master_consolidation \\
            > {{log}} 2>&1
        """
'''

        # Registry-driven plot-rule dispatch at master scope (P1b / TO-8). The
        # benchmarking set drives the five common renderers + the two conditional
        # sensitivity renderers (per_sim_per_sa, sensitivity_benchmarking), gated
        # by predicate_key against the method-local sa_event_pairs_sa /
        # _independent_vars threaded in via predicate_inputs. The export rule is a
        # set-invariant non-figure rule (Option B — NOT a renderer_selection
        # entry); the B-i interleave hook flushes it BETWEEN the unconditional and
        # conditional renderers, byte-matching the pre-refactor emission order.
        # Master uses f_consolidate_master_complete.flag (NOT the multisim
        # e_consolidate_complete flag).
        snakefile_content += self._emit_active_set_plot_rules(
            self._resolve_active_reporting_set(self.master_analysis),
            input_flag="_status/f_consolidate_master_complete.flag",
            predicate_inputs={
                "independent_vars": _independent_vars,
                "sa_event_pairs_sa": sa_event_pairs_sa,
            },
            interleave_after_unconditional=lambda: self._base_builder._build_export_scenario_status_rule(
                input_flag="_status/f_consolidate_master_complete.flag",
            ),
        )

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
        {self.python_executable} -m hhemt.render_report_runner \\
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
        from hhemt.scenario import compute_event_id_slug

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
        # Report-target invariant: only include a sub's (sa_id, event_id) pairs
        # when the sub passes the shared summary-existence predicate (matches
        # consolidate_sensitivity_datatree's whole-sub skip, Gotcha 36). The
        # c_run_* flag is a STRICTLY WEAKER signal (Gotcha 34) — a sim can have
        # run with its summary absent — so enumerating a per-sim plot/report
        # target on c_run produces an unsatisfiable target the renderer fails on.
        from hhemt.constants import sim_run_flag_per_sa

        sa_event_pairs_sa: list[str] = []
        sa_event_pairs_evt: list[str] = []
        analysis_dir_for_pairs = self.master_analysis.analysis_paths.analysis_dir

        def _sub_included_for_reprocess(sa_id, sub) -> bool:
            """Shared sub-inclusion predicate for SA_EVENT_PAIRS, completed_sa_ids,
            and the per-sa consolidate-emission loop, so the ~6785 equality
            assertion holds. On the consolidate/render paths a sub is included
            iff ALL its summaries exist (matches consolidate_sensitivity_datatree's
            whole-sub skip, Gotcha 36 -> no per-sa consolidate FileNotFoundError:
            the per-sa consolidate is a pure summary CONSUMER via
            consolidate_to_datatree/_retrieve_combined_output, which has no
            per-sub allow_incomplete). On the process path a sub with >=1
            completed sim (c_run) is ALSO included so its self-healed divergent
            events get a process rebuild rule: the FIX-2 self-heal
            (_reconcile_stale_process_flags_against_summaries) unlinked the stale
            d_process flag at reprocess entry (c_run present, summary absent); the
            per-EVENT c_run filter in the per-sa loop admits those events; the
            process-emit gate re-emits their rebuild rule; the per-sa consolidate
            then runs with --allow-incomplete via the existing
            had_conditional_process_emit mixed-state path. The start_with
            disjunct is INERT on the default consolidate/render paths (which have
            no self-heal — it is guarded on start_with == 'process')."""
            if _sub_analysis_summaries_complete(sub, enabled_models):
                return True
            if start_with == "process":
                for _evt_iloc in sub.df_sims.index:
                    _evt = compute_event_id_slug(sub._retrieve_weather_indexer_using_integer_index(_evt_iloc))
                    if (analysis_dir_for_pairs / sim_run_flag_per_sa(model_type, str(sa_id), _evt)).exists():
                        return True
            return False

        try:
            for sa_id_pair, sub_pair in self.sensitivity_analysis.sub_analyses.items():
                if not _sub_included_for_reprocess(sa_id_pair, sub_pair):
                    continue
                for event_iloc in sub_pair.df_sims.index:
                    ev = sub_pair._retrieve_weather_indexer_using_integer_index(event_iloc)
                    event_id_pair = compute_event_id_slug(ev)
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
from hhemt.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst

try:
    from importlib.metadata import version as _pkg_version
    _toolkit_version = _pkg_version("hhemt")
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
        {self.python_executable} -m hhemt.export_scenario_status \\
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
        from hhemt.constants import (
            consolidate_subanalysis_flag,
        )

        completed_sa_ids: list[str] = []
        for sa_id_check, sub_check in self.sensitivity_analysis.sub_analyses.items():
            # Shared start_with-aware sub-inclusion predicate (lockstep with
            # SA_EVENT_PAIRS and the per-sa loop) — keeps the ~6785 equality
            # assertion true. Summary-existence on consolidate/render;
            # + rebuildable-on-process. Replaces the prior per-event c_run scan.
            if _sub_included_for_reprocess(sa_id_check, sub_check):
                completed_sa_ids.append(str(sa_id_check))
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
            # ADR-2 OE-1: per-sa input stems derive from the single-source helper
            # so a future stem-grammar change cannot desync them from the outputs.
            _pfd_sa = _plot_output_template(
                renderer_kind="peak_flood_depth",
                subdir="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}",
                sa_id="{sa_id}",
                event_id="{event_id}",
            ).replace("__OUTPUT_EXT__", _e_pfd)
            _cf_sa = _plot_output_template(
                renderer_kind="conduit_flow",
                subdir="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}",
                sa_id="{sa_id}",
                event_id="{event_id}",
            ).replace("__OUTPUT_EXT__", _e_cf)
            rule_all_inputs.append(
                f'expand("{_pfd_sa}", zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)'
            )
            rule_all_inputs.append(
                f'expand("{_cf_sa}", zip=True, sa_id=SA_EVENT_PAIRS_SA, event_id=SA_EVENT_PAIRS_EVT)'
            )
        if _independent_vars:
            _e_bench = _ext["sensitivity_benchmarking"]
            # ADR-2 OE-1 + D3: benchmarking input stem derives from the helper
            # (renderer_kind=benchmarking, descriptor={independent_var}.vs.total).
            _bench_path = _plot_output_template(
                renderer_kind="benchmarking",
                subdir="plots/sensitivity/benchmarking",
                descriptor="{independent_var}.vs.total",
            ).replace("__OUTPUT_EXT__", _e_bench)
            rule_all_inputs.append(f'expand("{_bench_path}", independent_var={_independent_vars!r})')
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
        # Flag-name builders live in hhemt.constants (single
        # source of truth for new code; existing hardcoded sites are
        # tracked as a follow-up refactor).
        from hhemt.constants import (
            consolidate_subanalysis_flag,
            process_timeseries_flag_per_sa,
            sa_inputs_fingerprint_flag,
        )

        # sim_run_flag_per_sa is already imported at method scope (in the
        # SA_EVENT_PAIRS block above, for the shared _sub_included_for_reprocess
        # closure); the per-event c_run filter below reuses that binding.
        analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        subanalysis_flags: list[str] = []

        for sa_id, sub_analysis in self.sensitivity_analysis.sub_analyses.items():
            # Lockstep sub-inclusion gate (same shared closure as SA_EVENT_PAIRS
            # and completed_sa_ids) — keeps the ~6785 equality assertion true. A
            # summary-absent sub on the consolidate/render path gets NO per-sa
            # consolidate rule (its consolidate_to_datatree would raise
            # FileNotFoundError — there is no per-sub allow_incomplete). The
            # per-EVENT c_run filter below stays on c_run for process rebuild and
            # is intentionally NOT gated here.
            if not _sub_included_for_reprocess(sa_id, sub_analysis):
                continue
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
                runtime_min=240,
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
                # (R10) Architecture note — the emit gate below and the
                # consolidate-input routing at the d_process/c_run append sites
                # are CORRECT. The reprocess rebuild rule's correctness under
                # --rerun-triggers comes from (i) the deleted/absent d_process
                # output (missing-output reruns are trigger-independent) plus
                # (ii) the downstream consolidate_sa_* consuming the d_process
                # flag as input: — NOT from any declared dependency on the
                # deleted summary zarr. The d5d0084 regression was an UPSTREAM
                # stale-flag survival on the sensitivity regenerate_existing
                # reprocess arm (fixed in sensitivity_analysis.py), NOT a defect
                # in this gate. Do not add a summary-zarr input: here — it would
                # break the rerun semantics.
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
        {self._container_process_prefix}{self.python_executable} -m hhemt.process_timeseries_runner \\
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
        {self.python_executable} -m hhemt.consolidate_workflow \\
            {sub_config_args} \\
            --which {which} \\
{allow_incomplete_line}            --compression-level {compression_level} \\
            --flag-output {{output}} \\
            --rule-name consolidate_{prefix}{sa_id_rule} \\
            --sa-id {sa_id} \\
            > {{log}} 2>&1
        """

'''

        # Sanity assertion: the per-sa loop's subanalysis_flags must match the
        # up-front completed_sa_ids list — both now gate sub-inclusion on the
        # SAME shared _sub_included_for_reprocess predicate (whole-sub summary-
        # existence on consolidate/render; + rebuildable-on-process). Equality
        # (NOT subset) is the correct invariant: a mismatch in EITHER direction
        # is the workflow.py line-142 integration risk. Over-count -> the loop
        # emits a consolidate_{sa} flag rule all does not depend on (orphan
        # rule). Under-count -> rule all demands a consolidate_subanalysis flag
        # the loop never emitted -> Snakemake MissingInputException at DAG build.
        # Both lists iterate sub_analyses.items() in the same order, so the
        # order-sensitive list == comparison is exact.
        from hhemt.constants import consolidate_subanalysis_flag as _cons_flag

        _expected_subanalysis_flags = [_cons_flag(sa_id) for sa_id in completed_sa_ids]
        if subanalysis_flags != _expected_subanalysis_flags:
            raise RuntimeError(
                "generate_reprocess_master_snakefile_content: per-sa loop's "
                f"subanalysis_flags={subanalysis_flags!r} does not match the "
                f"up-front completed_sa_ids derivation={_expected_subanalysis_flags!r}; "
                "shared sub-inclusion invariant violated — the per-sa consolidate-"
                "emission gate and the completed_sa_ids gate must both call "
                "_sub_included_for_reprocess (workflow.py line-142 integration risk)."
            )

        # Master consolidation — aggregates the EMITTED per-sa flags into the
        # master flag + sensitivity_datatree.zarr; overwrite + allow-incomplete
        # baked. Uses the central flag-name builder for the master flag (Spec 1).
        from hhemt.constants import consolidate_master_flag

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
        {self.python_executable} -m hhemt.consolidate_workflow \\
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

        # Plot + export + render rules — registry-driven dispatch (P1b / TO-8),
        # identical to the production master (same benchmarking set + B-i
        # interleave hook), so the rendered report set is byte-equivalent. The
        # dispatcher running the SAME set on both surfaces makes the
        # historically hand-maintained "byte-equivalent" guarantee structural.
        snakefile_content += self._emit_active_set_plot_rules(
            self._resolve_active_reporting_set(self.master_analysis),
            input_flag="_status/f_consolidate_master_complete.flag",
            predicate_inputs={
                "independent_vars": _independent_vars,
                "sa_event_pairs_sa": sa_event_pairs_sa,
            },
            interleave_after_unconditional=lambda: self._base_builder._build_export_scenario_status_rule(
                input_flag="_status/f_consolidate_master_complete.flag",
            ),
        )

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
        {self.python_executable} -m hhemt.render_report_runner \\
            {master_config_args} \\
            --format {{wildcards.format}} \\
            --reprocess \\
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
    from hhemt.report_renderers._figure_emission import (
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
            output_path_template=_plot_output_template(
                renderer_kind="benchmarking",
                subdir="plots/sensitivity/benchmarking",
                descriptor="{independent_var}.vs.total",
            ),
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
        from hhemt.scenario import compute_event_id_slug

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
    from hhemt.report_renderers._figure_emission import (
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
    from hhemt.report_renderers._figure_emission import (
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
            output_path_template=_plot_output_template(
                renderer_kind="peak_flood_depth",
                subdir="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}",
                sa_id="{sa_id}",
                event_id="{event_id}",
            ),
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
            output_path_template=_plot_output_template(
                renderer_kind="conduit_flow",
                subdir="plots/sensitivity/per_sim/sa-{sa_id}/{event_id}",
                sa_id="{sa_id}",
                event_id="{event_id}",
            ),
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

    def _write_queued_sentinels_sensitivity(self, alloc_jobid: str | None) -> None:
        """mechanism (b) PENDING-recovery writer for the sensitivity path: write
        _status/_queued/{simulation_sa_{sa_id}_evt-{event_id}}.json under EACH
        sub-analysis's OWN analysis_dir (sensitivity markers/sentinels live per-sub,
        Spec D — the master-dir reconcile would miss them), for the full planned
        per-sub sim-token set, AFTER submit-success. The sensitivity sentinel token is
        per-(sa_id, event_id) with NO model_type segment (matching
        run_simulation_runner.py's _rule_token when args.sa_id is set, and the
        generate_master_snakefile_content emission gate). Toolkit-owns-sbatch (1_job)
        records the master allocation jobid; executor-owns-sbatch (batch_job) null.
        Delegates the atomic compare-and-write to the base builder's writer so the
        mtime-preserving contract (R12) is single-sourced."""
        from hhemt.scenario import compute_event_id_slug

        for sa_id, sub in self.sensitivity_analysis.sub_analyses.items():  # type: ignore[attr-defined]
            event_ids = [
                compute_event_id_slug(sub._retrieve_weather_indexer_using_integer_index(i))
                for i in range(len(sub.df_sims))
            ]
            tokens = [f"simulation_sa_{sa_id}_evt-{event_id}" for event_id in event_ids]
            self._base_builder._write_queued_sentinels(tokens, alloc_jobid, sub.analysis_paths.analysis_dir)

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
        override_hpc_restart_times_simulate: int | None = None,
        override_hpc_restart_times_other: int | None = None,
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

        # Retry overrides (P1 Decision 4, FQ3 SITE 5): the sensitivity path emits
        # rules via _base_builder (generate_snakemake_config global baseline +
        # _resolved_simulate_retries on the simulation_sa rules), so stash the
        # override knobs there, mirroring the diagnostics stash above.
        self._base_builder._override_hpc_restart_times_simulate = override_hpc_restart_times_simulate
        self._base_builder._override_hpc_restart_times_other = override_hpc_restart_times_other

        # Check if we should use 1-job mode based on config
        multi_sim_method = self.master_analysis.cfg_analysis.multi_sim_run_method

        sim_resources = self.master_analysis._resource_manager._get_simulation_resource_requirements()
        n_gpus_per_sim = sim_resources["n_gpus"]
        if n_gpus_per_sim > 0 and not self.system.gpu_compilation_backend:
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

            # mechanism (b): record the planned per-sub sim-token set under each sub's
            # _status/_queued/ AFTER submit-success. 1_job_many_srun_tasks is
            # toolkit-owns-sbatch — the master allocation jobid enables the wait-runner
            # in-loop probe (R8) for PENDING-recovered sensitivity wait-rules.
            if isinstance(result, dict) and result.get("success", True):
                self._write_queued_sentinels_sensitivity(result.get("job_id"))

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

            # mechanism (b): record the planned per-sub sim-token set under each sub's
            # _status/_queued/ AFTER submit-success. batch_job is executor-owns-sbatch —
            # jobid null, held on PRESENCE bounded by the mtime fail-safe (F1-O3, R12).
            if isinstance(result, dict) and result.get("success", True):
                self._write_queued_sentinels_sensitivity(None)

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

        The reprocess driver shares ``analysis_dir/.snakemake/`` with the run
        path and uses ``--nolock`` so it coexists with queued/running SLURM
        sim workers without touching the shared lock; the master
        ``_status/_orchestrator/`` liveness gate (not the Snakemake lock) is
        the concurrency authority and refuses fast only when a live
        orchestration driver for this master analysis exists.

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
            "--nolock",
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

        # Reprocess concurrency gate (R3): refuse fast with a WorkflowError —
        # never input() — when a live orchestration DRIVER for this master
        # analysis exists. Default-safe when no _orchestrator/ sentinel is
        # present (R6), and coexists with queued/running _submitted/ sim
        # WORKERS (R2). Skipped for dry runs (planning only — nothing is
        # submitted, no zarr is written).
        import os

        from hhemt import orchestrator_sentinels as _osent

        driver_id = _osent.new_driver_id()
        remove_self_sentinel = False
        if not dry_run:
            gate_err = self._base_builder._orchestrator_liveness_gate(
                analysis_dir=analysis_dir,
                exclude_driver_id=driver_id,
            )
            if gate_err is not None:
                raise gate_err
            # Reprocess self-sentinel (R5): write-own-then-scan-others mutual
            # exclusion vs a second reprocess. Always mode="local"/pid=os.getpid()
            # — reprocess overrides 1_job_many_srun_tasks -> batch_job but never
            # allocates its own tmux/sbatch driver, so it never produces a
            # tmux/sbatch self-sentinel. A blocking-local reprocess removes it in
            # the finally; a detached Popen reprocess leaves it (reclaimed by the
            # gate's ps -p {pid} arm once this login-node process exits).
            _osent.write_orchestrator_sentinel(
                analysis_dir,
                driver_id=driver_id,
                workflow_submission_mode="local",
                pid=os.getpid(),
            )
            remove_self_sentinel = True

        try:
            # Facade — reconciliation against analysis_dir/_status/_submitted/.
            # skip_lock_check=True bypasses the toolkit-side input() prompt; the
            # orchestrator-liveness gate above is the concurrency authority and
            # --nolock is on the subprocess. Phase 1's at-most-once guard still
            # protects reprocess from a parallel live sim driver double-submitting.
            self._base_builder._pre_snakemake_invocation_guards(
                snakefile_path,
                dry_run=dry_run,
                verbose=verbose,
                working_dir=analysis_dir,
                skip_lock_check=True,
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
            # Detached driver: leave the self-sentinel for the gate's liveness
            # reclaim — do NOT remove it in the finally.
            remove_self_sentinel = False
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
        finally:
            if remove_self_sentinel:
                _osent.remove_orchestrator_sentinel(analysis_dir, driver_id)
