"""Static-plot Snakefile generator (publication figures).

Emits one Snakemake rule per static-plot ID + a `rule all`. No render_report
rule, no report() wrappers — static plots are standalone publication files
(ADR-8 / OD-6 Dist-A). Modeled on bundle/snakefile_generator.py. The output
extension is pre-resolved from each per-plot config's `output_format` (NOT from
static_backend via _OUTPUT_EXT_BY_RENDERER, which has no static renderer kind),
so the RuleSpec.output_path_template is already a literal path.

The per-sim event selector is resolved at GENERATION time: the canonical ADR-2
plot ID carries the event-id slug (and optional sa-id) as `__evt.{slug}` /
`__sa.{id}` segments. The generator maps the slug to the positional integer
event_iloc the renderer CLI expects (the same `range(n_sims)` convention as
workflow.py's ILOC_BY_EVENT_ID) and threads `--event-iloc` / `--sa-id` onto the
rule's shell, so the renderer's static branch receives the selector exactly as
the report path does.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

from hhemt.config.static_plots import STATIC_PLOT_CONFIG_REGISTRY, StaticPlotBaseConfig
from hhemt.workflow import (
    RuleEmissionContext,
    RuleSpec,
    _emit_plot_rule,
    _emit_rule_all,
)

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis


def _load_static_config(path: Path) -> StaticPlotBaseConfig:
    """Deserialize a per-plot YAML into the registry-selected subclass."""
    raw = yaml.safe_load(Path(path).read_text())
    renderer_kind = raw.get("renderer_kind")
    model_cls = STATIC_PLOT_CONFIG_REGISTRY.get(renderer_kind)
    if model_cls is None:
        from hhemt.exceptions import ConfigurationError

        raise ConfigurationError(
            field="renderer_kind",
            message=(
                f"static-plot config {path} has renderer_kind={renderer_kind!r}, "
                f"not in STATIC_PLOT_CONFIG_REGISTRY (valid: {sorted(STATIC_PLOT_CONFIG_REGISTRY)})."
            ),
            config_path=Path(path),
        )
    return model_cls.model_validate(raw)


def _parse_plot_id_selectors(plot_id: str) -> tuple[str | None, str | None]:
    """Extract (event_id_slug, sa_id) from an ADR-2 canonical plot ID.

    Segments are joined by ``__``; the per-sim selector is ``evt.{slug}`` (the
    slug itself may contain ``.``) and the sensitivity selector is ``sa.{id}``.
    Returns (None, None) for a plot ID carrying neither (e.g. a per-analysis
    figure with no per-sim scope).
    """
    event_id: str | None = None
    sa_id: str | None = None
    for seg in plot_id.split("__"):
        if seg.startswith("evt."):
            event_id = seg[len("evt.") :]
        elif seg.startswith("sa."):
            sa_id = seg[len("sa.") :]
    return event_id, sa_id


def _resolve_event_iloc(scope_analysis: TRITONSWMM_analysis, event_id: str) -> int | None:
    """Map an event-id slug to the positional integer event_iloc.

    Mirrors workflow.py's ``iloc_by_event_id = {event_ids[i]: i for i in
    range(n_sims)}`` construction (positional iloc, NOT df_sims label index) so
    the resolved iloc matches what the report path's ``--event-iloc`` passes.
    Returns None when the slug is not found (caller raises ConfigurationError).
    """
    from hhemt.scenario import compute_event_id_slug

    n_sims = len(scope_analysis.df_sims)
    for i in range(n_sims):
        ev = scope_analysis._retrieve_weather_indexer_using_integer_index(i)
        if compute_event_id_slug(ev) == event_id:
            return i
    return None


def _scope_analysis_for_sa(analysis: TRITONSWMM_analysis, sa_id: str) -> TRITONSWMM_analysis:
    """Return the sub-analysis for an ``sa.{id}`` selector (sensitivity master).

    Mirrors _cli.py's --sa-id resolution so the generation-time event_iloc is
    computed against the SAME scope the renderer will operate on.
    """
    from hhemt.exceptions import ConfigurationError

    sub_analyses = analysis.sensitivity.sub_analyses
    if sa_id not in sub_analyses:
        raise ConfigurationError(
            field="static_config_id",
            message=(
                f"plot_id carries sa.{sa_id} but it is not in the master's sub_analyses "
                f"(available: {sorted(sub_analyses)})."
            ),
        )
    return sub_analyses[sa_id]


def _harvest_static_rule_specs(
    analysis: TRITONSWMM_analysis,
    ctx: RuleEmissionContext,
    *,
    static_plot_configs: list[Path],
    static_config_ids: list[str] | None = None,
) -> tuple[RuleSpec, ...]:
    """One RuleSpec per static-plot config in the PASSED static_plot_configs.

    report_kwargs=None (bare output); output_path_template is the LITERAL
    static_plots/{plot_id}.{ext} with ext pre-resolved from output_format.
    Iterates the passed list (override-resolved at the facade), not
    analysis.cfg_analysis.static_plot_configs, so an override is honored.
    When static_config_ids is non-None, only configs whose plot_id is in that
    set are emitted.
    """
    id_filter = set(static_config_ids) if static_config_ids is not None else None
    specs: list[RuleSpec] = []
    for cfg_path in static_plot_configs:
        scfg = _load_static_config(Path(cfg_path))
        if id_filter is not None and scfg.plot_id not in id_filter:
            continue
        ext = scfg.output_format  # pdf/svg/ps/eps/pgf/png — matplotlib-native

        # Per-sim / sensitivity selector resolution from the canonical plot ID.
        # Emit BOTH the id (rule identity) and the absolute config PATH so the
        # render subprocess is self-contained: it loads the config from the path
        # rather than re-searching cfg_analysis.static_plot_configs, which does NOT
        # carry a facade-supplied override_static_plot_configs list (that override
        # is resolved here at generation time but never persisted to the analysis
        # config the rule subprocess re-reads). Path is quoted for space-safety.
        event_id, sa_id = _parse_plot_id_selectors(scfg.plot_id)
        abs_cfg_path = Path(cfg_path).resolve()
        extra_flags: list[str] = [
            f"--static-config-id {scfg.plot_id}",
            f'--static-config-path "{abs_cfg_path}"',
        ]
        if sa_id is not None:
            extra_flags.append(f"--sa-id {sa_id}")
        if event_id is not None:
            scope = _scope_analysis_for_sa(analysis, sa_id) if sa_id is not None else analysis
            iloc = _resolve_event_iloc(scope, event_id)
            if iloc is None:
                from hhemt.exceptions import ConfigurationError

                raise ConfigurationError(
                    field="static_config_id",
                    message=(
                        f"plot_id {scfg.plot_id!r} carries evt.{event_id} but no simulation "
                        f"in the analysis resolves to that event-id slug."
                    ),
                    config_path=Path(cfg_path),
                )
            extra_flags.append(f"--event-iloc {iloc}")

        # Snakemake rule names must be valid Python identifiers — the canonical
        # plot_id's within-segment "." separator is illegal in a rule name, so
        # sanitize it to "_" for the rule name only (the output/log paths keep
        # the real plot_id, where "." is a valid filename char).
        rule_name_safe = "static_plot_" + scfg.plot_id.replace(".", "_")
        specs.append(
            RuleSpec(
                rule_name=rule_name_safe,
                # invariant: the renderer_kind string IS the renderer module name
                # under report_renderers/ — it is the `_cli.py` positional
                # `renderer` arg (workflow.py interpolates it as
                # `python -m hhemt.report_renderers._cli {renderer_module}`).
                renderer_module=scfg.renderer_kind,
                input_flags=(),
                output_path_template=f"static_plots/{scfg.plot_id}.{ext}",
                source_paths=(),
                wildcards=(),
                extra_cli_flags=tuple(extra_flags),
                extra_params=(),
                report_kwargs=None,  # bare output — no report() wrapper
                resources_yaml=_static_resources_yaml(),
                log_path_template=f"logs/static_plots/{scfg.plot_id}.log",
            )
        )
    return tuple(specs)


def _static_resources_yaml() -> str:
    """Small fixed resource block for a per-plot static render (no MPI/GPU).

    Flat `name=value, name=value` form — Snakemake's `resources:` directive
    parses a keyword-argument list, NOT a dict/set literal; a leading `{` is a
    parse error. `runtime` (minutes) is the canonical resource the SLURM
    executor maps to `--time`. Matches every existing `_emit_plot_rule` caller.
    """
    return "mem_mb=4000, runtime=30, cpus_per_task=1"


def _static_preamble(ctx: RuleEmissionContext) -> str:
    """Snakefile preamble: imports format_sources_rst (the params helper
    _emit_plot_rule references) + the configfile-free header. Mirrors
    bundle/snakefile_generator._build_preamble; static rules need no config:
    block because every parameter is passed via the --static-config-id CLI flag."""
    return "from hhemt.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst\n"


def generate_static_snakefile(
    analysis: TRITONSWMM_analysis,
    *,
    static_plot_configs: list[Path],
    config_args_str: str,
    static_backend: Literal["matplotlib", "plotly"] = "matplotlib",
    static_config_ids: list[str] | None = None,
) -> str:
    # config_args_str is supplied by submit_static_plots_workflow via
    # self._get_config_args(); RuleEmissionContext requires it (no default).
    # static_plot_configs is the override-resolved list passed down from the
    # facade; _harvest iterates THIS list (NOT analysis.cfg_analysis.
    # static_plot_configs), so a passed override is honored.
    ctx = RuleEmissionContext(
        python_executable="python",
        log_dir_rel="logs/static_plots",
        conda_env_path="",
        config_args_str=config_args_str,
        is_sensitivity=analysis.cfg_analysis.toggle_sensitivity_analysis,
        static_backend=static_backend,
    )
    rule_specs = _harvest_static_rule_specs(
        analysis,
        ctx,
        static_plot_configs=static_plot_configs,
        static_config_ids=static_config_ids,
    )
    plot_output_paths = tuple(f'"{spec.output_path_template}"' for spec in rule_specs)
    rule_all_block = _emit_rule_all(
        status_flags=(),
        plot_output_paths=plot_output_paths,
        render_report_targets=(),  # NO render_report
        ctx=ctx,
    )
    plot_rule_blocks = "".join(_emit_plot_rule(spec, ctx) for spec in rule_specs)
    return "\n".join([_static_preamble(ctx), rule_all_block, plot_rule_blocks])


def write_static_snakefile(
    analysis: TRITONSWMM_analysis,
    *,
    static_plot_configs: list[Path],
    config_args_str: str,
    static_backend: Literal["matplotlib", "plotly"] = "matplotlib",
    static_config_ids: list[str] | None = None,
) -> Path:
    """Overwrite {analysis_dir}/Snakefile.static unconditionally (never
    auto-detected by existence — mirrors the Snakefile.reprocess discipline)."""
    text = generate_static_snakefile(
        analysis,
        static_plot_configs=static_plot_configs,
        config_args_str=config_args_str,
        static_backend=static_backend,
        static_config_ids=static_config_ids,
    )
    out = analysis.analysis_paths.analysis_dir / "Snakefile.static"
    out.write_text(text)
    return out
