"""Bundle-side regeneration-scoped Snakefile generator.

Produces a regeneration-scoped Snakefile inside a bundle, containing
only the rules needed to re-render the analysis report locally — plot
rules + render_report + rule all. Simulation, processing, consolidation,
and sensitivity-orchestration rules are NOT emitted.

Public surface:

    generate_regeneration_snakefile(bundle_root, *, static_backend) -> str
        Return the Snakefile body as a string.

    write_regeneration_snakefile(bundle_root, *, static_backend) -> Path
        Write the Snakefile to ``{bundle_root}/Snakefile`` and return the path.
        The bundle's verbatim copy of the source-side Snakefile lives at
        ``{bundle_root}/Snakefile.source`` (preserved for debugging value
        per Plan Phase 2 D1).

Per Plan Phase 2 D3: ``static_backend`` is a required keyword-only param —
no in-code default. The cfg-level default is ``"plotly"`` per Decision 4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml

from TRITON_SWMM_toolkit.workflow import (
    RuleEmissionContext,
    RuleSpec,
    _emit_plot_rule,
    _emit_render_report_rule,
    _emit_rule_all,
    _output_ext_for,
)


_RENDER_REPORT_RESOURCES_YAML = (
    "        mem_mb=2000,\n"
    "        time_min=30,\n"
    "        cpus_per_task=1,"
)


def generate_regeneration_snakefile(
    bundle_root: Path,
    *,
    static_backend: Literal["matplotlib", "plotly"],
) -> str:
    """Generate the regeneration-scoped Snakefile body as a string."""
    bundle_root = bundle_root.resolve()
    cfg_analysis = yaml.safe_load((bundle_root / "cfg_analysis.yaml").read_text())
    is_sensitivity = bool(cfg_analysis.get("toggle_sensitivity_analysis", False))

    ctx = _make_rule_emission_context(
        bundle_root=bundle_root,
        is_sensitivity=is_sensitivity,
        static_backend=static_backend,
    )
    rule_specs = _harvest_rule_specs(
        bundle_root=bundle_root,
        is_sensitivity=is_sensitivity,
        cfg_analysis=cfg_analysis,
    )
    preamble = _build_preamble()
    config_block = _build_config_block(
        cfg_analysis=cfg_analysis, is_sensitivity=is_sensitivity
    )
    report_directive = 'report: "report/workflow_description.rst"\n'

    plot_output_paths = tuple(
        f'"{p}"'
        for spec in rule_specs
        for p in _expand_wildcards_to_existing_files(
            _resolve_output_path(spec, ctx), bundle_root
        )
    )
    render_report_targets = ('"analysis_report.html"', '"analysis_report.zip"')
    status_flags = _status_flags_for(is_sensitivity, cfg_analysis)

    rule_all_block = _emit_rule_all(
        status_flags=status_flags,
        plot_output_paths=plot_output_paths,
        render_report_targets=render_report_targets,
        ctx=ctx,
    )
    plot_rule_blocks = "".join(_emit_plot_rule(spec, ctx) for spec in rule_specs)
    render_report_block = _emit_render_report_rule(
        plot_output_paths=plot_output_paths,
        ctx=ctx,
        resources_yaml=_RENDER_REPORT_RESOURCES_YAML,
    )

    return "\n".join([
        preamble,
        config_block,
        report_directive,
        rule_all_block,
        plot_rule_blocks,
        render_report_block,
    ])


def write_regeneration_snakefile(
    bundle_root: Path,
    *,
    static_backend: Literal["matplotlib", "plotly"],
) -> Path:
    """Generate the Snakefile and write it to ``{bundle_root}/Snakefile``.

    Overwrites any existing ``Snakefile``. The verbatim source-side copy
    lives at ``Snakefile.source`` per Plan Phase 2 D1 and is untouched.
    """
    bundle_root = bundle_root.resolve()
    text = generate_regeneration_snakefile(bundle_root, static_backend=static_backend)
    out = bundle_root / "Snakefile"
    out.write_text(text + "\n")
    return out


def _make_rule_emission_context(
    *,
    bundle_root: Path,
    is_sensitivity: bool,
    static_backend: Literal["matplotlib", "plotly"],
) -> RuleEmissionContext:
    """Build a bundle-root-relative RuleEmissionContext.

    Bundle-relative config args; no conda env (consume-side does not
    invoke --use-conda).
    """
    return RuleEmissionContext(
        python_executable="python",
        log_dir_rel="_logs",
        conda_env_path="",
        config_args_str=(
            "--system-config cfg_system.yaml \\\n"
            "            --analysis-config cfg_analysis.yaml"
        ),
        is_sensitivity=is_sensitivity,
        static_backend=static_backend,
    )


def _harvest_rule_specs(
    *,
    bundle_root: Path,
    is_sensitivity: bool,
    cfg_analysis: dict,
) -> tuple[RuleSpec, ...]:
    """Construct the RuleSpec list for the regeneration-scoped Snakefile.

    Multi-sim rules: system_overview, per_sim_peak_flood_depth (wildcarded
    on event_id), per_sim_conduit_flow (wildcarded on event_id),
    per_analysis_summary_table, scenario_status_appendix,
    errors_and_warnings.

    Sensitivity-master rules add sensitivity_benchmarking (wildcarded on
    independent_var) and per_sim_per_sa_* (wildcarded on sa_id+event_id).
    """
    specs: list[RuleSpec] = []
    plots_dir = bundle_root / "plots"

    # system_overview (always emitted; source_paths from manifest sidecar if present)
    so_sidecar = plots_dir / "system_overview.manifest.json"
    so_sources = _load_source_paths(so_sidecar)
    specs.append(RuleSpec(
        rule_name="plot_system_overview",
        renderer_module="system_overview",
        input_flags=(),  # bundle-side regen-only: no status flag input
        output_path_template="plots/system_overview__OUTPUT_EXT__",
        source_paths=tuple(so_sources),
        wildcards=(),
        extra_cli_flags=(),
        extra_params=(),
        report_kwargs={
            "caption": "report/captions/system_map.rst",
            "category": "System Information",
            "labels": '{"figure": "System map"}',
        },
        resources_yaml="mem_mb=2000, time_min=10",
        log_path_template="_logs/plots/system_overview.log",
    ))

    # per_sim rules (wildcarded on event_id). Regen-mode: plot files
    # already exist in the bundle; the rules exist only as metadata for
    # `snakemake --report` to discover output paths. No upstream inputs,
    # no event_iloc lookups, no helper functions — keep the rules
    # parseable and stub-shaped.
    if not is_sensitivity:
        for renderer, figname in [
            ("per_sim_peak_flood_depth", "peak_flood_depth"),
            ("per_sim_conduit_flow", "conduit_flow"),
        ]:
            specs.append(RuleSpec(
                rule_name=f"plot_{renderer}",
                renderer_module=renderer,
                input_flags=(),
                output_path_template=f"plots/per_sim/{{event_id}}/{figname}__OUTPUT_EXT__",
                source_paths=(),
                wildcards=("event_id",),
                extra_cli_flags=(),
                extra_params=(),
                report_kwargs={
                    "caption": f"report/captions/per_sim_{figname}.rst",
                    "category": "Per Simulation Results",
                    "labels": (
                        '{"event_id": "{event_id}", "figure": "'
                        + figname.replace("_", " ").title() + '"}'
                    ),
                },
                resources_yaml="mem_mb=4000, time_min=15",
                log_path_template=f"_logs/plots/per_sim_{figname}_{{event_id}}.log",
            ))

    # per_analysis_summary (always emitted)
    pa_sidecar = plots_dir / "per_analysis" / "summary_table.manifest.json"
    pa_sources = _load_source_paths(pa_sidecar)
    specs.append(RuleSpec(
        rule_name="plot_per_analysis_summary_table",
        renderer_module="per_analysis_summary",
        input_flags=(),
        output_path_template="plots/per_analysis/summary_table__OUTPUT_EXT__",
        source_paths=tuple(pa_sources),
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
        log_path_template="_logs/plots/per_analysis_summary_table.log",
    ))

    # scenario_status_appendix and errors_and_warnings: always emitted per
    # Notes item 5 fallback option (source_paths=() when sidecar missing).
    for renderer, output_subpath, fig_label, category_kwargs in [
        (
            "scenario_status_appendix",
            "plots/appendix/scenario_status",
            "Per-scenario status table",
            {"category": "Appendix", "subcategory": "Scenario Status"},
        ),
        (
            "errors_and_warnings",
            "plots/errors_and_warnings/validation_report",
            "Validation report",
            {"category": "Errors and Warnings", "subcategory": "Validation Report"},
        ),
    ]:
        sidecar = plots_dir / (output_subpath + ".manifest.json")
        sources = _load_source_paths(sidecar) if sidecar.exists() else []
        specs.append(RuleSpec(
            rule_name=f"plot_{renderer}",
            renderer_module=renderer,
            input_flags=(),
            output_path_template=f"{output_subpath}__OUTPUT_EXT__",
            source_paths=tuple(sources),
            wildcards=(),
            extra_cli_flags=(),
            extra_params=(),
            report_kwargs={
                "caption": f"report/captions/{renderer}.rst",
                **category_kwargs,
                "labels": f'{{"figure": "{fig_label}"}}',
            },
            resources_yaml="mem_mb=1000, time_min=5",
            log_path_template=f"_logs/plots/{renderer}.log",
        ))

    # Sensitivity-master additional rules (stub-shaped — see per_sim
    # rationale: plots exist on disk, rules are metadata only).
    if is_sensitivity:
        specs.append(RuleSpec(
            rule_name="plot_sensitivity_benchmarking",
            renderer_module="sensitivity_benchmarking",
            input_flags=(),
            output_path_template="plots/sensitivity/benchmarking/{independent_var}_vs_total__OUTPUT_EXT__",
            source_paths=(),
            wildcards=("independent_var",),
            extra_cli_flags=(),
            extra_params=(),
            report_kwargs={
                "caption": "report/captions/sensitivity_benchmarking.rst",
                "category": "Key Results",
                "subcategory": "Benchmarking",
                "labels": '{"independent_var": "{independent_var}", "figure": "vs Total runtime"}',
            },
            resources_yaml="mem_mb=4000, time_min=10",
            log_path_template="_logs/plots/sensitivity_benchmarking_{independent_var}.log",
        ))
        for renderer_local, figname in [
            ("per_sim_per_sa_peak_flood_depth", "peak_flood_depth"),
            ("per_sim_per_sa_conduit_flow", "conduit_flow"),
        ]:
            specs.append(RuleSpec(
                rule_name=f"plot_{renderer_local}",
                renderer_module=renderer_local,
                input_flags=(),
                output_path_template=(
                    f"plots/sensitivity/per_sim/sa-{{sa_id}}/{{event_id}}/{figname}__OUTPUT_EXT__"
                ),
                source_paths=(),
                wildcards=("sa_id", "event_id"),
                extra_cli_flags=(),
                extra_params=(),
                report_kwargs={
                    "caption": f"report/captions/per_sim_{figname}.rst",
                    "category": "Per Simulation Results",
                    "labels": (
                        '{"sa_id": "{sa_id}", "event_id": "{event_id}", "figure": "'
                        + figname.replace("_", " ").title() + '"}'
                    ),
                },
                resources_yaml="mem_mb=4000, time_min=15",
                log_path_template=(
                    f"_logs/plots/per_sim_per_sa_{figname}_sa-{{sa_id}}_{{event_id}}.log"
                ),
            ))

    return tuple(specs)


def _load_source_paths(sidecar_path: Path) -> list[str]:
    """Read source_paths_relative from a plot manifest sidecar.

    Returns the bare relative-path list (lossy form); ``format_sources_rst``
    accepts ``list[str]`` and degrades gracefully (no variable sub-bullets).
    """
    if not sidecar_path.exists():
        return []
    data = json.loads(sidecar_path.read_text())
    return list(data.get("source_paths_relative", []))


def _build_preamble() -> str:
    """Emit the import preamble + _toolkit_version block (mirror of
    source-side workflow.py preamble).
    """
    return (
        "# Auto-generated by Bundle.regenerate_report\n"
        "\n"
        "import os\n"
        "import glob\n"
        "from datetime import datetime as _dt\n"
        "from TRITON_SWMM_toolkit.report_renderers._figure_emission import "
        "format_sources_rst as _fmt_sources_rst\n"
        "\n"
        "try:\n"
        '    from importlib.metadata import version as _pkg_version\n'
        '    _toolkit_version = _pkg_version("TRITON_SWMM_toolkit")\n'
        "except Exception:\n"
        '    _toolkit_version = "unknown"\n'
    )


def _build_config_block(*, cfg_analysis: dict, is_sensitivity: bool) -> str:
    """Emit the config[...] assignment block matching the source-side shape."""
    n_sims = int(cfg_analysis.get("n_sims", 0))
    analysis_id = cfg_analysis.get("analysis_id", "unknown")
    lines = [
        "# Config dict consumed by report_templates/workflow_description.rst.j2",
        f'config["analysis_id"] = {analysis_id!r}',
        'config["toolkit_version"] = _toolkit_version',
        f'config["n_sims"] = {n_sims}',
        f'config["is_sensitivity"] = {is_sensitivity}',
        'config["report"] = {"generated_at": _dt.now().isoformat(timespec="seconds")}',
    ]
    if is_sensitivity:
        n_sub = int(cfg_analysis.get("n_sub_analyses", 0))
        independent_vars = cfg_analysis.get("independent_vars", [])
        lines.append(f'config["n_sub_analyses"] = {n_sub}')
        lines.append(f'config["independent_vars"] = {independent_vars!r}')
        group_by = cfg_analysis.get("group_by_var")
        if group_by:
            lines.append(f'config["group_by_var"] = {group_by!r}')
    return "\n".join(lines) + "\n"


def _resolve_output_path(spec: RuleSpec, ctx: RuleEmissionContext) -> str:
    """Substitute __OUTPUT_EXT__ in the spec's output_path_template."""
    ext = _output_ext_for(ctx.static_backend, spec.renderer_module)
    return spec.output_path_template.replace("__OUTPUT_EXT__", ext)


def _expand_wildcards_to_existing_files(template: str, bundle_root: Path) -> tuple[str, ...]:
    """Glob-expand Snakemake wildcards in a path template against the
    bundle filesystem.

    rule_all and render_report inputs cannot be wildcarded (Snakemake
    needs concrete file paths to materialize the DAG). The
    regeneration-scoped Snakefile lists each existing plot file
    explicitly.

    Discovery strategy: substitute wildcards with ``*`` and glob first
    against the plot artifact itself; if no matches, fall back to
    globbing against the corresponding ``.manifest.json`` sidecar
    (since bundle fixtures may carry only the manifest sidecars, not
    the rendered plot files). Each wildcard match yields one expanded
    path. If neither matches, return the template verbatim so the
    Snakefile remains parseable for empty plot subsets.
    """
    import re as _re

    glob_pat = _re.sub(r"\{[^{}]+\}", "*", template)
    if glob_pat == template:
        return (template,)
    matches = sorted(str(p.relative_to(bundle_root)) for p in bundle_root.glob(glob_pat))
    if matches:
        return tuple(matches)
    # Fallback: glob for the manifest sidecar; the plot path is the manifest
    # stem with the originally-templated extension.
    ext_match = _re.search(r"\.[^./{}]+$", template)
    if not ext_match:
        return (template,)
    ext = ext_match.group(0)
    manifest_glob = glob_pat[: -len(ext)] + ".manifest.json"
    sidecars = sorted(bundle_root.glob(manifest_glob))
    if not sidecars:
        return (template,)
    return tuple(
        str(s.relative_to(bundle_root)).removesuffix(".manifest.json") + ext
        for s in sidecars
    )


def _status_flags_for(is_sensitivity: bool, cfg_analysis: dict) -> tuple[str, ...]:
    """Consume-side regen-only: returns empty tuple.

    The source-side rule_all references _status/*.flag inputs from
    simulation/processing/consolidation rules. The regen Snakefile emits
    no such rules and the bundle does not carry the flag files, so
    including them in consume-side rule_all would trigger
    MissingInputException at parse time. The render_report rule consumes
    plot outputs directly.
    """
    return ()
