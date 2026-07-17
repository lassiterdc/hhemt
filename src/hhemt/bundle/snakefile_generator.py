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

from hhemt.workflow import (
    RuleEmissionContext,
    RuleSpec,
    _emit_plot_rule,
    _emit_render_report_rule,
    _emit_rule_all,
    _output_ext_for,
)

_RENDER_REPORT_RESOURCES_YAML = "        mem_mb=2000,\n        time_min=30,\n        cpus_per_task=1,"


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
    config_block = _build_config_block(cfg_analysis=cfg_analysis, is_sensitivity=is_sensitivity)
    report_directive = 'report: "report/workflow_description.rst"\n'

    plot_output_paths = tuple(
        f'"{p}"'
        for spec in rule_specs
        for p in _expand_wildcards_to_existing_files(_resolve_output_path(spec, ctx), bundle_root)
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

    return "\n".join(
        [
            preamble,
            config_block,
            report_directive,
            rule_all_block,
            plot_rule_blocks,
            render_report_block,
        ]
    )


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
        config_args_str=("--system-config cfg_system.yaml \\\n            --analysis-config cfg_analysis.yaml"),
        is_sensitivity=is_sensitivity,
        static_backend=static_backend,
    )


def _harvest_rule_specs(
    *,
    bundle_root: Path,
    is_sensitivity: bool,
    cfg_analysis: dict,
) -> tuple[RuleSpec, ...]:
    """Construct the RuleSpec list for the regeneration-scoped Snakefile by
    iterating the active reporting set's renderer_selection (P1b / TO-8).

    Each figure's bundle facts (rule_name, renderer_module, output path,
    report_kwargs, wildcards, resources, log) live on the registry's
    rule_spec_template tuple, so the registry is the single source the source-side
    builders' report(category=) strings are cross-checked against
    (tests/test_reporting_set_cosourcing.py). Adding a set or renderer is a
    registry edit; this harvest carries no per-renderer branch. The harvest still
    owns the two runtime-only facts: source_paths (read from a plot manifest
    sidecar for non-wildcarded figures; () for wildcarded figures, whose concrete
    paths are glob-expanded downstream) and input_flags=() (regeneration-scoped --
    the bundle carries no _status/*.flag inputs).

    is_sensitivity selects the shipped set, mirroring
    config.report.resolve_active_reporting_set_name for reporting_set="default"
    (default -> benchmarking when sensitivity, else the standard set).
    """
    from hhemt.report_renderers._reporting_sets import (
        get_reporting_set,
        renderer_active,
    )

    # Read the active reporting-set selection from config rather than hardcoding
    # the default→benchmarking/default resolution: a bundle whose source run
    # selected a non-default set (e.g. "compute-sensitivity") must regenerate the
    # SAME renderer set the source side emitted, or the bundle Snakefile diverges
    # from the source Snakefile. Mirror config.report.resolve_active_reporting_set_name's
    # sentinel handling: the "default" sentinel (and an absent field) resolves to
    # "benchmarking" when sensitivity else the standard set; any other name is taken
    # verbatim.
    _report = (cfg_analysis or {}).get("report") or {}
    _set_name = _report.get("reporting_set") or "default"
    if _set_name == "default":
        _set_name = "benchmarking" if is_sensitivity else "default"
    active_set = get_reporting_set(_set_name)
    # Phase 3: per-plot disable. Filtering the harvest's renderer_selection drops
    # the renderer's RuleSpec(s), which in turn drops its plot_output_paths from
    # BOTH rule all and render_report (both derive from rule_specs in
    # generate_regeneration_snakefile), so the bundle stays in emission/enumeration
    # lockstep in one place.
    _disabled = _report.get("disabled_renderers") or []

    specs: list[RuleSpec] = []
    for sel in active_set.renderer_selection:
        if not renderer_active(sel.builder_key, _disabled):
            continue
        for tmpl in sel.rule_spec_template:
            if tmpl.wildcards:
                source_paths: tuple[str, ...] = ()
            else:
                sidecar = bundle_root / tmpl.output_path_template.replace("__OUTPUT_EXT__", ".manifest.json")
                source_paths = tuple(_load_source_paths(sidecar))
            specs.append(
                RuleSpec(
                    rule_name=tmpl.rule_name,
                    renderer_module=tmpl.renderer_module,
                    input_flags=(),  # bundle-side regen-only: no status flag input
                    output_path_template=tmpl.output_path_template,
                    source_paths=source_paths,
                    wildcards=tmpl.wildcards,
                    extra_cli_flags=(),
                    extra_params=(),
                    report_kwargs=dict(tmpl.report_kwargs),
                    resources_yaml=tmpl.resources_yaml,
                    log_path_template=tmpl.log_path_template,
                )
            )
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
        "from hhemt.report_renderers._figure_emission import "
        "format_sources_rst as _fmt_sources_rst\n"
        "\n"
        "try:\n"
        "    from importlib.metadata import version as _pkg_version\n"
        '    _toolkit_version = _pkg_version("hhemt")\n'
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
    return tuple(str(s.relative_to(bundle_root)).removesuffix(".manifest.json") + ext for s in sidecars)


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
