"""Combined-bundle regeneration-scoped Snakefile generator (PIP-1 Phase 5).

Emits the combined cross-experiment report as a FIRST-CLASS Snakemake ``--report``
(matching single-bundle report chrome/nav/CSS), reusing the module-level
``_emit_plot_rule`` / ``_emit_rule_all`` / ``RuleEmissionContext`` / ``RuleSpec``
primitives — the same single-source ``report()``-emission the bundle and static
generators use. Modeled on ``static_snakefile_generator.py`` (a dedicated sibling
that reuses the shared primitives rather than branching the single-analysis
generator, which is coupled to cfg_analysis/cfg_system/status-flags a combined
bundle does not have).

Scope: exactly the ``combined`` ReportingSet's two cross-experiment renderers
(``cross_experiment_compatibility`` -> "Cross-Experiment Compatibility",
``cross_experiment_intercomparison`` -> "Cross-Experiment Results"), both fixed
(no wildcards). The figures are PRE-RENDERED at emit/regen time by
``_combine._render_combined_report``'s direct-invoke loop (they are NEW and do not
pre-exist), so the plot-rule shells are INERT under the ``--touch`` + ``--report``
render path — the same dead-shell contract the bundle-regen plot rules carry (their
figures are pre-rendered/bundled too). This generator's sole job is to give
``snakemake --report`` the ``report()`` annotations + ``report:`` workflow-description
directive. Static backend is fixed to "plotly": the two combined renderers emit
``.html`` unconditionally (HTML-string ``emit_plot_with_sources`` branch), and their
``_OUTPUT_EXT_BY_RENDERER`` entries are ``.html`` for both backends.
"""

from __future__ import annotations

from pathlib import Path

from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.subprocess_utils import run_subprocess_with_tee
from hhemt.workflow import (
    RuleEmissionContext,
    RuleSpec,
    _emit_plot_rule,
    _emit_rule_all,
    _output_ext_for,
)

# The combined renderers emit HTML tables unconditionally; a combined bundle has no
# cfg_analysis, so static_backend is fixed. Both _OUTPUT_EXT_BY_RENDERER entries are
# .html, so this resolves .html for both figures.
_COMBINED_STATIC_BACKEND = "plotly"
_COMBINED_NAVBAR_TEXT = "Combined cross-experiment report"

# Static combined workflow-description (NO config refs, NO timestamp) so the rendered
# report is byte-deterministic. RST title underline must be >= title length.
_COMBINED_WORKFLOW_DESCRIPTION = """\
================================
Combined cross-experiment report
================================

Overview
--------

This report combines multiple completed experiment bundles into one
cross-experiment comparison. Browse the **Cross-Experiment Compatibility**
sidebar category for the metadata-compatibility summary across the combined
experiments, and the **Cross-Experiment Results** category for the
clean-vs-resume intercomparison of key-result summaries.

Each combined experiment retains its own intact render bundle under
``child_crates/``; run the per-experiment report from there for
single-experiment detail.
"""


def _combined_preamble() -> str:
    """Snakefile preamble: the format_sources_rst helper _emit_plot_rule's params
    reference. Combined rules need no config: block — the workflow-description and
    captions carry no config refs (only the plot-rule params.source_paths_rst)."""
    return "from hhemt.report_renderers._figure_emission import format_sources_rst as _fmt_sources_rst\n"


def _load_source_paths(sidecar_path: Path) -> list[str]:
    """Read source_paths_relative from a plot manifest sidecar (mirrors
    bundle.snakefile_generator._load_source_paths). The two figures + their manifests
    are written by the direct-invoke render before this generator runs."""
    import json

    if not sidecar_path.exists():
        return []
    data = json.loads(sidecar_path.read_text())
    return list(data.get("source_paths_relative", []))


def _harvest_combined_rule_specs(bundle_root: Path) -> tuple[RuleSpec, ...]:
    """One RuleSpec per combined-set figure template (both fixed / no wildcards).

    source_paths is harvested from the figure's manifest sidecar (the read-model the
    renderer declared), threaded into the caption's Sources block. No status-flag
    input; no wildcards; report_kwargs carried verbatim from the registry template.
    """
    rset = get_reporting_set("combined")
    specs: list[RuleSpec] = []
    for sel in rset.renderer_selection:
        for tmpl in sel.rule_spec_template:
            sidecar = bundle_root / tmpl.output_path_template.replace("__OUTPUT_EXT__", ".manifest.json")
            specs.append(
                RuleSpec(
                    rule_name=tmpl.rule_name,
                    renderer_module=tmpl.renderer_module,
                    input_flags=(),  # regen-only: figures pre-rendered, no upstream flag
                    output_path_template=tmpl.output_path_template,
                    source_paths=tuple(_load_source_paths(sidecar)),
                    wildcards=(),
                    extra_cli_flags=(),
                    extra_params=(),
                    report_kwargs=dict(tmpl.report_kwargs),
                    resources_yaml=tmpl.resources_yaml,
                    log_path_template=tmpl.log_path_template,
                )
            )
    return tuple(specs)


def _child_experiment_ids(bundle_root: Path) -> list[str]:
    """Sorted child experiment ids from the child_crates/ subdir names."""
    crates = bundle_root / "child_crates"
    if not crates.exists():
        return []
    return sorted(p.name for p in crates.iterdir() if p.is_dir())


def _harvest_per_experiment_rule_specs(bundle_root: Path) -> tuple[RuleSpec, ...]:
    """One concrete (non-wildcarded) RuleSpec per EXISTING child figure file, IN PLACE.

    Option B: for each child_crates/{eid}/, resolve the child's active ReportingSet,
    glob-expand each figure template's output pattern against the child's on-disk
    plots/, and emit one report() rule per concrete figure with the category rewrite
    category={eid} / subcategory={child figure's category}. Reuses the single-bundle
    harvest primitives (bundle.snakefile_generator._expand_wildcards_to_existing_files
    + _load_source_paths). Any on-disk plots/**/*.html no template claims lands under
    subcategory="Other" (defensive; never drop a rendered figure).
    """
    import json as _json
    import re as _re

    import yaml as _yaml

    from hhemt.bundle.snakefile_generator import (
        _expand_wildcards_to_existing_files,
        _load_source_paths,
    )
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    specs: list[RuleSpec] = []
    for eid in _child_experiment_ids(bundle_root):
        child = bundle_root / "child_crates" / eid
        cfg = _yaml.safe_load((child / "cfg_analysis.yaml").read_text())
        # Resolve the child's ACTUAL set (captures compute-sensitivity's eda figure),
        # falling back to the is_sensitivity heuristic that _harvest_rule_specs uses.
        set_name = (cfg.get("report") or {}).get("reporting_set")
        if not set_name or set_name == "default":
            set_name = "benchmarking" if cfg.get("toggle_sensitivity_analysis") else "default"
        active_set = get_reporting_set(set_name)

        claimed: set[Path] = set()
        for sel in active_set.renderer_selection:
            for tmpl in sel.rule_spec_template:
                pat = tmpl.output_path_template.replace("__OUTPUT_EXT__", ".html")
                for rel in _expand_wildcards_to_existing_files(pat, child):
                    fpath = child / rel
                    if not fpath.exists():
                        continue  # manifest-only fixture entry; nothing to embed
                    claimed.add(fpath)
                    out_rel = f"child_crates/{eid}/{rel}"
                    sidecar = fpath.with_suffix(".manifest.json")
                    if sidecar.exists():
                        plot_id = _json.loads(sidecar.read_text()).get("plot_id", fpath.stem)
                    else:
                        plot_id = fpath.stem
                    rk_child = tmpl.report_kwargs
                    sub = rk_child.get("category", "Other")
                    rn = _re.sub(r"[^A-Za-z0-9_]", "_", f"harvest__{eid}__{plot_id}")
                    specs.append(
                        RuleSpec(
                            rule_name=rn,
                            renderer_module=tmpl.renderer_module,
                            input_flags=(),
                            output_path_template=out_rel,  # concrete; no __OUTPUT_EXT__
                            source_paths=tuple(_load_source_paths(sidecar)),
                            wildcards=(),
                            extra_cli_flags=(),
                            extra_params=(("experiment_id", repr(eid)),),
                            report_kwargs={
                                "caption": "report/captions/_harvested.rst",
                                "category": eid,
                                "subcategory": sub,
                                "labels": _json.dumps({"experiment": eid, "figure": plot_id}),
                            },
                            resources_yaml="mem_mb=1000, time_min=5",
                            log_path_template=f"_logs/plots/harvest_{rn}.log",
                        )
                    )
        # Defensive: any child figure no template claimed -> {eid} / "Other".
        for fpath in sorted(child.glob("plots/**/*.html")):
            if fpath in claimed:
                continue
            rel = fpath.relative_to(child).as_posix()
            sidecar = fpath.with_suffix(".manifest.json")
            if sidecar.exists():
                plot_id = _json.loads(sidecar.read_text()).get("plot_id", fpath.stem)
            else:
                plot_id = fpath.stem
            rn = _re.sub(r"[^A-Za-z0-9_]", "_", f"harvest__{eid}__{plot_id}")
            specs.append(
                RuleSpec(
                    rule_name=rn,
                    renderer_module="cross_experiment_compatibility",  # any keyed renderer; shell inert
                    input_flags=(),
                    output_path_template=f"child_crates/{eid}/{rel}",
                    source_paths=tuple(_load_source_paths(sidecar)),
                    wildcards=(),
                    extra_cli_flags=(),
                    extra_params=(("experiment_id", repr(eid)),),
                    report_kwargs={
                        "caption": "report/captions/_harvested.rst",
                        "category": eid,
                        "subcategory": "Other",
                        "labels": _json.dumps({"experiment": eid, "figure": plot_id}),
                    },
                    resources_yaml="mem_mb=1000, time_min=5",
                    log_path_template=f"_logs/plots/harvest_{rn}.log",
                )
            )
    return tuple(specs)


def generate_combined_snakefile(bundle_root: Path) -> str:
    """Generate the combined regeneration-scoped Snakefile body as a string."""
    bundle_root = Path(bundle_root).resolve()
    ctx = RuleEmissionContext(
        python_executable="python",
        log_dir_rel="_logs",
        conda_env_path="",
        # INERT: the figures are pre-rendered; the plot-rule shells never run under
        # --touch + --report (the same dead-shell contract as the bundle-regen plot
        # rules). The cfg args are shape-parity only — a combined bundle has none.
        config_args_str="--system-config cfg_system.yaml \\\n            --analysis-config cfg_analysis.yaml",
        is_sensitivity=False,
        static_backend=_COMBINED_STATIC_BACKEND,
    )
    rule_specs = _harvest_combined_rule_specs(bundle_root) + _harvest_per_experiment_rule_specs(bundle_root)
    preamble = _combined_preamble()
    report_directive = 'report: "report/workflow_description.rst"\n'

    def _plot_out(spec: RuleSpec) -> str:
        ext = _output_ext_for(ctx.static_backend, spec.renderer_module)
        return '"' + spec.output_path_template.replace("__OUTPUT_EXT__", ext) + '"'

    plot_output_paths = tuple(_plot_out(spec) for spec in rule_specs)
    # NO render_report rule: the report is produced by the direct `snakemake --report`
    # call (render_combined_report_via_snakemake), so rule all lists only the figures.
    rule_all_block = _emit_rule_all(
        status_flags=(),
        plot_output_paths=plot_output_paths,
        render_report_targets=(),
        ctx=ctx,
    )
    plot_rule_blocks = "".join(_emit_plot_rule(spec, ctx) for spec in rule_specs)
    return "\n".join([preamble, report_directive, rule_all_block, plot_rule_blocks])


def write_combined_snakefile(bundle_root: Path) -> Path:
    """Overwrite {bundle_root}/Snakefile with the combined regeneration Snakefile."""
    bundle_root = Path(bundle_root).resolve()
    out = bundle_root / "Snakefile"
    out.write_text(generate_combined_snakefile(bundle_root) + "\n")
    return out


def stage_combined_report_artifacts(bundle_root: Path) -> None:
    """Stage report.css (DEFAULT_BRAND_THEME), captions, and the STATIC combined
    workflow-description. Overwrites the single-analysis workflow_description
    _emit_report_artifacts stages with the combined static one."""
    from hhemt.config.brand_theme import DEFAULT_BRAND_THEME
    from hhemt.workflow import _brand_theme_css_map, _emit_report_artifacts

    bundle_root = Path(bundle_root).resolve()
    _emit_report_artifacts(bundle_root, brand_theme=_brand_theme_css_map(DEFAULT_BRAND_THEME))
    (bundle_root / "report" / "workflow_description.rst").write_text(_COMBINED_WORKFLOW_DESCRIPTION)


def render_combined_report_via_snakemake(bundle_root: Path, *, formats: tuple[str, ...] = ("html", "zip")) -> None:
    """Stage artifacts + write the combined Snakefile + `snakemake --touch` the
    pre-rendered figures + `snakemake --report` (+ React surgery) per format.

    Mirrors Bundle.regenerate_report's touch->report->surgery sequence. The two
    figures already exist on disk (direct-rendered by _render_combined_report before
    this call); --touch marks them up-to-date so --report does not try to re-run their
    (inert) plot rules.
    """
    bundle_root = Path(bundle_root).resolve()
    stage_combined_report_artifacts(bundle_root)
    write_combined_snakefile(bundle_root)

    # Defense-in-depth stale-lock check (mirrors Bundle.regenerate_report).
    locks_dir = bundle_root / ".snakemake" / "locks"
    if locks_dir.exists() and any(locks_dir.iterdir()):
        lock_paths = sorted(p.name for p in locks_dir.iterdir())
        raise RuntimeError(
            f"Stale Snakemake locks under {locks_dir}: {lock_paths}. Run "
            f"`python -m snakemake --unlock --snakefile {bundle_root}/Snakefile "
            f"--directory {bundle_root}` to clear them."
        )

    logs_dir = bundle_root / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    touch_cmd = [
        "snakemake",
        "--snakefile",
        str(bundle_root / "Snakefile"),
        "--directory",
        str(bundle_root),
        "--cores",
        "1",
        "--touch",
        "--quiet",
    ]
    touch_proc = run_subprocess_with_tee(
        touch_cmd,
        logfile=logs_dir / "combined_touch.log",
        cwd=bundle_root,
        echo_to_stdout=False,
    )
    if touch_proc.returncode != 0:
        raise RuntimeError(
            f"combined snakemake --touch failed (exit {touch_proc.returncode}). See {logs_dir / 'combined_touch.log'}"
        )

    from hhemt.report_renderers._react_surgery import (
        apply_post_process_surgery,
        apply_post_process_surgery_to_zip,
    )

    # Dynamic per-experiment order: splice the sorted child experiment ids between the
    # cross-experiment results and the aggregate errors-and-warnings bookend, so each
    # {eid} top-level category is ordered (unknown categories otherwise sort last in
    # _react_surgery). Derived from child_crates/ on disk so regenerate_report() with no
    # re-merge recomputes it identically.
    _fixed = list(get_reporting_set("combined").category_order)  # [..Compat, ..Results, Errors and Warnings]
    _eids = _child_experiment_ids(bundle_root)
    if "Errors and Warnings" in _fixed:
        _i = _fixed.index("Errors and Warnings")
        category_order = _fixed[:_i] + _eids + _fixed[_i:]
    else:
        category_order = _fixed + _eids
    for fmt in formats:
        output_path = bundle_root / f"analysis_report.{fmt}"
        cmd = [
            "snakemake",
            "--snakefile",
            str(bundle_root / "Snakefile"),
            "--directory",
            str(bundle_root),
            "--cores",
            "1",
            "--report",
            str(output_path),
            "--report-stylesheet",
            str(bundle_root / "report" / "report.css"),
            "--quiet",
        ]
        proc = run_subprocess_with_tee(
            cmd,
            logfile=logs_dir / f"combined_report_{fmt}.log",
            cwd=bundle_root,
            echo_to_stdout=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"combined snakemake --report ({fmt}) failed (exit {proc.returncode}). "
                f"See {logs_dir / f'combined_report_{fmt}.log'}"
            )
        try:
            if fmt == "html":
                output_path.write_text(
                    apply_post_process_surgery(
                        output_path.read_text(),
                        bundle_mode=True,
                        navbar_text=_COMBINED_NAVBAR_TEXT,
                        category_order=category_order,
                    )
                )
            else:
                apply_post_process_surgery_to_zip(
                    output_path,
                    bundle_mode=True,
                    navbar_text=_COMBINED_NAVBAR_TEXT,
                    category_order=category_order,
                )
        except Exception:
            pass
