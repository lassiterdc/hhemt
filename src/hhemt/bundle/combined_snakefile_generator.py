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

import warnings
from pathlib import Path

from hhemt.report_plot_ids import humanize_plot_id
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


#: Model tokens stripped to collapse two single-model bundles of one experiment into one
#: base-experiment category (F1 Option C). tritonswmm is checked FIRST (it is the longer token).
_MODEL_TOKENS: tuple[str, ...] = ("_tritonswmm", "_triton")


def _base_experiment(eid: str) -> str:
    """The experiment id with its trailing model token stripped, so `synth_cc_clean_triton`
    and `synth_cc_clean_tritonswmm` both collapse to `synth_cc_clean` (one report section
    carrying BOTH model arms, F1)."""
    for _tok in _MODEL_TOKENS:
        if eid.endswith(_tok):
            return eid[: -len(_tok)]
    return eid


def _model_of(eid: str) -> str:
    """The model arm a child bundle carries ('triton' / 'tritonswmm'), from its eid suffix;
    '' when the eid carries no model token. Emitted in each harvested figure's labels so a
    reader (and the data-viz .model-pair layout) can distinguish the two arms."""
    for _tok in _MODEL_TOKENS:
        if eid.endswith(_tok):
            return _tok[1:]
    return ""


#: data-viz R5-1's stacked both-models CSS, INLINED into each composed page's <head> because the
#: report.css --report-stylesheet reaches only the report CHROME, never a report()'d figure's inner
#: (data-URI iframe) document (F1 round-6, FQ2). Consumed VERBATIM from the R5-1 spec; the sole
#: reconciliation delta is the .model-arm-frame rule (R5-1 assumed inlined figures/tables — hhemt
#: embeds each arm as a full self-contained plotly doc via <iframe srcdoc>, which needs an explicit
#: height). var(--uva-*) references resolve from the :root{} block _extract_root_css_vars pulls out
#: of the staged report.css (brand_theme stipulation: brand colors flow from brand_theme). The
#: fallbacks are NEUTRAL non-brand values so no UVA hex is hardcoded here.
_MODEL_STACK_CSS = """
.model-pair-page{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:1rem;}
.model-pair-heading{color:var(--uva-blue,#1a1a1a);font-size:1.15rem;margin:0 0 .5rem;}
.model-stack{display:flex;flex-direction:column;margin:1.5rem 0;}
.model-stack>.model-section{width:100%;padding:1rem 0;}
.model-stack>.model-section[data-model="tritonswmm"]{order:0;}
.model-stack>.model-section[data-model="triton"]{order:1;border-top:2px solid var(--uva-medium-gray,#cccccc);margin-top:.5rem;}
.model-stack>.model-section[data-model]::before{display:block;font-weight:bold;color:var(--uva-blue,#1a1a1a);margin-bottom:.5rem;font-size:1.05rem;}
.model-stack>.model-section[data-model="tritonswmm"]::before{content:"TRITON-SWMM";}
.model-stack>.model-section[data-model="triton"]::before{content:"TRITON";}
.model-stack>.model-section img,.model-stack>.model-section iframe,.model-stack>.model-section svg,.model-stack>.model-section table{max-width:100%;}
.model-arm-frame{width:100%;height:900px;border:0;overflow:auto;}
.model-stack>.model-section[data-model-absent],.model-stack>.model-section.model-absent{color:var(--uva-text-gray,#6b7280);background:var(--uva-light-gray,#f1f3f5);border:1px dashed var(--uva-medium-gray,#cccccc);border-radius:4px;padding:.75rem 1rem;font-style:italic;}
"""


def _extract_root_css_vars(bundle_root: Path) -> str:
    """The staged report.css :root{...} custom-property block, so a composed figure page's inlined
    R5-1 CSS resolves its var(--uva-*) references from the SAME brand_theme the report chrome uses
    (brand_theme stipulation). Graceful-absent: '' when report.css is not yet staged (e.g. under the
    CR4 monkeypatched test path), where _MODEL_STACK_CSS's neutral non-brand fallbacks apply."""
    import re as _re

    css = bundle_root / "report" / "report.css"
    if not css.exists():
        return ""
    m = _re.search(r":root\s*\{[^}]*\}", css.read_text(encoding="utf-8", errors="replace"))
    return m.group(0) if m else ""


def _compose_model_pair_page(
    base_eid: str, plot_id: str, ts_html: str, tri_html: str | None, root_css_vars: str = ""
) -> str:
    """Compose ONE self-contained page STACKING the TRITON-SWMM figure (top) and its pure-TRITON
    counterpart (below, when present) as a data-viz .model-stack / .model-section[data-model] column
    (F1 round-6: hhemt VMS 5.2 structure + data-viz R5-1 CSS, reconciled). Each arm is a full
    self-contained plotly document embedded via an isolated <iframe srcdoc> (EMPIRICALLY VERIFIED:
    snakemake --report base64-passes the composed page verbatim and the report app renders it in an
    <iframe src=data-uri>, so nested srcdoc iframes are proper opaque-origin docs — no id collision,
    no double-plotly-bundle, no figure-surgery). R5-1's CSS is INLINED here because report.css
    (--report-stylesheet) styles the report chrome only and never reaches a figure's inner document.
    tri_html is None for a coupled-only (SWMM-specific) figure -> only the tritonswmm section renders.
    The inner iframes carry an onload auto-grow (srcdoc inherits the composed page's opaque origin, so
    the DOM read succeeds in practice) with a try/catch fallback to the CSS 900px height."""
    import html as _html

    def _section(model: str, inner: str) -> str:
        return (
            f'<div class="model-section" data-model="{_html.escape(model, quote=True)}">'
            f'<iframe class="model-arm-frame" loading="lazy" '
            f'title="{_html.escape(base_eid)} {_html.escape(plot_id)} ({_html.escape(model)})" '
            "onload=\"try{this.style.height=(this.contentWindow.document.documentElement.scrollHeight+24)+'px';}catch(e){}\" "
            f'srcdoc="{_html.escape(inner, quote=True)}"></iframe></div>'
        )

    body = _section("tritonswmm", ts_html)
    if tri_html is not None:
        body += _section("triton", tri_html)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{_html.escape(base_eid)} — {_html.escape(humanize_plot_id(plot_id))}</title>"
        f"<style>{root_css_vars}\n{_MODEL_STACK_CSS}</style></head>"
        '<body class="model-pair-page">'
        f'<h2 class="model-pair-heading">{_html.escape(base_eid)} — {_html.escape(humanize_plot_id(plot_id))}</h2>'
        f'<div class="model-stack" data-plot="{_html.escape(plot_id, quote=True)}" '
        f'data-config="{_html.escape(base_eid, quote=True)}">{body}</div>'
        "</body></html>"
    )


def _harvest_per_experiment_rule_specs(bundle_root: Path) -> tuple[RuleSpec, ...]:
    """ONE composed report() page per (base-experiment, plot_id): the TRITON-SWMM figure on top,
    its pure-TRITON counterpart stacked BELOW on the SAME page (F1 within-page co-location, round-6;
    supersedes the round-3 base-experiment-CATEGORY collapse). The composed page carries the figure's
    ORIGINAL category (NO extra base-experiment sidebar entry -> minimal navigation). The pure-TRITON
    arm's figures already exist SWMM-free in the *_triton child bundle, so co-location needs no
    figure-surgery. Coupled-only figures (SWMM-specific, no pure-TRITON counterpart) render the
    tritonswmm section alone. Composed pages are written under paired_figures/ (OUTSIDE the crate-
    hashed child_crates/; CR4-safe -- the whole harvest runs under the CR4-monkeypatched
    _render_combined_report). R5-1's .model-stack CSS is inlined per page (FQ2); its var(--uva-*)
    resolves from the staged report.css :root{} (brand_theme stipulation)."""
    import json as _json
    import re as _re

    import yaml as _yaml

    from hhemt.bundle.snakefile_generator import (
        _expand_wildcards_to_existing_files,
        _load_source_paths,
    )
    from hhemt.config.eda import _RETIRED_EDA_FIGURE_STEMS
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    def _figures_of(eid: str) -> list[tuple]:
        """(plot_id, category, subcategory, rel, html_path, sidecar) for one child bundle,
        template-claimed figures first then defensively-globbed leftovers."""
        child = bundle_root / "child_crates" / eid
        cfg = _yaml.safe_load((child / "cfg_analysis.yaml").read_text())
        set_name = (cfg.get("report") or {}).get("reporting_set")
        if not set_name or set_name == "default":
            set_name = "benchmarking" if cfg.get("toggle_sensitivity_analysis") else "default"
        active_set = get_reporting_set(set_name)
        out: list[tuple] = []
        claimed: set[Path] = set()
        for sel in active_set.renderer_selection:
            for tmpl in sel.rule_spec_template:
                pat = tmpl.output_path_template.replace("__OUTPUT_EXT__", ".html")
                for rel in _expand_wildcards_to_existing_files(pat, child):
                    fpath = child / rel
                    if not fpath.exists():
                        continue
                    claimed.add(fpath)
                    sidecar = fpath.with_suffix(".manifest.json")
                    plot_id = _json.loads(sidecar.read_text()).get("plot_id", fpath.stem) if sidecar.exists() else fpath.stem
                    rk = tmpl.report_kwargs
                    out.append((plot_id, rk.get("category", "Other"), rk.get("subcategory", ""), rel, fpath, sidecar))
        for fpath in sorted(child.glob("plots/**/*.html")):
            if fpath in claimed:
                continue
            if fpath.stem in _RETIRED_EDA_FIGURE_STEMS:
                # A retired EDA figure (config-drift) left on a no-wipe re-render dir and
                # carried into the child bundle by the wholesale plots/ copytree. Never
                # surface it in the combined report (config/eda.py::_RETIRED_EDA_FIGURE_STEMS).
                continue
            sidecar = fpath.with_suffix(".manifest.json")
            plot_id = _json.loads(sidecar.read_text()).get("plot_id", fpath.stem) if sidecar.exists() else fpath.stem
            out.append((plot_id, "Other", "", fpath.relative_to(child).as_posix(), fpath, sidecar))
        return out

    root_css_vars = _extract_root_css_vars(bundle_root)

    # group child bundles by base experiment -> {model_arm: eid}
    by_base: dict[str, dict[str, str]] = {}
    for eid in _child_experiment_ids(bundle_root):
        # A child eid carrying no model token keys by its OWN eid, never a shared literal:
        # two model-less children under one base would otherwise overwrite each other and
        # one would be dropped with no signal.
        by_base.setdefault(_base_experiment(eid), {})[_model_of(eid) or eid] = eid

    paired_dir = bundle_root / "paired_figures"
    specs: list[RuleSpec] = []
    for base, arms in sorted(by_base.items()):
        ts_eid = arms.get("tritonswmm")
        tri_eid = arms.get("triton")
        # tritonswmm is primary (top); a triton-only base uses triton. A base whose children
        # carry NO model token (every pre-model-arm bundle, and every combine outside this
        # campaign) has neither, and falling through to `continue` harvested ZERO figures for
        # it silently. Such a base renders its single section alone, exactly as a coupled-only
        # figure does.
        primary_eid = ts_eid or tri_eid or next(
            (arms[k] for k in sorted(arms) if k not in ("tritonswmm", "triton")), None
        )
        if primary_eid is None:
            warnings.warn(
                f"Combined report: base experiment {base!r} contributed no harvestable "
                f"child bundle (arms={sorted(arms)!r}); its per-experiment figures are "
                "ABSENT from the combined report.",
                stacklevel=2,
            )
            continue
        # pure-TRITON counterpart figures by plot_id, only when the primary IS the coupled arm
        counterpart = {f[0]: f for f in _figures_of(tri_eid)} if (ts_eid and tri_eid) else {}
        for plot_id, category, subcategory, rel, fpath, sidecar in _figures_of(primary_eid):
            ts_html = fpath.read_text(encoding="utf-8", errors="replace")
            cf = counterpart.get(plot_id)
            tri_html = cf[4].read_text(encoding="utf-8", errors="replace") if cf is not None else None
            page_html = _compose_model_pair_page(base, plot_id, ts_html, tri_html, root_css_vars)
            (paired_dir / _re.sub(r"[^A-Za-z0-9_.]", "_", base)).mkdir(parents=True, exist_ok=True)
            # Per-base SUBDIRECTORY, so the page's filename STEM is exactly the plot id: the
            # Snakemake figure-card name derives from that stem, and the ADR-2 humanizer's
            # curated _RENDERER_KIND_LABELS table only fires when the stem's leading segment IS
            # the renderer kind. A flat "{base}__{plot_id}" stem makes the humanizer read the
            # base eid as the renderer kind and silently drops the curated label. The base is
            # not lost: it is the directory here and report_kwargs["subcategory"] below.
            page_rel = (
                f"paired_figures/{_re.sub(r'[^A-Za-z0-9_.]', '_', base)}"
                f"/{_re.sub(r'[^A-Za-z0-9_.]', '_', plot_id)}.html"
            )
            (bundle_root / page_rel).write_text(page_html, encoding="utf-8")
            srcs = list(_load_source_paths(sidecar))
            if cf is not None:
                srcs += list(_load_source_paths(cf[5]))
            rn = _re.sub(r"[^A-Za-z0-9_]", "_", f"paired__{base}__{plot_id}")
            specs.append(
                RuleSpec(
                    rule_name=rn,
                    renderer_module="cross_experiment_compatibility",  # any keyed renderer; shell inert (page pre-composed)
                    input_flags=(),
                    output_path_template=page_rel,
                    source_paths=tuple(srcs),
                    wildcards=(),
                    extra_cli_flags=(),
                    extra_params=(("experiment_id", repr(base)),),
                    report_kwargs={
                        "caption": "report/captions/_harvested.rst",
                        "category": category,  # the figure's ORIGINAL category (NO base-experiment category)
                        "subcategory": base,  # clean/resume grouped as a subcategory within the category
                        "labels": _json.dumps(
                            {
                                "experiment": base,
                                "figure": humanize_plot_id(plot_id),
                                "models": "tritonswmm+triton" if tri_html is not None else (_model_of(primary_eid) or "tritonswmm"),
                            }
                        ),
                    },
                    resources_yaml="mem_mb=1000, time_min=5",
                    log_path_template=f"_logs/plots/{rn}.log",
                )
            )
    return tuple(specs)


def _distinct_child_categories(bundle_root: Path) -> list[str]:
    """Ordered distinct figure categories the composed per-experiment pages use (each child figure's
    ORIGINAL report category), excluding the cross-experiment bookends. Round-6: replaces the base-
    experiment-category collapse in the combined sidebar order (line 371)."""
    import yaml as _yaml

    from hhemt.report_renderers._reporting_sets import get_reporting_set

    combined_cats = set(get_reporting_set("combined").category_order)
    seen: list[str] = []
    for eid in _child_experiment_ids(bundle_root):
        cfg = _yaml.safe_load((bundle_root / "child_crates" / eid / "cfg_analysis.yaml").read_text())
        set_name = (cfg.get("report") or {}).get("reporting_set")
        if not set_name or set_name == "default":
            set_name = "benchmarking" if cfg.get("toggle_sensitivity_analysis") else "default"
        active_set = get_reporting_set(set_name)
        # Only categories the set's rule-spec TEMPLATES actually declare can reach a composed
        # page. category_order is the sidebar VOCABULARY and is deliberately non-exhaustive in
        # both directions (tests/test_reporting_set_cosourcing.py): it also carries chrome-only
        # entries no figure ever bears -- notably "Simulation Health (placeholder)", a reserved
        # EMPTY slot _react_surgery step 5 injects and explicitly SUPPRESSES under bundle_mode.
        # Ordering such an entry re-introduces the literal via the step-4 ORDER comparator (F2).
        declared = {
            (tmpl.report_kwargs or {}).get("category", "Other")
            for sel in active_set.renderer_selection
            for tmpl in sel.rule_spec_template
        }
        for c in active_set.category_order:
            if c in declared and c not in seen and c not in combined_cats:
                seen.append(c)
    return seen


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
    _eids = _distinct_child_categories(bundle_root)  # round-6: composed pages carry the figure's original category
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
