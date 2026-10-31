"""Combined-bundle regeneration-scoped Snakefile generator.

A combined bundle joins several completed experiment bundles into one
cross-experiment report, and that report should look and navigate like a
single-bundle report: the same chrome, sidebar and CSS, produced by Snakemake
``--report``. The single-analysis generator cannot be reused for it, because it is
coupled to cfg_analysis, cfg_system and status flags that a combined bundle does
not carry. This module is a dedicated sibling generator, arranged like
``static_snakefile_generator.py``: it reuses the module-level ``_emit_plot_rule`` /
``_emit_rule_all`` / ``RuleEmissionContext`` / ``RuleSpec`` primitives, so the
``report()`` emission is the same code the bundle and static generators run.

It emits two populations of rules: one per figure template of the ``combined``
reporting set (the cross-experiment renderers, all fixed, with no wildcards), and
one per composed per-experiment page, which stacks the primary child bundle's figure
(the TRITON-SWMM arm when the experiment has one) above its pure-TRITON counterpart
when one exists (``_harvest_per_experiment_rule_specs``).

Every figure and page is already on disk by the time the report is rendered:
``_combine._render_combined_report`` direct-renders the cross-experiment figures,
and the harvest writes the composed pages while it builds their rules. The plot-rule
shells therefore never run under the ``--touch`` + ``--report`` render path, the
same as the bundle-regeneration plot rules: the Snakefile this module writes exists to
give ``snakemake --report`` the ``report()`` annotations and the ``report:``
workflow-description directive, not to produce figures. The render itself is driven by
``render_combined_report_via_snakemake``, which stages the report artifacts, writes the
Snakefile, runs ``snakemake --touch`` and then ``snakemake --report`` per format, and
applies the post-render surgery. The static backend is fixed to "plotly" because a
combined bundle has no cfg_analysis to read it from, and every combined renderer's
output extension is ``.html`` under both backends, so the fixed value changes no
output path.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from hhemt.report_plot_ids import escape_label_braces, humanize_plot_id
from hhemt.report_renderers._reporting_sets import get_reporting_set
from hhemt.subprocess_utils import run_subprocess_with_tee
from hhemt.workflow import (
    RuleEmissionContext,
    RuleSpec,
    _emit_plot_rule,
    _emit_rule_all,
    _output_ext_for,
)

# Every combined-set renderer emits HTML unconditionally, and _OUTPUT_EXT_BY_RENDERER
# resolves .html for each of them under both backends. A combined bundle has no
# cfg_analysis to read static_backend from, so it is fixed here, and the fixed value
# changes no output path.
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
cross-experiment comparison. The sidebar holds four cross-experiment sections:

- **Combine Provenance & Compatibility**: the metadata-compatibility summary
  across the combined experiments.
- **Cross-Experiment Results**: the clean-vs-resume intercomparison of
  key-result summaries, as a table and as maps.
- **Cross-Experiment Errors and Warnings**: each experiment's validation
  checks, side by side.
- **Cross-Experiment Disk Utilization**: each experiment's disk use, side by
  side.

Below them, the combined experiments' own figures are grouped by kind, under
the section each figure has in a single-experiment report (for example
System Information); a figure that no section claims is listed under Other.
Inside those sections, each combined experiment's figures appear under entries
whose names start with that experiment's name, with that experiment's model
runs shown together on one page.

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
    bundle.snakefile_generator._load_source_paths). Each combined-set figure and its
    manifest are written by the direct-invoke render (_combine._render_combined_report)
    before this generator runs; an absent sidecar yields no source paths."""
    import json

    if not sidecar_path.exists():
        return []
    data = json.loads(sidecar_path.read_text())
    return list(data.get("source_paths_relative", []))


def _harvest_combined_rule_specs(bundle_root: Path) -> tuple[RuleSpec, ...]:
    """One RuleSpec per combined-set figure template; every such template is fixed, with no wildcards.

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


#: Model tokens stripped to collapse two single-model bundles of one experiment onto one base
#: experiment, whose figures the harvest then pairs on one composed page. Matching is by suffix
#: and neither token is a suffix of the other, so the order of this tuple does not change which
#: token an eid matches.
_MODEL_TOKENS: tuple[str, ...] = ("_tritonswmm", "_triton")


def _base_experiment(eid: str) -> str:
    """The experiment id with its trailing model token stripped, so `synth_cc_clean_triton`
    and `synth_cc_clean_tritonswmm` both collapse to `synth_cc_clean`, the key under which the
    harvest pairs the two model arms of one experiment on one composed page."""
    for _tok in _MODEL_TOKENS:
        if eid.endswith(_tok):
            return eid[: -len(_tok)]
    return eid


def _model_of(eid: str) -> str:
    """The model arm a child bundle carries ('triton' / 'tritonswmm'), from its eid suffix;
    '' when the eid carries no model token. The harvest uses it twice: to key each base
    experiment's children by arm, and to pass the primary child's arm to
    _compose_model_pair_page as ``primary_model``, the top section's model -- "tritonswmm",
    "triton" for a TRITON-only base, or "" when the primary child's eid carries no model token."""
    for _tok in _MODEL_TOKENS:
        if eid.endswith(_tok):
            return _tok[1:]
    return ""


#: The stacked both-models CSS, INLINED into each composed page's <head> because the report.css
#: --report-stylesheet reaches only the report CHROME, never a report()'d figure's inner (data-URI
#: iframe) document, so a stylesheet rule would leave the composed page unstyled. The
#: .model-arm-frame rule exists because each arm is a full self-contained plotly document embedded
#: via <iframe srcdoc>, and an iframe needs an explicit height or it renders at the browser default.
#: var(--uva-*) references resolve from the :root{} block _extract_root_css_vars pulls out of the
#: staged report.css, so brand colors come from the brand_theme config like the chrome's. The
#: fallbacks are NEUTRAL non-brand values so no UVA hex is hardcoded here.
_MODEL_STACK_CSS = """
.model-pair-page{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:0;padding:1rem;}
.model-stack{display:flex;flex-direction:column;margin:1.5rem 0;}
.model-stack>.model-section{width:100%;padding:1rem 0;}
.model-stack>.model-section[data-model="tritonswmm"]{order:0;}
.model-stack>.model-section[data-model="triton"]{order:1;
border-top:2px solid var(--uva-medium-gray,#cccccc);margin-top:.5rem;}
.model-stack>.model-section[data-model]::before{display:block;font-weight:bold;color:var(--uva-blue,#1a1a1a);margin-bottom:.5rem;font-size:1.05rem;}
.model-stack>.model-section[data-model="tritonswmm"]::before{content:"TRITON-SWMM";}
.model-stack>.model-section[data-model="triton"]::before{content:"TRITON";}
.model-stack>.model-section img,.model-stack>.model-section iframe,
.model-stack>.model-section svg,.model-stack>.model-section table{max-width:100%;}
.model-arm-frame{width:100%;height:900px;border:0;overflow:auto;}
.model-stack>.model-section[data-model-absent],
.model-stack>.model-section.model-absent{color:var(--uva-text-gray,#6b7280);
background:var(--uva-light-gray,#f1f3f5);
border:1px dashed var(--uva-medium-gray,#cccccc);
border-radius:4px;padding:.75rem 1rem;font-style:italic;}
"""

#: Why the arm frame is re-measured. It used to be sized ONCE, from its own `onload` handler, as
#: `scrollHeight + 24`. That single measurement is taken when the inner document's `load`
#: event fires -- which is BEFORE any content that builds asynchronously after load exists in
#: the DOM. The reproduction-guide tables are exactly that: three Tabulator instances that
#: build from a CDN script and populate their mount divs after `load`, so the frame locked to
#: its pre-table height and everything below that point was clipped. This is COMBINED-ONLY
#: because only the composed page embeds an arm in an <iframe>; a per-arm report embeds the
#: figure document directly and has no frame to mis-size.
#:
#: The fix re-measures on ANY late content growth rather than hooking Tabulator specifically,
#: so the next async widget added to a figure page inherits the fix instead of re-opening the
#: bug. A ResizeObserver on the inner <html> and <body> is the generic instrument: it fires on
#: any layout-size change regardless of what caused it.
#:
#: Three details are load-bearing.
#: (1) The immediate `fit()` reproduces the previous one-shot behavior exactly, so a browser
#:     without ResizeObserver is no worse off than before -- this strictly adds a later
#:     correction, it never removes the original measurement.
#: (2) The 2px epsilon and the pass cap bound a REAL feedback loop, not a hypothetical one:
#:     growing the frame re-resolves any `vh`-sized element inside it, which changes
#:     scrollHeight, which fires the observer again. Simulated against a fake DOM whose
#:     scrollHeight depends on frame height: the bounded case (the `min(70vh, 640px)` table
#:     height report_renderers/metadata.py sets)
#:     settles in 2 writes, and an UNBOUNDED bare `70vh` settles in 11 -- because a `vh`
#:     fraction below 1 is a contraction, so the series converges on its own and the epsilon
#:     stops it. The pass cap therefore never engages for either, and is a backstop only for
#:     a pathological content function with gain >= 1. An earlier draft of this comment said
#:     the companion height bound is "what makes that loop convergent"; that overstated it --
#:     the bound makes convergence FASTER and the resting height LOWER, it is not what makes
#:     the loop terminate.
#: (3) The whole body stays inside try/catch, preserving the previous silent-fallback
#:     semantics if same-origin DOM access is ever refused.
_MODEL_STACK_JS = """
(function () {
  window.__fitArmFrame = function (f) {
    try {
      var w = f.contentWindow, d = w.document, r = d.documentElement;
      var passes = 0;
      var fit = function () {
        if (passes > 40) { return; }
        var h = Math.max(r.scrollHeight, d.body ? d.body.scrollHeight : 0) + 24;
        if (Math.abs(h - f.clientHeight) > 2) { passes++; f.style.height = h + 'px'; }
      };
      fit();
      var RO = w.ResizeObserver || window.ResizeObserver;
      if (RO) {
        var ro = new RO(fit);
        ro.observe(r);
        if (d.body) { ro.observe(d.body); }
      }
    } catch (e) {}
  };
})();
"""


def _extract_root_css_vars(bundle_root: Path) -> str:
    """The staged report.css :root{...} custom-property block, so a composed figure page's inlined
    _MODEL_STACK_CSS resolves its var(--uva-*) references from the SAME brand_theme config the report
    chrome uses. Graceful-absent: '' when report.css is not yet staged (e.g. when the byte-determinism
    test tests/test_combine_bundle.py::test_combined_emit_deterministic stubs out the render), where
    _MODEL_STACK_CSS's neutral non-brand fallbacks apply."""
    import re as _re

    css = bundle_root / "report" / "report.css"
    if not css.exists():
        return ""
    m = _re.search(r":root\s*\{[^}]*\}", css.read_text(encoding="utf-8", errors="replace"))
    return m.group(0) if m else ""


def _compose_model_pair_page(
    base_eid: str,
    plot_id: str,
    ts_html: str,
    tri_html: str | None,
    root_css_vars: str = "",
    member_labels: dict[str, str] | None = None,
    event_labels: dict[str, str] | None = None,
    *,
    primary_model: str = "tritonswmm",
) -> str:
    """Compose ONE self-contained page STACKING the primary child's figure (top) and its pure-TRITON
    counterpart (below, when present) as a .model-stack / .model-section[data-model] column styled by
    _MODEL_STACK_CSS. The top section's data-model is the keyword-only ``primary_model`` (default
    "tritonswmm"); the harvest passes the primary child's arm, which is "triton" for a TRITON-only
    base and "" for a base whose children carry no model token. Each arm is a full
    self-contained plotly document embedded via an isolated <iframe srcdoc> (EMPIRICALLY VERIFIED:
    snakemake --report base64-passes the composed page verbatim and the report app renders it in an
    <iframe src=data-uri>, so nested srcdoc iframes are proper opaque-origin docs — no id collision,
    no double-plotly-bundle, no figure-surgery). _MODEL_STACK_CSS is INLINED here because report.css
    (--report-stylesheet) styles the report chrome only and never reaches a figure's inner document.
    tri_html is None when there is no pure-TRITON counterpart (a coupled-only SWMM-specific figure, or
    a base with no TRITON-SWMM child to pair against) -> only the top section renders.
    The inner iframes carry an onload auto-grow (srcdoc inherits the composed page's opaque origin, so
    the DOM read succeeds in practice) with a try/catch fallback to the CSS 900px height."""
    import html as _html

    def _section(model: str, inner: str) -> str:
        return (
            f'<div class="model-section" data-model="{_html.escape(model, quote=True)}">'
            f'<iframe class="model-arm-frame" loading="lazy" '
            f'title="{_html.escape(base_eid)} {_html.escape(plot_id)} ({_html.escape(model or "model not recorded")})" '
            'onload="__fitArmFrame(this)" '
            f'srcdoc="{_html.escape(inner, quote=True)}"></iframe></div>'
        )

    body = _section(primary_model, ts_html)
    if tri_html is not None:
        body += _section("triton", tri_html)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{_html.escape(base_eid)} — "
        f"{_html.escape(humanize_plot_id(plot_id, member_labels, event_labels))}</title>"
        f"<style>{root_css_vars}\n{_MODEL_STACK_CSS}</style>"
        # Defined ONCE per composed page rather than inlined into each iframe's onload
        # attribute: one definition for both arms, and no attribute-escaping hazard around
        # the quotes and braces the fitter needs.
        f"<script>{_MODEL_STACK_JS}</script></head>"
        '<body class="model-pair-page">'
        # No visible heading. The `<h2 class="model-pair-heading">` that stood here carried the SAME
        # `{base_eid} — {humanize_plot_id(...)}` string as the <title> above, and the report
        # chrome ALREADY identifies the result: its `figure` label column carries
        # humanize_plot_id(...) (set in the harvest's labels below), and its subcategory
        # names the experiment. So the
        # reader saw `synth_cc_clean — Run metadata` twice in a row, which is exactly the
        # duplication the user reported: 'there are redundant headers. ## synth_cc_clean —
        # Run metadata appears twice in a row, and then TRITON-SWMM appears twice as well'.
        #
        # Measured on the delivered combined report: `grep -o 'Run metadata' | wc -l` -> 4,
        # two per experiment across synth_cc_clean and synth_cc_resume.
        #
        # The <title> is RETAINED and is not the duplicate: it names the browser tab and is
        # the only heading a reader gets when the composed page is opened standalone, where
        # no report chrome supplies the name. Dropping the visible <h2> costs nothing in the
        # embedded case (the chrome states it) and leaves the standalone case titled.
        #
        # This is the same defect class as the 'exact code that produced this' SHA caption
        # repaired in 7eeecca and the page-title question settled in f1ccc9f: a string that
        # is correct at its own emission site and redundant-or-false once the surrounding
        # context also supplies it.
        f'<div class="model-stack" data-plot="{_html.escape(plot_id, quote=True)}" '
        f'data-config="{_html.escape(base_eid, quote=True)}">{body}</div>'
        "</body></html>"
    )


def _harvest_per_experiment_rule_specs(bundle_root: Path) -> tuple[RuleSpec, ...]:
    """ONE composed report() page per (base-experiment, plot_id): the primary child's figure on top
    (the TRITON-SWMM arm when the base has one), its pure-TRITON counterpart stacked BELOW on the SAME
    page, so the two model arms are compared without leaving the page. The composed page carries the
    figure's ORIGINAL category, so pairing adds no base-experiment CATEGORY; the base experiment
    appears one level down instead, composed into the subcategory label (`{base} — {child
    subcategory}`, or `{base}` alone) in the report_kwargs below. The pure-TRITON arm's figures
    already exist SWMM-free in the *_triton child bundle, so co-location needs no figure-surgery.
    Figures with no pure-TRITON counterpart (SWMM-specific ones) render the top section alone.
    Composed pages are written under paired_figures/, OUTSIDE the crate-hashed child_crates/,
    so they leave the combined crate's bytes unchanged; the byte-determinism test
    (tests/test_combine_bundle.py::test_combined_emit_deterministic) stubs out
    _render_combined_report, under which this harvest runs. _MODEL_STACK_CSS is inlined per page;
    its var(--uva-*) resolves from the staged report.css :root{} block, i.e. the brand_theme config."""
    # One member_id -> derived-config-label map for every label this harvest emits --
    # the paired-page <title> and the sidebar `labels.figure` facet. Merged across
    # child crates because the combined bundle has no top-level scenario_status.csv.
    # Built ONCE here rather than per figure: the harvest emits one page per
    # (base-experiment, plot_id) pair and re-reading per page would re-parse the same
    # CSVs for every figure.
    from ..report_plot_ids import event_sidebar_labels_from_status, member_labels_from_status

    _member_labels: dict[str, str] = {}
    _event_labels: dict[str, str] = {}
    _crates_dir = bundle_root / "child_crates"
    if _crates_dir.exists():
        for _c in sorted(p for p in _crates_dir.iterdir() if p.is_dir()):
            _member_labels.update(member_labels_from_status(_c))
            _event_labels.update(event_sidebar_labels_from_status(_c))
    import json as _json
    import re as _re

    import yaml as _yaml

    from hhemt.bundle.snakefile_generator import (
        _expand_wildcards_to_existing_files,
        _load_source_paths,
    )
    from hhemt.config.eda import _RETIRED_EDA_FIGURE_STEMS
    from hhemt.config.report import resolve_reporting_set_names
    from hhemt.eda._sensitivity_figures import _PENDING_EDA_FIGURE_STEMS
    from hhemt.exceptions import ConfigurationError
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    def _figures_of(eid: str) -> list[tuple]:
        """(plot_id, category, subcategory, rel, html_path, sidecar) for one child bundle,
        template-claimed figures first then defensively-globbed leftovers."""
        child = bundle_root / "child_crates" / eid
        cfg = _yaml.safe_load((child / "cfg_analysis.yaml").read_text())
        # resolve_reporting_set_names carries both the absent-selection rule (no
        # selection resolves to "default") AND the registered-name check, and it
        # accepts the LIST this raw dict now holds -- a bundle's cfg_analysis.yaml is
        # dumped from the validated model, so it carries a list even when the source
        # config was written with a bare string. Hand-inlining the absent-selection
        # rule here drops the registered-name check, which is the divergence the shared
        # helper exists to prevent.
        _names = resolve_reporting_set_names(
            (cfg.get("report") or {}).get("reporting_set"),
            is_sensitivity=bool(cfg.get("toggle_sensitivity_analysis")),
        )
        if len(_names) > 1:
            # RAISE rather than harvest the first: this function globs a child's figures
            # from ONE set's output-path templates, so a multi-member child would be
            # harvested partially and the combined report would look complete while
            # omitting a member set's figures. Raising keeps a partial harvest loud.
            raise ConfigurationError(
                field="report.reporting_set",
                message=(
                    f"child experiment {eid!r} was run with a composed reporting set "
                    f"({', '.join(_names)}), which this combine path cannot read: it "
                    "reads one set's templates. Re-run combine once it composes, or "
                    "combine children that each name a single set."
                ),
                config_path=child / "cfg_analysis.yaml",
            )
        active_set = get_reporting_set(_names[0])
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
                    plot_id = (
                        _json.loads(sidecar.read_text()).get("plot_id", fpath.stem) if sidecar.exists() else fpath.stem
                    )
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
            if fpath.stem in _PENDING_EDA_FIGURE_STEMS:
                # A NOT-YET-DESIGNED placeholder figure, written by
                # eda/_sensitivity_figures.py::_pending_figure: a ~10 MB inlined-Plotly page carrying a title and no
                # panels. Distinct from the retired case above -- these stems are LIVE
                # opt-ins via eda.enabled_plots and must NOT be added to _EDA_DROPS, which
                # would strip them at config load. Suppress at the COMBINE guard only.
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
        primary_eid = (
            ts_eid or tri_eid or next((arms[k] for k in sorted(arms) if k not in ("tritonswmm", "triton")), None)
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
        for plot_id, category, subcategory, _rel, fpath, sidecar in _figures_of(primary_eid):
            ts_html = fpath.read_text(encoding="utf-8", errors="replace")
            cf = counterpart.get(plot_id)
            tri_html = cf[4].read_text(encoding="utf-8", errors="replace") if cf is not None else None
            page_html = _compose_model_pair_page(
                base,
                plot_id,
                ts_html,
                tri_html,
                root_css_vars,
                _member_labels,
                _event_labels,
                primary_model=_model_of(primary_eid),
            )
            (paired_dir / _re.sub(r"[^A-Za-z0-9_.]", "_", base)).mkdir(parents=True, exist_ok=True)
            # Per-base SUBDIRECTORY, so the page's filename STEM is exactly the plot id: the
            # Snakemake figure-card name derives from that stem, and humanize_plot_id looks the
            # stem's leading "__"-segment up in report_plot_ids._RENDERER_KIND_LABELS, so the
            # curated label is used only when that segment IS the renderer kind. A flat
            # "{base}__{plot_id}" stem makes the humanizer read the base eid as the renderer
            # kind and silently drops the curated label. The base is not lost: it is the
            # directory here and report_kwargs["subcategory"] below.
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
                    # any keyed renderer; shell inert (page pre-composed)
                    renderer_module="cross_experiment_compatibility",
                    input_flags=(),
                    output_path_template=page_rel,
                    source_paths=tuple(srcs),
                    wildcards=(),
                    extra_cli_flags=(),
                    extra_params=(("experiment_id", repr(base)),),
                    report_kwargs={
                        "caption": "report/captions/_harvested.rst",
                        "category": category,  # the figure's ORIGINAL category (NO base-experiment category)
                        # Requirement: the combined report preserves each child report's own
                        # navigation structure as nearly as the engine allows, including the
                        # child's subcategory grouping, which an earlier version lost.
                        #
                        # The child's OWN subcategory is composed with the experiment id
                        # rather than discarded. `_figures_of` already harvests it (it is the third
                        # tuple element, destructured as `subcategory` at the top of this loop) and
                        # this line previously dropped it on the floor, which is what destroyed the
                        # child's `Benchmarking` / `Compute-config EDA` / `Byte-for-byte comparison`
                        # grouping in the combined report.
                        #
                        # Composed rather than nested because the engine offers exactly TWO
                        # navigation levels -- `report(value, caption, category, subcategory,
                        # labels, patterns, htmlindex)`, verified by introspection against the
                        # installed snakemake 9.15.0 -- so the user's three-level
                        # `Key Results -> synth_cc_resume -> Benchmarking` is unreachable. This
                        # renders it as `Key Results -> [synth_cc_resume — Benchmarking]`: both
                        # tokens, in his order, one click instead of two.
                        #
                        # The `base`-only fallback is load-bearing, not defensive: several figures
                        # carry NO child subcategory (`_TMPL_SYSTEM_OVERVIEW` and
                        # `_TMPL_DISK_UTILIZATION` declare `category` and no `subcategory`), and an
                        # unconditional compose would give them a trailing-em-dash label.
                        "subcategory": f"{base} — {subcategory}" if subcategory else base,
                        # RETIRED CLAIM, kept because naming a retirement beats deleting it.
                        # A `figure` facet IS emitted here -- see `"labels"` below. What stood
                        # here said it must not be, and that claim is FALSE against the
                        # installed engine (snakemake 9.15.0): the result's NAME is rendered as
                        # visible text nowhere, its only consumer being
                        # `result_view_button.js`'s `download:` filename attribute, so no facet
                        # can restate it. The full refutation and its measurements are at the
                        # `figure` facet itself.
                        #
                        # The retired text, verbatim:
                        #   "NO "figure" facet here. On a harvested page the report result's own
                        #    NAME is already humanize_plot_id(plot_id), so a facet carrying the
                        #    same string restates the name rather than filtering by anything, and
                        #    renders the label twice. The per-primary rules can carry a "figure"
                        #    facet because their names and facets are independently authored;
                        #    this path derives both from one function."
                        #
                        # ONE defect behind two navigation symptoms. The former
                        # `experiment` facet was DEGENERATE at this level and that is what
                        # dead-ended the navigation: Snakemake builds a facet table whose
                        # COLUMNS are the label keys, and `experiment` restated the
                        # `subcategory` the reader had just clicked, so it could not
                        # partition. A facet that cannot partition renders ONE row where a
                        # figure list belongs -- which is why Key Results dead-ended before the
                        # benchmarking figures and Per Simulation Results inserted a redundant
                        # experiment level. The replacement facets are read off the plot id,
                        # whose grammar (report_plot_ids.canonical_plot_id) already encodes
                        # them: per-sim figures vary in the `member.` and `evt.` segments,
                        # benchmarking figures vary in the descriptor segment.
                        #
                        # TWO claims that stood here are RETIRED because measurement
                        # falsified them; they are named rather than deleted so the next
                        # reader does not re-derive them.
                        #
                        # (1) `models` was retired alongside `experiment` as "a pure function
                        # of `base`". That holds for which arms a BASE has children for; it
                        # FAILS for which arms a given FIGURE rendered, because `tri_html` is
                        # None whenever the pure-TRITON child has no counterpart at that plot
                        # id. Measured on the delivered bundle: `conduit_flow` occurs 0 times
                        # in the pure-TRITON child and 28 times in the coupled one, so inside
                        # the single subcategory `synth_cc_clean` 28 composed pages carry one
                        # arm and 28+ carry two. `models` partitions there, and it is the axis
                        # this campaign exists to expose -- so it is restored, supplied by the
                        # call site from the arm pair already in hand.
                        #
                        # (2) "The key set is FIXED (never conditional) because Snakemake does
                        # not validate key consistency across results." The premise is true and
                        # the conclusion does not follow: the per-arm reports already ship
                        # THREE distinct key sets in ONE report (8 results with {figure}, 1
                        # with {independent_var, figure}, 56 with {figure, member_id, event_id})
                        # and render correctly. The fixed set was what emitted an all-blank
                        # row -- measured 18 of them, and NO result populated all three.
                        # Absent facets are now OMITTED rather than emitted empty.
                        "labels": _json.dumps(
                            # `figure` FIRST and unconditional. The comment above said a
                            # `figure` facet would "restate the name rather than filtering by
                            # anything, and render the label twice". Measured against the
                            # INSTALLED engine (snakemake 9.15.0), that premise is false: the
                            # result's `name` is rendered as visible text NOWHERE. Columns are
                            # the union of LABEL keys (`abstract_results.js::getLabels`), and
                            # `name`'s only consumer in the bundled front end is
                            # `result_view_button.js`'s `download:` filename attribute. So
                            # this facet restates nothing visible -- it is the only identifying
                            # column the table can carry.
                            #
                            # It is also what stops figures disappearing. `getData()` keys each
                            # ROW on the joined label tuple and then does
                            # `entries.get(key).set(arrayKey(entryToggleLabels), path)`, so two
                            # figures sharing a tuple overwrite at the same key and the earlier
                            # one becomes unreachable -- silently, because the data layer still
                            # carries it (`render_results` keys on file path, so record counts
                            # look correct). Measured on the delivered combined report: 4
                            # figures present and unreachable, including
                            # "Cross-hardware raw byte identity over time" under Key Results.
                            #
                            # FIRST is load-bearing, not cosmetic. `getLabels()` orders columns
                            # by first appearance; `entryLabelValues.sort()` makes key 0 the
                            # primary sort; and `getData()` considers only `labels.slice(1)`
                            # for promotion to a toggle control, so a first key can never be
                            # silently converted from a column into a toggle.
                            {
                                # escape_label_braces because THIS value reaches a Snakemake
                                # `report(labels=...)` entry, where `expand_labels` re-expands
                                # it through `apply_wildcards` -- so a `{` in any user-authored
                                # member/event label aborts the render. `report_label_value` escapes
                                # for the per-primary rules; `humanize_plot_id` does not, because
                                # its other two consumers (the composed page `<title>` here and
                                # the React `name` field in `_react_surgery`) are never
                                # re-expanded and would show doubled braces to the reader. The
                                # escape therefore belongs at THIS call, not inside the humanizer.
                                "figure": escape_label_braces(humanize_plot_id(plot_id, _member_labels, _event_labels)),
                                **_plot_id_facets(
                                    plot_id,
                                    models=(
                                        "TRITON-SWMM + TRITON"
                                        if tri_html is not None
                                        else (
                                            "TRITON-SWMM only"
                                            if ts_eid
                                            else ("TRITON only" if tri_eid else "model not recorded")
                                        )
                                    ),
                                ),
                            }
                        ),
                    },
                    resources_yaml="mem_mb=1000, time_min=5",
                    log_path_template=f"_logs/plots/{rn}.log",
                )
            )
    return tuple(specs)


def _plot_id_facets(plot_id: str, *, models: str = "") -> dict[str, str]:
    """Split a canonical plot id into the facets that PARTITION a harvested figure set.

    Grammar (`report_plot_ids.py`): ``{renderer_kind}[__{descriptor}][__member.{member_id}]
    [__evt.{event_id}]``, "__" between segments and "." within one. This reads the two
    prefixed segments and files every other segment under a single ``variant`` facet; when an
    id carries more than one such segment, each overwrites the last and only the final one is
    kept.

    Why the parse lives HERE and not in `report_plot_ids.py`, which owns the grammar:
    that module is `layout_relevant` and its `non_breaking_allowlist` entry is scoped by
    `layout_signature` -- a content hash of the file as it stands. ANY edit to it, additive
    or not, changes the hash and re-fires CI Check B, which then demands a LAYOUT_VERSION
    bump plus a migration plus two golden fixtures for a change that alters no on-disk
    figure filename. `canonical_plot_id` remains the sole MINTER and this is a read-only
    consumer. A grammar change is therefore NOT detected here: a new prefixed segment is
    filed under ``variant`` like a descriptor, and only an empty segment value trips the
    guard below.

    Returns ONLY the facets that carry a value, so the key set is heterogeneous across
    results by design. Snakemake performs no key-consistency check on `labels`
    (`report/html_reporter/data/results.py` passes `res.labels` straight to the React app),
    and heterogeneity is not unvalidated territory: the per-arm reports already ship three
    distinct key sets in one report and render correctly. Emitting an absent facet as an
    empty string is what produced the blank index columns, so an absent facet is now
    omitted, and the guard below makes an all-empty return unrepresentable.
    """
    stem = plot_id
    for _ext in (".html", ".png", ".svg"):
        if stem.endswith(_ext):
            stem = stem[: -len(_ext)]
            break
    segments = stem.split("__")[1:]
    facets: dict[str, str] = {}
    for seg in segments:
        if seg.startswith("member."):
            facets["member"] = seg[len("member.") :]
        elif seg.startswith("evt."):
            facets["event"] = seg[4:]
        else:
            facets["variant"] = seg.replace(".", " ")
    if models:
        facets["models"] = models
    if not facets or any(v == "" for v in facets.values()):
        raise ValueError(
            f"combined report: plot id {plot_id!r} produced facets {facets!r}; "
            "every emitted facet must carry a non-empty value, and at least one "
            "facet must be present, or the report renders a blank index column."
        )
    return facets


def _distinct_child_categories(bundle_root: Path) -> list[str]:
    """Ordered distinct figure categories the composed per-experiment pages use (each child figure's
    ORIGINAL report category), excluding the combined set's own cross-experiment categories. These
    are what the composed pages carry, so they are what the combined sidebar order must rank."""
    import yaml as _yaml

    from hhemt.config.report import resolve_reporting_set_names
    from hhemt.exceptions import ConfigurationError
    from hhemt.report_renderers._reporting_sets import get_reporting_set

    combined_cats = set(get_reporting_set("combined").category_order)
    seen: list[str] = []
    for eid in _child_experiment_ids(bundle_root):
        cfg = _yaml.safe_load((bundle_root / "child_crates" / eid / "cfg_analysis.yaml").read_text())
        # See the twin in _figures_of. Same helper, same multi-member refusal --
        # this enumeration reads ONE set's declared categories, so a partial read would
        # silently drop a member set's sidebar categories from the combined report.
        _names = resolve_reporting_set_names(
            (cfg.get("report") or {}).get("reporting_set"),
            is_sensitivity=bool(cfg.get("toggle_sensitivity_analysis")),
        )
        if len(_names) > 1:
            raise ConfigurationError(
                field="report.reporting_set",
                message=(
                    f"child experiment {eid!r} was run with a composed reporting set "
                    f"({', '.join(_names)}), which this category enumeration cannot read: it "
                    "reads one set's declared categories. Re-run combine once the enumeration "
                    "composes, or combine children that each name a single set."
                ),
                config_path=bundle_root / "child_crates" / eid / "cfg_analysis.yaml",
            )
        active_set = get_reporting_set(_names[0])
        # Only categories that one of the set's rule-spec TEMPLATES declares can reach a
        # composed page: _figures_of takes each page's category from the template that
        # claimed its figure, and files an unclaimed leftover under "Other". category_order
        # is the set's sidebar ORDER, not an inventory of its figures, and a set may order a
        # category none of its templates declares ("Key Results" in the default set is one).
        # Deliberately NOT the whole category_order: an ordered name with no page adds no
        # sidebar entry, because the comparator _react_surgery installs only ranks
        # categories that exist, but the name is still written into that comparator's ORDER
        # literal in the rendered HTML. The filter keeps the combined order, and that
        # literal, to categories a composed page actually carries.
        declared = {
            (tmpl.report_kwargs or {}).get("category", "Other")
            for sel in active_set.renderer_selection
            for tmpl in sel.rule_spec_template
        }
        for c in active_set.category_order:
            if c in declared and c not in seen and c not in combined_cats:
                seen.append(c)
    return seen


def generate_combined_snakefile(bundle_root: Path) -> tuple[str, tuple[Path, ...]]:
    """Generate the combined regeneration-scoped Snakefile body as a string.

    Returns (body, expected_figures) — the absolute paths of every figure the
    emitted `rule all` requires, computed from the SAME traversal that wrote it so
    the two cannot disagree about which figures must exist.
    """
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

    def _plot_rel(spec: RuleSpec) -> str:
        """The rule's output path, extension resolved, UNQUOTED.

        Resolved once and quoted at the point of use so the Snakefile literal and
        the Path handed to the caller's preflight are provably the same string.
        Deriving the Path by stripping quotes back off the literal would be a
        second derivation and could drift.
        """
        ext = _output_ext_for(ctx.static_backend, spec.renderer_module)
        return spec.output_path_template.replace("__OUTPUT_EXT__", ext)

    plot_rel_paths = tuple(_plot_rel(spec) for spec in rule_specs)
    plot_output_paths = tuple('"' + rel + '"' for rel in plot_rel_paths)
    # NO render_report rule: the report is produced by the direct `snakemake --report`
    # call (render_combined_report_via_snakemake), so rule all lists only the figures.
    rule_all_block = _emit_rule_all(
        status_flags=(),
        plot_output_paths=plot_output_paths,
        render_report_targets=(),
        ctx=ctx,
    )
    plot_rule_blocks = "".join(_emit_plot_rule(spec, ctx) for spec in rule_specs)
    body = "\n".join([preamble, report_directive, rule_all_block, plot_rule_blocks])
    return body, tuple(bundle_root / rel for rel in plot_rel_paths)


def write_combined_snakefile(bundle_root: Path) -> tuple[Path, tuple[Path, ...]]:
    """Overwrite {bundle_root}/Snakefile; return (snakefile_path, expected_figures).

    The expected-figure set comes from the SAME traversal that produced the
    Snakefile, so the preflight in render_combined_report_via_snakemake and the
    rule-all enumeration cannot disagree about which figures must exist.
    """
    bundle_root = Path(bundle_root).resolve()
    out = bundle_root / "Snakefile"
    body, expected = generate_combined_snakefile(bundle_root)
    out.write_text(body + "\n")
    return out, expected


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

    Mirrors Bundle.regenerate_report's touch->report->surgery sequence. The
    cross-experiment figures already exist on disk (direct-rendered by
    _render_combined_report before this call); the paired per-experiment pages are
    written during this call, by the harvest that write_combined_snakefile triggers.
    --touch then marks all of them up-to-date so --report does not try to re-run their
    (inert) plot rules.
    """
    bundle_root = Path(bundle_root).resolve()
    stage_combined_report_artifacts(bundle_root)
    _, _expected_figures = write_combined_snakefile(bundle_root)

    # Preflight: a combined bundle is portability-scrubbed and carries NO root
    # cfg_system.yaml / cfg_analysis.yaml — only a single-analysis bundle does. The
    # normal path never needs them (figures are direct-rendered by
    # _render_combined_report and --touch marks them current, so no plot-rule shell
    # runs), but any invocation that DOES re-execute a rule dies inside _cli at
    # TRITONSWMM_system(cfg_system.yaml) with a bare FileNotFoundError naming a path
    # the user never created. Convert that silent cliff into an instruction.
    #
    # The expected set is threaded from write_combined_snakefile rather than
    # recomputed: _harvest_per_experiment_rule_specs COMPOSES AND WRITES the paired
    # pages, so re-invoking it here would rewrite the paired_figures/ tree as a side
    # effect of a read-only check AND raise from its own read_text() on exactly the
    # missing-figure case this guard exists to report.
    _unstaged = [p for p in _expected_figures if not p.exists()]
    if _unstaged and not (bundle_root / "cfg_system.yaml").exists():
        raise FileNotFoundError(
            f"Combined bundle {bundle_root} is missing {len(_unstaged)} pre-rendered "
            f"figure(s) (first: {_unstaged[0].name}) and carries no cfg_system.yaml, so "
            "the plot rules cannot be re-executed here. Combined bundles are rendered "
            "in-process, not by Snakemake: use CombinedBundle.regenerate_report() "
            "instead of `snakemake --forcerun` / a manual rule invocation."
        )

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

    # Sidebar order: the combined set's fixed cross-experiment categories, then the child
    # figure categories the composed pages carry, so every harvested category is ranked
    # (an unranked category sorts last, alphabetically, under _react_surgery's comparator).
    # Derived from child_crates/ on disk so regenerate_report() with no re-merge recomputes
    # it identically.
    _fixed = list(get_reporting_set("combined").category_order)  # the fixed cross-experiment categories
    _eids = _distinct_child_categories(bundle_root)  # child FIGURE categories, not experiment ids
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
        # The combined bundle has NO top-level scenario_status.csv -- its per-sim
        # data lives under child_crates/. Reading the root would yield {} and leave the
        # combined report's card names unresolved while the per-arm reports resolve,
        # so two views of the same figure would name it differently. Merge across
        # the children instead. member_ids are unique per child by construction, and a
        # collision would mean two children ran the same member, in which case
        # either label is correct.
        from ..report_plot_ids import event_sidebar_labels_from_status, member_labels_from_status

        _member_labels: dict[str, str] = {}
        _event_labels: dict[str, str] = {}
        _crates = bundle_root / "child_crates"
        if _crates.exists():
            for _child in sorted(p for p in _crates.iterdir() if p.is_dir()):
                _member_labels.update(member_labels_from_status(_child))
                _event_labels.update(event_sidebar_labels_from_status(_child))
        if fmt == "html":
            output_path.write_text(
                apply_post_process_surgery(
                    output_path.read_text(),
                    bundle_mode=True,
                    navbar_text=_COMBINED_NAVBAR_TEXT,
                    category_order=category_order,
                    member_labels=_member_labels,
                    event_labels=_event_labels,
                )
            )
        else:
            apply_post_process_surgery_to_zip(
                output_path,
                bundle_mode=True,
                navbar_text=_COMBINED_NAVBAR_TEXT,
                category_order=category_order,
                member_labels=_member_labels,
                event_labels=_event_labels,
            )
