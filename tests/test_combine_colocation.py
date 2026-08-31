"""F1 within-page co-location (round-6): both-models stacked on ONE composed page.

For any combined-report page carrying a TRITON-SWMM figure, the pure-TRITON counterpart is
stacked BELOW it on the SAME page (data-viz R5-1 `.model-stack` layout, inlined into the
composed page `<head>` because the report `--stylesheet` reaches only the report chrome).
`_compose_model_pair_page` builds the page; `_harvest_per_experiment_rule_specs` emits one
composed page per (base-experiment, plot_id), carrying the figure's ORIGINAL category.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath

from hhemt.bundle.combined_snakefile_generator import (
    _base_experiment,
    _compose_model_pair_page,
    _distinct_child_categories,
    _extract_root_css_vars,
    _harvest_per_experiment_rule_specs,
)
from hhemt.report_plot_ids import humanize_plot_id
from hhemt.report_renderers._react_surgery import apply_post_process_surgery
from hhemt.report_renderers._reporting_sets import get_reporting_set

_TS_HTML = "<html><head></head><body><div id='plotly-ts'>TRITON-SWMM figure</div></body></html>"
_TRI_HTML = "<html><head></head><body><div id='plotly-tri'>pure-TRITON figure</div></body></html>"


def test_compose_page_stacks_both_models() -> None:
    page = _compose_model_pair_page("synth_cc_clean", "config_diff_maps", _TS_HTML, _TRI_HTML)
    # the R5-1 stacked wrapper + both arms as data-model sections, tritonswmm ABOVE triton.
    # Anchor on the section <div> element, not a bare data-model= substring (the inlined <head>
    # CSS carries [data-model="triton"] selectors on every page).
    assert 'class="model-stack"' in page
    i_ts = page.index('<div class="model-section" data-model="tritonswmm"')
    i_tri = page.index('<div class="model-section" data-model="triton"')
    assert i_ts < i_tri, "TRITON-SWMM section must be emitted before the pure-TRITON section"
    # both arms embedded as isolated srcdoc iframes (empirically byte-faithful through the report app)
    assert page.count("srcdoc=") == 2
    assert 'class="model-arm-frame"' in page
    # the data-plot / data-config attributes honor R5-1's contract
    assert 'data-plot="config_diff_maps"' in page
    assert 'data-config="synth_cc_clean"' in page


def test_compose_page_inlines_model_stack_css() -> None:
    # R5-1's .model-stack CSS is INLINED into the composed page <head> (FQ2: report.css reaches
    # only chrome), including the reconciliation-delta .model-arm-frame height rule.
    page = _compose_model_pair_page("myexp", "peak_flood_depth", _TS_HTML, _TRI_HTML)
    head = page[: page.index("</head>")]
    assert ".model-stack" in head
    assert ".model-arm-frame" in head
    # branded section labels derived from data-model (R5-1 consumed verbatim).
    # Q6 (iter-2 dataviz): the pure-TRITON label is "TRITON" — "(uncoupled)" is redundant
    # (if it doesn't say TRITON-SWMM it's obviously just TRITON), per the user's H relabel.
    assert 'content:"TRITON-SWMM"' in head
    assert 'content:"TRITON"' in head


def test_compose_page_coupled_only_single_section() -> None:
    # A coupled-only (SWMM-specific) figure has no pure-TRITON counterpart -> one section only.
    page = _compose_model_pair_page("myexp", "conduit_flow", _TS_HTML, None)
    assert '<div class="model-section" data-model="tritonswmm"' in page
    assert '<div class="model-section" data-model="triton"' not in page
    assert page.count("srcdoc=") == 1


def test_extract_root_css_vars_reads_root_and_graceful_absent(tmp_path) -> None:
    bundle_root = tmp_path / "combined"
    (bundle_root / "report").mkdir(parents=True)
    # absent report.css -> "" (the composed page falls back to neutral non-brand values)
    assert _extract_root_css_vars(bundle_root) == ""
    # staged report.css -> the :root{} block flows through (brand_theme stipulation)
    (bundle_root / "report" / "report.css").write_text(
        ":root{--uva-blue:#232D4B;--uva-orange:#E57200;}\nbody{margin:0;}\n", encoding="utf-8"
    )
    root = _extract_root_css_vars(bundle_root)
    assert root.startswith(":root")
    assert "--uva-blue:#232D4B" in root
    # and it is injected into the composed page's <style>
    page = _compose_model_pair_page("myexp", "x", _TS_HTML, _TRI_HTML, root)
    assert "--uva-blue:#232D4B" in page[: page.index("</head>")]


def _write_child(bundle_root, eid: str, plot_id: str, body: str) -> None:
    """Minimal child_crates/{eid} bundle: empty cfg (-> 'default' set) + one plots/ figure + manifest."""
    child = bundle_root / "child_crates" / eid
    (child / "plots" / "maps").mkdir(parents=True, exist_ok=True)
    (child / "cfg_analysis.yaml").write_text("{}\n", encoding="utf-8")
    fig = child / "plots" / "maps" / f"{plot_id}.html"
    fig.write_text(body, encoding="utf-8")
    (child / "plots" / "maps" / f"{plot_id}.manifest.json").write_text(
        json.dumps({"plot_id": plot_id, "source_paths_relative": [f"data/{eid}.zarr"]}), encoding="utf-8"
    )


def test_harvest_stacks_both_models_one_spec_per_plot(tmp_path) -> None:
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "synth_cc_clean_tritonswmm", "config_diff_maps", _TS_HTML)
    _write_child(bundle_root, "synth_cc_clean_triton", "config_diff_maps", _TRI_HTML)

    specs = _harvest_per_experiment_rule_specs(bundle_root)

    # exactly ONE spec for (synth_cc_clean, config_diff_maps)
    paired = [s for s in specs if "config_diff_maps" in s.rule_name]
    assert len(paired) == 1
    spec = paired[0]
    # output under paired_figures/, NOT child_crates/
    assert spec.output_path_template.startswith("paired_figures/")
    # the composed page was written, stacks both arms, and carries source paths from BOTH arms
    page = (bundle_root / spec.output_path_template).read_text(encoding="utf-8")
    assert 'class="model-stack"' in page
    assert page.index('<div class="model-section" data-model="tritonswmm"') < page.index(
        '<div class="model-section" data-model="triton"'
    )
    assert page.count("srcdoc=") == 2
    # category is the figure's ORIGINAL category, NEVER the base-experiment name
    assert spec.report_kwargs["category"] != _base_experiment("synth_cc_clean_tritonswmm")
    assert spec.report_kwargs["subcategory"] == "synth_cc_clean"
    labels = json.loads(spec.report_kwargs["labels"])
    # Iter-11 items 5+9 retired the `experiment` facet: it restated the subcategory asserted
    # on the line above, so it could never partition the figures beneath that subcategory.
    # Asserted as the invariant -- no facet may restate the subcategory -- rather than as a
    # key-absence check, so a future facet re-introducing the same value under a different
    # key still fails.
    #
    # Iter-13 CORRECTION. This comment also claimed `models` was retired for the same reason,
    # "a pure function of `base`". That is FALSE and the claim is removed rather than
    # annotated, because a false rationale reads as ground truth to the next author. It holds
    # for which arms a BASE has children for; it fails for which arms a given FIGURE rendered,
    # since `tri_html` is None whenever the pure-TRITON child has no counterpart at that plot
    # id. Measured on the delivered bundle: `conduit_flow` occurs 0 times in the pure-TRITON
    # child and 28 times in the coupled one, so inside one subcategory 28 composed pages carry
    # one arm and 28+ carry two. `models` is RESTORED and is always populated; it passes the
    # assertion below because its values name model arms and never a base experiment.
    assert labels, "labels dict is empty — the probe is measuring nothing"
    assert spec.report_kwargs["subcategory"] not in labels.values()


def test_harvest_labels_carry_no_facet_restating_the_result_name(tmp_path) -> None:
    """A harvested page's labels must not repeat a string the reader can already SEE, and must
    distinguish figures that share a group.

    HISTORY, because the retired half reads as ground truth otherwise. This test formerly also
    forbade any facet value equal to `humanize_plot_id(plot_id)`, on the rationale that the
    result's NAME is shown and a matching facet "renders the label a second time, which is the
    duplicate-heading defect the user reported". Measured against the installed engine, the
    result NAME is rendered as visible text nowhere — its only consumer is a download filename
    attribute — so no facet can duplicate it. The user report IS real and is quoted at
    `combined_snakefile_generator.py:290-292` ("there are redundant headers. ## synth_cc_clean
    — Run metadata appears twice in a row"), but it describes the composed PAGE's <h2>
    duplicating the report chrome's own name rendering — fixed at Iter-10 K by removing that
    <h2> — and has nothing to do with a facet key in the labels table. The rationale borrowed a
    real complaint and misattributed its mechanism.
    The clause blocked the `figure` facet, which is the only column that can identify a row.

    What survives is the half that was well-founded: the SUBCATEGORY is visible, so a facet
    restating it renders twice and cannot partition. Asserted as the INVARIANT rather than as a
    key-absence check, so a differently-keyed facet carrying the same value still fails.
    """
    bundle_root = tmp_path / "combined"
    # Iter-13 fixture widening. TWO plot ids of DIFFERENT ADR-2 segment shapes, because the
    # per-spec assertions below are cross-spec claims and a one-plot-id fixture cannot
    # discriminate: the retired homogeneity clause compared paired[0] against itself and was
    # true for ANY implementation. `metadata` carries no segment (facets: models only); the
    # per-sim id carries member. + evt. (facets: member, event, models).
    plot_ids = ("metadata", "peak_flood_depth__member.gpu_0_r1__evt.event_index.0")
    for pid in plot_ids:
        _write_child(bundle_root, "expA_tritonswmm", pid, _TS_HTML)
        _write_child(bundle_root, "expA_triton", pid, _TRI_HTML)
    specs = _harvest_per_experiment_rule_specs(bundle_root)
    paired = [s for s in specs if s.output_path_template.startswith("paired_figures/")]
    # State the denominator rather than assuming it: an assertion over one spec cannot fail.
    assert len(paired) == len(plot_ids), (
        f"fixture yielded {len(paired)} harvested spec(s) for {len(plot_ids)} plot id(s) — "
        "the cross-spec assertions below would not discriminate"
    )
    by_plot = {PurePosixPath(s.output_path_template).stem: s for s in paired}
    assert set(by_plot) == set(plot_ids), f"unexpected harvested stems {sorted(by_plot)}"
    for plot_id, spec in sorted(by_plot.items()):
        labels = json.loads(spec.report_kwargs["labels"])
        # RETIRED, and the premise is named rather than deleted so it is not re-derived.
        # This asserted that no facet value may equal `humanize_plot_id(plot_id)`, on the
        # rationale that the result's NAME is already shown and a matching facet would
        # "render the label a second time". Measured against the INSTALLED engine
        # (snakemake 9.15.0): `res.name` is rendered as visible text NOWHERE. Its only
        # consumer in the bundled front end is `result_view_button.js`'s `download:`
        # filename attribute. `result_info.js` does not reference it; the result
        # breadcrumb carries `resultPath`, not the name; the results TABLE builds its
        # columns from label keys alone (`abstract_results.js::getLabels`). There is no
        # first rendering for a facet to duplicate, so the invariant guarded nothing and
        # BLOCKED the one facet that makes a row identifiable.
        #
        # The sibling assertion below is NOT retired and is the one that was well-founded:
        # the SUBCATEGORY is visible (it is the menu item the reader clicked), so a facet
        # restating it really does render twice and really cannot partition.
        # Iter-11 items 5+9: `experiment` was measured DEGENERATE at this level -- it restated
        # the subcategory, so the facet table rendered one row where a figure list belongs.
        # Asserted as the invariant rather than as a key list, so a differently-keyed correct
        # implementation still passes.
        assert spec.report_kwargs["subcategory"] not in labels.values()
        # Iter-13. The clause that stood here asserted a FIXED key set across harvested specs.
        # That doctrine is RETIRED by measurement: emitting an absent facet as "" is what
        # produced the blank index columns (18 all-blank rows in the delivered combined
        # report), and heterogeneity is not unvalidated territory -- the per-arm reports
        # already ship three distinct key sets in ONE report and render correctly. What
        # replaces it is the invariant the emitter now enforces.
        assert labels, f"{plot_id}: harvested spec emitted no facet at all"
        assert all(v != "" for v in labels.values()), f"{plot_id}: empty facet value in {labels}"
    # The two shapes must actually DIFFER, or the widened denominator is nominal rather than
    # real and every cross-spec claim above degenerates to a single-shape check.
    assert set(json.loads(by_plot[plot_ids[0]].report_kwargs["labels"])) != set(
        json.loads(by_plot[plot_ids[1]].report_kwargs["labels"])
    ), "fixture plot ids produced identical key sets — widen them"


def test_harvest_no_base_experiment_category(tmp_path) -> None:
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "expA_tritonswmm", "maps_fig", _TS_HTML)
    _write_child(bundle_root, "expA_triton", "maps_fig", _TRI_HTML)
    specs = _harvest_per_experiment_rule_specs(bundle_root)
    bases = {_base_experiment(e) for e in ("expA_tritonswmm", "expA_triton")}
    assert all(s.report_kwargs["category"] not in bases for s in specs)


def test_harvest_coupled_only_single_section(tmp_path) -> None:
    # only a tritonswmm child (no pure-TRITON counterpart) -> the composed page has one section
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "swmmonly_tritonswmm", "conduit_flow", _TS_HTML)
    specs = _harvest_per_experiment_rule_specs(bundle_root)
    paired = [s for s in specs if "conduit_flow" in s.rule_name]
    assert len(paired) == 1
    page = (bundle_root / paired[0].output_path_template).read_text(encoding="utf-8")
    assert '<div class="model-section" data-model="tritonswmm"' in page
    assert '<div class="model-section" data-model="triton"' not in page
    # Iter-13 CORRECTION. This comment claimed `models` was retired and that a coupled-only
    # base "is exactly the case where that facet held ONE value, which is what disqualified
    # it". Both halves are FALSE and are removed rather than annotated, because a false
    # rationale reads as ground truth to the next author. A coupled-only base is the case that
    # VINDICATES the facet: it is where `models` reads `TRITON-SWMM only` and thereby
    # distinguishes this page from the two-arm pages in the same subcategory. Holding one
    # value HERE is what partitioning looks like from inside a single-member partition -- the
    # facet is degenerate only if it holds one value across the WHOLE set, which measurement
    # refuted (28 one-arm vs 28+ two-arm pages under one base).
    #
    # The single-section property this test exists to pin is asserted on the two lines above,
    # unchanged. The assertion below is likewise unchanged and still holds: `models` is always
    # populated, so the labels dict is never empty.
    assert json.loads(paired[0].report_kwargs["labels"]) != {}


# --- combined sidebar category-order vocabulary (F2 placeholder suppression) ---------------


def _combined_order(bundle_root):
    """The category_order render_combined_report_via_snakemake threads into _react_surgery."""
    fixed = list(get_reporting_set("combined").category_order)
    eids = _distinct_child_categories(bundle_root)
    if "Errors and Warnings" in fixed:
        i = fixed.index("Errors and Warnings")
        return fixed[:i] + eids + fixed[i:]
    return fixed + eids


def test_distinct_child_categories_excludes_chrome_only_categories(tmp_path) -> None:
    """`_distinct_child_categories` must return the categories the COMPOSED PAGES carry --
    i.e. the child set's category_order filtered to categories its rule-spec templates
    actually declare -- not the whole declared sidebar vocabulary. "Simulation Health
    (placeholder)" is a RESERVED EMPTY slot injected by _react_surgery step 5 and explicitly
    suppressed under bundle_mode, so no composed page can ever carry it; ordering it puts the
    literal in the combined report's ORDER comparator (F2 regression)."""
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "expA_tritonswmm", "maps_fig", _TS_HTML)
    _write_child(bundle_root, "expA_triton", "maps_fig", _TRI_HTML)
    cats = _distinct_child_categories(bundle_root)
    assert "Simulation Health (placeholder)" not in cats, (
        f"chrome-only reserved slot leaked into the combined sidebar order: {cats}"
    )
    assert cats, "no per-experiment categories survived the filter"


def test_distinct_child_categories_no_model_token_shape(tmp_path) -> None:
    """G3/[Q28] model fungibility: identical for eids carrying NO model token."""
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "synth_multi_sim", "maps_fig", _TS_HTML)
    _write_child(bundle_root, "synth_multi_sim__1__b4b", "maps_fig", _TRI_HTML)
    cats = _distinct_child_categories(bundle_root)
    assert "Simulation Health (placeholder)" not in cats, (
        f"chrome-only reserved slot leaked into the combined sidebar order: {cats}"
    )


def test_combined_order_suppresses_placeholder_in_surgered_html(tmp_path) -> None:
    """End-to-end F2 surface: the ORDER comparator _react_surgery bakes into the combined
    report must not name the placeholder category (mirrors the assertion in
    tests/test_synth_combine_pip2.py::test_combine_pip2_roundtrip, without the 3-min roundtrip)."""
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "expA_tritonswmm", "maps_fig", _TS_HTML)
    _write_child(bundle_root, "expA_triton", "maps_fig", _TRI_HTML)
    html = "<html><body>x(a, b) => a.localeCompare(b)y</body></html>"
    out = apply_post_process_surgery(
        html, bundle_mode=True, navbar_text="Combined", category_order=_combined_order(bundle_root)
    )
    assert "Simulation Health (placeholder)" not in out


# --- composed-page card names (L1: curated ADR-2 label must survive) -----------------------


def test_composed_page_stem_is_the_bare_plot_id(tmp_path) -> None:
    """The Snakemake figure-card display name derives from the composed page's filename STEM,
    and `humanize_plot_id`'s curated _RENDERER_KIND_LABELS table only fires when the stem's
    leading segment IS the renderer kind. A flat "{base}__{plot_id}" stem makes the humanizer
    read the base eid as the renderer kind and silently drop the curated label, so the stem
    must be the bare plot id and the base must be carried by the directory instead."""
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "synth_cc_clean_tritonswmm", "config_diff_maps", _TS_HTML)
    _write_child(bundle_root, "synth_cc_clean_triton", "config_diff_maps", _TRI_HTML)

    specs = _harvest_per_experiment_rule_specs(bundle_root)
    paired = [s for s in specs if "config_diff_maps" in s.rule_name]
    assert len(paired) == 1
    rel = PurePosixPath(paired[0].output_path_template)

    assert rel.stem == "config_diff_maps", (
        f"composed-page stem must be the bare plot id, got {rel.stem!r} from {rel!s}"
    )
    # the base is not lost -- it is the directory
    assert rel.parent.name == "synth_cc_clean", f"base experiment not carried by the directory: {rel!s}"
    assert rel.parts[0] == "paired_figures"
    assert (bundle_root / rel).exists()


def test_composed_page_stem_humanizes_to_the_curated_label(tmp_path) -> None:
    """Anchored on a property true in BOTH states: the curated label for this renderer kind is
    'Config-diff maps' whatever the filename does. Pre-fix (flat
    "{base}__{plot_id}" stem) the humanized card name does NOT contain it; post-fix it does."""
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "synth_cc_clean_tritonswmm", "config_diff_maps", _TS_HTML)
    _write_child(bundle_root, "synth_cc_clean_triton", "config_diff_maps", _TRI_HTML)

    specs = _harvest_per_experiment_rule_specs(bundle_root)
    paired = [s for s in specs if "config_diff_maps" in s.rule_name]
    stem = PurePosixPath(paired[0].output_path_template).stem

    assert humanize_plot_id(stem) == "Config-diff maps", (
        f"composed-page card name lost its curated renderer-kind label: "
        f"humanize_plot_id({stem!r}) == {humanize_plot_id(stem)!r}"
    )


# NAVIGATION SHAPE — a POINTER, deliberately not a declaration.
#
# The combined-report navigation shape is the user's open decision. The three candidates are:
#
#   option 2 eid-first (LANDED)  f"{base} — {child}" if child else base
#   option 2 transposed          f"{child} — {base}" if child else base
#   option 3                     child or base
#
# The ONE place that changes is `combined_snakefile_generator.py`'s `"subcategory":` value
# (currently line 489). It is composed INLINE there and nowhere else.
#
# A `_SUBCATEGORY_SHAPE` lambda stood here and was RETIRED, because it was dead code carrying a
# false claim: nothing consumed it, and its comment said the three options "differ in this lambda
# and in nothing else" — so a future author would have edited it, watched this suite pass, and
# changed nothing. Do not re-introduce it. Wiring it into the test below would be worse than
# leaving it unused: that test asserts tuple DISTINCTNESS, which is invariant across all three
# shapes by construction, so consuming a shape helper there would make the test look
# navigation-sensitive while being unable to tell the options apart.


def test_harvested_figures_sharing_a_group_carry_distinct_label_tuples(tmp_path) -> None:
    """Two figures in ONE (category, subcategory) group must not share a label tuple.

    This is the invariant behind the user's "some of the key results aren't showing up".
    `abstract_results.js::getData` keys each rendered ROW on the joined label tuple and then
    does `entries.get(key).set(arrayKey(entryToggleLabels), path)` — so two figures sharing a
    tuple overwrite at the same key and the earlier one becomes UNREACHABLE. The data layer
    still carries it (`render_results` keys on file path), so record counts look correct and
    the loss is invisible to any check that counts figures. Measured on the delivered combined
    report: 4 figures present and unreachable.

    Asserted as distinct-tuple COUNT rather than as `"figure" in labels`, so a differently
    keyed implementation that also disambiguates still passes, and so the assertion
    discriminates on behaviour rather than on the new facet's name.
    """
    bundle_root = tmp_path / "combined"
    # Two segment-less plot ids: `_plot_id_facets` derives no facet from either, so pre-fix
    # both carry the identical single `models` facet and collide. Measured pre-fix: 2 figures,
    # 1 distinct tuple.
    for pid in ("system_overview", "disk_utilization"):
        _write_child(bundle_root, "expA_tritonswmm", pid, _TS_HTML)
        _write_child(bundle_root, "expA_triton", pid, _TRI_HTML)

    specs = [
        s
        for s in _harvest_per_experiment_rule_specs(bundle_root)
        if s.output_path_template.startswith("paired_figures/")
    ]
    assert len(specs) == 2, f"fixture yielded {len(specs)} harvested spec(s) — cannot discriminate"

    groups: dict[tuple[str, str], list[tuple[str, tuple]]] = {}
    for s in specs:
        rk = s.report_kwargs
        key = (rk["category"], rk["subcategory"])
        tup = tuple(sorted(json.loads(rk["labels"]).items()))
        groups.setdefault(key, []).append((PurePosixPath(s.output_path_template).stem, tup))

    for key, members in groups.items():
        distinct = {t for _, t in members}
        assert len(distinct) == len(members), (
            f"group {key}: {len(members)} figures collapse to {len(distinct)} rendered row(s). "
            f"Colliding tuples render one row and the earlier figures are unreachable. "
            f"Members: {[stem for stem, _ in members]}"
        )
