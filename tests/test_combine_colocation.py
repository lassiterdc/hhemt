"""Both model arms co-located on ONE composed combined-report page.

For any combined-report page carrying a TRITON-SWMM figure, the pure-TRITON counterpart is
stacked BELOW it on the SAME page (the `.model-stack` layout, inlined into the composed
page `<head>` because the stylesheet passed with `--report-stylesheet` styles only the
report chrome and never reaches a figure's inner document).
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
    """Property: a composed page carries both model arms as `.model-section` srcdoc iframes
    inside one `.model-stack`, the TRITON-SWMM section before the pure-TRITON one, and tags the
    stack with the plot id and the base experiment.

    Why it is required: the page exists to compare the two arms in one view; a missing arm, or
    the arms in the other order, puts the coupled result out of the position the layout labels.
    What kills it: dropping either section, emitting them in the other order, or inlining the
    figure documents instead of embedding each in its own srcdoc iframe.
    A second correct implementation still passes: one that changes the iframe titles, the
    loading attribute or the auto-fit script, since none of those is asserted.
    """
    page = _compose_model_pair_page("synth_cc_clean", "config_diff_maps", _TS_HTML, _TRI_HTML)
    # the stacked .model-stack wrapper + both arms as data-model sections, tritonswmm ABOVE triton.
    # Anchor on the section <div> element, not a bare data-model= substring (the inlined <head>
    # CSS carries [data-model="triton"] selectors on every page).
    assert 'class="model-stack"' in page
    i_ts = page.index('<div class="model-section" data-model="tritonswmm"')
    i_tri = page.index('<div class="model-section" data-model="triton"')
    assert i_ts < i_tri, "TRITON-SWMM section must be emitted before the pure-TRITON section"
    # both arms embedded, each in its own srcdoc iframe: two srcdoc= attributes on the page
    assert page.count("srcdoc=") == 2
    assert 'class="model-arm-frame"' in page
    # the data-plot / data-config attributes carry the plot id and the base experiment
    assert 'data-plot="config_diff_maps"' in page
    assert 'data-config="synth_cc_clean"' in page


def test_compose_page_inlines_model_stack_css() -> None:
    """Property: the composed page's own <head> carries the `.model-stack` CSS, the
    `.model-arm-frame` height rule, and the two section labels "TRITON-SWMM" and "TRITON".

    Why it is required: the report's --report-stylesheet styles only the report chrome and never
    reaches a figure's inner document, so CSS that is not inlined leaves the page unstyled and
    an iframe without an explicit height renders at the browser default.
    What kills it: moving the stack CSS into report.css, or dropping the arm-frame rule or
    either label rule.
    A second correct implementation still passes: one that reorders, rewrites or adds rules in
    the same inline <style> block, as long as those selectors and labels are present.
    """
    # The .model-stack CSS is INLINED into the composed page <head> because report.css reaches
    # only the report chrome, including the .model-arm-frame height rule an iframe needs.
    page = _compose_model_pair_page("myexp", "peak_flood_depth", _TS_HTML, _TRI_HTML)
    head = page[: page.index("</head>")]
    assert ".model-stack" in head
    assert ".model-arm-frame" in head
    # branded section labels derived from data-model.
    # The pure-TRITON label is "TRITON": "(uncoupled)" is redundant, because a label that does
    # not say TRITON-SWMM already reads as TRITON alone.
    assert 'content:"TRITON-SWMM"' in head
    assert 'content:"TRITON"' in head


def test_compose_page_coupled_only_single_section() -> None:
    """Property: with no pure-TRITON counterpart (`tri_html=None`), the composed page has the
    TRITON-SWMM section only, and exactly one srcdoc iframe.

    Why it is required: an empty second section reads to the viewer as a missing pure-TRITON
    result, when a coupled-only (SWMM-specific) figure has none by construction.
    What kills it: emitting the triton section unconditionally, or with an empty iframe.
    A second correct implementation still passes: one that also prints an explanatory note,
    provided the note is not a `data-model="triton"` section.
    """
    # A coupled-only (SWMM-specific) figure has no pure-TRITON counterpart -> one section only.
    page = _compose_model_pair_page("myexp", "conduit_flow", _TS_HTML, None)
    assert '<div class="model-section" data-model="tritonswmm"' in page
    assert '<div class="model-section" data-model="triton"' not in page
    assert page.count("srcdoc=") == 1


def test_extract_root_css_vars_reads_root_and_graceful_absent(tmp_path) -> None:
    """Property: with no staged report.css, `_extract_root_css_vars` returns ""; with one, it
    returns the stylesheet's `:root{...}` block, which then appears in the composed page's
    <head>.

    Why it is required: the composed page resolves its `var(--uva-*)` colours only through that
    block; where report.css is not staged yet, the page must fall back to its neutral colours
    rather than raise.
    What kills it: raising on the absent file, returning something other than the `:root`
    block, or not injecting it into the page.
    A second correct implementation still passes: one that normalizes whitespace inside the
    returned block.
    """
    bundle_root = tmp_path / "combined"
    (bundle_root / "report").mkdir(parents=True)
    # absent report.css -> "" (the composed page falls back to neutral non-brand values)
    assert _extract_root_css_vars(bundle_root) == ""
    # staged report.css -> the :root{} block flows through, so brand colours come from brand_theme
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
    """Property: two child bundles of one base experiment yield exactly ONE spec per plot id. It
    is written under paired_figures/, stacks both arms, keeps the figure's original category,
    takes the base experiment as its subcategory, and carries no facet that restates that
    subcategory.

    Why it is required: one spec per child would put the two arms on separate pages, a base-named
    category adds a sidebar level per experiment, and a facet equal to the subcategory renders
    twice and cannot partition the figures beneath it.
    What kills it: emitting one spec per child, writing under child_crates/, using the base as
    the category, or reinstating an `experiment` facet.
    A second correct implementation still passes: one that emits additional facets, provided
    none of their values equals the subcategory.
    """
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
    # The `experiment` facet was retired: it restated the subcategory asserted
    # on the line above, so it could never partition the figures beneath that subcategory.
    # Asserted as the invariant -- no facet may restate the subcategory -- rather than as a
    # key-absence check, so a future facet re-introducing the same value under a different
    # key still fails.
    #
    # `models` is always populated, and it varies by plot id because a FIGURE's arm set does:
    # `tri_html` is None whenever the pure-TRITON child has no counterpart at that plot id, so
    # within one subcategory some composed pages carry one arm and others two. Its values name
    # model arms and never a base experiment, so it passes the assertion below. Deliberately
    # NOT treated as a pure function of the base experiment: that holds for which arms a BASE
    # has children for, not for which arms a given figure rendered, and dropping the facet on
    # that ground would make one-arm and two-arm pages under one base indistinguishable.
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
    attribute — so no facet can duplicate it. The user report IS real and is quoted in
    `combined_snakefile_generator._compose_model_pair_page` ("there are redundant headers.
    ## synth_cc_clean — Run metadata appears twice in a row"), but it describes the composed
    PAGE's <h2> duplicating the name the report chrome already shows — fixed by removing that
    <h2> — and has nothing to do with a facet key in the labels table. The rationale borrowed a
    real complaint and misattributed its mechanism.
    The clause blocked the `figure` facet, which is the only column that can identify a row.

    What survives is the half that was well-founded: the SUBCATEGORY is visible, so a facet
    restating it renders twice and cannot partition. Asserted as the INVARIANT rather than as a
    key-absence check, so a differently-keyed facet carrying the same value still fails.

    Property: across two plot ids of different segment shapes, every harvested spec carries at
    least one facet, no empty facet value, and no facet equal to its subcategory, and the two
    specs' key sets differ.
    Why it is required: a facet equal to the subcategory renders one row where a figure list
    belongs, and an empty facet value renders a blank index column.
    What kills it: reinstating an `experiment` facet, emitting an absent facet as "", or forcing
    one fixed key set on every spec.
    A second correct implementation still passes: one that renames the facet keys, or adds
    facets, as long as no value is empty or equal to the subcategory.
    """
    bundle_root = tmp_path / "combined"
    # Fixture widening: TWO plot ids of DIFFERENT plot-id segment shapes, because the
    # per-spec assertions below are cross-spec claims and a one-plot-id fixture cannot
    # discriminate: the retired homogeneity clause compared paired[0] against itself and was
    # true for ANY implementation. `metadata` carries no segment, so its facets are
    # {figure, models}; the per-sim id carries member. + evt., so its facets are
    # {figure, member, event, models}. `figure` is emitted first and unconditionally.
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
        # The `experiment` facet was measured DEGENERATE at this level -- it restated
        # the subcategory, so the facet table rendered one row where a figure list belongs.
        # Asserted as the invariant rather than as a key list, so a differently-keyed correct
        # implementation still passes.
        assert spec.report_kwargs["subcategory"] not in labels.values()
        # The clause that stood here asserted a FIXED key set across harvested specs.
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
    """Property: no harvested spec's category is a base-experiment name.

    Why it is required: categories are the sidebar's top level, so a base name there adds one
    navigation level per experiment on top of the child report's own categories.
    What kills it: setting a spec's category to the base experiment.
    A second correct implementation still passes: one that files an unclaimed figure under any
    category other than a base name, such as "Other".
    """
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "expA_tritonswmm", "maps_fig", _TS_HTML)
    _write_child(bundle_root, "expA_triton", "maps_fig", _TRI_HTML)
    specs = _harvest_per_experiment_rule_specs(bundle_root)
    bases = {_base_experiment(e) for e in ("expA_tritonswmm", "expA_triton")}
    assert all(s.report_kwargs["category"] not in bases for s in specs)


def test_harvest_coupled_only_single_section(tmp_path) -> None:
    """Property: a base with only a TRITON-SWMM child yields one composed page with the
    TRITON-SWMM section alone and a non-empty labels dict.

    Why it is required: an empty second section reads as a missing result, and an empty labels
    dict renders a blank index row.
    What kills it: emitting the triton section without a counterpart, or emitting no facet when
    only one arm exists.
    A second correct implementation still passes: one that emits any non-empty facet set.
    """
    # only a tritonswmm child (no pure-TRITON counterpart) -> the composed page has one section
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "swmmonly_tritonswmm", "conduit_flow", _TS_HTML)
    specs = _harvest_per_experiment_rule_specs(bundle_root)
    paired = [s for s in specs if "conduit_flow" in s.rule_name]
    assert len(paired) == 1
    page = (bundle_root / paired[0].output_path_template).read_text(encoding="utf-8")
    assert '<div class="model-section" data-model="tritonswmm"' in page
    assert '<div class="model-section" data-model="triton"' not in page
    # On a coupled-only base `models` reads `TRITON-SWMM only`, which distinguishes this page
    # from the two-arm pages in the same subcategory. Deliberately NOT read as a degenerate
    # facet because it holds one value here: a facet is degenerate only when it holds one value
    # across the WHOLE set of pages it labels, and a single-member partition always shows one.
    #
    # The single-section property this test exists to pin is asserted on the two lines above,
    # unchanged. The assertion below is likewise unchanged and still holds: `models` is always
    # populated, so the labels dict is never empty.
    assert json.loads(paired[0].report_kwargs["labels"]) != {}


# --- combined sidebar category order: only categories a composed page carries -------------


def _combined_order(bundle_root):
    """The category_order render_combined_report_via_snakemake threads into _react_surgery."""
    fixed = list(get_reporting_set("combined").category_order)
    eids = _distinct_child_categories(bundle_root)
    return fixed + eids


def test_distinct_child_categories_excludes_chrome_only_categories(tmp_path) -> None:
    """`_distinct_child_categories` must return the categories the COMPOSED PAGES carry --
    i.e. the child set's category_order filtered to categories its rule-spec templates
    actually declare -- not the whole declared sidebar vocabulary. A name in the combined
    order that no page carries adds no sidebar entry, since _react_surgery's comparator only
    ranks categories the report already has and the surgery injects none of its own, but the
    name is still written into the comparator's ORDER literal in the rendered HTML.
    "Simulation Health (placeholder)" is the regression case asserted below: no composed page
    carries it, so it must not reach the order this function returns.

    Property: the returned categories are non-empty and do not include that name.
    Why it is required: see above; a name with no page writes a dead entry into the ORDER literal.
    What kills it: returning an empty list, or adding the name to the returned order.
    A second correct implementation still passes: one that reads the declared categories some
    other way, as long as only categories a composed page can carry are returned.
    Not established: no current reporting set orders that name (measured over REPORTING_SETS),
    so an implementation that returned a set's whole category_order unfiltered would also pass
    today; the filtering itself is not discriminated by this test."""
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "expA_tritonswmm", "maps_fig", _TS_HTML)
    _write_child(bundle_root, "expA_triton", "maps_fig", _TRI_HTML)
    cats = _distinct_child_categories(bundle_root)
    assert "Simulation Health (placeholder)" not in cats, (
        f"a category no composed page carries reached the combined sidebar order: {cats}"
    )
    assert cats, "no per-experiment categories survived the filter"


def test_distinct_child_categories_no_model_token_shape(tmp_path) -> None:
    """The same filtering holds for child eids carrying NO model token: the categories returned
    must not depend on how the children are named.

    Property: for two children whose eids carry no `_tritonswmm`/`_triton` token, the returned
    categories do not include "Simulation Health (placeholder)".
    Why it is required: eids outside the model-arm naming are the common case outside the
    synthetic campaign, and the filter must not depend on the token.
    What kills it: adding that name to the returned order when no model token is present.
    A second correct implementation still passes: any that returns the same filtered categories
    regardless of eid naming.
    Not established: as in the test above, no current reporting set orders that name, so this
    assertion does not discriminate a missing filter."""
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "synth_multi_sim", "maps_fig", _TS_HTML)
    _write_child(bundle_root, "synth_multi_sim__1__b4b", "maps_fig", _TRI_HTML)
    cats = _distinct_child_categories(bundle_root)
    assert "Simulation Health (placeholder)" not in cats, (
        f"a category no composed page carries reached the combined sidebar order: {cats}"
    )


def test_combined_order_suppresses_placeholder_in_surgered_html(tmp_path) -> None:
    """End-to-end surface: the ORDER comparator _react_surgery bakes into the combined
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


# --- composed-page card names (the curated renderer-kind label must survive) --------------


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

    assert rel.stem == "config_diff_maps", f"composed-page stem must be the bare plot id, got {rel.stem!r} from {rel!s}"
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
# The combined-report navigation shape has three candidate forms; the first is current:
#
#   base — child (current)       f"{base} — {child}" if child else base
#   child — base                 f"{child} — {base}" if child else base
#   child only                   child or base
#
# The ONE place that changes is the `"subcategory":` value in
# `combined_snakefile_generator._harvest_per_experiment_rule_specs`. It is composed INLINE
# there and nowhere else.
#
# A `_SUBCATEGORY_SHAPE` lambda stood here and was RETIRED, because it was dead code carrying a
# false claim: nothing consumed it, and its comment said the three shapes "differ in this lambda
# and in nothing else" — so a future author would have edited it, watched this suite pass, and
# changed nothing. Do not re-introduce it. Wiring it into the test below would be worse than
# leaving it unused: that test asserts tuple DISTINCTNESS, which is invariant across all three
# shapes by construction, so consuming a shape helper there would make the test look
# navigation-sensitive while being unable to tell the shapes apart.


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
