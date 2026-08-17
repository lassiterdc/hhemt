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
    assert labels["models"] == "tritonswmm+triton"


def test_harvest_labels_carry_no_facet_restating_the_result_name(tmp_path) -> None:
    """A harvested page's labels must not repeat the string Snakemake already shows as its name.

    The result's NAME on these pages is `humanize_plot_id(plot_id)`. A `figure` facet built from
    the same call therefore filters nothing and renders the label a second time, which is the
    duplicate-heading defect the user reported. Asserted as the INVARIANT — no facet value equals
    the humanized plot id — rather than as `"figure" not in labels`, so the test still fails if a
    future facet is added under a different key carrying the same string.
    """
    bundle_root = tmp_path / "combined"
    _write_child(bundle_root, "expA_tritonswmm", "metadata", _TS_HTML)
    _write_child(bundle_root, "expA_triton", "metadata", _TRI_HTML)
    specs = _harvest_per_experiment_rule_specs(bundle_root)
    paired = [s for s in specs if "metadata" in s.rule_name]
    assert paired, "no harvested metadata spec — the probe is measuring nothing"
    for spec in paired:
        labels = json.loads(spec.report_kwargs["labels"])
        assert labels, "labels dict is empty — the probe is measuring nothing"
        name_shown = humanize_plot_id("metadata")
        restating = [k for k, v in labels.items() if v == name_shown]
        assert not restating, f"facet(s) {restating} restate the result name {name_shown!r}"
        # the facets that actually partition the harvested set are still present
        assert labels["models"] == "tritonswmm+triton"
        assert labels["experiment"] == _base_experiment("expA_tritonswmm")


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
    assert json.loads(paired[0].report_kwargs["labels"])["models"] == "tritonswmm"


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
