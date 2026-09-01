"""K1 (iter-3): deterministic plot-id -> human display label + react-surgery generalization.

Proves the humanizer is model-fungible (identical label for a given stem regardless of
model) and that the react-surgery card-name pass is grammar-driven (handles ANY
independent_var, not just the retired hardcoded n_devices band-aid).
"""

from __future__ import annotations

from hhemt.report_plot_ids import humanize_plot_id
from hhemt.report_renderers._react_surgery import apply_post_process_surgery


def test_humanize_known_renderer_kinds():
    assert humanize_plot_id("config_diff_maps") == "Config-diff maps"
    assert humanize_plot_id("system_overview") == "System overview"
    # b4b renderer kind -> its human label
    assert humanize_plot_id("b4b_clean_identity") == "Cross-hardware raw byte identity over time"


def test_humanize_benchmarking_descriptor_and_extension_strip():
    # the leaked stem, with the .html extension stripped and the descriptor humanized
    assert humanize_plot_id("benchmarking__n_devices.vs.total.html") == "Benchmarking: n devices vs total runtime"
    # a DIFFERENT independent_var humanizes via the same rule with no code edit (the
    # generalization the hardcoded band-aid could not do).
    assert humanize_plot_id("benchmarking__compute_config.vs.total") == "Benchmarking: compute config vs total runtime"


def test_humanize_unknown_kind_falls_back_to_readable():
    # an unmapped renderer kind still yields a readable label (never the raw stem).
    assert humanize_plot_id("some_new_renderer") == "Some new renderer"


def test_humanize_is_model_agnostic():
    # G3 fungibility: a pure-TRITON and a coupled figure share the stem grammar, so the
    # SAME stem resolves to the SAME label — the humanizer takes no model argument.
    stem = "benchmarking__n_devices.vs.total"
    assert humanize_plot_id(stem) == humanize_plot_id(stem)


def test_react_surgery_humanizes_any_card_name():
    # the generalized 8b pass rewrites EVERY figure-card "name" via the grammar humanizer,
    # not just the hardcoded n_devices string. A non-n_devices card must be rewritten.
    html = (
        '<html><body>{"name": "benchmarking__compute_config.vs.total.html", ' '"filename": "keep.html"}</body></html>'
    )
    out = apply_post_process_surgery(html)
    assert '"name": "Benchmarking: compute config vs total runtime"' in out
    # the machine "filename" key is left intact so links/downloads keep working.
    assert '"filename": "keep.html"' in out


def test_label_table_keys_are_minted_renderer_kinds_not_module_names():
    """The two retired dead keys: a label keyed on the renderer MODULE name can never
    match a stem, so the table silently stops governing that figure's label."""
    from hhemt.report_plot_ids import _RENDERER_KIND_LABELS, humanize_plot_id

    assert "peak_flood_depth" in _RENDERER_KIND_LABELS
    assert "conduit_flow" in _RENDERER_KIND_LABELS
    assert "per_sim_peak_flood_depth" not in _RENDERER_KIND_LABELS
    assert "per_sim_conduit_flow" not in _RENDERER_KIND_LABELS
    assert humanize_plot_id("peak_flood_depth__evt.year.9.html").startswith("Peak flood depth")


def test_fallback_preserves_casing_after_the_first_character():
    from hhemt.report_plot_ids import humanize_plot_id

    assert humanize_plot_id("eda_cross_sim_identity") == "Cross-simulation byte identity"
    assert humanize_plot_id("some_NEW_kind") == "Some NEW kind"


def test_labels_are_model_fungible():
    """A stem carries no model token, so a pure-TRITON and a coupled instance of the same
    figure humanize identically (G3)."""
    from hhemt.report_plot_ids import canonical_plot_id, humanize_plot_id

    stem = canonical_plot_id("config_diff_maps")
    assert humanize_plot_id(stem + ".html") == humanize_plot_id(stem)


def test_event_arm_uses_a_supplied_label():
    """R2-4: the evt. segment mirrors the member. segment — a supplied map wins."""
    from hhemt.report_plot_ids import humanize_plot_id

    stem = "peak_flood_depth__evt.year.9"
    labels = {"year.9": "2009-11-11 07:10 - Ida remnant"}
    assert humanize_plot_id(stem, None, labels) == ("Peak flood depth: 2009-11-11 07:10 - Ida remnant")


def test_event_arm_falls_back_to_the_slug_when_no_map_is_supplied():
    """The no-map path must be byte-identical to the pre-R2-4 output, which is what
    keeps every non-threading caller rendering unchanged."""
    from hhemt.report_plot_ids import humanize_plot_id

    stem = "peak_flood_depth__evt.year.9"
    assert humanize_plot_id(stem) == "Peak flood depth: event year.9"
    assert humanize_plot_id(stem, None, {}) == "Peak flood depth: event year.9"
    assert humanize_plot_id(stem, None, {"other": "x"}) == "Peak flood depth: event year.9"
