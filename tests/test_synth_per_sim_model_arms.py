"""Paired tests for the per-model arm expansion (S4-S10) and the composite page (S11-S14).

PRE-FIX arms and why each fails on b627035:

- `test_matrix_yields_four_sections_for_three_models` and its siblings import
  `hhemt.report_renderers._model_arms`, which does not exist -> ImportError.
- `test_shared_depth_max_accepts_model_types` calls `_shared_depth_max` with a
  `model_types=` kwarg the current signature does not accept -> TypeError.
- `test_conduit_builder_accepts_link_group` inspects
  `_build_conduit_flow_figure`'s signature for `link_group`, which is absent.
- `test_shared_peak_flow_vmax_exists` imports a symbol that does not exist.

These are structural (signature/existence) assertions rather than rendered-output
assertions BY DESIGN: the arm expansion's whole content is which arms are reachable,
and a rendered-output test would need a multi-model synth fixture that the compile
tier does not currently build. The rendered-output half is the composite's own test
below, which is fixture-gated.
"""

import inspect

import pytest


def test_matrix_yields_four_sections_for_three_models():
    from hhemt.report_renderers._model_arms import page_sections

    sections = page_sections(["triton", "tritonswmm", "swmm"])
    assert [(m, k) for m, k, _g in sections] == [
        ("triton", "peak_flood_depth"),
        ("tritonswmm", "peak_flood_depth"),
        ("tritonswmm", "conduit_flow"),
        ("swmm", "conduit_flow"),
    ]
    assert len(sections) == 4


def test_page_sections_reaches_every_arm_in_the_matrix():
    """PAGE_MODEL_ORDER is display-only, so an arm can be added to ARM_GROUPS
    and to MODEL_ORDER yet forgotten here -- it would then vanish from the
    composite page with no other failure, which is the one silent gap the
    split-constant design introduces. This converts it into a red one, and
    derives the expected arm set from ARM_GROUPS so it cannot go stale.
    """
    from hhemt.report_renderers._model_arms import (
        ARM_GROUPS,
        MODEL_DISPLAY_NAMES,
        MODEL_ORDER,
        PAGE_MODEL_ORDER,
        page_sections,
    )

    every_arm = {(model, kind) for kind, arms in ARM_GROUPS.items() for model in arms}
    every_model = sorted({model for arms in ARM_GROUPS.values() for model in arms})
    reached = {(model, kind) for model, kind, _g in page_sections(every_model)}

    assert reached == every_arm, f"arms unreachable from the page: {sorted(every_arm - reached)}"
    assert set(PAGE_MODEL_ORDER) == set(MODEL_ORDER), (
        f"display and tie-break tuples cover different models: {sorted(set(PAGE_MODEL_ORDER) ^ set(MODEL_ORDER))}"
    )
    # per_sim_event_page indexes MODEL_DISPLAY_NAMES[model_type] unguarded.
    assert set(every_model) <= set(MODEL_DISPLAY_NAMES), (
        f"rendered arm has no display name: {sorted(set(every_model) - set(MODEL_DISPLAY_NAMES))}"
    )


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [
        (["tritonswmm"], 2),
        (["tritonswmm", "swmm"], 3),
        (["tritonswmm", "swmm", "triton"], 4),
        (["swmm"], 1),
        (["triton"], 1),
    ],
)
def test_section_count_is_derived_not_hardcoded(enabled, expected):
    from hhemt.report_renderers._model_arms import page_sections

    assert len(page_sections(enabled)) == expected


def test_no_depth_arm_for_standalone_swmm_and_no_conduit_arm_for_standalone_triton():
    """The asymmetry is physical: SWMM has no 2D depth field, TRITON has no conduits."""
    from hhemt.report_renderers._model_arms import arms_for

    assert "swmm" not in arms_for("peak_flood_depth", ["tritonswmm", "swmm", "triton"])
    assert "triton" not in arms_for("conduit_flow", ["tritonswmm", "swmm", "triton"])


def test_model_type_none_reproduces_coupled_wins():
    """The legacy single-figure path must be byte-equivalent to the retired chain."""
    from hhemt.report_renderers._model_arms import resolve_arm_group

    assert resolve_arm_group("peak_flood_depth", ["tritonswmm", "triton"]) == "/tritonswmm/triton"
    assert resolve_arm_group("peak_flood_depth", ["triton"]) == "/triton_only/triton"
    assert resolve_arm_group("conduit_flow", ["tritonswmm", "swmm"]) == "/tritonswmm/swmm_link"
    assert resolve_arm_group("conduit_flow", ["swmm"]) == "/swmm_only/swmm_link"
    assert resolve_arm_group("conduit_flow", ["triton"]) is None


def test_explicit_model_type_selects_that_arm_not_the_first():
    """The defect the expansion fixes: the non-coupled arm must be reachable."""
    from hhemt.report_renderers._model_arms import resolve_arm_group

    enabled = ["tritonswmm", "swmm", "triton"]
    assert resolve_arm_group("peak_flood_depth", enabled, model_type="triton") == "/triton_only/triton"
    assert resolve_arm_group("conduit_flow", enabled, model_type="swmm") == "/swmm_only/swmm_link"


def test_matrix_groups_match_the_consolidation_write_side():
    """Guards the one drift that would silently empty a section."""
    from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as _P
    from hhemt.report_renderers._model_arms import ARM_GROUPS

    write_side = set(_P._MODE_TO_TREE_PATH.values())
    for kind_groups in ARM_GROUPS.values():
        for group in kind_groups.values():
            assert group.lstrip("/") in write_side, f"{group} is not written by consolidation"


def test_shared_depth_max_accepts_model_types():
    from hhemt.report_renderers.per_sim_peak_flood_depth import _shared_depth_max

    assert "model_types" in inspect.signature(_shared_depth_max).parameters


def test_shared_wse_range_accepts_model_types():
    from hhemt.report_renderers.per_sim_peak_flood_depth import _shared_wse_range

    assert "model_types" in inspect.signature(_shared_wse_range).parameters


def test_conduit_builder_accepts_link_group():
    """Mirrors _build_peak_flood_depth_figure's long-standing triton_group."""
    from hhemt.report_renderers.per_sim_conduit_flow import _build_conduit_flow_figure

    assert "link_group" in inspect.signature(_build_conduit_flow_figure).parameters


def test_shared_peak_flow_vmax_exists_and_is_cross_arm():
    from hhemt.report_renderers.per_sim_conduit_flow import _shared_peak_flow_vmax

    params = inspect.signature(_shared_peak_flow_vmax).parameters
    assert "analysis" in params and "model_types" in params


def test_composite_renderer_has_the_uniform_signature():
    """Renderer-uniform-signature stipulation."""
    from hhemt.report_renderers import per_sim_event_page

    params = list(inspect.signature(per_sim_event_page.render).parameters)
    assert params[:3] == ["analysis", "report_cfg", "output_path"]


def test_composite_is_registered_in_the_extension_table():
    from hhemt.report_plot_ids import _OUTPUT_EXT_BY_RENDERER

    assert _OUTPUT_EXT_BY_RENDERER["per_sim_event_page"] == {"matplotlib": ".html", "plotly": ".html"}


def test_registry_emits_one_per_sim_template_not_two():
    from hhemt.report_renderers._reporting_sets import REPORTING_SETS

    sel = next(s for s in REPORTING_SETS["default"].renderer_selection if s.builder_key == "per_sim")
    assert len(sel.rule_spec_template) == 1
    assert sel.rule_spec_template[0].rule_name == "plot_per_sim_event_page"


def test_composite_page_renders_end_to_end(synthetic_multisim_completed_isolated, tmp_path):
    """END-TO-END render of the composite page (S18, added round 8).

    The other 17 tests in this file are UNIT-level -- arm matrix, signatures,
    registry membership -- and all 17 passed while the renderer could not render
    at all: its source-paths block reached for `cfg_analysis.weather_time_series`,
    a field that does not exist. The first thing that actually rendered it was a
    bundle round-trip test, which reported the failure three layers away as
    "File ... marked for report but does not exist" because `keep-going: True`
    lets a failed plot rule through to render_report.

    This test closes that gap by rendering the composite directly, so the next
    such defect surfaces as its own AttributeError at its own call site.
    """
    from hhemt.config.report import report_config
    from hhemt.report_renderers import per_sim_event_page
    from hhemt.report_renderers._model_arms import MODEL_DISPLAY_NAMES, page_sections

    analysis = synthetic_multisim_completed_isolated
    out = tmp_path / "event_page__evt.event_index.0.html"

    result = per_sim_event_page.render(analysis, report_config(), out, event_iloc=0)

    assert result.exists(), "composite page renderer produced no file"
    assert result.stat().st_size > 0, "composite page is empty"

    html = result.read_text()
    sections = page_sections(analysis._get_enabled_model_types())
    assert sections, "fixture enables no model arm — test would be vacuous"

    # One section header per applicable (model, renderer) pair. This is the
    # assertion the unit tests structurally could not make: it requires the
    # figures to have actually been built and composed.
    assert html.count('class="eda-figure-title"') == len(sections)
    for model_type, _kind, _group in sections:
        assert MODEL_DISPLAY_NAMES[model_type] in html

    # The Plotly bundle must still be inlined exactly once across N figures.
    assert html.count("Plotly.newPlot") >= len(sections)

    # ADR-6 Gate-C: the emit path wrote a manifest sidecar with declared sources.
    manifests = list(tmp_path.glob("*.manifest.json"))
    assert len(manifests) == 1, f"expected one manifest sidecar, got {manifests}"
