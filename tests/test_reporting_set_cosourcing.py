"""Co-sourcing guard: the bundle registry's rule_spec_template category MUST equal
the source-side ``_build_plot_rule_block_*`` ``report(category=)`` for the same
rule (P1b / F-I-3).

The bundle generator (``bundle/snakefile_generator.py``) data-drives its
``report(category=...)`` from each ``RendererSelection.rule_spec_template`` — the
single source. The source-side builders in ``workflow.py`` still hardcode their
own ``report(category=)`` strings (full source-side co-sourcing is a deferred
follow-up, A6). This test pins the two equal, keyed by RULE NAME, so the registry
duplication cannot silently drift from the source builders: if a source category
changes, this test fails until the registry template is updated to match.

Capture mechanism (structured comparison, not Snakefile-text regex): monkeypatch
``workflow._emit_plot_rule`` to record each emitted RuleSpec's
``rule_name -> report_kwargs["category"]`` while a source-side generator runs, then
compare against the registry templates. Keying on ``rule_name`` (not
``renderer_module``) is deliberate: the source ``per_sim_per_sa`` builder emits
rules named ``plot_per_sim_per_sa_*`` but with ``renderer_module`` ``per_sim_*``,
so ``rule_name`` is the stable cross-reference.
"""

from __future__ import annotations

from collections.abc import Callable

import hhemt.workflow as wf
from hhemt.report_renderers._reporting_sets import get_reporting_set


#: The report_kwargs keys the co-sourcing guard compares. `category` alone left
#: `labels` and `caption` unguarded, and both are hand-maintained in TWO places
#: (workflow.py's generators and _reporting_sets.py's templates) with nothing
#: comparing them -- so an edit to one copy shipped silently. `subcategory` IS
#: included: `_kwargs_subset` calls .get() on both sides, so a key absent from both
#: compares None == None, and absent-on-one-side-only is a true divergence the guard
#: should catch. (An earlier revision excluded it on the reasoning that a
#: None-vs-missing comparison would fail on rules that never declared one; that
#: reasoning was wrong, and excluding it narrowed the guard for no benefit.)
_CO_SOURCED_KEYS = ("category", "labels", "caption", "subcategory")


def _kwargs_subset(report_kwargs) -> dict[str, str]:
    """The co-sourced slice of one rule's ``report_kwargs``, missing keys as None."""
    return {k: report_kwargs.get(k) for k in _CO_SOURCED_KEYS}


def _capture_source_categories(generate_call: Callable[[], object]) -> dict[str, dict]:
    """Run a source-side generator with ``_emit_plot_rule`` patched to record each
    rule's co-sourced report_kwargs by ``rule_name``.
    Returns ``{rule_name: {category, labels, caption}}``."""
    captured: dict[str, dict] = {}
    orig = wf._emit_plot_rule

    def _cap(spec, ctx):
        captured[spec.rule_name] = _kwargs_subset(spec.report_kwargs)
        return orig(spec, ctx)

    wf._emit_plot_rule = _cap
    try:
        generate_call()
    finally:
        wf._emit_plot_rule = orig
    return captured


def _template_categories(set_name: str) -> dict[str, dict]:
    """Registry-side ``{rule_name: {category, labels, caption}}`` for every figure
    template in a set."""
    rset = get_reporting_set(set_name)
    return {
        tmpl.rule_name: _kwargs_subset(tmpl.report_kwargs)
        for sel in rset.renderer_selection
        for tmpl in sel.rule_spec_template
    }


def test_default_set_category_co_sourced(synth_multi_sim_analysis):
    """Every default-set rule_spec_template category equals the source-side
    multisim builder's ``report(category=)`` for the same rule (no drift)."""
    builder = synth_multi_sim_analysis._workflow_builder
    source = _capture_source_categories(
        lambda: builder.generate_snakefile_content(
            process_system_level_inputs=True,
            compile_TRITON_SWMM=True,
            prepare_scenarios=True,
            process_timeseries=True,
        )
    )
    templates = _template_categories("default")
    assert templates, "default set has no rule_spec_template figures"
    # Membership parity (Option I / iterate-selection guard): the registry must
    # emit EXACTLY the source generator's plot-rule set — no silently-dropped
    # renderer (registry missing a source rule) and no phantom (registry rule the
    # source never emits). Combined with the category check below, this pins the
    # {rule_name -> category} set equal, which makes the rendered report identical
    # to the source regardless of raw Snakefile emission order (snakemake re-sorts
    # within-category by output basename; cross-category by category_order).
    assert set(templates) == set(source), (
        "default-set membership parity broken: registry templates and source-side "
        "plot rules disagree.\n  only in registry: "
        f"{sorted(set(templates) - set(source))}\n  only in source: "
        f"{sorted(set(source) - set(templates))}"
    )
    for rule_name, category in templates.items():
        assert rule_name in source, (
            f"default-set template rule {rule_name!r} has no source-side counterpart "
            f"in the multisim generator; the bundle would emit a category the source "
            f"never produces. Source rules: {sorted(source)}."
        )
        assert category == source[rule_name], (
            f"category drift for {rule_name!r}: registry template={category!r} vs "
            f"source builder={source[rule_name]!r}. Update the rule_spec_template to "
            f"match the source-side report(category=)."
        )


def test_benchmarking_set_category_co_sourced(synth_sensitivity_analysis):
    """Every benchmarking-set rule_spec_template category equals the source-side
    sensitivity-master builder's ``report(category=)`` for the same rule."""
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder
    source = _capture_source_categories(
        lambda: builder.generate_master_snakefile_content(which="both", compression_level=5)
    )
    templates = _template_categories("benchmarking")
    assert templates, "benchmarking set has no rule_spec_template figures"
    # Membership parity (Option I / iterate-selection guard) — see the default-set
    # test for the rationale. synth_sensitivity fires BOTH conditional predicates
    # (sa_event_pairs + independent_vars), so the source emits the full
    # benchmarking figure set, making set-equality the correct assertion here.
    assert set(templates) == set(source), (
        "benchmarking-set membership parity broken: registry templates and "
        "source-side plot rules disagree.\n  only in registry: "
        f"{sorted(set(templates) - set(source))}\n  only in source: "
        f"{sorted(set(source) - set(templates))}"
    )
    for rule_name, category in templates.items():
        assert rule_name in source, (
            f"benchmarking-set template rule {rule_name!r} has no source-side "
            f"counterpart in the master generator. Source rules: {sorted(source)}."
        )
        assert category == source[rule_name], (
            f"category drift for {rule_name!r}: registry template={category!r} vs "
            f"source builder={source[rule_name]!r}. Update the rule_spec_template to "
            f"match the source-side report(category=)."
        )


def test_b4b_set_config_diff_maps_wired_and_ordered():
    """The report-wiring edit adds ``config_diff_maps`` (rule
    ``plot_eda_compute_sensitivity``) to the b4b set, declaring an ORDERED category.

    Two invariants, scoped to the template this wiring introduces:

    1. Presence — the ``eda_compute_sensitivity`` template is in the b4b set (guards
       against the edit being dropped/reverted, or the second-same-key-selection
       trap where ``_eda_rule_spec_templates()`` would silently ignore it).
    2. Ordered category — its ``report(category=)`` is a member of the set's
       ``category_order`` (so the figure lands in a positioned section rather than
       the implicit "everything else last" tail).

    Scope note: ``category_order`` is deliberately NON-exhaustive — the b4b set
    already inherits pre-existing templates (e.g. ``plot_scenario_status_appendix``
    → ``"Appendix"``) whose category is outside it and sorts at the tail, so a
    universal "every template category in category_order" assertion would (wrongly)
    fail on that inherited template. The full source-co-sourcing guard the
    default/benchmarking sets use is not mirrored here because the b4b set's
    ``eda_compute_sensitivity`` selection emits only when
    ``has_preserved_raw_outputs`` fires — a runtime gate covered by the
    ``reporting_set="b4b"`` dry-run parity check, not a construction-only unit test.
    """
    rset = get_reporting_set("b4b")
    templates = _template_categories("b4b")
    assert templates, "b4b set has no rule_spec_template figures"
    assert "plot_eda_compute_sensitivity" in templates, (
        "b4b set is missing the eda_compute_sensitivity (config_diff_maps) template; "
        "the report-wiring edit did not land (or was dropped as a second same-key "
        "eda_compute_sensitivity selection)."
    )
    # _template_categories now returns the co-sourced kwargs SUBSET per rule, so the
    # category must be selected out of it; `dict in set(...)` would raise TypeError:
    # unhashable type rather than failing an assertion.
    category = templates["plot_eda_compute_sensitivity"]["category"]
    assert category in set(rset.category_order), (
        f"config_diff_maps template declares category {category!r} not in the b4b "
        f"set's category_order {sorted(rset.category_order)}; it would render in the "
        f"untracked tail instead of a positioned section."
    )
