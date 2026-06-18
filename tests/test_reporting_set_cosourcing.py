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


def _capture_source_categories(generate_call: Callable[[], object]) -> dict[str, str]:
    """Run a source-side generator with ``_emit_plot_rule`` patched to record each
    rule's category by ``rule_name``. Returns ``{rule_name: category}``."""
    captured: dict[str, str] = {}
    orig = wf._emit_plot_rule

    def _cap(spec, ctx):
        captured[spec.rule_name] = spec.report_kwargs.get("category")
        return orig(spec, ctx)

    wf._emit_plot_rule = _cap
    try:
        generate_call()
    finally:
        wf._emit_plot_rule = orig
    return captured


def _template_categories(set_name: str) -> dict[str, str]:
    """Registry-side ``{rule_name: category}`` for every figure template in a set."""
    rset = get_reporting_set(set_name)
    return {
        tmpl.rule_name: tmpl.report_kwargs.get("category")
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
