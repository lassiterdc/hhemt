"""Unit tests for the registry-driven plot-rule dispatcher
(`workflow._ReportingSetDispatchMixin._emit_active_set_plot_rules`) — P1b.

The byte-identity gate (test_synth_reporting_sets_byte_identity.py) covers the
production generators, but its fixtures fire BOTH conditional predicates
(sa_event_pairs + independent_vars), so it never exercises the predicate-FALSE /
mixed paths or the B-i interleave-hook placement when the first conditional is
skipped. This module pins that load-bearing logic directly with stub builders, so
the export-rule interleave placement and the predicate gating are verified for
every true/false combination — not just the all-fire case.

Load-bearing invariants tested:
  1. Unconditional renderers always emit, in selection order.
  2. The `interleave_after_unconditional` hook (the export rule at master/reprocess)
     fires EXACTLY ONCE, immediately before the FIRST predicate-keyed entry, and
     fires regardless of whether that first conditional's predicate passes
     (matching the pre-refactor unconditional export emission).
  3. A conditional renderer emits iff its predicate passes against predicate_inputs.
  4. A set with no conditional entries and no hook (the multisim shape) emits only
     its unconditional renderers, in order, with no interleave.
"""

from __future__ import annotations

import TRITON_SWMM_toolkit.workflow as wf
from TRITON_SWMM_toolkit.report_renderers._reporting_sets import (
    RendererSelection,
    ReportingSet,
)


class _StubBuilder(wf._ReportingSetDispatchMixin):
    """Minimal mixin host with marker-returning stub builders. No `_base_builder`,
    so `getattr(self, "_base_builder", self)` self-resolves the common builders to
    these stubs (mirroring the base SnakemakeWorkflowBuilder dispatch path)."""

    def _build_plot_rule_block_system_overview(self, input_flag=None, ctx=None):
        return "SO\n"

    def _build_plot_rule_block_disk_utilization(self, input_flag=None, ctx=None):
        return "DISK\n"

    def _build_plot_rule_block_per_analysis_summary(self, input_flag=None, ctx=None):
        return "PA\n"

    def _build_plot_rule_block_scenario_status_appendix(self, input_flag=None, ctx=None):
        return "SS\n"

    def _build_plot_rule_block_errors_and_warnings(self, input_flag=None, ctx=None):
        return "EW\n"

    def _build_plot_rule_block_per_sim(self, ctx=None):
        return "PSIM\n"

    def _build_plot_rule_block_per_sim_per_sa(self, ctx=None):
        return "PSS\n"

    def _build_plot_rule_block_sensitivity_benchmarking(self, independent_vars, ctx=None):
        return "SB\n"


def _benchmarking_shaped_set() -> ReportingSet:
    """A set with two leading unconditional renderers then two conditional ones,
    matching the benchmarking shape that uses the interleave hook."""
    return ReportingSet(
        name="_test_benchmarking_shaped",
        category_order=(),
        renderer_selection=(
            RendererSelection("system_overview"),
            RendererSelection("disk_utilization"),
            RendererSelection("per_sim_per_sa", predicate_key="has_sa_event_pairs"),
            RendererSelection("sensitivity_benchmarking", predicate_key="has_independent_vars"),
        ),
        validator_key="none",
    )


def _export_hook():
    return "EXPORT\n"


def test_interleave_and_both_conditionals_fire():
    """Both predicates true: unconditionals, then EXPORT (interleave), then both
    conditionals — the production master/reprocess case."""
    out = _StubBuilder()._emit_active_set_plot_rules(
        _benchmarking_shaped_set(),
        input_flag="f",
        predicate_inputs={"sa_event_pairs_sa": [(0, 0)], "independent_vars": ["x"]},
        interleave_after_unconditional=_export_hook,
    )
    assert out == "SO\nDISK\nEXPORT\nPSS\nSB\n"


def test_interleave_fires_even_when_first_conditional_predicate_false():
    """sa_event_pairs empty (per_sim_per_sa skipped) but independent_vars present:
    EXPORT must STILL fire before the (skipped) first conditional, then SB emits.
    This is the load-bearing B-i property — export is unconditional and lands
    between the unconditional and conditional renderers regardless of the first
    conditional's predicate outcome."""
    out = _StubBuilder()._emit_active_set_plot_rules(
        _benchmarking_shaped_set(),
        input_flag="f",
        predicate_inputs={"sa_event_pairs_sa": [], "independent_vars": ["x"]},
        interleave_after_unconditional=_export_hook,
    )
    assert out == "SO\nDISK\nEXPORT\nSB\n"


def test_interleave_fires_when_all_conditionals_skip():
    """Both predicates false: EXPORT still fires after the unconditionals; both
    conditionals skip. The export rule is never gated."""
    out = _StubBuilder()._emit_active_set_plot_rules(
        _benchmarking_shaped_set(),
        input_flag="f",
        predicate_inputs={"sa_event_pairs_sa": [], "independent_vars": []},
        interleave_after_unconditional=_export_hook,
    )
    assert out == "SO\nDISK\nEXPORT\n"


def test_hook_fires_exactly_once():
    """The interleave hook is flushed once, before the FIRST predicate entry — not
    once per conditional. A counting hook proves single emission."""
    calls = {"n": 0}

    def _counting_hook():
        calls["n"] += 1
        return "EXPORT\n"

    out = _StubBuilder()._emit_active_set_plot_rules(
        _benchmarking_shaped_set(),
        input_flag="f",
        predicate_inputs={"sa_event_pairs_sa": [(0, 0)], "independent_vars": ["x"]},
        interleave_after_unconditional=_counting_hook,
    )
    assert calls["n"] == 1
    assert out.count("EXPORT\n") == 1


def test_no_hook_no_conditionals_multisim_shape():
    """Multisim shape: only unconditional renderers, no hook → just the
    unconditionals in order, no interleave (the caller emits export as a trailing
    sibling)."""
    multisim_shaped = ReportingSet(
        name="_test_multisim_shaped",
        category_order=(),
        renderer_selection=(
            RendererSelection("system_overview"),
            RendererSelection("per_sim"),
            RendererSelection("disk_utilization"),
        ),
        validator_key="none",
    )
    out = _StubBuilder()._emit_active_set_plot_rules(multisim_shaped, input_flag="f")
    assert out == "SO\nPSIM\nDISK\n"
