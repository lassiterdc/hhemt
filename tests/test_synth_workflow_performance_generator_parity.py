"""Four-generator parity for the Workflow performance page (`[Q160]`(7)).

WHY THIS FILE EXISTS, stated as the failure it catches rather than as coverage.

Adding an unconditional renderer touches, among other sites, one figure-ENUMERATION
entry per Snakefile generator. If the registry entry lands and ONE generator's
enumeration entry is missed, the rule is emitted into that generator's Snakefile,
nothing declares its output as an input, Snakemake never schedules it, and the section
is simply ABSENT from that generator's report. No exception, no warning, no
`MissingInputException` — that error fires only in the opposite case (enumeration
present, rule absent). The miss is therefore SILENT.

It matters here specifically because this campaign renders four arms through DIFFERENT
generators and then combines them: a missed reprocess-generator entry yields a combined
report in which some arms carry the Workflow performance section and others do not,
detectable only by eye.

There are FOUR generators, not three. `hhemt architecture.md` Gotcha 66 enumerates four
EDIT SITES and is stale in both directions — it counts the multisim `render_report` as a
separate literal list (it has since been derived from `_plot_items`) and it does not name
`reprocess_snakefile_generator.py`'s hand-rolled builder ladder. The generator census
below is the measured one:

1. `SnakemakeWorkflowBuilder.generate_snakefile_content`                  (multisim)
2. `SensitivityAnalysisWorkflowBuilder.generate_master_snakefile_content` (sensitivity master)
3. `...generate_reprocess_master_snakefile_content`                       (sensitivity reprocess master)
4. `reprocess_snakefile_generator.generate_reprocess_snakefile`           (non-sensitivity reprocess)

Generators 1 and 2 are ALSO covered by tests/test_reporting_set_cosourcing.py's
membership-parity assertion, which compares the registry's template set against the
source generator's emitted rule set. Generators 3 and 4 have no such guard, which is
exactly why the silent class is reachable there. This file covers all four uniformly
rather than only the uncovered two, so the census stays legible in one place.

The assertions are deliberately keyed on the RULE NAME and the OUTPUT PATH rather than
on any rendered content: those two strings are what the DAG planner matches on, and they
are what a missed enumeration entry drops.
"""

from __future__ import annotations

import pytest

from hhemt.reprocess_snakefile_generator import generate_reprocess_snakefile

_RULE = "rule plot_workflow_performance"
_OUTPUT = "plots/workflow_performance.html"


def _assert_emitted_and_enumerated(content: str, generator: str) -> None:
    """Both halves, because either alone is satisfiable while the page is missing.

    Rule-without-enumeration is the SILENT case this file exists for; enumeration-
    without-rule is the loud `MissingInputException` case. Asserting only one of them
    passes on the other's failure.
    """
    assert _RULE in content, f"{generator}: no `{_RULE}` emitted — the dispatcher/ladder entry is missing"
    assert content.count(f'"{_OUTPUT}"') >= 2, (
        f"{generator}: `{_OUTPUT}` appears fewer than twice — it must be the rule's "
        "`output:` AND appear in the rule-all enumeration. A rule whose output nothing "
        "demands is never scheduled, and the section goes silently missing from this "
        f"generator's report. Occurrences: {content.count(chr(34) + _OUTPUT + chr(34))}"
    )


def test_multisim_generator_emits_and_enumerates(synth_multi_sim_analysis):
    content = synth_multi_sim_analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )
    _assert_emitted_and_enumerated(content, "generate_snakefile_content (multisim)")


def test_sensitivity_master_generator_emits_and_enumerates(synth_sensitivity_analysis):
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder
    content = builder.generate_master_snakefile_content(which="both", compression_level=5)
    _assert_emitted_and_enumerated(content, "generate_master_snakefile_content (sensitivity master)")


@pytest.mark.parametrize("start_with", ["consolidate", "render"])
def test_sensitivity_reprocess_master_generator_emits_and_enumerates(synth_sensitivity_analysis, start_with):
    """Generator 3 — the one with no co-sourcing guard, and the one my own round-1
    census missed. Parameterised over the two start stages that emit the report tier."""
    builder = synth_sensitivity_analysis.sensitivity._workflow_builder
    content = builder.generate_reprocess_master_snakefile_content(
        which="both", compression_level=5, start_with=start_with
    )
    _assert_emitted_and_enumerated(content, f"generate_reprocess_master_snakefile_content(start_with={start_with!r})")


@pytest.mark.parametrize("start_with", ["consolidate", "render"])
def test_non_sensitivity_reprocess_generator_emits_and_enumerates(synth_multi_sim_analysis, start_with):
    """Generator 4 — the hand-rolled `if renderer_active(...)` ladder that bypasses the
    `_emit_active_set_plot_rules` dispatcher entirely, so a registry entry alone does
    NOT reach it."""
    content = generate_reprocess_snakefile(synth_multi_sim_analysis._workflow_builder, start_with=start_with)
    _assert_emitted_and_enumerated(content, f"generate_reprocess_snakefile(start_with={start_with!r})")
