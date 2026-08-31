"""FQ1: conduit_flow is SWMM-derived and must not be emitted for a TRITON-only analysis.

The load-bearing property is not "the figure disappears" but that EMISSION and
ENUMERATION move together. A renderer dropped from emission while left in `rule all`
yields ``MissingInputException`` at Snakemake parse time; the inverse yields an orphan
rule. Those two halves live in different functions (``_emit_plot_rule`` at the emission
site, ``_per_sim_per_member_rule_all_inputs`` for enumeration), so nothing but a test that
inspects the generated Snakefile can establish they agree.

Both arms below are SATISFYING states of the invariant, occupying different correct
positions: coupled must contain conduit_flow in both halves, TRITON-only in neither.
An implementation that gated only one half passes one arm and fails the other.
"""

from __future__ import annotations

import re

import pytest


def _conduit_counts(snakefile: str) -> tuple[int, int]:
    """(rule-definition count, rule-all entry count) for conduit_flow per-member figures."""
    rule_defs = len(re.findall(r"^rule plot_per_sim_per_member_conduit_flow:", snakefile, re.M))
    m = re.search(r"^rule all:\n    input:\n((?:        .*\n)+)", snakefile, re.M)
    rule_all_body = m.group(1) if m else ""
    rule_all_entries = rule_all_body.count("conduit_flow__member.")
    return rule_defs, rule_all_entries


@pytest.mark.parametrize(
    ("model_types", "expect_present"),
    [
        (["tritonswmm", "triton"], True),   # coupled: SWMM link outputs exist
        (["triton"], False),                # TRITON-only: they do not
    ],
    ids=["coupled-emits-conduit", "triton-only-omits-conduit"],
)
def test_conduit_flow_emission_and_enumeration_agree(
    synth_sensitivity_analysis, monkeypatch, model_types, expect_present
):
    master = synth_sensitivity_analysis.sensitivity._workflow_builder.experiment
    monkeypatch.setattr(master, "_get_enabled_model_types", lambda: list(model_types))

    builder = synth_sensitivity_analysis.sensitivity._workflow_builder
    generated = builder.generate_master_snakefile_content(which="both", compression_level=5)
    rule_defs, rule_all_entries = _conduit_counts(generated)

    if expect_present:
        assert rule_defs > 0, "coupled analysis must emit the conduit_flow rule"
        assert rule_all_entries > 0, "coupled analysis must enumerate conduit_flow in rule all"
    else:
        assert rule_defs == 0, (
            "TRITON-only analysis emitted a conduit_flow rule; the emission gate did not fire"
        )
        assert rule_all_entries == 0, (
            "TRITON-only analysis still enumerates conduit_flow in rule all — emission and "
            "enumeration have DESYNCED, which is MissingInputException at parse time"
        )

    # peak_flood_depth is model-agnostic and must survive BOTH arms. This is what
    # catches an over-firing gate that drops the whole per_member selection rather than
    # the one SWMM-derived template.
    assert re.search(r"^rule plot_per_sim_per_member_peak_flood_depth:", generated, re.M), (
        "peak_flood_depth must be emitted regardless of model type"
    )
