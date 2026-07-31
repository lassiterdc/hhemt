"""F1 combine-gate: auto-detect paired-model-arm downgrade.

The both-models report requires combining a pure-TRITON arm with a TRITON-SWMM arm.
`toggle_triton_model` / `toggle_tritonswmm_model` are BLOCKING experiment-identity
fields, so a naive combine aborts. `_downgrade_paired_model_arms` downgrades JUST the
paired model-toggle divergence to WARNING — and ONLY when the divergence set is exactly
the paired toggles AND the eids collapse to fewer base experiments (some base carries
both arms). These tests exercise that decision logic directly (the whole new behaviour
lives in the pure function; `check_bundle_compatibility` only wires it to bundle roots).
"""

from __future__ import annotations

from hhemt.bundle._compatibility import (
    CompatibilityDivergence,
    CompatibilityReport,
    CompatibilitySeverity,
    _base_experiment,
    _downgrade_paired_model_arms,
)


def _toggle_div(field_name: str, a: str, b: str) -> CompatibilityDivergence:
    return CompatibilityDivergence(
        field_name=field_name,
        bucket="experiment",
        severity=CompatibilitySeverity.BLOCKING,
        bundle_a=a,
        bundle_b=b,
        value_a=True,
        value_b=False,
    )


def _warning_div(field_name: str, a: str, b: str) -> CompatibilityDivergence:
    return CompatibilityDivergence(
        field_name=field_name,
        bucket="experiment",
        severity=CompatibilitySeverity.WARNING,
        bundle_a=a,
        bundle_b=b,
        value_a="x",
        value_b="y",
    )


def test_base_experiment_strips_model_token() -> None:
    # tritonswmm listed first (longer token) so it is not mis-stripped as `_triton` + `swmm`.
    assert _base_experiment("synth_cc_clean_triton") == "synth_cc_clean"
    assert _base_experiment("synth_cc_clean_tritonswmm") == "synth_cc_clean"
    assert _base_experiment("synth_cc_resume_tritonswmm") == "synth_cc_resume"
    # no model token -> unchanged
    assert _base_experiment("some_other_experiment") == "some_other_experiment"


def test_gate_admits_both_models_of_one_experiment() -> None:
    # myexp_triton vs myexp_tritonswmm: only the paired model toggles block; 2 eids -> 1 base.
    report = CompatibilityReport(
        divergences=[
            _toggle_div("toggle_triton_model", "myexp_triton", "myexp_tritonswmm"),
            _toggle_div("toggle_tritonswmm_model", "myexp_triton", "myexp_tritonswmm"),
        ]
    )
    assert report.is_compatible is False  # BLOCKING pre-downgrade
    _downgrade_paired_model_arms(report, ["myexp_triton", "myexp_tritonswmm"])
    assert report.is_compatible is True
    assert all(d.severity is CompatibilitySeverity.WARNING for d in report.divergences)


def test_gate_admits_four_way_clean_resume_x_models() -> None:
    # The 4-way synth_cc_{clean,resume}_{triton,tritonswmm}: model toggles BLOCK, the
    # clean-vs-resume sensitivity axis is WARNING; 4 eids -> 2 bases (collapse).
    eids = [
        "synth_cc_clean_triton",
        "synth_cc_clean_tritonswmm",
        "synth_cc_resume_triton",
        "synth_cc_resume_tritonswmm",
    ]
    report = CompatibilityReport(
        divergences=[
            _toggle_div("toggle_triton_model", eids[0], eids[1]),
            _toggle_div("toggle_tritonswmm_model", eids[0], eids[1]),
            _warning_div("sensitivity_analysis", eids[0], eids[2]),
        ]
    )
    assert report.is_compatible is False
    _downgrade_paired_model_arms(report, eids)
    assert report.is_compatible is True
    # the pre-existing sensitivity WARNING is untouched, still present and non-blocking
    assert any(d.field_name == "sensitivity_analysis" for d in report.divergences)


def test_gate_still_blocks_unrelated_single_arm_experiments() -> None:
    # expA_triton vs expB_tritonswmm differ in a NON-model identity field (case_name) too:
    # all_blocking is not a subset of the paired toggles -> no downgrade -> still blocks.
    report = CompatibilityReport(
        divergences=[
            _toggle_div("toggle_triton_model", "expA_triton", "expB_tritonswmm"),
            CompatibilityDivergence(
                field_name="case_name",
                bucket="experiment",
                severity=CompatibilitySeverity.BLOCKING,
                bundle_a="expA_triton",
                bundle_b="expB_tritonswmm",
                value_a="A",
                value_b="B",
            ),
        ]
    )
    _downgrade_paired_model_arms(report, ["expA_triton", "expB_tritonswmm"])
    assert report.is_compatible is False


def test_gate_still_blocks_two_unrelated_model_only_diff() -> None:
    # expA_triton vs expB_tritonswmm identical EXCEPT the model toggle: the blocking set is the
    # paired toggles, but bases {expA, expB} do NOT collapse (2 == 2) -> no downgrade -> blocks.
    report = CompatibilityReport(
        divergences=[
            _toggle_div("toggle_triton_model", "expA_triton", "expB_tritonswmm"),
            _toggle_div("toggle_tritonswmm_model", "expA_triton", "expB_tritonswmm"),
        ]
    )
    _downgrade_paired_model_arms(report, ["expA_triton", "expB_tritonswmm"])
    assert report.is_compatible is False
