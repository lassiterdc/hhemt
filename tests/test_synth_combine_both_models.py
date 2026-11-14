"""F1 both-models harvest (Option C): base-experiment collapse helpers.

Unit tests for the model-token stripping that collapses two single-model child bundles of
one experiment (synth_cc_clean_triton + synth_cc_clean_tritonswmm) into ONE base-experiment
report category carrying both model arms.
"""

from __future__ import annotations

from hhemt.bundle.combined_snakefile_generator import _base_experiment, _model_of


def test_base_experiment_strips_model_token():
    assert _base_experiment("synth_cc_clean_triton") == "synth_cc_clean"
    assert _base_experiment("synth_cc_clean_tritonswmm") == "synth_cc_clean"
    assert _base_experiment("synth_cc_resume_triton") == "synth_cc_resume"
    assert _base_experiment("synth_cc_resume_tritonswmm") == "synth_cc_resume"
    # No model token -> unchanged.
    assert _base_experiment("no_model_token") == "no_model_token"


def test_model_of_returns_arm_token():
    assert _model_of("synth_cc_clean_triton") == "triton"
    assert _model_of("synth_cc_resume_tritonswmm") == "tritonswmm"
    assert _model_of("no_model_token") == ""


def test_tritonswmm_checked_before_triton():
    # '_tritonswmm' is the LONGER token and must be matched first, else '_triton' would
    # never fire on a '..._tritonswmm' eid but '_tritonswmm' endswith check must win.
    assert _base_experiment("exp_tritonswmm") == "exp"
    assert _model_of("exp_tritonswmm") == "tritonswmm"


def test_four_eids_collapse_to_two_base_experiments():
    eids = [
        "synth_cc_clean_triton",
        "synth_cc_clean_tritonswmm",
        "synth_cc_resume_triton",
        "synth_cc_resume_tritonswmm",
    ]
    assert sorted({_base_experiment(e) for e in eids}) == ["synth_cc_clean", "synth_cc_resume"]
