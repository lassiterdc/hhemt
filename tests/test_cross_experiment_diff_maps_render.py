"""X3: the spatial diff-map renderer reports BOTH model arms and never draws empty axes."""

from __future__ import annotations

import json

from hhemt.report_renderers.cross_experiment_intercomparison_maps import (
    build_cross_experiment_diff_figure,
)
from tests._figure_invariant_helpers import assert_figure_invariants


def _write(tmp_path, pairs, experiments):
    (tmp_path / "combined_intercomparison.json").write_text(json.dumps({"experiments": experiments, "pairs": pairs}))
    return tmp_path


_EXPERIMENTS = [
    {"experiment": "e_clean_triton", "role": "clean"},
    {"experiment": "e_resume_triton", "role": "resume"},
    {"experiment": "e_clean_tritonswmm", "role": "clean"},
    {"experiment": "e_resume_tritonswmm", "role": "resume"},
]


def _pair(model, identical, variable="max_wlevel_m"):
    return {
        "config": "run_mode=serial|n_mpi=1|n_omp=1|n_gpus=0|n_nodes=1|partition=",
        "variable": variable,
        "event_iloc": 0,
        "identical": identical,
        "model": model,
        "max_abs_diff": 0.0 if identical else 0.02,
    }


def test_no_differences_yields_a_verdict_not_an_empty_plot(tmp_path):
    """User: 'no plot should be displayed if there are no differences, but the no
    difference result should be communicated.'"""
    root = _write(tmp_path, [_pair("TRITON", True), _pair("TRITON-SWMM", True)], _EXPERIMENTS)
    fig = build_cross_experiment_diff_figure(root)
    assert_figure_invariants(fig)
    # A verdict TABLE, not an axes-bearing figure.
    assert [t.type for t in fig.data] == ["table"]
    cells = fig.data[0].cells.values
    assert set(cells[0]) == {"TRITON", "TRITON-SWMM"}
    assert all("bit-identical" in v for v in cells[2])


def test_both_model_arms_are_reported(tmp_path):
    """The iter-2 defect: only one model's comparison surfaced."""
    root = _write(
        tmp_path,
        [_pair("TRITON", True), _pair("TRITON-SWMM", False)],
        _EXPERIMENTS,
    )
    fig = build_cross_experiment_diff_figure(root)
    assert_figure_invariants(fig)
    models = set(fig.data[0].cells.values[0])
    assert models == {"TRITON", "TRITON-SWMM"}


def test_compared_variable_is_named_as_a_peak_not_a_final_timestep(tmp_path):
    root = _write(tmp_path, [_pair("TRITON", True)], _EXPERIMENTS)
    fig = build_cross_experiment_diff_figure(root)
    text = " ".join(a.text or "" for a in fig.layout.annotations)
    assert_figure_invariants(fig)
    assert "max_wlevel_m" in text
    assert "PEAK water level" in text and "maximum over the whole simulation" in text


def test_pure_triton_combine_omits_the_swmm_truncation_caveat(tmp_path):
    """F9 deterministic-only: no SWMM arm => the SWMM caveat is inapplicable."""
    root = _write(tmp_path, [_pair("TRITON", True)], _EXPERIMENTS[:2])
    fig = build_cross_experiment_diff_figure(root)
    text = " ".join(a.text or "" for a in fig.layout.annotations)
    assert_figure_invariants(fig)
    assert "Nperiods" not in text
