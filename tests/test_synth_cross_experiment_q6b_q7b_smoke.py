"""Q6b / Q7b cross-experiment renderer smokes (iter-2, compile-free).

Q6b: cross_experiment_intercomparison carries a per-pair ``model`` column and groups the
Tabulator grid by model (``groupBy:"model"``) -> TRITON-SWMM and TRITON pairs render as
separate sections. Q7b: cross_experiment_disk_utilization pivots each child crate's
``_status/_du.json`` into one column per child and declares each ``_du.json`` as a source
(ADR-6 non-empty-source gate / Gotcha-53). Both drive the REAL renderer bodies on a
``types.SimpleNamespace`` stub analysis + hand-written on-disk fixtures -- no TRITON/SWMM
compile, sub-second. Deliberately does NOT reuse ``test_combine_colocation._write_child``
(that covers the harvest/pairing surface, not these two renderer deltas)."""

from __future__ import annotations

import json
import types
from pathlib import Path

from hhemt.du_sentinels import write_du_sentinel
from hhemt.report_renderers import (
    cross_experiment_disk_utilization,
    cross_experiment_intercomparison,
)


def _stub_analysis(analysis_dir: Path):
    return types.SimpleNamespace(
        analysis_paths=types.SimpleNamespace(analysis_dir=analysis_dir),
    )


def test_intercomparison_splits_by_model_q6b(tmp_path):
    """Q6b: the ``model`` column and per-model grouping survive the real render, and the
    read-model is declared as the sole source (non-empty-source gate)."""
    analysis_dir = tmp_path / "combined"
    analysis_dir.mkdir()
    cfg = "run_mode=serial|n_mpi=1|n_omp=1|n_gpus=0|n_nodes=1|partition=standard"
    (analysis_dir / "combined_intercomparison.json").write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment": "clean", "role": "clean"},
                    {"experiment": "resume", "role": "resume"},
                ],
                "pairs": [
                    {"config": cfg, "variable": "max_wlevel_m", "event_iloc": 0,
                     "identical": True, "max_abs_diff": 0.0, "model": "TRITON-SWMM"},
                    {"config": cfg, "variable": "max_wlevel_m", "event_iloc": 0,
                     "identical": True, "max_abs_diff": 0.0, "model": "TRITON"},
                ],
            }
        ),
        encoding="utf-8",
    )

    out = analysis_dir / "plots" / "cross_experiment_intercomparison.html"
    out.parent.mkdir(parents=True)
    cross_experiment_intercomparison.render(_stub_analysis(analysis_dir), report_cfg=None, output_path=out)

    html = out.read_text()
    assert "groupBy" in html, "Q6b: per-model grouping (groupBy) missing from the Tabulator options"
    assert "TRITON-SWMM" in html and "TRITON" in html, "Q6b: both model values must render"
    assert "model" in html, "Q6b: the model column must be present"

    manifest = json.loads((out.parent / "cross_experiment_intercomparison.manifest.json").read_text())
    assert manifest["source_paths_relative"] == ["combined_intercomparison.json"], (
        "Q6b: the read-model must be declared as the sole source (Gotcha-53)"
    )


def test_intercomparison_honest_placeholder_when_no_pairs(tmp_path):
    """No pairs -> honest placeholder (never a crash), read-model still declared."""
    analysis_dir = tmp_path / "combined"
    analysis_dir.mkdir()
    (analysis_dir / "combined_intercomparison.json").write_text(
        json.dumps({"experiments": [], "pairs": []}), encoding="utf-8"
    )
    out = analysis_dir / "plots" / "cross_experiment_intercomparison.html"
    out.parent.mkdir(parents=True)
    cross_experiment_intercomparison.render(_stub_analysis(analysis_dir), report_cfg=None, output_path=out)
    html = out.read_text()
    assert "No paired compute-configs" in html
    # N5: `heading` is shared with the populated path, so the asymmetry note must NOT
    # render here — an unguarded note would disclose a denominator that was never measured.
    assert "Why the row counts differ per model" not in html


def _intercomparison_html(tmp_path, pairs) -> str:
    """Render the intercomparison over a synthetic payload and return its HTML."""
    analysis_dir = tmp_path / "combined"
    analysis_dir.mkdir()
    (analysis_dir / "combined_intercomparison.json").write_text(
        json.dumps(
            {
                "experiments": [
                    {"experiment": "clean", "role": "clean"},
                    {"experiment": "resume", "role": "resume"},
                ],
                "pairs": pairs,
            }
        ),
        encoding="utf-8",
    )
    out = analysis_dir / "plots" / "cross_experiment_intercomparison.html"
    out.parent.mkdir(parents=True)
    cross_experiment_intercomparison.render(_stub_analysis(analysis_dir), report_cfg=None, output_path=out)
    return out.read_text()


def _pair(i: int, variable: str, model: str) -> dict:
    cfg = f"run_mode=serial|n_mpi={i}|n_omp=1|n_gpus=0|n_nodes=1|partition=standard"
    return {
        "config": cfg,
        "variable": variable,
        "event_iloc": 0,
        "identical": True,
        "max_abs_diff": 0.0,
        "model": model,
    }


def test_intercomparison_top_of_page_carries_only_the_title_ic1_g1():
    """IC-1 + G1: the page's leading block is the title and nothing else.

    Supersedes the two N5 denominator tests. Those asserted the presence of an
    explanatory paragraph that the IC-1 instruction removes outright, so they
    encoded a presentation decision that has since been reversed rather than a
    property the page must hold.

    Anchored on a property true in BOTH states -- the ``<h2>`` is present before
    and after -- so the assertion discriminates on the paragraph's presence
    rather than on wording that could not exist pre-fix. ``uncoupled`` is G1's
    banned word and occurred only inside that paragraph.
    """
    import tempfile

    pairs = [_pair(i, "max_wlevel_m", "TRITON") for i in range(14)]
    pairs += [_pair(i, "max_wlevel_m", "TRITON-SWMM") for i in range(14)]
    pairs += [_pair(i, "max_flow_cms", "TRITON-SWMM") for i in range(14)]

    with tempfile.TemporaryDirectory() as td:
        html = _intercomparison_html(Path(td), pairs)

    assert "<h2>Cross-Experiment Results: clean vs resume</h2>" in html
    assert "Why the row counts differ per model" not in html
    assert "uncoupled" not in html
    assert "<p>Experiments:" not in html


def test_cross_experiment_disk_utilization_pivots_per_child_q7b(tmp_path):
    """Q7b: one column per child crate (the per-model pivot) + each child's ``_du.json``
    declared as a source (ADR-6 / Gotcha-53). One source per child == one column per child."""
    analysis_dir = tmp_path / "combined"
    crates = analysis_dir / "child_crates"
    for eid, total, breakdown in (
        ("expA_tritonswmm", 3 * 1024 * 1024, {"sims": 2 * 1024 * 1024, "_status": 1024 * 1024}),
        ("expA_triton", 1 * 1024 * 1024, {"sims": 1024 * 1024}),
    ):
        write_du_sentinel(
            crates / eid / "_status" / "_du.json",
            disk_utilization_bytes=total,
            sub_path_breakdown=breakdown,
            scope="analysis",
            walk_errors=0,
        )

    out = analysis_dir / "plots" / "cross_experiment" / "disk_utilization.html"
    out.parent.mkdir(parents=True)
    cross_experiment_disk_utilization.render(_stub_analysis(analysis_dir), report_cfg=None, output_path=out)

    html = out.read_text()
    # Spec DU: the hand-rolled <table class='du-table'> is gone — the cross-experiment
    # disk-utilization table is now a Tabulator document, so the structural equivalent
    # is its container id. The heading literal is carried through body_heading_html.
    assert 'id="cross-experiment-disk-utilization"' in html and "Cross-Experiment Disk Utilization" in html
    assert "MiB" in html and "sims" in html and "_status" in html

    manifest = json.loads((out.parent / "disk_utilization.manifest.json").read_text())
    srcs = set(manifest["source_paths_relative"])
    assert any("expA_tritonswmm" in s and s.endswith("_du.json") for s in srcs), (
        f"Q7b: the coupled child's _du.json must be a declared source; got {sorted(srcs)}"
    )
    assert any("expA_triton" in s and s.endswith("_du.json") for s in srcs), (
        f"Q7b: the pure-TRITON child's _du.json must be a declared source; got {sorted(srcs)}"
    )


def test_cross_experiment_disk_utilization_honest_empty_state(tmp_path):
    """No child crates -> honest empty-state note + the expected dir declared as a source
    (ADR-6 D3 non-empty-source gate), never a crash."""
    analysis_dir = tmp_path / "combined"
    analysis_dir.mkdir()
    out = analysis_dir / "plots" / "cross_experiment" / "disk_utilization.html"
    out.parent.mkdir(parents=True)
    cross_experiment_disk_utilization.render(_stub_analysis(analysis_dir), report_cfg=None, output_path=out)
    html = out.read_text()
    assert "No child crates recorded" in html
    manifest = json.loads((out.parent / "disk_utilization.manifest.json").read_text())
    assert manifest["source_paths_relative"], "Q7b: empty-state must still declare a source (ADR-6 D3)"
