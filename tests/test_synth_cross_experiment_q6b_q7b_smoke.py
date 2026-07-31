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
    assert "No paired compute-configs" in out.read_text()


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
    assert "du-table" in html and "Cross-Experiment Disk Utilization" in html
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
