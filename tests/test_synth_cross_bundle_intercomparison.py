"""Focused unit tests for the combine-first cross-bundle intercomparison writer (Phase 5, S1).

Exercises ``bundle/_combine._write_combined_intercomparison`` and its helpers over SYNTHETIC
two-bundle fixtures (one all-clean, one all-resume single-arm master), with NO HPC / compile /
real analysis. Validates the North-Star derivation: classify each bundle clean/resume from its
bundled ``scenario_status.csv`` ``n_resumes``, pair each compute-config present in BOTH by the
run_mode/n_mpi/n_omp/n_gpus/n_nodes/partition identity, compare the key-result summaries
(``max_wlevel_m`` + ``max_flow_cms``) per event via ``compare_variable_exact``, and write real
``pairs`` into ``combined_intercomparison.json`` — NOT harvested from any per-arm verdict.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from hhemt.bundle import _combine


def _sub_nodes(sa_id: str, attrs: dict, wlevel: np.ndarray, flow: np.ndarray) -> dict:
    """One /sa_{id} node (carrying compute-config attrs) + its triton/swmm_link children,
    mirroring the as-built sensitivity_datatree.zarr shape read by _config_diff._load_subs."""
    return {
        sa_id: xr.Dataset(attrs=attrs),
        f"{sa_id}/tritonswmm/triton": xr.Dataset(
            {"max_wlevel_m": (("event_iloc", "y", "x"), wlevel)},
            coords={"event_iloc": [0], "y": [0, 1], "x": [0, 1]},
        ),
        f"{sa_id}/tritonswmm/swmm_link": xr.Dataset(
            {"max_flow_cms": (("event_iloc", "link_id"), flow)},
            coords={"event_iloc": [0], "link_id": ["L0", "L1"]},
        ),
    }


def _write_bundle(root, *, n_resumes: int, sa1_wlevel: np.ndarray, sa1_flow: np.ndarray) -> None:
    """Build a minimal single-arm bundle dir: scenario_status.csv (n_resumes) +
    sensitivity_datatree.zarr with two compute-configs (sa_0 serial, sa_1 mpi-8)."""
    root.mkdir(parents=True, exist_ok=True)
    # scenario_status.csv — the role source (clean iff every n_resumes == 0).
    lines = ["sa_id,n_resumes", f"sa_0,{n_resumes}", f"sa_1,{n_resumes}"]
    (root / "scenario_status.csv").write_text("\n".join(lines) + "\n")

    serial_attrs = {
        "sa_id": "sa_0",
        "run_mode": "serial",
        "n_mpi_procs": 1,
        "n_omp_threads": 1,
        "n_gpus": 0,
        "n_nodes": 1,
        "hpc.partition": "standard",
    }
    mpi_attrs = {
        "sa_id": "sa_1",
        "run_mode": "mpi",
        "n_mpi_procs": 8,
        "n_omp_threads": 1,
        "n_gpus": 0,
        "n_nodes": 1,
        "hpc.partition": "standard",
    }
    # sa_0 is BYTE-IDENTICAL clean-vs-resume; sa_1 differs by the caller-supplied arrays.
    sa0_wlevel = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    sa0_flow = np.array([[10.0, 20.0]])
    nodes: dict = {}
    nodes.update(_sub_nodes("sa_0", serial_attrs, sa0_wlevel, sa0_flow))
    nodes.update(_sub_nodes("sa_1", mpi_attrs, sa1_wlevel, sa1_flow))
    dt = xr.DataTree.from_dict(nodes)
    dt.to_zarr(root / "sensitivity_datatree.zarr", consolidated=False)


@pytest.fixture
def _stub_experiment_ids(monkeypatch):
    """Avoid the Bundle.from_directory / cfg_analysis.yaml requirement of the real
    _combined_experiment_ids — the writer's identity source is orthogonal to this test."""
    monkeypatch.setattr(
        _combine,
        "_combined_experiment_ids",
        lambda roots: [r.name for r in roots],
    )


def test_bundle_role_from_status(tmp_path):
    clean, resume = tmp_path / "clean", tmp_path / "resume"
    _write_bundle(clean, n_resumes=0, sa1_wlevel=np.zeros((1, 2, 2)), sa1_flow=np.zeros((1, 2)))
    _write_bundle(resume, n_resumes=2, sa1_wlevel=np.zeros((1, 2, 2)), sa1_flow=np.zeros((1, 2)))
    assert _combine._bundle_role_from_status(clean) == "clean"
    assert _combine._bundle_role_from_status(resume) == "resume"
    assert _combine._bundle_role_from_status(tmp_path / "missing") == "clean"  # absent -> clean


def test_cross_bundle_intercomparison_pairs(tmp_path, _stub_experiment_ids):
    clean, resume, out = tmp_path / "clean", tmp_path / "resume", tmp_path / "combined"
    out.mkdir()
    # clean sa_1 vs resume sa_1: max_wlevel_m differs by 0.5 at one cell; flow identical.
    clean_w = np.array([[[5.0, 6.0], [7.0, 8.0]]])
    resume_w = np.array([[[5.0, 6.0], [7.0, 8.5]]])  # +0.5 at [1,1]
    flow = np.array([[30.0, 40.0]])
    _write_bundle(clean, n_resumes=0, sa1_wlevel=clean_w, sa1_flow=flow)
    _write_bundle(resume, n_resumes=1, sa1_wlevel=resume_w, sa1_flow=flow)

    _combine._write_combined_intercomparison(out, [clean, resume])
    payload = json.loads((out / "combined_intercomparison.json").read_text())

    # experiments: one clean, one resume, sorted by experiment id.
    roles = {e["experiment"]: e["role"] for e in payload["experiments"]}
    assert roles == {"clean": "clean", "resume": "resume"}

    pairs = payload["pairs"]
    assert pairs, "cross-bundle pairing must find matching compute-configs in BOTH bundles"
    by_key = {(p["config"], p["variable"]): p for p in pairs}
    # Both compute-configs (serial sa_0, mpi-8 sa_1) pair on identity; both key-result vars present.
    serial_key = "run_mode=serial|n_mpi=1|n_omp=1|n_gpus=0|n_nodes=1|partition=standard"
    mpi_key = "run_mode=mpi|n_mpi=8|n_omp=1|n_gpus=0|n_nodes=1|partition=standard"
    assert (serial_key, "max_wlevel_m") in by_key
    assert (mpi_key, "max_wlevel_m") in by_key
    # sa_0 (serial) is byte-identical across arms.
    assert by_key[(serial_key, "max_wlevel_m")]["identical"] is True
    assert by_key[(serial_key, "max_wlevel_m")]["max_abs_diff"] == 0.0
    # sa_1 (mpi-8) depth differs by 0.5; flow identical.
    assert by_key[(mpi_key, "max_wlevel_m")]["identical"] is False
    assert by_key[(mpi_key, "max_wlevel_m")]["max_abs_diff"] == pytest.approx(0.5)
    assert by_key[(mpi_key, "max_flow_cms")]["identical"] is True


def test_cross_bundle_intercomparison_empty_when_not_clean_resume(tmp_path, _stub_experiment_ids):
    """Two clean bundles (no resume arm) -> honest empty pairs, no crash."""
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "combined"
    out.mkdir()
    _write_bundle(a, n_resumes=0, sa1_wlevel=np.zeros((1, 2, 2)), sa1_flow=np.zeros((1, 2)))
    _write_bundle(b, n_resumes=0, sa1_wlevel=np.zeros((1, 2, 2)), sa1_flow=np.zeros((1, 2)))
    _combine._write_combined_intercomparison(out, [a, b])
    payload = json.loads((out / "combined_intercomparison.json").read_text())
    assert payload["pairs"] == []
    assert all(e["role"] == "clean" for e in payload["experiments"])
