"""Pure-function unit tests for the Phase-2 sensitivity + magnitude EDA members.

Covers (per the phase DoD / R7-R10):

- ``compute_magnitude`` on synthetic 2-D depth grids with hand-computed
  max_abs / rmse / union-extent / τ-restricted %-diff (Type-8 p95) / newly-wet;
- each member returns an ``EdaResult`` whose ``verdict`` is a ``CheckResult``;
- the members read the FLAT per-scenario summaries via
  ``sub.process._retrieve_combined_output`` — NEVER the consolidated tree (the fake
  ``process`` here implements only the flat read; a consolidated-tree read would
  ``AttributeError``).

No real compile/run is exercised — these are pure/fake-fixture tests, HPC-free.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from hhemt.analysis_validation import CheckResult
from hhemt.eda._result import EdaResult
from hhemt.eda.compute_sensitivity import (
    check_cross_hardware_magnitude,
    check_rank_sensitivity,
    check_resume_sensitivity,
    compute_magnitude,
)

# --------------------------------------------------------------------------- #
# compute_magnitude — pure kernel                                             #
# --------------------------------------------------------------------------- #

# A 3x3 fixture pair with a hand-tractable structure. [2,0] is nodata (NaN) in the
# base and excluded from every reduction (exclusion, never nan-substitution-to-0).
_BASE = np.array([[0.00, 0.10, 0.20], [0.50, 1.00, 0.00], [np.nan, 0.04, 0.02]], dtype="float64")
_TEST = np.array([[0.00, 0.10, 0.25], [0.60, 1.00, 0.05], [np.nan, 0.04, 0.10]], dtype="float64")


def test_compute_magnitude_known_values():
    m = compute_magnitude(_BASE, _TEST, dry_threshold_m=0.0025, tau_m=0.03)

    # Union wetted footprint = every domain cell wet in either run except [0,0]
    # (0.0 in both) → 7 cells; only [1,2] flips wet-state (dry base, wet test).
    assert m["n_wet_union"] == 7
    assert m["n_extent_disagree"] == 1
    assert m["frac_extent_disagree"] == pytest.approx(1 / 7)

    # abs metrics over diff_wet = [0, 0.05, 0.10, 0, 0.05, 0, 0.08].
    diff_wet = np.array([0.0, 0.05, 0.10, 0.0, 0.05, 0.0, 0.08])
    assert m["max_abs_diff_m"] == pytest.approx(0.10)
    assert m["rmse_wetted_m"] == pytest.approx(float(np.sqrt(np.mean(diff_wet**2))))
    assert m["p95_abs_diff_m"] == pytest.approx(float(np.percentile(diff_wet, 95)))

    # τ-restricted %-diff over baseline-wet cells (base >= 0.03): 5 cells.
    assert m["n_baseline_wet"] == 5
    abs_pct = np.array([0.0, 25.0, 20.0, 0.0, 0.0])  # |(test-base)/base*100|
    assert m["pct_diff_p95"] == pytest.approx(float(np.nanpercentile(abs_pct, 95, method="median_unbiased")))
    assert m["pct_diff_median_signed"] == pytest.approx(0.0)

    # dry->wet newly-flooded: [1,2] and [2,2] cross from <τ to >=τ.
    assert m["n_newly_wet"] == 2

    # Disclosed metadata (D-MAG: method + both thresholds recorded).
    assert m["pct_p95_method"] == "median_unbiased"
    assert m["tau_m"] == pytest.approx(0.03)
    assert m["dry_threshold_m"] == pytest.approx(0.0025)


def test_compute_magnitude_shape_mismatch_raises():
    with pytest.raises(ValueError, match="grid shape mismatch"):
        compute_magnitude(np.zeros((2, 2)), np.zeros((3, 3)))


def test_compute_magnitude_all_dry_returns_zeros():
    dry = np.zeros((4, 4), dtype="float64")  # all below dry_threshold and τ
    m = compute_magnitude(dry, dry, dry_threshold_m=0.0025, tau_m=0.03)
    assert m["max_abs_diff_m"] == 0.0
    assert m["rmse_wetted_m"] == 0.0
    assert m["n_wet_union"] == 0
    assert m["n_baseline_wet"] == 0
    assert m["n_newly_wet"] == 0
    assert m["pct_diff_p95"] == 0.0


def test_compute_magnitude_tau_floor_excludes_below_tau():
    # A cell at 0.01 m (below τ=0.03) must NOT enter the %-diff sample even though a
    # large relative change there would blow up an ε-padded denominator.
    base = np.array([[0.01, 0.50]], dtype="float64")
    test = np.array([[0.05, 0.55]], dtype="float64")
    m = compute_magnitude(base, test, dry_threshold_m=0.0025, tau_m=0.03)
    # Only the 0.50 cell qualifies (base >= τ); its %-diff = 10.0.
    assert m["n_baseline_wet"] == 1
    assert m["pct_diff_p95"] == pytest.approx(10.0)
    # The 0.01→0.05 cell is a newly-wet dry→wet crossing, not a %-diff contributor.
    assert m["n_newly_wet"] == 1


def test_compute_magnitude_type8_method_is_used():
    # A skewed sample where Type-8 (median_unbiased) and Type-7 (linear) diverge at
    # p95, proving the method arg is honored rather than the numpy default.
    base = np.full((1, 20), 1.0, dtype="float64")
    # test depths chosen so (test-base)/base*100 spans a skewed 0..100 range.
    pct_targets = np.linspace(0.0, 100.0, 20)
    test = base * (1.0 + pct_targets / 100.0)
    m = compute_magnitude(base, test, dry_threshold_m=0.0025, tau_m=0.03)
    expected_t8 = float(np.nanpercentile(np.abs(pct_targets), 95, method="median_unbiased"))
    linear_t7 = float(np.nanpercentile(np.abs(pct_targets), 95, method="linear"))
    assert m["pct_diff_p95"] == pytest.approx(expected_t8)
    assert expected_t8 != pytest.approx(linear_t7)  # the two conventions genuinely differ here


# --------------------------------------------------------------------------- #
# Members — skip path on a non-sensitivity analysis                            #
# --------------------------------------------------------------------------- #

_NON_SENS_MASTER = SimpleNamespace(cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=False))


@pytest.mark.parametrize("member", [check_rank_sensitivity, check_resume_sensitivity, check_cross_hardware_magnitude])
def test_member_skips_non_sensitivity(member):
    res = member(_NON_SENS_MASTER, cfg_analysis=SimpleNamespace(), eda_cfg=SimpleNamespace())
    assert isinstance(res, EdaResult)
    assert res.skipped is True
    assert isinstance(res.verdict, CheckResult)
    assert res.verdict.passed is True
    assert res.artifact_path is None


# --------------------------------------------------------------------------- #
# Members — flat-read comparison on a fake sensitivity master                   #
# --------------------------------------------------------------------------- #


def _depth_ds(grid: np.ndarray) -> xr.Dataset:
    """A minimal flat-summary Dataset: max_wlevel_m over one event_iloc."""
    ny, nx = grid.shape
    return xr.Dataset(
        {"max_wlevel_m": (("event_iloc", "y", "x"), grid[None, :, :])},
        coords={"event_iloc": [0], "y": np.arange(ny), "x": np.arange(nx)},
    )


class _FakeProcess:
    """Implements ONLY the flat read (``_retrieve_combined_output`` + ``_MODE_CONFIG``).

    Deliberately has no ``open_datatree`` — a member that reached for the consolidated
    tree would ``AttributeError``, so a passing test proves the flat-read contract (R7).
    """

    def __init__(self, grid: np.ndarray, calls: list[str]):
        self._grid = grid
        self._MODE_CONFIG = {"max_depth": object()}
        self._calls = calls

    def _retrieve_combined_output(self, mode: str) -> xr.Dataset:
        self._calls.append(mode)
        if mode == "max_depth":
            return _depth_ds(self._grid)
        raise FileNotFoundError(mode)


def _fake_sub(analysis_dir, grid, *, run_mode, n_mpi, partition, n_gpus=0, calls=None):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    # Real .zarr store so the emit provenance gate (_validate_source_path) passes.
    xr.Dataset({"placeholder": (("a",), [1])}).to_zarr(analysis_dir / "analysis_datatree.zarr", mode="w")
    return SimpleNamespace(
        process=_FakeProcess(grid, calls if calls is not None else []),
        cfg_analysis=SimpleNamespace(
            run_mode=run_mode,
            n_mpi_procs=n_mpi,
            n_omp_threads=1,
            n_gpus=n_gpus,
            n_nodes=1,
            hpc_ensemble_partition=partition,
        ),
        analysis_paths=SimpleNamespace(analysis_dir=str(analysis_dir)),
    )


def _fake_master(tmp_path, subs: dict):
    master_dir = tmp_path / "master"
    master_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=True),
        sensitivity=SimpleNamespace(sub_analyses=subs),
        analysis_paths=SimpleNamespace(analysis_dir=str(master_dir)),
    )


def test_rank_sensitivity_passes_on_identical_flat_summaries(tmp_path):
    grid = np.array([[0.0, 0.5], [1.0, 0.04]], dtype="float64")
    calls: list[str] = []
    subs = {
        "mpi_9_r0": _fake_sub(tmp_path / "sa1", grid, run_mode="mpi", n_mpi=1, partition="standard", calls=calls),
        "mpi_10_r0": _fake_sub(
            tmp_path / "sa2", grid.copy(), run_mode="mpi", n_mpi=2, partition="standard", calls=calls
        ),
    }
    master = _fake_master(tmp_path, subs)

    res = check_rank_sensitivity(master, cfg_analysis=SimpleNamespace(), eda_cfg=SimpleNamespace())

    assert isinstance(res, EdaResult)
    assert res.skipped is False
    assert isinstance(res.verdict, CheckResult)
    assert res.verdict.passed is True
    assert res.plot_id == "eda_rank_sensitivity"
    # Artifact + verdict.json persisted under {master}/eda/.
    assert res.artifact_path.exists()
    verdict_json = res.artifact_path.parent / f"{res.plot_id}.verdict.json"
    assert verdict_json.exists()
    assert json.loads(verdict_json.read_text())["passed"] is True
    # The member read via the FLAT path (proves R7 — the fake has no consolidated tree).
    assert "max_depth" in calls


def test_rank_sensitivity_fails_on_divergent_flat_summaries(tmp_path):
    ref = np.array([[0.0, 0.5], [1.0, 0.04]], dtype="float64")
    div = ref.copy()
    div[1, 0] = 1.25  # rank-2 diverges from the rank-1 reference at one cell
    subs = {
        "mpi_9_r0": _fake_sub(tmp_path / "sa1", ref, run_mode="mpi", n_mpi=1, partition="standard"),
        "mpi_10_r0": _fake_sub(tmp_path / "sa2", div, run_mode="mpi", n_mpi=2, partition="standard"),
    }
    master = _fake_master(tmp_path, subs)

    res = check_rank_sensitivity(master, cfg_analysis=SimpleNamespace(), eda_cfg=SimpleNamespace())

    assert res.skipped is False
    assert res.verdict.passed is False
    assert any(d.get("variable") == "max_wlevel_m" for d in res.verdict.details)


def test_rank_sensitivity_skips_family_without_rank1_reference(tmp_path):
    # Only rank-2 and rank-4 present in the family → no n_mpi_procs==1 anchor (R10):
    # disclosed skip, NEVER a fall-back to a cross-run_mode global anchor.
    grid = np.array([[0.0, 0.5], [1.0, 0.04]], dtype="float64")
    subs = {
        "mpi_10_r0": _fake_sub(tmp_path / "sa2", grid, run_mode="mpi", n_mpi=2, partition="standard"),
        "mpi_11_r0": _fake_sub(tmp_path / "sa3", grid.copy(), run_mode="mpi", n_mpi=4, partition="standard"),
    }
    master = _fake_master(tmp_path, subs)

    res = check_rank_sensitivity(master, cfg_analysis=SimpleNamespace(), eda_cfg=SimpleNamespace())
    assert res.skipped is True
    assert isinstance(res.verdict, CheckResult)


def test_cross_hardware_reports_characterized_divergence(tmp_path):
    # 1-GPU vs 1-rank serial-CPU: passed=True regardless of divergence magnitude
    # (ADR-4 — the disclosed bounds ARE the verdict).
    cpu = np.array([[0.0, 0.5], [1.0, 0.04]], dtype="float64")
    gpu = cpu.copy()
    gpu[0, 1] = 0.55  # a real cross-hardware divergence
    subs = {
        "serial_0_r0": _fake_sub(tmp_path / "cpu", cpu, run_mode="serial", n_mpi=1, partition="standard"),
        "gpu_1_r0": _fake_sub(tmp_path / "gpu", gpu, run_mode="gpu", n_mpi=1, partition="gpu-a6000", n_gpus=1),
    }
    master = _fake_master(tmp_path, subs)

    res = check_cross_hardware_magnitude(master, cfg_analysis=SimpleNamespace(), eda_cfg=SimpleNamespace())
    assert res.skipped is False
    assert res.verdict.passed is True  # characterized-divergence, not an equality claim
    assert res.plot_id == "eda_cross_hardware_magnitude"
    assert "Characterized cross-hardware divergence" in res.verdict.summary


def test_resume_sensitivity_pairs_clean_and_resume_by_config(tmp_path, monkeypatch):
    # Same compute-config, one clean (n_resumes=0) + one resume (n_resumes=1). A
    # resumed sim must reproduce its clean counterpart bit-for-bit.
    grid = np.array([[0.0, 0.5], [1.0, 0.04]], dtype="float64")
    subs = {
        "serial_0_r0": _fake_sub(tmp_path / "clean", grid, run_mode="serial", n_mpi=1, partition="standard"),
        "serial_0_r1": _fake_sub(tmp_path / "resume", grid.copy(), run_mode="serial", n_mpi=1, partition="standard"),
    }
    master = _fake_master(tmp_path, subs)

    # df_status carries n_resumes per sa_id (R9): r1 was resumed once.
    import pandas as pd

    master.df_status = pd.DataFrame(
        {
            "sa_id": ["serial_0_r0", "serial_0_r1"],
            "event_iloc": [0, 0],
            "n_resumes": [0, 1],
        }
    )

    res = check_resume_sensitivity(master, cfg_analysis=SimpleNamespace(), eda_cfg=SimpleNamespace())
    assert res.skipped is False
    assert res.verdict.passed is True
    assert res.plot_id == "eda_resume_sensitivity"
