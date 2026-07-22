"""Phase 1: the DEM-resolution matrix builder + the R14 coupling-node pre-check."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from hhemt.config.synthetic_experiment import synthetic_experiment_config
from hhemt.exceptions import ProcessingError
from hhemt.synthetic_experiment import (
    assert_coupling_nodes_distinct,
    dem_resolution_matrix_rows,
)


def _write_hpc_yaml(path: Path, *, max_gpu: int = 8) -> Path:
    """A minimal per-cluster hpc_system_config YAML (the config's required
    hpc_system_config_yaml). Mirrors tests/test_synth_experiment_framework.py."""
    path.write_text(
        textwrap.dedent(
            f"""\
            system_name: test_synth_cluster
            partitions:
              gpu-a6000:
                max_runtime: 4320
                gpus_per_node: 8
                cpus_per_node: 40
                max_gpu: {max_gpu}
                gpu_hardware: a6000
                gpu_compilation_backend: CUDA
              gpu-a100-80:
                max_runtime: 4320
                gpus_per_node: 8
                cpus_per_node: 40
                max_gpu: {max_gpu}
                gpu_hardware: a100
                gpu_compilation_backend: CUDA
              standard:
                max_runtime: 4320
                cpus_per_node: 40
            """
        ),
        encoding="utf-8",
    )
    return path


def _cfg(hpc_yaml: Path, ladder: tuple[float, ...], **overrides) -> synthetic_experiment_config:
    """A synthetic_experiment_config carrying ``ladder`` at the shipped grid defaults
    unless an ``overrides`` key (n_cols/n_rows/rank_sweep) says otherwise. Runs every
    config validator, so a bad ``ladder`` raises here."""
    return synthetic_experiment_config(
        hpc_system_config_yaml=hpc_yaml,
        ensemble_partition="gpu-a6000",
        setup_partition="standard",
        dem_resolution_ladder=ladder,
        **overrides,
    )


def test_ladder_rejects_non_divisor(tmp_path):
    """A non-divisor rung inflates area-integrated metrics +7% to +30% (measured), and
    every run completes, so the error is silent. Reject at config load."""
    with pytest.raises(ValueError, match="not an integer divisor"):
        _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (3.5, 7.0, 15.0))


def test_ladder_rejects_non_constant_ratio(tmp_path):
    """(3.5, 7.0, 28.0) is divisor-valid at every rung (clears the divisor gate) but
    its ratios are 2 then 4 -- the constant-ratio gate must reject it. (3.5,7,21 would
    fail the divisor gate first and never reach the ratio check.)"""
    with pytest.raises(ValueError, match="CONSTANT refinement ratio"):
        _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (3.5, 7.0, 28.0))


def test_ladder_rejects_coarsest_first(tmp_path):
    with pytest.raises(ValueError, match="FINEST FIRST"):
        _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (14.0, 7.0, 3.5))


def test_rows_carry_the_resolution_and_a_fixed_compute_config(tmp_path):
    cfg = _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (3.5, 7.0, 14.0))
    rows = dem_resolution_matrix_rows(cfg, replicates=2)
    assert len(rows) == 3 * 2
    assert {r["system.target_dem_resolution"] for r in rows} == {3.5, 7.0, 14.0}
    # The control: ONE compute config across the whole sweep.
    assert {(r["run_mode"], r["n_mpi_procs"], r["n_omp_threads"], r["n_gpus"]) for r in rows} == {("serial", 1, 1, 0)}
    assert len({r["sa_id"] for r in rows}) == len(rows)
    assert all(r["sa_id"].startswith("dem_") for r in rows)


def test_coupling_precheck_returns_retained_counts_at_a_safe_ladder(tmp_path):
    """The DoD's canonical divisor-valid ladder 3.5/7/14 must PASS on the default
    64x120 grid -- every rung keeps all _N_COUPLING_NODES in-line coupling nodes in
    distinct DEM cells. (Iterating all of ``_nodes`` instead of the in-line set would
    wrongly reject 14 m, where the disconnected dummy_outfall bins into J1's cell.)"""
    cfg = _cfg(_write_hpc_yaml(tmp_path / "hpc.yaml"), (3.5, 7.0, 14.0))
    retained = assert_coupling_nodes_distinct(cfg)
    assert set(retained) == {3.5, 7.0, 14.0}
    assert retained == {3.5: 8, 7.0: 8, 14.0: 8}
    # Monotone: coarsening can only merge cells, never split them.
    assert retained[7.0] <= retained[3.5]
    assert retained[14.0] <= retained[7.0]


def test_coupling_precheck_raises_on_a_colliding_rung(tmp_path):
    """The deadlock this converts into a plan-time error is a HANG, not a crash -- the
    worst failure shape on HPC -- so a coarse-enough rung must fail LOUDLY here. A 32x32
    grid packs the 8 in-line nodes tightly enough that the divisor-valid 14 m rung merges
    two of them (J5+J6); the default grid's first in-line collision (56 m) is a non-divisor
    of the y-extent and would be rejected by the ladder validator before the guard runs.
    rank_sweep=(2,) keeps the small grid valid against the coupling invariant."""
    cfg = _cfg(
        _write_hpc_yaml(tmp_path / "hpc.yaml"),
        (3.5, 7.0, 14.0),
        n_cols=32,
        n_rows=32,
        rank_sweep=(2,),
    )
    with pytest.raises(ProcessingError, match="share DEM cell"):
        assert_coupling_nodes_distinct(cfg)
