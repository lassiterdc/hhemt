"""C2/G3: the config_diff panel SET must be identical across model arms.

Byte-identity grouping is model-DEPENDENT — measured, the coupled arm's identity partition
splits along CPU-vs-GPU while the pure-TRITON arm holds one group straddling Serial, OpenMP,
MPI, Hybrid and GPU. Same figure, same name, two arms, structurally different panels: a direct
G3 violation. The grouping key must therefore be a pure function of the COMPUTE attrs.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import hhemt.eda._config_diff as cd

_CONFIGS = [
    ("sa_serial_1", "serial", {"run_mode": "serial", "n_mpi_procs": 1, "n_omp_threads": 1, "n_gpus": 0}),
    ("sa_openmp_2", "openmp", {"run_mode": "openmp", "n_mpi_procs": 1, "n_omp_threads": 2, "n_gpus": 0}),
    ("sa_openmp_8", "openmp", {"run_mode": "openmp", "n_mpi_procs": 1, "n_omp_threads": 8, "n_gpus": 0}),
    ("sa_mpi_2", "mpi", {"run_mode": "mpi", "n_mpi_procs": 2, "n_omp_threads": 1, "n_gpus": 0}),
    ("sa_mpi_8", "mpi", {"run_mode": "mpi", "n_mpi_procs": 8, "n_omp_threads": 1, "n_gpus": 0}),
    ("sa_hybrid_4", "hybrid", {"run_mode": "hybrid", "n_mpi_procs": 2, "n_omp_threads": 2, "n_gpus": 0}),
    ("sa_gpu_a6000_1", "gpu", {"run_mode": "gpu", "n_gpus": 1, "hpc.partition": "gpu-a6000"}),
    ("sa_gpu_a6000_2", "gpu", {"run_mode": "gpu", "n_gpus": 2, "hpc.partition": "gpu-a6000"}),
    ("sa_gpu_a100_1", "gpu", {"run_mode": "gpu", "n_gpus": 1, "hpc.partition": "gpu-a100-80"}),
]


def _subs():
    out = {}
    for sa_id, run_mode, attrs in _CONFIGS:
        da = xr.DataArray(np.zeros((4, 4)), dims=("y", "x"))
        out[sa_id] = {
            "attrs": attrs,
            "label": cd._derive_config_label(attrs),
            "run_mode": run_mode,
            "n_resumes": 0,
            "wlevel": da,
            "flow": None,
        }
    return out


def _keys(groups):
    """Panel identity = the sorted distinct labels each group holds."""
    return sorted(tuple(sorted(set(g["labels"]))) for g in groups)


def test_grouping_ignores_the_byte_identity_partition(tmp_path, monkeypatch):
    """G3: two arms whose identity partitions DIFFER must still yield the same panel set.

    Pre-fix this fails because _group_by_identity keys on the partition itself, so the
    coupled-arm partition (CPU/GPU split) and the pure-TRITON partition (one straddling
    group) produce structurally different panels for the same compute configs.
    """
    subs = _subs()
    ids = [sa for sa, _, _ in _CONFIGS]

    # Coupled-arm shape: GPU configs mutually identical, CPU configs each distinct.
    coupled = {sa: ("G" if "gpu" in sa else sa) for sa in ids}
    # Pure-TRITON shape: ONE group straddling every run mode.
    pure = dict.fromkeys(ids, "ALL")

    monkeypatch.setattr(cd, "_identity_labels", lambda root: coupled)
    a = _keys(cd._group_by_config_class(subs, tmp_path))
    monkeypatch.setattr(cd, "_identity_labels", lambda root: pure)
    b = _keys(cd._group_by_config_class(subs, tmp_path))

    assert a == b, "panel set differs across model arms -> G3 violation"


def test_gpu_hardware_families_are_separate_panels():
    """'all the GPU groupings' is plural, and a6000 vs a100 is an identity-bearing
    distinction the config label already preserves."""
    groups = cd._group_by_config_class(_subs(), None)
    gpu_panels = [g for g in groups if any(rm == "gpu" for rm in g["run_modes"])]
    assert len(gpu_panels) == 2
    hardware = {cd._gpu_hardware(g["attrs"]) for g in gpu_panels}
    assert hardware == {"a6000", "a100-80"}


def test_group_sizes_are_non_vacuous():
    """'larger groups towards the top' is only meaningful if groups vary in membership."""
    sizes = sorted(len(set(g["labels"])) for g in cd._group_by_config_class(_subs(), None))
    assert max(sizes) > 1, "every group holds one config -> the size-ordering clause is vacuous"


def test_uniform_grid_guard_survives():
    """The cell-wise-subtraction precondition is independent of the grouping axis."""
    subs = _subs()
    subs["sa_mpi_8"]["wlevel"] = xr.DataArray(np.zeros((5, 5)), dims=("y", "x"))
    with pytest.raises(Exception, match="UNIFORM grid"):
        cd._group_by_config_class(subs, None)
