"""Unit tests for SnakemakeWorkflowBuilder._build_resource_block GPU branching.

Guards the gres-mode multi-GPU duplication fix (D2): only gres-mode N>=2 GPU
rules route through the executor's mpi/--ntasks path (mpi=True + tasks=N +
tasks_per_gpu=0). CPU, single-GPU gres, and Frontier gpus-mode emission is
byte-identical to the pre-fix behavior.
"""

import pytest


def _block(builder, **kw):
    """Call _build_resource_block with sane HPC defaults, overridable by kw."""
    defaults = dict(
        partition="standard",
        runtime_min=60,
        mem_mb=16000,
        nodes=1,
        tasks=1,
        cpus_per_task=4,
    )
    defaults.update(kw)
    return builder._build_resource_block(**defaults)


def test_gres_multi_gpu_routes_through_mpi_ntasks_path(synth_multi_sim_builder):
    """gres-mode N>=2: mpi=True, tasks=N, tasks_per_gpu=0, gres=gpu:hw:N; NO bare gpu=."""
    b = synth_multi_sim_builder._workflow_builder
    block = _block(
        b, tasks=4, gpus_total=2, gpus_per_node_config=8,
        gpu_hardware="a6000", gpu_alloc_mode="gres", mpi=False,
    )
    assert "mpi=True" in block
    assert "tasks=2," in block            # one task per requested GPU (not the MPI tasks=4)
    assert "tasks_per_gpu=0" in block
    assert 'gres="gpu:a6000:2"' in block
    # Strict subset (gpus_total=2 < gpus_per_node_config=8): NO --exclusive (partial-node).
    assert "slurm_extra" not in block
    # gres mode never emits the bare `gpu=N` resource line (gpus-mode only).
    # Match the resource LINE (newline + indent + `gpu=`) so this does not
    # collide with the `tasks_per_gpu=0` substring.
    assert "\n        gpu=" not in block


def test_gres_full_node_gpu_keeps_exclusive(synth_multi_sim_builder):
    """Full-node gres (gpus_total >= gpus_per_node_config): --exclusive IS emitted."""
    b = synth_multi_sim_builder._workflow_builder
    block = _block(
        b, tasks=8, gpus_total=8, gpus_per_node_config=8,
        gpu_hardware="a6000", gpu_alloc_mode="gres", mpi=False,
    )
    assert "mpi=True" in block
    assert "tasks=8," in block
    assert "tasks_per_gpu=0" in block
    assert 'gres="gpu:a6000:8"' in block
    assert 'slurm_extra="--exclusive"' in block


def test_gres_subset_4gpu_no_exclusive(synth_multi_sim_builder):
    """Strict subset (gpus_total=4 < gpus_per_node_config=8): NO --exclusive, task-triple intact."""
    b = synth_multi_sim_builder._workflow_builder
    block = _block(
        b, tasks=4, gpus_total=4, gpus_per_node_config=8,
        gpu_hardware="a6000", gpu_alloc_mode="gres", mpi=False,
    )
    assert "mpi=True" in block
    assert "tasks=4," in block
    assert "tasks_per_gpu=0" in block
    assert 'gres="gpu:a6000:4"' in block
    assert "slurm_extra" not in block


def test_single_gpu_gres_unchanged(synth_multi_sim_builder):
    """single-GPU gres: tasks=1, NO mpi, NO tasks_per_gpu — byte-identical to pre-fix."""
    b = synth_multi_sim_builder._workflow_builder
    block = _block(
        b, gpus_total=1, gpus_per_node_config=8,
        gpu_hardware="a6000", gpu_alloc_mode="gres", mpi=False,
    )
    assert "tasks=1," in block
    assert "mpi=True" not in block
    assert "tasks_per_gpu" not in block
    assert 'gres="gpu:a6000:1"' in block
    assert "slurm_extra" not in block


def test_frontier_gpus_mode_unchanged(synth_multi_sim_builder):
    """Frontier gpus-mode multi-GPU: tasks=1, gpu=N + gpu_model; NO mpi/tasks_per_gpu/gres."""
    b = synth_multi_sim_builder._workflow_builder
    block = _block(
        b, gpus_total=2, gpus_per_node_config=8,
        gpu_hardware="a100", gpu_alloc_mode="gpus", mpi=False,
    )
    assert "tasks=1," in block
    assert "gpu=2" in block
    assert 'gpu_model="a100"' in block
    assert "mpi=True" not in block
    assert "tasks_per_gpu" not in block
    assert "gres=" not in block
    assert "slurm_extra" not in block


def test_cpu_mpi_unchanged(synth_multi_sim_builder):
    """CPU MPI job: tasks=<rank count>, mpi=True (when mpi arg set); NO gpu/gres/tasks_per_gpu."""
    b = synth_multi_sim_builder._workflow_builder
    block = _block(b, tasks=4, gpus_total=0, mpi=True)
    assert "tasks=4," in block
    assert "mpi=True" in block
    assert "tasks_per_gpu" not in block
    assert "gpu" not in block and "gres" not in block
    assert "slurm_extra" not in block


def test_cpu_non_mpi_unchanged(synth_multi_sim_builder):
    """CPU non-MPI job: tasks=<count>, NO mpi/gpu/gres/tasks_per_gpu."""
    b = synth_multi_sim_builder._workflow_builder
    block = _block(b, tasks=2, gpus_total=0, mpi=False)
    assert "tasks=2," in block
    assert "mpi=True" not in block
    assert "tasks_per_gpu" not in block
    assert "slurm_extra" not in block


def test_resolution_helpers_read_hpc_system_config(synth_multi_sim_builder):
    """Phase 4 (4a): with cfg_hpc_system present, the resolution helpers read the
    new per-HPC-system config (default_account + per-partition topology) instead
    of the legacy cfg_analysis/cfg_system reads.

    This is the byte-identity foundation for 4c/4d: those phases delete the
    None-fallback branch of each helper, so every Snakefile-generating path must
    resolve through a non-None cfg_hpc_system first. The (4a-additive) helpers
    _resolve_gpu_hardware / _resolve_cpus_per_node / _resolve_additional_modules
    are exercised here too. A fresh builder is constructed and its cfg_hpc_system
    is overridden locally so the session-scoped fixture's other consumers (the
    GPU-branch tests above, which call _build_resource_block directly) are
    unaffected.
    """
    from pathlib import Path

    from TRITON_SWMM_toolkit.config.loaders import load_hpc_system_config
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    example = Path(__file__).parent / "fixtures" / "hpc_system_config_test.yaml"
    b = SnakemakeWorkflowBuilder(synth_multi_sim_builder)
    b.cfg_hpc_system = load_hpc_system_config(example)

    # default_account is read from the new config (a distinctive value the legacy
    # null hpc_account could not produce — proving the cfg_hpc_system branch fired).
    assert b._resolve_account() == b.cfg_hpc_system.default_account == "test_alloc"
    # Per-partition topology is read from the named PartitionSpec.
    assert b._resolve_gpus_per_node("test_partition") == 8
    assert b._resolve_cpus_per_node("test_partition") == 40
    assert b._resolve_gpu_hardware("test_partition") == "a6000"
    # gpu_allocation_flavor + additional_modules are unset in the example config,
    # gpu_allocation_flavor + additional_modules are unset in the example config.
    # 4c removed the legacy cfg_system fallbacks, so these resolve to the no-config
    # defaults: "gpus" for the alloc mode, None for the module string.
    assert b._resolve_gpu_alloc_mode() == "gpus"
    assert b._resolve_additional_modules() is None
