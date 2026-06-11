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
