# Plan: Fixing GPU Workflow Resource Handling

**Status:** Draft (Implementation-ready)
**Owner:** Toolkit maintainers
**Created:** 2026-02-11

## Goal

1. Add a **system config** switch that declares whether SLURM GPU allocation
   should use `--gres` or `--gpus`.
2. Centralize GPU-node math inside `_build_resource_block` so callers don’t
   repeat calculations.
3. Ensure Snakemake rule resources and the SLURM profile are built consistently
   with the preferred GPU allocation directive.

---

## Requirements

- New system config: `preferred_slurm_option_for_allocating_gpus` with allowed
  values `"gres" | "gpus"`.
- `_build_resource_block` should perform these calculations internally:

```python
if n_gpus > 0 and gpus_per_node_config < 1:
    raise ValueError("hpc_gpus_per_node must be set when requesting GPUs")
nodes_from_gpu = self._calculate_nodes_for_gpus(n_gpus, gpus_per_node_config)
sim_nodes = max(n_nodes, nodes_from_gpu)
gpus_per_node = math.ceil(n_gpus / sim_nodes) if n_gpus > 0 else 0
```

- Callers should not compute or pass `gpus_per_node` explicitly.
- `generate_snakemake_config()` must update the SLURM profile based on the
  preferred option (`gres` vs `gpus`).

---

## Proposed API Changes

### 1) System Config Field

**File:** `src/TRITON_SWMM_toolkit/config/system.py`

```python
from typing import Optional, Literal
from pydantic import Field

    preferred_slurm_option_for_allocating_gpus: Optional[Literal["gres", "gpus"]] = Field(
        "gres",
        description=(
            "Preferred SLURM GPU allocation directive. "
            "Set to 'gres' to emit --gres=gpu:..., or 'gpus' to emit "
            "--gpus/--gpus-per-node when supported by the cluster."
        ),
    )
```

---

## Core Refactor: `_build_resource_block`

### 2) New Signature

**File:** `src/TRITON_SWMM_toolkit/workflow.py`

```python
def _build_resource_block(
    self,
    partition: str | None,
    runtime_min: int,
    mem_mb: int,
    nodes: int,
    tasks: int,
    cpus_per_task: int,
    gpus_total: int = 0,
    gpus_per_node_config: int = 0,
    gpu_hardware: str | None = None,
    gpu_alloc_mode: Literal["gres", "gpus"] = "gres",
) -> str:
    ...
```

### 3) Centralized GPU Math + Resource Emission

```python
    if partition is None:
        raise ValueError("hpc partition must be set when generating SLURM resources")
    partition_name = partition

    if gpus_total > 0 and gpus_per_node_config < 1:
        raise ValueError("hpc_gpus_per_node must be set when requesting GPUs")

    nodes_from_gpu = self._calculate_nodes_for_gpus(gpus_total, gpus_per_node_config)
    sim_nodes = max(nodes, nodes_from_gpu)
    gpus_per_node = math.ceil(gpus_total / sim_nodes) if gpus_total > 0 else 0

    block = f"""        slurm_partition=\"{partition_name}\",
        runtime={runtime_min},
        tasks={tasks},
        cpus_per_task={cpus_per_task},
        mem_mb={mem_mb},
        nodes={sim_nodes}"""
    # EDITED
    if gpus_total > 0:
        if gpu_alloc_mode == "gpus":
            block += f",\n        gpu=\"{gpus_total}\""
            if gpu_hardware:
                block += f",\n        gpu_model=\"{gpu_hardware}\""
        else:
            if gpu_hardware:
                block += f",\n        gres=\"gpu:{gpu_hardware}:{gpus_per_node}\""
            else:
                block += f",\n        gres=\"gpu:{gpus_per_node}\""

    return block
```

---

## Call-Site Updates

### 4) Standard Workflow (generate_snakefile_content)

```python
sim_resources = self._build_resource_block(
    partition=self.cfg_analysis.hpc_ensemble_partition,
    runtime_min=hpc_time_min,
    mem_mb=mem_mb_per_sim,
    nodes=self.cfg_analysis.n_nodes or 1,
    tasks=mpi_ranks,
    cpus_per_task=omp_threads,
    gpus_total=self.cfg_analysis.n_gpus or 0,
    gpus_per_node_config=self.cfg_analysis.hpc_gpus_per_node or 0,
    gpu_hardware=self.system.cfg_system.gpu_hardware,
    gpu_alloc_mode=self.system.cfg_system.preferred_slurm_option_for_allocating_gpus,
)
```

### 5) Sensitivity Workflow

```python
sim_resources_sa = self._base_builder._build_resource_block(
    partition=sub_analysis.cfg_analysis.hpc_ensemble_partition,
    runtime_min=hpc_time,
    mem_mb=int(mem_per_cpu * n_mpi * n_omp * 1000),
    nodes=sub_analysis.cfg_analysis.n_nodes or 1,
    tasks=n_mpi,
    cpus_per_task=n_omp,
    gpus_total=sub_analysis.cfg_analysis.n_gpus or 0,
    gpus_per_node_config=sub_analysis.cfg_analysis.hpc_gpus_per_node or 0,
    gpu_hardware=self.system.cfg_system.gpu_hardware,
    gpu_alloc_mode=self.system.cfg_system.preferred_slurm_option_for_allocating_gpus or "gres",
)
```

---

## SLURM Profile Updates (batch_job)

**File:** `src/TRITON_SWMM_toolkit/workflow.py`
**Location:** `generate_snakemake_config(mode="slurm")`

```python
gpu_alloc_mode = self.system.cfg_system.preferred_slurm_option_for_allocating_gpus

slurm_sbatch = {
    "partition": "{resources.slurm_partition}",
    "time": "{resources.runtime}:00",
    "mem": "{resources.mem_mb}",
    "nodes": "{resources.nodes}",
    "ntasks": "{resources.tasks}",
    "cpus-per-task": "{resources.cpus_per_task}",
}

if gpu_alloc_mode == "gpus":
    if self.system.cfg_system.gpu_hardware:
        slurm_sbatch["gpus"] = (
            f"{self.system.cfg_system.gpu_hardware}:{{resources.gpu}}"
        )
    else:
        slurm_sbatch["gpus"] = "{resources.gpu}"
else:
    slurm_sbatch["gres"] = "{resources.gres}"
```

> This ensures the profile always aligns with the resource key emitted in
> Snakefiles (`resources.gres` vs `resources.gpus_per_node`).

---

## 1_job_many_srun_tasks SBATCH Header

**File:** `src/TRITON_SWMM_toolkit/workflow.py`
**Location:** `_generate_single_job_submission_script()`

```python
preferred = "gres"

if n_gpus_per_sim > 0:
    gpus_per_node = self.cfg_analysis.hpc_gpus_per_node
    assert gpus_per_node, "hpc_gpus_per_node required when requesting GPUs"

    if gpu_hardware:
        gpu_directive = f"#SBATCH --gres=gpu:{gpu_hardware}:{gpus_per_node}\n"
    else:
        gpu_directive = f"#SBATCH --gres=gpu:{gpus_per_node}\n"
```

---

## Validation / Guardrails

### Optional validation in analysis config

```python
if n_gpus > 0 and not hpc_gpus_per_node:
    raise ValueError("hpc_gpus_per_node must be set when requesting GPUs")
```

This is already enforced inside `_build_resource_block`, but validating early
keeps errors closer to config loading.

---

## Example Configs

### UVA GRES (default)

```yaml
preferred_slurm_option_for_allocating_gpus: gres
gpu_hardware: a100
hpc_gpus_per_node: 1
```

### GPU allocation via gpus-per-node

```yaml
preferred_slurm_option_for_allocating_gpus: gpus
hpc_gpus_per_node: 2
```

---

## Success Criteria

- GPU allocation math is centralized inside `_build_resource_block`.
- Callers do not compute `gpus_per_node` explicitly.
- `preferred_slurm_option_for_allocating_gpus` switches resource emission:
  - `gres` → `resources.gres` → `--gres=...`
  - `gpus` → `resources.gpus` → `--gpus=<hardware>:<count>` (or just count)
- Snakemake profile reflects the same preference so sbatch calls are consistent.
- Sensitivity analysis GPU jobs request GPUs correctly in SBATCH.

---

## Implementation Checklist

- [ ] Add `preferred_slurm_option_for_allocating_gpus` to system config.
- [ ] Refactor `_build_resource_block` with centralized GPU math.
- [ ] Update all call sites to pass `gpus_total` + `gpus_per_node_config` only.
- [ ] Update `generate_snakemake_config()` to emit `gres` vs `gpus`.
- [ ] Update 1-job SBATCH generation to respect preference.
- [ ] Validate with a GPU sensitivity dry run.
