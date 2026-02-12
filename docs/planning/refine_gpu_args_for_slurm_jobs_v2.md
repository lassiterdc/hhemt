# Refine GPU Arguments for SLURM Jobs (v2)

**Status:** Draft (Implementation-ready)
**Owner:** Toolkit maintainers
**Created:** 2026-02-11

## Goal

Standardize **all** SLURM GPU requests on `--gres` and derive node counts
dynamically from **GPUs per simulation** and **GPUs per node**.

This plan requires `hpc_gpus_per_node` whenever GPUs are requested and uses
that to compute the per-rule node count and `--gres=gpu:<count>` directives.

## Motivation

UVA’s Slurm installation accepts `--gres=gpu:<count>` but rejects
`--gpus=gpu:<count>`. To avoid incompatibilities across clusters, we will
use `--gres` exclusively and compute node counts for each rule using
`hpc_gpus_per_node`.

## Design Principles

1. **GRES-only** — all SLURM GPU requests use `--gres`.
2. **Correct semantics** — per-node GPU counts are explicit.
3. **Fail fast** — invalid GPU settings fail at config validation time.
4. **Production-ready** — code blocks below are paste-ready.

---

## Proposed API Changes

### 1) Require `hpc_gpus_per_node` for GPU workflows

Explicitly declares per-node GPU count when GPUs are requested.

```python
# src/TRITON_SWMM_toolkit/config/analysis.py

    hpc_gpus_per_node: Optional[int] = Field(
        None,
        description=(
            "GPUs per node on the HPC cluster. Required when GPUs are requested "
            "and used to compute per-rule node counts and --gres directives."
        ),
    )
```

### 3) Keep `gpu_hardware`

This allows hardware-specific requests such as:
`--gres=gpu:a100:2`.

```python
# src/TRITON_SWMM_toolkit/config/analysis.py

    gpu_hardware: Optional[str] = Field(
        None,
        description=(
            "Optional GPU hardware selector (e.g., 'a100', 'h200', 'rtx3090'). "
            "Used to qualify --gres=gpu:<hardware>:<count>."
        ),
    )
```

---

## Validation Rules (Fail Fast)

```python
# src/TRITON_SWMM_toolkit/config/analysis.py

    @model_validator(mode="before")
    @classmethod
    def check_consistency(cls, values):
        n_gpus = values.get("n_gpus") or 0
        multi_sim_method = values.get("multi_sim_run_method")
        gpu_hardware = values.get("gpu_hardware")
        hpc_gpus_per_node = values.get("hpc_gpus_per_node")

        if n_gpus == 0 and gpu_hardware:
            raise ValueError(
                "gpu_hardware is set but n_gpus is 0. Remove gpu_hardware or request GPUs."
            )

        if n_gpus > 0 and not hpc_gpus_per_node:
            raise ValueError(
                "hpc_gpus_per_node is required when GPUs are requested."
            )

        return values
```

---

## Workflow Updates (Production-Ready)

### 1) Compute nodes dynamically from GPU needs

**Semantics:** per-rule node count is derived from total GPUs and GPUs per node.

```python
# src/TRITON_SWMM_toolkit/workflow.py
# inside generate_snakefile_content (or a helper)

import math

def _calculate_nodes_for_gpus(total_gpus: int, gpus_per_node: int) -> int:
    if total_gpus <= 0:
        return 1
    return max(1, math.ceil(total_gpus / gpus_per_node))
```

Apply when building resource blocks (use the max of configured n_nodes and
GPU-derived nodes), and compute `gpus_per_node` from total GPUs and nodes:

```python
nodes_from_gpu = _calculate_nodes_for_gpus(n_gpus, gpus_per_node)
configured_nodes = self.cfg_analysis.n_nodes or 1
sim_nodes = max(configured_nodes, nodes_from_gpu)
gpus_per_node = math.ceil(n_gpus / sim_nodes) if n_gpus > 0 else 0

sim_resources = self._build_resource_block(
    partition=self.cfg_analysis.hpc_ensemble_partition,
    runtime_min=hpc_time_min,
    mem_mb=mem_mb_per_sim,
    nodes=sim_nodes,
    tasks=mpi_ranks,
    cpus_per_task=omp_threads,
    gpus_per_node=gpus_per_node,
)
```

### 2) Include gpus_per_node in resource blocks

Ensure the resource block includes `gpus_per_node` so the SLURM profile can
emit it as `--gres=gpu:<count>`.

```python
# src/TRITON_SWMM_toolkit/workflow.py
# inside _build_resource_block

    def _build_resource_block(
        self,
        partition: str | None,
        runtime_min: int,
        mem_mb: int,
        nodes: int,
        tasks: int,
        cpus_per_task: int,
        gpus_per_node: int | None = None,
    ) -> str:
        partition_name = partition or "standard"
        block = f"""        slurm_partition=\"{partition_name}\",
        runtime={runtime_min},
        tasks={tasks},
        cpus_per_task={cpus_per_task},
        mem_mb={mem_mb},
        nodes={nodes}"""
        if gpus_per_node:
            block += f",\n        gpus_per_node={gpus_per_node}"
        return block
```

### 3) Batch_job Snakemake SLURM executor (GRES-only)

**Semantics:** use `--gres` with per-node counts; do not pass `--gpus`.

```python
# src/TRITON_SWMM_toolkit/workflow.py
# inside generate_snakemake_config(mode="slurm")

slurm_sbatch = {
    "partition": "{resources.slurm_partition}",
    "time": "{resources.runtime}:00",
    "mem": "{resources.mem_mb}",
    "nodes": "{resources.nodes}",
    "ntasks": "{resources.tasks}",
    "cpus-per-task": "{resources.cpus_per_task}",
}

hardware = self.cfg_analysis.gpu_hardware
if hardware:
    slurm_sbatch["gres"] = f"gpu:{hardware}:{{resources.gpus_per_node}}"
else:
    slurm_sbatch["gres"] = "gpu:{resources.gpus_per_node}"

config.update(
    {
        "executor": "slurm",
        "jobs": max_concurrent,
        "latency-wait": 30,
        "max-jobs-per-second": 5,
        "max-status-checks-per-second": 10,
        "default-resources": [
            "nodes=1",
            "mem_mb=2000",
            "runtime=30",
            f"slurm_partition={slurm_partition}",
        ],
        "slurm": {"sbatch": slurm_sbatch},
    }
)
```

### 3) 1_job_many_srun_tasks SBATCH allocation

**Semantics:** per-node GPU requests. Always use `--gres`.

```python
# src/TRITON_SWMM_toolkit/workflow.py
# inside _generate_single_job_submission_script

n_gpus_per_sim = sim_resources["n_gpus"]
gpu_directive = ""
gpu_calculation = ""
gpu_cli_arg = ""

if n_gpus_per_sim > 0:
    gpus_per_node = self.cfg_analysis.hpc_gpus_per_node
    assert gpus_per_node, "hpc_gpus_per_node required when requesting GPUs"
    hardware = self.cfg_analysis.gpu_hardware

    if hardware:
        gpu_directive = f"#SBATCH --gres=gpu:{hardware}:{gpus_per_node}\n"
    else:
        gpu_directive = f"#SBATCH --gres=gpu:{gpus_per_node}\n"

    gpu_calculation = (
        "\n# Calculate total GPUs from SLURM allocation\n"
        f"TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * {gpus_per_node}))\n"
    )
    gpu_cli_arg = " --resources gpu=$TOTAL_GPUS"
```

---

## Platform Defaults (UVA Example)

```python
# src/TRITON_SWMM_toolkit/platform_configs.py

UVA_PLATFORM_CONFIG = PlatformConfig(
    ...,
    hpc_gpus_per_node=1,
)
```

---

## Example Configurations

### UVA (per-node via gres)

```yaml
run_mode: gpu
n_gpus: 1
multi_sim_run_method: batch_job
hpc_gpus_per_node: 1
gpu_hardware: a100
hpc_ensemble_partition: gpu
hpc_setup_and_analysis_processing_partition: gpu
hpc_account: mygroup
```

### Generic cluster (per-job via gpus)

```yaml
run_mode: gpu
n_gpus: 2
multi_sim_run_method: batch_job
hpc_gpus_per_node: 4
gpu_hardware: h200
hpc_ensemble_partition: gpu
hpc_setup_and_analysis_processing_partition: gpu
hpc_account: mygroup
```

### 1_job_many_srun_tasks (per-node gres)

```yaml
run_mode: gpu
n_gpus: 1
multi_sim_run_method: 1_job_many_srun_tasks
hpc_gpus_per_node: 4
gpu_hardware: rtx3090
hpc_total_nodes: 2
hpc_total_job_duration_min: 180
hpc_ensemble_partition: gpu
hpc_account: mygroup
```

---

## Success Criteria

- UVA: batch_job workflows submit with `--gres=gpu:<type>:<count>`.
- Other clusters: batch_job workflows also submit with `--gres`.
- 1_job_many_srun_tasks always uses `--gres` per node.
- Misconfigured GPU values fail fast with actionable errors.

---

## Implementation Checklist

- [ ] Require `hpc_gpus_per_node` for GPU workflows.
- [ ] Add validators for new fields (fail fast).
- [ ] Update Snakemake slurm sbatch config to use GRES only.
- [ ] Update 1_job_many_srun_tasks SBATCH generation.
- [ ] Update platform presets (UVA defaults to `gres`).
- [ ] Add test coverage for sbatch strings and profile generation.
