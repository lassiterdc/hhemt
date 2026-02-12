# Plan: Refine GPU GRES Arguments for SLURM Jobs

**Status:** Draft
**Owner:** Toolkit maintainers
**Created:** 2026-02-11

## Goal

Add an **optional analysis config parameter** that allows users to pass full SLURM
`--gres` arguments (including GPU model/type) to all SBATCH submissions generated
by the toolkit. This should work for:

- `multi_sim_run_method = batch_job` (Snakemake executor jobs)
- `multi_sim_run_method = 1_job_many_srun_tasks` (single job orchestrator)

## Why This Matters

UVA’s GPU guidance specifies that the *second* and *third* arguments of `--gres`
select **GPU hardware type** and **count**, e.g.:

```bash
#SBATCH --gres=gpu:a100:2
```

Today, we only allow a GPU count (via `n_gpus` or `hpc_gpus_per_node`), which
prevents users from requesting specific GPU hardware (e.g., `a100`, `h200`,
`rtx3090`) or richer GRES formats.

## Desired User Experience

```yaml
run_mode: gpu
n_gpus: 2
hpc_gpus_per_node: 2
hpc_gres: gpu:a100:2
```

This should yield:

```bash
#SBATCH --gres=gpu:a100:2
```

Users can also combine this with constraint directives via
`additional_SBATCH_params`:

```yaml
additional_SBATCH_params:
  - "--constraint=a100_80gb"
```

## Proposed Config Change

### ✅ `analysis.py` (new optional field)

**Paste-ready code chunk:**

```python
    hpc_gres: Optional[str] = Field(
        None,
        description=(
            "Optional SLURM --gres argument body (e.g., 'gpu:a100:2', "
            "'gpu:rtx3090:1'). If set, this value is used to emit SBATCH "
            "--gres=... directives for batch_job and 1_job_many_srun_tasks. "
            "If unset, GPU directives are derived from hpc_gpus_per_node."
        ),
    )
```

## Workflow Integration

We need to propagate `hpc_gres` to every SBATCH interface:

1. **Snakemake executor SBATCH payload** (batch_job mode)
2. **Single-job SBATCH header** (`1_job_many_srun_tasks`)

### 1) Snakemake executor SBATCH config

**File:** `src/TRITON_SWMM_toolkit/workflow.py`
**Location:** `generate_snakemake_config(mode="slurm")`

**Paste-ready code chunk:**

```python
            slurm_sbatch = {
                "partition": "{resources.slurm_partition}",
                "time": "{resources.runtime}:00",
                "mem": "{resources.mem_mb}",
                "nodes": "{resources.nodes}",
                "ntasks": "{resources.tasks}",
                "cpus-per-task": "{resources.cpus_per_task}",
                "gpus": "{resources.gpu}",
            }

            # Optional explicit gres override (e.g., gpu:a100:2)
            if self.cfg_analysis.hpc_gres:
                slurm_sbatch["gres"] = self.cfg_analysis.hpc_gres
```

### 2) Single-job SBATCH script

**File:** `src/TRITON_SWMM_toolkit/workflow.py`
**Location:** `_generate_single_job_submission_script()`

**Paste-ready code chunk:**

```python
        if n_gpus_per_sim > 0:
            # Prefer explicit hpc_gres if provided
            if self.cfg_analysis.hpc_gres:
                gpu_directive = f"#SBATCH --gres={self.cfg_analysis.hpc_gres}\n"
            else:
                gpus_per_node = self.cfg_analysis.hpc_gpus_per_node
                assert isinstance(
                    gpus_per_node, int
                ), "hpc_gpus_per_node required when using GPUs in 1_job_many_srun_tasks mode"
                gpu_directive = f"#SBATCH --gres=gpu:{gpus_per_node}\n"
```

### 3) Batch-job orchestration SBATCH (if desired)

**File:** `src/TRITON_SWMM_toolkit/workflow.py`
**Location:** `_submit_batch_job_workflow()`

**Paste-ready code chunk:**

```python
            gres_directive = ""
            if self.cfg_analysis.hpc_gres:
                gres_directive = f"#SBATCH --gres={self.cfg_analysis.hpc_gres}\n"

            script_content = f"""#!/bin/bash
... existing SBATCH lines ...
{gres_directive}#SBATCH --output=..."""
```

> Note: This is optional; the orchestration job itself may not require GPUs,
> but if users want a GPU-enabled Snakemake driver, this makes it possible.

## Validation & Edge Cases

### Validation

- `hpc_gres` is optional. If unset, use existing GPU count logic.
- If `hpc_gres` is set, do **not** override it with `hpc_gpus_per_node`.
- If `run_mode = gpu` and `multi_sim_run_method in {batch_job, 1_job_many_srun_tasks}`:
  ensure `hpc_gpus_per_node` is still required *only* if `hpc_gres` is not provided.

### Optional Validator Update

```python
            if (
                multi_sim_method in {"1_job_many_srun_tasks", "batch_job"}
                and gpus
                and not hpc_gres
                and not hpc_gpus_per_node
            ):
                raise ValueError(
                    "hpc_gpus_per_node is required unless hpc_gres is provided"
                )
```

## Examples

### UVA A100 (80GB) example

```yaml
run_mode: gpu
n_gpus: 2
hpc_gpus_per_node: 2
hpc_gres: gpu:a100:2
additional_SBATCH_params:
  - "--constraint=a100_80gb"
```

### UVA RTX3090 example

```yaml
run_mode: gpu
n_gpus: 1
hpc_gres: gpu:rtx3090:1
```

## Success Criteria

- Users can specify explicit GRES strings via `hpc_gres`.
- `batch_job` and `1_job_many_srun_tasks` SBATCH scripts include `--gres`.
- Snakemake executor jobs (batch_job) inherit `--gres` when requested.
- CPU-only workflows remain unchanged.

---

## Implementation Checklist

- [ ] Add `hpc_gres` to `analysis_config`
- [ ] Pass `hpc_gres` into Snakemake executor sbatch config
- [ ] Pass `hpc_gres` into `1_job_many_srun_tasks` SBATCH header
- [ ] (Optional) Add `hpc_gres` into batch-job orchestration SBATCH header
- [ ] Add validation rule to guard missing GPU info
- [ ] Update docs/examples if needed
