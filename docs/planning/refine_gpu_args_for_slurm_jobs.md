# Refine GPU GRES Args for SLURM Jobs (Plan)

**Status:** Implemented
**Owner:** Toolkit maintainers
**Created:** 2026-02-11

## Goal

Support explicit GPU hardware selection for **all SLURM-backed workflow execution
paths** while keeping the GPU **count** semantics correct across batch_job and
1_job_many_srun_tasks execution modes.

This plan:

- keeps `hpc_gpus_per_node` as the per-node capacity for single-job mode,
- adds a new optional `gpu_hardware` selector (e.g., `a100`, `h200`),
- uses `--gpus` with optional type for **batch_job** (total GPU count),
- uses `--gres` for **1_job_many_srun_tasks** (per-node request), and
- documents paste-ready code chunks for implementation.

## Motivation (UVA GPU requirements)

UVA GPU documentation requires `--gres=gpu` and supports **hardware-qualified gres**:

```bash
#SBATCH --gres=gpu:a100:2
```

Where:

- `a100` = GPU architecture
- `2` = count per node

We need explicit control of the gres string in all SLURM job submission paths.

## Design Principles

1. **Fail fast**: if GPUs are requested, configuration must include a per-node
   GPU count (`hpc_gpus_per_node`) and be internally consistent.
2. **No compatibility shims**: add a clear, explicit hardware selector instead
   of overloading existing fields.
3. **Correct semantics per mode**:
   - batch_job uses total GPU requests (via `--gpus`)
   - 1_job_many_srun_tasks uses per-node GPU requests (via `--gres`)
4. **Explicit and transparent**: errors should point to the misconfigured field.

## Proposed API Changes

### 1) Add `gpu_hardware` (keep `hpc_gpus_per_node`)

`gpu_hardware` is an **optional** GPU type selector for SLURM. When present, it
is used to qualify the GPU request. The **count** continues to come from
`hpc_gpus_per_node` (per-node) or `resources.gpu` (per-job total).

**Paste-ready update:** `src/TRITON_SWMM_toolkit/config/analysis.py`

```python
    gpu_hardware: Optional[str] = Field(
        None,
        description=(
            "Optional GPU hardware selector (e.g., 'a100', 'h200', 'rtx3090'). "
            "If provided, SLURM requests will qualify the GPU type using "
            "--gpus or --gres depending on run mode."
        ),
    )
```

## Validation Rules (Fail Fast)

When GPUs are used (`n_gpus > 0`), `hpc_gpus_per_node` must be present for
1_job_many_srun_tasks. For batch_job, validation ensures `resources.gpu` stays
consistent with the requested total GPUs. Reject invalid or inconsistent
configurations immediately.

**Paste-ready validator:** `src/TRITON_SWMM_toolkit/config/analysis.py`

```python
    @model_validator(mode="before")
    @classmethod
    def check_consistency(cls, values):
        # ... existing logic ...

        n_gpus = values.get("n_gpus") or 0
        multi_sim_method = values.get("multi_sim_run_method")
        hpc_gpus_per_node = values.get("hpc_gpus_per_node")
        gpu_hardware = values.get("gpu_hardware")

        if n_gpus == 0 and gpu_hardware:
            raise ValueError(
                "gpu_hardware is set but n_gpus is 0. Remove gpu_hardware or request GPUs."
            )

        if n_gpus > 0 and multi_sim_method == "1_job_many_srun_tasks":
            if not hpc_gpus_per_node:
                raise ValueError(
                    "hpc_gpus_per_node is required for 1_job_many_srun_tasks when GPUs are used."
                )

        return values
```

## Workflow Updates

### 1) Build GPU selectors by mode

We reconcile **per-job total GPU requests** with **per-node GRES** by splitting
the logic by multi-simulation mode:

- **batch_job** → use SLURM `--gpus` (total count) with optional type.
- **1_job_many_srun_tasks** → use SLURM `--gres` (per-node) with optional type.

### 2) Apply GRES in 1_job_many_srun_tasks SBATCH (per-node)

**Paste-ready block:** `_generate_single_job_submission_script`

```python
        # Build GPU directive if needed (use --gres for per-node specification)
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

### 3) Apply GPUS in Snakemake SLURM Executor (total GPUs)

Ensures **batch_job** jobs spawned by Snakemake request the exact total GPU
count needed by each rule. Use `--gpus` and optionally qualify with hardware.

**Paste-ready block:** `generate_snakemake_config(mode="slurm")`

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

            if self.cfg_analysis.gpu_hardware:
                slurm_sbatch["gpus"] = (
                    f"{self.cfg_analysis.gpu_hardware}:{{resources.gpu}}"
                )

            config.update(
                {
                    "executor": "slurm",
                    "jobs": max_concurrent,
                    "latency-wait": 30,
                    "max-jobs-per-second": 5,
                    "max-status-checks-per-second": 10,
                    "default-resources": [
                        f"nodes=1",
                        f"mem_mb=2000",
                        f"runtime=30",
                        f"slurm_partition={slurm_partition}",
                    ],
                    "slurm": {"sbatch": slurm_sbatch},
                }
            )
```

## Additional Notes

- For constraints like `#SBATCH --constraint=a100_80gb`, continue using
  `additional_SBATCH_params` (no new field required).
- The batch_job orchestration SBATCH script itself remains CPU-only; only the
  Snakemake worker jobs need `--gres`.
- If a future requirement emerges for constraints per job, add an explicit
  `hpc_constraint` field (not needed for this change).

## Collateral Updates

Add `gpu_hardware` in config/presets. Keep `hpc_gpus_per_node` in place.

**Configuration + presets**
- `src/TRITON_SWMM_toolkit/platform_configs.py`
- `src/TRITON_SWMM_toolkit/constants.py`
- `src/TRITON_SWMM_toolkit/case_study_catalog.py`

**Test fixtures + tests**
- `tests/fixtures/test_case_catalog.py`
- `tests/test_workflow_1job_profile_generation.py`
- `tests/test_workflow_1job_sbatch_generation.py`

## Example Configuration

```yaml
run_mode: gpu
n_gpus: 2
multi_sim_run_method: batch_job
gpu_hardware: a100
hpc_ensemble_partition: gpu
hpc_setup_and_analysis_processing_partition: gpu
hpc_account: mygroup
```

## Success Criteria

- `batch_job` Snakemake sbatch jobs request total GPUs via `--gpus`.
- `1_job_many_srun_tasks` SBATCH script requests per-node GPUs via `--gres`.
- Hardware selection is consistent across both paths when `gpu_hardware` is set.
- Misconfigured GPU settings fail fast with actionable error messages.

---

## Implementation Notes (2026-02-11)

- Added `gpu_hardware` to `analysis_config` with validation for GPU mode.
- Updated SLURM executor config to emit `--gpus=<type>:<count>` for batch_job.
- Updated 1_job_many_srun_tasks SBATCH generation to use `--gres=gpu:<type>:<count>`.
- Added `gpu_hardware` to `PlatformConfig` for reuse in presets and test cases.
- Extended SBATCH tests to validate hardware-qualified `--gres` strings.