# 1-Job Many `srun` Tasks: Goal, Findings, and Implementation Plan

## Goal

Enable Snakemake-based orchestration for HPC systems that require a single big
SLURM allocation, so that **`submit_workflow()` can be invoked from a shell and
submits one batch job** in which **each simulation is launched via `srun`**.
This corresponds to `multi_sim_run_method="1_job_many_srun_tasks"`.

## Findings

- `multi_sim_run_method` currently controls the execution strategy in
  `analysis._select_execution_strategy()`. The `SlurmExecutor` already enforces
  concurrency constraints using SLURM environment variables.
- `run_simulation.prepare_simulation_command()` already builds `srun` launch
  commands when running inside a SLURM job (`analysis.in_slurm`). Per direction,
  this behavior should **remain unchanged** and `srun` should be used for each
  simulation in big-job environments.
- `workflow.py` currently generates Snakemake profiles that submit **many**
  SLURM jobs (executor/cluster mode). This does **not** fit one-big-job systems
  where every rule must run inside a single allocation.
- `submit_workflow()` currently runs Snakemake directly (local or SLURM), but
  does not always submit a batch job. For big-job systems, it should **submit**
  a single job, and Snakemake should run *inside* that allocation.

## Current Implementation Summary

### ✅ Single-job Snakemake Profile (Resource-Aware)

`generate_snakemake_config(mode="single_job")` now:

- Uses `ResourceManager._get_simulation_resource_requirements()` to compute
  total allocation size.
- Sets Snakemake `cores` to **total CPUs in allocation** (not `max_concurrent`).
- Uses Snakemake GPU resource limits when GPUs are present:
  - `resources: ["gpu=<total_gpus>"]`
- Relies on per-rule `threads`/`resources` to control concurrency.

### ✅ Single-Job SLURM Submission Script

`_generate_single_job_submission_script()` now:

- Requires being inside a SLURM allocation (`analysis.in_slurm == True`).
- Requests **full allocation resources** for the ensemble:
  - `--nodes={total_nodes}`
  - `--ntasks={max_concurrent}`
  - `--cpus-per-task={cpus_per_sim}`
  - `--gpus={total_gpus}` (if any)
  - `--mem={total_mem_mb}`
- Uses `hpc_sbatch_time_upper_limit_min` to cap runtime if set.

### ✅ Submission Behavior

- `submit_workflow()` always submits a single SLURM job when
  `multi_sim_run_method == "1_job_many_srun_tasks"`.
- No dry-run behavior is used in this mode by design.

### ✅ Config Updates

- Added `hpc_sbatch_time_upper_limit_min` to `analysis_config`.
- Updated `hpc_max_simultaneous_sims` docstring to warn about high-resource
  sensitivity analysis configurations.

---

## Updated Summary of Files Modified

| File | Changes | Status |
|------|---------|--------|
| `workflow.py` | Resource-aware single-job profile + full allocation sbatch script | ✅ Complete |
| `config.py` | Added `hpc_sbatch_time_upper_limit_min` | ✅ Complete |
| `examples.py` | Added `retrieve_norfolk_frontier_sensitivity_minimal()` | ✅ Complete |
| `tests/conftest.py` | Added Frontier sensitivity fixtures | ✅ Complete |

---

## Success Criteria (Completed)

- [x] One-job Snakemake profile exists and respects SLURM allocation
- [x] `submit_workflow()` always submits a single batch job for
      `1_job_many_srun_tasks`
- [x] Each simulation runs via `srun` inside the allocation
- [x] Sensitivity workflows follow the same behavior
- [x] Added `hpc_sbatch_time_upper_limit_min` and applied it

---

## Next Steps

1. Run tests on Frontier HPC to validate end-to-end behavior.
2. Monitor performance and tune `hpc_max_simultaneous_sims` as needed.
