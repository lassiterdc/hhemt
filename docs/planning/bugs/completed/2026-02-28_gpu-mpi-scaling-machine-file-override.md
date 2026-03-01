# Bug Fix: GPU MPI Scaling — `--ntasks-per-gpu=1` Expands Task Count on Frontier

**Date**: 2026-02-28
**Status**: Complete — all fixes confirmed, test passes

---

## Problem

In the `frontier_sensitivity_suite`, 7 GPU sub-analyses (sa_0, sa_2, sa_5, sa_6,
sa_10, sa_11, sa_12) reported `actual_nTasks = 8` in TRITON's `log.out` despite
being configured with `n_mpi_procs` ∈ {1, 2, 3, 4, 5, 6, 7}. All affected runs
also showed `actual_total_gpus = 8`, regardless of the configured `n_gpus`. This
invalidated the GPU scaling data for these 7 sub-analyses — all ran identically to
sa_13 (the 8-GPU configuration).

---

## Root Cause

The GPU srun command used `--ntasks-per-gpu=1`. On Frontier, the parent sbatch job
holds all 8 GPUs on the allocated node. SLURM resolves `--ntasks-per-gpu=1` as
"1 task per GPU in the allocation", expanding task count to match the full node GPU
count (8), regardless of `--ntasks=N`.

**Source**: `slurm/src/interfaces/gres.c:8952-8962` (`_handle_ntasks_per_tres_step`).
The expansion overwrites `num_tasks` when `tmp * ntasks_per_tres > --ntasks`. This
is standard SLURM behavior — not Frontier-specific — but only manifests on
whole-node GPU allocations (where the parent job holds > 1 GPU).

TRITON and its machine file play no role. `nTasks` in `log.out` comes from
`MPI_Comm_size` (`main.cpp` line 78) — TRITON reports whatever MPI provides.
The toolkit already sets `-DTRITON_IGNORE_MACHINE_FILES=ON` for all builds
(`system.py` lines 533, 548, 565); no runtime equivalent exists.

---

## What Was Built

### Fix: Replace `--ntasks-per-gpu=1` with `--gpus-per-task=1`

In `src/TRITON_SWMM_toolkit/run_simulation.py` (GPU srun command construction),
`gpu_to_task_bind` was changed from `"--ntasks-per-gpu=1 "` to `"--gpus-per-task=1 "`.

`--gpus-per-task=1` assigns exactly 1 GPU *to* each task, setting a distinct
`ROCR_VISIBLE_DEVICES` per MPI rank. SLURM honors `--ntasks=N` without expansion.

Note: `--gpus-per-task` and `--ntasks-per-gpu` are mutually exclusive — SLURM errors
if both are specified (`slurm/src/common/slurm_opt.c:4871-4875`). This exclusion
extends to the environment variable `SLURM_NTASKS_PER_GPU` — if set by the batch
allocation (e.g., via Snakemake's SLURM executor injecting `--ntasks-per-gpu`),
srun will refuse `--gpus-per-task` even though the flag wasn't in the srun command.

**Commit**: `610657f` (2026-02-28)

**Empirically verified on Frontier** (interactive allocation, Tests A–F):
- `--ntasks=3 --ntasks-per-gpu=1` → 8 tasks launched (bug confirmed)
- `--ntasks=3 --gpus-per-task=1` → 3 tasks, each with a distinct GPU ✅
- `--ntasks=8 --gpus-per-task=1` → 8 tasks, each with a distinct GPU ✅
- Edge case: `--ntasks=1 --gpus-per-task=1 --overlap` → 1 task, all 8 GPUs visible in
  `ROCR_VISIBLE_DEVICES` — harmless for single-rank jobs (no inter-rank contention)

---

## Validation

After re-running affected sub-analyses (Frontier Run 11, Job 4157398):
- All 9 GPU sub-analyses: `actual_nTasks == n_gpus` ✅
- `actual_total_gpus == n_gpus` ✅
- `assert_analysis_workflow_completed_successfully` passes ✅
