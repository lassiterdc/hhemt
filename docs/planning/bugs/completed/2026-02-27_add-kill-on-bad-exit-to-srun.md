# Bug Fix: Add `--kill-on-bad-exit=1` to srun Commands

**Date**: 2026-02-27
**Priority**: Low (defensive improvement — does not prevent the root cause, only shortens recovery time)
**Related debugging report**: `frontier_sensitivity_suite/debugging_docs/debugging_report_20260227_2056.md` (Run 7)

---

## Problem

When an MPI job launched via `srun` experiences a partial task launch failure (e.g., some tasks fail to connect to the PMI controller due to a transient network event), the surviving tasks block indefinitely inside `MPI_Init` at a `PMI_Barrier` waiting for peers that will never arrive. The `srun` step hangs until the SBATCH job's wall-clock time limit kills it.

**Evidence from Run 7 (sa_31)**:
- `srun -N 2 --ntasks=8` dispatched; tasks 0–3 (local node) launched successfully, tasks 4–7 (remote node) failed with `Error configuring interconnect`
- Surviving tasks 0–3 blocked at `PMI_Barrier` inside `MPI_Init` — no timeout
- SLURM step (`STEP 4155594.32`) hung for **118 minutes** before the SBATCH wall-clock limit killed the entire job
- Result: the job's remaining time budget was consumed doing nothing; other sub-analyses that may have needed a retry could not run

**Why the hang is long**: Cray MPICH's `pmi_inet` `PMI_Barrier` has no configurable timeout. Once tasks 0–3 are stuck waiting for tasks 4–7, there is no application-level mechanism to detect the stall. The only exit is an external kill signal.

**Confirmed by**: `triton-specialist` analysis of MPICH PMI initialization path.

---

## Root Cause

`srun`'s default behavior when tasks exit with non-zero codes is to wait for all remaining tasks to exit naturally before returning. With partial failures, the surviving tasks never exit on their own. `--kill-on-bad-exit=1` changes this: if any task exits with a non-zero code, `srun` sends `SIGKILL` to all remaining tasks and returns promptly.

---

## Proposed Fix

Add `--kill-on-bad-exit=1` to both srun command constructions in `run_simulation.py`.

**File**: `src/TRITON_SWMM_toolkit/run_simulation.py`

**Location 1** — non-GPU srun (line ~558–570):
```python
# BEFORE:
launch_cmd_str = (
    f"srun "
    f"-N {n_nodes_per_sim} "
    f"--ntasks={n_mpi_procs} "
    f"--cpus-per-task={n_omp_threads} "
    "--cpu-bind=cores "
    "--overlap "
    f"{exe} {cfg}"
)

# AFTER:
launch_cmd_str = (
    f"srun "
    f"-N {n_nodes_per_sim} "
    f"--ntasks={n_mpi_procs} "
    f"--cpus-per-task={n_omp_threads} "
    "--cpu-bind=cores "
    "--overlap "
    "--kill-on-bad-exit=1 "
    f"{exe} {cfg}"
)
```

**Location 2** — GPU srun (line ~578–588):
```python
# AFTER:
launch_cmd_str = (
    f"srun "
    f"-N {n_nodes_per_sim} "
    f"--ntasks={n_gpus} "
    f"--cpus-per-task={n_omp_threads} "
    f"{gpu_to_task_bind}"
    "--cpu-bind=cores "
    "--overlap "
    "--kill-on-bad-exit=1 "
    f"{exe} {cfg}"
)
```

---

## Expected Impact

- **Partial launch failure (sa_31 class)**: srun exits within seconds of the first task failure instead of hanging for the remainder of the SBATCH wall-clock limit. The simulation runner logs a clean failure, Snakemake marks the rule as failed, and the job's remaining budget is preserved.
- **Normal runs**: No change. `--kill-on-bad-exit=1` only activates when a task exits with a non-zero code. Clean simulation exits (code 0) are unaffected.
- **GPU mode**: Same benefit — if a GPU task fails at launch, the step cleans up promptly.

---

## Risks and Considerations

- **None identified.** `--kill-on-bad-exit=1` is documented srun behavior, widely used, and strictly improves error recovery. It does not affect the simulation's execution or outputs on success.
- This does **not** prevent the underlying transient PMI `inet_connect` failure — it only reduces the time wasted when it occurs. The retry strategy remains the same.
- No smoke tests are needed for this change — it is only observable on Frontier under actual MPI launch failures, not testable locally.

---

## Verification

After implementation, if a partial task failure occurs on a future run:
- `model_tritonswmm_sa_XX_evt0.log` should show `srun: error: task N launch failed: Error configuring interconnect` followed almost immediately by the surviving tasks being killed (SIGKILL messages or rapid exit), rather than hanging
- `simulation_saXX_evt0.log` should show `Simulation status: simulation started but did not finish` within seconds of the task failure, not hours later
