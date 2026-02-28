# Bug Investigation: `--cpu-bind=cores` Causes srun Step Scheduling Latency Under High Concurrency

**Date**: 2026-02-27
**Priority**: Medium (affects job efficiency and allocation budget in 1_job_many_srun_tasks mode)
**Related debugging report**: `frontier_sensitivity_suite/debugging_docs/debugging_report_20260227_2056.md` (Run 7)
**Status**: Empirical testing required before any code change

---

## Observation

In Run 7 (Job 4155594), sa_26 (`hybrid, 8 MPI, 4 OMP, 1 node`) completed correctly but took **4050 seconds (67 minutes)** elapsed time. All 36 simulation rules were dispatched concurrently at ~17:37:25. Comparable hybrid simulations on 2–4 nodes completed in 75–220 seconds.

| SA | n_mpi | n_omp | n_nodes | srun CPUs/node | elapsed_s |
|----|-------|-------|---------|----------------|-----------|
| sa_25 | 4 | 8 | 4 | 8 | 84 |
| sa_30 | 4 | 16 | 2 | 32 | 75 |
| sa_33 | 4 | 32 | 4 | 32 | 75 |
| **sa_26** | 8 | 4 | **1** | **32** | **4050** |

TRITON's own model log shows a clean 120-second simulation — confirming the 3930-second remainder was scheduling/queue latency, not compute time.

---

## Hypothesis (from `snakemake-specialist`)

`--cpu-bind=cores` enforces strict CPU binding at SLURM step admission time. `--overlap` only bypasses node exclusivity — it does **not** bypass per-node CPU slot tracking. When 36 srun steps all launch simultaneously, each step holding CPU binding slots on shared nodes, SLURM's step scheduler may be unable to admit a 1-node step requesting 32 specifically-bound cores until enough prior steps release their slots. Multi-node steps (e.g., 4 nodes × 8 CPUs/node = 32 CPUs spread across 4 nodes) are less sensitive because the per-node demand is small.

**Key asymmetry**: sa_26's srun needed all 32 CPUs from a **single node**. Multi-node steps spread the same CPU count across multiple nodes, reducing per-node contention.

Snakemake's local executor is **not** the cause — it dispatches all 36 jobs to a `ThreadPoolExecutor` with 453 workers immediately and has no internal queuing gate after job selection.

---

## Investigation Plan

This hypothesis must be confirmed empirically before any code change. The `--cpu-bind=cores` flag is on the hot path for every non-GPU srun invocation.

**Unknowns**:
1. Does removing `--cpu-bind=cores` eliminate the scheduling latency for concurrent 1-node steps?
2. Is there a performance cost to removing `--cpu-bind=cores` that outweighs the scheduling benefit?
3. Does the issue reproduce consistently, or was it a one-off in Run 7?

---

## Empirical HPC Testing

### Test 1: Reproduce the latency with concurrent 1-node `--cpu-bind=cores` steps

Confirm that concurrent `--overlap --cpu-bind=cores` srun steps with high per-node CPU demand serialize at step admission.

Run inside an interactive or batch session with ≥2 nodes:

```bash
# On Frontier, inside salloc or small SBATCH with 2+ nodes
# Launch 8 concurrent 1-node steps with --cpu-bind=cores; record wall-clock start times
for i in $(seq 1 8); do
    (start=$(date +%s); srun -N 1 --ntasks=8 --cpus-per-task=4 --overlap --cpu-bind=cores sleep 10; end=$(date +%s); echo "step $i: started at $(date -d @$start +%T), waited $((end - start - 10))s") &
done
wait
```

Expected output if hypothesis is correct:
```
# Output (fill in):

```

### Test 2: Same setup without `--cpu-bind=cores`

```bash
for i in $(seq 1 8); do
    (start=$(date +%s); srun -N 1 --ntasks=8 --cpus-per-task=4 --overlap sleep 10; end=$(date +%s); echo "step $i: started at $(date -d @$start +%T), waited $((end - start - 10))s") &
done
wait
```

Expected output if hypothesis is correct (steps start nearly simultaneously):
```
# Output (fill in):

```

### Test 3: Check SLURM's step scheduler state during high concurrency

Run Test 1 again and in a separate terminal poll `squeue -s` while steps are pending:

```bash
watch -n 1 'squeue -s --job $SLURM_JOB_ID'
```

Expected if hypothesis is correct: step for sa_26-equivalent shows `PD` (pending) while others run.
```
# Output (fill in):

```

### Test 4 (optional): Measure performance cost of removing `--cpu-bind=cores`

If Tests 1–2 confirm the hypothesis, measure whether TRITON's runtime performance degrades without `--cpu-bind=cores`:

```bash
# Run actual TRITON simulations (or a proxy workload) in 1-node hybrid config
# With --cpu-bind=cores:
srun -N 1 --ntasks=8 --cpus-per-task=4 --overlap --cpu-bind=cores triton.exe TRITONSWMM.cfg
# Wall-clock time: ___

# Without --cpu-bind=cores:
srun -N 1 --ntasks=8 --cpus-per-task=4 --overlap triton.exe TRITONSWMM.cfg
# Wall-clock time: ___
```

```
# Output (fill in):

```

---

## Potential Fixes (pending empirical confirmation)

### Option A: Remove `--cpu-bind=cores` unconditionally in `1_job_many_srun_tasks` mode

Removes CPU binding enforcement for all srun steps when running in `1_job_many_srun_tasks`. Simplest fix; may have a performance cost for NUMA-sensitive workloads.

**Change location**: `src/TRITON_SWMM_toolkit/run_simulation.py` lines ~564 and ~585

```python
# BEFORE (both non-GPU and GPU blocks):
"--cpu-bind=cores "

# AFTER: remove the line entirely, or conditionally:
# (only in 1_job_many_srun_tasks mode — need to pass multi_sim_run_method into this scope)
```

### Option B: Conditionally omit `--cpu-bind=cores` only for 1-node steps in `1_job_many_srun_tasks`

More targeted: keeps CPU binding for multi-node steps (where it helps with NUMA locality) and removes it only for single-node steps (where it causes the scheduling bottleneck).

```python
cpu_bind_flag = "--cpu-bind=cores " if n_nodes_per_sim > 1 else ""
launch_cmd_str = (
    f"srun "
    f"-N {n_nodes_per_sim} "
    f"--ntasks={n_mpi_procs} "
    f"--cpus-per-task={n_omp_threads} "
    f"{cpu_bind_flag}"
    "--overlap "
    f"{exe} {cfg}"
)
```

### Option C: No change — accept scheduling latency for 1-node hybrid steps

If Test 4 shows a significant performance regression without `--cpu-bind=cores`, the latency may be acceptable given that 1-node hybrid steps are typically faster anyway (lower communication overhead, no cross-node fabric traffic).

---

## Decision criteria

- **Implement Option A or B** if: Tests 1–2 confirm the serialization, and Test 4 shows no significant performance regression without `--cpu-bind=cores`
- **Implement Option B** if: Option A degrades multi-node performance
- **Implement Option C (no change)** if: Test 4 shows meaningful performance regression that outweighs the scheduling latency improvement
- **Re-evaluate** if: Tests 1–2 do not reproduce the latency (may have been a one-off)
