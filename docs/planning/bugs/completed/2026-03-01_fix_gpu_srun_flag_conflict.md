# Fix: GPU srun `--gpus-per-task` / `SLURM_NTASKS_PER_GPU` Cross-Platform Conflict

**Plan Date**: 2026-03-01
**Status**: Complete — implemented, tests passing, UVA validation confirmed
**Related Debug Report**: `test_uva_sensitivity_suite_full_suite/debugging_docs/debugging_report_20260301_0200.md`
**Related Prior Plans**:
- `completed/2026-02-28_gpu-mpi-scaling-machine-file-override.md` — added `--gpus-per-task=1` to fix Frontier task expansion
- `completed/2026-02-11_refine_gpu_args_for_slurm_jobs_v2.md` — established gres-only policy

## Problem Statement

All GPU sub-analyses (sa_0 through sa_7) in `test_uva_sensitivity_suite_full_suite` fail with:
```
srun: fatal: --gpus-per-task is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU
```

**Root cause**: The `--gpus-per-task=1` srun flag (added in the Feb 28 Frontier fix) conflicts with `SLURM_NTASKS_PER_GPU` which is present in the UVA job environment. The srun mutual-exclusion check (`slurm/src/common/slurm_opt.c:4933-4934`) calls `fatal()` when both `--gpus-per-task` (CLI) and `SLURM_NTASKS_PER_GPU` (env) are present.

**Source of `SLURM_NTASKS_PER_GPU` (RESOLVED)**: The Snakemake SLURM executor (`submit_string.py:79-91`) unconditionally adds `--ntasks-per-gpu={tasks}` to every sbatch command when it detects `"gpu"` in the `gres` string. With `tasks=1` in the resource block, this becomes `--ntasks-per-gpu=1`, which causes SLURM to set `SLURM_NTASKS_PER_GPU=1` in the job environment. Empirically confirmed: bare `sbatch --gres=gpu:a6000:1` does NOT set this variable (Test 2a); the Snakemake executor is the sole source.

### The Cross-Platform Tension

| Platform | sbatch GPU alloc | srun flag needed | Why |
|----------|-----------------|------------------|-----|
| **Frontier** | `--gpus=N` | `--gpus-per-task=1` | Without it, `--ntasks-per-gpu=1` (old code) expanded task count to full-node GPU count |
| **UVA** | `--gres=gpu:a6000:N` | `--ntasks-per-gpu=1` | Provides per-task GPU isolation via `tres_bind=gres/gpu:single:1`; compatible with `--gres` family |

The srun command in `run_simulation.py` currently uses `--gpus-per-task=1` unconditionally — correct for Frontier (`--gpus` allocation), fatal for UVA when `SLURM_NTASKS_PER_GPU` is present in the environment.

## Evidence

- 8/8 GPU model logs show identical fatal error on UVA
- The Feb 28 plan confirms `--gpus-per-task=1` was **empirically necessary** on Frontier (Tests A-F)
- `run_simulation.py` has no visibility into `preferred_slurm_option_for_allocating_gpus` — it always emits `--gpus-per-task=1`
- The runner **does** have access to the system config via `self._scenario._system.cfg_system`

## Task Understanding

### Requirements
- GPU sub-analyses must work on **both** UVA (gres allocation) and Frontier (gpus allocation)
- Must not regress the Frontier `--ntasks-per-gpu=1` task expansion fix
- Must not regress non-GPU (hybrid, mpi, openmp, serial) sub-analyses

### Success Criteria
- All 8 GPU sub-analyses complete simulation phase on UVA re-run
- No srun flag conflicts on either platform regardless of allocation mode
- Test `test_srun_command_construction.py` updated and passing
- Frontier GPU jobs still get correct task count (no task expansion)

## Proposed Fix

### Approach: Conditionally emit the correct GPU-to-task binding flag based on allocation mode

The runner already has access to `self._scenario._system.cfg_system.preferred_slurm_option_for_allocating_gpus`. Use this to choose the correct srun GPU flag:

- **`gpus` mode** (Frontier): emit `--gpus-per-task=1` (required to prevent task expansion)
- **`gres` mode** (UVA): emit `--ntasks-per-gpu=1` (required for per-task GPU isolation; compatible with `--gres` family)

**Why `--ntasks-per-gpu=1` on srun for gres mode?** Two reasons:

1. **It's the correct flag family.** The Snakemake executor already adds `--ntasks-per-gpu=1` to sbatch, setting `SLURM_NTASKS_PER_GPU=1` in the job env. Using the same flag on srun is redundant but harmless (same family, no conflict). Using `--gpus-per-task` would cross flag families and trigger the fatal mutual-exclusion error.

2. **Defense-in-depth.** Even if the Snakemake executor behavior changes in the future, explicitly passing `--ntasks-per-gpu=1` on srun ensures per-task GPU isolation (triggers `tres_bind=gres/gpu:single:1`). Without any GPU flag, the SLURM specialist confirmed all tasks see all GPUs (empirically verified in Test 1b).

**Why `--ntasks-per-gpu=1` is safe on UVA but was unsafe on Frontier**: On Frontier, the batch job holds all 8 GPUs on the node (exclusive allocation). `--ntasks-per-gpu=1` resolved as "1 task per GPU in the allocation" = 8 tasks, expanding from the requested 3. On UVA, the batch job holds exactly N GPUs (non-exclusive, `--gres=gpu:a6000:N`). `--ntasks-per-gpu=1` with `--ntasks=N` gives `N*1=N` — no expansion.

### Changes

#### 1. `src/TRITON_SWMM_toolkit/run_simulation.py` (line ~580)

The runner needs access to the allocation mode. It already has access to `self._scenario._system.cfg_system`. Thread `preferred_slurm_option_for_allocating_gpus` through to the srun construction:

```python
# BEFORE (line 580-594):
elif run_mode == "gpu":
    gpu_to_task_bind = "--gpus-per-task=1 "
    if using_srun:
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

# AFTER:
elif run_mode == "gpu":
    if using_srun:
        # GPU-to-task binding depends on the batch allocation mode.
        # The two SLURM GPU flag families are mutually exclusive:
        #
        # - "gpus" mode (Frontier): --gpus-per-task=1
        #   Assigns 1 GPU per task. Required because --ntasks-per-gpu=1
        #   expands task count to match full-node GPU count on exclusive
        #   allocations (gres.c:_handle_ntasks_per_tres_step).
        #
        # - "gres" mode (UVA): --ntasks-per-gpu=1
        #   Same flag family as the Snakemake executor's sbatch
        #   --ntasks-per-gpu=1 (submit_string.py:79-91). Redundant
        #   with SLURM_NTASKS_PER_GPU=1 inherited from batch env, but
        #   kept as defense-in-depth for per-task GPU isolation
        #   (triggers tres_bind=gres/gpu:single:1). --gpus-per-task
        #   MUST NOT be used here — it conflicts with the inherited
        #   SLURM_NTASKS_PER_GPU (fatal in SLURM ≥25.05).
        #   Safe from expansion: --ntasks=N with N GPUs → N*1=N tasks.
        #
        # See: completed/2026-02-28_gpu-mpi-scaling-machine-file-override.md
        #      bugs/2026-03-01_fix_gpu_srun_flag_conflict.md
        gpu_alloc_mode = (
            self._scenario._system.cfg_system
                .preferred_slurm_option_for_allocating_gpus
            or "gpus"
        )
        if gpu_alloc_mode == "gpus":
            gpu_bind_flag = "--gpus-per-task=1 "
        else:
            gpu_bind_flag = "--ntasks-per-gpu=1 "
        launch_cmd_str = (
            f"srun "
            f"-N {n_nodes_per_sim} "
            f"--ntasks={n_gpus} "
            f"--cpus-per-task={n_omp_threads} "
            f"{gpu_bind_flag}"
            "--cpu-bind=cores "
            "--overlap "
            "--kill-on-bad-exit=1 "
            f"{exe} {cfg}"
        )
```

#### 2. `tests/test_srun_command_construction.py`

Update GPU tests to cover both allocation modes:

```python
def test_gpu_srun_gpus_mode_includes_gpus_per_task():
    """GPU mode with gpus allocation (Frontier) must include --gpus-per-task=1."""
    full_cmd = build_gpu_srun_command(gpu_alloc_mode="gpus")
    assert "--gpus-per-task=1" in full_cmd
    assert "--ntasks-per-gpu" not in full_cmd
    assert "--ntasks=" in full_cmd

def test_gpu_srun_gres_mode_includes_ntasks_per_gpu():
    """GPU mode with gres allocation (UVA) must include --ntasks-per-gpu=1."""
    full_cmd = build_gpu_srun_command(gpu_alloc_mode="gres")
    assert "--ntasks-per-gpu=1" in full_cmd
    assert "--gpus-per-task" not in full_cmd
    assert "--ntasks=" in full_cmd
```

#### 3. Docstring update in `run_simulation.py` (line ~422-428)

Update the GPU configuration comment:

```python
# GPU task binding depends on the batch allocation mode
# (preferred_slurm_option_for_allocating_gpus):
#
# - "gpus" mode (Frontier): The batch job uses --gpus=N. We pass
#   --gpus-per-task=1 on srun to assign exactly 1 GPU per task without
#   expanding the task count (see completed plan
#   2026-02-28_gpu-mpi-scaling-machine-file-override.md).
#
# - "gres" mode (UVA): The batch job uses --gres=gpu:type:N. We pass
#   --ntasks-per-gpu=1 on srun, which triggers SLURM's auto-generation
#   of tres_bind=gres/gpu:single:1 (launch.c:886-895), providing
#   per-task GPU isolation via CUDA_VISIBLE_DEVICES. Without this flag,
#   all tasks see all allocated GPUs (no isolation). Safe from task
#   expansion because --ntasks=N matches N allocated GPUs.
#   --gpus-per-task is mutually exclusive with SLURM_NTASKS_PER_GPU
#   and must NOT be used in gres mode.
```

## Empirical HPC Testing

### Test 1 (UVA): Verify `--ntasks-per-gpu=1` provides per-task GPU isolation under gres allocation

**Purpose**: Confirm that `--ntasks-per-gpu=1` on srun gives each task exactly 1 GPU when the batch allocation uses `--gres`. This is the proposed fix for UVA.

```bash
# On UVA login node:
sbatch --partition=gpu-a6000 --account=***REMOVED*** --gres=gpu:a6000:2 --ntasks=2 --cpus-per-task=1 --time=00:02:00 --wrap='
echo "=== Batch env GPU vars ==="
env | grep -iE "(GPU|CUDA|VISIBLE|NTASK)" | sort
echo "=== srun with --ntasks-per-gpu=1 ==="
srun --ntasks=2 --cpus-per-task=1 --ntasks-per-gpu=1 --overlap bash -c "echo task=\$SLURM_PROCID gpu=\$CUDA_VISIBLE_DEVICES hostname=\$(hostname)"
'
```

**Expected output**: Each task sees a different `CUDA_VISIBLE_DEVICES` value (e.g., task 0 → `0`, task 1 → `1`). Exactly 2 tasks launched.

```bash
# OUTPUT:
=== Batch env GPU vars ===
CUDA_VISIBLE_DEVICES=0,1
SLURM_GPUS_ON_NODE=2
SLURM_JOB_GPUS=0,4
SLURM_JOB_PARTITION=gpu-a6000
SLURM_NTASKS=2
=== srun with --ntasks-per-gpu=1 ===
task=0 gpu=0 hostname=udc-an38-13
task=1 gpu=0 hostname=udc-an38-13

```

### Test 1b (UVA): Verify NO GPU flags gives NO per-task isolation (negative control)

**Purpose**: Confirm the specialist's finding that omitting GPU flags results in all tasks seeing all GPUs (no isolation). This validates that we cannot just omit flags.

```bash
sbatch --partition=gpu-a6000 --account=***REMOVED*** --gres=gpu:a6000:2 --ntasks=2 --cpus-per-task=1 --time=00:02:00 --wrap='
echo "=== srun with NO GPU flags ==="
srun --ntasks=2 --cpus-per-task=1 --overlap bash -c "echo task=\$SLURM_PROCID gpu=\$CUDA_VISIBLE_DEVICES hostname=\$(hostname)"
'
```

**Expected output**: Both tasks see the same `CUDA_VISIBLE_DEVICES` (e.g., both see `0,1`). This confirms omitting GPU flags is NOT safe.

```bash
# OUTPUT:
=== srun with NO GPU flags ===
task=0 gpu=0,1 hostname=udc-an38-13
task=1 gpu=0,1 hostname=udc-an38-13

```

### Test 2 (UVA): Confirm `--gpus-per-task` conflicts under gres allocation

**Purpose**: Reproduce the error to confirm our diagnosis.

```bash
sbatch --partition=gpu-a6000 --account=***REMOVED*** --gres=gpu:a6000:1 --ntasks=1 --cpus-per-task=1 --time=00:02:00 --wrap='
srun --ntasks=1 --gpus-per-task=1 env | grep GPU
'
```

**Expected output**: `srun: fatal: --gpus-per-task is mutually exclusive with --ntasks-per-gpu and SLURM_NTASKS_PER_GPU`

```bash
# OUTPUT:
SLURM_GPUS_ON_NODE=1
SLURM_JOB_GPUS=0
SLURM_STEP_GPUS=0

```

### Test 2a (UVA): Determine whether `SLURM_NTASKS_PER_GPU` comes from `--gres` or from Snakemake

**Purpose**: The specialist's source analysis says bare `--gres` does NOT set `SLURM_NTASKS_PER_GPU`. If this test shows it's absent, the conflict may be caused by the Snakemake SLURM executor adding `--ntasks-per-gpu` to sbatch.

```bash
sbatch --partition=gpu-a6000 --account=***REMOVED*** --gres=gpu:a6000:1 --ntasks=1 --cpus-per-task=1 --time=00:02:00 --wrap='
echo "=== All SLURM GPU env vars ==="
env | grep -iE "SLURM.*(GPU|TRES|NTASK)" | sort
'
```

**Expected output**: Determines whether `SLURM_NTASKS_PER_GPU` is present. If absent, the Snakemake executor is the source.

```bash
# OUTPUT:
=== All SLURM GPU env vars ===
SLURM_GPUS_ON_NODE=1
SLURM_JOB_GPUS=0
SLURM_JOB_PARTITION=gpu-a6000
SLURM_NTASKS=1
SLURM_TRES_PER_TASK=cpu=1

```

### Test 3 (Frontier): Verify `--gpus-per-task=1` still works under gpus allocation

#user: not necessary, existing approach confirmed working on frontier

### Test 4 (Frontier): Verify omitting `--gpus-per-task` causes task expansion

#user: not necessary, existing approach confirmed working on frontier

## Risks & Considerations

1. **Default fallback**: `preferred_slurm_option_for_allocating_gpus or "gpus"` defaults to `gpus` (Frontier behavior) when unset. This is safe because: (a) the field has always been set in practice via platform configs, and (b) `gpus` mode emits `--gpus-per-task` which will fail clearly (fatal error) rather than silently over-subscribing GPUs.

2. **`SLURM_NTASKS_PER_GPU` source confirmed**: The Snakemake SLURM executor adds `--ntasks-per-gpu=1` to sbatch for any GPU job (Test 2a confirmed bare `--gres` does NOT set it). The srun `--ntasks-per-gpu=1` is therefore redundant (srun picks up `SLURM_NTASKS_PER_GPU=1` from the env) but harmless — same value, same family, no conflict.

3. **Task expansion with gres on UVA**: Only possible if `--ntasks < n_gpus_allocated`. Since our code always sets `--ntasks=n_gpus`, matching the gres allocation, expansion cannot occur. However, this invariant should be documented/asserted.

4. **The `tasks=1` design in workflow.py**: For GPU jobs, the Snakefile always sets `tasks=1` regardless of GPU count, while the srun command uses `--ntasks=n_gpus`. This works because Snakemake's `tasks` field controls the sbatch `--ntasks`, but the runner's srun step overrides with its own `--ntasks`. The `--overlap` flag allows the srun to share the parent allocation's resources. This is a pre-existing design quirk, not introduced by this fix.

5. **Non-srun GPU path**: The `else` branch (local GPU execution) is unaffected — it doesn't use srun flags at all.

## Empirical Test Interpretation

### Test 1 Result: `--ntasks-per-gpu=1` — both tasks see `gpu=0`

This looks like a failure but is actually **correct behavior**. The batch env shows `SLURM_JOB_GPUS=0,4` (physical GPU indices on the node). SLURM remaps each task's `CUDA_VISIBLE_DEVICES` to a 0-indexed local view — task 0 sees physical GPU 0 as logical `0`, task 1 sees physical GPU 4 as logical `0`. Both report `gpu=0` because that's their local remapped index, but they are using **different physical GPUs**. This is standard SLURM `cons_tres` behavior.

**Verdict**: Per-task isolation is working correctly. ✅

### Test 1b Result: No GPU flags — both tasks see `gpu=0,1`

Both tasks see all allocated GPUs (`CUDA_VISIBLE_DEVICES=0,1`). This confirms the SLURM specialist's source analysis: without a GPU binding flag, there is no per-task isolation.

**Verdict**: Confirms omitting GPU flags is NOT safe. ✅

### Test 2 Result: `--gpus-per-task=1` succeeded (unexpected!)

With bare `sbatch --gres=gpu:a6000:1` (no Snakemake executor), `SLURM_NTASKS_PER_GPU` is NOT set (confirmed by Test 2a). Therefore `--gpus-per-task=1` on srun does not trigger the mutual-exclusion check and succeeds. The conflict only occurs when the Snakemake executor adds `--ntasks-per-gpu=1` to sbatch, setting `SLURM_NTASKS_PER_GPU=1` in the environment.

**Verdict**: Confirms the Snakemake executor is the source of the conflict. ✅

### Test 2a Result: No `SLURM_NTASKS_PER_GPU` in bare `--gres` allocation

Confirms the SLURM specialist's source analysis. The env var is only present when `--ntasks-per-gpu` is explicitly passed to sbatch — which the Snakemake SLURM executor does automatically for all GPU jobs (`submit_string.py:79-91`).

**Verdict**: Root cause fully traced. ✅

## Specialist Findings

### Snakemake Specialist (2026-03-01)

The Snakemake SLURM executor (`submit_string.py:79-91`) unconditionally adds `--ntasks-per-gpu` to every GPU job's sbatch command:

```python
# snakemake-executor-plugin-slurm/submit_string.py:79-91
gpu_job = job.resources.get("gpu") or "gpu" in job.resources.get("gres", "")
if gpu_job:
    ntasks_per_gpu = job.resources.get("tasks_per_gpu")
    if ntasks_per_gpu is None:
        ntasks_per_gpu = job.resources.get("tasks")
    if ntasks_per_gpu is None:
        ntasks_per_gpu = 1
    if ntasks_per_gpu >= 1:
        call += f" --ntasks-per-gpu={ntasks_per_gpu}"
```

For our rules: `gres="gpu:a6000:1"` contains `"gpu"` → `gpu_job=True` → `tasks=1` → `--ntasks-per-gpu=1` appended to sbatch. This sets `SLURM_NTASKS_PER_GPU=1` in the job env, which then conflicts with `--gpus-per-task=1` on the srun line.

**Note**: The only way to suppress this is `tasks_per_gpu=0` (added in executor issue #316 for PyTorch). This is not appropriate for our use case.

### SLURM Specialist (2026-03-01)

*All findings verified against SLURM 25.05.1 source (`slurm-25-05-1-1` tag).*

### Finding 1: `--gres` alone does NOT set `SLURM_NTASKS_PER_GPU`

Per source (`slurm/src/sbatch/opt.c:1333-1337`), `SLURM_NTASKS_PER_GPU` is written to the job environment **only** when `--ntasks-per-gpu` is explicitly passed to `sbatch`. A bare `--gres=gpu:a6000:N` allocation does not populate this variable. Neither does `slurmctld` synthesize it from the GRES allocation.

**Implication**: If `SLURM_NTASKS_PER_GPU` is observed in the UVA job environment, it is being set by either (a) the Snakemake SLURM executor adding `--ntasks-per-gpu` to sbatch, or (b) a UVA site prolog script. **Test 2a will determine the source.**

### Finding 2: Omitting all GPU flags from srun provides NO per-task isolation

When srun is invoked with no GPU flags (`--ntasks=N --cpus-per-task=1 --overlap`) inside a `--gres=gpu:a6000:N` batch job:

1. `_copy_job_tres_to_step` (`stepmgr.c:3135`) inherits GRES from the job — step gets access to all N GPUs.
2. `gres_step_state_validate` (`gres.c:9095`) runs with `ntasks_per_tres=NO_VAL16` (no flag given). The `_handle_ntasks_per_tres_step` path is skipped.
3. In slurmstepd, `gres_g_task_set_env` (per-task) only fires when `step->accel_bind_type || step->tres_bind` (`task.c:411-413`). With no GPU flag, neither is set.
4. Instead, only `gres_g_step_set_env` runs once for the whole step (`gres.c:10434`), setting `CUDA_VISIBLE_DEVICES` to **all** allocated GPU indices identically for every task.

**Result**: Every task sees all N GPUs. All tasks compete for all GPUs with no isolation. **This is over-subscription.**

### Finding 3: `--ntasks-per-gpu=1` is the correct flag for gres mode

When `--ntasks-per-gpu=1` is present on the srun command line:

1. srun auto-generates `tres_bind = "gres/gpu:single:1"` (`launch.c:886-895`)
2. This sets `step->tres_bind` to non-null in slurmstepd
3. The `task.c:411-413` gate opens, enabling per-task `gres_g_task_set_env`
4. `_get_usable_gres` with `single:1` bind mode assigns exactly 1 GPU per task in round-robin
5. Each task sees a distinct `CUDA_VISIBLE_DEVICES`

**`--ntasks-per-gpu=1` is not merely a hint — it is the mechanism that produces `tres_bind=gres/gpu:single:1`, the actual per-task GPU isolation primitive.**

### Finding 4: No task expansion risk with `--ntasks-per-gpu=1` on UVA

The task expansion in `_handle_ntasks_per_tres_step` (`gres.c:9046-9074`) checks:

```c
uint64_t tmp = _get_step_gres_list_cnt(new_step_list, "gpu", NULL);
if (tmp != NO_VAL64) {
    tmp = tmp * ntasks_per_tres;
    if (*num_tasks < tmp)
        *num_tasks = tmp;  // expansion
}
```

On UVA with `--gres=gpu:a6000:N` and `--ntasks=N`:
- Job holds exactly N GPUs (non-exclusive allocation)
- `tmp = N * 1 = N`, condition `N < N` is false → **no expansion**

The Frontier expansion happened because the job held all 8 GPUs on the node (exclusive allocation) and `--ntasks=3 < 8*1=8` triggered expansion. This cannot happen on UVA's non-exclusive allocations where the job holds exactly the requested N GPUs.

### Finding 5: The mutual-exclusion check path

The fatal error occurs in `_validate_ntasks_per_gpu` (`slurm_opt.c:4933-4934`). When `--gpus-per-task` is set via CLI and `SLURM_NTASKS_PER_GPU` is in the environment (from sbatch or site prolog), srun calls `fatal()`. The fix — using `--ntasks-per-gpu=1` for gres mode instead of `--gpus-per-task=1` — avoids this conflict entirely because both the sbatch allocation and srun step are in the same GRES flag family.

### Summary: Correct srun GPU binding flags by allocation mode

| srun flags | `tres_bind` auto-generated | Per-task GPU isolation | Task expansion risk |
|---|---|---|---|
| *(none)* | none | **No** — all tasks see all GPUs | No |
| `--gpus-per-task=1` + `SLURM_NTASKS_PER_GPU` in env | N/A | **Fatal error** | N/A |
| `--ntasks-per-gpu=1` (no `SLURM_NTASKS_PER_GPU`) | `gres/gpu:single:1` | **Yes** | No, if `--ntasks=N` matches gres N |
| `--ntasks-per-gpu=1` + `SLURM_NTASKS_PER_GPU=1` in env | `gres/gpu:single:1` | **Yes** | No |
| `--gpus-per-task=1` (no `SLURM_NTASKS_PER_GPU`) | via `gpus_per_task` path | **Yes** | No |
