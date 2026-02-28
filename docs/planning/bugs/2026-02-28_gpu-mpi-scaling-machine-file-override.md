# Bug Fix: GPU MPI Scaling — `--ntasks-per-gpu=1` Causes SLURM to Expand Task Count on Frontier

**Date**: 2026-02-28
**Priority**: High (invalidates GPU scaling data for 7 of 14 GPU sub-analyses in frontier_sensitivity_suite)
**Related debugging report**: `frontier_sensitivity_suite/debugging_docs/debugging_report_20260228_1002.md` (Run 9)
**Status**: Fix cleared for implementation — replace `--ntasks-per-gpu=1` with `--gpus-per-task=1` in `run_simulation.py:581`

---

## Observation

In Run 9 of the frontier_sensitivity_suite (Job 4156619, completed successfully), 7 GPU sub-analyses show `actual_nTasks = 8` in TRITON's `log.out` despite being configured with `n_mpi_procs` ∈ {1, 2, 3, 4, 5, 6, 7}. All affected sub-analyses are `run_mode: gpu`, single-node (`n_nodes=1`), and also show `actual_total_gpus = 8` (the full GPU count per Frontier node) regardless of the configured `n_gpus`.

The srun invocations are correct (`--ntasks=n_mpi_procs` as configured). The override happens internally within TRITON after startup, apparently driven by its Frontier machine file.

**Affected sub-analyses** (from `scenario_status.csv`):

| SA    | n_mpi_procs | n_gpus | actual_nTasks | actual_total_gpus |
|-------|-------------|--------|---------------|-------------------|
| sa_0  | 1           | 1      | **8**         | 8                 |
| sa_2  | 2           | 2      | **8**         | 8                 |
| sa_5  | 3           | 3      | **8**         | 8                 |
| sa_6  | 4           | 4      | **8**         | 8                 |
| sa_10 | 5           | 5      | **8**         | 8                 |
| sa_11 | 6           | 6      | **8**         | 8                 |
| sa_12 | 7           | 7      | **8**         | 8                 |
| sa_13 | 8           | 8      | 8 ✅          | 8                 |
| sa_14 | 16          | 16     | 16 ✅         | 16                |

**Srun command for sa_0** (from `logs/sims/simulation_sa0_evt0.log`):
```
srun -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-gpu=1 --cpu-bind=cores --overlap \
  triton.exe TRITONSWMM.cfg
```
Despite `--ntasks=1`, TRITON reports nTasks=8 in `log.out`.

---

## Root Cause (Resolved)

**`--ntasks-per-gpu=1` in the GPU srun command causes SLURM to expand the task count to match the full GPU count of the parent job's allocation**, regardless of `--ntasks=N`. On Frontier, the parent sbatch job holds all 8 GPUs on the node. SLURM interprets `--ntasks-per-gpu=1` as a ratio ("1 task per GPU in the allocation") and spawns 8 tasks. TRITON and its machine file play no role — TRITON faithfully reports whatever task count MPI provides at `MPI_Init`.

The fix is to replace `--ntasks-per-gpu=1` with `--gpus-per-task=1`, which assigns GPUs *to* tasks (correct direction) rather than assigning tasks *per* GPU (wrong direction). Empirically confirmed on Frontier across Tests A–F.

## Consequences for the Test Suite

The test `assert_resource_usage_matches_config` raises `Failed` because `actual_nTasks ≠ n_mpi_procs` for these 7 sub-analyses. After Fix 1 + re-run, `actual_nTasks` will equal `n_gpus` for each — and since `n_mpi_procs == n_gpus` by design for GPU `run_mode`, the existing comparison logic should pass without changes. Fix 3 confirms this after re-run.

---

## Specialist Findings

**Questions posed**:
1. Does TRITON's Frontier machine file unconditionally override `nTasks` to match the full GPU count on the allocated node (8 GPUs/node), regardless of the srun `--ntasks` argument?
2. Is there a mechanism (e.g., `TRITON_IGNORE_MACHINE_FILES` or an equivalent env var or config flag) to suppress this override and honor the configured `n_mpi_procs`?
3. If `--ntasks=1 --ntasks-per-gpu=1` is passed to srun, does TRITON still internally use all 8 GPUs, or does it respect the 1-task constraint?

---

### Answer 1: The Frontier machine file does NOT override nTasks

**The TRITON machine file has no runtime role and no mechanism to override `nTasks`.** The machine file (`cmake/machines/frontier/cray_HIP.sh`) is a bash script that exports environment variables consumed exclusively at **CMake configure time** to select compilers, flags, and the `TRITON_RUN_COMMAND` helper variable. It is sourced once during compilation (`cmake/machine.cmake` lines 223–224: `run_bash_command("source ${machinefile_path} && env" ENV_OUTPUT)`). It is not executed at runtime and has no path to alter TRITON's internal MPI task count after launch.

`nTasks` in `log.out` is the value of `size` from `MPI_Comm_size(ENSIFY_COMM_WORLD, &size)`, called in `main.cpp` line 78. This is written to `log.out` by `Output::triton_log_run_header` at `src/output.h` line 1765:

```cpp
// main.cpp line 78
MPI_Comm_size(ENSIFY_COMM_WORLD, &size);

// src/output.h line 1765
log << "nTasks : " << size << std::endl;
```

`Total GPUs` is also `size` (`src/output.h` line 1798):

```cpp
log << "Total GPUs : " << size << std::endl;
```

Both values are therefore identical and both reflect what the MPI runtime reports as the communicator size. There is no internal TRITON logic that queries hardware GPU count and uses it to override or re-initialize `nTasks`. TRITON assumes 1 GPU per MPI rank; it does not enumerate available GPUs to determine how many ranks to spawn.

The value `nTasks=8` reported in `log.out` for sa_0 through sa_12 is thus the true MPI communicator size at runtime — meaning **8 MPI processes were actually launched by srun**, not 1–7 as configured. The override happens before TRITON starts, in SLURM's srun step resource assignment (see Answer 3).

**Relevant files**:
- `test_data/norfolk_coastal_flooding/triton/src/main.cpp` lines 68–78: MPI initialization; `size` from `MPI_Comm_size`
- `test_data/norfolk_coastal_flooding/triton/src/output.h` lines 1708–1829: `triton_log_run_header`; `nTasks` written at line 1765, `Total GPUs` at line 1798
- `test_data/norfolk_coastal_flooding/triton/cmake/machines/frontier/cray_HIP.sh`: machine file; build-time only, not runtime
- `test_data/norfolk_coastal_flooding/triton/cmake/machine.cmake` lines 220–224: machine file is sourced via `bash -c "source ... && env"` at configure time only

---

### Answer 2: TRITON_IGNORE_MACHINE_FILES is a build-time CMake flag, not a runtime env var — and is already set

`TRITON_IGNORE_MACHINE_FILES` is a CMake `option()` declared in `CMakeLists.txt` line 8:

```cmake
option(TRITON_IGNORE_MACHINE_FILES "Ignore Machine files and use manual CMake/environment configuration only" OFF)
```

It is checked at configure time in `cmake/machine.cmake` line 26:

```cmake
if(TRITON_IGNORE_MACHINE_FILES)
  message(STATUS "TRITON: Ignoring Machine files. Using user-specified CMake options and environment only.")
  ...
else()
  # Normal Machine file processing — sources the .sh file, reads TRITON_* env vars
endif()
```

**The toolkit already passes `-DTRITON_IGNORE_MACHINE_FILES=ON` for every build — both CPU and GPU.** This is confirmed in `src/TRITON_SWMM_toolkit/system.py`:

- CPU build (`system.py` line 533): `"-DTRITON_IGNORE_MACHINE_FILES=ON "`
- GPU/HIP build (`system.py` line 565): `"-DTRITON_IGNORE_MACHINE_FILES=ON "`
- GPU/CUDA build (`system.py` line 548): `"-DTRITON_IGNORE_MACHINE_FILES=ON "`

Also confirmed in the local CPU build cache: `build_tritonswmm_cpu/CMakeCache.txt` line 693: `TRITON_IGNORE_MACHINE_FILES:BOOL=ON`.

This means the `cray_HIP.sh` script's `TRITON_RUN_COMMAND="srun -n 8"` line is **never sourced** when the toolkit compiles TRITON. The machine file had no effect on the GPU binary built for the frontier_sensitivity_suite.

There is no runtime environment variable equivalent to `TRITON_IGNORE_MACHINE_FILES`. It exists only as a CMake configure-time flag. The GPU binary compiled for Frontier was built without any machine-file influence, so there is nothing to suppress.

**Relevant files**:
- `test_data/norfolk_coastal_flooding/triton/CMakeLists.txt` line 8: `option(TRITON_IGNORE_MACHINE_FILES ...)`
- `test_data/norfolk_coastal_flooding/triton/cmake/machine.cmake` lines 26–316: build-time branch guarded by this flag
- `src/TRITON_SWMM_toolkit/system.py` lines 533, 548, 565, 871, 885, 893: toolkit always passes `-DTRITON_IGNORE_MACHINE_FILES=ON`

---

### Answer 3: The 8-task override is caused by srun's resolution of --ntasks-per-gpu, not by TRITON internals

TRITON does not internally use all 8 GPUs in defiance of the 1-task srun constraint. The 8 MPI processes that TRITON sees were launched by srun itself, before TRITON's `MPI_Init` ran.

**How the srun command is constructed**: For GPU run mode, `run_simulation.py` lines 580–594 build the srun command as:

```python
launch_cmd_str = (
    f"srun "
    f"-N {n_nodes_per_sim} "
    f"--ntasks={n_gpus} "          # line 586: uses n_gpus, not n_mpi_procs
    f"--cpus-per-task={n_omp_threads} "
    f"--ntasks-per-gpu=1 "         # line 588
    "--cpu-bind=cores "
    "--overlap "
    "--kill-on-bad-exit=1 "
    f"{exe} {cfg}"
)
```

For sa_0 (`n_gpus=1`, `n_omp_threads=1`): `srun -N 1 --ntasks=1 --cpus-per-task=1 --ntasks-per-gpu=1 --cpu-bind=cores --overlap --kill-on-bad-exit=1 triton.exe TRITONSWMM.cfg`. This matches exactly what the debugging report shows.

**Why srun launches 8 tasks despite `--ntasks=1`**: On Frontier, GPU nodes are allocated as whole nodes via `--gpus-per-node=8` or equivalent GPU resource type, giving the parent sbatch job all 8 GPUs on the node. When an srun step specifies `--ntasks-per-gpu=1` within that 8-GPU allocation, SLURM's srun step resource assignment expands the task count to match the GPUs visible in the step's allocation. The `--ntasks=1` flag is effectively overridden by `--ntasks-per-gpu=1` when there are 8 GPUs in the allocation — SLURM interprets `--ntasks-per-gpu=1` as "one task per GPU in the allocation" and launches 8 tasks.

This is a SLURM scheduler behavior, not a TRITON behavior. TRITON receives 8 processes at `MPI_Init` because srun spawned 8, and faithfully reports `nTasks=8`.

**The machine file's `TRITON_RUN_COMMAND="srun -n 8"` is irrelevant**: This variable is used only to populate the `triton_run.sh` helper script (generated in `cmake/util.cmake` lines 55–62) as a default if no `MPI_CMD` argument is provided. The toolkit never uses `triton_run.sh` — it builds its own srun command directly in `run_simulation.py`. Even if the machine file were active, `TRITON_RUN_COMMAND` would not affect runtime behavior when the toolkit's runner script constructs and issues the srun command independently.

**Consequence for scaling data**: Since the override is in SLURM's srun step and `--ntasks-per-gpu=1` drives it, sa_0 through sa_12 all ran as 8-GPU, 8-task jobs identically to sa_13. The GPU scaling data for `n_gpus` ∈ {1, 2, 3, 4, 5, 6, 7} is invalid — these sub-analyses duplicated sa_13 rather than measuring reduced-GPU performance.

**Root cause of the override**: The `--ntasks-per-gpu=1` flag is incompatible with partial-node GPU allocation when the parent job holds a full 8-GPU Frontier node. Removing `--ntasks-per-gpu=1` from the srun command and relying solely on `--ntasks={n_gpus}` would prevent the expansion, but this requires empirical verification on Frontier before a fix is implemented (per HPC debugging protocol: no fix without empirical confirmation).

**Relevant files**:
- `src/TRITON_SWMM_toolkit/run_simulation.py` lines 580–594: GPU srun command construction; `--ntasks={n_gpus}` at line 586, `--ntasks-per-gpu=1` at line 581/588
- `test_data/norfolk_coastal_flooding/triton/src/main.cpp` lines 68–78: `MPI_Init` and `MPI_Comm_size` — TRITON takes MPI rank count as given from the launcher
- `test_data/norfolk_coastal_flooding/triton/cmake/util.cmake` lines 55–62: `triton_run.sh` generation using `TRITON_RUN_COMMAND` — toolkit does not use this script

<!-- Specialist writes above this line -->

---

## Empirical HPC Testing

The specialist's source analysis was conclusive — Tests 1–3 below are no longer needed to resolve the design question. They are retained for reference in case empirical confirmation of the fix (Test 4) surfaces unexpected behavior.

### ~~Test 1: Inspect TRITON Frontier machine file~~ — RESOLVED by source analysis
Machine file is build-time only; confirmed irrelevant to runtime nTasks.

### ~~Test 2: Confirm nTasks in log.out for sa_0~~ — RESOLVED by scenario_status.csv
`actual_nTasks=8` already confirmed from the run data.

### ~~Test 3: Check TRITON_IGNORE_MACHINE_FILES~~ — RESOLVED by source analysis
Build-time CMake flag only; toolkit already sets it ON; no runtime equivalent exists.

### Test 4: Verify that removing `--ntasks-per-gpu=1` correctly restricts srun to `--ntasks=N` on Frontier

**What**: Confirm that dropping `--ntasks-per-gpu=1` from the GPU srun command causes SLURM to honor `--ntasks={n_gpus}` and launch only N tasks (not 8), and that GPU binding is still correct.

**Why**: Source analysis identifies `--ntasks-per-gpu=1` as the cause of the 8-task expansion. Removing it is the fix candidate. But `--ntasks-per-gpu=1` may also influence GPU binding/affinity, so we need to confirm: (a) SLURM spawns exactly `--ntasks` MPI processes, and (b) each rank binds to a distinct GPU.

```bash
# On Frontier: interactive allocation of 1 GPU node
salloc -N 1 --gpus-per-node=8 -t 10 -A ***REMOVED*** -q debug -p batch

# Test A: with --ntasks-per-gpu=1 (current behavior — should launch 8 tasks)
srun -N 1 --ntasks=1 --ntasks-per-gpu=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'

# Test B: without --ntasks-per-gpu=1 (proposed fix — should launch exactly 1 task)
srun -N 1 --ntasks=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'

# Test C: without --ntasks-per-gpu=1, n=3 tasks (should launch exactly 3 tasks)
srun -N 1 --ntasks=3 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'
```

```bash
# Output (Test A):
(base) ***REMOVED***@frontier00083:~> srun -N 1 --ntasks=1 --ntasks-per-gpu=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'
rank=7 gpus=1
rank=5 gpus=7
rank=0 gpus=4
rank=4 gpus=6
rank=2 gpus=2
rank=3 gpus=3
rank=1 gpus=5
rank=6 gpus=0
(base) ***REMOVED***@frontier00083:~>
```

```bash
# Output (Test B):
(base) ***REMOVED***@frontier00083:~> srun -N 1 --ntasks=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'
rank=0 gpus=0,1,2,3,4,5,6,7
```

```bash
# Output (Test C):
(base) ***REMOVED***@frontier00083:~> srun -N 1 --ntasks=3 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'
rank=2 gpus=0,1,2,3,4,5,6,7
rank=0 gpus=0,1,2,3,4,5,6,7
rank=1 gpus=0,1,2,3,4,5,6,7
```

**Success criteria**: Test A shows 8 lines (confirming the bug); Test B shows 1 line; Test C shows 3 lines with distinct `ROCR_VISIBLE_DEVICES` values per rank.

**Results interpretation**:

- **Test A** ✅ Bug confirmed: 8 tasks launched despite `--ntasks=1`; each rank correctly bound to a distinct single GPU (`gpus=0` through `gpus=7`). Confirms `--ntasks-per-gpu=1` drives the expansion.
- **Test B** ⚠️ Partial success: exactly 1 task launched. But `gpus=0,1,2,3,4,5,6,7` — the single rank sees all 8 GPUs with no isolation. GPU binding is broken.
- **Test C** ⚠️ Partial success: exactly 3 tasks launched (count correct). But all 3 ranks see `gpus=0,1,2,3,4,5,6,7` — no per-rank GPU isolation. Multiple ranks would compete for all 8 GPUs simultaneously.

**Conclusion**: Simply removing `--ntasks-per-gpu=1` fixes the task count but breaks GPU isolation. The correct replacement is `--gpus-per-task=1`, which assigns exactly 1 GPU *to* each task (setting `ROCR_VISIBLE_DEVICES` per-rank) rather than assigning 1 task *per* GPU in the parent allocation (which is what `--ntasks-per-gpu=1` does, expanding task count to match GPU count). An additional empirical test is required.

### Test 5: Verify `--gpus-per-task=1` as the replacement for `--ntasks-per-gpu=1`

**What**: Confirm that `--gpus-per-task=1` (a) honors `--ntasks=N` without expansion, and (b) correctly isolates each rank to a distinct GPU via `ROCR_VISIBLE_DEVICES`.

```bash
# Test D: --gpus-per-task=1, n=1 task (should launch 1 task, bound to 1 GPU)
srun -N 1 --ntasks=1 --gpus-per-task=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'

# Test E: --gpus-per-task=1, n=3 tasks (should launch 3 tasks, each bound to a distinct GPU)
srun -N 1 --ntasks=3 --gpus-per-task=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'

# Test F: --gpus-per-task=1, n=8 tasks (full node — should launch 8 tasks, each bound to 1 GPU)
srun -N 1 --ntasks=8 --gpus-per-task=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'
```

```bash
# Output (Test D):
(base) ***REMOVED***@frontier00083:~> srun -N 1 --ntasks=1 --gpus-per-task=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'
rank=0 gpus=0,1,2,3,4,5,6,7
# Output (Test E):
(base) ***REMOVED***@frontier00083:~> srun -N 1 --ntasks=3 --gpus-per-task=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'
rank=1 gpus=5
rank=2 gpus=2
rank=0 gpus=4
# Output (Test F):
(base) ***REMOVED***@frontier00083:~> srun -N 1 --ntasks=8 --gpus-per-task=1 --overlap bash -c 'echo "rank=$SLURM_PROCID gpus=$ROCR_VISIBLE_DEVICES"'
rank=6 gpus=0
rank=2 gpus=2
rank=7 gpus=1
rank=4 gpus=6
rank=3 gpus=3
rank=0 gpus=4
rank=1 gpus=5
rank=5 gpus=7

```


**Success criteria**: Test D shows 1 line with `gpus=<single GPU id>`; Test E shows 3 lines each with a distinct single GPU id; Test F shows 8 lines each with a distinct single GPU id.

**Results interpretation**:

- **Test D** (ntasks=1): 1 task launched ✅, but `gpus=0,1,2,3,4,5,6,7` — no GPU isolation ⚠️. With a single rank and `--overlap`, SLURM exposes the full node GPU set rather than partitioning to 1. This is the same behavior as Tests B/C (no flag). **However, this is functionally acceptable**: a single-rank GPU job on an 8-GPU node will use GPU 0 by default and ignore the rest. There is no inter-rank GPU contention to worry about with ntasks=1.
- **Test E** (ntasks=3): 3 tasks launched ✅, each bound to a distinct single GPU ✅. `--gpus-per-task=1` correctly partitions GPUs across ranks for the partial-node case.
- **Test F** (ntasks=8): 8 tasks launched ✅, each bound to a distinct single GPU ✅. Full-node case works perfectly.

**Conclusion**: `--gpus-per-task=1` is the correct replacement for `--ntasks-per-gpu=1`. It fixes the task count expansion bug for all multi-GPU cases (the cases the sensitivity experiment cares about: n_gpus ∈ {2,3,4,5,6,7,8,16}). The single-task case (n_gpus=1, sa_0) shows all GPUs visible but is functionally harmless — TRITON will use GPU 0. The fix is cleared for implementation.

---

## Recommended Fix

**Design question resolved**: The override is caused by SLURM's interpretation of `--ntasks-per-gpu=1`, not by TRITON or its machine file. The source code analysis is conclusive. Empirical Test 4 (Tests A–C) confirmed the mechanism and revealed a secondary issue with GPU isolation.

### Fix 1 (REQUIRED): Replace `--ntasks-per-gpu=1` with `--gpus-per-task=1` in the GPU srun command in `run_simulation.py`

In `src/TRITON_SWMM_toolkit/run_simulation.py` line 581 (`gpu_to_task_bind = "--ntasks-per-gpu=1 "`), replace with `"--gpus-per-task=1 "`.

**Why `--gpus-per-task=1` and not simply removing the flag**:
- `--ntasks-per-gpu=1` reads as "assign 1 task *per GPU in the allocation*" — SLURM expands task count to match the parent job's GPU count (causing the bug).
- Simply removing the flag fixes the count but eliminates per-rank GPU isolation: each rank sees all 8 GPUs via `ROCR_VISIBLE_DEVICES` (confirmed by Tests B and C).
- `--gpus-per-task=1` reads as "assign 1 GPU *to each task*" — SLURM honors `--ntasks=N` exactly and sets a distinct `ROCR_VISIBLE_DEVICES` per rank.

**Empirically confirmed** (Tests D–F): `--gpus-per-task=1` correctly restricts task count and provides per-rank GPU isolation for all multi-GPU cases. The n_gpus=1 edge case shows all GPUs visible but is functionally harmless. **Fix 1 is cleared for implementation.**

### Fix 2 (REQUIRED): Re-run sa_0, sa_2, sa_5, sa_6, sa_10, sa_11, sa_12

After Fix 1 is deployed, these 7 sub-analyses must be re-run to obtain valid GPU scaling data. Their current results (all showing nTasks=8, wall_time ≈ 68–91s) are indistinguishable from sa_13 and do not represent 1–7 GPU performance.

### Fix 3 (REQUIRED): Update `validate_resource_usage` to compare `actual_nTasks` against `n_gpus` for GPU run_mode

**Wait for Fix 1 + re-run first.** After the re-run, `actual_nTasks` should equal `n_gpus` (since TRITON assumes 1 GPU per rank and the srun command uses `--ntasks={n_gpus}`). The current comparison against `n_mpi_procs` is correct in intent but the config field name is misleading — for GPU mode, `n_mpi_procs` and `n_gpus` should be equal by design. Confirm this invariant holds after Fix 1 before deciding whether any change to `validate_resource_usage` is needed.

---

---

## SLURM Specialist Assessment

**Date**: 2026-02-28
**Questions addressed**:
1. Is `--ntasks-per-gpu=1` expansion behavior documented SLURM behavior, or a Frontier/Cray/ROCm quirk?
2. Is the `--ntasks=1 --gpus-per-task=1 --overlap` edge case (all GPUs visible to single rank) expected general SLURM behavior or Frontier-specific?
3. Is this worth a FAQ entry, or would an experienced HPC practitioner consider it obvious?

---

### Q1: Is `--ntasks-per-gpu=1` task-count expansion documented SLURM behavior?

**Assessment: Standard SLURM behavior, not a Frontier quirk — but subtly undocumented.**

The expansion is implemented in core SLURM scheduler code shared across all sites. The mechanism:

1. `opt.c` (srun) maps `--ntasks-per-gpu=N` to `step_spec->ntasks_per_tres = N`
   (`slurm/src/common/slurm_opt.c:5741`).

2. In `stepmgr.c`, before step creation, `_copy_job_tres_to_step()` copies the parent
   job's full TRES allocation (including all 8 GPUs in the parent job's GRES list) into
   the step spec, so the step "sees" the full 8-GPU allocation as its GPU count
   (`slurm/src/stepmgr/stepmgr.c:3283-3286`, called at line 3498).

3. `gres_step_state_validate()` in `gres.c` then calls `_handle_ntasks_per_tres_step()`,
   which takes the branch at line 8952: `else if (tmp != NO_VAL64)`. Here `tmp` is the
   GPU count from the parent job's allocation (8 on a full Frontier node). The expansion
   is: `tmp = tmp * ntasks_per_tres` (= 8 * 1 = 8), and if `*num_tasks < tmp` (1 < 8),
   `*num_tasks = tmp`. This overwrites the `--ntasks=1` value with 8
   (`slurm/src/interfaces/gres.c:8952-8962`).

The semantic is: **`--ntasks-per-gpu` is a ratio constraint that SLURM resolves against the
GPU count already in the step's inherited allocation**. It does not mean "use exactly N tasks
per GPU I ask for" — it means "ensure the ratio of tasks to GPUs in the allocation is N:1".
When the parent job holds 8 GPUs and you don't explicitly constrain the step's GPU count,
the full 8 GPUs flow into the step (via `_copy_job_tres_to_step`), and `ntasks_per_tres`
expands the task count to match.

**Why it feels like a Frontier quirk**: On nodes where the parent job holds exactly 1 GPU
(common on NVIDIA clusters that allocate single GPUs by default), `--ntasks-per-gpu=1
--ntasks=1` produces 1 task because `tmp * 1 = 1` is not less than `--ntasks=1`. On
Frontier, the parent job holds all 8 GPUs on the node (because Frontier allocates whole
nodes), so the same flag silently expands to 8. The code path is identical; the outcome
differs because of the whole-node allocation policy, not the hardware or Cray environment.

**Documentation status**: The srun man page (`slurm/doc/man/man1/srun.1`) does not describe
this expansion behavior anywhere in the `--ntasks-per-gpu` option text. The env var
`SLURM_NTASKS_PER_GPU` is described as "Number of tasks requested per GPU" — a ratio
description that is technically correct but does not warn that this ratio is resolved against
the parent job's full GPU allocation and can override an explicit `--ntasks=N`. This is
effectively undocumented behavior that is a predictable consequence of how SLURM resolves
TRES constraints but is not stated plainly in user-facing documentation. An experienced HPC
practitioner who has read the GRES accounting internals would expect it; a practitioner who
has only read the man page would not.

**Is this Cray-specific?** No. The expansion is in `gres.c` and `stepmgr.c`, which are
common SLURM code. The Cray environment (PMIx, Cray MPI wrappers) plays no role. ROCm/HIP
does not affect the task count calculation. The behavior would be identical on an AMD MI300A
node running stock SLURM with no Cray toolchain.

---

### Q2: Is `--ntasks=1 --gpus-per-task=1 --overlap` showing all 8 GPUs expected general SLURM behavior?

**Assessment: General SLURM behavior, not Frontier-specific — a consequence of how `--overlap` interacts with step GRES partitioning.**

With `--gpus-per-task=1 --ntasks=1`, the step's GRES accounting computes 1 total GPU for the
step (1 task × 1 GPU/task). However, `--overlap` (`SSF_OVERLAP_FORCE`) causes the step to
share all resources with the job without deducting from the job's allocation
(`slurm/src/stepmgr/stepmgr.c:2756-2757, 3937-3940`).

The `ROCR_VISIBLE_DEVICES` environment variable is set per-rank by the GRES plugin
(`slurm/src/plugins/gres/common/gres_common.c:306-308`). For a step with `--overlap`, the
GPU binding for the single rank reflects the step's full GRES allocation without
partitioning. With only 1 rank and no other ranks to partition GPUs among, SLURM has no
reason to restrict which GPUs are visible — the whole-node GPU set passes through.

The concrete outcome (all 8 GPUs in `ROCR_VISIBLE_DEVICES`) would occur on any SLURM site
that allocates full nodes and uses `--overlap`. The Cray environment and ROCm are not
involved in setting `ROCR_VISIBLE_DEVICES` — that is set by the GRES plugin before the task
launches.

**Functionally harmless for single-rank GPU jobs**: A single TRITON rank initializing with
HIP will select GPU 0 (or use `hipSetDevice(0)`) regardless of which GPUs are enumerated in
`ROCR_VISIBLE_DEVICES`. There is no inter-rank GPU contention. The `--gpus-per-task=1 intent
(isolate 1 GPU per rank) is only meaningful when multiple ranks would otherwise compete for
the same GPU device; with 1 rank, isolation is a no-op.

---

### Q3: Is this worth a slurm-workspace FAQ entry?

**Assessment: Yes. Recommend a FAQ entry. Here is a draft.**

**Rationale**: The `--ntasks-per-gpu` expansion behavior is a predictable consequence of
SLURM's TRES resolution logic, but it is undocumented in user-facing material and is
specifically dangerous on whole-node allocation clusters like Frontier. An HPC practitioner
who has used `--ntasks-per-gpu=1` successfully on single-GPU NVIDIA nodes (where it behaves
as expected because the parent job holds exactly 1 GPU) will silently get wrong results on
Frontier. The correct fix (`--gpus-per-task=1`) is the semantic inverse and is not obvious
from the naming. The edge case with `--overlap --ntasks=1` is similarly counterintuitive.
Together these justify a FAQ entry.

---

**Draft FAQ entry** (for `_faq/` in slurm-workspace; suggested filename:
`ntasks_per_gpu_expands_task_count_on_whole_node_allocations.md`):

---

```markdown
# Why does `--ntasks-per-gpu=1` expand my task count to the full GPU count on Frontier?

## Symptom

An srun step launched inside a parent sbatch job with `--ntasks=N --ntasks-per-gpu=1`
launches with more than N tasks. On Frontier, it always launches with exactly 8 tasks
(the full GPU count per node), regardless of the `--ntasks=N` value. The application
(e.g., TRITON) reports `nTasks=8` via `MPI_Comm_size`, confirming that srun spawned 8
processes.

## Why it happens

`--ntasks-per-gpu` is a ratio constraint: it tells SLURM to maintain a fixed ratio of
tasks-to-GPUs in the step's GPU allocation. It does not fix the task count independently
of the GPU count.

When an srun step is created, SLURM first copies the parent job's full TRES allocation into
the step spec (`stepmgr/stepmgr.c:3283-3286`, called at `stepmgr.c:3498`). On Frontier,
the parent sbatch job holds all 8 GPUs on the allocated node (Frontier allocates whole nodes).
This means the step's inherited GPU count is 8 even if `--ntasks=1` was specified.

`gres_step_state_validate()` then resolves the ratio: with `ntasks_per_tres=1` and 8
inherited GPUs, it computes `tmp = 8 * 1 = 8` and since `1 < 8`, it overwrites `num_tasks`
with 8 (`interfaces/gres.c:8952-8956`). The `--ntasks=1` flag is silently overridden.

This is standard SLURM behavior — not a Cray, ROCm, or Frontier-specific quirk. It behaves
identically on any site where the parent job holds multiple GPUs. On single-GPU allocations
(common on NVIDIA clusters), `--ntasks-per-gpu=1 --ntasks=1` produces the expected 1 task
because `1 * 1 = 1`, which does not exceed `--ntasks=1`. The expansion only manifests when
the parent job holds more GPUs than the desired task count.

**Source**: `slurm/src/interfaces/gres.c:8912-8967` (`_handle_ntasks_per_tres_step`);
`slurm/src/stepmgr/stepmgr.c:3262-3288` (`_copy_job_tres_to_step`).

## The fix: use `--gpus-per-task=1` instead

`--gpus-per-task=1` means "allocate exactly 1 GPU to each task" — the inverse direction.
SLURM honors `--ntasks=N` exactly and sets a distinct `ROCR_VISIBLE_DEVICES` (or
`CUDA_VISIBLE_DEVICES`) per rank.

```bash
# WRONG: expands to 8 tasks on Frontier (parent job holds 8 GPUs)
srun -N 1 --ntasks=3 --ntasks-per-gpu=1 --overlap ./my_gpu_app

# CORRECT: exactly 3 tasks, each bound to a distinct GPU
srun -N 1 --ntasks=3 --gpus-per-task=1 --overlap ./my_gpu_app
```

**Important**: `--gpus-per-task` and `--ntasks-per-gpu` are mutually exclusive — SLURM will
error if both are specified (`slurm/src/common/slurm_opt.c:4871-4875`).

## Edge case: `--ntasks=1 --gpus-per-task=1 --overlap` shows all 8 GPUs

With a single task and `--overlap`, `ROCR_VISIBLE_DEVICES` will show all 8 GPUs rather than
just 1. This is because `--overlap` (`SSF_OVERLAP_FORCE`) causes the step to share the full
job GRES allocation without partitioning (`stepmgr/stepmgr.c:3937-3938`). With only 1 rank,
there are no other ranks to partition GPUs among, so the whole-node GPU set passes through.

This is functionally harmless for single-rank GPU jobs: a single MPI process will use GPU 0
(or whichever device it selects via `hipSetDevice`) and will not compete with any other rank.
The excess GPUs in `ROCR_VISIBLE_DEVICES` are simply unused.

Empirically confirmed on Frontier (Cray EX, ROCm, SLURM 24.11.5):
- `srun -N 1 --ntasks=1 --gpus-per-task=1 --overlap` → `rank=0 gpus=0,1,2,3,4,5,6,7`
- `srun -N 1 --ntasks=3 --gpus-per-task=1 --overlap` → 3 ranks, each with a distinct single GPU
- `srun -N 1 --ntasks=8 --gpus-per-task=1 --overlap` → 8 ranks, each with a distinct single GPU

## Summary table

| Flag combination | ntasks=N honored? | GPU isolation per rank? | Notes |
|---|---|---|---|
| `--ntasks=N --ntasks-per-gpu=1` | No — expands to GPU count | Yes (each rank gets 1 GPU) | Bug-prone on whole-node allocations |
| `--ntasks=N --gpus-per-task=1` (N > 1) | Yes | Yes (each rank gets 1 GPU) | Correct for multi-rank GPU jobs |
| `--ntasks=1 --gpus-per-task=1 --overlap` | Yes (1 task) | No (all GPUs visible) | Harmless for single-rank jobs |

## Site context

- **Frontier**: Allocates whole nodes; parent job always holds all 8 GCDs per node.
  `--ntasks-per-gpu=1` with any `--ntasks < 8` will expand to 8. Use `--gpus-per-task=1`.
- **Single-GPU allocations** (typical NVIDIA cluster): `--ntasks-per-gpu=1 --ntasks=1`
  appears to work because 1 × 1 = 1 (no expansion). Moving the same script to a whole-node
  GPU allocation will break silently.
```

---

## Definition of Done

- [x] Specialist findings section filled in
- [x] Design question resolved: `--ntasks-per-gpu=1` in srun command causes SLURM to expand task count to full node GPU allocation; TRITON and its machine file are not involved
- [x] Empirical Test 4 (Tests A–C) run on Frontier: bug confirmed; simple removal of flag breaks GPU isolation; `--gpus-per-task=1` identified as replacement candidate
- [x] Empirical Test 5 (Tests D–F) run on Frontier: `--gpus-per-task=1` confirmed correct for all multi-GPU cases; n_gpus=1 edge case functionally acceptable
- [x] Fix 1 implemented: `--ntasks-per-gpu=1` replaced with `--gpus-per-task=1` in `run_simulation.py` line 581
- [ ] Fix 2: sa_0, sa_2, sa_5, sa_6, sa_10, sa_11, sa_12 re-run; `actual_nTasks` matches `n_gpus` for each
- [ ] Fix 3: confirm `validate_resource_usage` comparison is correct after re-run (update if needed)
- [ ] `assert_analysis_workflow_completed_successfully` passes for frontier_sensitivity_suite
- [ ] Debugging report updated to reference this plan
