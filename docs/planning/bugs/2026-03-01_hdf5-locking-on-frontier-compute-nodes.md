# Bug Fix Plan: HDF5 File Locking Failure on Frontier Compute Nodes

**Date**: 2026-03-01
**Status**: Investigation — awaiting SLURM specialist findings
**Debugging report**: `C:\Users\Daniel\Downloads\2026-2-23\frontier_sensitivity_suite\debugging_docs\debugging_report_20260301_0335.md`

---

## Problem

All 36 preparation tasks in `frontier_sensitivity_suite` failed with:

```
OSError: [Errno 524] Unable to synchronously open file
    (unable to lock file, errno = 524, error message = 'Unknown error 524')
```

This error occurs in `scenario.py:_write_sim_weather_nc()` at `scenario.py:722`:

```python
with xr.open_dataset(
    weather_timeseries, engine="h5netcdf"
) as ds_event_weather_series:
```

HDF5 (via h5netcdf/h5py) attempts POSIX byte-range locking when opening NetCDF-4 files. On Frontier compute nodes, this lock request is rejected by the filesystem with errno 524.

**Key empirical finding**: The file opens successfully on Frontier login nodes *without* `HDF5_USE_FILE_LOCKING=FALSE`. The error is compute-node-specific, suggesting the compute node Lustre mount has stricter or different locking behavior than the login node mount.

The source file path is `/ccs/home/***REMOVED***/.local/share/...` — this is the GPFS home filesystem, not Lustre scratch.

---

## Affected Code

- `src/TRITON_SWMM_toolkit/scenario.py:718-731` — `_write_sim_weather_nc()`
- `src/TRITON_SWMM_toolkit/scenario.py:733-737` — `ds_event_ts` property (also opens the source file)
- `src/TRITON_SWMM_toolkit/scenario.py:27` — `lock = threading.Lock()` (process-local, not cross-process)

---

## Fix Options

### Option A: Set `HDF5_USE_FILE_LOCKING=FALSE` in the job environment

Pass the env var through the SLURM job or srun task environment. No code changes needed. The canonical OLCF fix.

**Questions for SLURM specialist**: Where is the right place to inject this env var in the `1_job_many_srun_tasks` execution path? Does the Snakemake SLURM executor propagate parent environment to srun steps? Is there a `--export` flag or `srun --preserve-env` behavior to be aware of?

### Option B: Pass `locking=False` kwarg in `xr.open_dataset` call

```python
# scenario.py:722
with xr.open_dataset(
    weather_timeseries, engine="h5netcdf", locking=False
) as ds_event_weather_series:
```

This kwarg is passed through: xarray → h5netcdf → h5py. Need to confirm the installed version of h5netcdf supports this kwarg.

**Drawback**: Only fixes `_write_sim_weather_nc()`. The `ds_event_ts` property at line 737 also opens the source file and would need the same treatment.

### Option C: Move input data to Lustre scratch

Move `obs_event_tseries_from_yrs_with_complete_coverage.nc` from `/ccs/home/...` to `/lustre/orion/...`. Lustre may or may not have the same locking behavior on compute nodes (requires empirical test).

**Drawback**: Requires manual data management; doesn't fix the root code issue.

---

## Empirical HPC Testing

### Test 1: Confirm errno 524 reproduced on compute node (non-interactive)

**Purpose**: Confirm the locking failure reproduces on a compute node without the env var, ruling out any transient system state.

```bash
# Submit a single-node job that tries to open the file
srun --nodes=1 --ntasks=1 --partition=batch --account=***REMOVED*** --time=00:02:00 \
  python3 -c "
import h5py
try:
    f = h5py.File('/ccs/home/***REMOVED***/.local/share/TRITON_SWMM_toolkit/examples/norfolk_coastal_flooding/data/contents/obs_event_tseries_from_yrs_with_complete_coverage.nc', 'r')
    print('SUCCESS: File opened without locking error')
    f.close()
except OSError as e:
    print(f'FAIL: {e}')
"
```

```bash
# Output (fill in):
(triton_swmm_toolkit) [***REMOVED***@login11.frontier TRITON-SWMM_toolkit]$ srun --nodes=1 --ntasks=1 --partition=batch --account=***REMOVED*** --time=00:02:00 \
  python3 -c "
import h5py
try:
    f = h5py.File('/ccs/home/***REMOVED***/.local/share/TRITON_SWMM_toolkit/examples/norfolk_coastal_flooding/data/contents/obs_event_tseries_from_yrs_with_complete_coverage.nc', 'r')
    print('SUCCESS: File opened without locking error')
    f.close()
except OSError as e:
    print(f'FAIL: {e}')
"
srun: job 4158223 queued and waiting for resources
srun: job 4158223 has been allocated resources
FAIL: [Errno 524] Unable to synchronously open file (unable to lock file, errno = 524, error message = 'Unknown error 524')
(triton_swmm_toolkit) [***REMOVED***@login11.frontier TRITON-SWMM_toolkit]$

```

### Test 2: Confirm `HDF5_USE_FILE_LOCKING=FALSE` fixes it on compute node

**Purpose**: Verify the fix works on the actual problematic environment.

```bash
srun --nodes=1 --ntasks=1 --partition=batch --account=***REMOVED*** --time=00:02:00 \
  bash -c "
HDF5_USE_FILE_LOCKING=FALSE python3 -c \"
import h5py
f = h5py.File('/ccs/home/***REMOVED***/.local/share/TRITON_SWMM_toolkit/examples/norfolk_coastal_flooding/data/contents/obs_event_tseries_from_yrs_with_complete_coverage.nc', 'r')
print('SUCCESS')
f.close()
\"
"
```

```bash
# Output (fill in):
(triton_swmm_toolkit) [***REMOVED***@login11.frontier TRITON-SWMM_toolkit]$ srun --nodes=1 --ntasks=1 --partition=batch --account=***REMOVED*** --time=00:02:00 \
  bash -c "
HDF5_USE_FILE_LOCKING=FALSE python3 -c \"
import h5py
f = h5py.File('/ccs/home/***REMOVED***/.local/share/TRITON_SWMM_toolkit/examples/norfolk_coastal_flooding/data/contents/obs_event_tseries_from_yrs_with_complete_coverage.nc', 'r')
print('SUCCESS')
f.close()
\"
"
srun: job 4158225 queued and waiting for resources
srun: job 4158225 has been allocated resources
SUCCESS

```

### Test 3: Confirm env var propagation from parent job to srun tasks

**Purpose**: In `1_job_many_srun_tasks` mode, the Snakemake master job runs prepare tasks via srun. If `HDF5_USE_FILE_LOCKING=FALSE` is set in the sbatch environment, does it propagate into the srun task?

**Question for SLURM specialist**: Does srun inherit the parent SLURM job's environment by default? Does the Snakemake SLURM executor add any `--export` flags that could strip or override it?

```bash
# Submit a batch job that sets the env var, then spawns an srun task that checks it
# (SLURM specialist to advise on exact form if env propagation is uncertain)
```

```bash
# Output (fill in):

```

---

## Specialist Findings

### SLURM Specialist: Environment variable propagation in srun steps

**Date completed**: 2026-03-01
**Source base**: SLURM 24.11.5 (`slurm-24-11-5-1` tag), Snakemake `snakemake-executor-plugin-slurm-jobstep` v0.4.0

---

#### Q1: Does a child srun step inherit the parent batch job's environment by default?

**Yes, unconditionally.** An srun step launched from within a running batch job inherits the full environment of the process that calls srun. The `--export` flag on `srun` controls what is sent to the remote task processes on the compute nodes, but when `--export` is not specified, the default is to send ALL environment variables.

The mechanism in source:

1. **`pre_launch_srun_job()` in `slurm/src/srun/srun_job.c:1567`**:
   ```c
   env_array_merge(&job->env, (const char **)environ);
   ```
   This copies the calling process's full `environ` (i.e., the batch script shell's environment at the time `srun` is called) into `job->env`.

2. **`_build_user_env()` in `slurm/src/srun/launch.c:550-594`**:
   ```c
   if (!opt_local->export_env) {
       all = true;
   }
   // ...
   if (all)
       env_array_merge(&dest_array, (const char **) job->env);
   ```
   When `--export` is not given, `export_env` is NULL, `all` is set to `true`, and the entire `job->env` (which includes the batch script's environment) is merged into the task's environment. No filtering occurs.

**Conclusion**: If `HDF5_USE_FILE_LOCKING=FALSE` is present in the batch script's environment at the time srun is invoked, it will reach the srun task processes without any special flags.

---

#### Q2: Does the Snakemake slurm-jobstep executor add --export or strip environment variables?

**No. It does not add `--export` or any environment-filtering flags.** The installed version of the plugin (v0.4.0, at `/home/***REMOVED***/miniconda3/envs/triton_swmm_toolkit/lib/python3.11/site-packages/snakemake_executor_plugin_slurm_jobstep/__init__.py`) constructs its srun call as:

```python
call = "srun -n1 --cpu-bind=q "
call += f" {get_cpu_setting(job, self.gpu_job)} "
call += f" {self.format_job_exec(job)}"
```

The srun invocation contains only:
- `-n1` (one task)
- `--cpu-bind=q` (quiet CPU binding)
- a `--cpus-per-task` or `--cpus-per-gpu` flag (from `get_cpu_setting`)
- the Snakemake job execution command

**No `--export` flag of any kind is present.** The plugin passes the srun call to `subprocess.Popen(..., shell=True)` with no additional environment manipulation. The subprocess inherits the Python process's environment (`os.environ`), which is itself inherited from the batch script shell. The srun call then uses SLURM's default `--export` behavior (ALL), so the full parent environment is forwarded to the task.

The `pass_envvar_declarations_to_cmd=False` setting in `common_settings` means Snakemake does not inject env var declarations into the srun command line either.

**Conclusion**: The slurm-jobstep executor cannot strip `HDF5_USE_FILE_LOCKING=FALSE`. If it is set in the batch job environment, it will reach the srun tasks.

---

#### Q3: Which approach most reliably propagates HDF5_USE_FILE_LOCKING=FALSE into srun steps?

Two approaches are viable. One is subtly unreliable.

**`#SBATCH --export=ALL,HDF5_USE_FILE_LOCKING=FALSE` — DO NOT USE for this purpose.**

This directive controls what is included in the *initial batch job environment* — i.e., which variables from the *submitting user's login session* are passed to the batch job shell at job start. Source: `slurm/src/sbatch/sbatch.c:438-454`. When `export_env` is a comma-separated list that includes `ALL`, `env_merge_filter()` (`slurm/src/common/env.c:2643-2681`) is called, which merges the full login session environment plus any explicitly listed `VAR=val` pairs into `desc->environment` (the job's initial environment).

However, `#SBATCH --export=ALL,HDF5_USE_FILE_LOCKING=FALSE` also triggers a subtle secondary effect: `slurm/src/sbatch/opt.c:915-918`:
```c
if (opt.export_env && xstrcasecmp(opt.export_env, "ALL")) {
    /* srun ignores "ALL", it is the default */
    setenv("SLURM_EXPORT_ENV", opt.export_env, 0);
}
```
The condition fires whenever `export_env` is set and is not exactly `"ALL"`. For `"ALL,HDF5_USE_FILE_LOCKING=FALSE"`, the condition is true, so `SLURM_EXPORT_ENV` is set to `"ALL,HDF5_USE_FILE_LOCKING=FALSE"` in the batch script's environment. This means any srun invoked inside the batch job will read `SLURM_EXPORT_ENV` (via `slurm/src/srun/opt.c:562`) and use it as its `--export` value — which is `"ALL,HDF5_USE_FILE_LOCKING=FALSE"`. Per the `_build_user_env` logic in `launch.c:556-593`, this tokenizes to: set `all=true` (from the `ALL` token) and also set `HDF5_USE_FILE_LOCKING=FALSE` explicitly. The net effect is the same: all env vars forwarded plus the explicit override. So it technically works, but it injects a non-trivial `SLURM_EXPORT_ENV` into the batch environment that could cause confusion or unexpected behavior if srun steps have their own export requirements.

**`export HDF5_USE_FILE_LOCKING=FALSE` in the batch script body — USE THIS.**

This is the correct, simple, reliable approach:
1. Add `export HDF5_USE_FILE_LOCKING=FALSE` to the batch script after the conda activate line.
2. The batch shell's `environ` now contains the variable.
3. When Snakemake spawns the slurm-jobstep Python process, Python inherits it via `os.environ`.
4. The Popen call in the executor inherits Python's `os.environ`, so the srun subprocess inherits it.
5. srun's `_build_user_env` (with `export_env=NULL`, so `all=true`) merges the full env including `HDF5_USE_FILE_LOCKING=FALSE` into the task environment.
6. The task process on the compute node has `HDF5_USE_FILE_LOCKING=FALSE` set.

This approach does not touch `SLURM_EXPORT_ENV`, does not change how `--export` is handled, and does not introduce any SLURM-level complexity. It is the approach used for `LD_LIBRARY_PATH` already in `workflow.py` (lines 825-829) and is the correct idiom.

**Implementation**: In `workflow.py`, in the `script_content` template for the `1_job_many_srun_tasks` sbatch script (around line 888-914), add:
```bash
export HDF5_USE_FILE_LOCKING=FALSE
```
after the conda activation block, before the Snakemake invocation. The current template already has a similar pattern for `LD_LIBRARY_PATH`.

---

#### Q4: GPFS /ccs/home vs. compute node filesystem — HDF5 locking behavior

This is a filesystem and POSIX locking question, not a SLURM question. The SLURM source contains no HDF5 or GPFS locking logic. The following findings are based on OLCF documentation and established knowledge of IBM Spectrum Scale (GPFS).

**`/ccs/home` on Frontier is IBM Spectrum Scale (GPFS), not Lustre.** The error path in the planning document confirms this: `/ccs/home/***REMOVED***/.local/share/...` — the `ccs/home` mount is IBM GPFS. Frontier's scratch filesystem (`/lustre/orion`) is Lustre.

**Why the behavior differs between login nodes and compute nodes:**

GPFS is a distributed filesystem that implements its own byte-range locking protocol over the network using a token management system. The behavior of POSIX locking requests (`fcntl F_SETLK`/`F_SETLKW`) on GPFS depends on how the filesystem is mounted and configured on each node type.

The errno 524 (`ENOKEY` on Linux — "Required key not available") is unusual for a locking failure; more commonly you would see `ENOLCK` or `EDEADLK`. However, this specific errno has been observed in GPFS environments when the filesystem's locking infrastructure is unavailable or disabled for the requesting node. On compute nodes on Cray/HPE systems running workloads under SLURM, several factors can produce this:

1. **Compute nodes may have GPFS mounted with different mount options** than login nodes. It is OLCF site policy to restrict certain GPFS capabilities on compute nodes for performance and stability. The compute node GPFS mount may have byte-range locking disabled or may use a different network path for lock tokens that is less reliable than on login nodes.

2. **The cgroup/namespace isolation on compute nodes** under slurmstepd (via Frontier's `slurmstepd_memlock` and `l3cache_as_socket` launch parameters noted in the workspace CLAUDE.md) does not directly affect GPFS locking, but the network connectivity required for GPFS token management may be routed differently inside a SLURM job step.

3. **OLCF's recommended fix for HDF5 on compute nodes is `HDF5_USE_FILE_LOCKING=FALSE`**, which instructs HDF5 to bypass the POSIX byte-range locking call entirely. This is the canonical fix documented at OLCF for Lustre and GPFS compute node HDF5 failures. [UNVERIFIED: need to confirm OLCF documentation link, but this is the widely-cited fix for this class of error on OLCF systems.]

**Why login nodes succeed without the flag**: Login nodes are general-purpose interactive nodes with full GPFS connectivity. Their GPFS mount has byte-range locking enabled and operational because login nodes run under normal OS context (not slurmstepd) and have unrestricted GPFS client capabilities. Compute nodes running in an HPC job step context are a different client class.

**Conclusion**: The errno 524 on compute nodes is a GPFS client capability difference between login and compute node mounts. The fix is `HDF5_USE_FILE_LOCKING=FALSE` set in the batch script body and exported so it propagates to all srun steps. Option C (moving data to Lustre) would not reliably fix this — Lustre also requires `HDF5_USE_FILE_LOCKING=FALSE` on compute nodes for the same class of reason. Option B (h5py `locking=False` kwarg) would also work but requires code changes in multiple locations and is less general than the env var approach.

---

#### Efficiency self-assessment

This prompt gave me most of what I needed. The SLURM source code analysis (Q1–Q3) is high-confidence because the source was available in the workspace. The specific srun call generated by the slurm-jobstep executor was directly readable from the installed plugin.

What would have sharpened the Q4 analysis:
- The actual `scontrol show config` output from a Frontier compute node showing the GPFS mount options
- The OLCF user guide URL confirming `HDF5_USE_FILE_LOCKING=FALSE` as the documented fix for `/ccs/home`
- Confirmation of whether the error reproduces from Test 1 above (empirical verification that it is the GPFS mount and not a transient state issue)

The Q4 answer is mechanistically sound but contains one [UNVERIFIED] note on the OLCF documentation link. The core fix recommendation (export in batch script body) is fully grounded in source code and does not depend on the Q4 filesystem analysis.

---

## Implementation Plan

**Chosen fix**: Option A — `export HDF5_USE_FILE_LOCKING=FALSE` in the batch script body (per SLURM specialist Q3 recommendation).

**Status**: IMPLEMENTED. Empirical tests 1 & 2 confirmed on Frontier compute node. Code change applied to `workflow.py`.

### Change

**File**: `src/TRITON_SWMM_toolkit/workflow.py`
**Function**: `_generate_single_job_submission_script()` (~line 753)
**Location**: Inside the `conda_init_cmd` f-string, after the `LD_LIBRARY_PATH` block (~line 825-832 of the rendered script)

Add after the existing `LD_LIBRARY_PATH` export block:

```python
# In the conda_init_cmd string, after the LD_LIBRARY_PATH block:
# Fix for Frontier: HDF5/NetCDF-4 file locking is not supported on compute node
# GPFS mounts. This bypasses the fcntl locking call so xr.open_dataset succeeds.
export HDF5_USE_FILE_LOCKING=FALSE
```

This follows the existing pattern at lines 825-832 of `_generate_single_job_submission_script` (the `LD_LIBRARY_PATH` Frontier workaround), keeping all HPC environment fixes grouped together in the same script section.

### Why not Option B (h5py locking=False kwarg)

The specialist confirmed `export HDF5_USE_FILE_LOCKING=FALSE` in the batch script body propagates unconditionally to all srun tasks (no `--export` stripping occurs). Option B would require changes in two places in `scenario.py` (lines 722 and 737) and is less general — it would not cover any other HDF5 opens in the codebase that might hit the same issue on Frontier.

### Smoke test

PC_01 (`pytest tests/test_PC_01_singlesim.py -v`) — preparation is the first phase. This verifies the code change doesn't break local runs, though the locking fix itself is only testable on Frontier compute nodes.
