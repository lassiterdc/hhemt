# Implementation Plan: Simplifying Environments to Address Snakemake Failures

**Plan Date**: 2026-02-23
**Related Bug Report**: `.debugging/test_uva_sensitivity_suite_gpu/debugging_report_20260223_211321.md`
**Supplementary Analysis**: `.debugging/test_uva_sensitivity_suite_gpu/snakemake_agent_perspective.md`

---

## Task Understanding

### Requirements

The `test_uva_sensitivity_suite_gpu` run produced **two distinct failures** that must be addressed independently:

1. **`OSError: [Errno 7] Argument list too long: 'scontrol'`** — crashes both the Snakemake master process and SLURM worker processes. Caused by `snakemake-executor-plugin-slurm` calling `scontrol` while inheriting a bloated shell environment that exceeds Linux's `ARG_MAX` kernel limit. Fix: trim PATH/LD_LIBRARY_PATH before Snakemake inherits them.

2. **`AttributeError: 'str' object has no attribute 'is_storage'`** — the sa_2 SLURM job (`9774868.log`) failed with this error inside the Snakemake executor stack, **not** with `OSError`. This is a Snakemake plugin API shape/version-compatibility issue where the plugin receives a plain `str` where it expects a Snakemake `IOFile` object. Fix: version-compatibility triage upstream; add a preflight version snapshot to improve future diagnosability.

> **Correction to initial debugging report**: The original report attributed sa_2's job failure to `OSError: [Errno 7]`. The `snakemake_agent_perspective.md` corrected this — the actual traceback in `9774868.log` is `AttributeError: 'str' object has no attribute 'is_storage'`. The PATH trim addresses failure 1 but will **not** resolve failure 2.

### Assumptions

- We cannot modify `snakemake-executor-plugin-slurm` (third-party). Our fix for failure 1 controls what environment the plugin inherits.
- The `is_storage` error (failure 2) is most likely a plugin version mismatch, to be confirmed via version capture and upstream reporting.
- No new config fields are needed — both fixes are pure code changes.
- The environment fix must be safe for both HPC and local (non-SLURM) runs.
- Keeping `bash -lc` (login shell) for simulation subprocesses is safer than switching to `bash -c` because the HPC module system relies on login shell initialization for the `module` shell function.
- GPU thread/task arithmetic mismatches visible in workflow summaries are **expected** behavior for GPU rules (the plugin uses `--ntasks-per-gpu`, not `--ntasks`) — not a failure indicator.

### Success Criteria

- `test_uva_sensitivity_suite_gpu` completes all 3 sub-analyses without `OSError: [Errno 7]`
- The `is_storage` error is either resolved by a plugin upgrade or has an upstream issue filed with full reproducibility information
- A version snapshot artifact is written to `logs/` before each workflow run
- Local smoke tests still pass
- No new config fields introduced

---

## Evidence from Codebase

### Where the bloated environment originates

**`workflow.py`, `_submit_tmux_workflow()`** generates `run_workflow_tmux.sh`, which does:
```bash
module purge && module load tmux miniforge gcc/11.4.0 openmpi/4.1.4 cuda/12.2.2
conda activate triton_swmm_toolkit
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
${CONDA_PREFIX}/bin/python -m snakemake ...
```
After `module load` and `conda activate`, `PATH` and `LD_LIBRARY_PATH` are enormous. Snakemake and its SLURM executor plugin inherit this in full. When the plugin calls `scontrol`, `ARG_MAX` is exceeded.

**`run_simulation.py`, `prepare_simulation_command()`** builds the env dict as:
```python
og_env = os.environ.copy()
env["LD_LIBRARY_PATH"] = f"{swmm_path}:{og_env.get('LD_LIBRARY_PATH', '$LD_LIBRARY_PATH')}"
env["PATH"] = og_env.get("PATH", "")
```
Then `run_simulation_runner.py` at line 233 invokes:
```python
proc = subprocess.Popen(cmd, env={**os.environ, **env}, ...)
```
where `cmd = ["bash", "-lc", full_cmd]` and `full_cmd` begins with:
```
export LD_LIBRARY_PATH="<huge value>"; export PATH="<huge value>"; ...
```

This double-embeds the full PATH: once in the subprocess `env=` kwarg, and again as a string argument to bash. The `simulation_sa1_evt0.log` shows PATH alone exceeds 600 characters, LD_LIBRARY_PATH exceeds 400. The combined environment across all variables approaches `ARG_MAX`.

### The two crash sites (failure 1: `OSError`)

1. **Master process crash** (in `tmux_session_*.log`): Snakemake master inherits the full module-loaded environment and calls `scontrol show config` via `get_min_job_age()` in the SLURM plugin's `job_status_query.py`. This call is made during active-job polling even when `sacct`/`squeue` are used for status — it is a separate probe for `MinJobAge`.
2. **Worker process crash** (in `.snakemake/log/2026-02-23T142358.033263.snakemake.log`): The Snakemake orchestrator also hit `OSError` when polling job status mid-run, ultimately crashing the master at step 10/15.

### The sa_2 job failure (failure 2: `AttributeError`)

The sa_2 SLURM job log (`9774868.log`) shows a different, independent failure:
```
AttributeError: 'str' object has no attribute 'is_storage'
```
This occurs inside the Snakemake executor stack when the plugin receives a plain `str` where it expects a Snakemake `IOFile` object (which carries `.is_storage`, `.flags`, and other metadata). This is a version-compatibility break in the `snakemake` + `snakemake-executor-plugin-slurm` + `snakemake-interface-executor-plugins` stack. The TRITON runner was **never launched** for sa_2 — there is no `logs/sims/simulation_sa2_evt0.log`.

### Files identified for change

- `src/TRITON_SWMM_toolkit/workflow.py` — `_submit_tmux_workflow()` generates the tmux script; add PATH trim and version snapshot
- `src/TRITON_SWMM_toolkit/run_simulation.py` — `prepare_simulation_command()` copies full PATH into env dict and embeds it in the bash command string
- `src/TRITON_SWMM_toolkit/run_simulation_runner.py` — Popen call starts from `{**os.environ, **env}`

---

## Implementation Strategy

### Chosen Approach

**Fix 1 (tmux script):** Use `env` prefix to trim PATH and LD_LIBRARY_PATH for the `python -m snakemake` invocation only, without affecting the rest of the tmux shell script. This is the least intrusive mechanism — tmux, conda, and other commands still run in the full environment.

```bash
# Before (bloated):
${CONDA_PREFIX}/bin/python -m snakemake ...

# After (trimmed):
SLURM_BIN=$(dirname "$(command -v scontrol 2>/dev/null || echo "/opt/slurm/current/bin/scontrol")")
env PATH="${CONDA_PREFIX}/bin:${SLURM_BIN}:/usr/local/bin:/usr/bin:/usr/sbin:/bin" \
    LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    ${CONDA_PREFIX}/bin/python -m snakemake ...
```

**Fix 2 (simulation subprocess):** Two changes in `prepare_simulation_command()`:

1. **Remove `env["PATH"] = og_env.get("PATH", "")`** — do not embed the full PATH in the env dict. Since we use `bash -lc` (login shell), the login shell rebuilds PATH from scratch via `/etc/profile` and module system init, which is exactly what we want. The subsequent `module load` in the command string then adds the correct HPC module paths.

2. **Remove `export PATH="..."` from the embedded shell command string** — it is redundant (the login shell sets PATH), and it embeds a huge string as a bash argument.

For `LD_LIBRARY_PATH` in the env dict, trim it to just `{swmm_path}:{CONDA_PREFIX}/lib` instead of appending the full accumulated `og_env["LD_LIBRARY_PATH"]`. The `export LD_LIBRARY_PATH=...` line embedded in the command string (which runs AFTER `module load`) already handles the final LD_LIBRARY_PATH correctly inside the subprocess.

**Fix 3 (Popen base environment):** In `run_simulation_runner.py`, replace `{**os.environ, **env}` with a curated minimal base environment. `os.environ` on HPC contains hundreds of variables from module loads. We only need: identity vars (HOME, USER), SLURM vars (for srun), LMOD vars (for `module load` to work in the login shell), and conda vars (for Python/conda activation). The `env` dict from `prepare_simulation_command()` overlays on top of this minimal base.

### Alternatives Considered

- **Patch `snakemake-executor-plugin-slurm` locally**: Brittle — overwritten on upgrades, requires fork maintenance.
- **Switch to `bash -c` (non-login shell) for simulation subprocess**: Avoids re-expansion of PATH during login, but `module load` is a shell function initialized by the login profile. Non-login shells don't source `/etc/profile`, so `module load` would fail unless we explicitly pass LMOD initialization variables. Unnecessary complexity given that Fix 2 already addresses the PATH bloat without changing shell type.
- **Config field for `minimal_path` on HPC**: Adds user-facing complexity for a problem that should be handled transparently in code.

### Trade-offs

- Using `bash -lc` means the login shell briefly re-expands PATH before `module load` in the command string adds the HPC paths. This is acceptable — the login shell re-initialization rebuilds a clean PATH, which is then extended (not copied from the caller).
- The `env` prefix approach in the tmux script means Snakemake's environment is very lean. If any Snakemake plugin or rule needs a specific env var that was stripped, it would fail. In practice, Snakemake rules use `conda:` directives for their own environments, so this should not be an issue.

---

## File-by-File Change Plan

### 1. `src/TRITON_SWMM_toolkit/workflow.py`

**Purpose**: Trim PATH and LD_LIBRARY_PATH before the `python -m snakemake` invocation in the generated `run_workflow_tmux.sh`.

**Location**: In `_submit_tmux_workflow()`, find where the tmux script content is assembled. The `${CONDA_PREFIX}/bin/python -m snakemake` line is the target.

**Change**: Prepend the `env` prefix invocation block immediately before the snakemake launch line. The SLURM bin is detected dynamically from `command -v scontrol` (available at script runtime on the login node) with a fallback to `/opt/slurm/current/bin`.

```python
# In the generated script template, replace:
#   ${CONDA_PREFIX}/bin/python -m snakemake ...
# With:
#   SLURM_BIN=$(dirname "$(command -v scontrol 2>/dev/null || echo "/opt/slurm/current/bin/scontrol")")
#   env PATH="${CONDA_PREFIX}/bin:${SLURM_BIN}:/usr/local/bin:/usr/bin:/usr/sbin:/bin" \
#       LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
#       ${CONDA_PREFIX}/bin/python -m snakemake ...
```

Also remove the `export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"` line in the tmux script template (if present), since this pre-bloating step becomes unnecessary when we trim at the `env` prefix stage.

**Impact**: Only the Snakemake process and its children see the trimmed environment. All other commands in the tmux script (tmux itself, conda, module) run in the full environment as before.

### 1b. `src/TRITON_SWMM_toolkit/workflow.py` — preflight version snapshot

**Purpose**: Write a `logs/snakemake_versions.txt` artifact before each workflow run so that future debugging can immediately identify the plugin stack in use. This directly addresses the `is_storage` failure's diagnosability — currently there is no record of which plugin versions were active during a failed run.

**Location**: In `_submit_tmux_workflow()`, immediately before the `env PATH=... python -m snakemake` invocation.

**Change**: Add a version capture block to the generated `run_workflow_tmux.sh`:

```bash
# Capture Snakemake plugin stack versions for future debugging.
mkdir -p logs
{
    echo "Captured: $(date -Iseconds)"
    echo "snakemake: $(${CONDA_PREFIX}/bin/python -m snakemake --version 2>/dev/null || echo 'unknown')"
    ${CONDA_PREFIX}/bin/pip show \
        snakemake-executor-plugin-slurm \
        snakemake-executor-plugin-slurm-jobstep \
        snakemake-interface-executor-plugins \
        snakemake-interface-common \
        2>/dev/null | grep -E "^(Name|Version):"
    echo "python: $(${CONDA_PREFIX}/bin/python --version 2>&1)"
    echo "env_size_bytes: $(env | wc -c)"
    echo "path_length_chars: ${#PATH}"
} > logs/snakemake_versions.txt
```

This block runs in the full (pre-trim) environment, so it accurately reflects what was loaded. The `env_size_bytes` and `path_length_chars` lines provide the before-trim size measurements needed to reproduce `ARG_MAX` diagnostics without manual instrumentation.

**Impact**: A new `logs/snakemake_versions.txt` file is written at the start of every workflow run. No behavior change to Snakemake execution itself.

### 2. `src/TRITON_SWMM_toolkit/run_simulation.py`

**Purpose**: Remove full PATH from the env dict and the embedded shell command string.

**Location**: `prepare_simulation_command()` — the section that builds `env` and `full_cmd`.

**Changes**:

**Change 2a**: Remove `env["PATH"] = og_env.get("PATH", "")`. Replace with nothing — do not include PATH in the env dict at all. The `bash -lc` login shell will rebuild PATH from `/etc/profile` and the `module load` in the command string adds the HPC paths.

**Change 2b**: Remove `export PATH="..."` from the embedded shell command string (`env_export_str` or equivalent). If PATH appears in the `env` dict that gets iterated to build export statements, it must be excluded from that loop, or the env dict must not contain PATH (follow-on from Change 2a).

**Change 2c**: Trim LD_LIBRARY_PATH in the env dict:
```python
# Before:
env["LD_LIBRARY_PATH"] = f"{swmm_path}:{og_env.get('LD_LIBRARY_PATH', '$LD_LIBRARY_PATH')}"

# After:
conda_lib = og_env.get("CONDA_PREFIX", "")
conda_lib_path = f"{conda_lib}/lib" if conda_lib else ""
env["LD_LIBRARY_PATH"] = ":".join(filter(None, [str(swmm_path), conda_lib_path]))
```
The `export LD_LIBRARY_PATH=...` already embedded in the command string (which runs AFTER `module load`) handles the final value inside the subprocess; no change needed there.

**Impact**: The env dict no longer contains the full accumulated PATH from the HPC module environment. The bash command string argument is significantly shorter. The subprocess still gets the correct PATH through the login shell initialization + module load sequence inside the command string.

### 3. `src/TRITON_SWMM_toolkit/run_simulation_runner.py`

**Purpose**: Replace `{**os.environ, **env}` with a curated minimal base environment.

**Location**: Line 233 (the `subprocess.Popen(cmd, env={**os.environ, **env}, ...)` call).

**Change**: Build a `clean_env` dict from selected `os.environ` keys, then overlay `env` on top:

```python
# Build minimal base environment for simulation subprocess.
# The bash -lc command string uses 'module load' to restore HPC paths,
# so we do not propagate the full accumulated module environment.
# Required vars:
#   - Identity: HOME, USER, LOGNAME (for bash, file permissions)
#   - SLURM_*: needed by srun to locate resources within the job allocation
#   - PMI*/PMIX*: needed for MPI launch coordination
#   - LMOD_*/MODULEPATH/MODULESHOME: needed for 'module load' to work in login shell
#   - CONDA_*: needed for conda activate / python resolution
clean_env = {}
passthrough_prefixes = ("SLURM_", "PMI", "PMIX", "LMOD_", "CONDA_")
passthrough_exact = ("HOME", "USER", "LOGNAME", "SHELL", "TERM", "TMPDIR",
                     "MODULEPATH", "MODULESHOME", "BASH_ENV")
for key, val in os.environ.items():
    if key in passthrough_exact or key.startswith(passthrough_prefixes):
        clean_env[key] = val
# Overlay simulation-specific env (LD_LIBRARY_PATH, OMP_*, CUDA_*, etc.)
clean_env.update(env)

proc = subprocess.Popen(cmd, env=clean_env, ...)
```

**Impact**: The subprocess no longer inherits the hundreds of variables accumulated by `module load` in the parent shell. Only the variables listed above are passed. The simulation command string's `module load` re-adds the HPC-specific paths inside the subprocess's bash session.

**Import changes**: No new imports needed (uses `os.environ` already in scope).

---

## Risks and Edge Cases

### Risk 1: `module load` fails in subprocess login shell if LMOD vars not captured

The `module` command is a shell function initialized by `/etc/profile` on login. If the LMOD initialization scripts rely on env vars not in `clean_env`, `module load` will fail silently or with "module: command not found".

**Mitigation**: The passthrough prefix `LMOD_` captures all LMOD variables. `MODULEPATH` and `MODULESHOME` are captured by `passthrough_exact`. Also capture `BASH_ENV` (some systems use this to source module init in non-interactive shells). If issues arise, the diagnostic is simple: check the simulation log for "module: command not found" and add the missing variable to `passthrough_exact`.

### Risk 2: `SLURM_JOB_NODELIST` is long even in the minimal env

For large HPC jobs, `SLURM_JOB_NODELIST` can itself be hundreds of characters. However, `srun` inside the simulation subprocess uses this variable to locate the job allocation. We cannot omit it. In practice, the SLURM nodelist for a single-simulation SLURM job (one node) is short. The `ARG_MAX` problem was caused by accumulated PATH/LD_LIBRARY_PATH, not by SLURM vars.

**Mitigation**: Accept this risk. If SLURM vars themselves become long for very large jobs, a follow-up fix can selectively drop `SLURM_JOB_NODELIST` (srun can discover the allocation through other means).

### Risk 3: Local (non-HPC) runs affected by missing PATH

On a local machine without SLURM, the `clean_env` approach in `run_simulation_runner.py` strips PATH entirely (since `og_env` has no LMOD vars and no SLURM vars). The subprocess's `bash -lc` will rebuild PATH from the user's login scripts, which is correct behavior on a local machine.

However, in test environments where login scripts are minimal or absent, the simulation subprocess might not find required tools.

**Mitigation**: Add the current `CONDA_PREFIX/bin` (if set) and standard system bins to `clean_env` unconditionally as a safe baseline, regardless of platform:
```python
system_fallback_path = "/usr/local/bin:/usr/bin:/usr/sbin:/bin"
conda_bin = os.environ.get("CONDA_PREFIX", "")
if conda_bin:
    clean_env.setdefault("PATH", f"{conda_bin}/bin:{system_fallback_path}")
else:
    clean_env.setdefault("PATH", system_fallback_path)
```
Using `setdefault` means the `env` dict overlay (which follows this line) can still override PATH if needed.

### Risk 4: tmux PATH trim leaves out a needed tool at script end

The `env` prefix approach scopes the trim to only the `python -m snakemake` process. The surrounding tmux shell script retains the full environment, so `tmux kill-session` and other script-level commands are unaffected.

### Risk 5: `command -v scontrol` unavailable on non-HPC systems

In the generated tmux script, `$(command -v scontrol 2>/dev/null || echo "/opt/slurm/current/bin/scontrol")` safely falls back. On a non-HPC system, `scontrol` is not present, the fallback path is a non-existent directory, but the `env PATH=...` invocation still works — the PATH just won't include a SLURM bin, which is fine since SLURM isn't used locally.

---

## Validation Plan

### Step 1: Version triage (completed 2026-02-23)

Version triage was completed. The installed stack on UVA HPC is:
- Snakemake: **9.15.0**
- snakemake-executor-plugin-slurm: **2.1.0**
- snakemake-executor-plugin-slurm-jobstep: **0.4.0**
- snakemake-interface-executor-plugins: **9.3.9**
- snakemake-interface-common: **1.22.0**

All packages import from the same conda env (`/home/***REMOVED***/.conda/envs/triton_swmm_toolkit`). No mixed-path install. No clear version incompatibility signal found. The `is_storage` `AttributeError` is still unexplained — it may be a transient state issue or a deterministic bug triggered by a specific execution path for sa_2.

**Next action for failure 2**: Run `--rerun-incomplete` (which will rerun `simulation_sa2_evt0`) to determine if the error is deterministic. If it reproduces, file an upstream issue on `snakemake-executor-plugin-slurm` with `9774868.log`, the Snakemake log, and the version list above.

### Step 2: User re-runs test case on HPC

### Step 3: Verify version snapshot artifact

After the workflow runs, confirm `logs/snakemake_versions.txt` exists and contains:
- A timestamp
- Version entries for all five packages
- `env_size_bytes` before trim (expect >200,000 on HPC)
- `path_length_chars` before trim

### Step 4: HPC re-run (user coordinates on UVA HPC — addresses failure 1)

After version triage (Step 1) and plugin upgrade if applicable, run `test_uva_sensitivity_suite_gpu` with `--rerun-incomplete` (sa_0 and sa_1 are already complete; only sa_2 and the master consolidation need to run):
- Verify: no `OSError: [Errno 7]` in `logs/tmux_session_*.log`
- Verify (if `is_storage` resolved): `logs/sims/simulation_sa2_evt0.log` is created (TRITON runner actually launched)
- Verify: `_status/sims/simulation_sa2_evt0_complete.flag` created
- Verify: `_status/consolidate_complete.flag` created
- Verify: `logs/snakemake_versions.txt` written with expected content

### Step 5: Full fresh HPC run

Run the complete 3-sub-analysis sensitivity suite from scratch to confirm all combinations work end-to-end.

---

## Documentation and Tracker Updates

- **`CLAUDE.md` Gotchas section**: Add a note about the ARG_MAX fix — that we use `env` prefix in the tmux script and a curated minimal env dict in simulation subprocesses to prevent inherited environment bloat causing `OSError: [Errno 7]` on HPC. Also note the version snapshot artifact (`logs/snakemake_versions.txt`).
- **`docs/planning/priorities.md`**: Mark this bug as resolved after HPC validation passes.
- **`run_simulation.py` `prepare_simulation_command()` docstring** (if one exists): Note that PATH is intentionally omitted from the env dict to prevent ARG_MAX overflow; the login shell rebuilds it.
- **Upstream issue** (if `is_storage` is not resolved by a plugin upgrade): File an issue on `snakemake-executor-plugin-slurm` with `9774868.log`, the tmux session log, and the version list from `logs/snakemake_versions.txt`.

Conditions: Update CLAUDE.md only after HPC validation confirms the fix works. Update priorities.md when the DoD checklist is complete.

---

## Decisions Needed from User

**Decision 1: Scope of `clean_env` in `run_simulation_runner.py`** — **RESOLVED: Option B**

Keep `{**os.environ, **env}` in `run_simulation_runner.py` unchanged. Only fix the tmux script (Fix 1 in `workflow.py`) and the `LD_LIBRARY_PATH` construction (Fix 2c in `run_simulation.py`). The `run_simulation_runner.py` Popen call is out of scope for this fix.

**Decision 2: `LD_LIBRARY_PATH` value in the env dict** — **RESOLVED: SWMM path only**

Set `env["LD_LIBRARY_PATH"] = str(swmm_path)` — the SWMM `bin/` directory only, with no appended HPC accumulation. The `module load` inside the bash command string and the subsequent `export LD_LIBRARY_PATH=...` at end of the command string handle all other library paths correctly inside the subprocess.

---

## Definition of Done

**Failure 1 (ARG_MAX / `OSError`):**
- [ ] `run_workflow_tmux.sh` generation (in `workflow.py`) uses `env PATH=... LD_LIBRARY_PATH=...` prefix when launching `python -m snakemake`, with PATH limited to conda bin + SLURM bin + system bins
- [ ] `prepare_simulation_command()` in `run_simulation.py` does not include `PATH` in the env dict and does not embed `export PATH="<full value>"` in the shell command string
- [ ] `prepare_simulation_command()` sets `env["LD_LIBRARY_PATH"] = str(swmm_path)` (SWMM bin only, no accumulated HPC paths)
- [ ] `run_simulation_runner.py` Popen call unchanged (Decision 1 = Option B)
- [ ] On UVA HPC: no `OSError: [Errno 7]` in `logs/tmux_session_*.log`

**Failure 2 (`is_storage` / `AttributeError`):**
- [x] Snakemake plugin stack versions captured on UVA HPC (see Validation Step 1 — triage complete 2026-02-23, no incompatibility found)
- [ ] `--rerun-incomplete` run attempted; `simulation_sa2_evt0` observed to either succeed or reproduce `AttributeError`
- [ ] If reproducible: upstream issue filed on `snakemake-executor-plugin-slurm` with `9774868.log`, Snakemake orchestrator log, and version list

**Preflight version snapshot:**
- [ ] `run_workflow_tmux.sh` generation writes `logs/snakemake_versions.txt` before each run, including package versions and environment size metrics
- [ ] `logs/snakemake_versions.txt` verified present and correct after a test run

**Overall:**
- [ ] Local smoke tests pass: `pytest tests/test_PC_01_singlesim.py` and `pytest tests/test_PC_02_multisim.py -m "not slow"`
- [ ] On UVA HPC: `test_uva_sensitivity_suite_gpu` completes all 3 sub-analyses end-to-end
- [ ] `CLAUDE.md` Gotchas updated
- [ ] `docs/planning/priorities.md` updated

---

## Self-Check Results

**Header/body alignment**: All section headers accurately match their content. "Evidence from Codebase" now correctly documents both failure modes with the `is_storage` correction from `snakemake_agent_perspective.md`. "File-by-File Change Plan" covers the three code files plus the version snapshot addition (1b). "Decisions Needed from User" frames genuinely blocking choices, not implementation details.

**Two-failure separation**: The plan clearly separates `OSError` (Fixes 1–3, our code) from `is_storage` (upstream triage, not our code). The Definition of Done enforces this separation with distinct checklist sections so neither failure can be mistakenly closed out by resolving only the other.

**Section necessity**: All sections present and necessary. "Decisions Needed from User" remains critical — Decision 1 on `clean_env` scope meaningfully changes implementation risk and should be resolved before coding begins.
