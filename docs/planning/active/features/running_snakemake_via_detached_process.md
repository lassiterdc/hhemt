# Plan: Run Snakemake via Detached Login-Node Process (Execution Mode `login_detached`)

**Status:** Draft (ready for review)
**Created:** 2026-02-11
**Owner:** Toolkit maintainers

## Goal

Provide an automated, **login-node detached** execution mode for Snakemake that:

- avoids running Snakemake inside an active SLURM allocation,
- still allows long-running workflows without an open terminal, and
- keeps the same `analysis.run()` user-facing workflow as today.

The proposed interface is:

```python
analysis.run(execution_mode="login_detached", ...)
```

This maps to **Approach 2, Interface 1** from the earlier discussion.

---

## Background (Why This Is Needed)

Snakemake prints a warning when invoked **inside** a SLURM job:

> “You are running snakemake in a SLURM job context. This is not recommended...”

In that context, SLURM can treat Snakemake’s submissions as job steps, and the
`sbatch` profile fields (e.g., `account`) may be ignored or overridden. This is
exactly what we observed in UVA runs, where the profile contained
`account: ***REMOVED***` but the generated `sbatch` calls omitted `--account`.

The safest fix is to **run Snakemake from the login node** and detach it
cleanly (nohup/setsid), so it can submit normal `sbatch` jobs using the
profile config without nested SLURM behavior.

---

## Proposed User Experience

```python
analysis.run(
    from_scratch=False,
    execution_mode="login_detached",
    dry_run=False,
    wait_for_job_completion=False,
    verbose=True,
)
```

Expected behavior:

- Snakemake runs **outside SLURM** from a detached background process.
- The Python call returns immediately with a log path and PID info.
- Users can close their terminal safely.

---

## Design Overview

### 1) Add a new execution mode

Extend the `analysis.run()` / workflow submission interface to accept:

```python
execution_mode: Literal["auto", "local", "slurm", "login_detached"]
```

### 2) Reuse existing slurm executor path

`login_detached` should **still use the normal Snakemake slurm executor**
(`--executor slurm` and the `slurm` profile). The only change is how the
Snakemake process is launched:

- detached (nohup/setsid)
- from a login node
- not inside SLURM

### 3) Avoid nested SLURM

The mode should detect if the caller is already inside SLURM and either:

- **warn** and refuse to run `login_detached`, or
- **force** a safer fallback (e.g., raise or revert to batch_job).

---

## Implementation Plan

### Step 1 — Add new execution mode plumbing

**Files:**
- `src/TRITON_SWMM_toolkit/analysis.py`
- `src/TRITON_SWMM_toolkit/workflow.py`

**Changes:**
- Accept `execution_mode="login_detached"` in `analysis.run()`
- Thread this mode through to `submit_workflow()`

### Step 2 — New detached login-node runner

Add a new helper in `workflow.py`, based on `_run_snakemake_slurm_detached`, but:

- uses `subprocess.Popen` with `start_new_session=True`
- uses `nohup` or `setsid` explicitly
- runs **only if not in SLURM** (`SLURM_JOB_ID` unset)

Proposed method signature:

```python
def _run_snakemake_slurm_login_detached(
    self,
    snakefile_path: Path,
    verbose: bool = True,
    dry_run: bool = False,
) -> dict:
    ...
```

### Step 3 — Add guard against nested SLURM

In `analysis.run()`:

```python
if execution_mode == "login_detached" and self.in_slurm:
    raise WorkflowError(
        "login_detached requires running from a login node (no SLURM allocation)."
    )
```

### Step 4 — Add logging and return payload

Return a dictionary containing:

- `success: True`
- `mode: "login_detached"`
- `snakemake_logfile`
- `pid` or `process` handle
- `message`

---

## CLI & API Integration

### API

Expose in `analysis.run()`:

```python
analysis.run(execution_mode="login_detached")
```

### CLI (optional future extension)

Add flag to CLI:

```bash
triton-swmm run --execution-mode login_detached
```

---

## Edge Cases / Guardrails

| Condition | Behavior |
|----------|----------|
| `SLURM_JOB_ID` set | Raise error or refuse to detach |
| profile missing `account` | Same behavior as current slurm executor |
| dry-run | still supported (snakemake `--dry-run`) |

---

## Success Criteria

1. `analysis.run(execution_mode="login_detached")` works on login node.
2. Snakemake runs in background without terminal attachment.
3. SBATCH calls include profile `account` directive (no “guess account” warning).
4. The run produces the same outputs as standard `execution_mode="slurm"`.

---

## Testing Plan

1. **Unit test:** verify `login_detached` rejects runs inside SLURM.
2. **Unit test:** verify it spawns background process with `start_new_session=True`.
3. **Integration test:** run a small workflow on login node and confirm
   `sbatch` calls include `--account` and no nested SLURM warning appears.

---

## Open Questions

1. Should `login_detached` use `nohup` or `setsid` explicitly?
2. Should `analysis.run()` auto-fallback to `batch_job` if `login_detached`
   is requested from inside SLURM?
3. Should the CLI expose this mode immediately or keep it API-only?
