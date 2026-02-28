# Plan: Automatically Unlock Snakemake After Interruptions

**Status:** Draft (implementation-ready)
**Owner:** Toolkit maintainers
**Created:** 2026-02-13

## Goal

Automatically detect and clear Snakemake lock files when a multi-sim Snakemake
workflow was interrupted (e.g., orchestrator job killed) so reruns proceed
without manual `--unlock` intervention.

## Problem Summary

When Snakemake is terminated unexpectedly, it can leave an active lock in the
workflow directory. Subsequent attempts to resume the run fail unless the user
manually executes:

```bash
snakemake --unlock
```

This interrupts batch workflows and complicates recovery. We need to detect a
lock condition and run an unlock step automatically before restarting.

**Lock behavior notes (from Snakemake FAQ):**
- Locks are created in `.snakemake/locks/` under the workflow directory.
- Lock files include lists of input/output files for the current run.
- Locks normally clear via `unlock()` on clean shutdown.
- Stale locks occur when the process is killed (SIGKILL, node eviction, etc.).
- The supported cleanup method is `snakemake --unlock`, which removes the lock
  directory via `Persistence.cleanup_locks()`.

## Proposed Fix (High Confidence)

Add a **preflight lock check** in the Snakemake orchestration flow:

1. Detect whether a lock exists in the Snakemake working directory.
2. If a lock is present, run `snakemake --unlock` with the same config/profile.
3. Resume the normal Snakemake invocation after the unlock completes.

This should be a no-op when no lock is present.

---

## Requirements

- Works in multi-sim batch mode (Slurm and local).
- Unlock uses the same workflow directory, Snakefile, and config as the run.
- Avoids infinite loops (unlock once per run attempt).
- Emits a clear log message when an unlock is performed.
- Uses the supported cleanup path (`snakemake --unlock`) instead of deleting
  lock files directly.

---

## Implementation Plan

### 1) Identify Snakemake Invocation Entry Points

Search for the orchestration methods that run Snakemake (expected in
`src/TRITON_SWMM_toolkit/workflow.py`, possibly `run_snakemake()` or similar).
These functions should be updated to include the lock preflight.

### 2) Add Lock Detection Helper

**File:** `src/TRITON_SWMM_toolkit/workflow.py` (or `utils.py` if used elsewhere)

```python
def _snakemake_lock_exists(workdir: Path) -> bool:
    """Return True if Snakemake lock files exist in the workflow directory."""
    lock_dir = workdir / ".snakemake" / "locks"
    if not lock_dir.exists():
        return False
    return any(lock_dir.iterdir())
```

Notes:
- Snakemake creates `.snakemake/locks/` with files when locked.
- If the directory exists but empty, treat as unlocked.
- A lock is considered active when any lock file overlaps with the run’s
  inputs/outputs; this check is still a safe proxy for detecting stale locks.

### 3) Add Unlock Helper

```python
def _unlock_snakemake(
    snakefile: Path,
    workdir: Path,
    configfile: Path | None,
    profile: Path | None,
    extra_args: list[str] | None = None,
) -> None:
    """Run `snakemake --unlock` with consistent configuration."""
    cmd = ["snakemake", "--unlock", "-s", str(snakefile), "--directory", str(workdir)]
    if configfile:
        cmd += ["--configfile", str(configfile)]
    if profile:
        cmd += ["--profile", str(profile)]
    if extra_args:
        cmd += extra_args
    subprocess.run(cmd, check=True)
```

Notes:
- Uses Snakemake’s supported cleanup (`Persistence.cleanup_locks()`), not manual
  deletion.
- Only run once per attempt to avoid re-entrant unlock loops.

### 4) Integrate Preflight in Snakemake Runner

At the start of the Snakemake execution function:

```python
if _snakemake_lock_exists(workdir):
    logger.warning("Detected Snakemake lock; running unlock before restart.")
    _unlock_snakemake(...)
```

Then proceed with the normal Snakemake command.

### 5) Logging and UX

- Log a concise warning when a lock is detected and unlocked.
- If unlock fails, surface the exception so the job stops clearly.

---

## Validation Plan

1. Simulate a lock by running Snakemake and terminating it mid-run.
2. Re-run the workflow and confirm:
   - The toolkit detects the lock.
   - `snakemake --unlock` is issued automatically.
   - The workflow resumes without manual steps.

---

## Success Criteria

- Multi-sim Snakemake runs recover cleanly after an interruption.
- No manual `--unlock` required when lock files exist.
- Unlock step is only executed when needed.

---

## Implementation Checklist

- [ ] Locate Snakemake orchestration entry points in `workflow.py` (or related modules)
- [ ] Add `_snakemake_lock_exists()` helper for lock detection
- [ ] Add `_unlock_snakemake()` helper to run `snakemake --unlock`
- [ ] Integrate preflight in Snakemake execution flow with logging
- [ ] Validate recovery from an interrupted run