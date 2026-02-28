# Snakemake Lock Detection and Auto-Unlock

**Written**: 2026-02-28 00:12
**Last edited**: 2026-02-28 00:45 — fix return_code=None→1; add WorkflowError to required imports; skip lock check on dry_run; split validation into local + HPC

---

## Task Understanding

### Requirements

When a Snakemake workflow is killed mid-run (e.g., SLURM time limit, SIGKILL), Snakemake leaves stale lock files in `.snakemake/locks/`. On the next `submit_workflow()` call, Snakemake immediately exits with `LockException`. This wasted a full debug queue allocation in Run 8 of `frontier_sensitivity_suite` (Job 4156506 consumed queue time just to print `LockException` and exit).

The fix: detect a stale lock **in Python before any submission or launch action**, prompt the user for permission to unlock, and if granted, run `snakemake --unlock` before proceeding.

### Why not in the generated bash scripts

The original plan placed the lock check inside `run_workflow_1job.sh` and `run_workflow_tmux.sh`. This is wrong for `1_job_many_srun_tasks` mode: `run_workflow_1job.sh` is submitted via `sbatch`, so the check would run **inside the SLURM allocation** — after the job has already been granted and queued compute time consumed. The check needs to happen before `sbatch` is called, on the Python side.

For `batch_job` (tmux) mode, the script runs on the login node so the timing is less critical, but Python is still the right place — it's consistent, interactive (the Python session is present), and avoids duplicating logic across bash templates.

### Assumptions

1. "User input" means the interactive Python session (notebook or CLI) — the same context from which `submit_workflow()` is called.
2. The prompt is `input()` in Python — simple, no dependency needed.
3. The lock check runs before any submission action in all three call paths: `_submit_single_job_workflow`, `_submit_tmux_workflow`, and `run_snakemake_local`.
4. The snakemake `--unlock` command is run via `subprocess` using the same Python executable already in `_get_snakemake_base_cmd()`, with `cwd` set to `analysis_dir`.
5. Lock detection: presence of any `*.lock` file inside `.snakemake/locks/` relative to the analysis directory.

### Success Criteria

- If `.snakemake/locks/` contains lock files, a warning is printed and the user is prompted: `"Snakemake lock detected. Run snakemake --unlock and proceed? [y/N]: "`
- If the user responds `y` or `Y`: runs `snakemake --unlock`, then continues with submission.
- If the user responds anything else: raises a clear exception with the manual unlock command.
- If no lock files exist: proceeds silently, no change in behavior.
- Behavior is identical across `1_job_many_srun_tasks`, `batch_job`, and `local` modes.

---

## Evidence from Codebase

- **`workflow.py:84–90`** (`_get_snakemake_base_cmd`): Returns `[sys.executable, "-m", "snakemake"]`. The unlock command should use this same base.
- **`workflow.py:1507–1538`** (`_submit_single_job_workflow`): Calls `sbatch` at line 1533 after generating the script. Lock check must go before this `sbatch` call.
- **`workflow.py:1932–2106`** (`_submit_tmux_workflow`): Launches tmux at line 2106. Lock check must go before this launch.
- **`workflow.py:927–955`** (`run_snakemake_local`): Calls `subprocess.run(cmd_args, ...)` at line 948. Lock check must go before this call.
- **`workflow.py:1624–1810`** (`_generate_batch_orchestration_script` / `run_workflow_batch_job.sh`): Deprecated — emits `DeprecationWarning` at line 1648, not routed to by `submit_workflow`. Skip.
- **`SensitivityAnalysisWorkflowBuilder`**: Delegates all submission to `self._base_builder._submit_single_job_workflow()` and `self._base_builder._submit_tmux_workflow()`. Fixing the base methods covers sensitivity analysis automatically.
- **Existing helper method pattern**: `_get_snakemake_base_cmd`, `_get_config_args`, `_get_module_load_prefix` are all small private helpers on `SnakemakeWorkflowBuilder`. The new method fits this pattern.
- **Lock files confirmed**: `.snakemake/locks/0.input.lock` and `.snakemake/locks/0.output.lock` — the glob `*.lock` is the correct check.

---

## Implementation Strategy

### Chosen Approach: Private helper method on `SnakemakeWorkflowBuilder`, called at three sites

Add a single method `_check_and_clear_snakemake_lock(self, snakefile_path: Path, dry_run: bool, verbose: bool) -> None` to `SnakemakeWorkflowBuilder`. It:

1. If `dry_run=True`: returns immediately (no lock check — dry runs don't submit, and the real submission call will check)
2. Checks for `*.lock` files in `analysis_dir / ".snakemake" / "locks"`
3. If found, prints a warning listing the lock files
4. Prompts with `input()` for `y/N`
5. On `y`: runs `snakemake --unlock` via subprocess; raises on failure
6. On anything else: raises `WorkflowError` with the manual unlock command
7. If no lock files: returns immediately

Call this method near the top of each of the three submission methods, before any launch action.

### Alternatives Considered

- **In the bash script templates** — wrong: `run_workflow_1job.sh` runs inside the SLURM allocation, so the check wastes queued compute time on a LockException before it can act. Rejected.
- **In `submit_workflow()` itself** — tempting since it's a single call site for `1_job_many_srun_tasks` and `batch_job`, but `run_snakemake_local` is called directly (not via `submit_workflow`) in some paths, and the dry-run calls in `_validate_single_job_dry_run` also invoke `run_snakemake_local`. Better to put the check in the actual submission methods so it fires exactly once per real submission attempt.
- **Auto-unlock without prompting** — simpler, but could silently destroy a lock held by an actually-running concurrent process. The prompt is a one-line safety gate.

### Trade-offs

- `input()` blocks if called in a non-interactive context (e.g., a CI pipeline or a headless script). This is acceptable — the lock scenario only arises from interrupted HPC runs, which are always started interactively. If needed in future, a `force_unlock: bool = False` parameter could be added to bypass the prompt.

---

## File-by-File Change Plan

### `src/TRITON_SWMM_toolkit/workflow.py`

**One new method + three call sites, all in `SnakemakeWorkflowBuilder`.**

#### New method: `_check_and_clear_snakemake_lock`

Add after `_get_snakemake_base_cmd` (around line 90), following the existing helper method pattern:

```python
def _check_and_clear_snakemake_lock(self, snakefile_path: Path, dry_run: bool, verbose: bool = True) -> None:
    """Check for a stale Snakemake lock and prompt the user to clear it.

    Snakemake leaves lock files in .snakemake/locks/ when a workflow is
    killed (e.g. SLURM time limit). If not cleared before the next run,
    Snakemake exits immediately with LockException, wasting any queued
    compute allocation.

    Skipped when dry_run=True — dry runs don't submit anything, so a lock
    is not dangerous, and the real submission call will check again.

    Parameters
    ----------
    snakefile_path : Path
        Path to the Snakefile (used to build the --unlock command).
    dry_run : bool
        If True, skip the lock check entirely.
    verbose : bool
        If True, print status messages.

    Raises
    ------
    WorkflowError
        If lock files are found and the user declines to unlock, or if
        snakemake --unlock itself fails.
    """
    if dry_run:
        return
    locks_dir = self.analysis_paths.analysis_dir / ".snakemake" / "locks"
    lock_files = list(locks_dir.glob("*.lock")) if locks_dir.exists() else []
    if not lock_files:
        return

    lock_names = ", ".join(f.name for f in lock_files)
    print(f"[Snakemake] WARNING: Stale lock files detected in {locks_dir}:", flush=True)
    print(f"[Snakemake]   {lock_names}", flush=True)
    print(
        "[Snakemake] This usually means a previous job was killed before Snakemake "
        "could clean up.\n"
        "[Snakemake] Only unlock if no other Snakemake process is currently running "
        "in this directory.",
        flush=True,
    )

    response = input("[Snakemake] Run snakemake --unlock and proceed? [y/N]: ").strip()
    if response.lower() != "y":
        unlock_cmd = (
            f"{sys.executable} -m snakemake "
            f"--unlock --snakefile {snakefile_path}"
        )
        raise WorkflowError(
            phase="pre-submission lock check",
            return_code=1,  # sentinel: user aborted (WorkflowError requires int, not None)
            stderr=(
                "Workflow submission aborted. If no other Snakemake process is running, "
                f"unlock manually and retry:\n  {unlock_cmd}"
            ),
        )

    unlock_cmd = self._get_snakemake_base_cmd() + [
        "--unlock",
        "--snakefile", str(snakefile_path),
    ]
    if verbose:
        print(f"[Snakemake] Running: {' '.join(unlock_cmd)}", flush=True)

    result = subprocess.run(
        unlock_cmd,
        cwd=str(self.analysis_paths.analysis_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise WorkflowError(
            phase="snakemake --unlock",
            return_code=result.returncode,
            stderr=result.stderr,
        )
    if verbose:
        print("[Snakemake] Unlock successful. Proceeding.", flush=True)
```

**Required import change**: `WorkflowError` is not currently imported in `workflow.py` (only `ConfigurationError` is, at line 28). Update line 28 to:
```python
from TRITON_SWMM_toolkit.exceptions import ConfigurationError, WorkflowError
```

#### Call site 1: `_submit_single_job_workflow` (~line 1507)

Insert immediately after the verbose print and before `generate_snakemake_config`:

```python
# Check for stale lock before consuming a SLURM allocation
self._check_and_clear_snakemake_lock(snakefile_path, dry_run=False, verbose=verbose)
```

`_submit_single_job_workflow` is only called for real submissions (never dry-run), so `dry_run=False` is always correct here.

#### Call site 2: `_submit_tmux_workflow` (~line 1932)

Insert immediately after the tmux availability check and before the session name generation / launch logic:

```python
# Check for stale lock before launching tmux session
self._check_and_clear_snakemake_lock(snakefile_path, dry_run=False, verbose=verbose)
```

Same as above — `_submit_tmux_workflow` is never called for dry runs.

#### Call site 3: `run_snakemake_local` (~line 869)

Insert immediately before the `subprocess.run(cmd_args, ...)` call that runs Snakemake:

```python
# Check for stale lock before running Snakemake locally (skipped on dry runs)
self._check_and_clear_snakemake_lock(snakefile_path, dry_run=dry_run, verbose=verbose)
```

`run_snakemake_local` receives a `dry_run` parameter and is called for both real runs and dry-run validation. Passing `dry_run=dry_run` threads the flag through correctly — the lock check is skipped during `_validate_single_job_dry_run` and fires only on real local submissions.

---

## Risks and Edge Cases

| Risk | Mitigation |
|------|------------|
| `input()` blocks in non-interactive context (CI, headless script) | Acceptable for current use cases; can add `force_unlock: bool = False` kwarg in future if needed |
| Lock files held by a genuinely running concurrent Snakemake process | The prompt is the safety gate — user should verify before answering `y` |
| `snakemake --unlock` fails (e.g. permissions) | `WorkflowError` raised with stderr; user sees exact error |
| Dry-run calls to `run_snakemake_local` | Handled — `dry_run` parameter passed through; lock check returns immediately when `dry_run=True` |
| `WorkflowError` constructor signature | Confirmed: `(phase: str, return_code: int, stderr: str = "")` — plan uses `return_code=1` (sentinel) correctly |

---

## Validation Plan

The lock detection and prompt logic is fully testable locally via `run_snakemake_local` (the `local` multi_sim_run_method path). Only the "submission proceeds after unlock" step requires HPC for `1_job_many_srun_tasks` and `batch_job` modes.

### Local tests (run on dev machine)

```bash
# Setup: use any existing analysis dir with a Snakefile
mkdir -p <analysis_dir>/.snakemake/locks
touch <analysis_dir>/.snakemake/locks/0.input.lock
touch <analysis_dir>/.snakemake/locks/0.output.lock
```

1. Call `submit_workflow()` with `multi_sim_run_method="local"` — verify the warning prints and prompt appears **before** Snakemake is invoked.
2. Answer `y` — verify `snakemake --unlock` runs (exit 0), lock files are removed, and Snakemake proceeds normally.
3. Answer `n` (or just press Enter) — verify `WorkflowError` is raised with the manual unlock command in the message.
4. Repeat with no lock files present — verify no prompt, no behavior change.
5. Call with `dry_run=True` while lock files exist — verify no prompt fires (lock check skipped).

### HPC verification (Frontier or UVA, after local tests pass)

6. With `multi_sim_run_method="1_job_many_srun_tasks"` and artificial lock files: call `submit_workflow()` — verify prompt fires before `sbatch` is invoked (i.e., no job ID is printed).
7. Answer `y` — verify unlock runs, then `sbatch` is called and a job ID is returned.
8. Repeat for `batch_job` mode — verify prompt fires before tmux session is created.

---

## Documentation and Tracker Updates

No architecture or convention changes needed.

---

## Decisions Needed from User

None.

---

## Definition of Done

- [x] `WorkflowError` added to import at `workflow.py:28`
- [x] `_check_and_clear_snakemake_lock(snakefile_path, dry_run, verbose)` method added to `SnakemakeWorkflowBuilder`
- [x] `dry_run=True` returns immediately (no prompt, no check)
- [x] `return_code=1` used in "user declined" `WorkflowError` (not `None`)
- [x] Called in `_submit_single_job_workflow` with `dry_run=False` before `sbatch`
- [x] Called in `_submit_tmux_workflow` with `dry_run=False` before tmux launch
- [x] Called in `run_snakemake_local` with `dry_run=dry_run` before `subprocess.run`
- [ ] `ruff format` and `ruff check` pass — `ruff format` applied; `ruff check` has 24 pre-existing violations in `workflow.py` (unused imports, f-strings, line-length). None introduced by this change. Tracked in `docs/planning/bugs/tech_debt_ruff_violations.md`.
- [x] Local tests 1–5 pass (prompt fires, unlock runs, n aborts, clean run silent, dry-run silent) — 2026-02-28
- [ ] HPC tests 6–8 verified (prompt fires before sbatch/tmux in all HPC modes)

---

## Self-Check Results

1. **Header/body alignment**: All sections match content.
2. **Section necessity**: All sections are present and non-redundant.
3. **Alignment with conventions**: Single helper method, no defaults except `verbose`, uses existing `WorkflowError` hierarchy, follows existing private helper pattern in `SnakemakeWorkflowBuilder`. Fail-fast with exception on declined unlock.
4. **Task-relevance**: Tightly scoped — one method, three call sites, no new abstractions.
