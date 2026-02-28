# SLURM Cancellation Function

**Written**: 2026-02-16
**Status**: ✅ Implemented

---

## What Was Done

Implemented `analysis.cancel()` for canceling `batch_job` workflows with cross-session persistence and verified termination. Added `analysis.get_slurm_job_status()` as a companion status checker. Customized SLURM job names to include `analysis_id` for easy identification in `squeue`.

**Simple usage**:
```python
tk = Toolkit("system.yaml", "analysis.yaml")
tk.analysis.cancel()  # Handles everything — no status check needed first
```

---

## What Was Built

### Persistent log fields (`log.py`, lines 577–581)

Added 5 `LogField` fields to `TRITONSWMM_analysis_log`:

```python
orchestrator_job_id: LogField[str]           # SLURM job ID of orchestrator
orchestrator_submission_time: LogField[str]  # ISO timestamp of submission
orchestrator_submission_mode: LogField[str]  # Value: "batch_job"
workflow_canceled: LogField[bool]            # Cancellation flag
workflow_cancellation_time: LogField[str]    # ISO timestamp of cancellation
```

Each `.set()` call triggers `log.write()`, persisting to `{analysis_dir}/log.json` immediately — enabling cross-session cancellation.

### Job name customization (`workflow.py`)

| Location | Change | Example result |
|----------|--------|----------------|
| Line 627 — worker jobs | `"job-name": f"{analysis_id}_{rule}"` | `norfolk_multi_sim_run_triton` |
| Line 1752 — orchestrator | `#SBATCH --job-name={analysis_id}_orchestrator` | `norfolk_multi_sim_orchestrator` |

All jobs queryable by `analysis_id` prefix via `squeue` and `scancel`.

### Job ID persistence (`workflow.py`, lines 1807–1816)

After successful sbatch submission, orchestrator job ID and timestamp are written to the analysis log automatically.

### `analysis.cancel()` (`analysis.py`, lines 2106–2438)

```python
def cancel(self, verbose: bool = True, wait_timeout: int = 30) -> dict
```

Behavior:
- Validates `multi_sim_run_method == "batch_job"` (raises `ValueError` otherwise)
- Pre-checks if jobs are running; exits gracefully if nothing to cancel
- Sends `scancel` to orchestrator by ID; `scancel --name "{analysis_id}_*"` for workers
- Polls `squeue` every 2s until jobs terminate (up to `wait_timeout`)
- Updates log with cancellation flag and timestamp

Return dict keys: `success`, `orchestrator_canceled`, `workers_canceled`, `jobs_were_running`, `orchestrator_job_id`, `analysis_id`, `message`, `errors`.

### `analysis.get_slurm_job_status()` (`analysis.py`, lines 2440–2539)

```python
def get_slurm_job_status(self, verbose: bool = False) -> dict
```

Queries orchestrator status via `scontrol`, counts active worker jobs in `squeue`, returns submission time and cancellation flag from log. Renamed from original plan's `get_workflow_status()` to avoid conflict with existing method.

### Tests (`tests/test_workflow_cancellation.py`)

- Error handling for non-batch_job modes
- Graceful handling when no jobs running
- Status checking before submission
- Log field persistence across sessions
- Cancellation flag persistence

### Example script (`examples/cancel_workflow_example.py`)

Four usage examples: same-session cancel, cross-session cancel, check-status-first, error handling.

---

## Key Design Decision: LogField for Persistence

Orchestrator metadata stored in `TRITONSWMM_analysis_log` via the existing `LogField[T]` pattern rather than a separate `~/.triton_swmm/workflows` registry. The analysis log already persists to `log.json` and is automatically loaded on reconstruction — no extra infrastructure needed.

---

## Known Limitations

- Only supports `batch_job` mode (raises `ValueError` for `local`, `1_job_many_srun_tasks`)
- Requires SLURM commands (`squeue`, `scancel`, `scontrol`)
- Jobs may still be terminating after 30s timeout (warning issued, not failure)

---

## Definition of Done

- [x] 5 `LogField` fields added to `TRITONSWMM_analysis_log`
- [x] Worker job names include `analysis_id` prefix
- [x] Orchestrator job name includes `analysis_id`
- [x] Job ID persisted to log after sbatch submission
- [x] `analysis.cancel()` implemented with verified termination
- [x] `analysis.get_slurm_job_status()` implemented
- [x] `tests/test_workflow_cancellation.py` created (5 tests)
- [x] `examples/cancel_workflow_example.py` created
