# Plan: Tmux-Based Workflow Orchestration for batch_job Mode

**Status:** Complete
**Created:** 2026-02-16
**Replaces:** Previous sbatch-based orchestrator approach

## Problem Statement

The current `batch_job` mode runs Snakemake inside an sbatch job. This creates **critical cancellation issues**:

1. **Orphaned jobs**: Canceling the orchestrator sbatch job does NOT cancel worker jobs
2. **Snakemake warning**: Snakemake explicitly warns against running inside SLURM context
3. **Signal handling broken**: Ctrl+C and `cancel_jobs()` don't work when orchestrator is killed via `scancel`
4. **Complex workarounds needed**: Would require job dependencies, UUID tracking, or comment-based parsing

**Snakemake's own documentation states:** Running Snakemake in sbatch is "not recommended, as it may lead to unexpected behavior."

## Solution: Tmux-Based Orchestration

Use **tmux sessions** to run Snakemake on the login node in a detached, persistent session. This approach:

- ✅ Enables proper signal handling (Ctrl+C works)
- ✅ Triggers Snakemake's built-in `cancel_jobs()` for worker cleanup
- ✅ Avoids nested SLURM context issues
- ✅ Allows reconnection to see live output
- ✅ Follows Snakemake best practices
- ✅ Simple `cancel()` implementation: just send SIGINT to the process

## Architecture Overview

### Current (sbatch-based):
```
User Python process
  └─> sbatch run_workflow.sh  (Job 1111111)
       └─> snakemake --executor slurm
            ├─> sbatch worker1  (Job 2222222)
            ├─> sbatch worker2  (Job 3333333)
            └─> sbatch worker3  (Job 4444444)

Problem: scancel 1111111 → Jobs 2222222+ become orphaned!
```

### New (tmux-based):
```
User Python process
  └─> tmux new-session -d "snakemake --executor slurm"
       └─> snakemake process (PID 12345)
            ├─> sbatch worker1  (Job 2222222)
            ├─> sbatch worker2  (Job 3333333)
            └─> sbatch worker3  (Job 4444444)

Solution: kill -SIGINT 12345 → Snakemake's cancel_jobs() cancels all workers!
```

## Implementation Design

### 1. Tmux Session Management

**Session naming convention:**
```
triton_swmm_{analysis_id}_{timestamp}
```

Example: `triton_swmm_norfolk_multi_sim_20260216_143052`

**Why this naming:**
- Includes `analysis_id` for easy identification
- Timestamp prevents conflicts from multiple runs
- `triton_swmm_` prefix for filtering all toolkit sessions

### 2. Submission Flow

```python
def _submit_tmux_workflow(
    self,
    snakefile_path: Path,
    wait_for_completion: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Submit Snakemake workflow in detached tmux session.

    Returns
    -------
    dict
        - success: bool
        - mode: str ("tmux")
        - session_name: str
        - snakemake_pid: int
        - message: str
    """
```

**Steps:**
1. Generate unique session name
2. Build Snakemake command with full paths (no relative paths)
3. Create detached tmux session: `tmux new-session -d -s {session_name} "{command}"`
4. Extract Snakemake PID from tmux session
5. Save session name + PID to analysis log
6. Optionally wait for completion (poll tmux session)

### 3. Persistent Logging

Add to `TRITONSWMM_analysis_log`:

```python
# Tmux workflow tracking
tmux_session_name: LogField[str] = Field(default_factory=LogField)
snakemake_pid: LogField[int] = Field(default_factory=LogField)
workflow_submission_time: LogField[str] = Field(default_factory=LogField)
workflow_submission_mode: LogField[str] = Field(default_factory=LogField)  # "tmux"
workflow_canceled: LogField[bool] = Field(default_factory=LogField)
workflow_cancellation_time: LogField[str] = Field(default_factory=LogField)
```

### 4. Cancellation Flow

```python
def cancel(self, verbose: bool = True, wait_timeout: int = 30) -> dict:
    """Cancel tmux-based workflow."""

    # 1. Check if tmux session exists
    session_name = self.log.tmux_session_name.get()
    if not session_name or not tmux_session_exists(session_name):
        return {"success": True, "jobs_were_running": False, ...}

    # 2. Get Snakemake PID from tmux session
    snakemake_pid = get_pid_in_tmux_session(session_name)

    # 3. Send SIGINT to Snakemake process
    os.kill(snakemake_pid, signal.SIGINT)

    # 4. Wait for Snakemake to finish canceling jobs
    wait_for_process_exit(snakemake_pid, timeout=wait_timeout)

    # 5. Verify worker jobs are canceled
    verify_no_worker_jobs_running()

    # 6. Kill tmux session
    tmux.kill_session(session_name)

    # 7. Update log
    self.log.workflow_canceled.set(True)
    self.log.workflow_cancellation_time.set(datetime.datetime.now().isoformat())
```

**Why this works:**
- SIGINT triggers Snakemake's `cancel_jobs()` method
- Snakemake calls `scancel` on all tracked worker job IDs
- Workers are cleaned up automatically by Snakemake itself
- No UUID tracking or comment parsing needed

### 5. Status Checking

```python
def get_workflow_status(self, verbose: bool = False) -> dict:
    """Check tmux workflow status."""

    session_name = self.log.tmux_session_name.get()

    if not session_name:
        return {"status": "never_submitted", ...}

    if not tmux_session_exists(session_name):
        return {"status": "completed", ...}

    # Session exists, check if Snakemake is still running
    snakemake_pid = get_pid_in_tmux_session(session_name)
    if process_exists(snakemake_pid):
        return {"status": "running", "pid": snakemake_pid, ...}
    else:
        return {"status": "crashed", ...}
```

## Configuration Changes

### Analysis Config

Add new field to `AnalysisConfig`:

```python
multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks", "tmux"]
```

**Migration strategy:**
- Default to `"tmux"` for new configs
- Existing `"batch_job"` configs continue to work (deprecated)
- Print warning when using `"batch_job"` suggesting migration to `"tmux"`

### Backward Compatibility

Keep `_submit_batch_job_workflow()` method but mark as deprecated:

```python
def _submit_batch_job_workflow(...):
    """
    DEPRECATED: Use tmux mode instead.

    This method runs Snakemake inside an sbatch job, which causes
    orphaned worker jobs when canceled. See docs for migration guide.
    """
    warnings.warn(
        "batch_job mode is deprecated due to orphaned job issues. "
        "Please use multi_sim_run_method='tmux' instead.",
        DeprecationWarning
    )
    # ... existing implementation
```

## Tmux Command Reference

### Key Operations

**Create detached session:**
```bash
tmux new-session -d -s session_name "command"
```

**Check if session exists:**
```bash
tmux has-session -t session_name 2>/dev/null
echo $?  # 0 = exists, 1 = doesn't exist
```

**Get PID of process in session:**
```bash
tmux list-panes -t session_name -F "#{pane_pid}"
# Returns PID of shell, need to get child processes
pgrep -P <shell_pid> -f snakemake
```

**Attach to session (for debugging):**
```bash
tmux attach -t session_name
# Detach with Ctrl+B, D
```

**Kill session:**
```bash
tmux kill-session -t session_name
```

**List all sessions:**
```bash
tmux list-sessions
```

## Error Handling

### Scenario: Tmux not available

```python
if not shutil.which("tmux"):
    raise EnvironmentError(
        "tmux is required for batch_job mode but not found in PATH. "
        "Please install tmux or use multi_sim_run_method='local'."
    )
```

### Scenario: Session name conflict

Use timestamp in session name to avoid conflicts. If session exists:
```python
if tmux_session_exists(session_name):
    raise WorkflowError(
        f"Tmux session '{session_name}' already exists. "
        "Please check if another workflow is running or kill the session manually."
    )
```

### Scenario: Can't find Snakemake PID

```python
if not snakemake_pid:
    warnings.warn(
        "Could not find Snakemake PID in tmux session. "
        "Cancellation may be incomplete. Killing tmux session anyway."
    )
    tmux.kill_session(session_name)
```

## Testing Strategy

### Unit Tests

**Test 1: Session creation and naming**
```python
def test_tmux_session_naming():
    session_name = workflow._generate_tmux_session_name()
    assert session_name.startswith("triton_swmm_")
    assert analysis.cfg_analysis.analysis_id in session_name
```

**Test 2: PID extraction**
```python
def test_get_snakemake_pid_from_tmux():
    # Create test session
    # Launch dummy process
    # Extract PID
    # Verify correct PID
```

**Test 3: Cancellation with SIGINT**
```python
def test_cancel_sends_sigint():
    # Submit workflow
    # Call cancel()
    # Verify SIGINT was sent to Snakemake PID
    # Verify session cleaned up
```

### Integration Tests (UVA)

**Test 4: Full workflow submission and cancellation**
```python
def test_UVA_tmux_workflow_submit_and_cancel():
    analysis.submit_workflow()
    time.sleep(30)  # Let workers start

    # Verify session exists
    assert tmux_session_exists(analysis.log.tmux_session_name.get())

    # Cancel
    result = analysis.cancel()
    assert result["success"]

    # Verify workers are gone
    assert no_worker_jobs_in_squeue(analysis.cfg_analysis.analysis_id)
```

**Test 5: Cross-session cancellation**
```python
def test_UVA_cancel_from_new_session():
    # Submit in session 1
    analysis1 = create_analysis()
    analysis1.submit_workflow()

    # Reload in session 2 (different Python process)
    analysis2 = create_analysis()  # Loads log with tmux_session_name
    result = analysis2.cancel()

    assert result["success"]
```

## Migration Guide (for users)

### Before (sbatch mode):
```yaml
# analysis_config.yaml
multi_sim_run_method: batch_job
hpc_total_job_duration_min: 1440
```

### After (tmux mode):
```yaml
# analysis_config.yaml
multi_sim_run_method: tmux
# Remove: hpc_total_job_duration_min (not needed for tmux)
```

**No other changes required** - submission and cancellation API stays the same:
```python
analysis.submit_workflow(wait_for_completion=False)
analysis.cancel()
```

## Advantages Summary

| Aspect | sbatch Mode | tmux Mode |
|--------|-------------|-----------|
| **Orphaned jobs** | ❌ Workers keep running | ✅ Workers auto-canceled |
| **Signal handling** | ❌ Broken (scancel kills process) | ✅ Works (SIGINT) |
| **Snakemake support** | ⚠️ Explicitly not recommended | ✅ Recommended approach |
| **cancel() complexity** | ❌ Requires UUID/comment parsing | ✅ Simple SIGINT |
| **Cross-session cancel** | ⚠️ Possible but complex | ✅ Simple (send SIGINT) |
| **Live monitoring** | ❌ Can't easily see output | ✅ `tmux attach` |
| **Debugging** | ❌ Logs only | ✅ Interactive session |

## Implementation Phases

### Phase 1: Core tmux submission (This PR)
- [ ] Add `_submit_tmux_workflow()` method
- [ ] Add tmux session name generation
- [ ] Add PID extraction logic
- [ ] Add log fields for tmux tracking
- [ ] Update `submit_workflow()` to route to tmux when `multi_sim_run_method="tmux"`

### Phase 2: Cancellation (This PR)
- [ ] Update `cancel()` to handle tmux sessions
- [ ] Implement SIGINT sending
- [ ] Implement wait-for-exit verification
- [ ] Add worker job verification

### Phase 3: Status and monitoring (This PR)
- [ ] Update `get_workflow_status()` for tmux sessions
- [ ] Add helper to attach to session for debugging
- [ ] Add session cleanup utilities

### Phase 4: Testing and migration (Follow-up)
- [ ] Add unit tests
- [ ] Add UVA integration tests
- [ ] Deprecate batch_job mode
- [ ] Update documentation
- [ ] Migrate example configs

## Open Questions

1. **Should we auto-detect if running on login node vs in allocation?**
   - For now: Let tmux run regardless, it works in both contexts
   - Future: Could add warning if inside SLURM allocation

2. **How to handle multiple simultaneous workflows?**
   - Timestamp in session name prevents conflicts
   - Each analysis has unique session

3. **What if user manually kills tmux session?**
   - Worker jobs become orphaned (same as sbatch problem)
   - Document: Always use `analysis.cancel()` for cleanup
   - Could add cron job to detect orphaned sessions

4. **Should we support tmux on Frontier?**
   - Yes - tmux is standard on most HPC systems
   - Check availability at submit time and fail fast

## Success Criteria

- [ ] Can submit workflow via tmux from login node
- [ ] Can cancel workflow and verify all workers are cleaned up
- [ ] Can cancel from new Python session (cross-session works)
- [ ] Can attach to tmux session to see live Snakemake output
- [ ] Orphaned job problem is eliminated
- [ ] Implementation is simpler than sbatch approach (less code)
