# Implementation: wait_for_completion for 1_job_many_srun_tasks Mode

## Overview

This document describes the implementation of the `wait_for_completion` parameter for the `1_job_many_srun_tasks` mode in the TRITON-SWMM toolkit. Previously, this mode would always return immediately after submitting the SLURM job, regardless of the `wait_for_completion` parameter setting.

## Problem Statement

The `wait_for_completion` parameter in `analysis.submit_workflow()` was ignored for `multi_sim_run_method = "1_job_many_srun_tasks"` mode. This caused tests that expected to check workflow outputs immediately after submission to fail because the job was still running or had not yet completed.

### Before
```python
result = analysis.submit_workflow(
    mode="auto",
    wait_for_completion=True,  # ❌ IGNORED - had no effect
    ...
)
# Returns immediately, even though job is still running
```

### After
```python
result = analysis.submit_workflow(
    mode="auto",
    wait_for_completion=True,  # ✅ NOW WORKS - blocks until job completes
    ...
)
# Blocks until SLURM job completes
assert result["success"] is True  # Job finished successfully
assert result["completed"] is True
assert result["exit_code"] == 0
```

## Implementation Details

### 1. New Method: `_wait_for_slurm_job_completion()`

**Location**: `src/TRITON_SWMM_toolkit/workflow.py:744-851`

This method polls for SLURM job completion using a two-stage approach:

#### Stage 1: Query Active Jobs
```python
squeue -j <job_id> -h -o "%T"
```
- Checks for jobs still in queue (PENDING, RUNNING, CONFIGURING, COMPLETING)
- Only works for active jobs
- Allows us to show real-time status updates

#### Stage 2: Query Historical Records
```python
sacct -j <job_id> -n -X -o "State,ExitCode"
```
- Retrieves completed job information from accounting database
- Returns state (COMPLETED, FAILED, CANCELLED, etc.) and exit code
- Persists after job leaves queue

#### Key Features
- **Smart Polling**: Only prints status when job state changes (prevents log spam)
- **Timeout Support**: Optional timeout parameter for long-running jobs
- **Exit Code Parsing**: Correctly extracts exit code from `sacct` output format (e.g., "1:0" → 1)
- **Failure Detection**: Distinguishes between successful completion (state=COMPLETED, exit=0) and failures

#### Return Value
```python
{
    "completed": bool,        # True if job finished successfully
    "state": str,             # SLURM state (COMPLETED, FAILED, TIMEOUT, etc.)
    "exit_code": int | None,  # Job exit code (None if timeout)
    "message": str            # Human-readable status
}
```

### 2. Updated Method: `_submit_single_job_workflow()`

**Location**: `src/TRITON_SWMM_toolkit/workflow.py:853-980`

#### New Parameters
- `wait_for_completion: bool = False` - Optional blocking behavior
- Updated docstring to document new return fields

#### New Behavior
1. Submits SBATCH job with `sbatch` command
2. Parses job ID from output
3. If `wait_for_completion=True`:
   - Calls `_wait_for_slurm_job_completion()`
   - Updates result dictionary with completion info
   - Sets `result["success"]` based on job completion status
4. If `wait_for_completion=False`:
   - Returns immediately (original behavior)

#### Return Value Structure
When `wait_for_completion=True`, adds to result:
```python
{
    "success": True,           # Based on job completion status
    "completed": True,         # True if COMPLETED with exit 0
    "state": "COMPLETED",      # SLURM job state
    "exit_code": 0,            # Job exit code
    ...
}
```

### 3. Updated Method: `submit_workflow()`

**Location**: `src/TRITON_SWMM_toolkit/workflow.py:1098-1102`

The public API now passes `wait_for_completion` through:
```python
result = self._submit_single_job_workflow(
    snakefile_path=snakefile_path,
    wait_for_completion=wait_for_completion,  # ✅ Now passed through
    verbose=verbose,
)
```

## Usage Examples

### Example 1: Non-blocking Submission (Default)
```python
# Don't wait - job runs in background
result = analysis.submit_workflow(
    wait_for_completion=False  # or omit (default)
)

print(f"Job {result['job_id']} submitted")
# Can check logs while job runs
import subprocess
subprocess.run(["tail", "-f", f"{analysis.analysis_dir}/snakemake.log"])
```

### Example 2: Blocking Submission
```python
# Wait for job to complete
result = analysis.submit_workflow(
    wait_for_completion=True
)

if result["success"]:
    print(f"✓ Job {result['job_id']} completed successfully")
    # Safe to check outputs now
    assert (analysis.analysis_dir / "output.nc").exists()
else:
    print(f"✗ Job failed: {result['message']}")
```

### Example 3: Test with Wait
```python
def test_snakemake_workflow(frontier_multisim_analysis):
    """Test workflow execution with automatic completion checking."""
    analysis = frontier_multisim_analysis

    result = analysis.submit_workflow(
        mode="auto",
        wait_for_completion=True,  # ✅ Now works!
        verbose=True,
    )

    # Can now safely check outputs
    assert result["success"], f"Job failed: {result['message']}"
    assert result["completed"] is True
    assert result["exit_code"] == 0

    # Verify workflow outputs exist
    assert (analysis.analysis_dir / "_status" / "output_consolidation_complete.flag").exists()
```

## Testing

### Unit Tests: `tests/test_workflow_single_job_wait.py`

14 comprehensive tests covering:

**Job Monitoring Tests** (6 tests)
- Successful completion detection
- Failure detection
- Timeout handling
- State transitions (PENDING → RUNNING → COMPLETING → COMPLETED)
- Cancelled job detection
- Exit code parsing

**Submission Tests** (6 tests)
- Submission without waiting
- Submission with wait (success case)
- Submission with wait (failure case)
- SBATCH submission failure handling
- Invalid job ID parsing
- Wait behavior with unparseable job ID

**Integration Tests** (2 tests)
- Parameter passing through call chain
- Default behavior (no wait)

### Run Tests
```bash
# Run all new tests
pytest tests/test_workflow_single_job_wait.py -v

# Run specific test class
pytest tests/test_workflow_single_job_wait.py::TestWaitForSlormJobCompletion -v

# Run with coverage
pytest tests/test_workflow_single_job_wait.py --cov=TRITON_SWMM_toolkit.workflow
```

### Integration Testing on Frontier

```bash
# SSH to Frontier (or in salloc allocation)
cd /path/to/TRITON-SWMM_toolkit

# Run Frontier tests with wait_for_completion
pytest tests/test_frontier_03_snakemake_multisim.py -v -s

# Monitor specific test
pytest tests/test_frontier_03_snakemake_multisim.py::test_snakemake_workflow_execution -v -s
```

Expected output:
```
[Snakemake] Submitted job 12345
[Snakemake] Waiting for SLURM job 12345 to complete...
[Snakemake] [0s] Job 12345: PENDING
[Snakemake] [5s] Job 12345: RUNNING
[Snakemake] [125s] Job 12345: COMPLETING
[Snakemake] [130s] Job 12345: COMPLETED ✓
```

## Impact Analysis

### Backward Compatibility ✅
- Default behavior unchanged: `wait_for_completion=False`
- Existing code without `wait_for_completion` parameter works as before
- Return value is backward compatible (new fields only added when waiting)

### Performance
- Polling interval: 30 seconds (configurable via method parameters)
- SLURM command overhead: ~1-2 seconds per query
- Minimal impact during polling (jobs run in background)

### Error Handling
- Graceful timeout support
- Clear error messages for failures
- Job state properly returned in all cases

## Known Limitations

1. **Polling Granularity**: 30-second intervals mean completion may be detected up to 30s after actual completion
   - Acceptable for typical multi-hour simulations
   - Configurable in method call if needed

2. **SLURM Command Availability**: Requires `squeue` and `sacct` on execution system
   - Standard on all major HPC clusters
   - Works on Frontier, UVA, and other SLURM systems

3. **Job Record Retention**: `sacct` relies on accounting database
   - Records typically retained 30+ days
   - Sufficient for all use cases

## Migration Guide

### For Test Writers
If you wrote tests that polled for job completion manually, you can now simplify:

**Before (Manual Polling)**
```python
result = analysis.submit_workflow(wait_for_completion=False)
job_id = result['job_id']

import subprocess, time
while True:
    check = subprocess.run(['squeue', '-j', job_id, '-h'], capture_output=True, text=True)
    if not check.stdout.strip():
        break
    time.sleep(30)

# Now check outputs
tst_ut.assert_analysis_completed(analysis)
```

**After (Using Built-in Wait)**
```python
result = analysis.submit_workflow(wait_for_completion=True)
assert result["success"] is True

# Now check outputs
tst_ut.assert_analysis_completed(analysis)
```

### For Production Users
No changes required unless you want to wait for jobs:

```python
# Old code - still works
result = analysis.submit_workflow(...)

# New option - wait for completion
result = analysis.submit_workflow(wait_for_completion=True)
```

## Troubleshooting

### Job Never Completes
```python
# Use timeout parameter
result = workflow_builder._wait_for_slurm_job_completion(
    job_id="12345",
    timeout=3600,  # 1 hour max
    verbose=True
)
```

### SLURM Commands Fail
If `squeue` or `sacct` fail, check:
1. SLURM environment is loaded
2. User has permissions to query jobs
3. Job ID is valid

### Need to Debug Job Status
```python
# Manual job status queries
subprocess.run(["squeue", "-j", job_id])
subprocess.run(["sacct", "-j", job_id, "-l"])
```

## References

- SLURM Documentation: https://slurm.schedmd.com/
- squeue Manual: `man squeue`
- sacct Manual: `man sacct`
- Plan Document: `docs/one_job_many_srun_tasks_plan.md`

## Maintenance Notes

When updating this code, ensure:
1. Maintain backward compatibility with `wait_for_completion=False`
2. Update tests if polling logic changes
3. Test on both UVA and Frontier (different sacct behaviors)
4. Document any new parameters or return fields
