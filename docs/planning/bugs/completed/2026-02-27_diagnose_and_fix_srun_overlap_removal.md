# Implementation Plan: Diagnose and Fix srun --overlap Removal

## Task Understanding

### Requirements

Commit `54b8529` removed `--overlap` (and changed `--cpu-bind=none` → `--cpu-bind=cores`) from all `srun` invocations inside `run_simulation.py`. The `test_uva_sensitivity_suite_cpu` sensitivity run on UVA HPC failed completely (0/20 simulations) immediately after this commit, with errors of the form:

```
srun: error: Unable to create step for job XXXXX: Job/step already completing or completed
```

The task is to:
1. Systematically validate, via test sbatch scripts on UVA, which srun argument combinations work correctly across the four execution modes (serial, openmp, mpi, hybrid) under `batch_job` execution
2. Restore correct srun arguments in `run_simulation.py` based on what is empirically validated
3. Increase `hpc_time_min_per_sim` in the test config to avoid time-limit masking

### Assumptions

- The `--overlap` flag is required for `batch_job` mode on UVA's standard partition because it allows the srun step to share the parent job's allocation without requesting exclusive sub-resources
- The `--cpu-bind=cores` change is likely benign or beneficial and should be kept (pending validation)
- `1_job_many_srun_tasks` mode uses a different resource model and may not need `--overlap`; however, reverting it requires understanding whether `--overlap` is needed there too

### Success Criteria

- All 20 sub-analyses in `test_uva_sensitivity_suite_cpu` complete successfully
- The srun command arguments in `run_simulation.py` are backed by empirical test results from UVA
- `hpc_time_min_per_sim` is at a reasonable minimum (≥ 10 min) for the benchmarking test

---

## Evidence from Codebase

- **`src/TRITON_SWMM_toolkit/run_simulation.py` (lines 527–556)**: The srun command is constructed in `prepare_simulation_command()`. After 54b8529, both CPU and GPU branches have `--cpu-bind=cores` but no `--overlap`. Before 54b8529 they had `--cpu-bind=none --overlap`.

- **`.debugging/test_uva_sensitivity_suite_cpu/logs/sims/simulation_sa0_evt0.log`**: Confirms the exact srun command used and the `"Unable to create step for job ... already completing or completed"` error, occurring ~106 seconds into a 2-minute job.

- **`.snakemake/slurm_logs/rule_simulation_sa0_evt0/9779623.log`**: Shows `Provided cores: 1`, `runtime=2`, and the `DUE TO TIME LIMIT` cancellation. The job ran until the time limit while srun was waiting for a step allocation that never materialized.

- **`sensitivity_analysis_definition.csv`**: 20 sub-analyses covering serial, openmp, mpi, hybrid modes; 3 use the `parallel` partition. Relevant because some modes (mpi/hybrid) request `tasks > 1` and have different srun resource semantics.

- **`src/TRITON_SWMM_toolkit/run_simulation_runner.py` (lines 35–63)**: `_raise_enriched_srun_error()` was added in 54b8529 and correctly surfaces the stderr, but the error "no known pattern matched" because the message `"already completing or completed"` is not yet in the pattern list.

- **`docs/planning/improving_srun_commands.md`** (created by 54b8529): Contains the rationale for the changes; should be updated to reflect the outcome of this investigation.

---

## Implementation Strategy

### Chosen Approach: Empirical Validation → Targeted Code Fix

1. Run the diagnostic sbatch scripts (from the debugging report) on UVA to confirm `--overlap` is necessary in `batch_job` mode
2. Once confirmed, restore `--overlap` in `run_simulation.py` (both CPU and GPU srun branches)
3. Conditionally apply `--overlap` only when appropriate: it is correct for `batch_job`; its applicability to `1_job_many_srun_tasks` mode needs verification
4. Add `"already completing or completed"` pattern to `_raise_enriched_srun_error()` to give actionable hints for this specific failure
5. Update `hpc_time_min_per_sim` in the benchmarking test config

### Alternatives Considered

- **Revert commit 54b8529 entirely**: Simple but throws away the GPU preflight check and enriched error reporting, which are valuable
- **Use `--exact` instead of `--overlap`**: `--exact` is a stricter cousin that prevents downscaling; doesn't solve the step-creation hang
- **Remove `srun` entirely for `batch_job` mode, use direct `mpirun`**: Would work for MPI modes but breaks GPU task binding and SLURM step tracking

### Trade-offs

The targeted fix preserves the beneficial parts of 54b8529 (GPU preflight, enriched errors, `--cpu-bind=cores`) while restoring the critical `--overlap` flag. The empirical validation step is non-optional — we need ground truth from UVA before committing to a specific argument set.

---

## File-by-File Change Plan

### `src/TRITON_SWMM_toolkit/run_simulation.py`

**Purpose**: Restore `--overlap` to srun command strings in `prepare_simulation_command()`

**Changes** (around lines 529–554):

```python
# CPU/non-GPU branch (line ~529):
launch_cmd_str = (
    f"srun "
    f"-N {n_nodes_per_sim} "
    f"--ntasks={n_mpi_procs} "
    f"--cpus-per-task={n_omp_threads} "
    "--cpu-bind=cores "
    "--overlap "          # ← restore
    f"{exe} {cfg}"
)

# GPU branch (line ~545):
launch_cmd_str = (
    f"srun "
    f"-N {n_nodes_per_sim} "
    f"--ntasks={n_gpus} "
    f"--cpus-per-task={n_omp_threads} "
    f"{gpu_to_task_bind}"
    "--cpu-bind=cores "
    "--overlap "          # ← restore
    f"{exe} {cfg}"
)
```

**Note**: If empirical testing shows `--overlap` behaves differently for `1_job_many_srun_tasks`, a conditional may be needed. The `using_srun` flag is already available; a second `in_batch_job_mode` check can be derived from `multi_sim_run_method`.

**Expected impact**: Fixes the `"Job/step already completing or completed"` failure for all modes under `batch_job` execution.

### `src/TRITON_SWMM_toolkit/run_simulation_runner.py`

**Purpose**: Add the "already completing or completed" pattern to `_raise_enriched_srun_error()` to provide an actionable hint when this error recurs

**Change** (around line 45):

```python
if "already completing or completed" in s or "unable to create step" in s:
    hints.append(
        "SLURM step creation failed. In batch_job mode, srun must be run with "
        "--overlap to share the parent job's resource allocation. Without it, "
        "srun waits for exclusive step resources that may never materialize, "
        "causing the job to time out. Check that --overlap is present in the srun command."
    )
```

**Expected impact**: Future occurrences of this error produce an immediately actionable diagnostic.

### Test config: `hpc_time_min_per_sim`

**Purpose**: The 2-minute limit is too short for meaningful benchmarking diagnostics. Increase it.

**File**: The test config at `/sfs/gpfs/tardis/.../tests/test_uva_sensitivity_suite_cpu/cfg_analysis.yaml`
(on UVA) or the test case factory in `tests/examples.py` / similar.

**Change**: `hpc_time_min_per_sim: 2` → `hpc_time_min_per_sim: 15` (at minimum; 30 preferred for the benchmarking suite which includes 64-OMP-thread runs)

### `docs/planning/improving_srun_commands.md`

**Purpose**: Update the planning document with findings from this investigation

**Change**: Add a section documenting that `--overlap` is required for `batch_job` mode and summarize the empirical UVA test results.

---

## Risks and Edge Cases

| Risk | Mitigation |
|------|-----------|
| `--overlap` may behave differently on `1_job_many_srun_tasks` mode | Test `1_job_many_srun_tasks` separately; add conditional if needed |
| `--cpu-bind=cores` might also need reverting for some modes | Include `--cpu-bind=cores` vs `--cpu-bind=none` tests in diagnostic scripts |
| UVA may have updated SLURM version between prior working runs and now | Check SLURM version on UVA login node (`srun --version`) before and after |
| `--overlap` is deprecated in newer SLURM versions | Verify it's still accepted; UVA uses SLURM 23.x which supports it |
| GPU branch fix may be harder to test without GPU allocation | Test on a GPU node using `a6000` GRES if possible; otherwise trust `--overlap` is symmetrically needed |

---

## Validation Plan

### Phase 1: Empirical UVA Testing (Required Before Code Change)

Run the four diagnostic scripts from the debugging report on UVA's login node:

```bash
# Script 1: serial (1 task)
sbatch /tmp/test_srun_overlap.sh

# Script 2: MPI (2 tasks)
sbatch /tmp/test_srun_mpi.sh

# Script 3: OpenMP (1 task, 4 cpus-per-task)
sbatch /tmp/test_srun_omp.sh

# Script 4: Hybrid (2 tasks × 2 cpus-per-task)
sbatch /tmp/test_srun_hybrid.sh
```

Expected result: Tests with `--overlap` pass; tests without `--overlap` fail or hang.

### Phase 2: Code Fix and Unit Tests

After restoring `--overlap`:

```bash
# Run existing srun command construction tests
pytest tests/test_srun_command_construction.py -v

# Run resource management tests
pytest tests/test_resource_management_1job_mode.py -v

# Run all local tests (smoke tests)
pytest tests/test_PC_01_singlesim.py tests/test_PC_02_multisim.py -v
```

Note: The local tests do NOT use srun (no SLURM environment), so they verify command construction logic but not execution. The UVA run is the true integration test.

### Phase 3: Full Sensitivity Suite Re-run on UVA

```bash
# On UVA: clear simulation flags, verify setup flags intact
ls _status/sims/ | wc -l  # should be 0 after clearing
ls _status/setup_complete.flag  # must exist
ls _status/*_complete.flag  # prepare flags must exist for all 20

# Re-run workflow (pickup-where-leftoff will skip preparation)
# Confirm via tmux session log that 20/20 simulations complete
```

---

## Documentation and Tracker Updates

- **`docs/planning/improving_srun_commands.md`**: Add findings — specifically that `--overlap` is mandatory for `batch_job` srun steps on UVA. Update with empirical test results.
- **`CLAUDE.md` Gotchas section**: Consider adding a note: "`srun` inside `batch_job` allocation requires `--overlap` to share parent job resources; omitting it causes step creation to hang until timeout"
- **Debugging report**: Already written at `.debugging/test_uva_sensitivity_suite_cpu/debugging_report_20260223_170000.md`

---

## Decisions Needed from User

1. **Should `--overlap` also be restored for `1_job_many_srun_tasks` mode?**
   - Currently, `1_job_many_srun_tasks` uses srun from *within the batch allocation* too, so `--overlap` is likely needed there as well.
   - Risk level if assumption wrong: **medium** (may hang or fail, but behavior is the same as current broken state)

2. **What is the right `hpc_time_min_per_sim` for the benchmarking suite?**
   - The 64-OMP-thread and hybrid-32-node runs may need significantly more time.
   - Suggested: 30 minutes. Confirm before updating config.

3. **Should `--cpu-bind=cores` be kept, or should we revert to `--cpu-bind=none`?**
   - `--cpu-bind=cores` is more correct for CPU locality but may interact with site binding policies.
   - The diagnostic scripts test both; user should confirm based on results.

---

## Definition of Done

- [ ] Empirical UVA test results confirm `--overlap` is required and `--cpu-bind=cores` is compatible
- [ ] `--overlap` restored in both CPU and GPU srun branches of `run_simulation.py`
- [ ] `"already completing or completed"` pattern added to `_raise_enriched_srun_error()`
- [ ] `hpc_time_min_per_sim` updated in benchmarking test config to agreed-upon value
- [ ] Unit tests (`test_srun_command_construction.py`) updated to assert `--overlap` is present
- [ ] Full `test_uva_sensitivity_suite_cpu` run completes with 20/20 simulations
- [ ] `docs/planning/improving_srun_commands.md` updated with empirical findings

---

## Self-Check Results

**Header/body alignment**: All 9 sections are present and match their content. The "File-by-File Change Plan" includes code snippets that accurately reflect the current state and proposed changes.

**Section necessity**: All sections present. The "Decisions Needed from User" section contains three genuine blocking questions. The "Validation Plan" distinguishes the critical empirical phase from the mechanical code/test phase.
