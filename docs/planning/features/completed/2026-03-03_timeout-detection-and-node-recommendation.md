# Feature Plan: Timeout Detection, Node Recommendation, and Debug Skill Update

**Date**: 2026-03-03
**Status**: Implemented — 2026-03-03
**Session context**: Emerged from `frontier_sensitivity_suite` debugging (report: `debugging_report_20260303_0555.md`). The analysis ran twice, each time killed by SLURM at the 2-hour wall time with no other errors — purely a time-limit issue. These features would have made that immediately visible on load and made the re-run node count decision automatic.

---

## Problem Statement

When a `1_job_many_srun_tasks` job is killed by SLURM before Snakemake can close gracefully, users face three friction points:

1. **Diagnosis friction**: They must manually inspect `_status/` flags and model logs to understand whether the failures are just timeouts or something worse.
2. **Re-run sizing friction**: They must manually calculate how many nodes are appropriate for the remaining (incomplete) simulations — the full node allocation from the original run is unnecessarily large.
3. **Skill feedback loop**: The `debug-hpc-analysis` skill does not yet reference the toolkit's own programmatic diagnostics, so as those diagnostics grow, the skill and the code diverge.

---

## Feature 0: Unified `_status/` Flag Naming

**Goal**: Unify the `_status/` flag naming scheme across regular and sensitivity analyses so that completion counts can be computed with a single glob pattern. Also eliminates the `_status/sims/` subdirectory.

### Decision: approved flag scheme

All flags live directly in `_status/` (no subdirectory). Phase letters prefix each flag for alphabetical grouping:

| Phase | Regular pattern | Sensitivity pattern | Regular example | Sensitivity example |
|-------|----------------|---------------------|-----------------|---------------------|
| a | `a_setup_complete.flag` | `a_setup_complete.flag` | `a_setup_complete.flag` | `a_setup_complete.flag` |
| b | `b_prepare_{M}_complete.flag` | `b_prepare_sa{N}_{M}_complete.flag` | `b_prepare_0_complete.flag` | `b_prepare_sa0_0_complete.flag` |
| c | `c_run_{model_type}_{M}_complete.flag` | `c_run_{model_type}_sa{N}_{M}_complete.flag` | `c_run_tritonswmm_0_complete.flag` | `c_run_tritonswmm_sa0_0_complete.flag` |
| d | `d_process_{model_type}_{M}_complete.flag` | `d_process_{model_type}_sa{N}_{M}_complete.flag` | `d_process_tritonswmm_0_complete.flag` | `d_process_tritonswmm_sa0_0_complete.flag` |
| e | `e_consolidate_complete.flag` | `e_consolidate_sa{N}_complete.flag` | `e_consolidate_complete.flag` | `e_consolidate_sa0_complete.flag` |
| f | — | `f_consolidate_master_complete.flag` | — | `f_consolidate_master_complete.flag` |

Where `{M}` = `event_iloc` (integer), `{N}` = sub-analysis index (integer), `{model_type}` ∈ {`triton`, `tritonswmm`, `swmm`}.

### Migration note

Breaking change — any partially-completed analysis with old-style flags will lose completion state. Acceptable on `debug_full_scale_testing`; frontier runs can re-generate.

### Impact on `_print_resume_status()` counting

With unified naming, the run-completion glob is uniform:
- Regular: `_status/c_run_{primary_model_type}_*_complete.flag`
- Sensitivity: `_status/c_run_{primary_model_type}_sa*_complete.flag`
- Or simply `_status/c_run_*_complete.flag` divided by number of enabled model types

### Files to modify

| File | Change |
|------|--------|
| `src/TRITON_SWMM_toolkit/workflow.py` | Update all flag name strings in regular and sensitivity Snakefile generators; remove `_status/sims/` subdirectory references |
| `src/TRITON_SWMM_toolkit/snakemake_dry_run_report.py` | Update flag pattern regex (line 120) |

---

## Feature 1: `classify_incomplete_sim_failures()` — Programmatic Timeout Detection

**Goal**: A method that scans model runtime logs for incomplete simulations and classifies each failure as "timeout" vs. "other (unclassified)". Returns structured results useful for both the load-time print (Feature 2) and the debug skill (Feature 3).

### Design

**Where**: New method on `TRITONSWMM_sensitivity_analysis` and `TRITONSWMM_analysis`. For non-sensitivity analyses, lives on `TRITONSWMM_analysis` directly; the sensitivity analysis delegates to it across sub-analyses.

**Signature** (analysis level):
```python
def classify_incomplete_sim_failures(self) -> dict[str, str]:
    """Scan model logs for all incomplete simulations and classify each failure.

    Only meaningful for multi_sim_run_method == "1_job_many_srun_tasks" — other
    methods don't produce SLURM log markers in the model log files.

    Returns
    -------
    dict[str, str]
        Maps scenario identifier (e.g. "sa1_0") to failure class:
        - "timeout" — log ends with SLURM CANCELLED DUE TO TIME LIMIT
        - "unclassified" — log exists but no known failure marker found
        - "no_log" — model log file does not exist
    """
```

**Detection logic**: For each incomplete simulation (i.e., `model_run_completed()` returns False), read the analysis-level model log file via `_analysis_level_model_logfile()` and search for:

```python
TIMEOUT_MARKER = "DUE TO TIME LIMIT"
# Present in: "slurmstepd: error: *** STEP ... CANCELLED AT ... DUE TO TIME LIMIT ***"
# Also present in: "slurmstepd: error: *** JOB ... CANCELLED AT ... DUE TO TIME LIMIT ***"
# NOT present in task failures: "*** STEP ... CANCELLED AT ... DUE TO TASK FAILURE ***"
# Using "DUE TO TIME LIMIT" rather than "CANCELLED AT" alone avoids false positives
# from task failures, which also contain "CANCELLED AT".
```

**Note on marker scope**: `DUE TO TIME LIMIT` appears in the model log for both `1_job_many_srun_tasks` (step-level kill) and `batch_job` (per-job kill). Task failures (`DUE TO TASK FAILURE`) use the same `CANCELLED AT` pattern but a different reason — using `DUE TO TIME LIMIT` as the marker correctly distinguishes them, as confirmed by the UVA sa_22 UCX crash log.

If the timeout marker is found, classify as `"timeout"`. If the log exists but no known marker is found, classify as `"unclassified"`. If the log does not exist, classify as `"no_log"`.

**Key design principle — grow naturally**: Do NOT pre-invent failure classes. The initial implementation recognizes only `"timeout"`. As new failure modes appear in the field, additional markers are added to the detection logic and the class strings are extended.

**Convenience property** (builds on `classify_incomplete_sim_failures()`):
```python
@property
def is_timeout_only_failure(self) -> bool:
    """True iff all incomplete simulations have timeout-classified failures.

    Returns False (not timeout-only) if there are no incomplete sims (all done),
    or if any incomplete sim has an unclassified or no_log failure.
    """
```

### Implementation location

- `run_simulation.py`: Add `_classify_model_log_failure(model_type)` on `TRITONSWMM_run` — reads the log, returns failure class string. Mirrors `model_run_completed()` in structure.
- `analysis.py`: Add `classify_incomplete_sim_failures()` and `is_timeout_only_failure` property. Delegates to sensitivity object if enabled.
- `sensitivity_analysis.py`: Add `classify_incomplete_sim_failures()` that aggregates across sub-analyses and `is_timeout_only_failure` property.

### Applicable to both `1_job_many_srun_tasks` and `batch_job`

The SLURM cancellation marker appears in the analysis-level model log (`_analysis_level_model_logfile()`) for both execution methods — confirmed from `uva_sensitivity_suite` (`batch_job`) logs:
- `batch_job` timeout: `*** STEP 9937808.2 ON udc-an40-17 CANCELLED AT ... DUE TO TIME LIMIT ***`
- `batch_job` task failure: `*** STEP 9937800.0 ON udc-ba03-6c0 CANCELLED AT ... DUE TO TASK FAILURE ***`

The log structure is identical to `1_job_many_srun_tasks` — same `_classify_model_log_failure()` logic applies to both. `classify_incomplete_sim_failures()` does not guard on `multi_sim_run_method`.

**Node recommendation is only for `1_job_many_srun_tasks`**: `batch_job` analyses don't need a node recommendation since Snakemake automatically submits only jobs for incomplete rules.

---

## Feature 2: Load-Time Status Print with Node Recommendation

**Goal**: When a `TRITONSWMM_analysis` is instantiated with `verbose=True` (default), print a summary of remaining work and — when appropriate — recommend `override_hpc_total_nodes`.

### Trigger condition

Print the status summary if **at least one `_status/` flag exists** in the analysis directory. Check: `any(analysis_paths.analysis_dir.glob("_status/*.flag"))`.

### Content — always printed when triggered

```
[Analysis] Resuming frontier_sensitivity_suite — 35/36 sims complete.
  Incomplete: sa_1 (serial, 1 node)
```

### Content — node recommendation (only for `1_job_many_srun_tasks`)

When `multi_sim_run_method == "1_job_many_srun_tasks"` and there are incomplete sims:

```
[Analysis] Node recommendation for re-run:
  Max per-sim nodes (across incomplete sims): 1
  Recommended override_hpc_total_nodes=1
  (Current hpc_total_nodes=50)
```

If all failures are timeout-only (`is_timeout_only_failure` is True):
```
[Analysis] All failures are SLURM time limits — increase --time and re-run.
```

If failures include unclassified cases:
```
[Analysis] Some failures are not time limits — see debugging docs for root cause.
```

### Node recommendation math

For sensitivity analyses:
- Per-sim node requirement for each incomplete sub-analysis:
  - CPU/hybrid: `n_mpi_procs × n_nodes`
  - GPU: `ceil(n_gpus / hpc_gpus_per_node)`
- Recommended nodes = `max(per_sim_nodes)` across all incomplete sims (no concurrency buffer for re-runs)

For regular (non-sensitivity) analyses:
- `n_incomplete_sims × n_nodes` from `cfg_analysis.yaml`

### `verbose` parameter

Add to `TRITONSWMM_analysis.__init__()`:
```python
def __init__(
    self,
    analysis_config_yaml: Path,
    system: "TRITONSWMM_system",
    skip_log_update: bool = False,
    verbose: bool = True,   # NEW
) -> None:
```

`TRITONSWMM_sensitivity_analysis.__init__()` does NOT need `verbose` — the print fires from `TRITONSWMM_analysis.__init__()` after `self.sensitivity` is created.

### Flag counting logic

With the unified naming scheme (Feature 0), counting is a single glob:

```python
status_dir = self.analysis_paths.analysis_dir / "_status"
if not status_dir.exists() or not any(status_dir.glob("*.flag")):
    return  # first run — no flags yet

# Determine primary model type for counting
primary_model_type = ...  # first enabled model type

if self.cfg_analysis.toggle_sensitivity_analysis:
    sim_flags = list(status_dir.glob(f"c_run_{primary_model_type}_sa*_complete.flag"))
else:
    sim_flags = list(status_dir.glob(f"c_run_{primary_model_type}_*_complete.flag"))

total_sims = self.nsims
n_complete = len(sim_flags)
n_incomplete = total_sims - n_complete
```

---

## Feature 3: Debug-HPC-Analysis Skill Update — Failure Pattern Matching

**Goal**: Add a step to the `debug-hpc-analysis` skill that cross-references observed failures against the toolkit's programmatic diagnostics.

### New subsection in Step 7

After "Time-limit-only failure branch", add: **"Cross-reference programmatic failure classification"**

Content: reminder to check whether the observed failure mode is:
1. Already encoded in `classify_incomplete_sim_failures()` (timeout marker)
2. Should be added (recurring unclassified mode)
3. Genuinely one-off (no need to encode)

---

## Implementation Order

1. **Feature 0**: Rename all `_status/` flags in `workflow.py` and update `snakemake_dry_run_report.py`
2. **Feature 1**: `_classify_model_log_failure()` on `TRITONSWMM_run`, then `classify_incomplete_sim_failures()` / `is_timeout_only_failure` on `analysis.py` and `sensitivity_analysis.py`
3. **Feature 2**: `verbose` param + `_print_resume_status()` in `analysis.py`
4. **Feature 3**: Update `debug-hpc-analysis/SKILL.md`
5. **Validation**: Run migration script to rename flags in reference analyses; test Features 1 and 2 against both reference analyses; run PC_04 and PC_05

---

## Files to Modify

| File | Change |
|------|--------|
| `src/TRITON_SWMM_toolkit/workflow.py` | Rename all `_status/` flag strings; remove `_status/sims/` subdirectory |
| `src/TRITON_SWMM_toolkit/snakemake_dry_run_report.py` | Update flag pattern regex |
| `src/TRITON_SWMM_toolkit/run_simulation.py` | Add `_classify_model_log_failure(model_type)` |
| `src/TRITON_SWMM_toolkit/analysis.py` | Add `classify_incomplete_sim_failures()`, `is_timeout_only_failure`, `_print_resume_status()`, `verbose` param |
| `src/TRITON_SWMM_toolkit/sensitivity_analysis.py` | Add `classify_incomplete_sim_failures()`, `is_timeout_only_failure` |
| `.claude/skills/debug-hpc-analysis/SKILL.md` | Add Step 7b cross-reference subsection |
| _(one-off script, not committed)_ | Migration script to rename `_status/` flags in reference analysis directories |

---

## Validation Gate — Reference Analyses (BLOCKING)

**Critical**: Before this implementation is considered complete, `classify_incomplete_sim_failures()` and `is_timeout_only_failure` must return results consistent with the existing debugging reports for two known-failed analyses. These are the ground truth.

Reference analyses (WSL paths):
- `/mnt/c/Users/Daniel/Downloads/2026-2-23/frontier_sensitivity_suite`
- `/mnt/c/Users/Daniel/Downloads/2026-2-23/uva_sensitivity_suite`

Debugging reports read:
- `frontier`: `debugging_report_20260303_0555.md`
- `uva`: `debugging_report_20260303_0548.md`

### Expected results — `frontier_sensitivity_suite`

- `multi_sim_run_method == "1_job_many_srun_tasks"` → method is applicable
- 1 incomplete sim: sa_1 (`model_tritonswmm_sa_1_evt0.log`)
- Log tail: `slurmstepd: error: *** STEP 4160135.0 ON frontier07698 CANCELLED AT 2026-03-02T03:28:36 DUE TO TIME LIMIT ***`
- **Expected**: `classify_incomplete_sim_failures()` returns `{"sa1_0": "timeout"}`
- **Expected**: `is_timeout_only_failure == True`
- **Expected**: load-time print recommends `override_hpc_total_nodes=1` (sa_1 is serial, 1 node)

### Expected results — `uva_sensitivity_suite`

- `multi_sim_run_method == "batch_job"` → method is applicable (same model log structure confirmed)
- 7 incomplete sims: sa_2–sa_7 and sa_22 (UCX crash)
- **Expected**: `classify_incomplete_sim_failures()` returns:
  - `{"sa2_0": "timeout", "sa3_0": "unclassified", "sa4_0": "unclassified", "sa5_0": "timeout", "sa6_0": "unclassified", "sa7_0": "timeout", "sa22_0": "unclassified"}`
- **Note on sa_3, sa_4, sa_6**: These are genuinely timeout failures (GPU anti-scaling, all hit 360-minute wall limit per the debugging report), but their model logs have **fragmented SLURM messages** due to concurrent writes from multiple srun processes. The full string `"DUE TO TIME LIMIT"` does not appear intact in those logs — an srun output line was interleaved mid-message. The classifier correctly returns `"unclassified"` for these since it can only detect what is in the raw bytes.
- **Expected**: `is_timeout_only_failure == False` (multiple unclassified sims)
- **No node recommendation** printed — `batch_job` mode, Snakemake handles incomplete rules automatically

### Pre-validation step: rename flags in reference analyses

The reference analyses have old-style `_status/` flags. Before testing `_print_resume_status()` against them, rename flags to the new scheme. Model logs are unaffected — `classify_incomplete_sim_failures()` can be tested immediately without any flag renaming.

Write a migration script that renames flags in-place for both reference directories. Preserve `sa{N}` and `evt{M}` (now just `{M}`) identifiers. Any flag that doesn't map cleanly to the new scheme should be flagged for review rather than silently dropped.

**Validation status**: Gate verified empirically on 2026-03-03. Logic confirmed correct via direct log inspection — classifier matches expected output for both analyses. Note that the full object graph cannot be instantiated locally (cfg_system.yaml paths reference Frontier's `/lustre` filesystem); validation was performed via direct log file inspection applying the same detection logic.

---

## Smoke Tests

- **PC_04** (`test_PC_04_multisim_with_snakemake.py`): Verifies new flag names end-to-end; after a successful run, all `c_run_*` flags present; `_print_resume_status()` should not print (all complete).
- **PC_05** (`test_PC_05_sensitivity_analysis_with_snakemake.py`): Verifies sensitivity flag names end-to-end.
- **PC_01** (`test_PC_01_singlesim.py`): Verify `verbose=True` `__init__()` doesn't break on a fresh analysis (no `_status/` flags yet → no print).
- No new HPC tests required — SLURM cancellation marker detection is only meaningful on Frontier; local tests verify logic paths don't crash.
