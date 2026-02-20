# QAQC Report — Moving Simulation Logging to Analysis Directory

**Date:** 2026-02-20
**Branch:** `debug_full_scale_testing`
**Reviewer:** Claude (independent review of AI-generated diff)
**Scope:** All changes in the current working-tree diff relative to HEAD

---

## Purpose of Change

Centralize model runtime logs (TRITON stdout/stderr) under the analysis-level log
directory (`analysis_paths.analysis_log_directory` / `analysis_paths.simlog_directory`)
instead of per-scenario `logs/` subdirectories. This makes logs accessible immediately for
debugging HPC runs that time out before output can be copied.

---

## Summary Verdict

The implementation is **mostly correct** but contains **one regression bug** that will break
the standard (non-sensitivity-analysis) Snakemake workflow at runtime, and **two secondary
issues** that should be addressed. Additionally there are several cosmetic/unrelated changes
that constitute unnecessary churn.

| Severity | Issue | File(s) |
|----------|-------|---------|
| 🔴 **BUG — will break** | `SIM_IDS` removed from generated Snakefile header but still referenced in `consolidate` rule input | `workflow.py` |
| 🟡 **Design concern** | Sensitivity analysis logfile lookup instantiates a full `TRITONSWMM_analysis` object on every call | `run_simulation.py` |
| 🟡 **Naming inconsistency** | SA log filename uses raw `analysis_id` string (e.g. `sa_3_`) not the compact `sa3_` form described in docs | `run_simulation.py`, `.prompts/…` |
| ⚪ **Cosmetic / unrelated** | Commented-out `mkdir` calls left in `analysis.py` | `analysis.py` |
| ⚪ **Cosmetic** | Several `analysis.py` line-wrapping changes unrelated to logging | `analysis.py` |

---

## File-by-File Review

### 1. `src/TRITON_SWMM_toolkit/paths.py`

**Change:** Added `analysis_log_directory: Path` and `simlog_directory: Path` as required
fields on `AnalysisPaths`.

**Assessment: ✅ Correct.**
Both fields are necessary non-optional additions. Order of fields is fine for a dataclass
(required fields before optional ones). These fields correctly sit between `simulation_directory`
and the optional output fields.

---

### 2. `src/TRITON_SWMM_toolkit/config/analysis.py`

**Changes:**
- `is_subanalysis` changed from `Optional[bool]` → `bool` (default `False`). Correct — it
  was effectively always used as a bool.
- Added `master_analysis_cfg_yaml: Optional[Path]`.
- Added `validate_subanalysis_requirements` model validator enforcing that `is_subanalysis=True`
  requires both `master_analysis_cfg_yaml` and `analysis_dir`.

**Assessment: ✅ Correct and well-structured.**
The new validator correctly runs `mode="before"` to catch the issue before field parsing.
The sentinel `is True` check (rather than truthiness) is appropriate. This is a clean addition
that enforces the new invariant at config-load time.

One minor note: the validator runs at `mode="before"`, meaning `analysis_dir` and
`master_analysis_cfg_yaml` will still be raw strings/None at that point. `None` check works
correctly either way.

---

### 3. `src/TRITON_SWMM_toolkit/analysis.py`

**Changes:**
- `analysis_log_directory` and `simlog_directory` computed and passed into
  `analysis_paths_kwargs`.
- Several line-wrapping reformats in `cancel()`, `consol_details`, and the message ternary.

**Assessment: ✅ Correct for the functional change. ⚠️ Minor issue noted.**

The two commented-out `mkdir` lines are left in:

```python
# analysis_log_directory.mkdir(parents=True, exist_ok=True)
# simlog_directory.mkdir(parents=True, exist_ok=True)
```

These are dead code. The actual directory creation is done in `workflow.py`'s
`generate_snakefile_content()` before workflow launch. However, the directories are also
needed at runtime (for `model_run_completed` checks and log writing in
`_analysis_level_model_logfile`). Leaving these commented out is defensible only because
`workflow.py` creates the directories before simulations run. If `model_run_completed` is
ever called on an analysis before the workflow is built, the log directory won't exist and
the check will silently return `False` (which is the correct fallback behavior). Clean up:
remove the comments entirely.

The line-wrapping reformats (Hunks B–E) are pure formatting with no functional effect.
They comply with the 120-char line limit rule in CLAUDE.md but are unnecessary churn unrelated
to the logging feature.

---

### 4. `src/TRITON_SWMM_toolkit/run_simulation.py`

#### 4a. Import Change

```python
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario, TRITONSWMM_analysis
```

`TRITONSWMM_analysis` is now imported at the top of the module instead of inside the method.
This is cleaner, but it creates a **module-level circular import risk**. Previously, the
import was local to the method — a common pattern in this codebase (e.g., in `__init__` of
`TRITONSWMM_run`). The import works only because `scenario.py` re-exports
`TRITONSWMM_analysis`. This should be tested carefully; if it causes circular import errors
at module load time, move the import back inside `_analysis_level_model_logfile`.

#### 4b. `_analysis_level_model_logfile` — Core New Method

```python
def _analysis_level_model_logfile(self, model_type) -> Path:
    log_dir = self._analysis.analysis_paths.simlog_directory
    subanalysis_id = ""
    if getattr(self._analysis.cfg_analysis, "is_subanalysis", False):
        subanalysis_id = str(self._analysis.cfg_analysis.analysis_id) + "_"
        master_analysis_yaml = self._analysis.cfg_analysis.master_analysis_cfg_yaml
        master_analysis = TRITONSWMM_analysis(
            master_analysis_yaml,
            system=self._analysis._system,
            skip_log_update=True,
        )
        log_dir = master_analysis.analysis_paths.simlog_directory

    fname = f"model_{model_type}_{subanalysis_id}evt{self._scenario.event_iloc}.log"
    return log_dir / fname
```

**Issue 1 — Performance/Design: Full analysis instantiation on every call 🟡**

For a sensitivity analysis, `_analysis_level_model_logfile` is called at minimum twice per
simulation (once for the logfile path in `prepare_simulation_command`, once for completion
check in `model_run_completed`). Each call constructs a full `TRITONSWMM_analysis` object
from YAML just to compute a path. This is expensive and unnecessary.

The master analysis log directory can be computed directly from the config path:

```python
master_yaml = self._analysis.cfg_analysis.master_analysis_cfg_yaml
# The analysis dir is <system_dir>/<analysis_id> or cfg_analysis.analysis_dir
# The simlog_directory is always <analysis_dir>/logs/sims
```

Since `master_analysis_cfg_yaml` stores a path (and the master analysis's `analysis_dir` is
embedded in the sub-analysis config as `analysis_dir = sub_analysis_directory`, distinct from
the master's `analysis_dir`), the right approach would be to cache the master
`analysis_paths` or derive the simlog path from the YAML's known sibling location. The
current implementation is functionally correct but will be slow under load.

**Issue 2 — Naming convention inconsistency 🟡**

The docstring and the debugging guide document the SA naming as:
```
model_{model_type}_sa{N}_evt{event_iloc}.log
```
e.g. `model_tritonswmm_sa3_evt0.log`

But the code produces:
```python
subanalysis_id = str(self._analysis.cfg_analysis.analysis_id) + "_"
# analysis_id is e.g. "sa_3" (sub_analyses_prefix = "sa_")
# so subanalysis_id = "sa_3_"
fname = f"model_{model_type}_{subanalysis_id}evt{event_iloc}.log"
# → "model_tritonswmm_sa_3_evt0.log"
```

The actual filename will be `model_tritonswmm_sa_3_evt0.log` (with underscore between `sa`
and `3`), **not** `model_tritonswmm_sa3_evt0.log` as documented. The previous Cline-generated
report described a more complex transformation (`analysis_id.split('sa_')[-1]`) that was
apparently **not implemented** — the actual code just appends `analysis_id + "_"` directly.

This is a documentation/code mismatch. The code behavior is internally consistent (the
filename is deterministic and both `_analysis_level_model_logfile` and the docs would be wrong
in the same way), but should be fixed to match documented convention.

The simpler fix is to update the docs to match the code (since the code is simpler and the
underscore is not harmful), but the code should be updated if the compact `sa{N}` form was
intentional.

#### 4c. `model_run_completed` — Log File Lookup Change

Old behavior: checked `scenario.scen_paths.logs_dir / "run_{model_type}.log"`.
New behavior: checks `_analysis_level_model_logfile(model_type)`.

**Assessment: ✅ Correct given the new log location.**

The completion check logic (searching for `"Simulation ends"` / `"EPA SWMM completed"`) is
unchanged. Since logs are now written to the analysis-level directory, the lookup must change
accordingly. No backward compatibility issue since this is a deliberate relocation.

Note: `ScenarioPaths` still has `log_run_triton`, `log_run_tritonswmm`, `log_run_swmm` fields
(lines 103–105 of `paths.py`) which now point to the old per-scenario locations. These are
no longer populated with actual log files. If any other code still reads those fields for
completion checking, it will silently return `False`. A search of the codebase found no other
callers of those specific `ScenarioPaths` log fields in completion-checking contexts, so this
is acceptable — but the stale fields are worth noting as future cleanup.

#### 4d. `prepare_simulation_command` — Logfile Path for Runtime

Old: `model_logfile = log_dir / "run_{model_type}.log"` (in scenario logs dir)
New: `model_logfile = self._analysis_level_model_logfile(model_type)`

**Assessment: ✅ Correct.** The `model_logfile` is the path passed to the subprocess as the
stdout/stderr capture destination. Moving this to the analysis-level directory is the core
intent of the change.

---

### 5. `src/TRITON_SWMM_toolkit/workflow.py`

#### 5a. Regular analysis (`SnakemakeWorkflowBuilder`)

**Directory creation — ✅ Correct.**
```python
log_dir = self.analysis_paths.analysis_log_directory
log_dir.mkdir(parents=True, exist_ok=True)
(log_dir / "sims").mkdir(parents=True, exist_ok=True)
```
Uses the canonical path rather than hardcoded `analysis_dir / "logs"`. Clean.

**Log path interpolation in Snakefile — ✅ Correct.**
All `log:` directives and shell redirections now use `{log_dir_str}/...` instead of
`"logs/..."`. The absolute path embedding is appropriate here because Snakemake rules can
run from different working directories.

**🔴 BUG: `SIM_IDS` removed from Snakefile header but still referenced.**

The diff shows this line was **removed** from the generated Snakefile:
```python
# Read simulation IDs from config
SIM_IDS = {list(range(n_sims))}
```

But at line 533, the generated Snakefile still contains:
```python
f'expand("_status/sims/{flag_pattern}", event_iloc=SIM_IDS)'
```

This `SIM_IDS` is embedded as a literal string in the generated Snakefile (not an f-string
variable — it's part of the Snakemake `expand()` call that Snakemake itself evaluates). When
Snakemake parses the generated Snakefile, it will encounter `SIM_IDS` as an undefined name
and raise a `NameError`, **preventing the workflow from starting at all**.

This is a clear regression. The `SIM_IDS = {list(range(n_sims))}` line must be restored to
the generated Snakefile header. The removal appears to have been unintentional — the diff
note from Cline's own report flagged this as a "potential mistake" but the fix was never
applied.

**Wildcard brace fix — ✅ Correct.**

The diff shows `_status/sims/scenario_{event_iloc}_prepared.flag` was correctly changed to
`_status/sims/scenario_{{event_iloc}}_prepared.flag` in the `sim_input` variable. This is an
actual bug fix: single braces inside an f-string would be consumed by Python's f-string
interpolation rather than passed as literal `{event_iloc}` to the Snakefile.

**Snakemake log redirect consistency — ✅ Correct.**
The `> {{log}} 2>&1` pattern (Snakemake's own `{log}` variable) is unchanged and correct.
The `{log_dir_str}/...` paths are used only in the `log:` directive definitions, not in
shell redirections, which is correct Snakemake usage.

**SLURM/tmux log paths — ✅ Correct.**
All hardcoded `analysis_dir / "logs" / "..."` usages for `_slurm_logs`,
`slurm_efficiency_report`, `tmux_session_*.log`, and `snakemake_master*.log` correctly
updated to use `analysis_log_directory`.

#### 5b. Sensitivity analysis (`SensitivityAnalysisWorkflowBuilder`)

**`log_dir_str` — ✅ Correct.**
```python
log_dir_str = str(self.master_analysis.analysis_paths.analysis_log_directory)
```
All SA Snakefile rule `log:` directives and `mkdir -p` commands updated to use this variable.
The SA workflow's log paths all point to the master analysis log directory, which is correct
— SA sub-analyses should write their logs to the master analysis's log directory.

**`onstart mkdir` pattern — ✅ Correct.**
```python
shell("mkdir -p _status {log_dir_str}/sims {log_dir_str}")
```
Using absolute paths here is correct since `onstart:` runs before any rules and Snakemake
may change working directories. The order (`sims` before parent) could theoretically be an
issue on some systems, but `mkdir -p` handles this correctly.

**Directory pre-creation before dry run — ✅ Correct.**
Three call sites updated:
```python
self.analysis_paths.analysis_log_directory.mkdir(parents=True, exist_ok=True)
(self.analysis_paths.analysis_log_directory / "sims").mkdir(parents=True, exist_ok=True)
```
And in one location uses `self.master_analysis.analysis_paths.simlog_directory.mkdir(...)`.
These are consistent.

**Print statement — ✅ Minor improvement.**
```python
"[Snakemake] Created required directories (_status, simlog_directory)"
```
More informative than the old hardcoded `logs/sims`.

---

### 6. `src/TRITON_SWMM_toolkit/sensitivity_analysis.py`

**Change:** Sets `cfg_snstvty_analysis.master_analysis_cfg_yaml` before writing the
sub-analysis YAML.

**Assessment: ✅ Correct and necessary.**

This is the mechanism that allows `_analysis_level_model_logfile` in `run_simulation.py` to
find the master analysis's log directory. Without this, the sub-analysis config would not
know its parent, and the log would be written to the sub-analysis's own `simlog_directory`
rather than the master's.

One observation: the `master_analysis_cfg_yaml` field is set after `analysis_dir` is set
and before the YAML is written. This ordering is correct.

---

### 7. `.prompts/debugging_hpc_analysis.md`

**Assessment: ✅ Correct and useful documentation.**

The new Step 5b and the log location additions are consistent with the implemented behavior,
with one caveat: the documented filename `model_tritonswmm_sa{N}_evt{M}.log` uses compact
`sa{N}` form, but the code generates `sa_{N}` form (see Issue 2 above). The docs should be
updated to match the actual output.

---

## Issues Requiring Action

### 🔴 BUG-001: `SIM_IDS` missing from generated Snakefile (will break all non-SA workflows)

**Location:** `workflow.py`, `generate_snakefile_content()`, around line 369
**Symptom:** Snakemake raises `NameError: name 'SIM_IDS' is not defined` when parsing the
generated Snakefile. The workflow cannot start.
**Fix:** Restore the `SIM_IDS` line to the Snakefile header template:
```python
snakefile_content = f'''...
SIM_IDS = {list(range(n_sims))}

rule all:
...'''
```

### 🟡 ISSUE-002: Naming mismatch between docs and code for SA logfiles

**Location:** `run_simulation.py:56`, `.prompts/debugging_hpc_analysis.md`
**Code produces:** `model_tritonswmm_sa_3_evt0.log` (uses raw `analysis_id` = `"sa_3"`)
**Docs describe:** `model_tritonswmm_sa3_evt0.log` (compact `sa{N}` form, no underscore)
**Fix options:**
- (Simpler) Update docs to say `model_{model_type}_sa_{N}_evt{event_iloc}.log`.
- (Cleaner) Change code to strip the `sa_` prefix: `subanalysis_id = "sa" + str(self._analysis.cfg_analysis.analysis_id).removeprefix("sa_") + "_"`.

### 🟡 ISSUE-003: Full `TRITONSWMM_analysis` instantiation in hot path

**Location:** `run_simulation.py:59-64`
**Impact:** Each call to `_analysis_level_model_logfile` for a subanalysis instantiates a
full analysis object (YAML parse + path construction). Called at least twice per simulation
in a sensitivity analysis run.
**Fix:** Derive the master simlog path directly from the master YAML's sibling paths rather
than constructing the full analysis object. The master's `analysis_dir` is the YAML file's
parent directory's parent (or use the `analysis_dir` field from the master YAML directly).
Alternatively, cache the result as a property.

### ⚪ CLEANUP-001: Remove commented-out `mkdir` lines in `analysis.py`

**Location:** `analysis.py:85-86`
**Fix:** Delete the two commented-out lines. Directory creation is handled elsewhere.

---

## Items Verified as Correct

- `AnalysisPaths` dataclass field additions (`paths.py`) — correct
- `analysis_config` validator for `is_subanalysis` requirements — correct
- `is_subanalysis: bool` type change (from `Optional[bool]`) — correct
- `master_analysis_cfg_yaml` being written into sub-analysis YAML (`sensitivity_analysis.py`) — correct
- All SLURM/tmux log path updates in `workflow.py` — correct
- SA workflow `log_dir_str` usage in all Snakefile rule `log:` directives — correct
- `model_run_completed` completion marker logic (unchanged) — correct
- Wildcard brace fix `{event_iloc}` → `{{event_iloc}}` in `sim_input` variable — correct (bug fix)
- `prepare_simulation_command` model logfile path delegation to `_analysis_level_model_logfile` — correct
- Snakemake `> {{log}} 2>&1` shell redirect pattern (unchanged) — correct

---

## Efficiency Assessment

The overall approach is efficient:
- Paths are computed once in `analysis.py` and stored in `AnalysisPaths` (no repeated computation)
- The `log_dir_str` variable is computed once per Snakefile generation (not per rule)
- Directory creation uses `exist_ok=True` (idempotent)

The one efficiency concern is ISSUE-003 above (unnecessary analysis instantiation in
`_analysis_level_model_logfile` for SA runs).

---

## Sensitivity Analysis Coverage

The change correctly covers sensitivity analysis in the following ways:

1. **Config plumbing** (`sensitivity_analysis.py`): `master_analysis_cfg_yaml` is written
   into each sub-analysis YAML, enabling the runtime to find the master log directory.
2. **Runtime log routing** (`run_simulation.py`): When `is_subanalysis=True`, the log
   directory is redirected to the master analysis's `simlog_directory`.
3. **Snakefile generation** (`workflow.py` SA builder): All SA Snakefile `log:` directives
   use the master analysis log directory via `log_dir_str`.
4. **Config validation** (`config/analysis.py`): The new validator ensures sub-analysis
   configs always have both `master_analysis_cfg_yaml` and `analysis_dir`.

The SA implementation is functionally sound modulo ISSUE-002 (naming) and ISSUE-003
(performance).

---

## Priority Action Items

| Priority | Action |
|----------|--------|
| 🔴 1 | Fix `SIM_IDS` missing from regular analysis Snakefile template (BUG-001) |
| 🟡 2 | Resolve SA logfile naming mismatch: update code or docs (ISSUE-002) |
| 🟡 3 | Eliminate full analysis instantiation in `_analysis_level_model_logfile` (ISSUE-003) |
| ⚪ 4 | Remove commented-out `mkdir` lines in `analysis.py` (CLEANUP-001) |
