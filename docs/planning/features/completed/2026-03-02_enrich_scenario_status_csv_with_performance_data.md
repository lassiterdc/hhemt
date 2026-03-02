# Enrich scenario_status.csv with Performance Summary Data

**Written**: 2026-03-02
**Last edited**: 2026-03-02 — post-implementation correction: fix two stale implementation details in File-by-File Change Plan

---

## Task Understanding

### Requirements

1. **Remove `actual_wall_time_s` and replace with `perf_Total`**: The existing `actual_wall_time_s` field (populated from `log.out` / `.rpt` internal timers) reflects only the most recent hotstart segment and is therefore inaccurate for restarted runs. It is removed entirely. `perf_Total` from the performance summary dataset replaces it — see Evidence section for hotstart-safety verification.

2. **Add performance breakdown columns**: Pull the per-category timing totals from the performance summary dataset into `df_status`. For TRITON/TRITONSWMM model types, add columns (in this order): `perf_Total`, `perf_Compute`, `perf_SWMM`, `perf_MPI`, `perf_Simulation`, `perf_IO`, `perf_Resize`, `perf_Other`. `perf_Init` is also included. These will be `NaN` for SWMM model type rows (no TRITON performance dataset for standalone SWMM) and for rows where processing has not completed.

3. **Reorder columns**: Establish a canonical, reader-friendly column ordering in `df_status` / `scenario_status.csv` that groups columns logically: identity/status first, performance breakdown in the middle, raw resource allocation columns last.

### Assumptions

- The performance summary dataset is a per-`event_iloc` xarray Dataset with variables `Compute`, `MPI`, `IO`, `Resize`, `SWMM`, `Other`, `Simulation`, `Init`, `Total` (shape: scalar per event_iloc after the `sum(timestep_min).mean(Rank)` reduction).
- For SWMM model type rows, no TRITON performance dataset exists; all `perf_*` columns should be `NaN`.
- Sensitivity analysis `df_status` (built by concatenating sub-analysis `df_status` frames) will inherit the new columns automatically since `sensitivity_analysis.df_status` delegates to `analysis.df_status` — no changes needed in `sensitivity_analysis.py`.
- `actual_wall_time_s` is **removed** from `df_status` entirely. The `_get_performance_summary_row()` helper provides `perf_Total` as its replacement. The log.out / .rpt parsing code for wall time (`parse_triton_log_file()` result and `retrieve_swmm_performance_stats_from_rpt()` result) will no longer be used for this purpose; the `actual_wall_time_s` row assignment lines in `df_status` are deleted.

### Success Criteria

- `df_status` has `perf_*` columns for all rows; values are `NaN` for SWMM rows and for TRITON/TRITONSWMM rows where processing has not completed.
- `actual_wall_time_s` column is absent from `df_status` and `scenario_status.csv`.
- Column ordering matches the canonical order defined below.
- `scenario_status.csv` produced by `export_scenario_status.py` reflects the new schema.
- Existing tests pass.

---

## Evidence from Codebase

- **`analysis.py:1938–2090`** — `df_status` property. Builds rows for each `(event_iloc, model_type)` pair. `actual_wall_time_s` is currently set at lines 2034 and 2046 from `parse_triton_log_file()`, and at line 2059 from `retrieve_swmm_performance_stats_from_rpt()` — these assignments will be removed. No performance dataset columns are currently included.
- **`process_simulation.py:379–478`** — `_export_performance_tseries()`. Parses per-timestep `performance*.txt` files, converts to deltas, handles counter resets (lines 459–463). The reset-handling is the mechanism that makes this dataset hotstart-safe.
- **`process_simulation.py:532–563`** — `_export_performance_summary()`. Reduces timeseries by `sum(timestep_min).mean(Rank)`. Writes to `scen_paths.output_tritonswmm_performance_summary` / `output_triton_only_performance_summary`. Gated by `_already_written()`.
- **`process_simulation.py:485–486`** — `TRITONSWMM_performance_summary` property: opens `scen_paths.output_tritonswmm_performance_summary`.
- **`paths.py:113–125`** — `ScenarioPaths` fields: `output_tritonswmm_performance_summary: Optional[Path]`, `output_triton_only_performance_summary: Optional[Path]`.
- **`analysis.py:105–114`** — Analysis-level path setup for performance summary files.
- **`log.py:378–399`** — `performance_summary_written: Optional[LogField[bool]]` — the log flag to gate on.
- **`sensitivity_analysis.py:725–765`** — `df_status` concatenates sub-analysis `df_status` frames. No changes needed here since it delegates to `analysis.df_status`.
- **`export_scenario_status.py:432–470`** — `export_scenario_status_to_csv()` simply calls `analysis.df_status` and writes to CSV. No changes needed here if `df_status` is correct.
- **Performance file format** (from `test_data`): columns are `Rank, Compute, MPI, IO, Resize, SWMM, Other, Simulation, Init, Total`.

### Hotstart-Safety Verification

The reset-handling logic in `_export_performance_tseries()` (lines 459–463) ensures the performance summary is correct across hotstart restarts:

1. TRITON's `performance*.txt` files report **cumulative** timers since process start.
2. When a hotstart resumes, the TRITON process restarts and the cumulative counters reset to 0.
3. The code takes `diff()` of cumulative values to get per-timestep deltas.
4. Reset detection: `(diff() <= 0).all(axis=1)` identifies timesteps where all metrics decreased simultaneously (counter reset).
5. At reset timesteps, the delta is replaced with the raw cumulative value — treating it as the start of a new accumulation segment.
6. `sum(timestep_min)` then accumulates all deltas across all segments, yielding true total compute time regardless of how many restarts occurred.

**Contrast with `log.out`**: The `TRITON total wall time [s]` field in `log.out` is written by the final TRITON process only. It reflects only the wall time of the last execution segment, not the sum across all hotstart restarts.

---

## Implementation Strategy

### Chosen Approach

Add a private helper `_get_performance_summary_for_row()` in `analysis.py` that:
1. Determines the correct `ScenarioPaths` performance summary path based on `model_type`.
2. Checks `log.performance_summary_written` — returns `None` dict if not written.
3. Opens the xarray dataset and extracts scalar values for each variable.
4. Returns a `dict[str, float | None]` keyed by `perf_<VarName>`.

Then, in the `df_status` property loop, call this helper per `(event_iloc, model_type)` row and merge the results. Apply column reordering at the end of the property using `_reorder_df_status_columns()`.

### Alternatives Considered

- **Read performance summary in `export_scenario_status.py` instead of `df_status`**: Would violate the principle that `df_status` is the single source of truth for scenario status. The CSV exporter is a thin wrapper over `df_status`.
- **Add performance columns to `df_snakemake_allocations`**: Wrong layer; that method is about Snakemake resource allocation metadata, not simulation output data.

### Trade-offs

- Performance summary files are only available post-processing. This is intentional — `NaN` for unprocessed rows is meaningful and correct.
- Opening per-scenario xarray files in a loop is slightly slower than a batch read, but is consistent with how `parse_triton_log_file` is already called per-row. Not a concern for the typical number of scenarios.

---

## File-by-File Change Plan

### `src/TRITON_SWMM_toolkit/analysis.py`

**Purpose**: Add performance summary columns and column reordering to `df_status`.

Changes:
1. Add private method `_get_performance_summary_row(event_iloc, model_type) -> dict[str, float | None]`:
   - For `model_type in ("triton", "tritonswmm")`: check `scen.get_log(model_type).performance_summary_written.get() is True` (uses `TRITONSWMM_model_log`, not `scen.log` which is the prep-only log). Call `self._retrieve_sim_run_processing_object(event_iloc)` to get the `TRITONSWMM_sim_post_processing` instance — this correctly chains through `TRITONSWMM_run`. Call `.TRITONSWMM_performance_summary` or `.TRITON_only_performance_summary` on it. Extract scalar values via `.values.item()` for each variable, prefix keys with `perf_`.
   - For `model_type == "swmm"`: return `{f"perf_{v}": None for v in PERF_VARS}`.
   - No try/except: the log-flag gate is a semantic guarantee that the file was successfully written; any subsequent exception is a real defect that should raise, not be silently masked as NaN.
2. In the `df_status` property loop (line ~1991), call `_get_performance_summary_row()` and add results to `row`.
3. Add private method `_reorder_df_status_columns(df) -> pd.DataFrame` implementing the canonical column order below.
4. Apply `_reorder_df_status_columns()` at the final return sites of `df_status` (two return paths: sensitivity and regular).
5. No new imports needed: `TRITONSWMM_sim_post_processing` is already imported; xarray is accessed via `_open()` inside `process_simulation.py`.

**Define `PERF_VARS` as a module-level constant** (all variable names from performance.txt, used to build the all-None dicts for SWMM rows):
```python
PERF_VARS = ["Compute", "MPI", "IO", "Resize", "SWMM", "Other", "Simulation", "Init", "Total"]
```

**Define `PERF_VARS_ORDERED` as a module-level constant** (the display order for `_reorder_df_status_columns()`):
```python
PERF_VARS_ORDERED = ["Total", "Compute", "SWMM", "MPI", "Simulation", "IO", "Resize", "Other", "Init"]
```

### `src/TRITON_SWMM_toolkit/process_simulation.py`

**Purpose**: Add missing `TRITON_only_performance_summary` property (counterpart to the existing `TRITONSWMM_performance_summary` property at line 485).

Change: Add one property after `TRITON_only_performance_tseries` (line 490):
```python
@property
def TRITON_only_performance_summary(self):
    return self._open(self.scen_paths.output_triton_only_performance_summary)
```

### `src/TRITON_SWMM_toolkit/export_scenario_status.py`

**Purpose**: No logic changes required. The CSV export is a thin wrapper over `df_status`. The new columns will appear automatically.

Minor: update module docstring to mention performance breakdown columns.

### `tests/utils_for_testing.py`

**Purpose**: Update `assert_scenario_status_csv()` — remove `actual_wall_time_s` from `required_columns` list (line 323), add `perf_Total` in its place.

---

## Canonical Column Order

The `_reorder_df_status_columns()` helper should produce this order. Columns not present (e.g., sensitivity columns, `n_nodes` when not in SLURM) are skipped silently.

**Group 1 — Identity & Status**
```
subanalysis_id          (sensitivity only)
sub_analysis_iloc       (sensitivity only)
event_iloc
model_type
scenario_setup
run_completed
scenario_directory
```

**Group 2 — Weather / Simulation Setup** (all remaining `df_sims` columns, dynamic — placed here by exclusion)

**Group 3 — Sensitivity Parameters** (sensitivity-specific setup columns, dynamic)

**Group 4 — Performance Breakdown** (from performance summary dataset; NaN if not processed)
```
perf_Total
perf_Compute
perf_SWMM
perf_MPI
perf_Simulation
perf_IO
perf_Resize
perf_Other
perf_Init
```

**Group 5 — Expected Resource Configuration**
```
run_mode
n_mpi_procs
n_omp_threads
n_gpus
n_nodes           (SLURM only)
backend_used
```

**Group 6 — Actual Resources (from logs)**
```
actual_nTasks
actual_omp_threads
actual_gpus
actual_total_gpus
actual_gpu_backend
actual_build_type
```

**Group 7 — Snakemake Allocated Resources**
```
snakemake_allocated_nTasks
snakemake_allocated_omp_threads
snakemake_allocated_total_cpus
snakemake_allocation_parse_error
```

**Implementation note**: The reordering helper builds the explicit ordered list, then appends any columns present in the DataFrame but not in the list (future-proofing for new columns not yet anticipated). This prevents silent column drops.

---

## Risks and Edge Cases

| Risk | Mitigation |
|------|-----------|
| `model_type == "tritonswmm"` uses `TRITONSWMM_performance_summary`; `model_type == "triton"` uses `TRITON_only_performance_summary`. Wrong path → all NaN for that row. | Use `model_type` branch to select correct path, with a test covering both branches. |
| `log.performance_summary_written` is `None` for SWMM scenarios (log field not initialized for that model type). | Check `model_type` before accessing the log field; SWMM branch always returns all-None dict. |
| Performance summary path is `None` (not configured in analysis) | `_get_performance_summary_row()` must guard for `None` path and return all-None dict. |
| Sensitivity sub-analyses: `sub_analysis.df_status` is called, which calls the regular `df_status` — column reordering applied there. Then sensitivity layer concatenates and merges. Reordering must survive the concat + merge. | Apply `_reorder_df_status_columns()` at the outer return site (after the sensitivity merge), not only inside the sub-analysis loop. |
| Future new performance variables from TRITON: if TRITON adds a new column to `performance*.txt`, the constant `PERF_VARS` will need updating. | `PERF_VARS` is a module-level constant — easy to find and update. |

---

## Validation Plan

Run the PC_04 smoke test (Snakemake local workflow, produces processed outputs):

```bash
conda run -n triton_swmm_toolkit pytest tests/test_PC_04_multisim_with_snakemake.py -v
```

After the test, verify:
1. `scenario_status.csv` exists in the analysis directory.
2. `perf_Compute`, `perf_Total`, etc. columns are populated for rows where `run_completed=True` and processing was done.
3. Rows where processing did not complete have `NaN` in all `perf_*` columns.
4. Column order in the CSV matches the canonical order defined above.

Also run PC_05 if sensitivity analysis is in scope:

```bash
conda run -n triton_swmm_toolkit pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py -v
```

Manual verification: open `scenario_status.csv` in a spreadsheet and confirm `actual_wall_time_s` column is absent, `perf_Total` is present and populated for fully processed runs, and column order matches the canonical order above.

---

## Documentation and Tracker Updates

- `architecture.md`: No structural change — `df_status` already exists. No update required.
- Module docstring in `export_scenario_status.py`: Update to mention performance breakdown columns.
- No `CONTRIBUTING.md` changes needed.

---

## Decisions Needed from User

None — all design decisions above are resolved. Implementation can proceed as specified.

---

## Definition of Done

- [ ] `PERF_VARS` and `PERF_VARS_ORDERED` constants defined in `analysis.py`
- [ ] `actual_wall_time_s` row assignments removed from `df_status` (lines ~2034, ~2046, ~2059 in `analysis.py`)
- [ ] `_get_performance_summary_row()` helper implemented: instantiates `TRITONSWMM_sim_post_processing(scen)`, uses `_open()` via its properties, handles all three model types + None paths + log flag guard
- [ ] `df_status` loop populates `perf_*` columns via helper (no `actual_wall_time_s`)
- [ ] `_reorder_df_status_columns()` implemented with full canonical column order (using `PERF_VARS_ORDERED`)
- [ ] Reordering applied at both `df_status` return sites (regular and sensitivity paths)
- [ ] `TRITON_only_performance_summary` property added to `process_simulation.py`
- [ ] `actual_wall_time_s` removed from `required_columns` in `tests/utils_for_testing.py`; `perf_Total` added
- [ ] Module docstring in `export_scenario_status.py` updated
- [ ] PC_04 smoke test passes
- [ ] `scenario_status.csv` manual inspection confirms `actual_wall_time_s` absent, correct column order, `perf_*` values populated
- [ ] Ruff format/check clean: `conda run -n triton_swmm_toolkit ruff check . && ruff format --check .`
- [ ] No new `# type: ignore` comments introduced

---

## Self-Check Results

1. **Header/body alignment**: All section headers match their content.
2. **Section necessity**: All sections carry actionable content. The canonical column order table is long but is the specification that the implementation helper must implement exactly — keeping it here avoids ambiguity.
3. **Alignment with CONTRIBUTING.md**: No backward-compat shims. No defaults added where choices should be explicit. `PERF_VARS` constant is used consistently. Fail-fast via try/except with logged warning (graceful but not silent).
4. **Task-relevance**: Verified — no stale content from discovery retained.
