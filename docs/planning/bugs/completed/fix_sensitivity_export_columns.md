# Implementation Plan: Improve Sensitivity Analysis Export Columns

## Task Understanding

### Requirements
1. Add `subanalysis_id` string column (e.g., `"sa_0"`, `"sa_1"`) to `sensitivity_analysis_definition.csv`
2. Add `subanalysis_id` string column to `scenario_status.csv` **only when the analysis is a sensitivity analysis**
3. Add `n_nodes` column (configured per-simulation node count) to `scenario_status.csv` **only when running in a SLURM context** (`analysis.in_slurm == True`); column is absent on local runs
4. `n_nodes` actual count is explicitly out of scope — configured count is sufficient and derivable

### Assumptions
- `subanalysis_id` values are `f"sa_{idx}"` matching the established `sub_analyses_prefix = "sa_"` convention (low risk)
- `n_nodes` reads from `cfg_analysis.n_nodes or 1` (consistent with how other resource fields default in `df_status`) (low risk)
- SLURM context is detected via `self.in_slurm` (already exists on `TRITONSWMM_analysis` at line 137: `"SLURM_JOB_ID" in os.environ or multi_sim_run_method == "1_job_many_srun_tasks"`) (low risk)
- SWMM-only model type gets `n_nodes = 1` when in SLURM context (low risk)
- `self.df_setup` has a default 0-based RangeIndex (confirmed: `_retrieve_df_setup()` at line 612 calls `pd.read_csv()` without `index_col`; `_create_sub_analyses()` at line 640 iterates `for idx, row in self.df_setup.iterrows()` using the same index to construct `sa_id = f"{self.sub_analyses_prefix}{idx}"`) (low risk)
- `case_study_catalog.py` calls `export_sensitivity_definition_csv()` but needs no changes — it is a thin call site that will benefit automatically from the added column (low risk)
- No backward compatibility shims needed per project philosophy

### Success Criteria
- `sensitivity_analysis_definition.csv` has a `subanalysis_id` column with values like `sa_0`, `sa_1`, ...
- `scenario_status.csv` for a sensitivity analysis has a `subanalysis_id` column
- `scenario_status.csv` for a SLURM run has an `n_nodes` column; column is absent on local runs
- Regular (non-sensitivity) analysis `df_status` is not broken

## Evidence from Codebase

- `src/TRITON_SWMM_toolkit/sensitivity_analysis.py:621–635` — `export_sensitivity_definition_csv()`: writes `self.df_setup.to_csv(output_path, index=True)` with integer index; no `subanalysis_id` string column
- `src/TRITON_SWMM_toolkit/sensitivity_analysis.py:716–751` — `df_status` property: adds `sub_analysis_iloc` (int) at line 744 but not `subanalysis_id` (string `"sa_{idx}"`)
- `src/TRITON_SWMM_toolkit/sensitivity_analysis.py:72–73` — `sub_analyses_prefix = "sa_"` confirms naming convention
- `src/TRITON_SWMM_toolkit/sensitivity_analysis.py:640–645` — `_create_sub_analyses()` iterates `for idx, row in self.df_setup.iterrows()` and constructs `sa_id = f"{self.sub_analyses_prefix}{idx}"`, confirming the index values drive the ID naming
- `src/TRITON_SWMM_toolkit/analysis.py:1971–2047` — row-building loop in regular analysis `df_status`: `n_nodes` is not currently included; `cfg_analysis.n_nodes` is available
- `src/TRITON_SWMM_toolkit/analysis.py:2050–2051` — sub-analysis returns early (`is_subanalysis=True`); `n_nodes` must be added inside the row-building loop before this branch
- `src/TRITON_SWMM_toolkit/config/analysis.py:51` — `n_nodes: Optional[int] = Field(1, ...)` — the config field to read
- `src/TRITON_SWMM_toolkit/export_scenario_status.py:432–470` — `export_scenario_status_to_csv()`: thin wrapper that calls `analysis.df_status` and writes it; no changes needed here if columns are added upstream
- `src/TRITON_SWMM_toolkit/case_study_catalog.py:119` — calls `export_sensitivity_definition_csv()`; no changes needed (thin call site, benefits automatically)

## Implementation Strategy

### Chosen Approach
Add the new columns directly in the `df_status` and `export_sensitivity_definition_csv` methods. `export_scenario_status.py` is a thin orchestrator — it writes whatever `df_status` returns, so enriching the source data is the right abstraction layer.

### Alternatives Considered
- **Mutate in `export_scenario_status.py`**: Would duplicate column-construction logic and decouple data definitions from the classes that own the sensitivity context
- **New helper method on sensitivity analysis**: Unnecessary indirection for a small, contained addition

### Trade-offs
Adding columns to `df_status` means any other caller also gets the new columns — this is desirable since `n_nodes` and `subanalysis_id` are generally useful for analysis consumers.

## File-by-File Change Plan

> Two source files change (column additions). One test file changes (stronger assertions).

### `src/TRITON_SWMM_toolkit/sensitivity_analysis.py`

**`export_sensitivity_definition_csv()` (line 621):**
- Before calling `to_csv`, insert a `subanalysis_id` column derived from the integer index:
  ```python
  df_export = self.df_setup.copy()
  df_export.insert(0, "subanalysis_id", [f"sa_{idx}" for idx in df_export.index])
  df_export.to_csv(output_path, index=True)
  ```
- Placing `subanalysis_id` as the first column (insert position 0) makes it immediately visible in the CSV
- The list comprehension mirrors the pattern in `_create_sub_analyses()` at line 645, which uses the same index values

**`df_status` property — line 744** (`sub_df_status["sub_analysis_iloc"] = sub_analysis_iloc`):
- Immediately after that assignment, add:
  ```python
  sub_df_status["subanalysis_id"] = f"sa_{sub_analysis_iloc}"
  ```

### `src/TRITON_SWMM_toolkit/analysis.py`

**Regular analysis `df_status` property, row-building loop (line 1980–2047):**
- Inside the `for model_type in enabled_models:` loop, after setting other resource fields, conditionally add `n_nodes` only when in SLURM context:
  ```python
  if self.in_slurm:
      if model_type == "swmm":
          row["n_nodes"] = 1
      else:
          row["n_nodes"] = self.cfg_analysis.n_nodes or 1
  ```
- This is in the same block as `n_mpi_procs`, `n_omp_threads`, `n_gpus`, so it is logically consistent
- `self.in_slurm` is already available on the analysis instance (line 137)

### `tests/test_PC_05_sensitivity_analysis_with_snakemake.py`

**`test_snakemake_sensitivity_workflow_dry_run` (line 189–193):**
- Replace the bare column-presence check with stronger assertions for `subanalysis_id`:
  ```python
  assert "subanalysis_id" in df_status.columns
  assert df_status["subanalysis_id"].str.startswith("sa_").all()
  expected_ids = [f"sa_{i}" for i in range(len(df_status))]
  assert df_status["subanalysis_id"].tolist() == expected_ids
  ```
- `n_nodes` is SLURM-conditional, so no assertion for it in a local dry-run test

## Risks and Edge Cases

| Risk | Mitigation |
|------|-----------|
| Sub-analysis `df_status` returns before snakemake join (line 2050–2051), so `n_nodes` must be in the row dict before that branch | `n_nodes` is added inside the row-building loop, which runs before the branch |
| Sensitivity `df_status` calls each sub-analysis `df_status` and concatenates — `n_nodes` and `subanalysis_id` flow through automatically | Verified by tracing the concatenation at lines 736–746 |
| `n_nodes` is `Optional[int]`; `or 1` guards against explicit `None` | Consistent with how `n_mpi_procs` and `n_omp_threads` are handled at lines 1993–2000 |
| Sub-analysis `df_status` is called with `is_subanalysis=True` — the sub-analysis object's `in_slurm` must be checked (not the master's) | `in_slurm` is set per-instance from env at init time; sub-analyses inherit the same SLURM environment, so `self.in_slurm` is correct in both regular and sub-analysis `df_status` calls |

## Validation Plan

**Primary smoke test**: `pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py`

- `test_snakemake_sensitivity_workflow_dry_run`: confirms `df_status` is correctly structured for a sensitivity analysis (Snakemake allocations join, no crash)
- The strengthened `subanalysis_id` assertions validate both column presence and exact value correctness
- `n_nodes` is SLURM-conditional so it will not appear in local dry-run results and is not testable here; validation of `n_nodes` requires an HPC run coordinated with the user

> PC_04 requires no changes — `n_nodes` is SLURM-only and won't appear in any local test run.

## Documentation and Tracker Updates

- No CLAUDE.md update required (no architectural change)
- No agent documentation update required

## Decisions Needed from User

None — all ambiguities resolved via code inspection and user confirmation. Assumptions are low-risk and consistent with existing patterns.

## Definition of Done

- [ ] `sensitivity_analysis_definition.csv` contains `subanalysis_id` column with string values `sa_0`, `sa_1`, ...
- [ ] `scenario_status.csv` from a sensitivity analysis run contains `subanalysis_id` column with correct `sa_{i}` values
- [ ] `scenario_status.csv` from a SLURM run contains `n_nodes` column reflecting `cfg_analysis.n_nodes or 1`; column is absent on local runs
- [ ] SWMM model type rows have `n_nodes = 1` when `in_slurm`
- [ ] No regression in `df_status` for regular (non-sensitivity) analyses
- [ ] `pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py::test_snakemake_sensitivity_workflow_dry_run` passes with strengthened `subanalysis_id` assertions
- [ ] `n_nodes` validated on next HPC run (SLURM-conditional, cannot be verified locally)

---

**Self-Check Results:**

1. **Header/body alignment**: All section headers accurately match their content — ✅
2. **Section necessity**: All sections present and actionable; no filler sections — ✅
3. **Internal consistency**: `n_nodes` is SLURM-conditional throughout (Requirements, Success Criteria, Implementation, DoD) — ✅
