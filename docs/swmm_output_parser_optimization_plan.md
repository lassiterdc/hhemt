# SWMM Output Parser Optimization Plan

## Executive Summary

This plan tracks performance and warning-hygiene improvements for `src/TRITON_SWMM_toolkit/swmm_output_parser.py`.

**Current status (2026-01-28):**
- Phase 3 single-pass parser is now the default/only RPT parsing path.
- Experimental mmap + parallel parsing were removed for simplicity.
- Reference tests now drop static geodataframe attributes for node/link tseries comparisons.
- Phase 4 dataframe construction refactor completed with new baseline timing.

**Performance tracking:**
- **Baseline (Phase 0):** 20.295690 seconds
- **Phase 1:** 19.89 seconds (2.0% savings)
- **Phase 2:** 19.14 seconds (5.7% savings)
- **Phase 3 (in progress):** single-pass parser adopted; new baseline pending
- **Phase 4 (2026-01-28):** unprofiled 1.18 seconds (baseline test), savings 19.12s (94.2%) vs Phase 0
- **Profiling run (2026-01-28):** unprofiled ~1.18 seconds (cProfile run ~43.6s due to profiler overhead)

---

## 1. Problem Analysis

### 1.1 Zarr Warnings

**Observed Warnings:**
```
UnstableSpecificationWarning: The data type (FixedLengthUTF32(length=N, endianness='little')) 
does not have a Zarr V3 specification.
```

These warnings appear for string coordinates with various lengths (3, 4, 7, 8, 10, 13 characters) when writing xarray Datasets to Zarr format.

**Root Cause:**
- The `return_dic_zarr_encodings()` function in `utils.py` preserves Unicode string dtypes (`<UN`) for coordinates
- Zarr V3 does not yet have a stable specification for fixed-length UTF-32 strings
- String coordinates like `node_id`, `link_id`, `type`, `Shape`, `InletNode`, `OutletNode`, etc. trigger these warnings

**Affected Code Path:**
```
swmm_output_parser.py → retrieve_SWMM_outputs_as_datasets()
    → return_swmm_outputs()
    → process_simulation.py → _export_SWMM_outputs()
    → utils.py → write_zarr()
    → return_dic_zarr_encodings() [handles Unicode coordinates]
```

### 1.2 Performance Bottlenecks

The following inefficiencies were identified in `swmm_output_parser.py`:

| Function | Issue | Impact |
|----------|-------|--------|
| `return_data_from_rpt()` | Nested loops with string operations, complex error handling | High |
| `format_rpt_section_into_dataframe()` | Creates Series objects in loops, inefficient string parsing | High |
| `return_swmm_outputs()` | Uses `iterrows()` for link_id processing | Medium |
| `convert_swmm_tdeltas_to_minutes()` | Iterates over each value individually | Medium |
| `convert_datavars_to_dtype()` | Try/except loops for type conversion | Low-Medium |
| `convert_coords_to_dtype()` | Try/except loops for type conversion | Low-Medium |
| `return_node_time_series_results_from_rpt()` | Multiple passes over file lines | Medium |

#### 1.2.1 Profiling Findings (2026-01-28)

**cProfile (cumtime, top hotspots):**
- `parse_rpt_single_pass` ~38.8s
- `_build_tseries_datasets` ~38.7s
- `create_tseries_ds` ~19.36s per call (two calls)
- `format_rpt_section_into_dataframe` ~32.4s
- Pandas internals dominate (Series/DataFrame construction, indexing, transpose)

**Line profiler (line-level hotspots):**
- `parse_rpt_single_pass`: 99.3% of time in `_build_tseries_datasets` call
- `format_rpt_section_into_dataframe` (43.996s total):
  - `pd.Series(...).astype(str)` ~23.8%
  - `s_vals.iloc[idx_val] = substring` ~28.8%
  - `new_row = s_vals.to_frame().T` ~18.8%
  - `pd.concat(lst_series, ...)` ~6.5%
  - `return_data_from_rpt(...)` ~6.8%
- `return_data_from_rpt` (2.86s total):
  - list comprehension splitting lines ~21.4%
  - `pd.Series(dict_content_lengths)` ~22.0%
  - `s_lengths.mode().iloc[0]` ~23.5%
  - `idx_problem_rows = ...` ~28.1%

---

## 2. Implementation Phases

### Phase 1: Quick Wins (Zarr Warnings + Simple Optimizations) ✅ Complete

**Objective:** Suppress Zarr warnings and implement straightforward performance improvements without changing function signatures.

**Tasks:**

1. **Suppress Zarr V3 string warnings in `utils.py`** ✅
   - Implemented `warnings.filterwarnings` context manager around Zarr write operations
   - Documented the reason for suppression with a comment

2. **Vectorize `convert_swmm_tdeltas_to_minutes()`** ✅
   - Implemented vectorized pandas `str.extract()` flow

3. **Replace `iterrows()` in `return_swmm_outputs()`** ✅
   - Replaced with `_clean_link_id()` helper

4. **Simplify string parsing in `format_rpt_section_into_dataframe()`** ✅
   - Simplified substring parsing while preserving newline tokens

**Estimated Effort:** 2-4 hours
**Risk Level:** Low

---

### Phase 2: Core Parser Optimization ✅ Complete

**Objective:** Refactor the main parsing functions for significant performance gains.

**Tasks:**

1. **Optimize `return_data_from_rpt()`** ✅
   - Precompiled regex, cached substring lookups, helper for normal-row selection

2. **Streamline `return_node_time_series_results_from_rpt()`** ✅
   - Single-pass parsing with reduced lookups

3. **Consolidate DataFrame operations in `return_swmm_outputs()`** ✅
   - Combined DataFrames before xarray conversion and batched dtype conversions

4. **Optimize type conversion functions** ✅
   - Numeric prechecks + safe fallbacks to reduce exception-driven loops

**Estimated Effort:** 4-8 hours
**Risk Level:** Medium

---

### Phase 3: Advanced Optimization (Optional)

**Objective:** Implement more complex optimizations for maximum performance.

**Tasks:**

1. **Single-pass RPT file parser** ✅
   - Implemented a state machine that extracts all sections in one file read
   - Now the only RPT parsing path

2. **Memory-mapped file reading** (Deferred)
   - Removed from codebase to reduce complexity
   - Revisit if large-file performance becomes a bottleneck

3. **Parallel processing for multiple entities** (Deferred)
   - Removed from codebase to keep deterministic, single-threaded parsing
   - Revisit if profiling demonstrates benefit

**Estimated Effort:** 8-16 hours
**Risk Level:** High

---

### Phase 4: DataFrame Construction Refactor (New)

**Objective:** Reduce pandas overhead in `format_rpt_section_into_dataframe`, `return_data_from_rpt`, and `create_tseries_ds` based on line-profiler hotspots.

**Tasks (proposed):**
1. **Replace per-row Series creation in `format_rpt_section_into_dataframe()`**
   - Build rows as Python lists/dicts and use `pd.DataFrame.from_records()` once
   - Avoid `.to_frame().T` and repeated `.iloc` assignments
2. **Reduce per-section concatenations**
   - Accumulate row dicts/lists and instantiate DataFrame once per section
3. **Optimize `return_data_from_rpt()` length checks**
   - Replace `pd.Series(...).mode()` with `collections.Counter` for mode length
   - Track problematic rows using list indices to avoid pandas overhead
4. **Vectorize datetime parsing in `create_tseries_ds()`**
   - Convert `date_time` column once per combined DataFrame (avoid per-key `to_datetime`)
5. **Document and validate behavior**
   - Preserve newline handling, `ltr` filtering, and current error corrections
6. **Re-run profiling and report savings**
   - Run `python -m cProfile -o profiling/retrieve_profile.out profiling/profile_runner.py`
   - Run `kernprof -l -v profiling/profile_runner.py`
   - Record new unprofiled runtime and compare to 18.29s baseline

**Status:** ✅ Complete

**Observed impact:**
- Baseline timing (retrieve_SWMM_outputs_as_datasets): 1.18s
- Savings: 19.12s (94.2%) vs Phase 0 baseline (20.30s)
- Line profiler shows `format_rpt_section_into_dataframe` no longer dominated by Series creation

**Low-hanging follow-ups (optional):**
- Remove `sorted(...)` over row indices in `format_rpt_section_into_dataframe` (dict order already preserves line order).
- Cache `len(lst_col_headers)` in `format_rpt_section_into_dataframe` to avoid repeated length checks.
- Pre-split lines once in `parse_rpt_single_pass` to reduce repeated `line.split(" ")` calls across summary/time-series parsing.
- Use `pd.to_datetime(..., format=...)` if date format is fixed to avoid inference overhead.

**Estimated Effort:** 6-10 hours
**Risk Level:** Medium

---

## 3. Testing Strategy

### 3.1 Reference Data

Reference outputs are stored in:
```
test_data/swmm_refactoring_reference/
├── hydraulics.rpt          # Input RPT file
├── hydro.inp               # Input INP file
├── hydro.out               # Binary output (alternative source)
├── hydro.rpt               # Alternative RPT file
├── SWMM_link_summary.zarr/ # Reference link summary output
├── SWMM_link_tseries.zarr/ # Reference link time series output
├── SWMM_node_summary.zarr/ # Reference node summary output
└── SWMM_node_tseries.zarr/ # Reference node time series output
```

### 3.2 Test File Structure

Create a new test file: `tests/test_swmm_output_parser_refactoring.py`

**Test Categories:**

1. **Output Equivalence Tests**
   - Compare refactored output against reference `.zarr` files
   - Verify all non-null numeric values match within tolerance
   - Verify all string values match exactly
   - Verify coordinate values and dimensions match

2. **Warning Suppression Tests**
   - Verify no `UnstableSpecificationWarning` appears during execution
   - Use `pytest.warns()` or `warnings.catch_warnings()` to capture

3. **Performance Regression Tests**
   - Measure execution time before and after changes
   - Ensure no significant performance degradation
   - Optional: Assert minimum speedup threshold
   - Capture benchmark output from `test_retrieve_swmm_outputs_baseline` for time savings

4. **Edge Case Tests**
   - Empty sections in RPT file
   - Missing data values
   - Malformed lines (existing error handling)

### 3.3 Comparison Methodology

```python
def compare_zarr_datasets(ds_new: xr.Dataset, ds_ref: xr.Dataset, rtol=1e-5, atol=1e-8):
    """
    Compare two xarray Datasets for equivalence.
    
    Returns:
        tuple: (is_equivalent: bool, differences: dict)
    """
    differences = {}
    
    # Check dimensions match
    if set(ds_new.dims) != set(ds_ref.dims):
        differences['dims'] = {
            'new': set(ds_new.dims),
            'ref': set(ds_ref.dims)
        }
    
    # Check coordinates match
    for coord in ds_ref.coords:
        if coord not in ds_new.coords:
            differences[f'missing_coord_{coord}'] = True
        elif not np.array_equal(ds_new[coord].values, ds_ref[coord].values):
            differences[f'coord_{coord}'] = 'values differ'
    
    # Check data variables
    for var in ds_ref.data_vars:
        if var not in ds_new.data_vars:
            differences[f'missing_var_{var}'] = True
            continue
            
        new_vals = ds_new[var].values
        ref_vals = ds_ref[var].values
        
        # Handle numeric vs string comparison
        if np.issubdtype(ref_vals.dtype, np.number):
            # Numeric comparison with tolerance
            mask = ~(np.isnan(ref_vals) & np.isnan(new_vals))
            if not np.allclose(new_vals[mask], ref_vals[mask], rtol=rtol, atol=atol, equal_nan=True):
                differences[f'var_{var}'] = 'numeric values differ'
        else:
            # String/object comparison
            if not np.array_equal(new_vals, ref_vals):
                differences[f'var_{var}'] = 'string values differ'
    
    return len(differences) == 0, differences
```

---

## 4. Success Criteria

### Phase 1 Completion Criteria
- [x] No `UnstableSpecificationWarning` warnings during test execution (suppressed in `write_zarr()`)
- [x] All reference output comparisons pass (`pytest tests/test_swmm_output_parser_refactoring.py`)
- [x] `tests/test_PC_02_multisim.py` passes without warnings
- [x] Execution time ≤ original (no regression)

### Phase 2 Completion Criteria
- [x] All Phase 1 criteria maintained
- [x] Measurable performance improvement (target: 20-50% reduction in processing time)
- [x] Code complexity reduced (fewer nested loops)

### Phase 3 Completion Criteria
- [ ] All Phase 2 criteria maintained
- [ ] Performance does not regress vs Phase 2 baseline
- [ ] Optional: measurable improvement if single-pass gains are confirmed

### Phase 4 Completion Criteria
- [x] All Phase 3 criteria maintained
- [x] `format_rpt_section_into_dataframe` no longer dominated by per-row Series creation
- [x] End-to-end runtime improves vs Phase 2 baseline (target: ≥10% reduction)
- [x] Profiling summary updated with new timings and savings

---

## 5. Rollback Plan

If issues are discovered after implementation:

1. **Git-based rollback:** All changes should be committed incrementally with clear messages
2. **Feature flag:** Consider adding a `use_optimized_parser` flag during transition
3. **Reference preservation:** Keep reference `.zarr` files unchanged for regression testing

---

## 6. Files to Modify

| File | Changes |
|------|---------|
| `src/TRITON_SWMM_toolkit/swmm_output_parser.py` | Single-pass parser + legacy/mmap/parallel removal |
| `src/TRITON_SWMM_toolkit/utils.py` | Warning suppression for `write_zarr()` |
| `tests/test_swmm_output_parser_refactoring.py` | Drop static geodataframe vars from reference datasets |

---

## 7. Dependencies

- No new package dependencies required
- Existing dependencies: `numpy`, `pandas`, `xarray`, `zarr`, `warnings`, `re`

---

## Appendix A: Current Function Call Graph

```
retrieve_SWMM_outputs_as_datasets()
└── return_swmm_outputs()
    ├── return_node_time_series_results_from_rpt()
    │   ├── create_tseries_ds()
    │   │   └── format_rpt_section_into_dataframe()
    │   │       └── return_data_from_rpt()
    │   └── [repeated for links]
    ├── convert_datavars_to_dtype()
    ├── convert_coords_to_dtype()
    ├── return_swmm_system_outputs()
    ├── return_lines_for_section_of_rpt()
    ├── format_rpt_section_into_dataframe()
    ├── convert_swmm_tdeltas_to_minutes()
    └── xr.merge()
```

---

## Appendix B: Sample RPT File Structure

The SWMM `.rpt` file contains multiple sections that are parsed:

1. **Element Count** - Validation marker
2. **Flow Units** - System configuration
3. **Flow Routing Continuity** - Error metrics
4. **Node Flooding Summary** - Flood statistics per node
5. **Node Inflow Summary** - Flow statistics per node
6. **Link Flow Summary** - Flow statistics per link
7. **Node Time Series Results** - Time series data per node
8. **Link Time Series Results** - Time series data per link (implied)

Each section has a header line with dashes (`------------`) marking the start and end of the data table.

# Prompts
1. **Code Quality Review:**
   - Read the modified files and confirm the implementation matches the phase goals in docs/swmm_output_parser_optimization_plan.md
   - Check for any code smells, unnecessary complexity, or missed opportunities for simplification
   - Verify no duplicate code was introduced
   - Confirm no functionality was accidentally removed or altered

2. **Completeness Check:**
   - Compare what was done against the phase checklist - were all items truly completed?
   - Were there any edge cases or aspects mentioned in the plan that weren't addressed?

3. **Consistency Verification:**
   - Does the new code follow the same patterns/conventions as existing code?
   - Are naming conventions consistent?
   - Do new helper functions/classes have appropriate docstrings?

4. **Potential Issues:**
   - Are there any potential regression risks not covered by the smoke tests?
   - Any tight coupling introduced that could cause issues in future phases?
   - Any technical debt created that should be noted for Final phase (Polish)?

5. **Summary Report:**
   - Rate the phase completion: Excellent / Good / Needs Improvement
   - List any concerns or recommendations for follow-up
   - How much time was saved in this round of refactoring?

6. **Plan immediate action based on review:**
   - If there are any changes that should be completed as part of this phase based on your review, create an action plan now.

7. **Plan next phase**
   - Create a plan for updating swmm_output_parser_optimization_plan.md and next_action_prompt.md before moving onto the next phase. Be sure to note the time savings achieved with this round of work.

## final prompt
Please review and validate the latest changes that that were just completed and confirm that we are ready to proceed with the next phase as a new task. If substantive code changes were made, ensure that smoke testing was re-done successfully.

## next action prompt
Please proceed with Phase [N] of the refactoring defined in @/docs/next_action_prompt.md . Please double check that all items have been addressed before smoke testing. Ask me for verification before smoke testing.