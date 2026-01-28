# SWMM Output Parser Optimization Plan

## Executive Summary

This plan tracks performance and warning-hygiene improvements for `src/TRITON_SWMM_toolkit/swmm_output_parser.py`.

**Current status (2026-01-27):**
- Phases 1-2 complete with output parity maintained.
- Warning suppression in `utils.write_zarr()` and `Zone.Identifier` cleanup completed.
- Refactoring suite + multisim + warnings-as-errors tests passed.
- Baseline timing: **19.14s vs 20.30s** (5.7% savings) from Phase 2 changes.

**Performance tracking:**
- **Baseline (Phase 0):** 20.295690 seconds
- **Phase 1:** 19.89 seconds (2.0% savings)
- **Phase 2:** 19.14 seconds (5.7% savings)

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

1. **Single-pass RPT file parser**
   - Create a state machine that extracts all sections in one file read
   - Eliminate multiple iterations over `rpt_lines`

2. **Memory-mapped file reading**
   - Use `mmap` for large RPT files
   - Reduce memory allocation overhead

3. **Parallel processing for multiple entities**
   - Process node and link time series concurrently
   - Use `concurrent.futures` for I/O-bound operations

**Estimated Effort:** 8-16 hours
**Risk Level:** High

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
- [ ] Significant performance improvement (target: 50-80% reduction)
- [ ] Memory usage reduced for large files

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
| `src/TRITON_SWMM_toolkit/swmm_output_parser.py` | Main optimization target |
| `src/TRITON_SWMM_toolkit/utils.py` | Add warning suppression to `write_zarr()` |
| `tests/test_swmm_output_parser_refactoring.py` | New test file (create) |

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
   - Create a plan for updating refactoring_plan.md and next_action_prompt.md before moving onto the next phase. Be sure to note the time savings achieved with this round of work.

## final prompt
Please review and validate the latest changes that that were just completed and confirm that we are ready to proceed with the next phase as a new task. If substantive code changes were made, ensure that smoke testing was re-done successfully.

## next action prompt
Please proceed with Phase [N] of the refactoring defined in @/docs/next_action_prompt.md . Please double check that all items have been addressed before smoke testing. Ask me for verification before smoke testing.