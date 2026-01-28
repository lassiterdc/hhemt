# SWMM Output Parser Optimization Plan

## Executive Summary

This document outlines a comprehensive plan to optimize `src/TRITON_SWMM_toolkit/swmm_output_parser.py` for improved performance and to address Zarr V3 warnings that appear during test execution. The current implementation takes approximately 92 seconds to process SWMM outputs in `tests/test_PC_02_multisim.py`, and generates multiple `UnstableSpecificationWarning` messages related to fixed-length UTF-32 string types.

**Status Update (2026-01-27):** Phase 1 warning suppression is partially complete. The test suite now passes with strict warning checks for `tests/test_swmm_output_parser_refactoring.py` after:
- Suppressing Zarr V3 string warnings in `write_zarr()`.
- Cleaning Windows `Zone.Identifier` files from reference `.zarr` test data.
- Explicitly setting `join="outer"` in `xr.merge()` to remove xarray FutureWarnings.
- Fixing NaN handling in `convert_swmm_tdeltas_to_minutes()` (still iterative, not yet vectorized).
- Closing subprocess stdout in `run_subprocess_with_tee()` to avoid unclosed-file warnings.

**Performance Tracking:** Baseline timing for `retrieve_SWMM_outputs_as_datasets()` is now recorded in the refactoring test suite. The benchmark prints elapsed time and savings relative to the baseline on each run to quantify time savings per phase.

- **Baseline (Phase 0):** 20.295690 seconds (captured 2026-01-27)
- **Phase 1+:** Use the benchmark output to update savings after each optimization step

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

### Phase 1: Quick Wins (Zarr Warnings + Simple Optimizations)

**Objective:** Suppress Zarr warnings and implement straightforward performance improvements without changing function signatures.

**Tasks:**

1. **Suppress Zarr V3 string warnings in `utils.py`** ✅
   - Implemented `warnings.filterwarnings` context manager around Zarr write operations
   - Documented the reason for suppression with a comment

2. **Vectorize `convert_swmm_tdeltas_to_minutes()`**
   - Pending: current implementation only fixes NaN handling (still iterative)
   - Planned: replace loop with pandas `str.extract()` and vectorized arithmetic
   - Expected speedup: 10-50x for large datasets

3. **Replace `iterrows()` in `return_swmm_outputs()`**
   - Pending: still using `iterrows()` for link_id cleanup
   - Planned: use vectorized pandas operations for link_id processing
   - Expected speedup: 5-20x for this section

4. **Simplify string parsing in `format_rpt_section_into_dataframe()`**
   - Pending: still using explicit substring filtering loops
   - Planned: use `filter(bool, line.split())` pattern
   - Reduce intermediate object creation

**Estimated Effort:** 2-4 hours
**Risk Level:** Low

---

### Phase 2: Core Parser Optimization

**Objective:** Refactor the main parsing functions for significant performance gains.

**Tasks:**

1. **Optimize `return_data_from_rpt()`**
   - Pre-compile regex patterns
   - Use list comprehensions instead of nested loops
   - Batch process problem row corrections

2. **Streamline `return_node_time_series_results_from_rpt()`**
   - Reduce dictionary lookups
   - Use more efficient data structures
   - Consider using `io.StringIO` for line processing

3. **Consolidate DataFrame operations in `return_swmm_outputs()`**
   - Merge DataFrames before xarray conversion
   - Reduce number of `.to_xarray()` calls
   - Batch type conversions

4. **Optimize type conversion functions**
   - Pre-determine dtypes based on column names
   - Use pandas `convert_dtypes()` where applicable
   - Reduce exception handling overhead

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
- [ ] `tests/test_PC_02_multisim.py` passes without warnings
- [ ] Execution time ≤ original (no regression)

### Phase 2 Completion Criteria
- [ ] All Phase 1 criteria maintained
- [ ] Measurable performance improvement (target: 20-50% reduction in processing time)
- [ ] Code complexity reduced (fewer nested loops)

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
