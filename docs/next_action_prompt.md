. # Phase 1 Implementation Prompt for AI Agent

## Context

You are implementing Phase 1 of the SWMM Output Parser Optimization Plan. This phase focuses on "quick wins" - suppressing Zarr warnings and implementing straightforward performance improvements without changing function signatures.

**Reference Document:** `docs/swmm_output_parser_optimization_plan.md`

---

## Objective

Complete remaining Phase 1 quick wins and confirm the broader pipeline runs warning-free. Recent fixes already removed strict-warning failures in `tests/test_swmm_output_parser_refactoring.py`, but several optimization items remain.

---

## ✅ Completed: Suppress Zarr V3 String Warnings

### File to Modify
`src/TRITON_SWMM_toolkit/utils.py`

### What to Do
Modify the `write_zarr()` function to suppress `UnstableSpecificationWarning` from Zarr when writing datasets with string coordinates.

### Implementation

Add a `warnings.filterwarnings` context manager around the `ds.to_zarr()` call:

```python
import warnings

def write_zarr(ds, fname_out, compression_level, chunks: str | dict = "auto"):
    encoding = return_dic_zarr_encodings(ds, compression_level)
    if chunks == "auto":
        chunks = return_dic_autochunk(ds)
    ds = ds.chunk(chunks)
    
    # Suppress Zarr V3 warnings for fixed-length UTF-32 string types.
    # These warnings occur because Zarr V3 does not yet have a stable specification
    # for fixed-length Unicode strings. The data is still written correctly.
    # See: https://github.com/zarr-developers/zarr-extensions/tree/main/data-types
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*does not have a Zarr V3 specification.*",
            category=UserWarning,  # UnstableSpecificationWarning is a subclass
        )
        ds.to_zarr(fname_out, mode="w", encoding=encoding, consolidated=False)
```

### Verification
`pytest tests/test_swmm_output_parser_refactoring.py` passes without warnings after this change (strict warning checks enabled).

---

## Task 1: Vectorize `convert_swmm_tdeltas_to_minutes()`

### File to Modify
`src/TRITON_SWMM_toolkit/swmm_output_parser.py`

### Current Implementation (Iterative, NaN handling fixed)
```python
def convert_swmm_tdeltas_to_minutes(s_tdelta):
    lst_tdeltas_min = []
    for val in s_tdelta:
        if pd.Series(val).isna()[0]:
            lst_tdeltas_min.append(np.nan)
            continue
        lst_val_substrings_all = val.split(" ")
        lst_val_substring_data = []
        for val in lst_val_substrings_all:
            if len(val) > 0:
                lst_val_substring_data.append(val)

        days = int(lst_val_substring_data[0])
        hh_mm = lst_val_substring_data[-1]
        hr = int(hh_mm.split(":")[0])
        min = int(hh_mm.split(":")[1])
        tdelta = (
            pd.Timedelta(days, unit="D")
            + pd.Timedelta(hr, unit="hr")
            + pd.Timedelta(min, unit="min")
        )
        lst_tdeltas_min.append(tdelta.total_seconds() / 60)
    return lst_tdeltas_min
```

### Target Implementation (Vectorized)
```python
def convert_swmm_tdeltas_to_minutes(s_tdelta):
    """
    Convert SWMM time delta strings (e.g., "0  05:30") to minutes.
    
    Vectorized implementation using pandas string methods for improved performance.
    
    Parameters
    ----------
    s_tdelta : pd.Series or list-like
        Series of time delta strings in format "D  HH:MM" where D is days
        
    Returns
    -------
    list
        List of time deltas in minutes (float), with NaN for invalid entries
    """
    # Convert to Series if not already
    if not isinstance(s_tdelta, pd.Series):
        s_tdelta = pd.Series(s_tdelta)
    
    # Handle empty series
    if len(s_tdelta) == 0:
        return []
    
    # Extract days and time components using regex
    # Pattern matches: optional whitespace, digits (days), whitespace, HH:MM
    pattern = r'^\s*(\d+)\s+(\d+):(\d+)'
    extracted = s_tdelta.astype(str).str.extract(pattern)
    
    # Convert to numeric, coercing errors to NaN
    days = pd.to_numeric(extracted[0], errors='coerce')
    hours = pd.to_numeric(extracted[1], errors='coerce')
    minutes = pd.to_numeric(extracted[2], errors='coerce')
    
    # Calculate total minutes: days*1440 + hours*60 + minutes
    total_minutes = days * 1440 + hours * 60 + minutes
    
    return total_minutes.tolist()
```

### Verification
The function should return identical results for the same inputs. Test with:
```python
# Test case
test_input = ["0  05:30", "1  12:00", "0  00:15", None]
# Expected: [330.0, 1800.0, 15.0, nan]
```

---

## Task 2: Replace `iterrows()` in `return_swmm_outputs()`

### File to Modify
`src/TRITON_SWMM_toolkit/swmm_output_parser.py`

### Current Implementation (Inefficient)
```python
for idx, row in df_link_flow_summary.iterrows():
    link_id = row.link_id
    try:
        link_id = str(int(link_id))
    except Exception:
        lst_val_substrings = link_id.split(" ")
        for substring in lst_val_substrings:
            if len(substring) > 0:
                link_id = substring
                break
    df_link_flow_summary.loc[idx, "link_id"] = link_id
```

### New Implementation (Vectorized)
Replace the entire `for idx, row in df_link_flow_summary.iterrows():` block with:

```python
def _clean_link_id(link_id):
    """Clean a single link_id value."""
    try:
        # Try to convert to int then string (removes decimal points)
        return str(int(float(link_id)))
    except (ValueError, TypeError):
        # If conversion fails, extract first non-empty substring
        if isinstance(link_id, str):
            parts = link_id.split()
            return parts[0] if parts else link_id
        return str(link_id)

df_link_flow_summary["link_id"] = df_link_flow_summary["link_id"].apply(_clean_link_id)
```

Note: The helper function `_clean_link_id` should be defined at module level or as a nested function within `return_swmm_outputs()`.

### Alternative (Pure Vectorized)
If all link_ids follow a consistent pattern, this can be further optimized:

```python
# Vectorized approach using str accessor
df_link_flow_summary["link_id"] = (
    df_link_flow_summary["link_id"]
    .astype(str)
    .str.split()
    .str[0]
)
```

---

## Task 3: Simplify String Parsing in `format_rpt_section_into_dataframe()`

### File to Modify
`src/TRITON_SWMM_toolkit/swmm_output_parser.py`

### Current Pattern (in `return_data_from_rpt()`)
```python
for substring in line.split(" "):
    if (len(substring) > 0) and (substring not in lst_substrings_to_ignore):
        lst_substrings_with_content.append(substring)
```

### Improved Pattern
```python
# More efficient: filter empty strings and ignored substrings in one pass
lst_substrings_with_content = [
    s for s in line.split() 
    if s and s not in lst_substrings_to_ignore
]
```

Or using `filter()`:
```python
lst_substrings_with_content = list(filter(
    lambda s: s and s not in lst_substrings_to_ignore,
    line.split()
))
```

### Note
The `line.split()` without arguments automatically splits on whitespace and removes empty strings, which is more efficient than `line.split(" ")` followed by filtering.

---

## Additional Maintenance (Warning Hygiene)

To keep strict warning checks green, ensure Windows `Zone.Identifier` files do not exist inside `test_data/swmm_refactoring_reference/*.zarr` directories. They trigger `ZarrUserWarning` during `xr.open_dataset()`.

- If present, remove them:
  ```bash
  find test_data -name '*Zone.Identifier*' -print -delete
  ```

## Testing Requirements

After implementing all changes, run the following tests:

### 1. Run the Refactoring Test Suite
```bash
pytest tests/test_swmm_output_parser_refactoring.py -v
```

### 1a. Print Baseline Timing and Savings
```bash
pytest tests/test_swmm_output_parser_refactoring.py -k retrieve_swmm_outputs_baseline -s
```
This prints:
- Elapsed time for `retrieve_SWMM_outputs_as_datasets`
- Baseline time (20.295690s)
- Savings in seconds and percent

### 2. Run the Original Multi-sim Test (still pending)
```bash
pytest tests/test_PC_02_multisim.py -v
```

### 3. Verify No Warnings (still pending)
```bash
pytest tests/test_PC_02_multisim.py -v -W error::UserWarning
```
This should pass without raising any warnings as errors.

---

## Commit Strategy

Make separate commits for each task:

1. `fix(utils): suppress Zarr V3 string type warnings`
2. `perf(swmm_output_parser): vectorize convert_swmm_tdeltas_to_minutes`
3. `perf(swmm_output_parser): replace iterrows with vectorized link_id cleaning`
4. `perf(swmm_output_parser): simplify string parsing with list comprehensions`

---

## Success Criteria

- [x] `pytest tests/test_swmm_output_parser_refactoring.py` passes (all reference comparisons)
- [ ] `pytest tests/test_PC_02_multisim.py` passes with 0 warnings
- [ ] No functional changes to output data (verified by reference comparison)
- [ ] Code is cleaner and more readable

**Note:** Phase 1 is considered complete only when `tests/test_swmm_output_parser_refactoring.py` passes all tests.

---

## Files Summary

| File | Action |
|------|--------|
| `src/TRITON_SWMM_toolkit/utils.py` | Warning suppression already applied |
| `src/TRITON_SWMM_toolkit/swmm_output_parser.py` | Pending performance refactors |
| `tests/test_swmm_output_parser_refactoring.py` | Tests already passing |
