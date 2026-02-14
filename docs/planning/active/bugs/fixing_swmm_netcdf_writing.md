# Fixing SWMM NetCDF Writing (StringDType Error)

**Status:** Reviewed (implementation-ready with refinements)
**Created:** 2026-02-12
**Reviewed:** 2026-02-12
**Owner:** Toolkit maintainers

## Goal

Prevent `ValueError: unsupported dtype for netCDF4 variable: StringDType()` when
writing SWMM outputs to NetCDF (`engine="h5netcdf"`). Ensure that:

- Numeric SWMM variables remain numeric (floats with `NaN` for missing values)
- String variables are encoded using **NetCDF-compatible numpy unicode/object dtypes**
- Output writing succeeds for RPT files containing orifice link rows

## Problem Summary

During SWMM RPT parsing, **orifice conduits** sometimes omit `max_velocity` and
`max_over_full_flow`. The current parser inserts **empty strings** to pad the row
to the expected length:

```
Orifice conduits do not return max velocity or max over full flow. Filling with empty string
```

Those empty strings propagate through parsing and conversion, which can produce
Pandas `StringDType()` columns. The NetCDF writer (via `h5netcdf`) rejects
`StringDType()` and raises:

```
ValueError: unsupported dtype for netCDF4 variable: StringDType()
```

### Root cause chain

1. `return_data_from_rpt()` inserts `""` for missing numeric fields (orifice rows)
2. `convert_datavars_to_dtype()` falls back to `str` for mixed numeric/string columns
3. Pandas `StringDType()` is emitted in the xarray Dataset
4. NetCDF writer rejects `StringDType()`

## Proposed Fix (High Confidence)

We’ll fix this in three layers:

1. **Parser-level normalization:** Use `np.nan` (not empty strings) for missing
   numeric values during orifice row correction.
2. **String dtype coercion:** Ensure string data variables are coerced to
   numpy unicode/object instead of pandas `StringDType()`.
3. **NetCDF safety check:** Final pre-write guard for any remaining
   `StringDType()` variables.

This provides a durable solution even if upstream parsing changes later.

---

## Implementation Plan (Paste-Ready)

### 1) Parser Fix: Replace Empty Strings with NaN for Orifice Rows

**File:** `src/TRITON_SWMM_toolkit/swmm_output_parser.py`
**Function:** `return_data_from_rpt()`

**Before** (current logic):
```python
            elif (
                solution
                == "Orifice conduits do not return max velocity or max over full flow. Filling with empty string"
            ):
                problem_row_list.insert(5, "")
                problem_row_list.insert(6, "")
                dict_line_contents_aslist[idx_int] = problem_row_list
                print(f"Properly parsed values:\n{dict_line_contents_aslist[idx_int]}")
```

**After** (use NaN for numeric fields):
```python
            elif (
                solution
                == "Orifice conduits do not return max velocity or max over full flow. Filling with empty string"
            ):
                # Insert NaN placeholders instead of empty strings to preserve numeric dtype
                problem_row_list.insert(5, np.nan)
                problem_row_list.insert(6, np.nan)
                dict_line_contents_aslist[idx_int] = problem_row_list
                print(f"Properly parsed values:\n{dict_line_contents_aslist[idx_int]}")
```

**Rationale:** Numeric columns should be numeric even when values are missing.
NaN preserves float dtype and avoids downstream string coercion.

---

### 2) String Coercion: Avoid Pandas `StringDType`

**File:** `src/TRITON_SWMM_toolkit/swmm_output_parser.py`
**Function:** `convert_datavars_to_dtype()`

Add a small helper to coerce string columns to numpy unicode/object.

**Insert helper near convert_datavars_to_dtype**:
```python
def _coerce_to_numpy_unicode(values: np.ndarray) -> np.ndarray:
    """Coerce string-like arrays to numpy unicode to avoid pandas StringDType."""
    # Check dtype.name attribute to properly detect pandas StringDtype
    if hasattr(values, 'dtype') and getattr(values.dtype, 'name', '') == 'string':
        return values.astype(object)
    if values.dtype.kind == "U":  # Already numpy unicode
        return values
    # Fallback for other string-like types
    return values.astype(str)
```

**Update string conversion block in `convert_datavars_to_dtype()`**:
```python
            else:
                try:
                    data_to_convert = ds[var]
                    if dtype == str:
                        # Ensure numpy unicode/object, not pandas StringDType
                        data_to_convert = xr.apply_ufunc(
                            _coerce_to_numpy_unicode, data_to_convert
                        )
                        ds[var] = data_to_convert
                    else:
                        ds[var] = ds[var].astype(dtype)
                    converted = True
                except (ValueError, TypeError):
                    first_attempt = False
                    continue
```

**Rationale:** This ensures string variables are represented in a NetCDF-compatible
format rather than `StringDType()`.

---

### 3) NetCDF Guard: Pre-Write Dtype Sanitization

**File:** `src/TRITON_SWMM_toolkit/utils.py`
**Functions:** `write_netcdf()` and `write_zarr_then_netcdf()`

Add a final safety check to coerce any `StringDType()` variables before writing.

**Insert helper function near the write functions:**
```python
def _sanitize_string_dtypes_for_netcdf(ds: xr.Dataset) -> xr.Dataset:
    """Convert pandas StringDType variables to numpy unicode/object for NetCDF."""
    for var in ds.data_vars:
        if getattr(ds[var].dtype, "name", "") == "string":
            ds[var] = ds[var].astype(object)
    return ds
```

**Update `write_netcdf()` to use the guard:**
```python
def write_netcdf(
    ds, fname_out, compression_level: int = 5, chunks: str | dict = "auto"
):
    encoding = return_dic_netcdf_encodings(ds, compression_level)
    if chunks == "auto":
        chunk_dict = return_dic_autochunk(ds)
    else:
        chunk_dict = chunks
    try:
        ds = ds.chunk(chunk_dict)
    except NotImplementedError:
        ds = ds.copy(deep=False)

    # Final guard: no pandas StringDType in NetCDF
    ds = _sanitize_string_dtypes_for_netcdf(ds)

    ds.to_netcdf(fname_out, encoding=encoding, engine="h5netcdf")
    return
```

**Update `write_zarr_then_netcdf()` to use the guard:**
```python
def write_zarr_then_netcdf(
    ds, fname_out, compression_level: int = 5, chunks: str | dict = "auto"
):
    if chunks == "auto":
        chunks = return_dic_autochunk(ds)
    ds = ds.chunk(chunks)
    # first write to zarr, then write to netcdf
    write_zarr(ds, f"{fname_out}.zarr", compression_level, chunks)
    # open and write
    ds = xr.open_dataset(
        f"{fname_out}.zarr", engine="zarr", chunks="auto", consolidated=False
    )

    # Sanitize before NetCDF write (important: do this AFTER reopening from zarr)
    ds = _sanitize_string_dtypes_for_netcdf(ds)

    write_netcdf(ds, fname_out, compression_level, chunks)
    # delete zarr
    try:
        fast_rmtree(f"{fname_out}.zarr")
    except Exception as e:
        print(f"Could not remove zarr folder {fname_out}.zarr due to error {e}")
    return
```

**Rationale:** This safety net prevents regressions if upstream parsing ever
reintroduces `StringDType()`. The guard is applied in both functions since
`write_zarr_then_netcdf()` reopens the dataset from zarr, which might
reintroduce StringDtype.

---

## Code Review Notes (2026-02-12)

### Files Verified
- ✅ `src/TRITON_SWMM_toolkit/swmm_output_parser.py:1006-1013` - Confirmed orifice parsing issue
- ✅ `src/TRITON_SWMM_toolkit/utils.py:613-626` - Confirmed `write_netcdf()` lacks StringDtype guard
- ✅ `src/TRITON_SWMM_toolkit/utils.py:582-601` - Confirmed `write_zarr_then_netcdf()` also needs guard

### Refinements Made to Plan
1. **String dtype detection**: Changed from `values.dtype == "string"` to `getattr(values.dtype, 'name', '') == 'string'` for proper pandas StringDtype detection
2. **Coverage extended**: Added guard to both `write_netcdf()` and `write_zarr_then_netcdf()` since the latter reopens datasets from zarr
3. **Guard placement in `write_zarr_then_netcdf()`**: Must sanitize AFTER reopening from zarr (line after `xr.open_dataset()`) to catch any StringDtype reintroduced during deserialization

### Implementation Notes
- `numpy` already imported as `np` at line 9 of `swmm_output_parser.py` (no import needed)
- The log message update in parser fix will change from "Filling with empty string" to reflect NaN usage
- The sanitization guard is defensive programming; layers 1-2 should prevent StringDtype, but layer 3 ensures safety

---

## Validation Plan

### 1. Reproduce the failure (baseline)
Run the existing multi-sim test that produced `process_swmm_0.log` and confirm
the error is present before changes.

### 2. Apply changes and re-run
Confirm that SWMM outputs write successfully:

- `processed/SWMM_only_node_tseries.nc`
- `processed/SWMM_only_link_tseries.nc`

### 3. Verify dtypes
Open the output and verify numeric columns are floats (with NaN where expected):

```python
import xarray as xr
ds = xr.open_dataset("SWMM_only_link_tseries.nc")
print(ds["max_velocity_mps"].dtype)  # float
```

### 4. Regression coverage (optional)
Add a lightweight unit test to ensure no StringDtype appears in SWMM datasets
created from RPT files.

**Suggested test location:** `tests/test_swmm_output_parser.py`

```python
def test_no_string_dtype_in_parsed_rpt_datasets():
    """Verify that parsed RPT datasets don't contain pandas StringDtype."""
    # Parse a known RPT file with orifice links
    ds_nodes, ds_links = retrieve_SWMM_outputs_as_datasets(...)

    # Check all data variables for StringDtype
    for var in ds_nodes.data_vars:
        dtype_name = getattr(ds_nodes[var].dtype, 'name', '')
        assert dtype_name != 'string', f"Variable {var} has StringDtype"

    for var in ds_links.data_vars:
        dtype_name = getattr(ds_links[var].dtype, 'name', '')
        assert dtype_name != 'string', f"Variable {var} has StringDtype"
```

---

## Risks & Considerations

- Using `NaN` for missing numeric values is a better semantic match than empty
  strings, but will slightly change the output (previously empty string).
- If downstream tools expect empty strings, they should be updated to handle
  NaN (recommended).
- The NetCDF guard is intentionally conservative; it only acts on
  `StringDType()` and leaves valid unicode arrays intact.

---

## Success Criteria

- SWMM outputs write without `StringDType()` errors
- Orifice rows are parsed with NaN in missing numeric columns
- NetCDF output variables retain correct dtypes (float for numeric, unicode for string)

---

## Implementation Summary

### Three-Layer Defense Strategy

This fix uses defense-in-depth to ensure NetCDF compatibility:

| Layer | Location | Purpose | Catches |
|-------|----------|---------|---------|
| **1. Parser** | `swmm_output_parser.py:1006-1013` | Use `np.nan` for missing numeric values | Root cause: orifice parsing |
| **2. Coercion** | `swmm_output_parser.py` (new helper) | Convert string vars to numpy unicode | DataFrame → xarray conversion issues |
| **3. Guard** | `utils.py` (both write functions) | Pre-write StringDtype sanitization | Any upstream regressions |

### Files to Modify

1. **`src/TRITON_SWMM_toolkit/swmm_output_parser.py`**
   - Lines 1006-1013: Replace `""` with `np.nan` in orifice parsing
   - Add `_coerce_to_numpy_unicode()` helper function
   - Update `convert_datavars_to_dtype()` string conversion block

2. **`src/TRITON_SWMM_toolkit/utils.py`**
   - Add `_sanitize_string_dtypes_for_netcdf()` helper function
   - Update `write_netcdf()` to call sanitizer before writing
   - Update `write_zarr_then_netcdf()` to call sanitizer after reopening zarr

3. **`tests/test_swmm_output_parser.py` (optional)**
   - Add regression test for StringDtype detection

### Expected Impact

- **Breaking change**: Orifice rows will contain `NaN` instead of `""` for missing velocity/flow
- **Backward compatibility**: Per CLAUDE.md, backward compatibility is not a priority for this single-developer codebase
- **Downstream tools**: Any tools expecting empty strings should be updated to handle `NaN` (recommended semantic improvement)
