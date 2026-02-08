# PC_04 Snakemake Multisim Workflow Fixes (TRITON/SWMM-only refactor)

**Date:** 2026-02-06  
**Status:** Completed  
**Scope:** `tests/test_PC_04_multisim_with_snakemake.py` end-to-end workflow coverage +
TRITON/SWMM output processing fixes required for Snakemake execution

## Why This Document Exists

After the refactor that separated TRITON-only and SWMM-only execution paths, the
Snakemake multisim test (`test_PC_04_multisim_with_snakemake.py`) failed to validate
the same end-to-end outputs that were already verified in:

- `tests/test_PC_01_singlesim.py`
- `tests/test_PC_02_multisim.py`

The failures exposed gaps in TRITON/SWMM output processing and in the test’s
cross-model assertions. This document records the fixes so another AI can verify
the changes quickly.

---

## High-Level Outcomes

1. **Snakemake end-to-end workflow now passes** for local multisim.
2. **TRITON-SWMM TRITON summaries are generated reliably**, even when TRITON-only
   outputs are processed in parallel.
3. **TRITON-SWMM SWMM time series are parsed from the correct hydraulics outputs**
   in `output/swmm/`, not the (missing) legacy `hydro.rpt` path.
4. **Cross-model link-variable comparison now normalizes capacity naming on both
   sides**, matching the single/multi sim tests.

---

## Tests Involved

- Primary test: `tests/test_PC_04_multisim_with_snakemake.py::test_snakemake_workflow_end_to_end`
- Validation references:
  - `tests/test_PC_01_singlesim.py`
  - `tests/test_PC_02_multisim.py`

Final verification command:

```bash
pytest -k test_snakemake_workflow_end_to_end tests/test_PC_04_multisim_with_snakemake.py -q
```

Result: **1 passed**.

---

## Root Causes & Fixes

### 1. TRITON-SWMM TRITON summary missing during consolidation

**Symptom:** Snakemake `consolidate` rule failed because
`TRITONSWMM_TRITON_summary.nc` was missing for scenario 0.

**Root cause:** TRITON-only processing was clearing raw TRITON outputs early, which
removed coupled raw files before TRITON-SWMM summary generation.

**Fixes:**

1. **Search for coupled raw TRITON outputs using actual output directories** instead
   of only `raw_triton_output_dir`.

   **File:** `src/TRITON_SWMM_toolkit/process_simulation.py`  
   **Method:** `_export_TRITONSWMM_TRITON_outputs()`

   Added a search over:
   - `scen_paths.out_tritonswmm`
   - `scen_paths.out_triton`
   - `sim_folder/output`
   - `sim_folder/build/output`

2. **Guard raw TRITON cleanup to require the coupled TRITON timeseries file**, not
   just the shared log flag.

   **File:** `src/TRITON_SWMM_toolkit/process_simulation.py`  
   **Method:** `_clear_raw_TRITON_outputs()`

   Replaced:
   ```python
   tritonswmm_ok = bool(self.log.TRITON_timeseries_written.get())
   ```
   With:
   ```python
   tritonswmm_ok = self._already_written(self.scen_paths.output_tritonswmm_triton_timeseries)
   ```

**Effect:** Coupled TRITON summary now always exists before consolidation.

---

### 2. TRITON-SWMM SWMM time series were empty (0 nodes)

**Symptom:** Cross-model test failed with missing node IDs; coupled SWMM datasets
had zero nodes.

**Root cause:** The coupled model writes SWMM hydraulics output to
`output/swmm/hydraulics.out` and `hydraulics.rpt`, while processing was still
looking for the legacy `swmm/hydro.rpt` path.

**Fixes:**

1. **Prefer `hydraulics.out` in `output/swmm/` for coupled SWMM time series parsing**.

   **File:** `src/TRITON_SWMM_toolkit/process_simulation.py`  
   **Method:** `_export_SWMM_outputs()`

   New resolution order for TRITON-SWMM:
   - `output/swmm/hydraulics.out` (preferred)
   - `output/swmm/hydraulics.rpt` (fallback)
   - legacy `swmm/hydro.rpt` if it exists

2. **Update SWMM raw output clearing to use `output/swmm/hydraulics.*`**, ensuring
   cleanup does not miss files.

   **File:** `src/TRITON_SWMM_toolkit/process_simulation.py`  
   **Method:** `_clear_raw_SWMM_outputs()`

**Effect:** Coupled SWMM time series now include full node/link outputs, and the
cross-model comparison proceeds to variable validation.

---

### 3. Link variable comparison mismatch (capacity vs capacity_setting)

**Symptom:** End-to-end test failed with “Link time series data variables do not match”.

**Root cause:** SWMM-only outputs use `capacity` (from `.out`), while the test
normalization only handled the SWMM-only side; coupled outputs also use `capacity`.

**Fix:** Normalize `capacity` → `capacity_setting` for **both** SWMM-only and
TRITON-SWMM outputs.

**File:** `tests/test_PC_04_multisim_with_snakemake.py`

Updated block:
```python
if "capacity" in swmm_link_vars:
    swmm_link_vars = (swmm_link_vars - {"capacity"}) | {"capacity_setting"}
if "capacity" in tritonswmm_link_vars:
    tritonswmm_link_vars = (tritonswmm_link_vars - {"capacity"}) | {
        "capacity_setting"
    }
```

**Effect:** Link-variable assertions now match the normalization logic used in
`test_PC_01_singlesim.py`.

---

## Files Modified

### Code
- `src/TRITON_SWMM_toolkit/process_simulation.py`
  - Added coupled raw-output search for TRITON-SWMM TRITON outputs.
  - Tightened TRITON raw-output cleanup guard.
  - Prefer `output/swmm/hydraulics.out` for TRITON-SWMM SWMM parsing.
  - Updated SWMM cleanup to consider `output/swmm/hydraulics.*`.

### Tests
- `tests/test_PC_04_multisim_with_snakemake.py`
  - Normalized link variable naming (`capacity` → `capacity_setting`) for both
    SWMM-only and TRITON-SWMM outputs.

---

## Quick Verification Checklist

1. **Run the end-to-end Snakemake test:**
   ```bash
   pytest -k test_snakemake_workflow_end_to_end tests/test_PC_04_multisim_with_snakemake.py -q
   ```
   Expect: `1 passed`.

2. **Inspect processed outputs for a scenario:**
   - `processed/TRITONSWMM_TRITON_summary.nc`
   - `processed/TRITONSWMM_SWMM_node_tseries.nc`
   - `processed/TRITONSWMM_SWMM_link_tseries.nc`
   - `processed/SWMM_only_*_tseries.nc`

3. **Confirm TRITON-SWMM SWMM outputs are non-empty:**
   ```python
   import xarray as xr
   ds = xr.open_dataset("processed/TRITONSWMM_SWMM_node_tseries.nc")
   assert ds["node_id"].size > 0
   ```

---

## Notes for Future Work

- If coupled SWMM artifacts are moved again, update `_export_SWMM_outputs()`
  and `_clear_raw_SWMM_outputs()` accordingly.
- The capacity naming mismatch is inherent to SWMM `.out` parsing (capacity)
  vs RPT-based naming (capacity_setting). Keep test normalization in sync with
  whichever parser is used.
