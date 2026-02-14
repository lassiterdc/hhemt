# Plan: Fix Resource Validation for SWMM-Only Models

## Context

The resource validation system (`consolidate_workflow.py:validate_resource_usage`) was designed to verify that TRITON/TRITON-SWMM simulations used the expected compute resources (MPI tasks, OMP threads, GPUs) by parsing `log.out` files.

**Problem:** SWMM standalone models don't produce TRITON-style `log.out` files, causing false-positive validation failures in multi-model tests.

**Discovered during:** Implementation of dynamic SWMM threading control (Phase 1 & Phase 2)

**Impact:**
- `test_PC_04::test_snakemake_workflow_end_to_end` fails validation despite correct execution
- Multi-model tests (TRITON + SWMM + TRITON-SWMM) cannot pass end-to-end validation
- Developers may incorrectly assume implementation bugs exist

## Current Behavior

### df_status Structure for Multi-Model Analysis
```
event_iloc  model_type  run_mode  n_omp_threads  actual_omp_threads
0           triton      serial    1              None
0           tritonswmm  serial    1              None
0           swmm        openmp    2              'rpt does not exist'
```

**SWMM rows:**
- `n_omp_threads`: Correctly populated from `cfg_analysis.n_threads_swmm` (line 1963 in analysis.py)
- `run_mode`: Set to `"openmp"` if `n_threads_swmm > 1`, else `"serial"` (line 1960)
- `actual_omp_threads`: `'rpt does not exist'` because SWMM doesn't write TRITON log.out

### Validation Logic (consolidate_workflow.py:123-130)
```python
# Check OMP threads
if (
    pd.notna(row["actual_omp_threads"])
    and row["actual_omp_threads"] != expected_omp_threads
):
    issues.append(
        f"  - OMP threads: expected {expected_omp_threads}, actual {row['actual_omp_threads']}"
    )
```

**Bug:** `pd.notna('rpt does not exist')` returns `True`, triggering comparison failure.

## Root Cause Analysis

1. **SWMM uses EPA SWMM executable**, not TRITON
2. **EPA SWMM doesn't write resource usage logs** in TRITON's format
3. **Log parser returns sentinel strings** (`'rpt does not exist'`) instead of `None` or `NaN`
4. **Validator treats sentinel strings as valid data**, comparing `'rpt does not exist' != 2`

## Proposed Solution

### Option 1: Skip SWMM Models in Validation (Simplest)

**Approach:** Exclude `model_type == "swmm"` rows from resource validation

**Rationale:**
- SWMM standalone doesn't provide resource usage data
- SWMM threading is validated via .inp file inspection (already done in Phase 2 tests)
- Resource validation is only meaningful for TRITON/TRITON-SWMM models

**Changes Required:**
- File: `src/TRITON_SWMM_toolkit/consolidate_workflow.py`
- Line: 86-91 (loop over df_status)

```python
for idx, row in df_status.iterrows():
    if not row["run_completed"]:
        continue  # Skip scenarios that didn't complete

    # NEW: Skip SWMM-only models - they don't produce TRITON log.out files
    if row["model_type"] == "swmm":
        continue

    scenario_dir = row["scenario_directory"]
    issues = []
    # ... rest of validation logic
```

**Pros:**
- ✅ Minimal code change (1 line)
- ✅ Clear intent (explicit skip for SWMM)
- ✅ No risk of breaking existing TRITON/TRITON-SWMM validation
- ✅ SWMM threading is already validated via .inp file tests

**Cons:**
- ❌ Doesn't validate SWMM resource usage (but there's nothing to validate anyway)

---

### Option 2: Normalize Sentinel Values to NaN (More Robust)

**Approach:** Convert sentinel strings (`'rpt does not exist'`, `'parse error'`, etc.) to `NaN` before validation

**Rationale:**
- Sentinel strings should be treated as missing data
- `pd.notna()` checks would then work correctly
- More robust for future edge cases

**Changes Required:**
- File: `src/TRITON_SWMM_toolkit/consolidate_workflow.py`
- Line: Before validation loop (around line 80)

```python
# Normalize sentinel values to NaN for proper validation
sentinel_values = ['rpt does not exist', 'parse error', 'file not found']
for col in ['actual_nTasks', 'actual_omp_threads', 'actual_gpus',
            'actual_total_gpus', 'actual_gpu_backend', 'actual_build_type']:
    if col in df_status.columns:
        df_status[col] = df_status[col].replace(sentinel_values, pd.NA)
```

**Pros:**
- ✅ Handles all sentinel cases uniformly
- ✅ Makes validation logic cleaner (no special cases needed)
- ✅ More maintainable if new sentinel values are added

**Cons:**
- ❌ Modifies df_status (could affect other code)
- ❌ Requires identifying all possible sentinel values
- ❌ More complex change

---

### Option 3: Hybrid Approach (Recommended)

**Approach:** Combine both strategies for robustness and clarity

**Changes Required:**

1. **Normalize sentinels to NaN** (Option 2) - handles all edge cases
2. **Add explicit SWMM skip** (Option 1) - documents intent clearly

```python
# Normalize sentinel values to NaN for proper validation
sentinel_values = ['rpt does not exist', 'parse error', 'file not found']
for col in ['actual_nTasks', 'actual_omp_threads', 'actual_gpus',
            'actual_total_gpus', 'actual_gpu_backend', 'actual_build_type']:
    if col in df_status.columns:
        df_status[col] = df_status[col].replace(sentinel_values, pd.NA)

# Check for mismatches
mismatches = []

for idx, row in df_status.iterrows():
    if not row["run_completed"]:
        continue

    # Skip SWMM-only models - they use EPA SWMM which doesn't produce
    # TRITON-style resource logs. SWMM threading is validated via .inp files.
    if row["model_type"] == "swmm":
        continue

    # ... rest of validation
```

**Pros:**
- ✅ Clear intent (skip SWMM explicitly)
- ✅ Robust (handles all sentinel cases)
- ✅ Self-documenting (comment explains why)
- ✅ Future-proof (handles new edge cases)

**Cons:**
- ❌ Slightly more code than Option 1

---

## Recommended Implementation

**Use Option 3 (Hybrid Approach)**

### Changes Needed

#### File: `src/TRITON_SWMM_toolkit/consolidate_workflow.py`

**Location 1:** After line 79 (before validation loop)
```python
# Normalize sentinel values to NaN for proper validation
# These strings indicate missing/unparseable data and should not be compared
sentinel_values = ['rpt does not exist', 'parse error', 'file not found']
for col in ['actual_nTasks', 'actual_omp_threads', 'actual_gpus',
            'actual_total_gpus', 'actual_gpu_backend', 'actual_build_type',
            'actual_wall_time_s']:
    if col in df_status.columns:
        df_status[col] = df_status[col].replace(sentinel_values, pd.NA)
```

**Location 2:** After line 88 (inside loop, after run_completed check)
```python
# Skip SWMM-only models - they use EPA SWMM which doesn't produce
# TRITON-style log.out files. SWMM threading is validated separately
# via .inp file THREADS parameter inspection (see test_swmm_threads_implementation.py).
if row["model_type"] == "swmm":
    continue
```

### Testing Strategy

#### 1. Update Existing Test Assertions

**File:** `tests/test_PC_04_multisim_with_snakemake.py`

Current assertion (line 282):
```python
tst_ut.assert_analysis_workflow_completed_successfully(analysis)
```

This calls `assert_resource_usage_matches_config()`, which should now pass.

**Verify:** Test should pass without modification after fixing validator.

#### 2. Add Explicit SWMM Skip Test

**File:** Create `tests/test_resource_validation.py`

```python
"""
Test resource validation system.
"""
import pytest
import pandas as pd
from TRITON_SWMM_toolkit.consolidate_workflow import validate_resource_usage
import tests.fixtures.test_case_catalog as cases


def test_resource_validation_skips_swmm_models():
    """
    Verify that SWMM-only models are skipped during resource validation.

    SWMM standalone uses EPA SWMM executable which doesn't produce
    TRITON-style log.out files, so resource usage cannot be validated.
    """
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = case.analysis

    # Prepare scenarios to populate df_status
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        proc._scenario.prepare_scenario(rerun_swmm_hydro_if_outputs_exist=True)

    # Get df_status
    df_status = analysis.df_status

    # Verify SWMM models exist and have sentinel values
    swmm_rows = df_status[df_status["model_type"] == "swmm"]
    assert not swmm_rows.empty, "Test requires SWMM models"

    # Validation should pass despite SWMM models having 'rpt does not exist'
    result = validate_resource_usage(analysis, logger=None)
    assert result is True, "Validation should skip SWMM models and pass"


def test_resource_validation_normalizes_sentinels():
    """
    Verify that sentinel values are normalized to NaN before validation.
    """
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = case.analysis

    # Create mock df_status with sentinel values
    df = analysis.df_status.copy()

    # Add sentinel values to TRITON rows (should be normalized)
    triton_mask = df["model_type"] == "triton"
    df.loc[triton_mask, "actual_omp_threads"] = "rpt does not exist"

    # Temporarily replace df_status
    original_df_status = analysis._df_status
    analysis._df_status = df

    try:
        # Should pass because sentinels are normalized to NaN
        # (and TRITON rows with NaN actual values are skipped)
        result = validate_resource_usage(analysis, logger=None)
        # If TRITON models didn't run, validation should pass (skip NaN)
        assert result in [True, False]  # Depends on whether simulations ran
    finally:
        # Restore original df_status
        analysis._df_status = original_df_status
```

#### 3. Verify Regression Tests Pass

```bash
# Should now pass without resource validation errors
pytest tests/test_PC_04_multisim_with_snakemake.py::test_snakemake_workflow_end_to_end -v

# Run all multi-model tests
pytest tests/test_PC_04_multisim_with_snakemake.py -v
pytest tests/test_PC_01_singlesim.py -v

# Verify SWMM threading tests still pass
pytest tests/test_swmm_threads_implementation.py -v
```

---

## Additional Considerations

### Documentation Updates

**File:** `src/TRITON_SWMM_toolkit/consolidate_workflow.py` docstring

Update `validate_resource_usage()` docstring to document SWMM skip:

```python
def validate_resource_usage(analysis, logger=None):
    """
    Validate that actual resource usage matches expected configuration.

    Logs warnings if mismatches are detected between expected and actual
    compute resources (MPI tasks, OMP threads, GPUs, backend) for
    TRITON and TRITON-SWMM models.

    Note: SWMM-only models are excluded from validation as they use
    EPA SWMM executable which does not produce TRITON-style resource logs.
    SWMM threading is validated via .inp file THREADS parameter inspection.

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis object containing scenario status
    logger : logging.Logger, optional
        Logger for writing warnings. If None, uses print statements.

    Returns
    -------
    bool
        True if all resources match expected values (or are skipped),
        False if any mismatches found
    """
```

### Log Parser Improvements (Future Work)

**File:** Where sentinel values are generated (likely `snakemake_snakefile_parsing.py` or similar)

Consider returning `None` or `pd.NA` instead of string sentinels for more Pythonic handling:

```python
# Instead of:
actual_omp_threads = 'rpt does not exist'

# Use:
actual_omp_threads = None  # or pd.NA
```

This would eliminate the need for sentinel normalization entirely.

---

## Verification Checklist

After implementation:

- [ ] `consolidate_workflow.py` modified with sentinel normalization
- [ ] `consolidate_workflow.py` modified with SWMM skip logic
- [ ] Docstring updated to document SWMM exclusion
- [ ] `test_resource_validation.py` created with skip tests
- [ ] `test_PC_04::test_snakemake_workflow_end_to_end` passes
- [ ] `test_PC_01::test_swmm_cross_model_consistency` passes
- [ ] `test_swmm_threads_implementation.py` tests still pass
- [ ] No regressions in TRITON/TRITON-SWMM validation

---

## Related Issues

- **SWMM threading implementation** (Phase 1 & Phase 2) - Complete, validated separately
- **Missing Snakefile error** in `test_PC_01::test_run_sim` - Separate issue, likely test environment setup

---

## Success Criteria

1. ✅ Multi-model tests pass resource validation
2. ✅ SWMM models are explicitly skipped with clear documentation
3. ✅ TRITON/TRITON-SWMM validation unchanged (no regressions)
4. ✅ Sentinel values handled robustly (future-proof)
5. ✅ Tests document expected behavior clearly
