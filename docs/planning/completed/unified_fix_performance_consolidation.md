# Unified Plan: Fix Performance Consolidation + Complete MODE_CONFIG Refactor

**Status:** ✅ Implemented (2026-02-13)
**Owner:** Toolkit maintainers
**Created:** 2026-02-13
**Implemented:** 2026-02-13
**Location:** Top-level `active/` (spans both bug fix and refactor categories)
**Supersedes:**
- `bugs/fix_missinig_triton_only_performance_summary_and_raise_errors_if_failed.md` (deleted)
- `refactors/refac_make_tritonswmm_performance_used_in_MODE_CONFIG.md` (deleted)

## Executive Summary

This plan unifies two related improvements:
1. **Bug fix**: TRITON-only sensitivity analyses missing master performance summaries
2. **Refactor**: Route TRITONSWMM performance through `_MODE_CONFIG` for consistency

By implementing both together, we achieve a cleaner, more maintainable solution with
unified validation across all performance consolidation paths.

## Current Problems

### Problem 1: TRITON-only Performance Missing in Sensitivity Analysis

**Symptom**: `test_PC_06_sensitivity_analysis_triton_and_swmm_only.py::test_sensitivity_analysis_triton_only_execution`
passes (workflow reports success) but assertion fails because `TRITON_only_performance.zarr`
is missing at master analysis level.

**Root Cause**: `sensitivity_analysis.py::consolidate_TRITONSWMM_performance_summaries()`
(lines 535-558) only runs when `toggle_tritonswmm_model == True`. For TRITON-only runs,
the entire performance consolidation block is skipped.

**Evidence**:
- Sub-analysis files exist: `subanalyses/sa_*/TRITON_only_performance.zarr`
- Master file missing: No file at master analysis root
- Log flag incorrectly null: `"triton_only_performance_analysis_summary_created": null`

### Problem 2: Inconsistent Performance Consolidation Architecture

**Symptom**: Performance outputs handled inconsistently compared to other outputs.

**Current State**:
- `triton_only_performance` **is** in `_MODE_CONFIG` ✅
- `tritonswmm_performance` **is not** in `_MODE_CONFIG` ❌ (uses bespoke method)

**Issues**:
- Duplicate consolidation logic (shared `_consolidate_outputs()` vs special-case method)
- Different validation patterns across output types
- Risk of divergence between performance and spatial outputs

### Problem 3: Silent Consolidation Failures

**Symptom**: Workflow marks consolidation as success even when output files don't exist.

**Root Cause**: Log flags set immediately after `_write_output()` without validating
file existence. If write fails silently or path issues occur, flag still gets set.

**Locations**:
- `processing_analysis.py::_consolidate_outputs()` (line 405)
- `processing_analysis.py::consolidate_TRITONSWMM_performance_summaries()` (line 329)
- `sensitivity_analysis.py::consolidate_TRITONSWMM_performance_summaries()` (line 553)

## Unified Solution

### Phase 1: Complete the `_MODE_CONFIG` Pattern

**Goal**: Add `tritonswmm_performance` to `_MODE_CONFIG` to match `triton_only_performance`.

**File**: `src/TRITON_SWMM_toolkit/processing_analysis.py`

```python
_MODE_CONFIG = {
    # ... existing entries ...
    "triton_only_performance": (
        "output_triton_only_performance_summary",
        "output_triton_only_performance_summary",
        "triton_only_performance_analysis_summary_created",
        None,  # Non-spatial
    ),
    # NEW: Add TRITONSWMM performance
    "tritonswmm_performance": (
        "output_tritonswmm_performance_summary",
        "output_tritonswmm_performance_summary",
        "tritonswmm_performance_analysis_summary_created",
        None,  # Non-spatial
    ),
}
```

### Phase 2: Add Fail-Fast Validation to Consolidation

**Goal**: Ensure files actually exist before setting success flags.

**File**: `src/TRITON_SWMM_toolkit/processing_analysis.py`

**Location**: In `_consolidate_outputs()`, after line 404 (`self._write_output(...)`)

```python
def _consolidate_outputs(
    self,
    ds_combined_outputs: xr.Dataset | xr.DataArray,
    mode: str,
    overwrite_outputs_if_already_created: bool = False,
    verbose: bool = False,
    compression_level: int = 5,
):
    # ... existing code ...

    self._write_output(
        ds_combined_outputs, fname_out, compression_level, chunks, verbose
    )

    # NEW: Validate output was actually created before setting success flag
    if fname_out.suffix == ".zarr":
        if not fname_out.exists() or not (fname_out / ".zgroup").exists():
            raise RuntimeError(
                f"Zarr consolidation failed for mode '{mode}': "
                f"output missing or incomplete at {fname_out}"
            )
    elif not fname_out.exists():
        raise RuntimeError(
            f"Consolidation failed for mode '{mode}': "
            f"output not created at {fname_out}"
        )

    proc_log.set(True)  # Only set after validation passes
    elapsed_s = time.time() - start_time
    self._analysis.log.add_sim_processing_entry(
        fname_out, get_file_size_MiB(fname_out), elapsed_s, True
    )
    return
```

### Phase 3: Migrate Standard Analysis to Use MODE_CONFIG

**Goal**: Replace bespoke `consolidate_TRITONSWMM_performance_summaries()` with
unified `consolidate_outputs_for_mode()`.

**File**: `src/TRITON_SWMM_toolkit/analysis.py`

**Before**:
```python
if cfg_sys.toggle_tritonswmm_model:
    if verbose:
        print("Consolidating TRITON-SWMM coupled model outputs...", flush=True)
    _consolidate("tritonswmm_triton")
    _consolidate("tritonswmm_swmm_node")
    _consolidate("tritonswmm_swmm_link")
    self.process.consolidate_TRITONSWMM_performance_summaries(...)  # Special case
```

**After**:
```python
if cfg_sys.toggle_tritonswmm_model:
    if verbose:
        print("Consolidating TRITON-SWMM coupled model outputs...", flush=True)
    _consolidate("tritonswmm_triton")
    _consolidate("tritonswmm_swmm_node")
    _consolidate("tritonswmm_swmm_link")
    _consolidate("tritonswmm_performance")  # Now uses MODE_CONFIG
```

### Phase 4: Fix Sensitivity Analysis Performance Consolidation

**Goal**: Enable performance consolidation for both TRITONSWMM and TRITON-only models
in sensitivity workflows using the unified MODE_CONFIG pipeline.

**File**: `src/TRITON_SWMM_toolkit/sensitivity_analysis.py`

**Replace** the current special-case block (lines 535-558):

```python
# OLD CODE (lines 535-558):
if cfg_sys.toggle_tritonswmm_model:
    # consolidate performance summaries (independent of 'which' parameter)
    start_time = time.time()
    ds_performance = self._combine_TRITONSWMM_performance_per_subanalysis()
    proc_log = (
        self.master_analysis.log.tritonswmm_performance_analysis_summary_created
    )
    fname_out = (
        self.master_analysis.analysis_paths.output_tritonswmm_performance_summary
    )
    self.master_analysis.process._write_output(...)
    proc_log.set(True)
    elapsed_s = time.time() - start_time
    self.master_analysis.log.add_sim_processing_entry(...)

return
```

**With** unified MODE_CONFIG approach:

```python
# NEW CODE: Use MODE_CONFIG for both model types
cfg_sys = self.master_analysis._system.cfg_system

# Determine which performance mode to use
if cfg_sys.toggle_tritonswmm_model:
    perf_mode = "tritonswmm_performance"
elif cfg_sys.toggle_triton_model:
    perf_mode = "triton_only_performance"
else:
    # No TRITON-based model enabled, skip performance consolidation
    return

# Combine performance from all sub-analyses
ds_performance = self._combine_TRITONSWMM_performance_per_subanalysis()

# Use unified consolidation pipeline (includes fail-fast validation from Phase 2)
self.master_analysis.process._consolidate_outputs(
    ds_performance,
    mode=perf_mode,
    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
    verbose=verbose,
    compression_level=compression_level,
)

return
```

**Key insight**: The `_combine_TRITONSWMM_performance_per_subanalysis()` method
(lines 384-414) **already has correct conditional logic** to handle both model types
(lines 397-404), so we can reuse it for both cases. The MODE_CONFIG routing ensures
correct paths and log flags are used.

### Phase 5: Deprecate Bespoke Method

**Goal**: Remove duplicate code path to prevent future divergence.

**File**: `src/TRITON_SWMM_toolkit/processing_analysis.py`

**Option A (preferred)**: Remove `consolidate_TRITONSWMM_performance_summaries()` entirely.

**Option B (safer)**: Keep as deprecated wrapper:

```python
def consolidate_TRITONSWMM_performance_summaries(
    self,
    overwrite_outputs_if_already_created: bool = False,
    verbose: bool = False,
    compression_level: int = 5,
):
    """
    DEPRECATED: Use consolidate_outputs_for_mode('tritonswmm_performance') instead.

    This method is maintained for backward compatibility but will be removed in a
    future version. All new code should use the MODE_CONFIG consolidation pipeline.
    """
    return self.consolidate_outputs_for_mode(
        "tritonswmm_performance",
        overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
        verbose=verbose,
        compression_level=compression_level,
    )
```

## Implementation Order

1. **Phase 1**: Add `tritonswmm_performance` to `_MODE_CONFIG`
2. **Phase 2**: Add fail-fast validation to `_consolidate_outputs()`
3. **Phase 3**: Migrate standard analysis consolidation
4. **Phase 4**: Fix sensitivity analysis consolidation
5. **Phase 5**: Deprecate/remove bespoke method

**Rationale**: Phases 1-2 establish the foundation. Phase 3 migrates existing working
code. Phase 4 fixes the bug. Phase 5 cleans up.

## Testing Plan

### Regression Tests (ensure no breakage)

1. **Standard multi-sim with TRITONSWMM**:
   ```bash
   pytest tests/test_PC_04_multisim_with_snakemake.py -v
   ```
   - Verify `TRITONSWMM_performance.zarr` still created
   - Check log flag set correctly

2. **Standard multi-sim with TRITON-only**:
   ```bash
   pytest tests/test_PC_01_singlesim.py -v  # (if has TRITON-only variant)
   ```
   - Verify `TRITON_only_performance.zarr` created
   - Check log flag set correctly

3. **Sensitivity with TRITONSWMM**:
   ```bash
   pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py -v
   ```
   - Verify master `TRITONSWMM_performance.zarr` created
   - Check consolidation flag set correctly

### Bug Fix Validation

4. **TRITON-only sensitivity** (primary bug fix):
   ```bash
   pytest tests/test_PC_06_sensitivity_analysis_triton_and_swmm_only.py::test_sensitivity_analysis_triton_only_execution -v
   ```
   - **Success criteria**:
     - Master directory contains `TRITON_only_performance.zarr`
     - Log flag `triton_only_performance_analysis_summary_created` is `True`
     - Test assertion `assert_analysis_summaries_created()` **passes**

5. **SWMM-only sensitivity** (verify no regression):
   ```bash
   pytest tests/test_PC_06_sensitivity_analysis_triton_and_swmm_only.py::test_sensitivity_analysis_swmm_only_execution -v
   ```

### Fail-Fast Validation Tests

6. **Simulate missing output**:
   - Temporarily delete consolidated output after write
   - Re-run consolidation
   - **Success criteria**: Workflow raises `RuntimeError` and does NOT set log flag

7. **Simulate incomplete zarr**:
   - Create zarr directory without `.zgroup` file
   - Re-run consolidation
   - **Success criteria**: Workflow raises `RuntimeError` mentioning "incomplete"

## Success Criteria

- ✅ All existing tests pass without modification
- ✅ TRITON-only sensitivity produces master `TRITON_only_performance.zarr`
- ✅ TRITONSWMM performance uses MODE_CONFIG (no special-case method)
- ✅ Consolidation fails fast if output files missing
- ✅ Log flags only set when files actually exist
- ✅ Unified consolidation pipeline for all output types

## Benefits

1. **Bug fix**: TRITON-only sensitivity analyses now produce master performance files
2. **Consistency**: All output types use same consolidation pipeline
3. **Robustness**: Fail-fast validation prevents silent failures
4. **Maintainability**: Single code path to maintain, test, and debug
5. **Future-proof**: Easier to add new output types (follow MODE_CONFIG pattern)

## Risks & Mitigation

**Risk**: Breaking existing workflows that depend on bespoke method.
- **Mitigation**: Phase 5 offers Option B (deprecated wrapper) for gradual migration.

**Risk**: Strict validation may surface previously hidden issues.
- **Mitigation**: This is actually desirable—better to fail loud than succeed silently.

**Risk**: zarr validation logic may have edge cases.
- **Mitigation**: Test with both `.zarr` and `.nc` outputs. Check for `.zgroup`
  (always present in zarr) rather than `.zmetadata` (only in consolidated stores).

## Notes

- Performance datasets are **non-spatial** (`spatial_coords=None` in MODE_CONFIG)
- `_chunk_for_writing()` already handles `spatial_coords=None` correctly
- This refactor aligns with CLAUDE.md philosophy: "Backward compatibility is NOT a
  priority"—clean code > maintaining deprecated APIs
- Both `.zarr` and `.nc` output formats tested and supported

## Related Documentation

- `CLAUDE.md`: Development philosophy (backward compatibility, log-based checks)
- `tests/utils_for_testing.py::assert_analysis_summaries_created()`: Test assertion
  that validates file existence
- `.claude/agents/output-processing.md`: Output processing patterns
- `.claude/agents/sensitivity-analysis.md`: Sensitivity workflow architecture
