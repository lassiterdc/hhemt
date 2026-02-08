# PC04 Review Fixes — Implementation Complete

## Summary

All code changes from the PC04 review have been successfully implemented. The changes address 8 issues identified during code review, with a focus on:
- Documenting the TRITON output path bug
- Simplifying path resolution logic
- Adding fail-fast error handling
- Centralizing workaround code with TODO tags
- Extracting test helpers

## Changes Made

### 1. Bug Documentation
**File:** `docs/implementation/triton_output_path_bug.md` (NEW)
- Documents that TRITON-SWMM ignores CFG `output_folder` for SWMM outputs and `log.out`
- Establishes `TODO(TRITON-OUTPUT-PATH-BUG)` tag convention
- Lists all affected code locations for future cleanup

### 2. Path Resolution Simplification
**Files:** `src/TRITON_SWMM_toolkit/run_simulation.py`, `src/TRITON_SWMM_toolkit/process_simulation.py`

**Before:** Multi-candidate directory search in 3 separate locations
**After:** Use configured paths directly with single fallback

- `_triton_swmm_raw_output_directory`: Now checks `out_tritonswmm` first, then falls back to `output/`
- `raw_triton_output_dir`: Returns the raw binary directory (`bin/` subdirectory)
- Removed inline candidate searches from `_export_TRITONSWMM_TRITON_outputs` and `_export_TRITON_only_outputs`

### 3. Centralized SWMM Output Path Workaround
**File:** `src/TRITON_SWMM_toolkit/run_simulation.py`

New property `coupled_swmm_output_file`:
- Checks bug location (`output/swmm/`) first
- Falls back to configured path
- Returns `None` if not found
- Tagged with `TODO(TRITON-OUTPUT-PATH-BUG)`

### 4. Fail-Fast Error Handling
**File:** `src/TRITON_SWMM_toolkit/process_simulation.py`

Changed 3 silent returns to explicit errors:
- `_export_TRITONSWMM_TRITON_outputs` (2 locations)
- `_export_SWMM_outputs` tritonswmm branch

Removed redundant file existence guard (kept only log-based check).

### 5. TODO Tags for log.out References
**Files:** `src/TRITON_SWMM_toolkit/analysis.py`, `src/TRITON_SWMM_toolkit/sensitivity_analysis.py`

Added `TODO(TRITON-OUTPUT-PATH-BUG)` comments before `log_out_path` assignments to document the overwrite issue with multi-model runs.

### 6. Capacity Normalization Helper
**File:** `tests/utils_for_testing.py` (NEW function)

Extracted `normalize_swmm_link_vars()` helper:
- Handles `capacity` (binary parser) vs `capacity_setting` (text parser) naming difference
- Both represent the same physical quantity (fraction of conduit filled, 0-1 range)
- Used in `test_PC_01_singlesim.py` and `test_PC_04_multisim_with_snakemake.py`

**Bug fix:** PC01 had a no-op normalization (checked for `"capacity_setting"`, then replaced it with itself). Now uses the correct helper.

### 7. CLAUDE.md Updates
**File:** `CLAUDE.md`

Added two sections:
- **Log-Based Check Philosophy** (after Backward Compatibility): Prefer `_already_written()` over file existence checks
- **Gotchas #5-6**: Documented TRITON output path bug and log.out overwrite issue

## Verification

Run these tests to verify the changes:

```bash
# Smoke test (faster)
pytest tests/test_PC_02_multisim.py -q

# Primary E2E (the test that was previously broken)
pytest tests/test_PC_04_multisim_with_snakemake.py -k test_snakemake_workflow_end_to_end -q
```

Expected outcome: Both tests should pass. The PC04 test now uses simplified path resolution and fail-fast error handling, making failures more debuggable.

## Files Modified

| File | Lines Changed | Change Type |
|------|---------------|-------------|
| `docs/implementation/triton_output_path_bug.md` | +55 | NEW |
| `src/.../run_simulation.py` | +32, -17 | Simplify paths; add property |
| `src/.../process_simulation.py` | +23, -39 | Simplify paths; fail-fast errors |
| `src/.../analysis.py` | +3 | TODO tag |
| `src/.../sensitivity_analysis.py` | +3 | TODO tag |
| `tests/utils_for_testing.py` | +23 | NEW function |
| `tests/test_PC_01_singlesim.py` | +2, -6 | Use helper; fix bug |
| `tests/test_PC_04_multisim_with_snakemake.py` | +2, -5 | Use helper |
| `CLAUDE.md` | +14 | Philosophy + gotchas |

**Total:** ~100 lines changed across 9 files (net reduction due to simplification)

## Key Insights

1. **Diagnostic confirmed expectations**: TRITON raw outputs DO go to configured directories (`out_triton/bin/`, `out_tritonswmm/bin/`). Only SWMM outputs and `log.out` have the bug.

2. **Path simplification safe**: Since TRITON respects `output_folder` for raw outputs, we could safely remove the 4-candidate search pattern.

3. **Log-based checks prevent race conditions**: File existence checks can give false positives (incomplete/corrupt files), while `_already_written()` verifies successful completion.

4. **Capacity normalization was needed**: The no-op bug in PC01 showed that manual normalization was error-prone. The extracted helper prevents future mistakes.

5. **TODO tags enable future cleanup**: When TRITON-SWMM developer fixes the output path bug, searching for `TODO(TRITON-OUTPUT-PATH-BUG)` will find all workarounds to remove.

## Next Steps

When the TRITON-SWMM executable is fixed to respect `output_folder`:
1. Search for `TODO(TRITON-OUTPUT-PATH-BUG)` tags
2. Remove `coupled_swmm_output_file` property and all references
3. Simplify `_export_SWMM_outputs` and `_clear_raw_SWMM_outputs` tritonswmm branches
4. Remove hardcoded `log_out_path` assignments
5. Delete `docs/implementation/triton_output_path_bug.md`
6. Remove gotchas #5-6 from CLAUDE.md
