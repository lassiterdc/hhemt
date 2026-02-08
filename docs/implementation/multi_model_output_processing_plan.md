# Multi-Model Output Processing Implementation Record

**Status:** ✅ Complete
**Started:** 2026-02-05
**Last Updated:** 2026-02-07

## Overview

This work implemented model-aware output processing for all supported model types:
1. **TRITON-only**
2. **TRITON-SWMM (coupled)**
3. **SWMM-only**

The original implementation plan has been executed. This document is now a
completed implementation record (not an active task plan).

## Completed Scope

- Added model-type routing for timeseries and summary exporters.
- Implemented TRITON-only processing and summary generation.
- Implemented SWMM-only summary generation and model-aware SWMM processing paths.
- Updated workflow integration so process rules pass model-appropriate `--which`
  behavior.
- Added runner-side model-aware validation for processing requests.
- Addressed pathing and type-annotation issues found during implementation.
- Integrated with later model-specific logging architecture updates.

## Key Outcomes

- All enabled model types now produce expected processed outputs.
- Export methods fail clearly when required model-specific inputs are missing.
- Routing logic avoids duplicate code while preserving model-specific behavior.
- Workflow/runners align with model-specific process rules.

## Regression Validation Order

Canonical smoke/regression order used for this implementation:
1. `pytest tests/test_PC_01_singlesim.py`
2. `pytest tests/test_PC_02_multisim.py`
3. `pytest tests/test_PC_04_multisim_with_snakemake.py`
4. `pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py`

## Related Files

- `src/TRITON_SWMM_toolkit/process_simulation.py`
- `src/TRITON_SWMM_toolkit/process_timeseries_runner.py`
- `src/TRITON_SWMM_toolkit/workflow.py`
- `src/TRITON_SWMM_toolkit/run_simulation.py`
- `src/TRITON_SWMM_toolkit/log.py`

## Notes

- Historical in-progress checklists/debugging placeholders were removed to avoid
  contradictory status signals (e.g., "complete" vs "not started").
- For active priorities and next actionable work, see:
  - `docs/planning/priorities.md`
  - `docs/planning/cruft_cleanup_plan.md`
