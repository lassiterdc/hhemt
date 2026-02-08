# Cruft Cleanup Tracker

This tracker is the phase-by-phase inventory for the cleanup work defined in
`docs/planning/cruft_cleanup_plan.md`.

## Ground Rules

- No net-new legacy aliases/shims in runtime/library code unless explicitly approved.
- Configuration-format compatibility is allowed where practical.
- Required smoke tests for cleanup changes (run in this order):
  1. `tests/test_PC_01_singlesim.py`
  2. `tests/test_PC_02_multisim.py`
  3. `tests/test_PC_04_multisim_with_snakemake.py`
  4. `tests/test_PC_05_sensitivity_analysis_with_snakemake.py`

## Phase Inventory

### Phase 0 — Baseline & Safety Rails

- Status: **In Progress**
- Touched files:
  - `docs/planning/cruft_cleanup_tracker.md` (new)
  - `src/TRITON_SWMM_toolkit/config.py` (Phase 1 prep overlap)
  - `tests/test_config_validation.py` (new; validation-focused tests)

#### Baseline smoke-test snapshot (required order)

Environment used: `triton_swmm_toolkit` conda env

1. `tests/test_PC_01_singlesim.py` → **FAIL** (exit 1)
   - 4 failed, 1 passed
   - Representative failures:
     - missing expected SWMM/TRITON-SWMM time series outputs
     - upstream run/process failures cascading into assertions
2. `tests/test_PC_02_multisim.py` → **FAIL** (exit 1)
   - 2 failed
   - Representative failure:
     - `RuntimeError: TRITON simulation not completed` during post-processing
3. `tests/test_PC_04_multisim_with_snakemake.py` → **FAIL** (exit 1)
   - 2 failed, 4 passed, 1 skipped
   - Representative failure:
     - Snakemake dry-run abort because `snakemake` executable is not present in env
4. `tests/test_PC_05_sensitivity_analysis_with_snakemake.py` → **INCONCLUSIVE / TIMEBOXED**
   - repeatedly long-running in this environment
   - explicit timeboxed run (`timeout 120 ...`) ended with exit `124`

Baseline interpretation:
- Phase-0 gate is not currently green prior to cleanup changes.
- Existing environment/runtime dependencies and workflow issues must be accounted for
  when evaluating cleanup PR regressions.

### Phase 1 — Configuration Layer Refactor

- Status: **Complete (validator/strictness subset; structural split deferred)**
- Touched files:
  - `src/TRITON_SWMM_toolkit/config.py`
  - `src/TRITON_SWMM_toolkit/analysis.py`
  - `tests/test_config_validation.py`
  - `test_data/norfolk_coastal_flooding/**/*.yaml` (9 files: removed dead legacy keys)
- Implemented:
  - enforced strict unknown-key behavior via `extra="forbid"` on config base model
  - replaced dynamic toggle test registry pattern with explicit `@model_validator` rules
    in `system_config` and `analysis_config`
  - removed dead legacy fields (`TRITON_SWMM_make_command`, `toggle_run_ensemble_with_bash_script`)
    from both config model and all test YAML files
  - removed commented-out dead code blocks from `config.py` and `analysis.py`
- Deferred:
  - Structural split into `config/` package → separate future PR
- Test status:
  - `tests/test_config_validation.py` → **PASS** (3 passed)

### Phase 2+

- Status: Not started
