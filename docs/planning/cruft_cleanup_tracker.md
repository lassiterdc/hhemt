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

1. `tests/test_PC_01_singlesim.py` → **PASS** (5 passed)
2. `tests/test_PC_02_multisim.py` → **PASS** (2 passed)
3. `tests/test_PC_04_multisim_with_snakemake.py` → **PASS** (6 passed, 1 skipped)
4. `tests/test_PC_05_sensitivity_analysis_with_snakemake.py` → **PASS** (5 passed, ~12.5 min)
   - Note: this suite is legitimately slow — the fixture runs `process_system_level_inputs`
     (DEM coarsening) once per test function (~54s each). No artificial timeout should be applied.

Baseline interpretation:
- All smoke tests are green. Phase-0 gate is met.

### Phase 1 — Configuration Layer Refactor

- Status: **Complete**
- Touched files:
  - `src/TRITON_SWMM_toolkit/config.py` → deleted; replaced by `config/` package
  - `src/TRITON_SWMM_toolkit/config/__init__.py` (new; docstring only, no re-exports)
  - `src/TRITON_SWMM_toolkit/config/base.py` (new; cfgBaseModel)
  - `src/TRITON_SWMM_toolkit/config/system.py` (new; system_config)
  - `src/TRITON_SWMM_toolkit/config/analysis.py` (new; analysis_config)
  - `src/TRITON_SWMM_toolkit/config/loaders.py` (new; load_* functions)
  - `src/TRITON_SWMM_toolkit/analysis.py` (import updated)
  - `src/TRITON_SWMM_toolkit/system.py` (import updated)
  - `src/TRITON_SWMM_toolkit/examples.py` (import updated)
  - `src/TRITON_SWMM_toolkit/case_study_catalog.py` (import updated)
  - `src/TRITON_SWMM_toolkit/gui.py` (removed dead SimulationConfig/ConfigGUI code)
  - `tests/test_config_validation.py` (imports updated)
  - `tests/fixtures/test_case_builder.py` (imports updated)
  - `scripts/check_doc_freshness.py` (`"config.py"` → `"config/"` in filename keys)
  - `test_data/norfolk_coastal_flooding/**/*.yaml` (9 files: removed dead legacy keys)
- Implemented:
  - enforced strict unknown-key behavior via `extra="forbid"` on config base model
  - replaced dynamic toggle test registry pattern with explicit `@model_validator` rules
    in `system_config` and `analysis_config`
  - removed dead legacy fields (`TRITON_SWMM_make_command`, `toggle_run_ensemble_with_bash_script`)
    from both config model and all test YAML files
  - removed commented-out dead code blocks
  - split `config.py` (642 lines, mixed concerns) into single-responsibility submodules
  - no compatibility shims — all import sites updated immediately
- Test status:
  - `tests/test_config_validation.py` → **PASS** (3 passed)
  - All 4 smoke tests → **PASS** (PC_01 through PC_05)

### Phase 2 — Remove Legacy/Obsolete Runtime Paths

- Status: **Complete**
- Touched files:
  - `src/TRITON_SWMM_toolkit/run_simulation.py`
  - `src/TRITON_SWMM_toolkit/scenario.py`
  - `src/TRITON_SWMM_toolkit/analysis.py`
  - `src/TRITON_SWMM_toolkit/log.py`
  - `src/TRITON_SWMM_toolkit/process_simulation.py`
  - `src/TRITON_SWMM_toolkit/process_timeseries_runner.py`
- Removed:
  - `_obsolete_retrieve_sim_launcher()` and `_obsolete_run_sim()` methods
  - `SimEntry` and `SimLog` Pydantic classes (simlog tracking fully retired)
  - `sim_log` field from `TRITONSWMM_model_log` (was always empty `{"run_attempts": {}}`)
  - `latest_simlog` property, `_latest_sim_status()`, `sim_run_status()`, `_simulation_run_statuses`
  - Commented-out srun/mpirun/gpu command alternatives and simlog tracking blocks
- Net: 305 lines deleted, 13 inserted
- All smoke tests pass (PC_01 through PC_05)

### Phase 3+

- Status: Not started
