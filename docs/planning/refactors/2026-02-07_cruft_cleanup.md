# Cruft Cleanup

**Written**: 2026-02-07
**Last edited**: 2026-02-28 — merged plan and tracker into single doc

---

## Purpose

Structured, repo-wide cleanup of accumulated cruft across source code, test suite, and documentation. Aligns with the project philosophy: no backward-compatibility shims, clean replacement over alias preservation, log-based completion semantics.

---

## Governing Philosophy

- **No net-new legacy aliases/shims** in runtime/library code unless explicitly approved
- **Configuration-format compatibility** is allowed where practical
- **Prefer log-based completion checks** over file-existence checks
- **Smoke test sequence** (run in order after each phase):
  1. `pytest tests/test_PC_01_singlesim.py`
  2. `pytest tests/test_PC_02_multisim.py`
  3. `pytest tests/test_PC_04_multisim_with_snakemake.py`
  4. `pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py`

---

## Phase 0 — Baseline & Safety Rails ✅ Complete

Established smoke test baseline (all 4 tests green). Created this tracker.

---

## Phase 1 — Configuration Layer Refactor ✅ Complete

Split `config.py` (642 lines, mixed concerns) into `config/` package:

| File | Contents |
|------|----------|
| `config/base.py` | `cfgBaseModel` |
| `config/system.py` | `system_config` |
| `config/analysis.py` | `analysis_config` |
| `config/loaders.py` | `load_*` functions |

Changes:
- `extra="forbid"` enforced on `cfgBaseModel`
- Dynamic `toggle_tests` registry replaced with explicit `@model_validator` rules
- Dead legacy fields removed (`TRITON_SWMM_make_command`, `toggle_run_ensemble_with_bash_script`)
- All import sites updated immediately (no compatibility shims)
- Removed dead `SimulationConfig`/`ConfigGUI` code from `gui.py`

All smoke tests pass.

---

## Phase 2 — Remove Legacy/Obsolete Runtime Paths ✅ Complete

Net: 305 lines deleted, 13 inserted.

Removed from `run_simulation.py`, `scenario.py`, `analysis.py`, `log.py`, `process_simulation.py`, `process_timeseries_runner.py`:
- `_obsolete_retrieve_sim_launcher()` and `_obsolete_run_sim()` methods
- `SimEntry` and `SimLog` Pydantic classes (simlog tracking retired)
- `sim_log` field from `TRITONSWMM_model_log`
- `latest_simlog` property, `_latest_sim_status()`, `sim_run_status()`, `_simulation_run_statuses`
- Commented-out srun/mpirun/gpu command alternatives and simlog tracking blocks

All smoke tests pass.

---

## Phase 3 — Orchestration Deduplication ⏸ Not Started

Blocked: awaiting Phase 4 completion.

Goal: extract shared command-building and subprocess-launch patterns; unify model-routing logic repeated across `analysis.py`, `workflow.py`, `sensitivity_analysis.py`.

---

## Phase 4 — Logging & Error Contract Normalization 🔄 Partially Complete

### 4a–4c ✅ Complete

Touched: `exceptions.py` (NEW), `__init__.py`, `system.py`, `scenario.py`.

- Custom exception hierarchy: `TRITONSWMMError`, `CompilationError`, `ConfigurationError`, `SimulationError`, `ProcessingError`, `WorkflowError`, `SLURMError`, `ResourceAllocationError`
- System/compilation layer: `CompilationError` with full context (logfile, return_code, model_type, backend)
- Scenario/run layer: no silent failures found

All smoke tests pass (PC_01: 5 passed 163s, PC_02: 2 passed 183s).

### 4d–4f ⏸ Deferred (non-critical)

- **4d**: Output processing logging (`process_simulation.py`, `swmm_output_parser.py`)
- **4e**: Workflow orchestration logging (`workflow.py`, `analysis.py`, `execution.py`)
- **4f**: Config validation error standardization (`config/system.py`, `config/analysis.py`)

Rationale: print→logger conversions without critical functionality impact. Current exception handling covers all critical paths.

---

## Phase 5 — Workaround Containment ⏸ Blocked

Blocked by upstream TRITON-SWMM fix for `output_folder` directive.

Goal: centralize all `TODO(TRITON-OUTPUT-PATH-BUG)` logic behind minimal interfaces. See `docs/implementation/triton_output_path_bug.md`.

---

## Phase 6 — Test Suite Cleanup 🔄 In Progress

### 6c ✅ Complete
Made test diagnostic prints opt-in via `verbose` parameter in `utils_for_testing.py`.

### 6d ✅ Complete
Assertion helper audit, implementation, migration, and documentation.

New helpers in `tests/utils_for_testing.py`:
- `assert_model_outputs_exist()` — consolidates 19 multi-model check patterns
- `assert_file_exists()` — standardizes 18 path existence checks
- `assert_phases_complete()` — leverages `WorkflowStatus` for phase validation
- `assert_model_simulations_complete()` — model-specific completion checking

Migrated 16 path existence patterns across PC, PILOT, UVA, Frontier tests.

### 6a.1 ✅ Complete
Platform parametrization pilot (`test_PILOT_platform_parametrized_workflow.py`): 2 tests × 3 platforms with automatic platform skipping.

### 6b.1 ✅ Complete
Fixture usage audit: 24 fixtures analyzed, 67% consolidation opportunity identified.

### 6b.2 ✅ Complete
Unified fixture API design: parametrized fixtures with platform selection, incremental migration strategy. Target: 24 → 8–10 fixtures.

### 6b.2.1 ✅ Complete
Pilot implementation validated locally. Ready for Phase 6b.2.2.

### Next steps
- **6b.2.2**: Expand unified fixture to UVA/Frontier platforms
- **6a.2**: Expand parametrization to more test files

---

## Phase 7 — Documentation Cleanup & Archival Policy ⏸ Not Started

Goal: classify all docs as Authoritative / Active Plan / Historical Archive / Delete. Prune stale implementation snapshots. Add doc freshness checklist for future PRs.

---

## Acceptance Criteria for Complete

1. No obsolete runtime paths (`_obsolete`, legacy aliases) without justification
2. Core orchestration paths deduplicated
3. Logging/error behavior consistent and actionable
4. Test suite has reduced duplication with preserved coverage intent
5. Documentation triaged and internally consistent
6. All decisions aligned with conventions.md philosophy
