# Development Priorities

**Last Updated:** 2026-02-28 (Frontier sensitivity suite complete ✅; UVA CPU suite complete ✅)
**Status:** Active — update this document as work progresses.

---

## Recently Completed Work

For reference, these major efforts are done and tested:

- [x] Multi-model integration (TRITON-only, TRITON-SWMM, SWMM-only concurrent execution)
- [x] Multi-model output processing (all 3 model types, timeseries + summaries)
- [x] Model-specific logs (`log_triton.json`, `log_tritonswmm.json`, `log_swmm.json`)
- [x] Log-based completion checking (replaces CFG-file-existence checks)
- [x] 1-job-many-srun-tasks SBATCH redesign (dynamic concurrency from SLURM allocation)
- [x] Wait-for-completion SLURM polling (two-stage squeue + sacct)
- [x] Conda activation in SLURM fix
- [x] Examples/test utilities refactor (67% reduction, platform configs centralized)
- [x] Shared orchestration core (analysis.run() API, WorkflowResult, workflow status reporting)
- [x] CLI refactor to use orchestration API (70 lines → 47 lines, thin adapter pattern)
- [x] Toolkit API facade (high-level notebook-friendly interface, comprehensive docstrings, example notebook)

---

## Tier 1: Cruft Cleanup & Code Quality

Incremental improvements that reduce maintenance burden. Can be done independently.

- [x] **Remove legacy/obsolete runtime paths** (`2026-02-07_cruft_cleanup.md` Phase 2)
  - Removed `_obsolete_*` methods and dead launch paths in `run_simulation.py`
  - Removed deprecated `SimLog` / `SimEntry` classes from `log.py`
  - Deleted commented-out simlog code
  - _Ref:_ `docs/planning/refactors/2026-02-07_cruft_cleanup.md` Phase 2

- [x] **Logging & error normalization** (`2026-02-07_cruft_cleanup.md` Phase 4, Phases 4a-4c)
  - ✅ Custom exception hierarchy (TRITONSWMMError, CompilationError, ConfigurationError, etc.)
  - ✅ System/compilation layer error handling with full context (logfile, return_code, model_type, backend)
  - ✅ Scenario/run layer error handling (no silent failures found)
  - ⏸️ Deferred: print→logger conversions in output processing, workflow orchestration, config validation (Phases 4d-4f) — non-critical cosmetic improvements
  - _Ref:_ `docs/planning/refactors/2026-02-07_cruft_cleanup.md` Phase 4

- [ ] **Workaround containment** (`2026-02-07_cruft_cleanup.md` Phase 5)
  - Centralize `TODO(TRITON-OUTPUT-PATH-BUG)` logic behind minimal interfaces
  - Currently in 4 source files — see `docs/implementation/triton_output_path_bug.md`
  - _Blocked by:_ Upstream TRITON-SWMM fix for `output_folder` directive

- [ ] **Test suite cleanup** (`2026-02-07_cruft_cleanup.md` Phase 6) — **In Progress**
  - [ ] **Phase 6a**: Parametrize repeated platform test patterns (6a.1 pilot ✅ complete, next: 6a.2 expand)
  - [ ] **Phase 6b**: Consolidate fixture factories (6b.1 audit ✅, 6b.2 design ✅, 6b.2.1 pilot ✅ validated, next: 6b.2.2 expand)
  - [x] **Phase 6c**: Reduce unconditional diagnostic prints (✅ complete)
  - [x] **Phase 6d**: Standardize assertions (✅ complete: audit, helpers, migration, documentation)
  - _Status:_ Phase 6b.2.1 validated (pilot test passes), ready for Phase 6b.2.2 (expand to UVA/Frontier)
  - _Ref:_ `docs/planning/refactors/2026-02-09_test_suite_cleanup_phase6_plan.md`, `docs/planning/refactors/completed/2026-02-09_test_fixture_audit_phase6b1.md`, `docs/planning/refactors/2026-02-09_test_fixture_consolidation_phase6b2_design.md`

---

## Tier 2: Config Refactor

Foundation for CLI/API work. Should be done before Tier 3.

- [x] **Split `config.py` into focused modules** (`2026-02-07_config_py_refactor_plan.md` Phase 1)
  - Created `config/` package: base, system, analysis, loaders (validation/display deferred)
  - Updated all import sites immediately (no compatibility shims)
  - Removed dead `gui.py` code (broken SimulationConfig/ConfigGUI)
  - _Ref:_ `docs/planning/refactors/completed/2026-02-07_config_py_refactor_plan.md`, `docs/planning/refactors/2026-02-07_cruft_cleanup.md` Phase 1

- [x] **Strict validation + validator cleanup** (`2026-02-07_config_py_refactor_plan.md` Phase 2)
  - `extra="forbid"` enforced on `cfgBaseModel`
  - Dynamic `toggle_tests` registry replaced with explicit `@model_validator` rules
  - Dead legacy fields and commented-out code removed

- [x] **Front-end validation checklist** (implement preflight checks)
  - ✅ Validation architecture (ValidationResult, ValidationIssue, preflight_validate)
  - ✅ System config validation (paths, toggle dependencies, model selection)
  - ✅ Analysis config validation (weather data, run-mode consistency, HPC sanity)
  - ✅ Data cross-consistency (event alignment, storm tide, units)
  - ✅ Analysis.validate() method integration
  - ⏸️ Deferred: Command/profile selection (CLI layer), testcase inheritance, environment checks
  - _Ref:_ `docs/planning/refactors/2026-02-07_frontend_validation_checklist.md`, `src/TRITON_SWMM_toolkit/validation.py`

---

## Tier 3: CLI & API

Depends on Tier 2 (config refactor). Major user-facing changes.

- [x] **Finalize CLI contract** (Phase 1 of implementation roadmap) — **Complete (80%)**
  - ✅ Define error classes and exit code mapping (CLIValidationError, WorkflowPlanningError)
  - ✅ Finalize argument contract and validation matrix (cli.py with all v1 arguments)
  - ✅ Finalize profile model: production, testcase, case-study (profile_catalog.py + 10 tests)
  - ✅ Implement list actions (--list-testcases, --list-case-studies)
  - ✅ **Complete CLI unit test suite (01-03):** 45 tests passing
    - test_cli_01_validation.py: 22 tests (argument validation)
    - test_cli_02_exit_codes.py: 9 tests (exit code mapping)
    - test_cli_03_actions.py: 14 tests (list actions)
  - ✅ Wire CLI to Analysis orchestration (system/analysis instantiation, preflight validation, workflow submission)
  - ⏸️ Implement profile resolution with 6-tier precedence (deferred - testcase/case-study profiles blocked)
  - ⏸️ Complete CLI integration tests (04-05, blocked by need for real workflow execution)
  - _Ref:_ `docs/planning/refactors/2026-02-08_cli_implementation_design.md`

- [x] **Shared orchestration core** (Phase 2 of implementation roadmap)
  - ✅ High-level analysis.run() API with mode translation (fresh/resume/overwrite)
  - ✅ WorkflowResult structured return type
  - ✅ Workflow status reporting (get_workflow_status() method + --status CLI flag)
  - _Ref:_ `docs/planning/features/2026-02-07_implementation_roadmap.md`, `docs/planning/features/completed/2026-02-09_workflow_status_reporting_plan.md`

- [x] **Implement `triton-swmm run` CLI command** (Phase 3 of implementation roadmap) — **Partial (70%)**
  - ✅ Refactored CLI to use analysis.run() orchestration API (32% code reduction)
  - ✅ Wire arguments to orchestration layer (mode translation: fresh/resume/overwrite)
  - ✅ Discovery actions implemented (--list-testcases, --list-case-studies)
  - ⏸️ Dry-run summary output (current implementation sufficient)
  - ⏸️ Profile resolution for testcase/case-study (blocked by catalog implementation)
  - _Ref:_ `docs/planning/features/2026-02-07_implementation_roadmap.md`

- [ ] **Profile catalog (`tests_and_case_studies.yaml`)** support
  - Implement HPC inheritance and merge semantics
  - _Ref:_ `docs/planning/features/2026-02-07_hpc_inheritance_spec.md`

- [x] **Frontier & UVA HPC validation** (Tier 4) — **Complete**
  - ✅ Fixed `--ntasks-per-gpu=1` → `--gpus-per-task=1` (GPU task expansion bug on Frontier)
  - ✅ Added `--kill-on-bad-exit=1` to srun (prevents Cray PMI hang)
  - ✅ Tmux-based Snakemake orchestration for `batch_job` mode (avoids nested SLURM context)
  - ✅ Automatic Snakemake lock detection and unlock on resume
  - ✅ ARG_MAX PATH trim in `workflow.py` (LD_LIBRARY_PATH trim reverted — broke Frontier)
  - ✅ UVA CPU suite: sa_19 moved to `parallel` partition; time limit raised to 90 min; ≥32 MPI rank configs removed

- [x] **API facade & notebook UX** (Phase 4 of implementation roadmap) — **Complete**
  - ✅ Implemented `Toolkit` high-level API with from_configs() and run() methods
  - ✅ Comprehensive docstrings with usage examples for all public methods
  - ✅ Created example notebook (examples/toolkit_quickstart.ipynb) with 10 usage scenarios
  - ✅ Exported Toolkit in package __init__.py for convenient import
  - ✅ Properties for analysis_dir and n_simulations access
  - ✅ Validated with 45 passing CLI unit tests
  - _Ref:_ `src/TRITON_SWMM_toolkit/toolkit.py`

---

## Tier 4: HPC & Performance (as needed)

These are driven by specific HPC usage needs rather than architectural improvements.

- [ ] **Local GPU workflow support**
  - Add `--resources gpu=<N>` to local Snakemake when `local_gpus_for_workflow` configured
  - Small, self-contained change
  - _Priority:_ Low until GPU testing begins
  - _Ref:_ `docs/planning/features/2026-02-07_local_gpu_workflow_support_plan.md`

- [x] **Frontier end-to-end validation**
  - ✅ Full GPU sensitivity suite (36 sub-analyses) completed — Run 11, Job 4157398 (2026-02-28)
  - ✅ `assert_analysis_workflow_completed_successfully` passes: all `actual_nTasks == n_gpus`
  - ✅ `--gpus-per-task=1` fix confirmed; `--kill-on-bad-exit=1` prevents PMI hang
  - ✅ UVA CPU sensitivity suite also complete (20/20 sub-analyses, all flags present)
  - _Ref:_ `docs/planning/bugs/completed/2026-02-28_gpu-mpi-scaling-machine-file-override.md`

- [ ] **Tool provisioning & reliability** (Phase 5 of implementation roadmap)
  - `--redownload` behavior with provenance logging
  - Resume/from-scratch safeguards
  - _Ref:_ `docs/planning/features/2026-02-07_implementation_roadmap.md`

---

## Known Upstream Issues

- **TRITON output path bug**: TRITON-SWMM executable ignores `output_folder` for SWMM
  artifacts and `log.out`. Workarounds tagged `TODO(TRITON-OUTPUT-PATH-BUG)`.
  See `docs/implementation/triton_output_path_bug.md`.

---

## Documentation Index

### Active Implementation Docs (still relevant)

| Document | Status | Topic |
|----------|--------|-------|
| `implementation/multi_model_integration.md` | ✅ Complete | Architecture and decisions for 3-model concurrent execution |
| `implementation/multi_model_output_processing_plan.md` | ✅ Complete | 9-phase output processing implementation |
| `implementation/log_based_completion_implementation.md` | ✅ Complete | Migration from CFG-existence to log-file completion checks |
| `implementation/1_job_many_srun_tasks_redesign.md` | ✅ Complete | Dynamic SLURM concurrency, SBATCH simplification |
| `implementation/triton_output_path_bug.md` | Active | Upstream bug documentation and workaround locations |

### Active Planning Docs

| Document | Topic |
|----------|-------|
| `planning/refactors/2026-02-07_cruft_cleanup.md` | 7-phase cleanup roadmap (Phases 0-7) with status tracker |
| `planning/features/2026-02-07_implementation_roadmap.md` | 6-phase CLI/API convergence roadmap |
| `planning/features/2026-02-07_hpc_inheritance_spec.md` | `tests_and_case_studies.yaml` schema |
| `planning/features/2026-02-07_local_gpu_workflow_support_plan.md` | Local GPU workflow resource limiting |
| `planning/refactors/2026-02-08_cli_implementation_design.md` | CLI implementation design |
| `planning/refactors/2026-02-07_frontend_validation_checklist.md` | Preflight validation checks |
| `planning/refactors/completed/2026-02-07_config_py_refactor_plan.md` | config.py split into focused modules |

### Archived (completed, historical context only)

See `docs/archived/README.md` for index of 12 archived documents.

