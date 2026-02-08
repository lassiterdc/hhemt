# Development Priorities

**Last Updated:** 2026-02-07
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

---

## Tier 1: Cruft Cleanup & Code Quality

Incremental improvements that reduce maintenance burden. Can be done independently.

- [ ] **Remove legacy/obsolete runtime paths** (`cruft_cleanup_plan.md` Phase 2)
  - Remove `_obsolete_*` methods and dead launch paths in `run_simulation.py`
  - Collapse fallback branches for retired structures
  - Remove deprecated `SimLog` / `SimEntry` classes from `log.py`
  - Delete commented-out simlog code
  - _Ref:_ `docs/planning/cruft_cleanup_plan.md`

- [ ] **Logging & error normalization** (`cruft_cleanup_plan.md` Phase 4)
  - Replace ad-hoc `print(...)` in library paths with structured logging
  - Standardize exception boundaries and message format
  - Remove silent returns for failure states
  - _Ref:_ `docs/planning/cruft_cleanup_plan.md`

- [ ] **Workaround containment** (`cruft_cleanup_plan.md` Phase 5)
  - Centralize `TODO(TRITON-OUTPUT-PATH-BUG)` logic behind minimal interfaces
  - Currently in 4 source files — see `docs/implementation/triton_output_path_bug.md`
  - _Blocked by:_ Upstream TRITON-SWMM fix for `output_folder` directive

- [ ] **Test suite cleanup** (`cruft_cleanup_plan.md` Phase 6)
  - Parametrize repeated platform test patterns
  - Consolidate fixture factories
  - Reduce unconditional diagnostic prints
  - Standardize assertions around completion semantics
  - _Ref:_ `docs/planning/cruft_cleanup_plan.md`

---

## Tier 2: Config Refactor

Foundation for CLI/API work. Should be done before Tier 3.

- [ ] **Split `config.py` into focused modules** (`config_py_refactor_plan.md` Phase 1)
  - Create `config/` package: base, system, analysis, validation, loaders, display
  - Update all import sites immediately (no compatibility shims)
  - _Ref:_ `docs/planning/refactors/config_py_refactor_plan.md`

- [x] **Strict validation + validator cleanup** (`config_py_refactor_plan.md` Phase 2)
  - `extra="forbid"` enforced on `cfgBaseModel`
  - Dynamic `toggle_tests` registry replaced with explicit `@model_validator` rules
  - Dead legacy fields and commented-out code removed from `config.py` and `analysis.py`

- [ ] **Front-end validation checklist** (implement preflight checks)
  - Command/profile selection, config integrity, toggle dependencies, run-mode consistency
  - HPC sanity checks, data cross-consistency, environment checks
  - _Ref:_ `docs/planning/refactors/frontend_validation_checklist.md`

---

## Tier 3: CLI & API

Depends on Tier 2 (config refactor). Major user-facing changes.

- [ ] **Finalize CLI contract** (Phase 1 of implementation roadmap)
  - Finalize argument contract and validation matrix
  - Define error classes and exit code mapping
  - Finalize profile model: production, testcase, case-study
  - _Ref:_ `docs/planning/cli_command_spec.md`, `docs/planning/cli_vision.md`

- [ ] **Shared orchestration core** (Phase 2 of implementation roadmap)
  - Consolidate run/setup/processing flow behind one orchestration layer
  - Make CLI a thin adapter, API calls same path
  - _Ref:_ `docs/planning/implementation_roadmap.md`

- [ ] **Implement `triton-swmm run` CLI command** (Phase 3 of implementation roadmap)
  - Wire arguments to Snakemake targets/options
  - Add dry-run and argument-resolution summary output
  - Add testcase/case-study discovery actions
  - _Ref:_ `docs/planning/cli_command_spec.md`

- [ ] **Profile catalog (`tests_and_case_studies.yaml`)** support
  - Implement HPC inheritance and merge semantics
  - _Ref:_ `docs/planning/hpc_inheritance_spec.md`

- [ ] **API facade & notebook UX** (Phase 4 of implementation roadmap)
  - Implement `Toolkit`-style high-level API
  - Return structured result objects
  - _Ref:_ `docs/planning/api_vision.md`

---

## Tier 4: HPC & Performance (as needed)

These are driven by specific HPC usage needs rather than architectural improvements.

- [ ] **Local GPU workflow support**
  - Add `--resources gpu=<N>` to local Snakemake when `local_gpus_for_workflow` configured
  - Small, self-contained change
  - _Priority:_ Low until GPU testing begins
  - _Ref:_ `docs/planning/local_gpu_workflow_support_plan.md`

- [ ] **Frontier end-to-end validation**
  - Run full test suites on Frontier with multi-model + GPU
  - Validate 1-job-many-srun-tasks mode on real cluster
  - _Priority:_ Next cluster access window

- [ ] **Tool provisioning & reliability** (Phase 5 of implementation roadmap)
  - `--redownload` behavior with provenance logging
  - Resume/from-scratch safeguards
  - _Ref:_ `docs/planning/implementation_roadmap.md`

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
| `planning/cruft_cleanup_plan.md` | 7-phase cleanup roadmap (Phases 0-7) |
| `planning/implementation_roadmap.md` | 6-phase CLI/API convergence roadmap |
| `planning/cli_vision.md` | Snakemake-first CLI design principles |
| `planning/cli_command_spec.md` | Formal `triton-swmm run` command contract |
| `planning/api_vision.md` | Python API layers and parity requirements |
| `planning/hpc_inheritance_spec.md` | `tests_and_case_studies.yaml` schema |
| `planning/local_gpu_workflow_support_plan.md` | Local GPU workflow resource limiting |
| `planning/refactors/config_py_refactor_plan.md` | config.py split into focused modules |
| `planning/refactors/frontend_validation_checklist.md` | Preflight validation checks |

### Archived (completed, historical context only)

See `docs/archived/README.md` for index of 12 archived documents.
