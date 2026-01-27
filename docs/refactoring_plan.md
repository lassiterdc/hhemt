# TRITON-SWMM Toolkit Refactoring Plan

**Date:** January 26, 2026 | **Status:** Phase 5 Complete ✅ - Continuation Phases 6-10 Planned | **Goal:** Decompose `TRITONSWMM_analysis` god class and continue refactoring

---

## Executive Summary

Refactor `TRITONSWMM_analysis` (1,400+ lines, 50+ methods) into focused components. Phases 1-4 completed successfully with full test validation. Phases 5-10 planned for continued refactoring.

**Refactoring Strategy:**
- ✅ **Prioritize codebase reduction** - Aggressively remove duplicate code
- ✅ **Not concerned with backward compatibility** - Internal APIs can change
- ✅ **Preserve core functionality** - Public entry points remain stable
- ✅ **Update tests as needed** - Allow test modifications for improved clarity

**Phase Completion Criteria:**
- ✅ Code changes complete
- ✅ **All 4 smoke tests passing**
- ✅ Public entry points stable (but internal APIs can change)
- ✅ Log file structures unchanged
- ✅ Dead code identified and removed

---

# Part 1: Completed Phases (1-4) - Historical Record

## Phase 1: Extract Resource Management ✅ COMPLETE

**Status:** Code complete, all tests passing

**What Changed:**
- Created `src/TRITON_SWMM_toolkit/resource_management.py` with `ResourceManager` class
- Extracted 3 methods from `analysis.py`:
  - `calculate_effective_max_parallel()` - Calculates effective parallelism based on CPU, GPU, memory, and SLURM constraints
  - `_get_slurm_resource_constraints()` - Extracts SLURM resource constraints from environment variables
  - `_parse_slurm_tasks_per_node()` - Parses SLURM_TASKS_PER_NODE format
- `analysis.py` now delegates to `self._resource_manager`
- **All 4 smoke tests passing:** 22/22 tests passed ✅

---

## Phase 2: Extract Execution Strategies ✅ COMPLETE

**Status:** Code complete, all tests passing

**What Changed:**
- Created `src/TRITON_SWMM_toolkit/execution.py` with execution strategy classes
- Extracted execution logic into 3 strategy classes:
  - `SerialExecutor` - Sequential simulation execution
  - `LocalConcurrentExecutor` - Parallel execution on local machines using ThreadPoolExecutor
  - `SlurmExecutor` - Parallel execution on HPC using SLURM srun tasks
- **Removed ~260 lines of duplicate code** from `analysis.py`
- Updated `tests/test_PC_02_multisim.py` to use new API
- **All 4 smoke tests passing:** 22/22 tests passed ✅

---

## Phase 3: Extract Workflow Generation ✅ COMPLETE

**Status:** Code complete, all tests passing

**What Changed:**
- Created `src/TRITON_SWMM_toolkit/workflow.py` with `SnakemakeWorkflowBuilder` class
- Extracted 6 workflow methods from `analysis.py`:
  - `generate_snakefile_content()`, `generate_snakemake_config()`, `write_snakemake_config()`
  - `run_snakemake_local()`, `run_snakemake_slurm()`, `submit_workflow()`
- `analysis.py` now delegates all workflow operations to `self._workflow_builder`
- **All 4 smoke tests passing:** 22/22 tests passed ✅

---

## Phase 4: Simplify Facade ✅ COMPLETE

**Status:** Code complete, all tests passing

**What Changed:**
- Removed ~100 lines of unused imports and dead code across `analysis.py` and `sensitivity_analysis.py`
- Fixed typo methods: `retreive_*` → `retrieve_*`, `consolidate_analysis_outptus()` → `consolidate_analysis_outputs()`
- Removed duplicate imports and unused utilities
- **All 4 smoke tests passing:** 22/22 tests passed ✅

---

## Phase 5: Simplify Logging Infrastructure ✅ COMPLETE

**Status:** Code complete, all tests passing

**What Changed:**
- Created helper functions `_create_logfield_validator()` and `_create_logfielddict_validator()` to eliminate repetitive validator code
- Created shared `_logfield_serializer()` function to eliminate repetitive serializer code
- Consolidated validators in `TRITONSWMM_scenario_log`:
  - 25 boolean LogField validators → 1 consolidated validator using helper function
  - 1 Path LogField validator using helper function
  - 1 LogFieldDict validator using helper function
- Consolidated validators in `TRITONSWMM_analysis_log`:
  - 11 boolean LogField validators → 1 consolidated validator using helper function
- Consolidated serializers in both log classes to use single shared function
- **Removed ~150 lines of boilerplate code** through consolidation
- Added docstrings to LogField and LogFieldDict classes
- **All 4 smoke tests passing:** 22/22 tests passed ✅

**Key Improvements:**
- Adding new LogField now requires only field definition + adding field name to consolidated validator/serializer
- Reduced from 4 places to 3 places (field definition + validator list + serializer list)
- Helper functions make type coercion explicit and reusable
- No changes to external API or log file structure

---

# Part 2: Continuation Phases (5-10)

## Priority Order Table

| Phase | Name | Impact | Risk | Lines Reduced | Notes |
|-------|------|--------|------|---------------|-------|
| **5** | Simplify Logging Infrastructure | Medium | Low-Medium | ~200 | Reduces cascading changes |
| **6** | Extract SWMM Output Parsing | High | Low | ~500 | Pure function extraction |
| **7** | Remove Delegation Wrappers | Medium | Low | ~100 | Quick wins in analysis.py |
| **8** | Extract Scenario Preparation | High | Medium | ~300 | Decompose scenario.py |
| **9** | Unify Sensitivity Workflow | Medium | Medium | ~200 | Reuse SnakemakeWorkflowBuilder |
| **10** | Fix Naming & Polish | Low | Low | ~50 | Typos, docstrings |

---

## Phase 5: Simplify Logging Infrastructure

**Goal:** Reduce boilerplate in `log.py` and eliminate cascading change requirements

**Current Problems:**
- `LogField` and `LogFieldDict` classes require manual registration with validators and serializers for every field
- Adding a new log field requires changes in 3-4 places:
  1. Field definition in the class
  2. `_load_logfield` validator registration
  3. Type coercion validator registration
  4. `serialize_logfield` serializer registration
- Same fields listed in `@field_validator` decorators multiple times
- Tight coupling - every class using logging must know about `LogField.set()` and `LogField.get()` patterns

**Target Architecture:**
- Reduce boilerplate by using a single decorator or metaclass approach
- Auto-register fields based on type annotations
- Simplify the `LogField` API to be more Pythonic
- Consider using Pydantic's built-in field validators more effectively

**Example Improvement:**
```python
# BEFORE: Manual registration in 4 places
class MyLogModel(BaseModel):
    my_field: LogField = LogField()
    
    @field_validator("my_field", mode="before")
    def _load_my_field(cls, v): ...
    
    @field_validator("my_field")
    def _coerce_my_field(cls, v): ...
    
    @field_serializer("my_field")
    def serialize_my_field(self, v): ...

# AFTER: Single declaration with auto-registration
class MyLogModel(BaseModel):
    my_field: LogField = LogField(auto_register=True)
```

**Risk:** Low-Medium - Logging is used throughout, but changes are internal to `log.py`

**Target:** ~200 lines reduction in boilerplate

---

## Phase 6: Extract SWMM Output Parsing

**Goal:** Move SWMM parsing functions from `process_simulation.py` (~1100 lines) to dedicated module

**Functions to Extract to new `swmm_output_parser.py`:**
- `retrieve_SWMM_outputs_as_datasets()` - Main entry point for SWMM output retrieval
- `return_swmm_outputs()` - Returns SWMM outputs from .out file
- `return_swmm_system_outputs()` - Returns system-level SWMM outputs
- `return_lines_for_section_of_rpt()` - Extracts lines from RPT file sections
- `return_node_time_series_results_from_rpt()` - Parses node timeseries from RPT
- `return_node_time_series_results_from_outfile()` - Parses node timeseries from .out file
- `format_rpt_section_into_dataframe()` - Formats RPT sections as DataFrames
- `return_data_from_rpt()` - Generic RPT data extraction
- All helper functions for RPT parsing (~500 lines total)

**Benefits:**
- Clear separation of concerns: simulation processing vs. output parsing
- Easier to test SWMM parsing logic in isolation
- Reduces `process_simulation.py` from ~1100 to ~600 lines
- Makes SWMM parsing reusable across different contexts

**Risk:** Low - These are pure functions with no class dependencies

**Target:** ~500 lines moved to new module

---

## Phase 7: Remove Delegation Wrappers in analysis.py

**Goal:** Clean up thin wrapper methods that just delegate to extracted components

**Methods to Remove/Simplify:**

From Phase 1 (ResourceManager):
- `_parse_slurm_tasks_per_node()` → Direct access to `_resource_manager.parse_slurm_tasks_per_node()`
- `_get_slurm_resource_constraints()` → Direct access to `_resource_manager.get_slurm_resource_constraints()`

From Phase 3 (SnakemakeWorkflowBuilder):
- `_generate_snakefile_content()` → Direct access to `_workflow_builder.generate_snakefile_content()`
- `_generate_snakemake_config()` → Direct access to `_workflow_builder.generate_snakemake_config()`
- `_write_snakemake_config()` → Direct access to `_workflow_builder.write_snakemake_config()`
- `_run_snakemake_local()` → Direct access to `_workflow_builder.run_snakemake_local()`
- `_run_snakemake_slurm()` → Direct access to `_workflow_builder.run_snakemake_slurm()`

**Approach:**
- Update all call sites to use `self._resource_manager.*` or `self._workflow_builder.*` directly
- Remove the thin wrapper methods from `analysis.py`
- This makes the delegation explicit and reduces indirection

**Risk:** Low - Simple mechanical refactoring

**Target:** ~100 lines reduction

---

## Phase 8: Extract Scenario Preparation Logic

**Goal:** Decompose `TRITONSWMM_scenario` (~700 lines, 25+ methods) into focused components

**Current State:**
- `scenario.py` mixes multiple responsibilities:
  - Weather/boundary condition file generation
  - SWMM model building and modification
  - TRITON config generation
  - Hydrograph file creation

**New Modules:**

### 1. `scenario_inputs.py` - Weather/Boundary Condition File Generation
Extract methods:
- `_write_swmm_rainfall_dat_files()` - Generates rainfall input files
- `_write_swmm_waterlevel_dat_files()` - Generates water level input files
- `_create_external_boundary_condition_files()` - Creates boundary condition files
- `_write_hydrograph_files()` - Generates hydrograph input files

### 2. `swmm_model_builder.py` - SWMM Model Generation
Extract methods:
- `_create_swmm_model_from_template()` - Creates SWMM model from template
- `_update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell()` - Updates SWMM model structure
- `_run_swmm_hydro_model()` - Executes SWMM hydraulic model

**Benefits:**
- Clear separation: input generation vs. model building
- Easier to test each component independently
- Reduces `scenario.py` complexity significantly
- Makes scenario preparation logic more maintainable

**Risk:** Medium - These methods have interdependencies with scenario state

**Target:** ~300 lines reduction in scenario.py

---

## Phase 9: Unify Sensitivity Analysis Workflow Generation

**Goal:** Refactor `TRITONSWMM_sensitivity_analysis` to reuse `SnakemakeWorkflowBuilder`

**Current Problem:**
- `sensitivity_analysis.py` has its own `_generate_master_snakefile_content()` (~200 lines)
- Duplicates workflow generation logic from `workflow.py`
- Inconsistent with the refactored `TRITONSWMM_analysis` approach

**Approach:**
1. Create `SensitivityAnalysisWorkflowBuilder` that extends or composes `SnakemakeWorkflowBuilder`
2. Move `_generate_master_snakefile_content()` logic to the new builder
3. Extract common workflow patterns into shared base class or utility functions
4. Reduce `sensitivity_analysis.py` to orchestration only

**Benefits:**
- Consistent workflow generation across analysis types
- Reduces code duplication
- Easier to maintain and extend workflow generation
- Follows the same pattern as Phases 1-3

**Risk:** Medium - Sensitivity analysis has unique workflow requirements

**Target:** ~200 lines reduction, improved consistency

---

## Phase 10: Fix Naming Inconsistencies and Polish

**Goal:** Address remaining code quality issues

**Tasks:**

1. **Fix Typos:**
   - Search for any remaining `retreive` → `retrieve` typos
   - Fix any other spelling errors in method names

2. **Remove Unused Imports:**
   - Run import analysis to find unused imports
   - Clean up any remaining unused imports across all modules

3. **Add Missing Docstrings:**
   - Ensure all public methods have docstrings
   - Add type hints where missing
   - Document complex internal methods

4. **Ensure Consistent Error Handling:**
   - Review error handling patterns across modules
   - Ensure consistent exception types and messages
   - Add proper error context where needed

5. **Code Quality Checks:**
   - Run linters (flake8, pylint) and address issues
   - Ensure consistent code formatting
   - Remove any remaining commented-out code

**Risk:** Low - These are non-functional improvements

**Target:** ~50 lines reduction, improved code quality

---

## Global Architectural Overview

### System Architecture

```
TRITON-SWMM Toolkit
├── Configuration Layer (config.py, paths.py)
│   ├── Pydantic validation models
│   ├── YAML config loaders
│   └── Path management
│
├── System Setup Layer (system.py)
│   ├── DEM processing
│   ├── Manning's coefficient processing
│   └── TRITON-SWMM compilation
│
├── Analysis Orchestration Layer (analysis.py) ✅ REFACTORED
│   ├── Scenario iteration/management
│   ├── Delegates to ResourceManager ✅
│   ├── Delegates to ExecutionStrategy ✅
│   ├── Delegates to SnakemakeWorkflowBuilder ✅
│   └── Output consolidation coordination
│
├── Resource Management (resource_management.py) ✅ NEW
│   ├── SLURM constraint parsing
│   ├── Parallelism calculation
│   └── Resource allocation
│
├── Execution Strategies (execution.py) ✅ NEW
│   ├── SerialExecutor
│   ├── LocalConcurrentExecutor
│   └── SlurmExecutor
│
├── Workflow Generation (workflow.py) ✅ NEW
│   ├── Snakefile generation
│   ├── Config generation
│   └── Workflow execution
│
├── Scenario Management Layer (scenario.py) ⚠️ NEXT TARGET
│   ├── Individual scenario setup
│   ├── SWMM model generation
│   ├── Boundary condition creation
│   └── TRITON config generation
│
├── Execution Layer (run_simulation.py)
│   ├── Command building
│   ├── Subprocess management
│   └── Checkpoint recovery
│
├── Post-Processing Layer
│   ├── Per-scenario processing (process_simulation.py) ⚠️ NEXT TARGET
│   ├── Ensemble consolidation (processing_analysis.py)
│   └── Sensitivity analysis (sensitivity_analysis.py) ⚠️ NEXT TARGET
│
├── Logging & State Layer (log.py) ⚠️ NEXT TARGET
│   ├── JSON-based state tracking
│   └── Pydantic log models
│
├── Visualization Layer (plot_*.py)
│   ├── System plots
│   ├── Analysis plots
│   └── Utility functions
│
└── CLI Entry Points (*_runner.py, setup_workflow.py, etc.)
    └── Subprocess entry points for Snakemake
```

### Public API Surface (Must Remain Stable)

**Entry Points:**
1. `TRITONSWMM_system(system_config_yaml)` - System setup
2. `TRITONSWMM_system.add_analysis(analysis_config_yaml)` → `TRITONSWMM_analysis`
3. `TRITONSWMM_analysis.submit_workflow()` - Snakemake-based execution
4. `TRITONSWMM_sensitivity_analysis.submit_workflow()` - Sensitivity analysis
5. All CLI modules (`setup_workflow`, `run_single_simulation`, etc.)

**Key Public Methods:**
- `TRITONSWMM_analysis.run_prepare_scenarios_serially()`
- `TRITONSWMM_analysis.run_simulations_concurrently()`
- `TRITONSWMM_analysis.consolidate_TRITON_and_SWMM_simulation_summaries()`
- `TRITONSWMM_analysis.df_status` (property)

---

## Refactoring Philosophy (Preserved from Phases 1-4)

**Core Principles:**
- ✅ **Prefer correctness and clarity over minimal diffs**
- ✅ **Internal refactors may be invasive if they reduce conceptual complexity**
- ✅ **Large rewrites allowed only if hidden behind stable entry points**
- ✅ **Prefer deleting code over preserving unused abstractions**
- ✅ **Never refactor more than one subsystem at a time**
- ✅ **All 4 smoke tests must pass after each phase**

**What We Can Change:**
- Internal method signatures
- Class structure and organization
- Module boundaries
- Implementation details

**What Must Stay Stable:**
- Public API entry points
- Log file structures
- CLI interfaces
- Test behavior (functionality, not implementation)

---

## Smoke Test Requirements

**All phases must pass these tests before completion:**

```bash
conda activate triton_swmm_toolkit
cd /home/***REMOVED***/dev/TRITON-SWMM_toolkit
python -m pytest tests/test_PC_01_singlesim.py tests/test_PC_02_multisim.py tests/test_PC_04_multisim_with_snakemake.py tests/test_PC_05_sensitivity_analysis_with_snakemake.py -v
```

**Test Coverage:**
1. `test_PC_01_singlesim.py` - Single simulation end-to-end (7 tests)
2. `test_PC_02_multisim.py` - Multi-simulation concurrent execution (2 tests)
3. `test_PC_04_multisim_with_snakemake.py` - Snakemake workflow (7 tests)
4. `test_PC_05_sensitivity_analysis_with_snakemake.py` - Sensitivity analysis (6 tests)

**Phase Completion Criteria:**
- ✅ Code changes complete
- ✅ All 22 tests passing
- ✅ No public API changes
- ✅ Log file structures unchanged
- ✅ Dead code identified and removed

---

## Progress Checklist

### Phase 1: Resource Management ✅ COMPLETE
- [x] Create resource_management.py
- [x] Define ResourceManager class
- [x] Move calculate_effective_max_parallel()
- [x] Move _get_slurm_resource_constraints()
- [x] Move _parse_slurm_tasks_per_node()
- [x] Update analysis.py to use ResourceManager
- [x] Run smoke tests (all 4 passing)
- [x] Document architecture overview

### Phase 2: Execution Strategies ✅ COMPLETE
- [x] Create execution.py
- [x] Define ExecutionStrategy protocol
- [x] Implement SerialExecutor
- [x] Implement LocalConcurrentExecutor
- [x] Implement SlurmExecutor
- [x] Update analysis.py strategy selection
- [x] Remove old execution methods (~260 lines deleted)
- [x] Update tests to use new API
- [x] Run smoke tests (22/22 passing)

### Phase 3: Workflow Generation ✅ COMPLETE
- [x] Create workflow.py
- [x] Define SnakemakeWorkflowBuilder class
- [x] Move generate_snakefile_content()
- [x] Move generate_snakemake_config()
- [x] Move write_snakemake_config()
- [x] Move run_snakemake_local()
- [x] Move run_snakemake_slurm()
- [x] Move submit_workflow() orchestration
- [x] Update analysis.py to use builder
- [x] Run smoke tests (22/22 passing)

### Phase 4: Simplify Facade ✅ COMPLETE
- [x] Search codebase for usage of dead code candidates
- [x] Clean up analysis.py (remove unused imports, fix duplicates)
- [x] Remove typo methods (retreive_*, consolidate_analysis_outptus)
- [x] Clean up sensitivity_analysis.py (remove unused imports)
- [x] Fix typo method calls in sensitivity_analysis.py
- [x] Update test files
- [x] Run smoke tests (22/22 passing)

### Phase 5: Simplify Logging Infrastructure ✅ COMPLETE
- [x] Analyze current LogField implementation
- [x] Design auto-registration mechanism
- [x] Implement decorator or metaclass approach
- [x] Update LogField and LogFieldDict classes
- [x] Update all log model classes to use new approach
- [x] Remove manual validator/serializer registrations
- [x] Run smoke tests (22/22 passing)
- [x] Document new logging patterns

### Phase 6: Extract SWMM Output Parsing
- [ ] Create swmm_output_parser.py
- [ ] Move retrieve_SWMM_outputs_as_datasets()
- [ ] Move return_swmm_outputs()
- [ ] Move return_swmm_system_outputs()
- [ ] Move return_lines_for_section_of_rpt()
- [ ] Move return_node_time_series_results_from_rpt()
- [ ] Move return_node_time_series_results_from_outfile()
- [ ] Move format_rpt_section_into_dataframe()
- [ ] Move return_data_from_rpt()
- [ ] Move all RPT parsing helper functions
- [ ] Update process_simulation.py imports
- [ ] Run smoke tests (22/22 passing)

### Phase 7: Remove Delegation Wrappers
- [ ] Identify all delegation wrapper methods in analysis.py
- [ ] Update call sites to use direct component access
- [ ] Remove _parse_slurm_tasks_per_node() wrapper
- [ ] Remove _get_slurm_resource_constraints() wrapper
- [ ] Remove _generate_snakefile_content() wrapper
- [ ] Remove _generate_snakemake_config() wrapper
- [ ] Remove _write_snakemake_config() wrapper
- [ ] Remove _run_snakemake_local() wrapper
- [ ] Remove _run_snakemake_slurm() wrapper
- [ ] Run smoke tests (22/22 passing)

### Phase 8: Extract Scenario Preparation
- [ ] Create scenario_inputs.py
- [ ] Move _write_swmm_rainfall_dat_files()
- [ ] Move _write_swmm_waterlevel_dat_files()
- [ ] Move _create_external_boundary_condition_files()
- [ ] Move _write_hydrograph_files()
- [ ] Create swmm_model_builder.py
- [ ] Move _create_swmm_model_from_template()
- [ ] Move _update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell()
- [ ] Move _run_swmm_hydro_model()
- [ ] Update scenario.py to use new components
- [ ] Run smoke tests (22/22 passing)

### Phase 9: Unify Sensitivity Workflow
- [ ] Create SensitivityAnalysisWorkflowBuilder
- [ ] Extract common workflow patterns to base class
- [ ] Move _generate_master_snakefile_content() logic
- [ ] Update sensitivity_analysis.py to use builder
- [ ] Remove duplicate workflow generation code
- [ ] Run smoke tests (22/22 passing)
- [ ] Verify sensitivity analysis workflow unchanged

### Phase 10: Fix Naming & Polish
- [ ] Search for remaining typos (retreive, etc.)
- [ ] Run import analysis and remove unused imports
- [ ] Add missing docstrings to public methods
- [ ] Add type hints where missing
- [ ] Review and standardize error handling
- [ ] Run linters (flake8, pylint)
- [ ] Remove commented-out code
- [ ] Run smoke tests (22/22 passing)
- [ ] Final documentation update

---

## Validation Strategy

### After Each Phase

1. **Run full smoke test suite**
2. **Check invariants:**
   - Simulation order matches (serial execution)
   - Resource allocation decisions match
   - SLURM command construction matches
   - Log entries identical
   - File paths unchanged

3. **Regression detection:**
   ```bash
   # Capture baseline before phase
   pytest tests/ -v > baseline_tests.log
   
   # After phase
   pytest tests/ -v > phase_N_tests.log
   diff baseline_tests.log phase_N_tests.log
   ```

### Rollback Strategy

- Git commit after each phase passes validation
- Tag stable points: `refactor-phase-5`, `refactor-phase-6`, etc.
- Keep backups of modified files until phase complete

---

## Benefits for AI Agents

### Reduced Context Requirements

**After Phases 1-4:**
- Modifying SLURM logic: Only read `resource_management.py` (~300 lines)
- Modifying execution: Only read `execution.py` (~430 lines)
- Modifying workflows: Only read `workflow.py` (~500 lines)
- Clear component boundaries reduce context by 75%

**After Phases 5-10:**
- Modifying logging: Only read `log.py` (reduced boilerplate)
- Modifying SWMM parsing: Only read `swmm_output_parser.py` (~500 lines)
- Modifying scenario prep: Only read `scenario_inputs.py` or `swmm_model_builder.py`
- Each component has explicit inputs/outputs

### Local Reasoning

- Each component testable in isolation
- Unit tests with mocked dependencies
- Integration tests verify component interactions
- Side effects explicit through method signatures

---

**Last Updated:** January 26, 2026 - Phase 5 Complete ✅ - Continuation Phases 6-10 Planned - All 22 Tests Passing
