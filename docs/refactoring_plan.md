# TRITON-SWMM Toolkit Refactoring Plan

**Date:** January 26, 2026 | **Status:** Phase 4 Complete ✅ - All Phases Finished | **Goal:** Decompose `TRITONSWMM_analysis` god class

---

## Executive Summary

Refactor `TRITONSWMM_analysis` (1,400+ lines, 50+ methods) into focused components. Conducted in 4 phases with full test validation after each phase.

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
├── Analysis Orchestration Layer (analysis.py) ⚠️ GOD CLASS
│   ├── Scenario iteration/management
│   ├── Execution strategies (serial, concurrent, SLURM)
│   ├── Resource management (SLURM constraints)
│   ├── Snakemake workflow generation
│   └── Output consolidation coordination
│
├── Scenario Management Layer (scenario.py)
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
│   ├── Per-scenario processing (process_simulation.py)
│   ├── Ensemble consolidation (processing_analysis.py)
│   └── Sensitivity analysis (sensitivity_analysis.py)
│
├── Logging & State Layer (log.py)
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

### Data Flow

```
User Config (YAML)
    ↓
System Setup (DEM, Manning's, Compilation)
    ↓
Analysis Creation (scenario list, paths)
    ↓
Scenario Preparation (SWMM models, boundary conditions)
    ↓
Simulation Execution (TRITON-SWMM run)
    ↓
Output Processing (timeseries extraction)
    ↓
Consolidation (ensemble-level summaries)
    ↓
Visualization & Analysis
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

### High-Risk Coupling Points

1. **Circular Type Dependencies:**
   - `analysis.py` ↔ `scenario.py` ↔ `run_simulation.py` ↔ `process_simulation.py`
   - All use `TYPE_CHECKING` imports to break cycles

2. **Config Sprawl:**
   - `analysis_config` has 40+ fields mixing simulation params, HPC config, output preferences

3. **Subprocess Python Executable:**
   - Hardcoded `"python"` vs `sys.executable` causes environment issues
   - Fixed by using `cfg_analysis.python_path` or defaulting to `"python"`

---

## Problem Statement

### God Class: `TRITONSWMM_analysis`

**Current State:** 1,400+ lines, 50+ methods mixing 6+ responsibilities:
1. Scenario iteration/management
2. Serial execution orchestration
3. Concurrent execution (ThreadPool/ProcessPool)
4. SLURM resource management and constraint parsing
5. Snakemake workflow generation
6. Output consolidation coordination

**Impact:**
- Every change risks breaking unrelated functionality
- Impossible to test individual concerns in isolation
- High cognitive load for modifications
- Difficult for AI agents to reason about locally

---

## Target Architecture

```
TRITONSWMM_analysis (Thin Facade)
├── ResourceManager ✅ (Phase 1 - COMPLETE)
│   ├── calculate_effective_max_parallel()
│   ├── _get_slurm_resource_constraints()
│   └── _parse_slurm_tasks_per_node()
│
├── ExecutionStrategy (Phase 2)
│   ├── SerialExecutor
│   ├── LocalConcurrentExecutor
│   └── SlurmExecutor
│
├── SnakemakeWorkflowBuilder (Phase 3)
│   ├── generate_snakefile_content()
│   └── generate_snakemake_config()
│
└── OutputConsolidator (already exists as TRITONSWMM_analysis_post_processing)
```

---

## Phase 1: Extract Resource Management ✅ COMPLETE

**Status:** Code complete, all tests passing

**What Changed:**
- Created `src/TRITON_SWMM_toolkit/resource_management.py` with `ResourceManager` class
- Extracted 3 methods from `analysis.py`:
  - `calculate_effective_max_parallel()` - Calculates effective parallelism based on CPU, GPU, memory, and SLURM constraints
  - `_get_slurm_resource_constraints()` - Extracts SLURM resource constraints from environment variables
  - `_parse_slurm_tasks_per_node()` - Parses SLURM_TASKS_PER_NODE format
- `analysis.py` now delegates to `self._resource_manager`
- **All 4 smoke tests passing:**
  - `test_PC_01_singlesim.py` - 7/7 passed ✅
  - `test_PC_02_multisim.py` - 2/2 passed ✅
  - `test_PC_04_multisim_with_snakemake.py` - 7/7 passed ✅
  - `test_PC_05_sensitivity_analysis_with_snakemake.py` - 6/6 passed ✅

**Changes Summary:**
```python
# analysis.py - BEFORE
class TRITONSWMM_analysis:
    def calculate_effective_max_parallel(...):
        # 100+ lines of SLURM logic
    
    def _get_slurm_resource_constraints(...):
        # 80+ lines of env var parsing
    
    def _parse_slurm_tasks_per_node(...):
        # 20+ lines of string parsing

# analysis.py - AFTER
from TRITON_SWMM_toolkit.resource_management import ResourceManager

class TRITONSWMM_analysis:
    def __init__(...):
        self._resource_manager = ResourceManager(self.cfg_analysis, self.in_slurm)
    
    def calculate_effective_max_parallel(...):
        return self._resource_manager.calculate_effective_max_parallel(...)
    
    # Thin wrappers that delegate to ResourceManager
```

---

## Phase 2: Extract Execution Strategies ✅ COMPLETE

**Status:** Code complete, all tests passing

**Goal:** Separate "how to run" from "what to run"

**Created:** `src/TRITON_SWMM_toolkit/execution.py` (~430 lines)

**What Changed:**
- Created `src/TRITON_SWMM_toolkit/execution.py` with execution strategy classes
- Extracted execution logic into 3 strategy classes:
  - `SerialExecutor` - Sequential simulation execution
  - `LocalConcurrentExecutor` - Parallel execution on local machines using ThreadPoolExecutor
  - `SlurmExecutor` - Parallel execution on HPC using SLURM srun tasks
- Added `_select_execution_strategy()` method to `analysis.py` to choose appropriate executor
- Updated `run_simulations_concurrently()` to delegate to `self._execution_strategy`
- **Removed ~260 lines of duplicate code** from `analysis.py`:
  - Deleted `run_simulations_concurrently_on_local_machine()` (~80 lines)
  - Deleted `run_simulations_concurrently_on_SLURM_HPC_using_many_srun_tasks()` (~180 lines)
- Updated `tests/test_PC_02_multisim.py` to use new API
- **All 4 smoke tests passing:**
  - `test_PC_01_singlesim.py` - 7/7 passed ✅
  - `test_PC_02_multisim.py` - 2/2 passed ✅
  - `test_PC_04_multisim_with_snakemake.py` - 7/7 passed ✅
  - `test_PC_05_sensitivity_analysis_with_snakemake.py` - 6/6 passed ✅

### Components Extracted

#### 1. ExecutionStrategy Protocol
```python
from typing import Protocol, List, Tuple, Optional

class ExecutionStrategy(Protocol):
    """Protocol for simulation execution strategies."""
    
    def execute_simulations(
        self,
        launch_functions: List[Tuple],
        max_concurrent: Optional[int],
        verbose: bool
    ) -> List[str]:
        """Execute simulations and return completion statuses."""
        ...
```

#### 2. SerialExecutor
**Extract from:** `run_sims_in_sequence()`

```python
class SerialExecutor:
    """Sequential simulation execution."""
    
    def __init__(self, analysis: "TRITONSWMM_analysis"):
        self.analysis = analysis
    
    def execute_simulations(
        self,
        launch_functions: List[Tuple],
        max_concurrent: Optional[int] = None,
        verbose: bool = True
    ) -> List[str]:
        """Execute simulations sequentially."""
        results = []
        for launcher, finalize_sim in launch_functions:
            proc, start_time, sim_logfile, lf = launcher()
            finalize_sim(proc, start_time, sim_logfile, lf)
            results.append("completed")
        self.analysis._update_log()
        return results
```

#### 3. LocalConcurrentExecutor
**Extract from:** `run_simulations_concurrently_on_local_machine()`

#### 4. SlurmExecutor
**Extract from:** `run_simulations_concurrently_on_SLURM_HPC_using_many_srun_tasks()`

### Changes to `analysis.py`

```python
from TRITON_SWMM_toolkit.execution import (
    SerialExecutor,
    LocalConcurrentExecutor,
    SlurmExecutor
)

class TRITONSWMM_analysis:
    def __init__(self, ...):
        self._resource_manager = ResourceManager(...)
        self._execution_strategy = self._select_execution_strategy()
    
    def _select_execution_strategy(self):
        method = self.cfg_analysis.multi_sim_run_method
        if method == "1_job_many_srun_tasks":
            return SlurmExecutor(self)
        elif method == "local":
            return LocalConcurrentExecutor(self)
        else:
            return SerialExecutor(self)
    
    def run_simulations_concurrently(self, launch_functions, max_concurrent=None, verbose=True):
        """Delegate to execution strategy."""
        return self._execution_strategy.execute_simulations(
            launch_functions, max_concurrent, verbose
        )
```

**Risk:** Medium - Must preserve exact execution order and logging behavior

---

## Phase 3: Extract Workflow Generation ✅ COMPLETE

**Status:** Code complete, all tests passing

**Goal:** Make Snakemake workflow generation testable

**Created:** `src/TRITON_SWMM_toolkit/workflow.py`

**What Changed:**
- Created `src/TRITON_SWMM_toolkit/workflow.py` with `SnakemakeWorkflowBuilder` class
- Extracted 5 workflow methods from `analysis.py`:
  - `generate_snakefile_content()` - Generates Snakefile content for workflow
  - `generate_snakemake_config()` - Generates dynamic Snakemake configuration
  - `write_snakemake_config()` - Writes config to analysis directory
  - `run_snakemake_local()` - Executes workflow on local machine
  - `run_snakemake_slurm()` - Executes workflow on SLURM HPC
  - `submit_workflow()` - Main entry point that orchestrates workflow submission
- `analysis.py` now delegates all workflow operations to `self._workflow_builder`
- **All 4 smoke tests passing:**
  - `test_PC_01_singlesim.py` - 7/7 passed ✅
  - `test_PC_02_multisim.py` - 2/2 passed ✅
  - `test_PC_04_multisim_with_snakemake.py` - 7/7 passed ✅
  - `test_PC_05_sensitivity_analysis_with_snakemake.py` - 6/6 passed ✅

**Changes Summary:**
```python
# analysis.py - BEFORE
class TRITONSWMM_analysis:
    def _generate_snakefile_content(...):
        # 150+ lines of Snakefile generation
    
    def submit_workflow(...):
        # 100+ lines of workflow orchestration
    
    def _generate_snakemake_config(...):
        # 70+ lines of config generation
    
    def _write_snakemake_config(...):
        # 20+ lines of file writing
    
    def _run_snakemake_local(...):
        # 70+ lines of local execution
    
    def _run_snakemake_slurm(...):
        # 150+ lines of SLURM execution

# analysis.py - AFTER
from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

class TRITONSWMM_analysis:
    def __init__(...):
        self._workflow_builder = SnakemakeWorkflowBuilder(self)
    
    def _generate_snakefile_content(...):
        return self._workflow_builder.generate_snakefile_content(...)
    
    def submit_workflow(...):
        return self._workflow_builder.submit_workflow(...)
    
    # All other workflow methods delegate to builder
```

---

## Phase 4: Simplify Facade ✅ COMPLETE

**Status:** Code complete, all tests passing

**Goal:** Clean up `analysis.py` and `sensitivity_analysis.py` to reduce cruft and improve maintainability

**Actual Impact:** Removed ~100 lines of unused imports and dead code across both files

### Specific Tasks

#### 1. Remove Unused Imports from `analysis.py`
The following imports are no longer used after Phases 1-3:
- `subprocess` - Moved to workflow.py
- `shutil` - Not used anywhere
- `yaml` - Moved to workflow.py
- `ProcessPoolExecutor` - Not used (imported twice)
- `traceback` - Not used
- `redirect_stdout` - Not used
- `redirect_stderr` - Not used
- `threading` - Not used
- `psutil` - Moved to workflow.py
- `warnings` - Not used
- `Mode` from constants - Not used

#### 2. Fix Duplicate Imports in `analysis.py`
The following are imported multiple times:
- `as_completed` from `concurrent.futures` (imported twice)
- `Optional` from `typing` (imported twice)
- `Path` (imported multiple times - once from pathlib, then shadowed)

#### 3. Remove Dead Code Methods from `analysis.py`
The following methods should be removed:
- `retreive_scenario_timeseries_processing_launchers()` (line ~582) - Typo duplicate of `retrieve_scenario_timeseries_processing_launchers()`
- `consolidate_analysis_outptus()` (line ~618) - Typo in function name (should be `consolidate_analysis_outputs`)
- `minutes_to_hhmmss()` (line ~1486) - Module-level utility, verify usage before removing

#### 4. Clean Up `sensitivity_analysis.py` Imports
The following imports are unused:
- `subprocess` - Not used
- `shutil` - Not used
- `Mode` from constants - Not used
- `pprint` - Not used
- `json` - Not used

#### 5. Fix Typo Method Calls in `sensitivity_analysis.py`
The following calls reference typo methods that will be removed:
- Line ~489: `sub_analysis.retreive_scenario_timeseries_processing_launchers()` → `retrieve_scenario_timeseries_processing_launchers()`
- Line ~556: `sub_analysis.consolidate_analysis_outptus()` → Must be updated based on what we do with the typo method

#### 6. Optional Cleanup
- Consider removing or documenting commented-out code in both files
- Verify all remaining methods have proper docstrings
- Check for any other typos or inconsistencies
- (Future) Consider extracting `_generate_master_snakefile_content()` (~200 lines) to reuse `SnakemakeWorkflowBuilder`

### Implementation Steps
1. Search codebase for usage of `minutes_to_hhmmss()` and `consolidate_analysis_outptus()`
2. Remove unused imports from both `analysis.py` and `sensitivity_analysis.py`
3. Consolidate duplicate imports in `analysis.py`
4. Remove dead code methods from `analysis.py`
5. Fix typo method calls in `sensitivity_analysis.py`
6. Run smoke tests to verify no breakage (especially test_PC_05_sensitivity_analysis_with_snakemake.py)
7. Update this document to mark Phase 4 complete

---

## Dead Code Candidates

### Confirmed Issues

1. **Typo/Duplicate Methods:**
   - `retreive_scenario_timeseries_processing_launchers()` (line 582) - typo, duplicate of `retrieve_scenario_timeseries_processing_launchers()` (line 308)
   - `consolidate_analysis_outptus()` (line 618) - typo in function name

2. **Potentially Unused:**
   - `minutes_to_hhmmss()` at module level (line 1486) - utility function that may not be called

### Removal Strategy

**Phase 1.5 (Optional Cleanup):**
- Remove duplicate `retreive_scenario_timeseries_processing_launchers()`
- Fix or remove `consolidate_analysis_outptus()` typo
- Search codebase for `minutes_to_hhmmss` usage

**Future Phases:**
- Conduct comprehensive dead code analysis using tools like `vulture` or `coverage`
- Remove unused imports
- Consolidate duplicate logic

---

## Smoke Test Requirements

**All phases must pass these tests before completion:**

1. `pytest tests/test_PC_01_singlesim.py -v` - Single simulation end-to-end
2. `pytest tests/test_PC_02_multisim.py -v` - Multi-simulation concurrent execution
3. `pytest tests/test_PC_04_multisim_with_snakemake.py -v` - Snakemake workflow
4. `pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py -v` - Sensitivity analysis

**Test Command:**
```bash
conda activate triton_swmm_toolkit
cd /home/***REMOVED***/dev/TRITON-SWMM_toolkit
python -m pytest tests/test_PC_01_singlesim.py tests/test_PC_02_multisim.py tests/test_PC_04_multisim_with_snakemake.py tests/test_PC_05_sensitivity_analysis_with_snakemake.py -v
```

**Phase Completion Criteria:**
- ✅ Code changes complete
- ✅ All 4 smoke tests passing
- ✅ No public API changes
- ✅ Log file structures unchanged
- ✅ Dead code identified

---

## Test Improvement Guidelines

Tests may be modified to:
1. **Strengthen coverage** - Add assertions for edge cases
2. **Improve clarity** - Better test names, more explicit assertions
3. **Accommodate improvements** - Update when refactoring improves behavior
4. **Fix brittleness** - Replace hardcoded paths with dynamic values

**Restrictions:**
- Must not reduce coverage
- Must not bypass validation of core functionality
- Changes must be documented in commit messages

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
- [x] Identify dead code candidates

### Phase 2: Execution Strategies ✅ COMPLETE
- [x] Create execution.py
- [x] Define ExecutionStrategy protocol
- [x] Implement SerialExecutor
- [x] Implement LocalConcurrentExecutor
- [x] Implement SlurmExecutor
- [x] Update analysis.py strategy selection
- [x] Remove old execution methods (~260 lines deleted)
- [x] Update tests to use new API (test_PC_02_multisim.py)
- [x] Run smoke tests (22/22 passing)
- [x] Verify execution behavior unchanged

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
- [x] Verify Snakefiles unchanged

### Phase 4: Simplify Facade ✅ COMPLETE
- [x] Search codebase for usage of dead code candidates
  - [x] Check for `minutes_to_hhmmss` usage (not used, safe to remove)
  - [x] Check for `consolidate_analysis_outptus` usage (typo method used in sensitivity_analysis.py)
- [x] Clean up analysis.py
  - [x] Remove unused imports (`subprocess`, `shutil`, `yaml`, `ProcessPoolExecutor`, `traceback`, `redirect_stdout`, `redirect_stderr`, `threading`, `psutil`, `warnings`, `Mode`)
  - [x] Fix duplicate imports (`as_completed`, `Optional`, `Path`)
  - [x] Remove `retreive_scenario_timeseries_processing_launchers()` typo method
  - [x] Rename `consolidate_analysis_outptus()` to `consolidate_analysis_outputs()`
  - [x] Remove `minutes_to_hhmmss()` function
- [x] Clean up sensitivity_analysis.py
  - [x] Remove unused imports (`subprocess`, `shutil`, `Mode`, `pprint`, `json`)
  - [x] Fix typo method call: `retreive_scenario_timeseries_processing_launchers()` → `retrieve_scenario_timeseries_processing_launchers()`
  - [x] Fix typo method call: `consolidate_analysis_outptus()` → `consolidate_analysis_outputs()`
- [x] Update test files (test_PC_02_multisim.py)
- [x] Run smoke tests (22/22 passing ✅)
- [x] Update refactoring plan to mark Phase 4 complete

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
- Tag stable points: `refactor-phase-1`, `refactor-phase-2`, etc.
- Keep `analysis_v1.py.backup` until Phase 4 complete

---

## Benefits for AI Agents

### Reduced Context Requirements

**Before:** 1,400 lines of context required for any change

**After Phase 1:** 
- Modifying SLURM logic: Only read `resource_management.py` (~300 lines)
- Clear component boundaries reduce context by 75%

**After Phase 2:**
- Adding execution backend: Only implement `ExecutionStrategy` protocol
- Modifying local execution: Only touch `LocalConcurrentExecutor`

**After All Phases:**
- Each component has explicit inputs/outputs
- Changes contained within component boundaries
- Side effects explicit through method signatures

### Local Reasoning

- Each component testable in isolation
- Unit tests with mocked dependencies
- Integration tests verify component interactions

---

**Last Updated:** January 26, 2026 - Phase 4 Complete, All Phases Finished ✅ - All 22 Tests Passing
