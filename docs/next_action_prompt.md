# Phase 13: Type Safety & Final Polish

**Date:** January 27, 2026 | **Status:** Ready for Implementation | **Goal:** Add type hints, run type checker, and finalize refactoring

**Previous Phase:** Phase 12 (Documentation Quality) completed - Comprehensive docstrings added to 5 priority files ✅

---

## Objective

Complete the refactoring with type safety improvements and final code quality polish. This phase adds type hints to public methods, runs static analysis tools, and evaluates project structure for pythonic best practices.

**Expected Impact:**
- Type hints on public methods in 5 priority files
- mypy type checking passing (or documented exceptions)
- pylint high-priority issues resolved
- File/folder structure evaluated for pythonic improvements
- Refactoring marked complete

**Risk Level:** Low-Medium - Type hints are additive, but may reveal latent type issues

**Priority:** High - Completes the refactoring initiative

---

## Priority Files for Type Hints

### 1. `analysis.py` - Main Orchestration Class
**Public Methods to Add Type Hints:**
- `__init__()` - Constructor
- `consolidate_TRITON_and_SWMM_simulation_summaries()` - Output consolidation
- `consolidate_TRITON_simulation_summaries()` - TRITON output consolidation
- `consolidate_SWMM_simulation_summaries()` - SWMM output consolidation
- `print_cfg()` - Configuration display
- `retrieve_prepare_scenario_launchers()` - Scenario preparation launcher creation
- `retrieve_scenario_timeseries_processing_launchers()` - Timeseries processing launcher creation
- `calculate_effective_max_parallel()` - Parallelism calculation
- `run_python_functions_concurrently()` - Concurrent function execution
- `run_prepare_scenarios_serially()` - Serial scenario preparation
- `run_sim()` - Single simulation execution
- `process_sim_timeseries()` - Timeseries processing
- `submit_workflow()` - Workflow submission
- Properties: `scenarios_not_created`, `scenarios_not_run`, `TRITON_summary`, `df_status`

### 2. `sensitivity_analysis.py` - Sensitivity Analysis Orchestration
**Public Methods to Add Type Hints:**
- `__init__()` - Constructor
- `submit_workflow()` - Workflow submission
- Properties: `all_scenarios_created`, `all_sims_run`, `all_TRITON_timeseries_processed`, etc.

### 3. `workflow.py` - Workflow Generation
**Public Methods to Add Type Hints:**
- `__init__()` - Constructor
- `generate_snakefile_content()` - Snakefile generation
- `generate_snakemake_config()` - Config generation
- `write_snakemake_config()` - Config writing
- `run_snakemake_local()` - Local execution
- `run_snakemake_slurm()` - SLURM execution
- `submit_workflow()` - Workflow submission

### 4. `execution.py` - Execution Strategies
**Public Methods to Add Type Hints:**
- `SerialExecutor.execute_simulations()` - Serial execution
- `LocalConcurrentExecutor.execute_simulations()` - Local concurrent execution
- `SlurmExecutor.execute_simulations()` - SLURM execution

### 5. `resource_management.py` - Resource Management
**Public Methods to Add Type Hints:**
- `__init__()` - Constructor
- `calculate_effective_max_parallel()` - Parallelism calculation
- `_get_slurm_resource_constraints()` - SLURM constraint extraction
- `_parse_slurm_tasks_per_node()` - SLURM task parsing

---

## Type Hint Guidelines

### Basic Principles
- Use `from typing import` for complex types (List, Dict, Optional, Union, Callable, etc.)
- Use built-in types where possible (Python 3.9+: `list`, `dict`, `tuple`)
- Use `Optional[Type]` for parameters that can be None
- Use `Union[Type1, Type2]` for parameters that accept multiple types
- Use `Callable[[ArgTypes], ReturnType]` for function parameters
- Use `TYPE_CHECKING` guard for circular import type hints

### Examples

```python
from typing import List, Dict, Optional, Literal, Callable
from pathlib import Path

def method_with_types(
    param1: str,
    param2: int | None = None,
    param3: List[str] = None,
    verbose: bool = False,
) -> Dict[str, any]:
    """Method with type hints."""
    pass

@property
def property_with_type(self) -> pd.DataFrame:
    """Property with return type hint."""
    return self._df_status
```

### Common Types in This Codebase
- `Path` - File paths
- `pd.DataFrame` - Pandas DataFrames
- `xr.Dataset` - xarray Datasets
- `Literal["option1", "option2"]` - String literals
- `List[Callable[[], None]]` - List of callable functions
- `Dict[str, any]` - Dictionary with string keys
- `Optional[int]` or `int | None` - Optional integers

---

## Implementation Steps

### Step 1: Add Type Hints to Priority Files (Focus on Public Methods)
1. Start with `analysis.py` - most critical file
2. Then `sensitivity_analysis.py`
3. Then `workflow.py`
4. Then `execution.py`
5. Finally `resource_management.py`

**Guidelines:**
- Focus on public methods first (methods without leading `_`)
- Add return type hints to all methods
- Add parameter type hints to all parameters
- Use `TYPE_CHECKING` guard for circular imports if needed
- Don't worry about 100% coverage - focus on public API

### Step 2: Run mypy Type Checker
```bash
conda activate triton_swmm_toolkit
cd /home/***REMOVED***/dev/TRITON-SWMM_toolkit

# Run mypy on priority files
mypy src/TRITON_SWMM_toolkit/analysis.py \
     src/TRITON_SWMM_toolkit/sensitivity_analysis.py \
     src/TRITON_SWMM_toolkit/workflow.py \
     src/TRITON_SWMM_toolkit/execution.py \
     src/TRITON_SWMM_toolkit/resource_management.py \
     --ignore-missing-imports \
     --no-strict-optional
```

**Address Critical Issues:**
- Fix type errors that indicate real bugs
- Add `# type: ignore` comments for known false positives
- Document any unresolved type issues in comments

**Success Criteria:** No critical type errors (warnings acceptable)

### Step 3: Run pylint and Address High-Priority Issues
```bash
conda activate triton_swmm_toolkit
cd /home/***REMOVED***/dev/TRITON-SWMM_toolkit

# Run pylint on priority files
pylint src/TRITON_SWMM_toolkit/analysis.py \
       src/TRITON_SWMM_toolkit/sensitivity_analysis.py \
       src/TRITON_SWMM_toolkit/workflow.py \
       src/TRITON_SWMM_toolkit/execution.py \
       src/TRITON_SWMM_toolkit/resource_management.py \
       --disable=C,R,W0212,W0613
```

**Focus on:**
- E-level errors (syntax, undefined names, etc.)
- F-level errors (fatal issues)
- High-impact W-level warnings (unused variables, etc.)

**Ignore:**
- C-level (convention) - not critical
- R-level (refactor) - already addressed in previous phases
- W0212 (protected-access) - acceptable for internal APIs
- W0613 (unused-argument) - acceptable for interface compliance

**Success Criteria:** No E or F level errors, minimal high-impact warnings

### Step 4: Analyze Folder Structure and File Organization

**Current Structure Assessment:**
```
src/TRITON_SWMM_toolkit/
├── Core orchestration: analysis.py, sensitivity_analysis.py
├── Workflow: workflow.py, execution.py, resource_management.py
├── Scenario: scenario.py, scenario_inputs.py
├── SWMM: swmm_*.py (6 files)
├── Processing: process_*.py, processing_analysis.py
├── Plotting: plot_*.py (3 files)
├── Runners: *_runner.py (4 files)
├── Utilities: utils.py, subprocess_utils.py, swmm_utils.py
├── Configuration: config.py, paths.py, constants.py
├── Logging: log.py
├── System: system.py
├── CLI: cli.py, gui.py
└── Legacy: __main__.py, examples.py
```

**Evaluation Criteria:**
1. **File Naming:**
   - Are all files using snake_case? ✅ (Yes)
   - Are names descriptive and clear? ✅ (Mostly yes)
   - Any files that could be renamed for clarity?

2. **Folder Structure:**
   - Should files be grouped into subdirectories?
   - Potential groupings:
     - `core/` - analysis.py, sensitivity_analysis.py
     - `workflow/` - workflow.py, execution.py, resource_management.py
     - `scenario/` - scenario.py, scenario_inputs.py
     - `swmm/` - swmm_*.py files
     - `processing/` - process_*.py, processing_analysis.py
     - `plotting/` - plot_*.py files
     - `runners/` - *_runner.py files
     - `utils/` - utils.py, subprocess_utils.py, swmm_utils.py
     - `config/` - config.py, paths.py, constants.py

3. **Module Organization:**
   - Are imports clean and organized?
   - Any circular dependencies?
   - Any modules that are too large?

**Deliverable:** Document findings and recommendations (no implementation required unless critical)

### Step 5: Final Documentation Update
Update `docs/refactoring_plan.md`:
- Mark Phase 13 as complete
- Add summary of type hints added
- Document any unresolved type issues
- Mark refactoring initiative as COMPLETE
- Update "Last Updated" line

### Step 6: Validation
Run all smoke tests:
```bash
conda activate triton_swmm_toolkit
cd /home/***REMOVED***/dev/TRITON-SWMM_toolkit
python -m pytest tests/test_PC_01_singlesim.py tests/test_PC_02_multisim.py tests/test_PC_04_multisim_with_snakemake.py tests/test_PC_05_sensitivity_analysis_with_snakemake.py -v
```

**Success Criteria:** All 22 tests passing

---

## Key Constraints

✅ **DO:**
- Add type hints to public methods in 5 priority files
- Run mypy and address critical type errors
- Run pylint and address E/F level errors
- Document folder structure findings
- Update refactoring_plan.md to mark complete

❌ **DON'T:**
- Change functionality or behavior
- Modify public API signatures (except adding type hints)
- Touch log file structures
- Break any tests
- Spend excessive time on perfect type coverage (focus on public API)
- Reorganize folder structure without discussion (document recommendations only)

---

## Expected Results

**Before:**
- No type hints on public methods
- mypy not run
- pylint not run
- Folder structure not evaluated

**After:**
- Type hints on all public methods in 5 priority files
- mypy passing (or documented exceptions)
- pylint E/F errors resolved
- Folder structure evaluated and documented
- Refactoring marked COMPLETE
- All 22 tests passing

---

## Validation Checklist

- [ ] Added type hints to all public methods in analysis.py
- [ ] Added type hints to all public methods in sensitivity_analysis.py
- [ ] Added type hints to all public methods in workflow.py
- [ ] Added type hints to all public methods in execution.py
- [ ] Added type hints to all public methods in resource_management.py
- [ ] Ran mypy and addressed critical type errors
- [ ] Ran pylint and addressed E/F level errors
- [ ] Analyzed folder structure and documented findings
- [ ] Updated refactoring_plan.md to mark Phase 13 complete
- [ ] All 22 smoke tests passing (test_PC_01, test_PC_02, test_PC_04, test_PC_05)
- [ ] No changes to public API signatures (except type hints)
- [ ] No changes to log file structures
- [ ] Refactoring initiative marked COMPLETE

---

## Notes

- Phase 13 completes the refactoring initiative
- Type hints are additive and should not break existing code
- Focus on public API - internal methods can be typed later
- Document any unresolved type issues for future work
- Folder structure recommendations are for discussion, not immediate implementation

---

**Last Updated:** January 27, 2026 - Phase 13 Implementation Guide
