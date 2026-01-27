# Phase 9: Unify Sensitivity Analysis Workflow Generation

**Date:** January 27, 2026 | **Status:** Ready for Implementation | **Goal:** Refactor sensitivity analysis to reuse `SnakemakeWorkflowBuilder` and eliminate duplicate workflow generation code

**Previous Phase:** Phase 8 (Extract Scenario Preparation Logic) completed - 22/22 tests passing ✅

---

## Objective

Refactor `TRITONSWMM_sensitivity_analysis` to reuse the `SnakemakeWorkflowBuilder` class, eliminating ~200 lines of duplicate workflow generation code and creating consistency across analysis types.

**Expected Impact:**
- Reduces `sensitivity_analysis.py` by ~200 lines
- Eliminates duplicate workflow generation logic
- Creates consistent workflow patterns across regular and sensitivity analyses
- Makes workflow generation easier to maintain and extend

**Risk Level:** Medium - Sensitivity analysis has unique workflow requirements that must be preserved

---

## Current Problem

**Duplicate Workflow Generation:**
- `sensitivity_analysis.py` has its own `_generate_master_snakefile_content()` method (~200 lines)
- This duplicates workflow generation logic from `workflow.py`
- Inconsistent with the refactored `TRITONSWMM_analysis` approach (Phases 1-3)
- Makes it harder to maintain and extend workflow generation

**Code Smell:**
```python
# sensitivity_analysis.py
class TRITONSWMM_sensitivity_analysis:
    def _generate_master_snakefile_content(self):
        # ~200 lines of Snakefile generation
        # Duplicates patterns from SnakemakeWorkflowBuilder
        ...
```

---

## Target Architecture

### Option 1: Composition (Recommended)

Create a `SensitivityAnalysisWorkflowBuilder` that **composes** `SnakemakeWorkflowBuilder`:

```python
class SensitivityAnalysisWorkflowBuilder:
    """
    Builds Snakemake workflows for sensitivity analysis.
    
    Composes SnakemakeWorkflowBuilder to reuse common workflow patterns
    while adding sensitivity-specific workflow generation.
    """
    
    def __init__(self, sensitivity_analysis: "TRITONSWMM_sensitivity_analysis"):
        self.sensitivity_analysis = sensitivity_analysis
        # Reuse base workflow builder for common patterns
        self._base_builder = SnakemakeWorkflowBuilder(sensitivity_analysis)
        
    def generate_master_snakefile_content(self) -> str:
        """Generate master Snakefile for sensitivity analysis."""
        # Use base builder for common patterns
        # Add sensitivity-specific rules
        ...
```

### Option 2: Inheritance

Create a `SensitivityAnalysisWorkflowBuilder` that **extends** `SnakemakeWorkflowBuilder`:

```python
class SensitivityAnalysisWorkflowBuilder(SnakemakeWorkflowBuilder):
    """
    Extends SnakemakeWorkflowBuilder for sensitivity analysis workflows.
    """
    
    def generate_master_snakefile_content(self) -> str:
        """Generate master Snakefile for sensitivity analysis."""
        # Override to add sensitivity-specific workflow
        ...
```

**Recommendation:** Use **Option 1 (Composition)** because:
- Sensitivity analysis workflow is fundamentally different (master + sub-workflows)
- Composition is more flexible and explicit
- Avoids inheritance complexity

---

## Implementation Steps

### Step 1: Analyze Current Workflow Generation

1. Read `sensitivity_analysis.py` to understand `_generate_master_snakefile_content()`
2. Identify common patterns with `SnakemakeWorkflowBuilder`
3. Identify sensitivity-specific requirements
4. Map out data flow and dependencies

### Step 2: Extract Common Patterns

1. Identify reusable workflow generation patterns in `SnakemakeWorkflowBuilder`
2. Consider extracting common patterns to utility functions if needed
3. Document which parts can be reused vs. need customization

### Step 3: Create `SensitivityAnalysisWorkflowBuilder`

1. Create new class in `workflow.py` (or new `sensitivity_workflow.py` if large)
2. Implement composition pattern with `SnakemakeWorkflowBuilder`
3. Move `_generate_master_snakefile_content()` logic to new builder
4. Add comprehensive docstrings

### Step 4: Update `sensitivity_analysis.py`

1. Import `SensitivityAnalysisWorkflowBuilder`
2. Initialize in `__init__`:
   ```python
   self._workflow_builder = SensitivityAnalysisWorkflowBuilder(self)
   ```
3. Replace method calls with delegation:
   - `self._generate_master_snakefile_content()` → `self._workflow_builder.generate_master_snakefile_content()`
4. Remove the old `_generate_master_snakefile_content()` method

### Step 5: Validate

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
- Reuse `SnakemakeWorkflowBuilder` patterns where possible
- Maintain sensitivity analysis workflow behavior exactly
- Add type hints and comprehensive docstrings
- Test sensitivity analysis workflow thoroughly

❌ **DON'T:**
- Change sensitivity analysis workflow behavior
- Modify public API methods
- Touch log file structures
- Break any sensitivity analysis tests

---

## Expected Results

**Before:**
```python
# sensitivity_analysis.py (~800 lines)
class TRITONSWMM_sensitivity_analysis:
    def _generate_master_snakefile_content(self):
        # ~200 lines of workflow generation
        # Duplicates patterns from workflow.py
        ...
```

**After:**
```python
# sensitivity_analysis.py (~600 lines, down from ~800)
class TRITONSWMM_sensitivity_analysis:
    def __init__(self, ...):
        self._workflow_builder = SensitivityAnalysisWorkflowBuilder(self)
    
    # Delegates to workflow builder
    # ~200 lines removed

# workflow.py or sensitivity_workflow.py (~200 lines added)
class SensitivityAnalysisWorkflowBuilder:
    """Builds Snakemake workflows for sensitivity analysis."""
    
    def __init__(self, sensitivity_analysis):
        self.sensitivity_analysis = sensitivity_analysis
        self._base_builder = SnakemakeWorkflowBuilder(sensitivity_analysis)
    
    def generate_master_snakefile_content(self) -> str:
        """Generate master Snakefile for sensitivity analysis."""
        # Reuses base builder patterns
        # Adds sensitivity-specific rules
        ...
```

---

## Validation Checklist

- [ ] Analyzed _generate_master_snakefile_content() in sensitivity_analysis.py
- [ ] Identified common patterns with SnakemakeWorkflowBuilder
- [ ] Identified sensitivity-specific requirements
- [ ] Created SensitivityAnalysisWorkflowBuilder class
- [ ] Moved workflow generation logic to new builder
- [ ] Updated sensitivity_analysis.py to use new builder
- [ ] Removed old _generate_master_snakefile_content() method
- [ ] Added type hints and docstrings to new class
- [ ] All 22 smoke tests passing (test_PC_01, test_PC_02, test_PC_04, test_PC_05)
- [ ] No changes to public API
- [ ] No changes to log file structures
- [ ] Sensitivity analysis workflow behavior unchanged

---

## Notes

- This phase follows the same pattern as Phases 1-3 (extract to focused component)
- Sensitivity analysis has unique workflow requirements (master + sub-workflows)
- Composition pattern recommended over inheritance for flexibility
- The new builder acts as a "strategy" object that encapsulates workflow generation
- This creates consistency across all workflow generation in the toolkit

---

**Last Updated:** January 27, 2026 - Phase 9 Implementation Guide
