# Phase 7: Remove Delegation Wrappers in analysis.py

**Date:** January 27, 2026 | **Status:** Ready for Implementation | **Goal:** Clean up thin wrapper methods that delegate to extracted components

---

## Objective

Remove delegation wrapper methods from `analysis.py` that simply forward calls to `ResourceManager` and `SnakemakeWorkflowBuilder`. This makes the delegation explicit and reduces unnecessary indirection.

**Expected Impact:**
- Reduces `analysis.py` by ~100 lines
- Makes component delegation explicit and clear
- Reduces method call indirection
- Improves code readability

**Risk Level:** Low - Simple mechanical refactoring with clear call sites

---

## Methods to Remove

### From Phase 1 (ResourceManager) - 2 wrappers

**1. `_parse_slurm_tasks_per_node()`**
```python
# CURRENT: Wrapper method in analysis.py
def _parse_slurm_tasks_per_node(self, tasks_per_node_str: str) -> dict:
    return self._resource_manager.parse_slurm_tasks_per_node(tasks_per_node_str)

# AFTER: Direct access
# Replace: self._parse_slurm_tasks_per_node(...)
# With: self._resource_manager.parse_slurm_tasks_per_node(...)
```

**2. `_get_slurm_resource_constraints()`**
```python
# CURRENT: Wrapper method in analysis.py
def _get_slurm_resource_constraints(self) -> dict:
    return self._resource_manager.get_slurm_resource_constraints()

# AFTER: Direct access
# Replace: self._get_slurm_resource_constraints()
# With: self._resource_manager.get_slurm_resource_constraints()
```

### From Phase 3 (SnakemakeWorkflowBuilder) - 5 wrappers

**3. `_generate_snakefile_content()`**
```python
# CURRENT: Wrapper method in analysis.py
def _generate_snakefile_content(self, ...) -> str:
    return self._workflow_builder.generate_snakefile_content(...)

# AFTER: Direct access
# Replace: self._generate_snakefile_content(...)
# With: self._workflow_builder.generate_snakefile_content(...)
```

**4. `_generate_snakemake_config()`**
```python
# CURRENT: Wrapper method in analysis.py
def _generate_snakemake_config(self) -> dict:
    return self._workflow_builder.generate_snakemake_config()

# AFTER: Direct access
# Replace: self._generate_snakemake_config()
# With: self._workflow_builder.generate_snakemake_config()
```

**5. `_write_snakemake_config()`**
```python
# CURRENT: Wrapper method in analysis.py
def _write_snakemake_config(self, config: dict):
    return self._workflow_builder.write_snakemake_config(config)

# AFTER: Direct access
# Replace: self._write_snakemake_config(...)
# With: self._workflow_builder.write_snakemake_config(...)
```

**6. `_run_snakemake_local()`**
```python
# CURRENT: Wrapper method in analysis.py
def _run_snakemake_local(self, ...):
    return self._workflow_builder.run_snakemake_local(...)

# AFTER: Direct access
# Replace: self._run_snakemake_local(...)
# With: self._workflow_builder.run_snakemake_local(...)
```

**7. `_run_snakemake_slurm()`**
```python
# CURRENT: Wrapper method in analysis.py
def _run_snakemake_slurm(self, ...):
    return self._workflow_builder.run_snakemake_slurm(...)

# AFTER: Direct access
# Replace: self._run_snakemake_slurm(...)
# With: self._workflow_builder.run_snakemake_slurm(...)
```

---

## Implementation Steps

### Step 1: Find All Call Sites
Search `analysis.py` for all calls to the 7 wrapper methods:
```bash
grep -n "_parse_slurm_tasks_per_node\|_get_slurm_resource_constraints\|_generate_snakefile_content\|_generate_snakemake_config\|_write_snakemake_config\|_run_snakemake_local\|_run_snakemake_slurm" src/TRITON_SWMM_toolkit/analysis.py
```

### Step 2: Update Call Sites
For each call site, update to use direct component access:
- `self._parse_slurm_tasks_per_node(...)` → `self._resource_manager.parse_slurm_tasks_per_node(...)`
- `self._get_slurm_resource_constraints()` → `self._resource_manager.get_slurm_resource_constraints()`
- `self._generate_snakefile_content(...)` → `self._workflow_builder.generate_snakefile_content(...)`
- `self._generate_snakemake_config()` → `self._workflow_builder.generate_snakemake_config()`
- `self._write_snakemake_config(...)` → `self._workflow_builder.write_snakemake_config(...)`
- `self._run_snakemake_local(...)` → `self._workflow_builder.run_snakemake_local(...)`
- `self._run_snakemake_slurm(...)` → `self._workflow_builder.run_snakemake_slurm(...)`

### Step 3: Remove Wrapper Methods
Delete all 7 wrapper method definitions from `analysis.py`

### Step 4: Validate
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
- Update all call sites before removing wrapper methods
- Use direct component access (`self._resource_manager.*`, `self._workflow_builder.*`)
- Keep method signatures and behavior identical
- Verify no other files import these wrapper methods

❌ **DON'T:**
- Change any logic or behavior
- Modify the component classes themselves
- Touch any public API methods
- Skip any call sites

---

## Expected Results

**Before:**
```python
class TRITONSWMM_analysis:
    def some_method(self):
        constraints = self._get_slurm_resource_constraints()  # Wrapper call
        config = self._generate_snakemake_config()  # Wrapper call
        
    def _get_slurm_resource_constraints(self):  # Wrapper method
        return self._resource_manager.get_slurm_resource_constraints()
        
    def _generate_snakemake_config(self):  # Wrapper method
        return self._workflow_builder.generate_snakemake_config()
```

**After:**
```python
class TRITONSWMM_analysis:
    def some_method(self):
        constraints = self._resource_manager.get_slurm_resource_constraints()  # Direct call
        config = self._workflow_builder.generate_snakemake_config()  # Direct call
        
    # Wrapper methods removed - ~100 lines deleted
```

---

## Validation Checklist

- [ ] Found all call sites for 7 wrapper methods
- [ ] Updated all call sites to use direct component access
- [ ] Removed all 7 wrapper method definitions
- [ ] Verified no other files import these methods
- [ ] All 22 smoke tests passing (test_PC_01, test_PC_02, test_PC_04, test_PC_05)
- [ ] No changes to public API
- [ ] No changes to component classes

---

## Notes

- This is a pure refactoring - no logic changes
- Makes delegation explicit and reduces indirection
- Improves code clarity by showing which component handles each responsibility
- Sets up cleaner architecture for future phases

---

**Last Updated:** January 27, 2026 - Phase 7 Implementation Guide
