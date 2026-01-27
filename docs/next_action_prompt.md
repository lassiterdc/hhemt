# Phase 10: Fix Naming Inconsistencies and Polish

**Date:** January 27, 2026 | **Status:** Ready for Implementation | **Goal:** Final code quality improvements and cleanup

**Previous Phase:** Phase 9 (Unify Sensitivity Analysis Workflow Generation) completed - 22/22 tests passing ✅

---

## Objective

Address remaining code quality issues, fix naming inconsistencies, remove unused code, and ensure consistent documentation across the codebase.

**Expected Impact:**
- Improved code readability and maintainability
- Consistent naming conventions
- Complete documentation coverage
- Clean, linter-compliant codebase
- ~50-100 lines reduction through cleanup

**Risk Level:** Low - Non-functional improvements only

---

## Tasks

### 1. Fix Typos and Naming Inconsistencies

**Search for remaining typos:**
```bash
# Search for common typos
grep -r "retreive" src/TRITON_SWMM_toolkit/
grep -r "outptus" src/TRITON_SWMM_toolkit/
grep -r "occured" src/TRITON_SWMM_toolkit/
grep -r "sucessful" src/TRITON_SWMM_toolkit/
```

**Known issues to fix:**
- Any remaining `retreive` → `retrieve` typos
- Check for other common spelling errors in method names
- Ensure consistent naming patterns across modules

---

### 2. Remove Unused Imports

**Files with potential unused imports (from Phase 9 review):**
- `sensitivity_analysis.py` - Several imports appear unused after refactoring:
  - `analysis_config`
  - `TRITONSWMM_run`
  - `TRITONSWMM_sim_post_processing`
  - `TRITONSWMM_analysis_post_processing`
  - `print_json_file_tree`
  - `TRITONSWMM_analysis_log`
  - `TRITONSWMM_analysis_plotting`

**Approach:**
1. Use automated tools to detect unused imports:
   ```bash
   # Using autoflake
   autoflake --check --remove-all-unused-imports src/TRITON_SWMM_toolkit/*.py
   
   # Or using pylint
   pylint --disable=all --enable=unused-import src/TRITON_SWMM_toolkit/
   ```

2. Manually verify each unused import before removal
3. Remove confirmed unused imports
4. Run smoke tests to ensure no breakage

---

### 3. Add Missing Docstrings

**Priority files for docstring improvements:**

1. **`sensitivity_analysis.py`:**
   - Class docstring is incomplete: "Docstring for TRITONSWMM_sensitivity_analysis"
   - Should describe purpose, responsibilities, and usage

2. **All public methods without docstrings:**
   - Search for methods without docstrings:
     ```bash
     # Find methods without docstrings
     grep -A 1 "def " src/TRITON_SWMM_toolkit/*.py | grep -v '"""'
     ```

3. **Complex internal methods:**
   - Methods with >20 lines should have docstrings explaining logic
   - Methods with non-obvious behavior should be documented

**Docstring Format (Google Style):**
```python
def method_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Brief one-line description.
    
    Longer description if needed, explaining what the method does,
    when to use it, and any important details.
    
    Parameters
    ----------
    param1 : Type1
        Description of param1
    param2 : Type2
        Description of param2
    
    Returns
    -------
    ReturnType
        Description of return value
    
    Raises
    ------
    ExceptionType
        When this exception is raised
    """
```

---

### 4. Add Missing Type Hints

**Check for missing type hints:**
```bash
# Use mypy to find missing type hints
mypy --disallow-untyped-defs src/TRITON_SWMM_toolkit/
```

**Priority:**
1. All public method parameters and return types
2. Class attributes in `__init__` methods
3. Complex internal methods

**Note:** Don't add type hints to simple, obvious cases where they add no value.

---

### 5. Review and Standardize Error Handling

**Check error handling patterns:**

1. **Consistent exception types:**
   - Use `ValueError` for invalid input values
   - Use `FileNotFoundError` for missing files
   - Use `RuntimeError` for runtime failures
   - Use `TypeError` for type mismatches

2. **Error messages should be descriptive:**
   ```python
   # BAD
   raise ValueError("Invalid input")
   
   # GOOD
   raise ValueError(
       f"Invalid analysis_id '{analysis_id}': must be alphanumeric, "
       f"got '{analysis_id}' with invalid characters"
   )
   ```

3. **Add context to exceptions:**
   ```python
   try:
       result = process_data(data)
   except Exception as e:
       raise RuntimeError(
           f"Failed to process data for scenario {scenario_id}: {e}"
       ) from e
   ```

---

### 6. Run Code Quality Checks

**Linters to run:**

1. **flake8** - Style guide enforcement:
   ```bash
   flake8 src/TRITON_SWMM_toolkit/ --max-line-length=100 --ignore=E203,W503
   ```

2. **pylint** - Code quality:
   ```bash
   pylint src/TRITON_SWMM_toolkit/ --disable=C0103,R0913,R0914
   ```

3. **mypy** - Type checking:
   ```bash
   mypy src/TRITON_SWMM_toolkit/ --ignore-missing-imports
   ```

**Address high-priority issues:**
- Fix any errors (E-level in flake8)
- Fix critical warnings (C-level in pylint)
- Consider fixing other warnings if they improve code quality

---

### 7. Remove Commented-Out Code

**Search for commented-out code:**
```bash
# Find commented-out code blocks
grep -n "^[[:space:]]*#.*def \|^[[:space:]]*#.*class \|^[[:space:]]*#.*import " src/TRITON_SWMM_toolkit/*.py
```

**Guidelines:**
- Remove commented-out code blocks (use git history if needed)
- Keep explanatory comments that add value
- Keep TODO/FIXME comments if they're actionable

---

### 8. Assess File Organization (Optional - Time Permitting)

**Current structure:**
```
src/TRITON_SWMM_toolkit/
├── Core orchestration: analysis.py, sensitivity_analysis.py
├── Components: resource_management.py, execution.py, workflow.py
├── Scenario: scenario.py, scenario_inputs.py
├── SWMM: swmm_*.py (5 files)
├── Processing: process_*.py, processing_*.py
├── Plotting: plot_*.py (3 files)
├── Utilities: utils.py, subprocess_utils.py, paths.py
├── Config/Log: config.py, log.py, constants.py
├── CLI runners: *_runner.py (4 files)
└── Other: system.py, examples.py, gui.py, cli.py
```

**Potential improvements (for future consideration):**
- Group related files into subdirectories (e.g., `swmm/`, `plotting/`, `cli/`)
- Consider renaming files to be more descriptive
- Document any organizational decisions in README or architecture docs

**Note:** File reorganization is optional and should only be done if time permits and benefits are clear.

---

## Implementation Steps

### Step 1: Automated Cleanup
1. Run autoflake to identify unused imports
2. Run flake8 to identify style issues
3. Create a list of issues to address

### Step 2: Manual Fixes
1. Fix typos and naming inconsistencies
2. Remove confirmed unused imports
3. Add missing docstrings (prioritize public methods)
4. Add missing type hints (prioritize public methods)
5. Review and improve error messages

### Step 3: Code Quality
1. Run linters again and address remaining issues
2. Remove commented-out code
3. Ensure consistent formatting

### Step 4: Validation
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
- Fix obvious typos and naming issues
- Remove confirmed unused imports
- Add docstrings to public methods
- Improve error messages for clarity
- Run linters and address high-priority issues

❌ **DON'T:**
- Change functionality or behavior
- Modify public API methods
- Touch log file structures
- Break any tests
- Make large structural changes (save for future phases)

---

## Expected Results

**Before:**
- Some typos in method names
- Unused imports in several files
- Incomplete docstrings
- Inconsistent error handling
- Some linter warnings

**After:**
- No typos in method names
- All imports used or removed
- Complete docstrings on public methods
- Consistent, descriptive error messages
- Clean linter output (or documented exceptions)
- All 22 tests passing

---

## Validation Checklist

- [ ] Searched for and fixed all typos
- [ ] Removed all unused imports
- [ ] Added docstrings to public methods
- [ ] Added type hints where missing
- [ ] Reviewed and improved error handling
- [ ] Ran flake8 and addressed issues
- [ ] Ran pylint and addressed critical issues
- [ ] Removed commented-out code
- [ ] All 22 smoke tests passing (test_PC_01, test_PC_02, test_PC_04, test_PC_05)
- [ ] No changes to public API
- [ ] No changes to log file structures
- [ ] Documentation updated

---

## Notes

- This is the final cleanup phase of the refactoring project
- Focus on high-value improvements that enhance maintainability
- Don't spend excessive time on minor style issues
- Document any decisions about what NOT to change
- After Phase 10, the refactoring project is complete!

---

**Last Updated:** January 27, 2026 - Phase 10 Implementation Guide
