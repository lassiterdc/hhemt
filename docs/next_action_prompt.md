# Phase 11: Code Quality Polish

**Date:** January 27, 2026 | **Status:** Ready for Implementation | **Goal:** Complete code quality improvements

**Previous Phase:** Phase 10 (Fix Naming Inconsistencies and Polish) completed - 22/22 tests passing ✅

---

## Objective

Complete remaining code quality improvements that were deferred from Phase 10. These enhancements will further improve code maintainability and ensure comprehensive code quality across the codebase.

**Expected Impact:**
- Comprehensive unused import cleanup across all files
- Complete documentation coverage for public methods
- Enhanced type safety with type hints
- Consistent error handling patterns
- Linter-compliant codebase

**Risk Level:** Low - Non-functional improvements only

**Priority:** Medium - Completes the code quality improvements started in Phase 10

---

## Tasks

### 1. Full Unused Import Scan

**Scope:** All source files in `src/TRITON_SWMM_toolkit/`

**Approach:**
```bash
# Using autoflake to detect unused imports
autoflake --check --remove-all-unused-imports src/TRITON_SWMM_toolkit/*.py

# Or using pylint
pylint --disable=all --enable=unused-import src/TRITON_SWMM_toolkit/
```

**Files to Check:**
- All `.py` files in `src/TRITON_SWMM_toolkit/`
- Focus on files modified during Phases 1-10
- Verify each unused import before removal

---

### 2. Comprehensive Public Method Docstring Audit

**Priority Files:**
1. `analysis.py` - Main orchestration class
2. `sensitivity_analysis.py` - Sensitivity analysis orchestration
3. `workflow.py` - Workflow generation
4. `execution.py` - Execution strategies
5. `resource_management.py` - Resource management

**Docstring Format (NumPy/Google Style):**
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

**Search for Methods Without Docstrings:**
```bash
# Find public methods without docstrings
grep -A 1 "def [^_]" src/TRITON_SWMM_toolkit/*.py | grep -v '"""'
```

---

### 3. Add Type Hints to Public Methods

**Priority:**
1. All public method parameters and return types
2. Class attributes in `__init__` methods
3. Complex internal methods

**Check for Missing Type Hints:**
```bash
# Use mypy to find missing type hints
mypy --disallow-untyped-defs src/TRITON_SWMM_toolkit/
```

**Note:** Don't add type hints to simple, obvious cases where they add no value.

---

### 4. Review and Standardize Error Handling

**Check Error Handling Patterns:**

1. **Consistent Exception Types:**
   - Use `ValueError` for invalid input values
   - Use `FileNotFoundError` for missing files
   - Use `RuntimeError` for runtime failures
   - Use `TypeError` for type mismatches

2. **Error Messages Should Be Descriptive:**
   ```python
   # BAD
   raise ValueError("Invalid input")
   
   # GOOD
   raise ValueError(
       f"Invalid analysis_id '{analysis_id}': must be alphanumeric, "
       f"got '{analysis_id}' with invalid characters"
   )
   ```

3. **Add Context to Exceptions:**
   ```python
   try:
       result = process_data(data)
   except Exception as e:
       raise RuntimeError(
           f"Failed to process data for scenario {scenario_id}: {e}"
       ) from e
   ```

---

### 5. Run Code Quality Checks

**Linters to Run:**

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

**Address High-Priority Issues:**
- Fix any errors (E-level in flake8)
- Fix critical warnings (C-level in pylint)
- Consider fixing other warnings if they improve code quality

---

### 6. Remove Commented-Out Code

**Search for Commented-Out Code:**
```bash
# Find commented-out code blocks
grep -n "^[[:space:]]*#.*def \|^[[:space:]]*#.*class \|^[[:space:]]*#.*import " src/TRITON_SWMM_toolkit/*.py
```

**Guidelines:**
- Remove commented-out code blocks (use git history if needed)
- Keep explanatory comments that add value
- Keep TODO/FIXME comments if they're actionable

---

## Implementation Steps

### Step 1: Automated Cleanup
1. Run autoflake to identify unused imports
2. Run flake8 to identify style issues
3. Create a list of issues to address

### Step 2: Manual Fixes
1. Remove confirmed unused imports
2. Add missing docstrings (prioritize public methods)
3. Add missing type hints (prioritize public methods)
4. Review and improve error messages

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
- Remove confirmed unused imports
- Add docstrings to public methods
- Improve error messages for clarity
- Run linters and address high-priority issues

❌ **DON'T:**
- Change functionality or behavior
- Modify public API methods
- Touch log file structures
- Break any tests
- Make large structural changes

---

## Expected Results

**Before:**
- Some unused imports in various files
- Some public methods without docstrings
- Some missing type hints
- Some linter warnings

**After:**
- All imports used or removed
- Complete docstrings on public methods
- Type hints on public methods
- Consistent, descriptive error messages
- Clean linter output (or documented exceptions)
- All 22 tests passing

---

## Validation Checklist

- [ ] Removed all unused imports across all source files
- [ ] Added docstrings to all public methods
- [ ] Added type hints to public methods
- [ ] Reviewed and improved error handling
- [ ] Ran flake8 and addressed issues
- [ ] Ran pylint and addressed critical issues
- [ ] Ran mypy and addressed type issues
- [ ] Removed commented-out code
- [ ] All 22 smoke tests passing (test_PC_01, test_PC_02, test_PC_04, test_PC_05)
- [ ] No changes to public API
- [ ] No changes to log file structures
- [ ] Documentation updated

---

## Notes

- Phase 11 completes the code quality improvements started in Phase 10
- Focus on high-value improvements that enhance maintainability
- Don't spend excessive time on minor style issues
- Document any decisions about what NOT to change
- After Phase 11, the refactoring project is fully complete!

---

**Last Updated:** January 27, 2026 - Phase 11 Implementation Guide (Optional)
