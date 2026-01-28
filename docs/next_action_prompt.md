# Phase 1 Implementation Prompt for AI Agent

## Context

You are starting Phase 1 of the test refactor plan focused on PC-prefixed tests. The goal is to reduce duplication while preserving or strengthening test coverage.

**Reference Document:** `docs/test_refactor_plan.md`

---

## Objective

Refactor `test_PC_*` tests by introducing shared fixtures, helper assertions, and parametrized Snakefile checks.

## Phase 1 Tasks: PC Tests Refactor

### Task 1: Add Shared Fixtures

**Goals:**
- Add reusable analysis fixtures in `tests/conftest.py`
- Ensure fixtures preserve `start_from_scratch` semantics

### Task 2: Add Helper Assertions

**Goals:**
- Add shared assertion helpers in `tests/utils_for_testing.py`
- Consolidate repeated log checks and Snakefile validation logic

### Task 3: Refactor PC Test Modules

**Goals (optional):**
- Update `test_PC_01_singlesim.py`
- Update `test_PC_02_multisim.py`
- Update `test_PC_04_multisim_with_snakemake.py`
- Update `test_PC_05_sensitivity_analysis_with_snakemake.py`

### Task 4: Parametrize Snakefile Variants

**Goals:**
- Replace repeated Snakefile config tests with `pytest.mark.parametrize`
- Ensure inclusion/exclusion of flags is still validated

---

## Testing Requirements

After Phase 6 changes:

### 1. PC Test Suite
```bash
pytest -k "test_PC" -v
```

### 2. PC Tests With Warnings as Errors
```bash
pytest -k "test_PC" -v -W error::UserWarning
```

---

## Commit Strategy

Make separate commits for each task (or grouped by function):

1. `tests(pc): add shared fixtures + helpers`
2. `tests(pc): refactor PC test modules`
3. `tests(pc): parametrize Snakefile checks`

---

## Success Criteria

- [ ] Shared fixtures + helpers added
- [ ] PC tests refactored + parametrized
- [ ] PC test suite passes

**Note:** Phase 1 is complete once PC tests are refactored and all `test_PC_*` cases pass.

---

## Files Summary

| File | Action |
|------|--------|
| `tests/conftest.py` | Add shared fixtures |
| `tests/utils_for_testing.py` | Add helper assertions |
| `tests/test_PC_01_singlesim.py` | Refactor |
| `tests/test_PC_02_multisim.py` | Refactor |
| `tests/test_PC_04_multisim_with_snakemake.py` | Refactor + parametrize |
| `tests/test_PC_05_sensitivity_analysis_with_snakemake.py` | Refactor + parametrize |
| `docs/next_action_prompt.md` | This prompt |