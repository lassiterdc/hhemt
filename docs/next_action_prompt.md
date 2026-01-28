# Phase 2 Implementation Prompt for AI Agent

## Context

You are starting Phase 2 of the test refactor plan focused on non-PC test suites (Frontier/UVA). The goal is to reuse Phase 1 fixtures/helpers and reduce duplication while preserving or strengthening test coverage.

**Reference Document:** `docs/test_refactor_plan.md`

---

## Objective

Refactor non-PC tests (Frontier/UVA) by introducing shared fixtures/helpers and parametrized Snakefile checks.

## Phase 2 Tasks: Non-PC Tests Refactor

### Task 1: Reuse Shared Fixtures

**Goals:**
- Reuse analysis fixtures in `tests/conftest.py`
- Ensure fixtures preserve `start_from_scratch` semantics for new suites

### Task 2: Reuse Helper Assertions

**Goals:**
- Apply shared assertion helpers in `tests/utils_for_testing.py`
- Consolidate repeated log checks and Snakefile validation logic

### Task 3: Refactor Non-PC Test Modules

**Targets:**
- Update `tests/test_frontier_01_1core_multisim.py`
- Update `tests/test_frontier_02_all_compute_configs.py`
- Update `tests/test_UVA_01_1core_multisim.py`
- Update `tests/test_UVA_02_multisim_with_snakemake.py`
- Update `tests/test_UVA_03_sensitivity_analysis_with_snakemake.py`
- Update `tests/test_UVA_04_multiCPU_sensitivity_analysis_minirun.py`

### Task 4: Parametrize Snakefile Variants

**Goals:**
- Replace repeated Snakefile config tests with `pytest.mark.parametrize`
- Ensure inclusion/exclusion of flags is still validated
---

## Testing Requirements

### 1. Frontier Tests (if available)
```bash
pytest -k "test_frontier" -v
```

### 2. UVA Tests (if available)
```bash
pytest -k "test_UVA" -v
```
```

---

## Commit Strategy

Make separate commits for each task (or grouped by function):

1. `tests(non-pc): refactor Frontier tests`
2. `tests(non-pc): refactor UVA tests`
3. `tests(non-pc): parametrize Snakefile checks`

---

## Success Criteria

- [ ] Frontier/UVA tests refactored to use shared fixtures/helpers
- [ ] Snakefile checks parametrized where applicable
- [ ] Relevant non-PC test suites pass in their environments

**Note:** Phase 2 is complete once non-PC tests are refactored and the relevant test suites pass.

---

## Files Summary

| File | Action |
|------|--------|
| `tests/test_frontier_01_1core_multisim.py` | Refactor |
| `tests/test_frontier_02_all_compute_configs.py` | Refactor |
| `tests/test_UVA_01_1core_multisim.py` | Refactor |
| `tests/test_UVA_02_multisim_with_snakemake.py` | Refactor + parametrize |
| `tests/test_UVA_03_sensitivity_analysis_with_snakemake.py` | Refactor + parametrize |
| `tests/test_UVA_04_multiCPU_sensitivity_analysis_minirun.py` | Refactor |
| `docs/next_action_prompt.md` | This prompt |