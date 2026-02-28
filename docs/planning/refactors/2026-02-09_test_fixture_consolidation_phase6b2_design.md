# Phase 6b.2: Unified Fixture API Design

**Date**: 2026-02-09
**Phase**: Tier 1 Phase 6b.2 - Design unified fixture API
**Prerequisite**: Phase 6b.1 audit complete ✅

---

## 1. Executive Summary

**Goal**: Reduce 24 fixtures → 8 through parametrization while improving clarity and maintainability.

**Approach**: Incremental "expand then contract" refactoring
1. Add new unified fixtures **alongside** existing ones
2. Validate with pilot test conversions
3. Migrate remaining tests incrementally
4. Remove old fixtures only after full validation

**Risk mitigation**: Keep both APIs operational during transition, validate at each step.

---

## 2. Current State Analysis

### 2.1 Duplication Pattern

conftest.py has 25 fixtures following this pattern:
```
{location}_{platform}_{test_type}_{variant}_analysis_{cached}
```

**Dimensions:**
- Location: norfolk (only value currently)
- Platform: local (implicit), uva, frontier
- Test type: single_sim, multi_sim, sensitivity
- Variant: all_models, triton_only, swmm_only, gpu, minimal
- Cached: (default), _cached

**Examples:**
- `norfolk_multi_sim_analysis` (local, fresh)
- `norfolk_uva_multisim_analysis_cached` (UVA, cached)
- `norfolk_frontier_multisim_gpu_analysis` (Frontier GPU, fresh)

### 2.2 Problems with Current Approach

1. **Explosion of fixtures**: Adding a new platform/variant requires creating N new fixtures
2. **No clear naming convention**: "multi_sim" vs "multisim", inconsistent order
3. **Duplication**: Same logic repeated 25 times with minor variations
4. **Hard to extend**: Adding "minimal" variant to all platforms = 9 new fixtures
5. **Discovery**: Hard to know which fixture to use for a given need

---

## 3. Design Principles

### 3.1 Core Principles

1. **Single Responsibility**: One fixture per test type, parametrized by platform
2. **Explicit over Implicit**: Platform selection should be obvious
3. **Fail Fast**: Skip unavailable platforms immediately with clear messages
4. **Backward Compatible (temporarily)**: Keep old fixtures during migration
5. **Easy to Extend**: Adding new platform/variant touches minimal code

### 3.2 Naming Convention

**Standard fixture names:**
```
norfolk_{test_type}_analysis[_{variant}]
```

Where:
- `test_type`: `multi_sim`, `single_sim`, `sensitivity`
- `variant` (optional): `all_models`, `triton_only`, `swmm_only`, `gpu`, `minimal`
- Platform: specified via parametrization, not name

**Caching:**
- Default: `start_from_scratch=True` (fresh)
- Cached: Use `_cached` suffix or separate fixture

---

## 4. Proposed API

### 4.1 Core Unified Fixtures

```python
# conftest.py

import pytest
import tests.fixtures.test_case_catalog as cases
import tests.utils_for_testing as tst_ut


# ========== Phase 6b.2: Unified Fixture API (Pilot) ==========


@pytest.fixture(
    params=[
        pytest.param("local", id="local"),
        pytest.param("uva", marks=pytest.mark.skipif(
            not tst_ut.on_UVA_HPC(), reason="UVA platform only"
        ), id="uva"),
        pytest.param("frontier", marks=pytest.mark.skipif(
            not tst_ut.on_frontier(), reason="Frontier platform only"
        ), id="frontier"),
    ]
)
def platform(request):
    """Platform selection for parametrized fixtures."""
    return request.param


@pytest.fixture
def norfolk_multi_sim_unified(platform):
    """Multi-simulation analysis (platform-parametrized).

    Replaces:
    - norfolk_multi_sim_analysis (local)
    - norfolk_uva_multisim_analysis (UVA)
    - norfolk_frontier_multisim_analysis (Frontier)

    Usage:
        def test_something(norfolk_multi_sim_unified):
            analysis = norfolk_multi_sim_unified
            # Test runs 3x (local, UVA, Frontier) with automatic skipping
    """
    # Platform-specific catalog selection
    if platform == "local":
        case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
            start_from_scratch=True
        )
    elif platform == "uva":
        case = cases.UVA_TestCases.retrieve_norfolk_UVA_multisim_1cpu_case(
            start_from_scratch=True
        )
    elif platform == "frontier":
        case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_cpu_serial_case(
            start_from_scratch=True
        )
    else:
        pytest.fail(f"Unknown platform: {platform}")

    return case.analysis


@pytest.fixture
def norfolk_multi_sim_unified_cached(platform):
    """Multi-simulation analysis (cached, platform-parametrized)."""
    if platform == "local":
        case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
            start_from_scratch=False
        )
    elif platform == "uva":
        case = cases.UVA_TestCases.retrieve_norfolk_UVA_multisim_1cpu_case(
            start_from_scratch=False
        )
    elif platform == "frontier":
        case = cases.Frontier_TestCases.retrieve_norfolk_frontier_multisim_cpu_serial_case(
            start_from_scratch=False
        )
    else:
        pytest.fail(f"Unknown platform: {platform}")

    return case.analysis
```

### 4.2 Usage Examples

**Before (old API):**
```python
def test_workflow_generation(norfolk_multi_sim_analysis):
    analysis = norfolk_multi_sim_analysis
    # Test logic (runs on local only)


def test_workflow_generation_uva(norfolk_uva_multisim_analysis):
    analysis = norfolk_uva_multisim_analysis
    # Same test logic, different fixture
```

**After (unified API):**
```python
def test_workflow_generation(norfolk_multi_sim_unified):
    analysis = norfolk_multi_sim_unified
    # Test logic (runs 3x: local, UVA, Frontier with auto-skip)
```

### 4.3 Platform-Specific Tests

For tests that truly need platform-specific logic:
```python
def test_slurm_specific(norfolk_multi_sim_unified, platform):
    analysis = norfolk_multi_sim_unified

    if platform == "local":
        pytest.skip("SLURM test not applicable to local")

    # SLURM-specific test logic
```

---

## 5. Implementation Phases

### Phase 6b.2.1: Pilot (Local Only) - Day 1

**Goal**: Validate unified API pattern with minimal risk

**Actions:**
1. Add `platform` fixture to conftest.py (local param only)
2. Add `norfolk_multi_sim_unified` fixture (local only)
3. Keep all existing fixtures untouched
4. Convert 1-2 workflow generation tests to use new fixture
5. Run smoke tests to validate

**Success criteria:**
- Pilot tests pass with new fixture
- Existing tests still pass with old fixtures
- No regressions in PC_01-05

### Phase 6b.2.2: Expand to All Platforms - Day 2

**Goal**: Add UVA/Frontier parameters after local validated

**Actions:**
1. Add UVA and Frontier params to `platform` fixture
2. Update `norfolk_multi_sim_unified` with platform branching
3. Add `norfolk_multi_sim_unified_cached` variant
4. Convert 2-3 more tests to validate auto-skip behavior
5. Test on local (should skip UVA/Frontier)

**Success criteria:**
- Tests using unified fixture run 3x (or skip appropriately)
- Platform-specific tests skip cleanly
- Smoke tests still pass

### Phase 6b.2.3: Add Sensitivity Fixtures - Day 3

**Goal**: Extend pattern to sensitivity analysis tests

**Actions:**
1. Add `norfolk_sensitivity_unified` fixture
2. Add `norfolk_sensitivity_unified_cached` variant
3. Convert PC_05 to use new fixtures
4. Validate sensitivity workflow still works

**Success criteria:**
- PC_05 passes with unified fixtures
- Platform variations work correctly

---

## 6. Migration Strategy (Future Phases)

### Phase 6b.3: Migrate All Tests

Convert remaining tests file-by-file:
1. test_PC_04 (workflow generation) ← Lowest risk
2. test_PC_02 (execution tests)
3. test_PC_01 (single sim + all models)
4. Platform-specific tests (UVA, Frontier)

### Phase 6b.4: Remove Old Fixtures

Only after 100% migration:
1. Remove old fixtures from conftest.py
2. Remove direct catalog calls from tests
3. Update documentation

---

## 7. Open Questions

### 7.1 Caching Strategy

**Question**: How to handle `start_from_scratch` with parametrized fixtures?

**Options:**
1. **Separate fixtures** (`_unified` vs `_unified_cached`) ← **Recommended**
   - Pros: Clear, explicit, matches current pattern
   - Cons: 2x fixture count (but still way better than 25)

2. **Parameter** (`@pytest.fixture(params=[True, False])`)
   - Pros: Single fixture
   - Cons: Every test runs 2x (fresh + cached) even if only needs one

**Decision**: Use separate fixtures (Option 1)

### 7.2 Model Variant Fixtures

**Question**: How to handle `all_models`, `triton_only`, `swmm_only`?

**Options:**
1. **Separate fixtures** for each variant
2. **Parameter** to unified fixture
3. **Model-specific fixtures** not unified

**Decision**: Keep model variants as separate fixtures for now (low duplication, high specificity)

### 7.3 GPU Fixtures

**Question**: Unify GPU fixtures or keep separate?

**Decision**: Keep GPU as separate variant fixture (highly specific, low duplication)

---

## 8. Expected Outcomes

### 8.1 Fixture Count Reduction

**Before:**
- 25 fixtures in conftest.py

**After (Phase 6b.2 complete):**
- 6 unified fixtures (multi_sim, sensitivity × fresh/cached × 3 platforms)
- 8 variant fixtures (all_models, triton_only, swmm_only, gpu, etc.)
- ~14 total (44% reduction)

**After (Phase 6b.3-4 complete):**
- Further consolidation possible
- Target: ~8-10 fixtures (60-67% reduction)

### 8.2 Test Count Changes

Tests using unified fixtures will run multiple times (once per platform param):
- 1 test → 3 test items (local, UVA, Frontier)
- Skip markers prevent cross-platform execution

**Example:**
```
test_workflow_generation[local]      PASSED
test_workflow_generation[uva]        SKIPPED (UVA platform only)
test_workflow_generation[frontier]   SKIPPED (Frontier platform only)
```

---

## 9. Risk Assessment

### 9.1 Risks

1. **Cache invalidation**: Parametrized fixtures may break cached test dependencies
   - Mitigation: Keep cached/fresh separation

2. **Test explosion**: Unified fixtures multiply test count
   - Mitigation: Skip markers prevent actual execution on unavailable platforms

3. **Platform detection bugs**: Auto-skip logic must be robust
   - Mitigation: Test skip behavior explicitly in pilot

### 9.2 Rollback Plan

If unified fixtures cause problems:
1. Keep old fixtures operational (they're not removed until Phase 6b.4)
2. Revert test file changes (git reset specific files)
3. Document issues and redesign

---

## 10. Success Metrics

Phase 6b.2 complete when:
- ✅ `platform` fixture implemented
- ✅ 2-3 unified fixtures created (multi_sim, sensitivity × fresh/cached)
- ✅ 3-5 tests converted to use unified fixtures
- ✅ All smoke tests pass (PC_01-05)
- ✅ Skip behavior validated (tests skip on unavailable platforms)
- ✅ Documentation updated (this design doc + CLAUDE.md)

---

## 11. Next Steps

1. ✅ Complete this design document
2. [ ] Get approval for pilot (Phase 6b.2.1)
3. [ ] Implement pilot with local-only parametrization
4. [ ] Validate pilot with 1-2 test conversions
5. [ ] Expand to full platform coverage (6b.2.2)
6. [ ] Add sensitivity fixtures (6b.2.3)
7. [ ] Plan Phase 6b.3 (full migration)

---

**End of Design Document**
