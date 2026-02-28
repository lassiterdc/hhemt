# Test Fixture Usage Audit (Phase 6b.1)

**Date**: 2026-02-09
**Phase**: Tier 1 Phase 6b.1 - Fixture consolidation audit
**Purpose**: Understand current fixture architecture before consolidation

---

## Executive Summary

**Current State**:
- **24 pytest fixtures** defined in `tests/conftest.py`
- **3 platform-specific catalog classes** (`Local_TestCases`, `UVA_TestCases`, `Frontier_TestCases`)
- **14 factory methods** in `test_case_catalog.py`
- **6 tests** bypass fixtures and call catalog directly
- **8 platform detection calls** scattered across test files

**Key Patterns Identified**:
1. **Naming Convention**: `norfolk_{platform}_{type}_{variant}_analysis[_cached]`
2. **Duplication**: Each platform has nearly identical fixture patterns
3. **Mixed Access**: Some tests use fixtures, others call catalog directly
4. **Cache Pattern**: Paired fixtures (fresh vs cached) for faster test iteration

**Consolidation Opportunities**:
- Reduce 24 fixtures to ~6-8 parametrized fixtures
- Eliminate platform-specific catalog classes (use conditional logic)
- Standardize access pattern (all tests use fixtures)

---

## 1. Fixture Inventory

### 1.1 Platform-Specific Fixtures

**Pattern**: Each platform has similar fixture sets

| Platform | Multi-sim | Multi-sim (cached) | Sensitivity | Sensitivity (cached) | GPU |
|----------|-----------|-------------------|-------------|---------------------|-----|
| **Local (PC)** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **UVA** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Frontier** | ✅ | ✅ | ✅ | ✅ | ✅ (CPU+GPU) |

**Observation**: 85% duplication across platforms - only config paths differ

### 1.2 Model-Configuration Fixtures

**Purpose**: Test multi-model integration (TRITON/SWMM combinations)

| Fixture | Models Enabled | Usage Count |
|---------|---------------|-------------|
| `norfolk_triton_only_analysis` | TRITON only | 3 tests |
| `norfolk_swmm_only_analysis` | SWMM only | 2 tests |
| `norfolk_all_models_analysis` | All 3 models | 5 tests (PC_01) |
| `norfolk_triton_and_tritonswmm_analysis` | TRITON + coupled | 2 tests |

**Observation**: These are **content-specific**, not platform-specific - good candidates to keep separate

### 1.3 Fixture Naming Pattern

```
norfolk_{platform}_{test_type}_{variant}_analysis[_cached]
  │         │           │          │            └─ Optional: cached version
  │         │           │          └─ Optional: cpu/gpu/minimal/full
  │         │           └─ Type: multisim, sensitivity, single
  │         └─ Platform: (none)=local, uva, frontier
  └─ Location: Always "norfolk"
```

**Examples**:
- `norfolk_multi_sim_analysis` - Local multi-simulation
- `norfolk_frontier_multisim_gpu_analysis` - Frontier GPU multi-sim
- `norfolk_uva_sensitivity_analysis_cached` - UVA sensitivity (cached)

---

## 2. Catalog Architecture

### 2.1 Catalog Class Hierarchy

```python
GetTS_TestCases (base class)
├── Local_TestCases
│   ├── retrieve_norfolk_multi_sim_test_case()
│   ├── retrieve_norfolk_single_sim_test_case()
│   ├── retrieve_norfolk_triton_only_test_case()
│   ├── retrieve_norfolk_swmm_only_test_case()
│   ├── retrieve_norfolk_all_models_test_case()
│   └── retrieve_norfolk_cpu_config_sensitivity_case()
├── UVA_TestCases
│   ├── retrieve_norfolk_UVA_multisim_1cpu_case()
│   ├── retrieve_norfolk_UVA_sensitivity_CPU_minimal()
│   └── retrieve_norfolk_UVA_sensitivity_CPU_full_ensemble_short_sims()
└── Frontier_TestCases
    ├── retrieve_norfolk_frontier_multisim_gpu_case()
    ├── retrieve_norfolk_frontier_multisim_cpu_serial_case()
    ├── retrieve_norfolk_frontier_sensitivity_minimal()
    └── retrieve_norfolk_frontier_sensitivity_suite()
```

### 2.2 Catalog Method Responsibilities

Each method:
1. Detects platform (via hostname/env vars)
2. Selects appropriate config paths for platform
3. Calls `retrieve_TRITON_SWMM_test_case()` constructor
4. Returns constructed test case

**Problem**: Platform detection happens in 3 places (utils, catalog subclasses, test skipif)

---

## 3. Usage Patterns

### 3.1 Fixture-Based Tests (Majority)

**Count**: 25 test files use fixtures
**Pattern**:
```python
def test_something(norfolk_multi_sim_analysis):
    analysis = norfolk_multi_sim_analysis
    # test logic
```

**Advantages**:
- Clean test code
- Pytest handles lifecycle
- Easy to see fixture dependency

### 3.2 Direct Catalog Calls (6 Tests)

**Files**:
- `test_multi_model_integration.py`
- `test_resource_management_1job_mode.py`
- `test_swmm_output_parser_refactoring.py`
- `test_workflow_1job_dry_run.py`
- `test_workflow_1job_profile_generation.py`
- `test_workflow_1job_sbatch_generation.py`

**Pattern**:
```python
def test_something():
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = case.analysis
    # test logic
```

**Problem**: Inconsistent with fixture-based tests, harder to parametrize

---

## 4. Platform Detection

### 4.1 Detection Utilities (`tests/utils_for_testing.py`)

```python
def uses_slurm() -> bool
def on_frontier() -> bool
def on_UVA_HPC() -> bool
```

### 4.2 Usage Locations

| Location | Purpose |
|----------|---------|
| Test skipif decorators | Skip tests on wrong platform (4 instances) |
| Catalog subclasses | Select platform configs (implicit in class) |
| Fixture conftest | Choose which catalog class to call |

**Problem**: Tripled detection logic - should be centralized

---

## 5. Cache Pattern Analysis

### 5.1 Fresh vs Cached Fixtures

**Purpose**: Speed up test iteration by reusing compiled executables

| Type | start_from_scratch | Use Case |
|------|-------------------|----------|
| **Fresh** | `True` | First run, test setup logic |
| **Cached** | `False` | Test execution/processing (reuse setup) |

**Usage**:
- Setup tests use fresh fixtures (test_PC_00, test_PC_01 setup)
- Execution tests use cached fixtures (test_PC_01 run/process, test_PC_02)

### 5.2 Fixture Pairs

Every major fixture has a cached variant:
- `norfolk_multi_sim_analysis` / `norfolk_multi_sim_analysis_cached`
- `norfolk_all_models_analysis` / `norfolk_all_models_analysis_cached`
- etc. (12 pairs total = 24 fixtures)

**Observation**: This pattern works well, should be preserved in consolidation

---

## 6. Cross-Platform Test Duplication

### 6.1 Duplicated Test Files

Similar tests exist for each platform:

| Test Type | Local | UVA | Frontier |
|-----------|-------|-----|----------|
| 1-core multisim | ❌ | ✅ UVA_01 | ✅ frontier_01 |
| Snakemake multisim | ✅ PC_04 | ✅ UVA_02 | ✅ frontier_03 (CPU), frontier_04 (GPU) |
| Sensitivity analysis | ✅ PC_05 | ✅ UVA_03, UVA_04 | ✅ frontier_05 |

**Duplication**:
- Workflow generation tests: Nearly identical across platforms
- Dry-run tests: Platform-agnostic logic
- Execution tests: Some platform-specific differences (SLURM flags)

---

## 7. Consolidation Recommendations

### 7.1 Phase 6b.2 - Unified Fixture API

**Proposed Structure**:

```python
# conftest.py - Single fixture per test type

@pytest.fixture(params=["local", "uva", "frontier"])
def norfolk_analysis(request, platform):
    """Multi-simulation analysis (platform-parametrized)."""
    if platform == "uva" and not on_UVA_HPC():
        pytest.skip("UVA platform not available")
    # Similar for frontier

    case = get_test_case(
        location="norfolk",
        test_type="multi_sim",
        platform=platform,
        start_from_scratch=True
    )
    return case.analysis

@pytest.fixture
def norfolk_analysis_cached(norfolk_analysis):
    """Cached version - reuses setup from norfolk_analysis."""
    # Implementation that reuses existing case
```

**Benefits**:
- 24 fixtures → ~8 fixtures (67% reduction)
- Platform-agnostic test code
- Easy to add new platforms

### 7.2 Phase 6b.3 - Simplified Catalog

**Proposed**:

```python
# test_case_catalog.py - Single class with conditional logic

class TestCaseFactory:
    @classmethod
    def get_case(
        cls,
        location: str = "norfolk",
        test_type: Literal["multi_sim", "sensitivity", "single"],
        platform: Literal["local", "uva", "frontier"] = "local",
        variant: str | None = None,  # "gpu", "minimal", etc.
        start_from_scratch: bool = True,
    ):
        """Unified test case factory with platform selection."""
        # Platform detection
        if platform == "auto":
            platform = detect_platform()

        # Config selection based on platform + test_type
        config_paths = cls._get_config_paths(platform, test_type, variant)

        # Delegate to builder
        return retrieve_TRITON_SWMM_test_case(
            cfg_system_yaml=config_paths.system,
            cfg_anlysys_yaml=config_paths.analysis,
            start_from_scratch=start_from_scratch,
        )
```

**Benefits**:
- Single class instead of 3 subclasses
- Platform selection explicit parameter
- Easy to understand and extend

### 7.3 Phase 6b.4 - Migrate Direct Calls

**Change**:
```python
# Before
case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
    start_from_scratch=True
)
analysis = case.analysis

# After
def test_something(norfolk_analysis):
    analysis = norfolk_analysis  # Use fixture
```

**Files to update**: 6 test files with direct catalog calls

---

## 8. Risk Assessment

### 8.1 High Risk Areas

1. **Platform-specific test breakage** - Tests may have subtle platform dependencies
2. **Cache invalidation** - Cached fixtures share state, parametrization may break this
3. **Execution tests** - Platform-specific SLURM configurations may not parametrize cleanly

### 8.2 Mitigation Strategies

1. **Incremental migration** - Keep old fixtures alongside new until validated
2. **Smoke tests after each change** - Run PC_01-05 to catch regressions
3. **Start with workflow generation** - Lowest risk tests (no actual execution)

---

## 9. Implementation Roadmap

### Phase 6b.2: Unified Fixture API (3 days)

1. Create parametrized `norfolk_analysis` fixture (local platform only)
2. Validate with PC_04 workflow generation tests
3. Add UVA/Frontier parameters after local validated
4. Keep old fixtures alongside new

### Phase 6b.3: Simplified Catalog (2 days)

1. Create `TestCaseFactory` class with `get_case()` method
2. Migrate `_get_config_paths()` from subclasses to factory
3. Update `conftest.py` to use new factory
4. Remove old catalog subclasses after validation

### Phase 6b.4: Migrate Direct Calls (1 day)

1. Update 6 test files to use fixtures instead of direct calls
2. Remove direct imports of catalog classes
3. Run full test suite to validate

### Total Timeline: 6 days (part-time)

---

## 10. Success Metrics

- **Fixture count**: 24 → 8 (67% reduction)
- **Catalog classes**: 3 → 1 (unified factory)
- **Direct catalog calls**: 6 → 0 (all via fixtures)
- **Platform detection**: 3 locations → 1 (utils only)
- **All smoke tests**: Still passing after migration

---

## 11. Next Steps

1. ✅ Complete this audit (Phase 6b.1)
2. [ ] Get approval to proceed with Phase 6b.2
3. [ ] Implement unified fixture API pilot (local platform only)
4. [ ] Validate with workflow generation tests
5. [ ] Expand to all platforms after pilot success

---

## Appendix: Fixture Reference Table

| Fixture Name | Platform | Type | Cached | Users |
|--------------|----------|------|--------|-------|
| norfolk_single_sim_analysis | Local | Single | No | 5 tests (PC_00) |
| norfolk_multi_sim_analysis | Local | Multi | No | 4 tests (PC_02, PC_04) |
| norfolk_multi_sim_analysis_cached | Local | Multi | Yes | 2 tests (PC_02) |
| norfolk_all_models_analysis | Local | Multi | No | 1 test |
| norfolk_all_models_analysis_cached | Local | Multi | Yes | 4 tests (PC_01) |
| norfolk_sensitivity_analysis | Local | Sens | No | 1 test (PC_05) |
| norfolk_frontier_multisim_analysis | Frontier | Multi | No | 2 tests |
| norfolk_frontier_multisim_analysis_cached | Frontier | Multi | Yes | 5 tests |
| norfolk_frontier_multisim_gpu_analysis | Frontier | Multi+GPU | No | 1 test |
| norfolk_frontier_sensitivity_analysis | Frontier | Sens | No | 0 tests |
| norfolk_uva_multisim_analysis | UVA | Multi | No | 3 tests |
| norfolk_uva_multisim_analysis_cached | UVA | Multi | Yes | 4 tests |
| norfolk_uva_sensitivity_analysis | UVA | Sens | No | 1 test |
| norfolk_triton_only_analysis | Local | Model-specific | No | 3 tests |
| norfolk_swmm_only_analysis | Local | Model-specific | No | 2 tests |

**Total**: 24 fixtures, 15 unique fixture bases (ignoring cached variants)

---

**End of Audit**
