# Test Assertion Pattern Audit (Phase 6d.1)

**Date**: 2026-02-09
**Phase**: Tier 1 Phase 6d.1 - Assertion standardization audit
**Purpose**: Analyze current assertion patterns before standardization

---

## Executive Summary

**Current State**:
- **3 helper functions** exist (`assert_scenarios_setup`, `assert_scenarios_run`, `assert_timeseries_processed`)
- **Limited usage**: Only 4+3+3 = 10 calls across entire test suite
- **Snakefile helpers**: Well-used (11+16 = 27 calls for rules/flags validation)
- **Most assertions**: Manual/ad-hoc patterns (poor error messages)

**Key Findings**:
1. **Good helper functions exist but are underutilized** (10 uses total)
2. **Snakefile validation helpers are well-adopted** (27 uses)
3. **Most tests use ad-hoc patterns** with poor failure messages
4. **Multi-model assertions** scattered (19 instances, no standard pattern)
5. **Path existence checks** common (18 instances) but inconsistent

**Consolidation Opportunities**:
- Expand usage of existing helpers (10 → 50+ instances)
- Create multi-model assertion helper (consolidate 19 instances)
- Standardize path existence checks (18 instances)
- Add model-specific output validation helpers

---

## 1. Assertion Pattern Categories

### 1.1 Existing Helper Functions (Phase 6c)

**Location**: `tests/utils_for_testing.py`

| Helper | Usage Count | Status | Quality |
|--------|-------------|--------|---------|
| `assert_scenarios_setup()` | 4 | ✅ Good | Clear failure messages with counts |
| `assert_scenarios_run()` | 3 | ✅ Good | Shows failed simulation indices |
| `assert_timeseries_processed()` | 3 | ✅ Good | Multi-model aware |
| `assert_snakefile_has_rules()` | 11 | ✅ Excellent | Clear missing rule list |
| `assert_snakefile_has_flags()` | 16 | ✅ Excellent | Clear missing flag list |

**Example (Good):**
```python
# Current good pattern from utils_for_testing.py
def assert_scenarios_setup(analysis, verbose=False):
    if not analysis.all_scenarios_created:
        pytest.fail(
            f"Scenario setup failed for {len(analysis.scenarios_not_created)} "
            f"of {len(analysis.df_sims)} scenarios. Run with pytest -v for details."
        )
```

**Usage Example:**
```python
# test_PC_01_singlesim.py:24
tst_ut.assert_scenarios_setup(analysis)
```

**Why This Works**:
- ✅ Clear failure message with count
- ✅ Tells you how many failed out of total
- ✅ Suggests verbose mode for details
- ✅ Optionally prints failed scenario list

### 1.2 Direct Property Assertions

**Count**: 8 instances (minimal usage)
**Pattern**: `assert analysis.property`

**Example:**
```python
# Would be: assert analysis.all_scenarios_created
# But this is RARE - most tests use helpers
```

**Problem**: Direct property checks give poor error messages
```python
assert analysis.all_scenarios_created
# Failure: AssertionError (no context!)
```

**Status**: ✅ Already mostly avoided (only 8 instances)

### 1.3 Manual Loop Assertions

**Count**: 5 loop patterns
**Pattern**: `for event_iloc in analysis.df_sims.index: ...`

**Example** (test_PC_01_singlesim.py):
```python
for event_iloc in analysis.df_sims.index:
    proc = analysis._retrieve_sim_run_processing_object(event_iloc)
    for model_type in enabled_models:
        if model_type == "tritonswmm":
            proc.write_timeseries_outputs(...)
```

**Purpose**: Process each scenario individually, not assertion per se
**Status**: ⚠️ Not really "assertions" - these are execution loops

### 1.4 Multi-Model Pattern Assertions

**Count**: 19 instances
**Pattern**: Check `model_types_enabled` / `enabled_models`

**Example** (test_PC_01_singlesim.py):
```python
enabled_models = tst_ut.get_enabled_model_types(analysis)

for event_iloc in analysis.df_sims.index:
    proc = analysis._retrieve_sim_run_processing_object(event_iloc)

    for model_type in enabled_models:
        if model_type == "tritonswmm":
            proc.write_timeseries_outputs(which="both", model_type=model_type)
        elif model_type == "triton":
            proc.write_timeseries_outputs(which="TRITON", model_type=model_type)
        elif model_type == "swmm":
            proc.write_timeseries_outputs(which="SWMM", model_type=model_type)
```

**Problem**: Repeated pattern across many tests
**Opportunity**: Create helper for multi-model output validation

### 1.5 Path/File Existence Assertions

**Count**: 18 instances
**Pattern**: `assert path.exists()`

**Examples**:
```python
# test_PC_04_multisim_with_snakemake.py:30
assert snakefile_path.exists()

# test_PILOT_platform_parametrized_workflow.py:83
assert snakefile_path.exists(), f"Snakefile not created at {snakefile_path}"
```

**Problem**: Inconsistent - some have messages, some don't
**Opportunity**: Standardize to always include path in message

### 1.6 DataFrame Status Assertions

**Count**: 1 instance
**Pattern**: `assert df_status...`

**Status**: Extremely rare pattern

---

## 2. Test File Assertion Complexity

### 2.1 Most Assertion-Heavy Files

| File | Assertions | Primary Type |
|------|------------|--------------|
| test_PC_04_multisim_with_snakemake.py | 27 | Snakefile validation |
| test_frontier_05_snakemake_sensitivity_analysis_CPU.py | 17 | Snakefile validation |
| test_UVA_03_sensitivity_analysis_with_snakemake.py | 17 | Snakefile validation |
| test_PC_05_sensitivity_analysis_with_snakemake.py | 16 | Snakefile validation |
| test_PC_01_singlesim.py | 10 | Mixed (uses helpers) |

**Observation**: Snakemake workflow tests have the most assertions, primarily using the well-designed Snakefile helpers.

### 2.2 Helper Function Adoption

**Files using assert_scenarios_setup** (4):
- test_PC_01_singlesim.py
- test_PC_02_multisim.py
- test_UVA_01_1core_multisim.py
- test_frontier_01_1core_multisim.py

**Files using assert_scenarios_run** (3):
- test_PC_01_singlesim.py
- test_PC_02_multisim.py
- test_frontier_01_1core_multisim.py

**Files using assert_timeseries_processed** (3):
- test_PC_01_singlesim.py
- test_PC_02_multisim.py
- test_frontier_01_1core_multisim.py

**Pattern**: Helper adoption is concentrated in **simulation execution tests** (PC_01, PC_02, UVA_01, frontier_01).

**Missing**: Workflow tests (PC_04, PC_05, etc.) don't use the helper pattern at all - they validate Snakefiles instead.

---

## 3. Assertion Quality Analysis

### 3.1 High-Quality Patterns (Keep/Expand)

**Snakefile Validation Helpers**: ✅ **Excellent**
```python
tst_ut.assert_snakefile_has_rules(content, ["setup", "run_simulation", ...])
tst_ut.assert_snakefile_has_flags(content, ["--compression-level 5", ...])
```

**Why Excellent**:
- Clear failure message showing missing items
- Lists exactly what's missing
- Multi-model aware (rule name variants)

**Scenario/Simulation Helpers**: ✅ **Good**
```python
tst_ut.assert_scenarios_setup(analysis)
tst_ut.assert_scenarios_run(analysis)
tst_ut.assert_timeseries_processed(analysis, which="both")
```

**Why Good**:
- Shows count of failures
- Suggests verbose mode for details
- Multi-model aware

### 3.2 Medium-Quality Patterns (Improve)

**Path Existence with Message**: ⚠️ **Inconsistent**
```python
# Good (has context):
assert snakefile_path.exists(), f"Snakefile not created at {snakefile_path}"

# Bad (no context):
assert snakefile_path.exists()
```

**Recommendation**: Standardize to always include path

### 3.3 Low-Quality Patterns (Avoid)

**Direct Property Checks**: ❌ **Poor**
```python
assert analysis.all_scenarios_created
# Failure: AssertionError
# (no indication of HOW MANY failed or WHICH scenarios)
```

**Status**: Already rare (8 instances) - good!

---

## 4. Missing Assertion Helpers

Based on patterns identified, these helpers would be valuable:

### 4.1 Multi-Model Output Validation

**Current Pattern** (repeated 19 times):
```python
enabled_models = tst_ut.get_enabled_model_types(analysis)
for model_type in enabled_models:
    if model_type == "tritonswmm":
        # validate TRITON-SWMM outputs
    elif model_type == "triton":
        # validate TRITON outputs
    elif model_type == "swmm":
        # validate SWMM outputs
```

**Proposed Helper**:
```python
def assert_model_outputs_exist(
    analysis,
    model_types: List[str] = None,  # None = auto-detect
    check_timeseries: bool = True,
    check_summaries: bool = True,
    verbose: bool = False
) -> None:
    """Assert expected outputs exist for all enabled model types.

    Checks:
    - TRITON-SWMM: TRITON.nc, SWMM_nodes.nc, SWMM_links.nc
    - TRITON-only: TRITON.nc
    - SWMM-only: SWMM_nodes.nc, SWMM_links.nc

    Raises AssertionError with details of missing outputs.
    """
```

**Usage**:
```python
# Instead of 10+ lines of if/elif checking
tst_ut.assert_model_outputs_exist(analysis)
```

### 4.2 Phase Completion Validation

**Current Gap**: No helper for checking multiple phases at once

**Proposed Helper**:
```python
def assert_phases_complete(
    analysis,
    phases: List[str] = ["setup", "preparation", "simulation", "processing"],
    verbose: bool = False
) -> None:
    """Assert specified workflow phases completed for all scenarios.

    Uses WorkflowStatus to check completion state.
    """
```

**Usage**:
```python
# Check that setup and preparation are done before testing execution
tst_ut.assert_phases_complete(analysis, phases=["setup", "preparation"])
```

### 4.3 Standardized Path Assertion

**Current Pattern** (18 instances, inconsistent):
```python
assert path.exists()  # No message
assert path.exists(), f"File not found: {path}"  # Has message
```

**Proposed Helper**:
```python
def assert_file_exists(path: Path, description: str = None) -> None:
    """Assert file exists with clear error message.

    Parameters
    ----------
    path : Path
        Path to check
    description : str, optional
        Description of what this file is (e.g., "Snakefile", "SWMM input")
    """
    if not path.exists():
        desc = f" ({description})" if description else ""
        pytest.fail(f"Expected file{desc} not found: {path}")
```

**Usage**:
```python
# Clear, consistent pattern
tst_ut.assert_file_exists(snakefile_path, "Snakefile")
```

### 4.4 Model-Specific Completion Check

**Current Gap**: No helper for checking if specific model type completed

**Proposed Helper**:
```python
def assert_model_simulations_complete(
    analysis,
    model_type: Literal["triton", "tritonswmm", "swmm"],
    verbose: bool = False
) -> None:
    """Assert all simulations completed for specific model type.

    Checks model-specific log fields for completion.
    """
```

---

## 5. Adoption Patterns

### 5.1 Why Snakefile Helpers Are Well-Adopted

**Usage**: 27 instances (11 rules + 16 flags)
**Success Factors**:
1. **Clear purpose**: Validate Snakefile content
2. **Easy to use**: One-liners with list of expectations
3. **Great failure messages**: Shows exactly what's missing
4. **Multi-model aware**: Handles model-specific rule variants

### 5.2 Why Scenario/Sim Helpers Are Underused

**Usage**: 10 instances (should be 50+)
**Likely Reasons**:
1. **Newer additions**: Added in Phase 6c, not yet widely adopted
2. **Mixed test types**: Workflow tests don't need them (validate Snakefiles, not execution)
3. **Execution focus**: Only relevant for tests that actually run simulations

**Adoption Opportunity**: Migrate existing manual checks to use helpers

---

## 6. Recommendations

### 6.1 Phase 6d.2 - Implement New Helpers

**Priority 1** (High value, clear pattern):
- `assert_model_outputs_exist()` - Consolidate 19 multi-model checks

**Priority 2** (Standardization):
- `assert_file_exists()` - Standardize 18 path existence checks

**Priority 3** (New capability):
- `assert_phases_complete()` - Leverage WorkflowStatus
- `assert_model_simulations_complete()` - Model-specific checking

### 6.2 Phase 6d.3 - Migration Strategy

**Migration Targets** (estimated):
- Multi-model patterns: 19 instances → use `assert_model_outputs_exist()`
- Path existence: 18 instances → use `assert_file_exists()`
- Phase checks: Add new usage of `assert_phases_complete()`

**Keep As-Is**:
- ✅ Snakefile helpers (already excellent, well-adopted)
- ✅ Existing scenario/sim/timeseries helpers (good quality)

**Total Migration**: ~37 assertion patterns

### 6.3 Documentation

**Add to CLAUDE.md** under "Testing" section:
```markdown
### Test Assertion Patterns

Use standardized assertion helpers for consistency:

**Workflow Phase Completion**:
```python
tst_ut.assert_scenarios_setup(analysis)      # All scenarios created
tst_ut.assert_scenarios_run(analysis)        # All simulations complete
tst_ut.assert_timeseries_processed(analysis) # All outputs processed
```

**Multi-Model Output Validation**:
```python
tst_ut.assert_model_outputs_exist(analysis)  # All enabled models have outputs
```

**Snakefile Validation**:
```python
tst_ut.assert_snakefile_has_rules(content, ["setup", "run_simulation", ...])
tst_ut.assert_snakefile_has_flags(content, ["--compression-level 5", ...])
```

**File Existence**:
```python
tst_ut.assert_file_exists(path, description="Snakefile")
```

All helpers provide verbose mode: pass `verbose=True` or run with `pytest -v`.
```

---

## 7. Risk Assessment

### 7.1 Implementation Risk

**Phase 6d.2** (Add new helpers): **LOW**
- Additive changes only
- No existing code affected
- Can be implemented incrementally

**Phase 6d.3** (Migrate existing assertions): **MEDIUM**
- Changes test logic
- Must preserve test behavior
- Can be done incrementally (migrate file by file)

### 7.2 Mitigation Strategy

1. **Implement helpers first** (Phase 6d.2) without changing existing tests
2. **Validate helpers** with new pilot tests
3. **Migrate incrementally** (Phase 6d.3):
   - Start with PC tests (most stable)
   - Keep both patterns temporarily
   - Run smoke tests after each migration
   - Only remove old pattern after validation

---

## 8. Success Metrics

**Phase 6d.2 Complete When**:
- ✅ 4 new helpers implemented
- ✅ All helpers have comprehensive docstrings
- ✅ Verbose mode supported for all helpers
- ✅ Test with pilot tests (prove they work)

**Phase 6d.3 Complete When**:
- ✅ 80%+ of assertions use standardized helpers
- ✅ All smoke tests still passing
- ✅ CLAUDE.md documents assertion patterns
- ✅ Clearer test failure messages validated

---

## 9. Timeline

**Phase 6d.2** (Implement helpers): 2-3 hours
- 4 new helper functions
- Docstrings and examples
- Optional: Pilot tests demonstrating usage

**Phase 6d.3** (Migrate assertions): 3-4 hours
- ~37 assertion patterns to migrate
- File-by-file approach
- Validation after each file

**Total Phase 6d**: 5-7 hours

---

## 10. Next Steps

1. ✅ Complete this audit (Phase 6d.1) ← **Done**
2. [ ] Get approval to proceed with Phase 6d.2
3. [ ] Implement 4 new assertion helpers
4. [ ] Add pilot tests demonstrating new helpers
5. [ ] Update CLAUDE.md with assertion guidelines
6. [ ] Migrate existing assertions incrementally (Phase 6d.3)

---

**End of Audit**
