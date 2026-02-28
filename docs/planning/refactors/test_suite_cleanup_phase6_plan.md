# Test Suite Cleanup - Phase 6 Implementation Plan

**Status**: Planning
**Created**: 2026-02-09
**Owner**: Implementation team
**Parent**: `docs/planning/cruft_cleanup_plan.md` Phase 6

---

## Context

Phase 6 of the cruft cleanup addresses test suite maintainability. Phase 6c (diagnostic prints) is complete. This plan covers the remaining phases 6a, 6b, and 6d.

**Current State**:
- 31 test files across 3 platform categories (PC/UVA/Frontier)
- Some parametrization exists (9 instances, mostly for config variations)
- Fixture factories spread across `conftest.py`, `test_case_builder.py`, `test_case_catalog.py`
- Platform detection utilities in `utils_for_testing.py`
- All smoke tests passing (PC_01 through PC_05)

**Goals**:
1. Reduce test duplication through parametrization
2. Consolidate fixture creation patterns
3. Standardize completion assertion helpers

---

## Phase 6a: Parametrize Repeated Platform Test Patterns

### Current Duplication Pattern

Many tests exist in three variants:
- `test_PC_XX_*.py` - Local machine tests
- `test_UVA_XX_*.py` - UVA HPC tests (skip if not on UVA)
- `test_frontier_XX_*.py` - Frontier tests (skip if not on Frontier)

**Example**: Snakemake workflow tests
- `test_PC_04_multisim_with_snakemake.py` (7 tests)
- `test_UVA_02_multisim_with_snakemake.py` (similar tests for UVA)
- `test_frontier_03_snakemake_multisim_CPU.py` (similar tests for Frontier)

### Proposed Approach

**Option A: Parametrize by Platform** (Recommended)
- Create unified test files with platform as a parameter
- Use `pytest.mark.skipif` with platform detection
- Consolidate platform-specific config fixtures

**Example transformation**:
```python
# Before: 3 separate files with similar tests
# test_PC_04_multisim_with_snakemake.py
def test_workflow_generation(norfolk_analysis):
    # test logic

# test_UVA_02_multisim_with_snakemake.py
@pytest.mark.skipif(not on_UVA_HPC(), reason="UVA only")
def test_workflow_generation(norfolk_analysis_uva):
    # same test logic

# After: 1 parametrized file
@pytest.mark.parametrize("platform,skip_condition", [
    ("local", lambda: False),
    ("uva", lambda: not on_UVA_HPC()),
    ("frontier", lambda: not on_frontier()),
])
def test_workflow_generation(platform, skip_condition, request):
    if skip_condition():
        pytest.skip(f"Platform {platform} not available")
    fixture_name = f"norfolk_analysis_{platform}"
    analysis = request.getfixturevalue(fixture_name)
    # test logic once
```

**Benefits**:
- Reduces test file count (31 → ~15 files)
- Makes cross-platform consistency visible
- Easier to add new platform variants
- Test logic written once, executed everywhere applicable

**Risks**:
- More complex fixture management
- May obscure platform-specific edge cases
- Requires careful parametrization of config differences

**Decision**: Start with low-risk tests (workflow generation, dry-run) before tackling execution tests.

### Implementation Steps

1. **Pilot with workflow generation tests** (Phase 6a.1)
   - Target: `test_*_multisim_with_snakemake.py` workflow generation tests
   - Create parametrized version
   - Validate all platforms still work
   - Keep originals until validated

2. **Expand to dry-run tests** (Phase 6a.2)
   - Workflow dry-run tests are safe (no actual execution)
   - Good candidates for parametrization

3. **Evaluate execution tests** (Phase 6a.3)
   - Execution tests may have legitimate platform differences
   - May keep separate or use conditional logic within parametrized tests

### Exit Criteria

- At least 5 test files consolidated via parametrization
- All smoke tests still passing
- Platform-specific tests still skip appropriately
- Test output clearly shows which platform variant failed

---

## Phase 6b: Consolidate Fixture Factories

### Current State

Fixture creation is scattered:

**`tests/conftest.py`** (38 lines)
- Defines `norfolk_multi_sim_analysis` and `norfolk_multi_sim_analysis_cached`
- Calls into `test_case_catalog.py`

**`tests/fixtures/test_case_builder.py`** (157 lines)
- `retrieve_TRITON_SWMM_test_case` class for manual test case construction
- Platform-specific config paths
- Used by catalog

**`tests/fixtures/test_case_catalog.py`** (500+ lines)
- `GetTS_TestCases` with methods like `retrieve_norfolk_multi_sim_test_case()`
- Platform detection and config selection
- `Local_TestCases`, `UVA_TestCases`, `Frontier_TestCases` subclasses

### Problems

1. **Unclear entry point**: Should tests use conftest fixtures or call catalog directly?
2. **Duplication**: Platform-specific config paths repeated across files
3. **Mixed responsibilities**: Catalog does both selection and construction
4. **Hard to extend**: Adding new test case requires touching multiple files

### Proposed Consolidation

**Single Responsibility Principle**:
- **`conftest.py`**: Declare fixtures only, delegate to catalog
- **`test_case_catalog.py`**: Select appropriate config/platform
- **`test_case_builder.py`**: Construct test case from config (no platform logic)

**Simplified flow**:
```
Test → conftest fixture → catalog.get_case(platform) → builder.build(config)
```

### Implementation Steps

1. **Audit current fixture usage** (Phase 6b.1)
   - List all test files and which fixtures they use
   - Identify duplicate fixture patterns
   - Document platform-specific config variations

2. **Create unified fixture API** (Phase 6b.2)
   - Define standard fixture names: `norfolk_analysis`, `norfolk_analysis_cached`
   - Add platform variant fixtures: `norfolk_analysis_uva`, etc.
   - All fixtures go through same catalog → builder flow

3. **Refactor catalog** (Phase 6b.3)
   - Simplify to single class with `@classmethod` for each test case
   - Platform detection happens once at method entry
   - Returns constructed test case (delegates to builder)

4. **Update all test imports** (Phase 6b.4)
   - Change direct catalog calls to use conftest fixtures
   - Remove redundant fixture definitions from test files

### Exit Criteria

- Single, clear fixture API in conftest.py
- All test files use standard fixtures
- Platform logic centralized in one place
- Adding new test case touches only 2 files (catalog + conftest)

---

## Phase 6d: Standardize Assertions Around Completion Semantics

### Current State

Completion checks are inconsistent:

**Direct property access**:
```python
assert analysis.all_scenarios_created
assert analysis.all_sims_run
```

**Helper functions** (`utils_for_testing.py`):
```python
assert_scenarios_setup(analysis)
assert_scenarios_run(analysis)
assert_timeseries_processed(analysis, which="both")
```

**Manual checks**:
```python
for event_iloc in analysis.df_sims.index:
    assert analysis.df_status.loc[event_iloc, "setup"] == True
```

### Problems

1. **Inconsistent patterns**: Some tests use helpers, others check properties directly
2. **Verbose failures**: Direct property checks give poor error messages
3. **Repeated logic**: Multi-model checking logic duplicated across tests
4. **Hard to debug**: Failure doesn't indicate *which* scenario failed

### Proposed Standardization

**Create assertion helpers for common patterns**:

```python
# New helpers in utils_for_testing.py

def assert_all_phases_complete(
    analysis,
    phases: List[str] = ["setup", "preparation", "simulation", "processing"],
    verbose: bool = False
) -> None:
    """Assert all specified phases complete for all scenarios.

    Raises AssertionError with details of incomplete phases/scenarios.
    """
    # Implementation using WorkflowStatus

def assert_model_outputs_exist(
    analysis,
    model_types: List[str] = None,  # Auto-detect if None
    check_timeseries: bool = True,
    check_summaries: bool = True,
    verbose: bool = False
) -> None:
    """Assert expected outputs exist for all enabled model types."""
    # Check log fields + file existence

def assert_no_failed_scenarios(analysis, verbose: bool = False) -> None:
    """Assert no scenarios have failed status."""
    # Check df_status for failures

def assert_completion_matches_log(scenario, model_type: str) -> None:
    """Assert log completion matches actual output files."""
    # Verify log-based vs file-existence consistency
```

### Benefits

- **Better error messages**: Shows exactly which scenario/phase/model failed
- **Consistent interface**: All tests use same assertion style
- **Less duplication**: Common logic in one place
- **Easier debugging**: Verbose mode shows progress

### Implementation Steps

1. **Audit current assertion patterns** (Phase 6d.1)
   - Grep for all completion checks in tests
   - Categorize by pattern (property, helper, manual)
   - Identify most common patterns

2. **Implement new helpers** (Phase 6d.2)
   - Start with most common patterns
   - Add verbose mode for debugging
   - Include clear failure messages with counts

3. **Migrate tests incrementally** (Phase 6d.3)
   - Start with PC tests (most stable)
   - Update platform tests after parametrization (Phase 6a complete)
   - Keep old assertions alongside new until validated

4. **Document assertion conventions** (Phase 6d.4)
   - Add docstring examples to helpers
   - Update CLAUDE.md with test assertion guidelines

### Exit Criteria

- 80%+ of tests use standardized assertion helpers
- All smoke tests passing with new assertions
- Clear, actionable failure messages
- Documented test assertion patterns in CLAUDE.md

---

## Risk Mitigation

**Test breakage risk**: HIGH
- Mitigation: Work incrementally, run smoke tests after each change
- Mitigation: Keep original test files until parametrized versions validated
- Mitigation: Start with low-risk tests (workflow generation, dry-run)

**Platform test coverage loss**: MEDIUM
- Mitigation: Ensure skip conditions preserve existing skip behavior
- Mitigation: Test on available platforms before removing originals

**Fixture confusion**: MEDIUM
- Mitigation: Clear naming convention for platform-specific fixtures
- Mitigation: Document fixture usage in conftest.py docstrings

---

## Implementation Order

Recommended sequence to minimize risk:

1. **Phase 6b.1**: Audit fixture usage (safe, information gathering)
2. **Phase 6a.1**: Pilot parametrization with workflow generation tests
3. **Phase 6d.1**: Audit assertion patterns (safe, information gathering)
4. **Phase 6d.2**: Implement new assertion helpers (additive, no breaking changes)
5. **Phase 6b.2-4**: Consolidate fixtures (after understanding usage patterns)
6. **Phase 6a.2-3**: Expand parametrization (after fixture consolidation)
7. **Phase 6d.3**: Migrate to new assertions (final step, after structure stable)

---

## Timeline Estimate

- **Phase 6a**: 3-5 days (pilot + expansion + validation)
- **Phase 6b**: 2-3 days (audit + refactor + migrate)
- **Phase 6d**: 2-3 days (implement helpers + migrate tests)
- **Total**: 7-11 days (part-time equivalent)

---

## Success Metrics

- Test file count reduced by ~30% (31 → ~22 files)
- Fixture creation centralized to 2 files (conftest + catalog)
- 80%+ of tests use standardized assertions
- All smoke tests passing (PC_01 through PC_05)
- No loss of platform test coverage
- Clearer test failure messages

---

## Next Steps

1. ✅ Create this plan document
2. [ ] Get approval to proceed with Phase 6a.1 (pilot parametrization)
3. [ ] Implement Phase 6b.1 (audit fixture usage)
4. [ ] Execute phases in recommended order
5. [ ] Update `cruft_cleanup_tracker.md` as phases complete

---

## References

- `docs/planning/cruft_cleanup_plan.md` - Parent plan document
- `docs/planning/cruft_cleanup_tracker.md` - Progress tracker
- `tests/utils_for_testing.py` - Current test utilities
- `tests/conftest.py` - Current fixture definitions
- `tests/fixtures/test_case_catalog.py` - Test case selection logic
