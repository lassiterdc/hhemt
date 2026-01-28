# Test Refactor Assessment + Multi-Phase Plan

## Context
The current `tests/test_PC_*` suite contains repeated setup logic (retrieving analysis objects, generating Snakefiles, checking logs) and duplicated assertion blocks. This increases maintenance cost and risks divergence in expected behaviors. The same patterns appear in other test groups (Frontier/UVA), so a phased refactor should start with PC-prefixed tests and then generalize.

## Assessment of Duplication (PC Tests)
**Repeated setup patterns**
- `GetTS_TestCases` calls with `start_from_scratch` in nearly every test
- `analysis = <case>.system.analysis` repeated verbatim
- Snakemake Snakefile generation repeated with small parameter changes

**Repeated assertion patterns**
- “Snakefile contains rules/flags” blocks repeated across tests
- “Timeseries processed / summaries created” checks repeated with the same log flags
- “Local mode detection” repeated in multiple modules

**Boilerplate file IO**
- Writing Snakefile to disk + re-reading
- Creating logs/status directories for dry-run tests

## Goals
- Reduce duplicated code in PC tests
- Preserve or strengthen assertions (e.g., more consistent Snakefile validation)
- Introduce reusable fixtures + helper assertions to make future tests cheaper to write
- Final phase: apply the same refactor patterns to non-PC tests

---

## Phase 1 (PC-prefixed tests) — Refactor + Consolidate

### Deliverables
1. **Shared fixtures** in `tests/conftest.py`
   - `norfolk_single_sim_analysis`
   - `norfolk_multi_sim_analysis`
   - `norfolk_sensitivity_analysis`
   - Optional callable/parametrized fixtures for `start_from_scratch`

2. **Helper utilities** in `tests/utils_for_testing.py`
   - `assert_snakefile_has_rules(content, rules)`
   - `assert_snakefile_has_flags(content, flags)`
   - `assert_timeseries_processed(analysis, which = "both")`
   - `assert_summaries_created(analysis, which="both")`
   - `write_snakefile(analysis, content)`

3. **Parametrized tests** for Snakefile configurations
   - Replace repeated `generate_snakefile_content` blocks with `pytest.mark.parametrize`

4. **Refactored PC tests**
   - `test_PC_01_singlesim.py`
   - `test_PC_02_multisim.py`
   - `test_PC_04_multisim_with_snakemake.py`
   - `test_PC_05_sensitivity_analysis_with_snakemake.py`

### Success Criteria
- All `test_PC_*` tests pass unchanged
- No loss of assertion strength (ideally stronger via shared checks)
- Boilerplate reduced substantially and new tests are shorter to write

### Suggested Test Command
```bash
pytest -k "test_PC" -v
```

---

## Phase 2 (Non-PC tests) — Apply the Same Patterns

### Targets
- `tests/test_frontier_*`
- `tests/test_UVA_*`
- Any other non-PC suites with similar setup/assertion duplication

### Actions
- Reuse Phase 1 fixtures + helpers
- Replace duplicated setup and assertions with shared helpers
- Introduce parametrized variants where meaningful (workflow config, dry-run, submission checks)

### Success Criteria
- Non-PC tests still pass in their respective environments
- 50%+ reduction in repeated setup/assert blocks

---

## Risks + Mitigations
- **Risk:** Changing fixtures might alter test execution order or side effects
  - **Mitigation:** Keep fixture scopes function-level and preserve `start_from_scratch` semantics
- **Risk:** Over-generalized helpers obscure failures
  - **Mitigation:** Helper assertions should print logs and include clear failure messages

---

## Files Affected (Phase 1)
| File | Planned Changes |
|------|-----------------|
| `tests/conftest.py` | Add shared fixtures |
| `tests/utils_for_testing.py` | Add shared assertion helpers |
| `tests/test_PC_01_singlesim.py` | Replace repeated setup/assert blocks |
| `tests/test_PC_02_multisim.py` | Replace repeated setup/assert blocks |
| `tests/test_PC_04_multisim_with_snakemake.py` | Parametrize Snakefile checks + helpers |
| `tests/test_PC_05_sensitivity_analysis_with_snakemake.py` | Parametrize + helpers |

## Files Affected (Phase 2)
| File Group | Planned Changes |
|------------|-----------------|
| `tests/test_frontier_*` | Reuse fixtures + helpers |
| `tests/test_UVA_*` | Reuse fixtures + helpers |
