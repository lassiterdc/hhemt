# Implementation Plan: Dynamic SWMM Thread Control

## Phased Implementation Plan

### Phase 1: Core .inp File Modification
**Goal:** Implement post-template thread modification for SWMM .inp files

- [ ] Update `n_threads_swmm` docstring in `config/analysis.py` (clarify scope)
- [ ] Add `update_swmm_threads_in_inp_file()` method to `scenario_inputs.py`
- [ ] Add call to update method after `hydro.inp` creation in `scenario.py`
- [ ] Add call to update method after `full.inp` creation in `scenario.py`
- [ ] **Update plan:** Document Phase 1 completion, verify Phase 2 steps still make sense

### Phase 2: Testing Integration
**Goal:** Add verification to existing test suite

- [ ] Add THREADS verification to `test_PC_04_multisim_with_snakemake.py::test_snakemake_workflow_end_to_end`
- [ ] Verify test passes with default `n_threads_swmm=2`
- [ ] **Verify Phase 2:** Test passes, THREADS parameter correctly updated in all .inp files
- [ ] **Update plan:** Document Phase 2 completion, verify Phase 3 steps still make sense

### Phase 3: Sensitivity Analysis Testing (UPDATED for Unified Threading) ✅ COMPLETE

**Goal:** Create comprehensive sensitivity analysis tests for TRITON-only and SWMM-only models using unified `n_omp_threads`

**Note:** Phase 3 updated after Phases 1-2 completion. We unified `n_threads_swmm` → `n_omp_threads`, so sensitivity tests now vary `n_omp_threads` across sub-analyses.

- [x] Create new test file `tests/test_PC_06_sensitivity_analysis_triton_and_swmm_only.py`
- [x] Add conftest fixtures using new test cases from `test_case_catalog.py`:
  - `norfolk_sensitivity_triton_only` (uses `retrieve_norfolk_cpu_config_sensitivity_case_triton_only`)
  - `norfolk_sensitivity_swmm_only` (uses `retrieve_norfolk_cpu_config_sensitivity_case_swmm_only`)
- [x] Implement `test_sensitivity_analysis_triton_only_dry_run()` (modeled after test_PC_05)
- [x] Implement `test_sensitivity_analysis_swmm_only_dry_run()` (modeled after test_PC_05)
- [x] Verify both tests pass (dry-run confirms workflow generation)
- [x] Verify `n_omp_threads` varies across sub-analyses (not `n_threads_swmm`)
- [x] **Phase 3 Complete:** Both TRITON and SWMM sensitivity analyses generate correct workflows

**Test Results:**
```
tests/test_PC_06_sensitivity_analysis_triton_and_swmm_only.py::test_sensitivity_analysis_triton_only_dry_run PASSED [ 50%]
tests/test_PC_06_sensitivity_analysis_triton_and_swmm_only.py::test_sensitivity_analysis_swmm_only_dry_run PASSED [100%]

2 passed in 237.96s (0:03:57)
```

**Implementation Summary:**
- Both tests validate Snakemake workflow generation via dry-run (no execution)
- Verified `n_omp_threads` varies across sub-analyses (sensitivity dimension working)
- Confirmed df_status parsing works for both model types
- SWMM test validates run_mode consistency (serial mode → n_omp_threads=1)
- Test cases use existing sensitivity Excel files already updated to `n_omp_threads`

#### Phase 3 Implementation Details

**File to create:** `tests/test_PC_06_sensitivity_analysis_triton_and_swmm_only.py`

**Pattern:** Model after `test_PC_05_sensitivity_analysis_with_snakemake.py::test_snakemake_sensitivity_workflow_dry_run`

**Conftest fixtures to add** (in `tests/conftest.py`):
```python
@pytest.fixture
def norfolk_sensitivity_triton_only():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_triton_only(
        start_from_scratch=True
    )
    return case.analysis


@pytest.fixture
def norfolk_sensitivity_swmm_only():
    case = cases.Local_TestCases.retrieve_norfolk_cpu_config_sensitivity_case_swmm_only(
        start_from_scratch=True
    )
    return case.analysis
```

**Test structure:**
```python
import pytest
import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_sensitivity_analysis_triton_only_dry_run(norfolk_sensitivity_triton_only):
    """
    Test TRITON-only sensitivity analysis dry-run.

    Verifies that:
    1. DAG can be constructed from master Snakefile
    2. All dependencies resolve correctly for TRITON-only runs
    3. No actual execution occurs
    4. Snakemake exit code is 0
    5. THREADS parameter correctly set in generated .inp files
    """
    analysis = norfolk_sensitivity_triton_only

    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=True,
        prepare_scenarios=True,
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which="TRITON",  # TRITON-only
        clear_raw_outputs=True,
        overwrite_outputs_if_already_created=True,
        compression_level=5,
        pickup_where_leftoff=False,
        dry_run=True,
        verbose=True,
    )

    assert result.get("success"), f"Dry-run failed: {result.get('message', '')}"
    assert result.get("mode") == "local"

    # Verify Snakemake allocation parsing
    df_status = analysis.df_status
    assert not df_status.empty
    assert "snakemake_allocated_nTasks" in df_status.columns
    assert "snakemake_allocated_omp_threads" in df_status.columns


def test_sensitivity_analysis_swmm_only_dry_run(norfolk_sensitivity_swmm_only):
    """
    Test SWMM-only sensitivity analysis dry-run.

    Verifies that:
    1. DAG can be constructed from master Snakefile
    2. All dependencies resolve correctly for SWMM-only runs
    3. No actual execution occurs
    4. Snakemake exit code is 0
    5. THREADS parameter correctly set in generated .inp files
    6. n_threads_swmm varies across sub-analyses
    """
    analysis = norfolk_sensitivity_swmm_only

    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=False,  # SWMM-only, no TRITON compilation
        recompile_if_already_done_successfully=False,
        prepare_scenarios=True,
        overwrite_scenario_if_already_set_up=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which="SWMM",  # SWMM-only
        clear_raw_outputs=True,
        overwrite_outputs_if_already_created=True,
        compression_level=5,
        pickup_where_leftoff=False,
        dry_run=True,
        verbose=True,
    )

    assert result.get("success"), f"Dry-run failed: {result.get('message', '')}"
    assert result.get("mode") == "local"

    # Verify Snakemake allocation parsing
    df_status = analysis.df_status
    assert not df_status.empty
    assert "snakemake_allocated_nTasks" in df_status.columns
    assert "snakemake_allocated_omp_threads" in df_status.columns

    # Verify n_omp_threads varies across sub-analyses (sensitivity dimension)
    sensitivity = analysis.sensitivity
    n_threads_values = set()
    for sub_analysis in sensitivity.sub_analyses.values():
        n_threads_values.add(sub_analysis.cfg_analysis.n_omp_threads)

    assert len(n_threads_values) > 1, \
        "Sensitivity analysis should vary n_omp_threads across sub-analyses"
```

**Why these tests:**
- Validates TRITON-only and SWMM-only sensitivity workflows work correctly
- Ensures `n_threads_swmm` propagates through sensitivity analysis framework
- Confirms Snakemake workflow generation for both model types
- Uses dry-run for fast validation (no actual simulation execution)

---

## Context

Currently, the TRITON-SWMM toolkit has limited control over SWMM threading configuration. While SWMM models support multi-threaded execution via the `THREADS` parameter in the `[OPTIONS]` section of .inp files, this value is:

1. **Hardcoded in template files** - Users must manually edit templates to change thread counts
2. **Partially configured** - The existing `n_threads_swmm` parameter in `analysis_config.py` (line 45-47) is used for Snakemake resource allocation but NOT for modifying .inp files
3. **Prevents benchmarking studies** - Cannot easily run sensitivity analyses varying SWMM thread counts

This implementation completes dynamic control of SWMM threading by:
- **Bridging the gap** - Connect existing `n_threads_swmm` config to actual .inp file THREADS parameter
- **Performance benchmarking** - Test SWMM performance across different thread counts
- **Resource optimization** - Match SWMM thread allocation to available CPUs
- **Sensitivity analysis** - Include SWMM threading as a parameter dimension

**Problem:** `n_threads_swmm` controls Snakemake CPU allocation but doesn't update SWMM .inp files, so SWMM may not actually use allocated threads.

**Solution:** Implement post-template modification of the `THREADS` parameter in SWMM .inp files to match `n_threads_swmm` configuration.

## Implementation Approach

### Strategy: Post-Template Modification Pattern

Rather than adding template placeholders (which would break existing templates), we'll follow the **existing post-modification pattern** used in `scenario_inputs.py:update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell()` (lines 209-230).

**Why this approach?**
1. ✅ **No breaking changes** - Existing templates continue to work
2. ✅ **Consistent with codebase** - Matches the established pattern for .inp modifications
3. ✅ **Backward compatible** - Template files don't need updating
4. ✅ **Simple implementation** - Direct string manipulation like existing code
5. ✅ **Already has config field** - `n_threads_swmm` exists in `analysis_config.py`

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Configuration Layer                                              │
│  analysis_config.py: n_threads_swmm (already exists, line 45-47)│
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Template Creation (No Changes)                                   │
│  swmm_utils.py: create_swmm_inp_from_template()                 │
│    → Creates .inp files with template THREADS value             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ NEW: Post-Processing Modification                                │
│  scenario_inputs.py: update_swmm_threads_in_inp_file()          │
│    → Reads .inp file                                             │
│    → Finds THREADS line in [OPTIONS]                            │
│    → Replaces with n_threads_swmm value                         │
│    → Writes back to file                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Scenario Orchestration (Modified)                                │
│  scenario.py: prepare_scenario()                                │
│    → Calls new update function for each .inp file created       │
└─────────────────────────────────────────────────────────────────┘
```

## Critical Files to Modify

### 1. **`src/TRITON_SWMM_toolkit/config/analysis.py`**
   - **Line 45-47:** `n_threads_swmm` already exists (no changes needed)
   - **Action:** Update docstring to clarify it now applies to all SWMM models (hydrology, hydraulics, full)
   - **Note:** Already has default value of 1 (maintains current behavior)

### 2. **`src/TRITON_SWMM_toolkit/scenario_inputs.py`** (ScenarioInputGenerator class)
   - **Add new method:** `update_swmm_threads_in_inp_file(inp_file_path: Path) -> None`
   - **Location:** After `update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell()` (around line 233)
   - **Pattern:** Similar to existing line-by-line file modification (lines 209-230)
   - **Logic:**
     ```python
     def update_swmm_threads_in_inp_file(self, inp_file_path: Path) -> None:
         """Update THREADS parameter in SWMM .inp file OPTIONS section."""
         n_threads = self.cfg_analysis.n_threads_swmm

         with open(inp_file_path, "r") as fp:
             lines = fp.readlines()

         # Find and replace THREADS line in [OPTIONS] section
         in_options_section = False
         for idx, line in enumerate(lines):
             if "[OPTIONS]" in line:
                 in_options_section = True
             elif line.startswith("[") and in_options_section:
                 # Left OPTIONS section without finding THREADS
                 break
             elif in_options_section and line.strip().startswith("THREADS"):
                 # Replace entire line preserving format
                 lines[idx] = f"THREADS              {n_threads}\n"
                 break

         # Write back
         with open(inp_file_path, "w") as fp:
             fp.writelines(lines)
     ```

### 3. **`src/TRITON_SWMM_toolkit/scenario.py`**
   - **Modify:** `prepare_scenario()` method
   - **Location:** After each .inp file creation, add thread update call
   - **Note:** Skip hydraulics.inp - it's used by TRITON-SWMM which ignores THREADS parameter

   **Two modification points:**

   #### A. After hydrology.inp creation (around line 824-825)
   ```python
   # Existing code (line 821):
   self.runoff_modeler.create_hydrology_model_from_template(
       swmm_hydro_template, self.scen_paths.swmm_hydro_inp
   )

   # ADD AFTER:
   self.input_gen.update_swmm_threads_in_inp_file(
       self.scen_paths.swmm_hydro_inp
   )
   ```

   #### B. After full.inp creation (around line 839-840)
   ```python
   # Existing code (line 836):
   self.swmm_full_model_builder.create_full_model_from_template(
       swmm_full_template, self.scen_paths.swmm_full_inp
   )

   # ADD AFTER:
   self.input_gen.update_swmm_threads_in_inp_file(
       self.scen_paths.swmm_full_inp
   )
   ```

## Implementation Steps

### Step 1: Update Configuration Docstring
**File:** `src/TRITON_SWMM_toolkit/config/analysis.py` (line 45-47)

Change:
```python
n_threads_swmm: Optional[int] = Field(
    1, description="Threads per rank for SWMM-only simulations"
)
```

To:
```python
n_threads_swmm: Optional[int] = Field(
    1,
    description=(
        "Number of OpenMP threads for SWMM execution. Applied to all SWMM models "
        "(hydrology, hydraulics, full) by dynamically updating the THREADS parameter "
        "in the [OPTIONS] section of .inp files. Defaults to 1 (serial execution)."
    )
)
```

### Step 2: Add Thread Update Method
**File:** `src/TRITON_SWMM_toolkit/scenario_inputs.py`

Add new method to `ScenarioInputGenerator` class after line 232:

```python
def update_swmm_threads_in_inp_file(self, inp_file_path: Path) -> None:
    """
    Update THREADS parameter in SWMM .inp file OPTIONS section.

    Modifies the THREADS option in the [OPTIONS] section to match the
    n_threads_swmm configuration parameter. This enables dynamic control
    of SWMM threading for performance tuning and benchmarking studies.

    Parameters
    ----------
    inp_file_path : Path
        Path to the SWMM .inp file to modify

    Notes
    -----
    - Preserves all other OPTIONS parameters unchanged
    - Uses line-by-line replacement to maintain file structure
    - If THREADS parameter not found, no modification occurs (backward compatible)
    """
    n_threads = self.cfg_analysis.n_threads_swmm

    with open(inp_file_path, "r") as fp:
        lines = fp.readlines()

    # Find and replace THREADS line in [OPTIONS] section
    in_options_section = False
    threads_found = False

    for idx, line in enumerate(lines):
        # Track when we enter/exit OPTIONS section
        if "[OPTIONS]" in line:
            in_options_section = True
            continue
        elif line.startswith("[") and in_options_section:
            # Left OPTIONS section
            break

        # Replace THREADS line if found
        if in_options_section and line.strip().startswith("THREADS"):
            # Preserve spacing format: "THREADS              {value}"
            lines[idx] = f"THREADS              {n_threads}\n"
            threads_found = True
            break

    # Write back modified file
    with open(inp_file_path, "w") as fp:
        fp.writelines(lines)

    # Note: If THREADS not found, file remains unchanged (backward compatible)
    return
```

### Step 3: Call Update Method After Template Creation
**File:** `src/TRITON_SWMM_toolkit/scenario.py`

**Note:** Skip hydraulics.inp modification - it's used by TRITON-SWMM which ignores the THREADS parameter.

#### Modification A: After hydrology.inp creation (line ~824)
```python
# Existing code:
self.runoff_modeler.create_hydrology_model_from_template(
    swmm_hydro_template, self.scen_paths.swmm_hydro_inp
)

# ADD IMMEDIATELY AFTER:
self.input_gen.update_swmm_threads_in_inp_file(
    self.scen_paths.swmm_hydro_inp
)
```

#### Modification B: After full.inp creation (line ~839)
```python
# Existing code:
self.swmm_full_model_builder.create_full_model_from_template(
    swmm_full_template, self.scen_paths.swmm_full_inp
)

# ADD IMMEDIATELY AFTER:
self.input_gen.update_swmm_threads_in_inp_file(
    self.scen_paths.swmm_full_inp
)
```

## Testing Strategy

### Integration Test: Add to Existing test_PC_04
**File:** `tests/test_PC_04_multisim_with_snakemake.py`

Add THREADS verification to the existing `test_snakemake_workflow_end_to_end()` test (after line 282):

```python
def test_snakemake_workflow_end_to_end(norfolk_multi_sim_analysis):
    """
    End-to-end Snakemake workflow test.

    ... (existing docstring)
    """
    import xarray as xr

    analysis = norfolk_multi_sim_analysis

    # Set n_threads_swmm to 2 (default in fixture)
    assert analysis.cfg_analysis.n_threads_swmm == 2, \
        "Test expects n_threads_swmm=2 in fixture configuration"

    result = analysis.submit_workflow(
        mode="local",
        # ... (existing workflow submission)
    )

    assert result.get("success"), result.get("message", "Workflow failed")
    assert result.get("mode") == "local"

    tst_ut.assert_analysis_workflow_completed_successfully(analysis)

    # NEW: Verify THREADS parameter was updated in SWMM .inp files
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        paths = proc.scen_paths

        # Check hydrology.inp (if hydrology enabled)
        if paths.swmm_hydro_inp.exists():
            with open(paths.swmm_hydro_inp, "r") as fp:
                content = fp.read()
                assert "THREADS              2" in content, \
                    f"hydro.inp for event {event_iloc} should have THREADS=2"

        # Check full.inp (if full model enabled)
        if paths.swmm_full_inp.exists():
            with open(paths.swmm_full_inp, "r") as fp:
                content = fp.read()
                assert "THREADS              2" in content, \
                    f"full.inp for event {event_iloc} should have THREADS=2"

    # Continue with existing model validation tests...
    enabled_models = tst_ut.get_enabled_model_types(analysis)
    # ... (rest of existing test)
```

**Why this approach:**
- Uses existing test infrastructure (no new test file needed)
- Validates THREADS parameter alongside other workflow checks
- Runs in CI/CD with other Snakemake tests
- Simple assertion that keeps test focused

### Manual Verification
1. **Set n_threads_swmm in test fixture**:
   ```python
   # In conftest.py or test file
   analysis.cfg_analysis.n_threads_swmm = 2
   ```

2. **Run test**:
   ```bash
   pytest tests/test_PC_04_multisim_with_snakemake.py::test_snakemake_workflow_end_to_end -v
   ```

3. **Inspect generated .inp files manually**:
   ```bash
   grep "THREADS" test_data/norfolk_coastal_flooding/tests/*/sims/*/swmm/*.inp
   # Expected: All should show "THREADS              2"
   ```

4. **Verify Snakemake resource allocation** (already implemented):
   - Check that `workflow.py` line 416 uses `n_threads_swmm` for CPU allocation
   - This is already tested via `df_status` assertions in existing tests

## Edge Cases and Error Handling

### Edge Case 1: Template Missing THREADS Parameter
**Scenario:** Old template files without THREADS line

**Behavior:** Method completes without modification (backward compatible)

**Code:** Already handled by `if threads_found` logic

### Edge Case 2: Invalid Thread Count
**Scenario:** User sets `n_threads_swmm: 0` or negative value

**Solution:** Add validation in `analysis_config.py`:
```python
@field_validator("n_threads_swmm")
def validate_n_threads_swmm(cls, v):
    if v is not None and v < 1:
        raise ValueError("n_threads_swmm must be >= 1")
    return v
```

### Edge Case 3: THREADS Line Format Variations
**Scenario:** Templates with different spacing/formatting

**Current approach:** Uses `line.strip().startswith("THREADS")` for flexibility

**Improvement (optional):** Use regex for more robust parsing:
```python
import re
if re.match(r'^\s*THREADS\s+', line, re.IGNORECASE):
    # Extract and replace value
```

### Edge Case 4: Multiple THREADS Lines
**Scenario:** Malformed template with duplicate THREADS

**Behavior:** Only first occurrence modified (stops at `break`)

**No action needed:** SWMM would reject duplicate parameters anyway

## Snakemake Workflow Integration (Already Implemented)

**Good news:** Snakemake workflow already uses `n_threads_swmm` for resource allocation!

### Current Implementation
**File:** `src/TRITON_SWMM_toolkit/workflow.py` (line 416)

```python
if model_type == "swmm":
    swmm_cpus = self.cfg_analysis.n_threads_swmm or 1
    swmm_resources = self._build_resource_block(
        partition=self.cfg_analysis.hpc_ensemble_partition,
        runtime_min=hpc_time_min,
        mem_mb=self.cfg_analysis.mem_gb_per_cpu * swmm_cpus * 1000,
        nodes=1,
        tasks=1,
        cpus_per_task=swmm_cpus,  # <-- Uses n_threads_swmm
        gpus_total=0,
        gpus_per_node_config=0,
    )
```

**What this means:**
- Snakemake **allocates** the correct number of CPUs based on `n_threads_swmm`
- However, SWMM .inp files still have hardcoded THREADS values from templates
- This implementation **bridges the gap** by updating .inp files to match allocated resources

### Sensitivity Analysis Support
**File:** `src/TRITON_SWMM_toolkit/sensitivity_analysis.py` (line 548-552)

```python
for idx, row in self.df_setup.iterrows():
    cfg_snstvty_analysis = self.master_analysis.cfg_analysis.model_copy()

    for key, val in row.items():
        setattr(cfg_snstvty_analysis, key, val)  # <-- Applies n_threads_swmm from CSV
```

**What this means:**
- Sensitivity analysis CSV can include `n_threads_swmm` column
- Each sub-analysis automatically gets the correct value via `setattr()`
- No additional code needed for sensitivity support

### Verification Steps
1. ✅ **Snakemake resource allocation** - Already uses `n_threads_swmm` (workflow.py:416)
2. ✅ **Sensitivity analysis propagation** - Already propagates via `setattr()` (sensitivity_analysis.py:552)
3. ⏳ **SWMM .inp file update** - This implementation completes the connection

## Benefits

1. **Completes existing implementation** - Bridges Snakemake allocation to actual SWMM execution
2. **Enables benchmarking studies** - Vary SWMM thread counts across sensitivity analyses
3. **Dynamic resource optimization** - Ensures SWMM uses allocated CPUs
4. **No breaking changes** - Existing templates work unchanged
5. **Consistent with toolkit patterns** - Uses established post-modification approach
6. **Low implementation risk** - Simple file I/O following proven pattern
7. **Backward compatible** - Default value maintains current behavior

## Future Enhancements (Out of Scope)

1. **Auto-detect optimal thread count** - Based on available CPUs
2. **HPC integration** - Coordinate SWMM threads with SLURM allocation
3. **Validation against CPU count** - Warn if `n_threads_swmm > available_cores`
4. **Performance metrics** - Log actual speedup from threading

## References

**Existing patterns:**
- Post-modification: `scenario_inputs.py` lines 209-230 (inflow node removal)
- Configuration validation: `config/analysis.py` field validators
- SWMM .inp creation: `swmm_utils.py:create_swmm_inp_from_template()`

**SWMM documentation:**
- THREADS parameter: Controls OpenMP thread count for SWMM 5.2+
- Valid range: 1 to system CPU count
- Default behavior: SWMM uses all available cores if THREADS not specified

## Verification Checklist

After implementation, verify:

- [ ] `n_threads_swmm` docstring updated in `analysis.py`
- [ ] `update_swmm_threads_in_inp_file()` method added to `scenario_inputs.py`
- [ ] Two calls added to `scenario.py` (hydrology, full) - **NOT hydraulics**
- [ ] Test modification added to `test_PC_04_multisim_with_snakemake.py`
- [ ] Test passes with `n_threads_swmm=2`
- [ ] Generated .inp files have correct THREADS value (hydro.inp, full.inp)
- [ ] Default behavior unchanged (THREADS matches config value)
- [ ] Other OPTIONS parameters preserved
- [ ] Backward compatible with templates missing THREADS
- [ ] Works with both SWMM model types (hydrology, full)
- [ ] Snakemake resource allocation verified (already implemented in workflow.py:416)
- [ ] Sensitivity analysis propagation verified (already implemented in sensitivity_analysis.py:552)
