# Multi-Model Output Processing Implementation Plan

**Status:** In Progress
**Started:** 2026-02-05
**Last Updated:** 2026-02-05 (Phases 1-3 ~90% ✅ | SWMM-only outputs pending)

## Overview

This document tracks the implementation of multi-model output processing for the TRITON-SWMM toolkit. The goal is to enable proper processing of outputs from all three model types:
1. **TRITON-only** (standalone 2D hydrodynamic)
2. **TRITON-SWMM** (coupled 2D surface + 1D drainage)
3. **SWMM-only** (standalone EPA SWMM)

## Context

**Problem:** Output paths were defined for all three model types in `paths.py`, but processing infrastructure (`process_simulation.py`) only handles TRITON-SWMM coupled model outputs.

**Gap Analysis:**
- ✅ TRITON-SWMM: 8/8 paths used, full processing implemented
- ❌ TRITON-only: 0/4 paths used, no processing implemented
- ⚠️ SWMM-only: 2/4 paths used (timeseries extraction works, summaries missing)

## Implementation Phases

### Phase 1: Add Model-Type Detection Infrastructure ✅ COMPLETED

**Objective:** Add runtime detection of enabled model types

#### 1.1 Add `model_types_enabled` Property
**File:** `src/TRITON_SWMM_toolkit/run_simulation.py`
**Status:** ✅ Completed

**Location:** After line 64 (after `performance_file` property)

```python
@property
def model_types_enabled(self):
    """Return list of enabled model types for this scenario.

    Returns:
        List of strings: ['triton_only', 'tritonswmm', 'swmm_only']
    """
    sys_cfg = self._scenario._system.cfg_system
    enabled = []
    if sys_cfg.toggle_triton_model:
        enabled.append('triton_only')
    if sys_cfg.toggle_tritonswmm_model:
        enabled.append('tritonswmm')
    if sys_cfg.toggle_swmm_model:
        enabled.append('swmm_only')
    return enabled
```

**Acceptance Criteria:**
- [x] Correctly reflects system config toggles
- [x] Returns list of enabled model type strings
- [ ] Used throughout processing code for routing decisions (Phase 3+)

**Note:** Initial plan incorrectly included `raw_swmm_output` property, which doesn't make sense in multi-model framework. Processing methods should directly use explicit paths from `scen_paths` based on model type being processed.

---

### Phase 2: Fix Type Annotations & Model-Aware Methods ✅ COMPLETED

**File:** `src/TRITON_SWMM_toolkit/process_simulation.py`
**Status:** ✅ Completed

#### 2.1 Fix `_export_SWMM_outputs()` Type Annotation
**Change (line 406):** Fixed `model=Literal[...]` → `model: Literal[...]`

#### 2.2 Update Call Site (line 128-133)
**Change:** Added explicit `model="tritonswmm"` parameter to call in `_process_simulation_outputs()`

#### 2.3 Make `_clear_raw_SWMM_outputs()` Model-Aware (line 602)
**Change:** Added `model: Literal["swmm", "tritonswmm"]` parameter to avoid silent failures
- Uses `scen_paths.swmm_full_out_file` for standalone SWMM
- Uses `scen_paths.swmm_hydraulics_rpt` for TRITON-SWMM coupled
- Updated both call sites (lines 448, 480) to pass `model` parameter

#### 2.4 Fix Typo in Error Message (line 239)
**Change:** `"output_tritonswmm_performance_timeserie"` → `"output_tritonswmm_performance_timeseries"`

#### 2.5 Fix Missing Directory Creation (scenario.py, line 45)
**Change:** Added `processed_output_folder.mkdir(parents=True, exist_ok=True)` after defining the folder

#### 2.6 Fix SWMM Hydraulics Output Path (scenario.py, line 72)
**Change:** Updated `swmm_hydraulics_rpt` to point to `sim_folder / "output" / "swmm" / "hydraulics.rpt"`
- TRITON-SWMM writes SWMM outputs to `output/swmm/`, not `swmm/` directory

**Acceptance Criteria:**
- [x] Method signatures are valid Python
- [x] Type checking passes for updated methods
- [x] No default model assumptions - explicit model parameter required
- [x] All smoke tests pass (test_PC_01_singlesim.py: 4/4 ✅)

---

### Phase 3: Multi-Model Output Processing Router ✅ MOSTLY COMPLETE

**File:** `src/TRITON_SWMM_toolkit/process_simulation.py`
**Status:** ✅ 90% Complete (SWMM-only pending debug)

#### 3.1 Add `_export_TRITON_only_outputs()` Method
**Location:** After line ~402

**Implementation:**
```python
def _export_TRITON_only_outputs(
    self,
    overwrite_if_exist: bool = False,
    clear_raw_outputs: bool = True,
    verbose: bool = False,
    comp_level: int = 5,
):
    """Process TRITON-only model outputs (no SWMM coupling)."""
    fname_out = self._validate_path(
        self.scen_paths.output_triton_only_timeseries,
        "output_triton_only_timeseries",
    )

    # Check if already written
    if self._already_written(fname_out) and not overwrite_if_exist:
        if verbose:
            print(f"{fname_out.name} already written. Not overwriting.")
        if clear_raw_outputs:
            self._clear_raw_TRITON_outputs()
        return

    start_time = time.time()

    # Use out_triton directory (not out_tritonswmm)
    fldr_out_triton = self._scenario.scen_paths.out_triton
    if fldr_out_triton is None or not fldr_out_triton.exists():
        raise ValueError(
            f"TRITON-only output directory not found: {fldr_out_triton}. "
            "Ensure toggle_triton_model is enabled and simulation has run."
        )

    raw_out_dir = fldr_out_triton / self._analysis.cfg_analysis.TRITON_raw_output_type

    # Read TRITON outputs (same logic as coupled model)
    lst_files = list(raw_out_dir.glob("*"))
    if not lst_files:
        raise FileNotFoundError(f"No TRITON output files found in {raw_out_dir}")

    ds_all_tsteps = xr.open_mfdataset(
        lst_files,
        combine="nested",
        concat_dim="time",
        engine="h5netcdf" if fname_out.suffix == ".nc" else "zarr",
        parallel=True,
    )

    # Add event identifier
    ds_all_tsteps = ds_all_tsteps.expand_dims({"event_iloc": [self._scenario.event_iloc]})

    # Write output
    self._write_output(ds_all_tsteps, fname_out, comp_level, verbose)

    elapsed_s = time.time() - start_time
    self.log.add_sim_processing_entry(
        fname_out, get_file_size_MiB(fname_out), elapsed_s, True
    )
    self.log.TRITON_only_timeseries_written.set(True)

    if clear_raw_outputs:
        self._clear_raw_TRITON_outputs()

    return
```

#### 3.2 Rename Existing Method
**Change:** Rename `_export_TRITON_outputs()` → `_export_TRITONSWMM_TRITON_outputs()`

**Rationale:** Makes it explicit that this processes TRITON outputs from the coupled model.

#### 3.3 Add Router Method
**Replace old `_export_TRITON_outputs()` with:**

```python
def _export_TRITON_outputs(
    self,
    overwrite_if_exist: bool = False,
    clear_raw_outputs: bool = True,
    verbose: bool = False,
    comp_level: int = 5,
):
    """Route to appropriate TRITON processing method based on model type."""
    model_types = self._run.model_types_enabled

    if "tritonswmm" in model_types:
        self._export_TRITONSWMM_TRITON_outputs(
            overwrite_if_exist, clear_raw_outputs, verbose, comp_level
        )
    elif "triton" in model_types:
        self._export_TRITON_only_outputs(
            overwrite_if_exist, clear_raw_outputs, verbose, comp_level
        )
    else:
        if verbose:
            print("No TRITON model enabled, skipping TRITON output processing")
```

**What Was Actually Implemented:**

#### Timeseries Processing ✅
- ✅ Added `_export_TRITON_only_outputs()` - processes from `out_triton/bin/`
- ✅ Renamed `_export_TRITON_outputs()` → `_export_TRITONSWMM_TRITON_outputs()`
- ✅ Added router `_export_TRITON_outputs()` that dispatches based on `model_types_enabled`
- ✅ Updated SWMM timeseries router in `write_timeseries_outputs()` to process both tritonswmm and swmm_only

#### Summary Processing ✅ (with improvements)
- ✅ Made `_export_TRITON_summary(model_type)` model-aware with parameter
- ✅ Made `_export_SWMM_summaries(model_type)` model-aware with parameter
- ✅ Both methods validate input timeseries exists and fail loudly if not
- ✅ Router in `write_summary_outputs()` checks timeseries existence before creating summaries
- ✅ Uses conditional processing: only creates summaries for models where timeseries exist

**Key Design Decision:** Instead of separate methods per model, we made existing methods take `model_type` parameter and route based on which timeseries files exist. This is cleaner and avoids duplicate code.

**Acceptance Criteria:**
- [x] TRITON-only outputs use `output_triton_only_*` paths
- [x] TRITON-SWMM outputs work with `output_tritonswmm_triton_*` paths
- [x] Router correctly dispatches based on enabled models
- [x] TRITON-only timeseries + summary created successfully
- [x] TRITON-SWMM all 8 outputs created successfully
- [ ] SWMM-only outputs created (0/4 - needs debugging)

**Test Results:**
```bash
pytest tests/test_PC_01_singlesim.py::test_process_sim
# TRITON-only: ✅ 2/2 files (timeseries + summary)
# TRITON-SWMM: ✅ 8/8 files
# SWMM-only: ❌ 0/4 files (processing runs but files not created)
```

---

### Phase 4: Add TRITON-only Summary Generation

**File:** `src/TRITON_SWMM_toolkit/process_simulation.py`
**Status:** ⬜ Not Started

#### 4.1 Add `_export_TRITON_only_summary()` Method
**Location:** After line ~719 (after existing `_export_TRITON_summary`)

```python
def _export_TRITON_only_summary(
    self,
    overwrite_if_exist: bool = False,
    verbose: bool = False,
    comp_level: int = 5,
):
    """Create TRITON-only summary from full timeseries."""
    fname_out = self._validate_path(
        self.scen_paths.output_triton_only_summary,
        "output_triton_only_summary",
    )

    if self._already_written(fname_out) and not overwrite_if_exist:
        if verbose:
            print(f"{fname_out.name} already written. Not overwriting.")
        return

    start_time = time.time()

    # Load full timeseries
    ds_full = self._open(self.scen_paths.output_triton_only_timeseries)

    # Summarize (reuse existing function)
    target_dem_res = self._system.cfg_system.target_dem_resolution
    ds_summary = summarize_triton_simulation_results(
        ds_full, self._scenario.event_iloc, target_dem_res
    )

    # Add compute time
    df = pd.DataFrame(
        index=[self._scenario.event_iloc],
        data=dict(compute_time_min=[self._scenario.sim_compute_time_min]),
    )
    df.index.name = "event_iloc"
    da_compute_time = df.to_xarray()["compute_time_min"]
    ds_summary["compute_time_min"] = da_compute_time

    # Write
    self._write_output(ds_summary, fname_out, comp_level, verbose)
    elapsed_s = time.time() - start_time
    self.log.add_sim_processing_entry(
        fname_out, get_file_size_MiB(fname_out), elapsed_s, True
    )
    self.log.TRITON_only_summary_written.set(True)

    return
```

#### 4.2 Convert `_export_TRITON_summary()` to Router
**Modify existing method:**

```python
def _export_TRITON_summary(
    self,
    overwrite_if_exist: bool = False,
    verbose: bool = False,
    comp_level: int = 5,
):
    """Route to appropriate TRITON summary method based on model type."""
    model_types = self._run.model_types_enabled

    if "tritonswmm" in model_types:
        self._export_TRITONSWMM_TRITON_summary(
            overwrite_if_exist, verbose, comp_level
        )
    elif "triton" in model_types:
        self._export_TRITON_only_summary(
            overwrite_if_exist, verbose, comp_level
        )
```

**Also rename:** `_export_TRITON_summary()` internals → `_export_TRITONSWMM_TRITON_summary()`

**Acceptance Criteria:**
- [ ] TRITON-only summary generated correctly
- [ ] Uses same summarization logic as coupled model
- [ ] Writes to `output_triton_only_summary` path
- [ ] Logging updated appropriately

---

### Phase 5: Complete SWMM-only Summary Generation

**File:** `src/TRITON_SWMM_toolkit/process_simulation.py`
**Status:** ⬜ Not Started

#### 5.1 Refactor `_export_SWMM_summaries()` to Accept Model Parameter

**Current issue:** Method hardcoded for TRITON-SWMM paths only

**Changes (line ~720):**

```python
def _export_SWMM_summaries(
    self,
    model: Literal["swmm", "tritonswmm"] = "tritonswmm",
    overwrite_if_exist: bool = False,
    verbose: bool = False,
    comp_level: int = 5,
):
    """Create SWMM node and link summaries from full timeseries."""
    start_time = time.time()

    # Select paths based on model type
    if model == "tritonswmm":
        f_out_nodes = self._validate_path(
            self.scen_paths.output_tritonswmm_node_summary,
            "output_tritonswmm_node_summary",
        )
        f_out_links = self._validate_path(
            self.scen_paths.output_tritonswmm_link_summary,
            "output_tritonswmm_link_summary",
        )
        # Load from coupled model timeseries
        ds_nodes_full = self._open(self.scen_paths.output_tritonswmm_node_time_series)
        ds_links_full = self._open(self.scen_paths.output_tritonswmm_link_time_series)
    else:  # model == "swmm"
        f_out_nodes = self._validate_path(
            self.scen_paths.output_swmm_only_node_summary,
            "output_swmm_only_node_summary",
        )
        f_out_links = self._validate_path(
            self.scen_paths.output_swmm_only_link_summary,
            "output_swmm_only_link_summary",
        )
        # Load from SWMM-only timeseries
        ds_nodes_full = self._open(self.scen_paths.output_swmm_only_node_time_series)
        ds_links_full = self._open(self.scen_paths.output_swmm_only_link_time_series)

    # Check if already written
    nodes_already_written = self._already_written(f_out_nodes)
    links_already_written = self._already_written(f_out_links)

    if (nodes_already_written and links_already_written) and not overwrite_if_exist:
        if verbose:
            print(f"{f_out_nodes.name} and {f_out_links.name} already written.")
        return

    # Rest of existing implementation (summarization logic) unchanged
    # ... (lines 739-801)
```

#### 5.2 Update `write_summary_outputs()` Router

**Modify to pass model type:**

```python
def write_summary_outputs(
    self,
    which: Literal["TRITON", "SWMM", "both"] = "both",
    overwrite_if_exist: bool = False,
    verbose: bool = False,
    compression_level: int = 5,
):
    """Create summary files from full timeseries."""
    model_types = self._run.model_types_enabled

    # TRITON-SWMM performance summary
    if "tritonswmm" in model_types and which in {"TRITON", "both"}:
        self._export_TRITONSWMM_performance_summary(
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            comp_level=compression_level,
        )

    # TRITON summary
    if (which == "both") or (which == "TRITON"):
        self._export_TRITON_summary(
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            comp_level=compression_level,
        )

    # SWMM summary
    if (which == "both") or (which == "SWMM"):
        # Determine SWMM model type
        if "swmm" in model_types:
            swmm_model = "swmm"
        elif "tritonswmm" in model_types:
            swmm_model = "tritonswmm"
        else:
            if verbose:
                print("No SWMM model enabled, skipping SWMM summary")
            return

        self._export_SWMM_summaries(
            model=swmm_model,
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            comp_level=compression_level,
        )
```

**Acceptance Criteria:**
- [ ] SWMM-only summaries generated
- [ ] Uses all 4 `output_swmm_only_*` paths
- [ ] Logging tracks both node and link summaries
- [ ] Test with SWMM-only configuration passes

---

### Phase 6: Update Logging Properties

**Files:** `src/TRITON_SWMM_toolkit/log.py`, `src/TRITON_SWMM_toolkit/process_simulation.py`
**Status:** ⬜ Not Started

#### 6.1 Add LogField Entries to Scenario Log

**File:** `src/TRITON_SWMM_toolkit/log.py`

```python
# TRITON-only logging
TRITON_only_timeseries_written: LogField[bool] = LogField(default=False)
TRITON_only_summary_written: LogField[bool] = LogField(default=False)
raw_TRITON_only_outputs_cleared: LogField[bool] = LogField(default=False)

# SWMM-only logging
SWMM_only_node_timeseries_written: LogField[bool] = LogField(default=False)
SWMM_only_link_timeseries_written: LogField[bool] = LogField(default=False)
SWMM_only_node_summary_written: LogField[bool] = LogField(default=False)
SWMM_only_link_summary_written: LogField[bool] = LogField(default=False)
raw_SWMM_only_outputs_cleared: LogField[bool] = LogField(default=False)
```

#### 6.2 Update Property Methods

**File:** `src/TRITON_SWMM_toolkit/process_simulation.py`

**Update `TRITON_outputs_processed` property:**

```python
@property
def TRITON_outputs_processed(self) -> bool:
    """Check if TRITON outputs processed (model-aware)."""
    model_types = self._run.model_types_enabled

    if "tritonswmm" in model_types:
        triton = self._already_written(
            self.scen_paths.output_tritonswmm_triton_timeseries
        )
        self.log.TRITON_timeseries_written.set(triton)
        return triton
    elif "triton" in model_types:
        triton = self._already_written(
            self.scen_paths.output_triton_only_timeseries
        )
        self.log.TRITON_only_timeseries_written.set(triton)
        return triton
    else:
        return False
```

**Add similar logic for SWMM properties**

**Acceptance Criteria:**
- [ ] All model types have proper logging fields
- [ ] Properties correctly detect which model ran
- [ ] Log fields persist to JSON correctly
- [ ] Backward compatibility maintained

---

### Phase 7: Update `write_timeseries_outputs()` Public API

**File:** `src/TRITON_SWMM_toolkit/process_simulation.py`
**Status:** ⬜ Not Started

**Modify (lines 96-136):**

```python
def write_timeseries_outputs(
    self,
    which: Literal["TRITON", "SWMM", "both"] = "both",
    clear_raw_outputs: bool = True,
    overwrite_if_exist: bool = False,
    verbose: bool = False,
    compression_level: int = 5,
):
    """Process outputs based on enabled model types."""
    scen = self._scenario
    model_types = self._run.model_types_enabled

    if not self._scenario.sim_run_completed:
        raise RuntimeError(
            f"Simulation not completed. Log: {self._scenario.latest_simlog}"
        )

    print(f"Processing run results for scenario {scen.event_iloc}", flush=True)

    # Performance timeseries (only for TRITON-SWMM coupled)
    if "tritonswmm" in model_types and which in {"TRITON", "both"}:
        self._export_TRITONSWMM_performance_tseries(
            comp_level=compression_level,
            verbose=verbose,
            overwrite_if_exist=overwrite_if_exist,
        )

    # TRITON outputs (router handles both TRITON-only and TRITON-SWMM)
    if (which == "both") or (which == "TRITON"):
        if "triton" in model_types or "tritonswmm" in model_types:
            self._export_TRITON_outputs(
                overwrite_if_exist,
                clear_raw_outputs,
                verbose,
                compression_level,
            )
            print(f"Processed TRITON outputs for scenario {scen.event_iloc}", flush=True)

    # SWMM outputs (router handles both SWMM-only and TRITON-SWMM)
    if (which == "both") or (which == "SWMM"):
        if "swmm" in model_types:
            self._export_SWMM_outputs(
                model="swmm",
                overwrite_if_exist=overwrite_if_exist,
                clear_raw_outputs=clear_raw_outputs,
                verbose=verbose,
                comp_level=compression_level,
            )
            print(f"Processed SWMM outputs for scenario {scen.event_iloc}", flush=True)
        elif "tritonswmm" in model_types:
            self._export_SWMM_outputs(
                model="tritonswmm",
                overwrite_if_exist=overwrite_if_exist,
                clear_raw_outputs=clear_raw_outputs,
                verbose=verbose,
                comp_level=compression_level,
            )
            print(f"Processed SWMM outputs for scenario {scen.event_iloc}", flush=True)

    return
```

**Acceptance Criteria:**
- [ ] Correctly processes outputs for any combination of enabled models
- [ ] Doesn't try to process outputs for disabled models
- [ ] Maintains backward compatibility
- [ ] All tests pass

---

### Phase 8: Update Workflow Integration

**File:** `src/TRITON_SWMM_toolkit/workflow.py`
**Status:** ⬜ Not Started

**Fix `--which` flag generation (lines 396-427):**

```python
# CURRENT (BUG):
if model_type == "triton":
    which_arg = "TRITON"
elif model_type == "tritonswmm":
    which_arg = which  # BUG: Uses configured value
elif model_type == "swmm":
    which_arg = "SWMM"

# FIXED:
if model_type == "triton":
    which_arg = "TRITON"
elif model_type == "tritonswmm":
    which_arg = "both"  # Always process both for coupled model
elif model_type == "swmm":
    which_arg = "SWMM"
```

**Acceptance Criteria:**
- [ ] `process_triton` rule calls with `--which TRITON`
- [ ] `process_tritonswmm` rule calls with `--which both`
- [ ] `process_swmm` rule calls with `--which SWMM`
- [ ] Workflow tests pass

---

### Phase 9: Update Runner Script Validation

**File:** `src/TRITON_SWMM_toolkit/process_timeseries_runner.py`
**Status:** ⬜ Not Started

**Add model-aware validation (lines 134-177):**

```python
# Add model-type detection
model_types_enabled = run.model_types_enabled

# Verify processing request matches enabled models
if args.which == "TRITON":
    if "triton" not in model_types_enabled and "tritonswmm" not in model_types_enabled:
        logger.error(
            f"TRITON processing requested but no TRITON model enabled "
            f"for scenario {args.event_iloc}"
        )
        return 1

if args.which == "SWMM":
    if "swmm" not in model_types_enabled and "tritonswmm" not in model_types_enabled:
        logger.error(
            f"SWMM processing requested but no SWMM model enabled "
            f"for scenario {args.event_iloc}"
        )
        return 1

if not scenario.sim_run_completed:
    logger.error(
        f"Simulation not completed for scenario {args.event_iloc}. "
        f"Cannot process outputs. Log: {scenario.latest_simlog}"
    )
    return 1
```

**Acceptance Criteria:**
- [ ] Clear error messages when processing incompatible model types
- [ ] Doesn't fail when model is disabled
- [ ] Still validates simulation completion

---

## Testing Strategy

**Smoke Test:** `tests/test_PC_01_singlesim.py`

**After each phase:**
```bash
pytest tests/test_PC_01_singlesim.py -v -s
```

**Final integration:**
```bash
pytest tests/test_multi_model_integration.py -v
pytest tests/test_PC_01_singlesim.py -v
```

**Assertion Functions to Validate:**
- `assert_model_simulation_run()` - Validates model-specific runs
- `assert_model_outputs_processed()` - Validates model-specific processing
- `get_enabled_model_types()` - Helper for test configuration

---

## Progress Tracking

### Completed Tasks
- [x] Gap analysis of unused paths
- [x] Comprehensive implementation plan created
- [ ] Phase 1: Critical bug fixes
- [ ] Phase 2: Type annotation fix
- [ ] Phase 3: TRITON output processing split
- [ ] Phase 4: TRITON-only summary generation
- [ ] Phase 5: SWMM-only summary generation
- [ ] Phase 6: Logging updates
- [ ] Phase 7: Public API updates
- [ ] Phase 8: Workflow integration
- [ ] Phase 9: Runner validation

### Known Issues
- Pre-existing bug in `swmm_output_parser.py` for `.out` file processing (unrelated to this work)
- `_export_SWMM_outputs()` has wrong type annotation syntax

### Next Steps
1. Start with Phase 1 to fix critical infrastructure gaps
2. Test incrementally after each phase
3. Document any deviations from plan
4. Update this document with completion status

---

## Resuming Work

**To resume this implementation:**

1. **Check progress:**
   ```bash
   git diff src/TRITON_SWMM_toolkit/run_simulation.py
   git diff src/TRITON_SWMM_toolkit/process_simulation.py
   ```

2. **Identify last completed phase** by checking task list above

3. **Run smoke test** to verify current state:
   ```bash
   pytest tests/test_PC_01_singlesim.py::test_process_sim -v
   ```

4. **Continue with next incomplete phase**

5. **Update this document** with completion status

---

## Critical Files Modified

- `src/TRITON_SWMM_toolkit/run_simulation.py` - Model detection, raw output paths
- `src/TRITON_SWMM_toolkit/process_simulation.py` - Processing routers, model-specific methods
- `src/TRITON_SWMM_toolkit/log.py` - Logging fields for all model types
- `src/TRITON_SWMM_toolkit/workflow.py` - Snakemake rule parameters
- `src/TRITON_SWMM_toolkit/process_timeseries_runner.py` - Model-aware validation
- `tests/utils_for_testing.py` - Assertion functions (already updated)

---

## Dependencies & Shared Utilities

**No new utilities needed** - existing functions work for all model types:
- `summarize_triton_simulation_results()` ✓
- `summarize_swmm_simulation_results()` ✓
- `retrieve_SWMM_outputs_as_datasets()` ✓
- `_write_output()` ✓
- `_validate_path()` ✓

---

## SWMM-Only Output Processing Debugging Plan

**Status:** ⬜ Not Started
**Priority:** High (blocks Phase 3 completion)

### Problem Statement

When `test_process_sim` runs, it reports:
```
Processed SWMM-only outputs for scenario 0
```
But the output files don't exist:
```
Failed: swmm output processing incomplete:
  - SWMM-only node timeseries (SWMM_only_node_tseries.nc) - scenario 0
  - SWMM-only link timeseries (SWMM_only_link_tseries.nc) - scenario 0
  - SWMM-only node summary (SWMM_only_node_summary.nc) - scenario 0
  - SWMM-only link summary (SWMM_only_link_summary.nc) - scenario 0
```

### Diagnostic Steps

#### Step 1: Verify Raw SWMM Output Exists
**Check:** Does `swmm_full_out_file` exist before processing?

```python
# In _export_SWMM_outputs(), add at the beginning when model=="swmm":
if model == "swmm":
    f_inp = self.scen_paths.swmm_full_inp
    swmm_timeseries_result_file = self.scen_paths.swmm_full_out_file

    # DIAGNOSTIC: Check file existence
    print(f"[DEBUG] SWMM-only processing:")
    print(f"  swmm_full_inp: {f_inp}")
    print(f"  swmm_full_out_file: {swmm_timeseries_result_file}")
    print(f"  inp exists: {f_inp.exists() if f_inp else None}")
    print(f"  out exists: {swmm_timeseries_result_file.exists() if swmm_timeseries_result_file else None}")
```

**Expected:** Both files should exist
**If not:** Check if standalone SWMM simulation actually ran

#### Step 2: Trace Through retrieve_SWMM_outputs_as_datasets()
**Check:** Does the parsing function succeed or fail silently?

```python
# Wrap the call in try/except to catch any errors
try:
    ds_nodes, ds_links = retrieve_SWMM_outputs_as_datasets(
        f_inp,
        swmm_timeseries_result_file,
    )
    print(f"[DEBUG] Successfully parsed SWMM outputs")
    print(f"  ds_nodes shape: {ds_nodes.dims}")
    print(f"  ds_links shape: {ds_links.dims}")
except Exception as e:
    print(f"[ERROR] Failed to parse SWMM outputs: {e}")
    import traceback
    traceback.print_exc()
    raise
```

**Expected:** Should successfully create datasets
**If fails:** Error message will indicate parsing issue

#### Step 3: Check Path Configuration
**Check:** Are `output_swmm_only_*` paths correctly initialized in scenario.py?

```python
# In test, after creating proc object:
proc = analysis._retrieve_sim_run_processing_object(0)
paths = proc.scen_paths
print(f"output_swmm_only_node_time_series: {paths.output_swmm_only_node_time_series}")
print(f"output_swmm_only_link_time_series: {paths.output_swmm_only_link_time_series}")
print(f"  Both are None: {paths.output_swmm_only_node_time_series is None}")
```

**Expected:** Paths should be set to `processed/SWMM_only_*.nc`
**If None:** Issue in scenario.py path initialization (likely toggle check)

#### Step 4: Verify _write_output() is Called
**Check:** Does execution reach the write step?

```python
# Before _write_output() calls:
print(f"[DEBUG] About to write SWMM-only node timeseries to: {f_out_nodes}")
self._write_output(ds_nodes, f_out_nodes, comp_level, verbose)
print(f"[DEBUG] Write completed, file exists: {f_out_nodes.exists()}")
```

**Expected:** Should show write completion
**If not reached:** Logic error in conditional checks

#### Step 5: Check for Silent Exception Catching
**Search:** Look for broad try/except blocks that might swallow errors

```bash
grep -n "except.*:" src/TRITON_SWMM_toolkit/process_simulation.py | grep -A2 "pass"
```

**Expected:** No silent exception catching in SWMM processing path
**If found:** Remove or make it re-raise after logging

### Hypotheses (Ranked by Likelihood)

1. **Path Configuration Issue (80%):** `output_swmm_only_*` paths are None because `toggle_swmm_model` check is failing in scenario.py initialization
   - **Test:** Print paths in test to verify
   - **Fix:** Correct toggle check in scenario.py lines ~177-194

2. **Raw Output Deleted Before Processing (15%):** `swmm_full_out_file` is deleted by time we try to process it
   - **Test:** Check file existence at start of `_export_SWMM_outputs()`
   - **Fix:** Adjust order of operations or clear_raw_outputs logic

3. **Silent Exception (5%):** Error during write is caught and ignored
   - **Test:** Add try/except with explicit logging
   - **Fix:** Remove silent catch or add proper error handling

### Implementation Plan

1. Add diagnostic prints to `_export_SWMM_outputs()` for `model=="swmm"` case
2. Run test and capture debug output
3. Based on output, implement fix:
   - If paths are None → Fix scenario.py initialization
   - If file missing → Check simulation ran and adjust clear timing
   - If parsing fails → Debug swmm_output_parser.py
4. Remove diagnostic prints after fix confirmed
5. Update this plan with findings

### Success Criteria

- [ ] All 4 SWMM-only output files created successfully
- [ ] Test assertion `assert_model_outputs_processed(analysis, "swmm")` passes
- [ ] No debug/diagnostic prints remaining in code
- [ ] Phase 3 marked as 100% complete
