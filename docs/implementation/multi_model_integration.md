# Multi-Model Integration Plan: TRITON, TRITON-SWMM, and SWMM

**Status:** ✅ Complete (all 7 phases implemented and tested)
**Last Updated:** 2026-02-07

## Current Status

All 7 implementation phases are **COMPLETE** and all tests pass:
- ✅ Phase 1: Test Infrastructure
- ✅ Phase 2: Compilation
- ✅ Phase 3: CFG Generation
- ✅ Phase 4: Scenario Preparation
- ✅ Phase 5: Simulation Execution
- ✅ Phase 6: Output Processing
- ✅ Phase 7: Workflow Integration

Debugging issues (template config defaults, race conditions) were resolved through
subsequent work on model-specific logs (commit d0e7b7a) and multi-model output
processing. See `multi_model_output_processing_plan.md` for output processing details.

**Key Configuration Change:** Model toggles moved from `analysis_config` to `system_config` (2026-01-31) because compilation happens at system level before analysis exists. This follows the "configuration near first use" architectural pattern.

### Configuration Toggle Location

The model toggles are now in `system_config.yaml`:
```yaml
toggle_triton_model: True
toggle_tritonswmm_model: True
toggle_swmm_model: True
```

Previously these were in `analysis_config.yaml`, but system initialization needs them to create build directories and compilation paths, so they were moved to system config.

## Overview

This plan integrates pure TRITON and pure SWMM models into the existing TRITON-SWMM workflow, enabling users to run any combination of the three model types **concurrently** in a single analysis.

## Architecture Decisions

### Execution Model
- **SWMM execution**: Compiled EPA SWMM executable (not PySwmm) for controllability
- **Concurrent simulation**: All three model types run in parallel via separate Snakemake rules
- **Resource allocation**:
  - TRITON/TRITON-SWMM: CPUs and/or GPUs as configured
  - SWMM: CPUs only (multithreaded, no GPU support) - limit to reasonable thread count (e.g., 4)

### Directory Structure (Single Event)
All models run in the **same scenario directory** with model-specific output subdirectories:
```
{scenario_dir}/
├── logs/                          # Centralized log directory
│   ├── run_triton.log
│   ├── run_tritonswmm.log
│   └── run_swmm.log
├── out_triton/                    # TRITON-only outputs
│   ├── triton_tseries.nc
│   └── triton_summary.nc
├── out_tritonswmm/                # Coupled model outputs
│   ├── tritonswmm_triton_tseries.nc
│   ├── tritonswmm_swmm_node_tseries.nc
│   └── tritonswmm_swmm_link_tseries.nc
├── out_swmm/                      # SWMM-only outputs
│   ├── swmm_node_tseries.nc
│   └── swmm_link_tseries.nc
├── TRITON.cfg                     # TRITON-only CFG
├── TRITONSWMM.cfg                 # Coupled model CFG
├── swmm_full.inp                  # SWMM-only input
└── ...

> **Note:** The coupled TRITON-SWMM executable still writes its SWMM artifacts
> (e.g., `hydraulics.out`, `hydraulics.rpt`, node/link outputs) under the
> legacy `output/swmm/` folder. The `output_folder` setting in `TRITONSWMM.cfg`
> redirects TRITON outputs (bin/cfg/performance) to `out_tritonswmm/` but does
> not relocate SWMM artifacts produced by the coupled binary.
```

### Output Format and Dimensions
- **Default format**: NetCDF (not Zarr - better I/O performance for many-file scenarios)
- **Model dimension**: All timeseries outputs include `model` as a dimension/coordinate
- **Filename convention**: `{model}_{variable}_tseries.nc` (e.g., `triton_surface_tseries.nc`)

### Logging Approach
- Follow existing patterns from `analysis.py` and `scenario.py`
- Use `LogField[T]` pattern for JSON-persisted logging
- Model type tracked in `analysis.df_status`

---

## Snakemake Workflow Structure

With all three models enabled, the workflow generates **separate rules** for each model type:

```
rule all:
    input: all output flags

# Phase 1: Setup (compiles all enabled models)
rule setup:
    output: "_status/setup_complete.flag"

# Phase 2: Scenario Preparation (one rule, prepares all models)
rule prepare_scenario:
    input: setup_complete
    output: "_status/sims/scenario_{event_iloc}_prepared.flag"

# Phase 3: Simulation (separate rules per model - run concurrently)
rule run_triton:
    input: scenario_prepared
    output: "_status/sims/triton_{event_iloc}_complete.flag"
    threads: {triton_cpus}
    resources: gpus={triton_gpus}

rule run_tritonswmm:
    input: scenario_prepared
    output: "_status/sims/tritonswmm_{event_iloc}_complete.flag"
    threads: {tritonswmm_cpus}
    resources: gpus={tritonswmm_gpus}

rule run_swmm:
    input: scenario_prepared
    output: "_status/sims/swmm_{event_iloc}_complete.flag"
    threads: 4  # SWMM multithreaded, no GPU

# Phase 4: Output Processing (separate rules per model - run concurrently)
rule process_triton:
    input: triton_complete
    output: "_status/sims/triton_{event_iloc}_processed.flag"

rule process_tritonswmm:
    input: tritonswmm_complete
    output: "_status/sims/tritonswmm_{event_iloc}_processed.flag"

rule process_swmm:
    input: swmm_complete
    output: "_status/sims/swmm_{event_iloc}_processed.flag"

# Phase 5: Consolidation
rule consolidate:
    input: all processing flags
    output: "_status/consolidation_complete.flag"
```

---

## Implementation Phases

### Phase 1: Test Infrastructure
**Status:** ⬜ Not Started

#### Files to Modify
- `tests/conftest.py` - Add multi-model fixtures
- `tests/fixtures/test_case_catalog.py` - Add test case builders
- `tests/utils_for_testing.py` - Add model-type assertion helpers
- `tests/test_PC_01_singlesim.py` - Add per-model tests

#### Key Tasks
1. **Test fixtures for each configuration**:
   - `norfolk_triton_only_analysis` - TRITON only
   - `norfolk_swmm_only_analysis` - SWMM only
   - `norfolk_all_models_analysis` - All three enabled

2. **Assertion helpers**:
   - `assert_triton_compiled(analysis)`
   - `assert_swmm_compiled(analysis)`
   - `assert_model_simulation_run(analysis, model_type)`
   - `assert_model_outputs_processed(analysis, model_type)`

3. **SWMM output consistency tests**:
   - Verify SWMM output NetCDF files have same data variables
   - Verify dimensions and coordinates match across model types
   - Use existing `swmm_output_parser.py` functions for extraction

4. **Model type in df_status**:
   - Add `model_type` column to `analysis.df_status`

#### Acceptance Criteria
- [ ] Tests define expected paths for model-specific outputs
- [ ] Tests verify SWMM .out parsing produces consistent NetCDF structure
- [ ] Tests verify model type is tracked in df_status

---

### Phase 2: Compilation
**Status:** ⬜ Not Started

#### Files to Modify
- `src/TRITON_SWMM_toolkit/system.py` - Add compilation methods
- `src/TRITON_SWMM_toolkit/paths.py` - Add paths

#### Key Changes

1. **`compile_TRITON_only()`**:
   - CMake flag: `-DTRITON_ENABLE_SWMM=OFF`
   - Build dir: `build_triton_cpu/` and `build_triton_gpu/`

2. **`compile_SWMM()`**:
   - Clone from `SWMM_git_URL` at `SWMM_tag_key`
   - Build standalone executable using CMake
   - Build dir: `swmm_build/`

3. **Rename existing directories**:
   - `build_cpu` → `build_tritonswmm_cpu`
   - `build_gpu` → `build_tritonswmm_gpu`

4. **Logging**: Follow LogField pattern for compilation status

#### Acceptance Criteria
- [ ] TRITON compiles with explicit `-DTRITON_ENABLE_SWMM=OFF`
- [ ] EPA SWMM compiles as standalone executable
- [ ] Compilation status logged consistently

---

### Phase 3: CFG Generation
**Status:** ⬜ Not Started

#### Files to Modify
- `src/TRITON_SWMM_toolkit/scenario.py` - CFG generation

#### Key Changes

1. **Model-specific CFG files**:
   - `TRITON.cfg` - `inp_filename` commented out
   - `TRITONSWMM.cfg` - `inp_filename` present (existing)

2. **`output_folder` argument**:
   - TRITON: `output_folder="out_triton"`
   - TRITON-SWMM: `output_folder="out_tritonswmm"`
   - Replace if exists, append if not

3. **No build folder copy** when using `output_folder`

#### Acceptance Criteria
- [ ] TRITON.cfg has `#inp_filename=...` (commented)
- [ ] Both CFGs have model-specific `output_folder`
- [ ] Build folder not copied to scenario directory

---

### Phase 4: Scenario Preparation
**Status:** ⬜ Not Started

#### Files to Modify
- `src/TRITON_SWMM_toolkit/scenario.py`
- `src/TRITON_SWMM_toolkit/paths.py`

#### Key Changes

1. **Model-specific paths in ScenarioPaths**:
   ```python
   # CFG files
   triton_cfg: Optional[Path]
   triton_swmm_cfg: Path

   # Executables
   sim_triton_executable: Optional[Path]
   sim_tritonswmm_executable: Path
   sim_swmm_executable: Optional[Path]

   # Output directories
   out_triton: Optional[Path]
   out_tritonswmm: Optional[Path]
   out_swmm: Optional[Path]

   # Log files (in logs/ subdirectory)
   log_run_triton: Optional[Path]
   log_run_tritonswmm: Optional[Path]
   log_run_swmm: Optional[Path]
   ```

2. **`prepare_scenario()` updates**:
   - Prepares all enabled model types
   - Creates logs/ subdirectory
   - Creates model-specific output directories

#### Acceptance Criteria
- [ ] Scenario prep creates logs/ directory
- [ ] Model-specific output directories created
- [ ] All required files generated per model type

---

### Phase 5: Simulation Execution
**Status:** ⬜ Not Started

#### Files to Modify
- `src/TRITON_SWMM_toolkit/run_simulation.py`
- `src/TRITON_SWMM_toolkit/run_simulation_runner.py`

#### Key Changes

1. **Separate runner scripts per model** (or single runner with `--model-type` flag):
   ```bash
   python -m TRITON_SWMM_toolkit.run_simulation_runner \
       --model-type triton \
       --event-iloc 0 \
       --system-config ... \
       --analysis-config ...
   ```

2. **SWMM execution via compiled executable**:
   ```python
   def _run_swmm(self):
       cmd = [str(self.swmm_executable), str(inp_file), str(rpt_file), str(out_file)]
       subprocess.run(cmd, check=True)
   ```

3. **Model-specific log files**:
   - `logs/run_triton.log`
   - `logs/run_tritonswmm.log`
   - `logs/run_swmm.log`

4. **Logging**: Update LogField for each model type's completion status

#### Acceptance Criteria
- [ ] Each model runs with correct executable
- [ ] SWMM uses compiled executable (not PySwmm)
- [ ] Logs written to model-specific files
- [ ] Completion status tracked per model

---

### Phase 6: Output Processing
**Status:** ⬜ Not Started

#### Files to Modify
- `src/TRITON_SWMM_toolkit/process_simulation.py`
- `src/TRITON_SWMM_toolkit/process_timeseries_runner.py`
- `src/TRITON_SWMM_toolkit/swmm_output_parser.py` (use existing functions)

#### Key Changes

1. **Model-specific output processing**:
   - Runner accepts `--model-type` flag
   - Reads from model-specific output directory
   - Writes to model-specific timeseries files

2. **SWMM .out parsing**:
   - Use existing `swmm_output_parser.py` functions
   - Extract timeseries from `.out` file
   - Output format: NetCDF (default)

3. **Output filename convention**:
   - `{model}_{variable}_tseries.nc`
   - Examples: `triton_surface_tseries.nc`, `swmm_node_tseries.nc`

4. **Model dimension in outputs**:
   - All timeseries include `model` as dimension/coordinate
   - Enables easy comparison across model types

#### Acceptance Criteria
- [ ] Each model's outputs processed independently
- [ ] SWMM timeseries extracted from .out file
- [ ] NetCDF outputs include `model` dimension
- [ ] Output file naming follows convention

---

### Phase 7: Workflow Integration
**Status:** ⬜ Not Started

#### Files to Modify
- `src/TRITON_SWMM_toolkit/workflow.py`
- `src/TRITON_SWMM_toolkit/setup_workflow.py`
- `src/TRITON_SWMM_toolkit/consolidate_workflow.py`

#### Key Changes

1. **Separate Snakemake rules per model**:
   - `run_triton`, `run_tritonswmm`, `run_swmm`
   - `process_triton`, `process_tritonswmm`, `process_swmm`

2. **Resource allocation**:
   - TRITON/TRITON-SWMM: `threads={cpus}`, `resources: gpus={gpus}`
   - SWMM: `threads=4` (upper limit), no GPU

3. **Status flags per model**:
   - `_status/sims/triton_{event_iloc}_complete.flag`
   - `_status/sims/tritonswmm_{event_iloc}_complete.flag`
   - `_status/sims/swmm_{event_iloc}_complete.flag`

4. **Consolidation aggregates all model types**:
   - Collects outputs from all enabled models
   - Updates `df_status` with model-type column

#### Acceptance Criteria
- [ ] Three simulation rules run concurrently
- [ ] Three processing rules run concurrently
- [ ] SWMM rule has thread limit, no GPU
- [ ] df_status includes model_type column

---

## Smoke Tests

After each phase:
- [ ] `pytest tests/test_PC_01_singlesim.py`
- [ ] `pytest tests/test_PC_02_multisim.py`

After Phase 7:
- [ ] `pytest tests/test_PC_04_multisim_with_snakemake.py`
- [ ] `pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py`

---

## Progress Tracking

| Phase | Status | Tests Pass | Notes |
|-------|--------|------------|-------|
| 1. Test Infrastructure | ✅ | ✅ | |
| 2. Compilation | ✅ | ✅ | |
| 3. CFG Generation | ✅ | ✅ | |
| 4. Scenario Preparation | ✅ | ✅ | |
| 5. Simulation Execution | ✅ | ✅ | |
| 6. Output Processing | ✅ | ✅ | |
| 7. Workflow Integration | ✅ | ✅ | |

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SWMM execution | Compiled EPA SWMM | Better controllability |
| Concurrent models | Separate Snakemake rules | Independent execution |
| SWMM threads | Limit to 4 CPUs | Reasonable default, no GPU |
| Output format | NetCDF | Better I/O performance |
| Directory structure | Single dir, model-specific outputs | Clean organization |
| Log organization | `logs/` subdirectory | Centralized logging |
| CMake flag | Explicit `-DTRITON_ENABLE_SWMM=OFF` | No ambiguity |
| Build dir naming | Renamed to `build_tritonswmm_*` | Consistency |
| df_status | Add model_type column | Track per-model status |

---

## Critical Files

1. `src/TRITON_SWMM_toolkit/system.py` - Compilation
2. `src/TRITON_SWMM_toolkit/scenario.py` - CFG, scenario prep
3. `src/TRITON_SWMM_toolkit/run_simulation.py` - Execution
4. `src/TRITON_SWMM_toolkit/workflow.py` - Snakemake generation
5. `src/TRITON_SWMM_toolkit/swmm_output_parser.py` - SWMM output parsing
6. `src/TRITON_SWMM_toolkit/paths.py` - Path definitions

