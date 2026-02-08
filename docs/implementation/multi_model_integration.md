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

All seven phases listed in this plan were completed and validated. The detailed
per-phase design text below is preserved as historical implementation context,
but the active status is captured in the progress table and current status
sections of this document.

---

## Regression Validation Order

Canonical smoke/regression order used for this integration work:
1. `pytest tests/test_PC_01_singlesim.py`
2. `pytest tests/test_PC_02_multisim.py`
3. `pytest tests/test_PC_04_multisim_with_snakemake.py`
4. `pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py`

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

