# NOTE: this was resolved manually on 2/9/2026 but qaqc would be helpful.

# TRITON Output Path Bug

## Bug Description

TRITON-SWMM (the external C++ executable) writes certain output files to hardcoded paths,
ignoring the `output_folder` directive in the CFG configuration file.

### Affected Outputs

| Output | Expected Location | Actual Location | Status |
|--------|------------------|-----------------|--------|
| TRITON raw outputs (H, QX, QY, MH binaries) | `out_tritonswmm/bin/` | `out_tritonswmm/bin/` | ✅ Correct |
| TRITON-only raw outputs | `out_triton/bin/` | `out_triton/bin/` | ✅ Correct |
| Coupled SWMM outputs (.out, .rpt) | `out_tritonswmm/` | `sim_folder/output/swmm/` | ❌ Bug |
| `log.out` (TRITON log) | Per-model directory | `sim_folder/output/log.out` | ❌ Bug |

### `log.out` Overwrite Issue

When both TRITON-only and TRITON-SWMM models are enabled (multi-model mode), both write
to `sim_folder/output/log.out`. The last model to finish overwrites the other's log. This
means resource-usage parsing (nTasks, OMP threads, GPUs, backend) may be incorrect for
one model type.

## Workaround Convention

All workaround code is tagged with:
```
TODO(TRITON-OUTPUT-PATH-BUG)
```

### Affected Code Locations

- `src/TRITON_SWMM_toolkit/run_simulation.py` — `coupled_swmm_output_file` property
- `src/TRITON_SWMM_toolkit/process_simulation.py` — `_export_SWMM_outputs` tritonswmm branch
- `src/TRITON_SWMM_toolkit/process_simulation.py` — `_clear_raw_SWMM_outputs` tritonswmm branch
- `src/TRITON_SWMM_toolkit/analysis.py` — `log_out_path` (~line 1270)
- `src/TRITON_SWMM_toolkit/sensitivity_analysis.py` — `log_out_path` (~line 663)

## Resolution

Once the TRITON-SWMM developer fixes the executable to respect the `output_folder` directive:
1. Search for `TODO(TRITON-OUTPUT-PATH-BUG)` across the codebase
2. Remove all workaround code
3. Delete this document
