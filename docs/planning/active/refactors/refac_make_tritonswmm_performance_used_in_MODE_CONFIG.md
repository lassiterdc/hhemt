# Refactor Plan: Route TRITONSWMM Performance Through `_MODE_CONFIG`

**Date:** 2026-02-13
**Status:** Draft (implementation-ready)
**Owner:** Toolkit maintainers

## Goal

Make TRITONSWMM performance consolidation follow the same `_MODE_CONFIG` pipeline
as other summary outputs (including `triton_only_performance`). This removes a
special-case method and centralizes consolidation logic in the shared
`consolidate_outputs_for_mode()` flow.

## Why This Refactor

Today, performance outputs are handled inconsistently:

- `triton_only_performance` **is** in `_MODE_CONFIG`
- `tritonswmm_performance` **is not** in `_MODE_CONFIG`

Instead, TRITONSWMM performance uses a bespoke method:

```python
consolidate_TRITONSWMM_performance_summaries()
```

This creates a split pattern (special-case vs shared pipeline) with duplicated
logging and validation. Best practice here is a single consolidation API to
reduce maintenance risk and avoid future drift.

## Desired End State

- Add `tritonswmm_performance` to `_MODE_CONFIG`.
- Use `consolidate_outputs_for_mode("tritonswmm_performance")` rather than a
  dedicated method.
- Retire or deprecate `consolidate_TRITONSWMM_performance_summaries()` to avoid
  two separate consolidation paths.

## Design Notes

- Performance datasets are **non-spatial**, so `spatial_coords=None`.
- `_chunk_for_writing()` already supports `spatial_coords=None`.
- Shared `_consolidate_outputs()` already handles:
  - log updates
  - overwrite checks
  - output writing

## Implementation Plan

### 1) Add `_MODE_CONFIG` entry

**File:** `src/TRITON_SWMM_toolkit/processing_analysis.py`

Add a new key:

```python
"tritonswmm_performance": (
    "output_tritonswmm_performance_summary",
    "output_tritonswmm_performance_summary",
    "tritonswmm_performance_analysis_summary_created",
    None,
),
```

**Production-ready diff:**

```python
# File: src/TRITON_SWMM_toolkit/processing_analysis.py

_MODE_CONFIG = {
    # ... existing modes ...
    "triton_only_performance": (
        "output_triton_only_performance_summary",
        "output_triton_only_performance_summary",
        "triton_only_performance_analysis_summary_created",
        None,
    ),
    "tritonswmm_performance": (
        "output_tritonswmm_performance_summary",
        "output_tritonswmm_performance_summary",
        "tritonswmm_performance_analysis_summary_created",
        None,
    ),
}
```

### 2) Route TRITONSWMM performance through shared path

**File:** `src/TRITON_SWMM_toolkit/analysis.py`

In `consolidate_TRITON_and_SWMM_simulation_summaries()`, replace:

```python
self.process.consolidate_TRITONSWMM_performance_summaries(...)
```

with:

```python
self.process.consolidate_outputs_for_mode(
    "tritonswmm_performance",
    overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
    verbose=verbose,
    compression_level=compression_level,
)
```

**Production-ready diff:**

```python
# File: src/TRITON_SWMM_toolkit/analysis.py

if cfg_sys.toggle_tritonswmm_model:
    if verbose:
        print("Consolidating TRITON-SWMM coupled model outputs...", flush=True)
    _consolidate("tritonswmm_triton")
    _consolidate("tritonswmm_swmm_node")
    _consolidate("tritonswmm_swmm_link")
    _consolidate("tritonswmm_performance")
```

### 3) Deprecate or remove the bespoke method

**File:** `src/TRITON_SWMM_toolkit/processing_analysis.py`

Option A (preferred): remove `consolidate_TRITONSWMM_performance_summaries()`.

Option B (safer transition): keep method but replace its contents with a call
to `consolidate_outputs_for_mode("tritonswmm_performance")` and add a comment
stating it is deprecated and will be removed.

**Production-ready diff (wrapper option):**

```python
# File: src/TRITON_SWMM_toolkit/processing_analysis.py

def consolidate_TRITONSWMM_performance_summaries(
    self,
    overwrite_outputs_if_already_created: bool = False,
    verbose: bool = False,
    compression_level: int = 5,
):
    """Deprecated. Use consolidate_outputs_for_mode('tritonswmm_performance')."""
    return self.consolidate_outputs_for_mode(
        "tritonswmm_performance",
        overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
        verbose=verbose,
        compression_level=compression_level,
    )
```

### 4) Update sensitivity consolidation (if needed)

If sensitivity analysis uses TRITONSWMM performance consolidation directly,
ensure it also routes through the shared mode. (This depends on current call
sites; see `sensitivity_analysis.py`.)

**Important finding:** TRITON-only sensitivity runs currently **do not**
consolidate performance summaries at the master level. The sensitivity
consolidation only handles:

- TRITON spatial summaries
- SWMM node/link summaries
- TRITONSWMM performance summaries (when coupled model is enabled)

There is **no master-level combine** for `triton_only_performance`, which is a
likely cause of missing `TRITON_only_performance.zarr` in sensitivity analyses.

**Production-ready fix (add TRITON-only performance combine):**

```python
# File: src/TRITON_SWMM_toolkit/sensitivity_analysis.py

def _combine_triton_performance_per_subanalysis(self):
    cfg_sys = self.master_analysis._system.cfg_system
    if cfg_sys.toggle_tritonswmm_model:
        return self._combine_TRITONSWMM_performance_per_subanalysis()
    if cfg_sys.toggle_triton_model:
        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            ds = sub_analysis.process.triton_only_performance_summary
            ds = ds.assign_coords(coords={"sub_analysis_iloc": sub_analysis_iloc})
            ds = ds.expand_dims("sub_analysis_iloc")
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)
            lst_ds.append(ds)
        return xr.combine_by_coords(lst_ds, combine_attrs="drop", join="outer")
    raise ValueError("No TRITON model enabled for performance consolidation.")

def _performance_output_mode(self) -> str:
    cfg_sys = self.master_analysis._system.cfg_system
    if cfg_sys.toggle_tritonswmm_model:
        return "tritonswmm_performance"
    if cfg_sys.toggle_triton_model:
        return "triton_only_performance"
    raise ValueError("No TRITON model enabled for performance consolidation.")
```

**Hook into `consolidate_subanalysis_outputs`:**

```python
if which in ["TRITON", "both"]:
    # ... existing TRITON/SWMM consolidation ...
    perf_mode = self._performance_output_mode()
    ds_performance = self._combine_triton_performance_per_subanalysis()
    self.master_analysis.process._consolidate_outputs(
        ds_performance,
        mode=perf_mode,
        overwrite_outputs_if_already_created=overwrite_outputs_if_already_created,
        verbose=verbose,
        compression_level=compression_level,
    )
```

## Testing Plan

1. Run `test_PC_04_multisim_with_snakemake.py` to verify standard consolidation.
2. Run `test_PC_05_sensitivity_analysis_with_snakemake.py` to validate
   sensitivity consolidation (if performance outputs enabled).
3. Optional: add a small unit test for `_MODE_CONFIG` containing the new key.

**Additional check for TRITON-only sensitivity:**

4. Run `tests/test_PC_06_sensitivity_analysis_triton_and_swmm_only.py` and
   verify the master analysis directory contains `TRITON_only_performance.zarr`
   (or `.nc` if configured).

## Success Criteria

- TRITONSWMM performance outputs are consolidated via `_MODE_CONFIG`.
- No duplicate code path for performance summaries.
- Existing tests pass without modification.

## Notes / Risks

- If any downstream code calls `consolidate_TRITONSWMM_performance_summaries()`
  directly, keep it as a wrapper to preserve behavior until call sites are
  migrated.
- This refactor aligns performance outputs with the unified consolidation
  contract and prevents further divergence between model types.
