# Plan: Allow Users to Specify Processed DEM/Manning’s Paths

**Status:** Draft (implementation-ready)
**Owner:** Toolkit maintainers
**Created:** 2026-02-13
**Scope:** `system_config` + system input processing and validation

---

## Goal

Allow users to optionally pass **pre-processed** DEM and Manning’s raster file paths in
`system_config`. When these are provided, the toolkit should **skip** creating
processed outputs in `TRITONSWMM_system.process_system_level_inputs()` and use the
user-supplied files directly throughout the system and analysis workflows.

This enables workflows where preprocessing is done externally (e.g., custom GIS
pipelines), while keeping current defaults for users who rely on the built-in
processing steps.

---

## Deliverables

1. **Config schema updates** to accept `processed_dem` and `processed_mannings`.
2. **System path routing** to use user-provided paths when set.
3. **Processing bypass** (skip DEM/Manning’s generation when inputs supplied).
4. **Validation updates** to allow bypassing raw inputs when processed files are provided.
5. **Optional log synchronization** to mark processed files as complete when provided.
6. **Documentation** updates and example YAML snippet.

---

## Success Criteria

- If `processed_dem` is provided and exists, **no DEM processing occurs** and the
  provided file is used everywhere `sys_paths.dem_processed` is referenced.
- If `processed_mannings` is provided and exists, **no Manning’s processing occurs**
  and the provided file is used everywhere `sys_paths.mannings_processed` is referenced.
- Validation accepts missing `DEM_fullres` / landuse inputs **when processed files are provided**.
- Existing workflows **continue to work unchanged** when the new fields are not set.

---

## Design Overview

### Key Paths and Flow

1. `system_config` is loaded → `TRITONSWMM_system.__init__` builds `SysPaths`.
2. `SysPaths.dem_processed` and `SysPaths.mannings_processed` are used throughout
   scenario preparation, plotting, analysis, and post-processing.
3. `process_system_level_inputs()` always runs `create_dem_for_TRITON()` and
   `create_mannings_file_for_TRITON()` (unless constant mannings is enabled).

### New Behavior (with user-supplied processed files)

If `system_config.processed_dem` or `system_config.processed_mannings` are set:

✅ `SysPaths` should point at those paths

✅ `process_system_level_inputs()` should **skip** processing for those inputs

✅ Validation should accept missing raw inputs when processed equivalents exist

---

## Implementation Plan

### 1) Add new optional fields to `system_config`

**File:** `src/TRITON_SWMM_toolkit/config/system.py`

Add new optional fields with clear docstrings:

```python
from typing import Optional
from pydantic import Field
from pathlib import Path

    processed_dem: Optional[Path] = Field(
        None,
        description=(
            "Optional path to a pre-processed DEM formatted for TRITON-SWMM "
            "(ASCII raster with TRITON header). When provided, DEM processing "
            "is skipped and this file is used directly."
        ),
    )

    processed_mannings: Optional[Path] = Field(
        None,
        description=(
            "Optional path to a pre-processed Manning's raster formatted for TRITON-SWMM "
            "(ASCII raster). When provided, Manning's processing is skipped and this "
            "file is used directly."
        ),
    )
```

---

### 2) Route `SysPaths` to user-supplied files when present

**File:** `src/TRITON_SWMM_toolkit/system.py`

In `TRITONSWMM_system.__init__`, replace the current `SysPaths` assignments for
`dem_processed` and `mannings_processed`:

```python
        dem_processed_path = (
            self.cfg_system.processed_dem
            if self.cfg_system.processed_dem is not None
            else system_dir / f"elevation_{self.cfg_system.target_dem_resolution:.2f}m.dem"
        )

        mannings_processed_path = (
            self.cfg_system.processed_mannings
            if self.cfg_system.processed_mannings is not None
            else system_dir / f"mannings_{self.cfg_system.target_dem_resolution:.2f}m.dem"
        )

        self.sys_paths = SysPaths(
            dem_processed=dem_processed_path,
            mannings_processed=mannings_processed_path,
            ...
        )
```

This ensures downstream usage (scenarios, plots, processing) automatically uses
the provided files.

---

### 3) Skip system-level processing when files provided

**File:** `src/TRITON_SWMM_toolkit/system.py`

Update `process_system_level_inputs()`:

```python
    def process_system_level_inputs(self, overwrite_outputs_if_already_created: bool = False, verbose: bool = False):
        if self.cfg_system.processed_dem is None:
            self.create_dem_for_TRITON(overwrite_outputs_if_already_created, verbose)
        elif verbose:
            print("Using user-provided processed DEM; skipping DEM processing.")

        if not self.cfg_system.toggle_use_constant_mannings:
            if self.cfg_system.processed_mannings is None:
                self.create_mannings_file_for_TRITON(overwrite_outputs_if_already_created, verbose)
            elif verbose:
                print("Using user-provided processed Manning's; skipping Manning's processing.")
```

This preserves current behavior when processed inputs are not provided.

---

### 4) Validation updates for optional processed inputs

**File:** `src/TRITON_SWMM_toolkit/validation.py`

#### 4.1 Core path checks

Allow `DEM_fullres` to be optional if `processed_dem` is provided:

```python
def _validate_system_paths(cfg: system_config, result: ValidationResult):
    required_paths = {
        "system_directory": cfg.system_directory,
        "watershed_gis_polygon": cfg.watershed_gis_polygon,
        "SWMM_hydraulics": cfg.SWMM_hydraulics,
        "TRITONSWMM_software_directory": cfg.TRITONSWMM_software_directory,
        "triton_swmm_configuration_template": cfg.triton_swmm_configuration_template,
    }

    # DEM is required only if processed_dem is not provided
    if cfg.processed_dem is None:
        required_paths["DEM_fullres"] = cfg.DEM_fullres
    else:
        # still validate processed DEM exists
        if not Path(cfg.processed_dem).exists():
            result.add_error(
                field="system.processed_dem",
                message="Processed DEM file does not exist",
                current_value=str(cfg.processed_dem),
                fix_hint="Provide a valid processed_dem path or remove processed_dem",
            )
```

#### 4.2 Manning’s dependencies

Allow skipping landuse-based inputs if `processed_mannings` is provided:

```python
def _validate_toggle_dependencies_system(cfg: system_config, result: ValidationResult):
    if cfg.toggle_use_constant_mannings:
        ...
    else:
        # If processed_mannings is provided, landuse inputs are not required
        if cfg.processed_mannings is None:
            if cfg.landuse_lookup_file is None:
                result.add_error(...)
```

Also verify the processed file exists when provided:

```python
    if cfg.processed_mannings is not None and not Path(cfg.processed_mannings).exists():
        result.add_error(
            field="system.processed_mannings",
            message="Processed Manning's file does not exist",
            current_value=str(cfg.processed_mannings),
            fix_hint="Provide a valid processed_mannings path or remove processed_mannings",
        )
```

---

### 5) Consistency checks between processed DEM and Manning’s

**File:** `src/TRITON_SWMM_toolkit/system.py` or `validation.py`

Because Manning’s rasters may not include an ASCII header, the alignment check should
compare the **opened rasters** for coordinate and dimension consistency instead of
comparing headers.

**Raster coordinate/dimension check** (recommended):

```python
if cfg.processed_dem and cfg.processed_mannings:
    rds_dem = rxr.open_rasterio(cfg.processed_dem)
    rds_mannings = rxr.open_rasterio(cfg.processed_mannings)

    if rds_dem.shape != rds_mannings.shape:
        result.add_warning(
            field="system.processed_mannings",
            message="Processed DEM and Manning's shapes differ. Ensure grids align.",
            current_value=str(cfg.processed_mannings),
            fix_hint="Provide aligned rasters or regenerate using toolkit processing",
        )

    if not np.allclose(rds_dem.x.values, rds_mannings.x.values):
        result.add_warning(
            field="system.processed_mannings",
            message="Processed DEM and Manning's x-coordinates differ.",
            current_value=str(cfg.processed_mannings),
            fix_hint="Provide aligned rasters or regenerate using toolkit processing",
        )

    if not np.allclose(rds_dem.y.values, rds_mannings.y.values):
        result.add_warning(
            field="system.processed_mannings",
            message="Processed DEM and Manning's y-coordinates differ.",
            current_value=str(cfg.processed_mannings),
            fix_hint="Provide aligned rasters or regenerate using toolkit processing",
        )
```

---

### 6) Optional: synchronize logs when processed inputs exist

**File:** `src/TRITON_SWMM_toolkit/system.py`

If user provides processed files and they exist, log fields should reflect
completion to avoid status checks showing setup as incomplete:

```python
if self.cfg_system.processed_dem and self.sys_paths.dem_processed.exists():
    self.log.dem_processed.set(True)
    # Optionally set shape by reading raster

if self.cfg_system.processed_mannings and self.sys_paths.mannings_processed.exists():
    self.log.mannings_processed.set(True)
```

This ensures `analysis.get_workflow_status()` correctly reflects setup completion.

---

### 7) Documentation + examples

Add example YAML snippet to planning doc or README:

```yaml
# Example: use pre-processed rasters
processed_dem: /path/to/processed_dem.dem
processed_mannings: /path/to/processed_mannings.dem

# Raw inputs can be omitted if processed files are provided
# DEM_fullres: ... (optional if processed_dem set)
# landuse_lookup_file: ... (optional if processed_mannings set)
```

---

## Production-Ready Code Chunks

> The following code snippets are ready to paste into the target files.

### A) `system_config` schema extension

```python
# File: src/TRITON_SWMM_toolkit/config/system.py

    processed_dem: Optional[Path] = Field(
        None,
        description=(
            "Optional path to a pre-processed DEM formatted for TRITON-SWMM "
            "(ASCII raster with TRITON header). When provided, DEM processing "
            "is skipped and this file is used directly."
        ),
    )

    processed_mannings: Optional[Path] = Field(
        None,
        description=(
            "Optional path to a pre-processed Manning's raster formatted for TRITON-SWMM "
            "(ASCII raster). When provided, Manning's processing is skipped and this "
            "file is used directly."
        ),
    )
```

### B) `SysPaths` routing in `TRITONSWMM_system.__init__`

```python
# File: src/TRITON_SWMM_toolkit/system.py

        dem_processed_path = (
            self.cfg_system.processed_dem
            if self.cfg_system.processed_dem is not None
            else system_dir / f"elevation_{self.cfg_system.target_dem_resolution:.2f}m.dem"
        )

        mannings_processed_path = (
            self.cfg_system.processed_mannings
            if self.cfg_system.processed_mannings is not None
            else system_dir / f"mannings_{self.cfg_system.target_dem_resolution:.2f}m.dem"
        )

        self.sys_paths = SysPaths(
            dem_processed=dem_processed_path,
            mannings_processed=mannings_processed_path,
            ...
        )
```

### C) Processing bypass in `process_system_level_inputs()`

```python
# File: src/TRITON_SWMM_toolkit/system.py

    def process_system_level_inputs(
        self, overwrite_outputs_if_already_created: bool = False, verbose: bool = False
    ):
        if self.cfg_system.processed_dem is None:
            self.create_dem_for_TRITON(overwrite_outputs_if_already_created, verbose)
        elif verbose:
            print("Using user-provided processed DEM; skipping DEM processing.")

        if not self.cfg_system.toggle_use_constant_mannings:
            if self.cfg_system.processed_mannings is None:
                self.create_mannings_file_for_TRITON(
                    overwrite_outputs_if_already_created, verbose
                )
            elif verbose:
                print("Using user-provided processed Manning's; skipping Manning's processing.")
```

### D) Validation changes (core path + Manning’s dependencies)

```python
# File: src/TRITON_SWMM_toolkit/validation.py

def _validate_system_paths(cfg: system_config, result: ValidationResult):
    required_paths = {
        "system_directory": cfg.system_directory,
        "watershed_gis_polygon": cfg.watershed_gis_polygon,
        "SWMM_hydraulics": cfg.SWMM_hydraulics,
        "TRITONSWMM_software_directory": cfg.TRITONSWMM_software_directory,
        "triton_swmm_configuration_template": cfg.triton_swmm_configuration_template,
    }

    if cfg.processed_dem is None:
        required_paths["DEM_fullres"] = cfg.DEM_fullres
    else:
        if not Path(cfg.processed_dem).exists():
            result.add_error(
                field="system.processed_dem",
                message="Processed DEM file does not exist",
                current_value=str(cfg.processed_dem),
                fix_hint="Provide a valid processed_dem path or remove processed_dem",
            )

def _validate_toggle_dependencies_system(cfg: system_config, result: ValidationResult):
    if cfg.toggle_use_constant_mannings:
        if cfg.constant_mannings is None:
            result.add_error(...)
    else:
        if cfg.processed_mannings is None:
            if cfg.landuse_lookup_file is None:
                result.add_error(...)

    if cfg.processed_mannings is not None and not Path(cfg.processed_mannings).exists():
        result.add_error(
            field="system.processed_mannings",
            message="Processed Manning's file does not exist",
            current_value=str(cfg.processed_mannings),
            fix_hint="Provide a valid processed_mannings path or remove processed_mannings",
        )
```

---

## Decisions (Confirmed)

1. **`DEM_fullres` is NOT required** when `processed_dem` is provided.
2. **Landuse inputs are optional** when `processed_mannings` is provided.
3. **Use raster coordinate/dimension checks** (not header checks) for alignment,
   since Manning’s may not include a header.

---

## Testing Notes (Recommended)

1. **Happy path**: Provide `processed_dem` and `processed_mannings`, skip processing,
   ensure system setup and analysis run normally.
2. **Partial path**: Provide only `processed_dem`, ensure Manning’s is still generated
   unless constant mannings is enabled.
3. **Validation**: Missing `DEM_fullres` should pass when `processed_dem` exists.
4. **Header mismatch warning** (if enabled): ensure warnings show but processing continues.

---

## Proposed Documentation Snippet

```yaml
# Optional: provide pre-processed inputs
processed_dem: /data/my_project/processed/elevation_2.00m.dem
processed_mannings: /data/my_project/processed/mannings_2.00m.dem

# Raw inputs can be omitted when processed files are provided
# DEM_fullres: ... (optional if processed_dem set)
# landuse_lookup_file: ... (optional if processed_mannings set)
```

---

## Next Step

Once approved, implement the changes in:

- `src/TRITON_SWMM_toolkit/config/system.py`
- `src/TRITON_SWMM_toolkit/system.py`
- `src/TRITON_SWMM_toolkit/validation.py`

and update any example configs or README snippets as needed.
