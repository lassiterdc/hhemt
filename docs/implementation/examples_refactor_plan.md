# TRITON-SWMM Examples/Test Utilities: Full Structural Refactor Plan

**Date**: 2026-01-30  
**Status**: Proposed  
**Scope**: `src/TRITON_SWMM_toolkit/examples.py` + test utilities usage  

## Goal

Restructure the `examples.py` module to separate production-facing example utilities from
test/benchmark infrastructure, reduce duplication, and make configuration generation and
system setup composable and testable.

This plan targets a **full structural refactor**, not just cleanup, while preserving
current behavior and test coverage.

## Current Pain Points (Summary)

1. **Mixed concerns in one module**
   - Production example loading, HydroShare download logic, synthetic weather generation,
     and test-case presets live in one file.

2. **Heavy, side-effectful constructors**
   - `retrieve_TRITON_SWMM_test_case.__init__()` does filesystem writes, creates configs,
     generates weather, and processes inputs in one step.

3. **Config duplication in presets**
   - `GetTS_TestCases` repeats HPC configuration blocks across many methods.

4. **Environment-specific hard-coded paths**
   - Example: `/home/***REMOVED***/.conda/...` or `/scratch/***REMOVED***/...`.

5. **Unclear public API vs. test-only utilities**
   - Tests import utilities that are bundled with the package, making it unclear which
     functions are stable for end users.

## Success Criteria

- Production example utilities remain easy to use and documented.
- Test utilities are isolated and data-driven, minimizing duplication.
- Config generation is separated from file I/O and system processing.
- Hard-coded user paths are removed.
- Existing tests keep passing with minimal changes.

---

## Refactor Overview

### Proposed Module Layout

```
src/TRITON_SWMM_toolkit/examples/
├── __init__.py
├── norfolk.py              # Norfolk example load + template fill
├── hydroshare.py           # HydroShare download/sign-in utilities
├── weather.py              # Synthetic weather generation utilities
├── test_case_builder.py    # Builder for synthetic test cases
└── presets.py              # Data-driven test-case presets

tests/
├── fixtures/
│   ├── example_cases.py    # Optional test-only entry points
│   └── platform_configs.py # Optional test-only platform presets
```

> Note: If the project prefers not to create a package subdirectory, these can be
> standalone modules with a consistent naming scheme, e.g. `examples_hydroshare.py`.

---

## Phase 1: Extraction + API Preservation

### 1. Create an `examples` package
- Move or wrap existing public entrypoints into `examples/__init__.py`.
- Preserve current imports by keeping `examples.py` as a backward-compatible shim:
  ```python
  from .examples import *
  ```
  or forwarding selected public names.

### 2. Extract Norfolk template utilities
**Move to** `examples/norfolk.py`:
- `load_config_filepath`
- `load_config_file_as_dic`
- `return_filled_template_yaml_dictionary`
- `get_norfolk_data_and_package_directory_mapping_dict`
- `load_norfolk_system_config`
- `TRITON_SWMM_examples` (renamed: `NorfolkExamples` or similar)

### 3. Extract HydroShare logic
**Move to** `examples/hydroshare.py`:
- `download_data_from_hydroshare`
- `sign_into_hydroshare`

Ensure HydroShare dependency remains optional (same behavior).

---

## Phase 2: Refactor Test Case Builder

### 4. Replace `retrieve_TRITON_SWMM_test_case` with a builder

**New class**: `TestCaseBuilder` (in `examples/test_case_builder.py`)

**Responsibilities split into explicit steps**:

```python
builder = TestCaseBuilder(cfg_system_yaml, analysis_name, ...)

builder.apply_system_overrides(...)
builder.apply_analysis_overrides(...)

builder.write_system_config()
builder.write_analysis_config()
builder.generate_weather_timeseries()
builder.process_system_inputs()
```

Benefits:
- Allows dry-run and unit testing of each step
- Reduces side effects in `__init__`
- Makes failures easier to localize

### 5. Consolidate synthetic weather logic
**Move to** `examples/weather.py`:
- `create_short_intense_weather_timeseries`
- `_create_reduced_weather_file_for_testing_if_it_does_not_exist` (if still needed)

---

## Phase 3: Presets & Duplication Reduction

### 6. Convert `GetTS_TestCases` into data-driven presets

Define preset configurations as dicts or dataclasses:

```python
PRESETS = {
    "UVA_multisim": CasePreset(
        analysis_name="UVA_multisim",
        analysis_overrides={...},
        system_overrides={...},
        n_events=8,
    ),
    "frontier_multisim_GPU": CasePreset(...),
}
```

Provide a single entrypoint:

```python
def create_case(preset_name: str, **overrides) -> TestCaseBuilder:
    preset = PRESETS[preset_name]
    return TestCaseBuilder.from_preset(preset, **overrides)
```

### 7. Extract platform base configurations
Define base configs (UVA / Frontier / local) once, then merge overrides.

---

## Phase 4: Hard-Coded Paths & Environment Awareness

### 8. Remove hard-coded paths

Replace:
- `python_path="/home/..."` → `sys.executable`
- `/scratch/***REMOVED***/...` → derive from `$USER` or provide as input

### 9. Add runtime-safe defaults
- Add optional parameters for data directory and Python executable
- Provide fallback to package root or `Path.home()`

---

## Phase 5: Backward Compatibility + Docs

### 10. Backward compatibility shim
Keep `examples.py` for one release cycle:
- Re-export old names
- Emit warnings for deprecated classes

### 11. Documentation updates
- Update `docs/usage.rst` or add a new doc showing:
  - Example data download
  - Synthetic test case creation
  - Preset selection via `create_case()`

---

## Implementation Checklist

1. [ ] Create new examples package layout
2. [ ] Move Norfolk config utilities to `norfolk.py`
3. [ ] Move HydroShare utilities to `hydroshare.py`
4. [ ] Build `TestCaseBuilder` with explicit steps
5. [ ] Move weather generation logic to `weather.py`
6. [ ] Replace `GetTS_TestCases` with data-driven presets
7. [ ] Remove hard-coded paths and add defaults
8. [ ] Add backward compatibility shim in `examples.py`
9. [ ] Update tests to new imports
10. [ ] Update docs with new example API

---

## Expected Impact

**Short-term**:
- Cleaner organization and safer test utilities
- Improved portability across user environments

**Long-term**:
- Easier to add new example datasets and HPC presets
- Better composability for users who need partial setup steps

---

## Next Steps

If approved, implement in the following order:
1. Extraction of utilities (low risk)
2. Builder refactor (medium risk)
3. Preset consolidation (medium risk)
4. Test + doc updates (high confidence)

If you want a smaller scope first, we can execute **Phases 1–2 only** as a separate PR.