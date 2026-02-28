# Config Package Split — Implementation Plan

## Objective

Split `src/TRITON_SWMM_toolkit/config.py` (642 lines, mixed concerns) into a
focused `config/` package with single-responsibility submodules. No compatibility
shims. All import sites updated immediately per project philosophy in CLAUDE.md.

---

## Current State

`config.py` contains four distinct concerns in one file:

1. Base class + display utilities + toggle-validation helper (`cfgBaseModel`, lines 18-137)
2. SystemConfig schema + validators (`system_config`, lines 139-317)
3. AnalysisConfig schema + validators (`analysis_config`, lines 320-624)
4. Loader functions (`load_system_config*`, `load_analysis_config`, lines 627-641)

### Phase 1 Completion Status

Phase 1 from `config_py_refactor_plan.md` is complete:
- `extra="forbid"` applied to `cfgBaseModel`
- Explicit `@model_validator` rules replace dynamic toggle registry

---

## Target Package Layout

```
src/TRITON_SWMM_toolkit/config/
    __init__.py          # Minimal package init; no re-exports
    base.py              # cfgBaseModel base class, display helpers, validate_from_toggle
    system.py            # system_config Pydantic model + validators
    analysis.py          # analysis_config Pydantic model + validators
    loaders.py           # load_system_config, load_system_config_from_dict, load_analysis_config
```

Deferred to future phases (per config_py_refactor_plan.md):
- `validation.py` — reusable extracted validators (future)
- `display.py` — tabulate/tree display helpers (future)
- `profiles.py` — TestsAndCaseStudiesConfig (future)

---

## Module Contents

### `config/__init__.py`
Minimal docstring only. No re-exports — all consumers import from submodules directly.

### `config/base.py`
- `class cfgBaseModel(BaseModel)`

### `config/system.py`
- `class system_config(cfgBaseModel)`

### `config/analysis.py`
- `class analysis_config(cfgBaseModel)`

### `config/loaders.py`
- `load_system_config_from_dict(cfg_dict)`
- `load_system_config(cfg_yaml: Path)`
- `load_analysis_config(cfg_yaml: Path)`

---

## Import Sites That Must Be Updated

| File | New import |
|------|-----------|
| `src/TRITON_SWMM_toolkit/system.py` | `from TRITON_SWMM_toolkit.config.loaders import load_system_config` |
| `src/TRITON_SWMM_toolkit/analysis.py` | `from TRITON_SWMM_toolkit.config.loaders import load_analysis_config` |
| `src/TRITON_SWMM_toolkit/examples.py` | `from TRITON_SWMM_toolkit.config.loaders import (load_system_config, load_system_config_from_dict)` |
| `src/TRITON_SWMM_toolkit/case_study_catalog.py` | `from TRITON_SWMM_toolkit.config.analysis import analysis_config` |
| `src/TRITON_SWMM_toolkit/gui.py` | Remove broken `SimulationConfig` import (does not exist) |
| `tests/fixtures/test_case_builder.py` | `from TRITON_SWMM_toolkit.config.loaders import ...` + `from TRITON_SWMM_toolkit.config.analysis import analysis_config` |
| `tests/test_config_validation.py` | Split across `.analysis`, `.loaders`, `.system` |
| `scripts/check_doc_freshness.py` | Update `"config.py"` filename keys to `"config/"` |

---

## Migration Order (Single Atomic Commit)

1. Create `config/` package directory
2. Create `config/base.py` (cfgBaseModel)
3. Create `config/system.py` (system_config)
4. Create `config/analysis.py` (analysis_config)
5. Create `config/loaders.py` (loader functions)
6. Create `config/__init__.py` (docstring only)
7. Delete `config.py`
8. Update all import sites
9. Run all smoke tests

---

## Dependency Graph (No Circular Imports)

```
plot_utils  ←── base.py
                  ↑
            system.py    analysis.py
                  ↑           ↑
                  └─────┬─────┘
                     loaders.py
```

---

## Risks

| Risk | Mitigation |
|------|-----------|
| `gui.py` references non-existent `SimulationConfig` | Already broken — remove import + dead GUI references |
| `scripts/check_doc_freshness.py` filename keys | Update string from `config.py` to `config/` |

---

## Future Phases

Once the package exists:
- Extract `validate_from_toggle` into `config/validation.py`
- Move display methods into `config/display.py`
- Add `config/profiles.py` for TestsAndCaseStudiesConfig
