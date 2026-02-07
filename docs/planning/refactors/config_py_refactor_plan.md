# `config.py` Refactor Plan (Strict + Backward-Compatible)

## Objective

Refactor `src/TRITON_SWMM_toolkit/config.py` to be cleaner, stricter, and easier
to maintain **without breaking downstream usage patterns**.

---

## Non-Breaking Contract (Hard Requirement)

The refactor must preserve current downstream expectations:

1. Existing code can continue using `cfg_system` and `cfg_analysis` exactly as-is.
2. `load_system_config(...)` and `load_analysis_config(...)` remain valid entrypoints.
3. Existing field names used throughout analysis/workflow code remain unchanged
   unless a deliberate migration is planned and documented.
4. No required call-site edits across simulation/workflow modules for v1 refactor.

This means we refactor internals and validation architecture, not external model
contracts.

---

## Current Pain Points in `config.py`

1. **File size / mixed concerns**: schema definitions, loading, validation,
   display helpers, and utility logic are all in one module.
2. **Toggle validation pattern is brittle**: mutable class-level `toggle_tests`
   with runtime population is harder to reason about and debug.
3. **Error UX inconsistency**: model-level printing and exceptions are mixed,
   which makes CLI/API error rendering less consistent.
4. **Strictness gaps**: unknown/unused config keys should fail loudly to prevent
   silent user confusion.
5. **Maintenance friction**: adding new profile/config types (e.g., testcase /
   case-study inheritance) will further bloat the current file.

---

## Design Principles

1. **Strict by default**
   - Disallow unknown keys (`extra="forbid"`) for user-facing config models.
2. **Single responsibility per module**
   - Split schema, validation helpers, loading, and display utilities.
3. **Readable explicit validators**
   - Prefer explicit `@model_validator` rules over dynamic toggle registries.
4. **Consistent error reporting**
   - Raise structured validation errors; formatting is handled at loader/CLI/API layer.
5. **Backward-compatible public surface**
   - Keep core config object behavior and naming stable.

---

## Target Module Layout

Proposed package structure:

```text
src/TRITON_SWMM_toolkit/config/
  __init__.py
  base.py                # shared base class + common config policy
  system.py              # SystemConfig schema + validators
  analysis.py            # AnalysisConfig schema + validators
  profiles.py            # TestsAndCaseStudiesConfig schemas (new)
  validation.py          # reusable dependency/path/range validators
  loaders.py             # yaml loading + strict/legacy mode entrypoints
  display.py             # tabulate/tree display helpers
```

Compatibility shim:

- Keep `src/TRITON_SWMM_toolkit/config.py` as a thin compatibility module that
  re-exports current public symbols (`system_config`, `analysis_config`,
  `load_system_config`, `load_analysis_config`, etc.) during migration.

---

## Strict Validation Policy

Apply to all user-facing config models:

- `model_config = ConfigDict(extra="forbid")`

Behavioral intent:

- Unknown/unused keys are validation errors.
- Error message must identify field path and offending key.
- Avoid silent acceptance of stale or misspelled arguments.

Optional migration safety:

- Add an explicit temporary legacy loader mode (`strict=False`) only if needed,
  with deprecation warning and planned removal date.

---

## Validator Redesign

### Replace dynamic toggle registry

Current pattern uses mutable `toggle_tests`. Replace with explicit model-level
validators in each schema:

- `SystemConfig`: toggle-dependent field requirements (e.g., manning’s,
  hydrology, standalone SWMM)
- `AnalysisConfig`: run mode consistency + HPC/job mode dependencies +
  storm-tide/sensitivity dependencies

### Keep path checks intentional

Path existence checks should be explicit and context-aware (required-now vs
resolved-later), rather than broad implicit behavior that may block legitimate
deferred paths.

---

## Integration with New Profile Planning

Add a typed config model for curated profiles:

- `TestsAndCaseStudiesConfig`
  - `defaults`
  - `testcases`
  - `case_studies`

And a single precedence resolver implementing:

1. CLI args
2. selected testcase/case-study entry
3. profile defaults
4. analysis config
5. system config
6. internal defaults

This aligns with planning docs under `docs/planning/`.

---

## Migration Plan (Phased)

### Phase 1 — Structural split with compatibility shim

- Create new `config/` package modules.
- Move existing schemas and loaders with minimal behavior changes.
- Keep `config.py` re-export shim for compatibility.

### Phase 2 — Strictness + validator cleanup

- Enable `extra="forbid"` in strict mode.
- Replace toggle registry with explicit validators.
- Normalize error message style.

### Phase 3 — Profile schema + resolver

- Add `TestsAndCaseStudiesConfig`.
- Implement inheritance/merge resolver for testcase/case-study flows.

### Phase 4 — Hardening and docs

- Add/expand tests.
- Update documentation with strictness and migration guidance.

---

## Backward Compatibility Verification Checklist

1. Existing code paths still obtain `cfg_system` and `cfg_analysis` objects.
2. Existing workflows/tests that load system/analysis YAMLs still run.
3. No downstream import breakage from `TRITON_SWMM_toolkit.config`.
4. Error behavior improves without changing successful-run behavior.

---

## Test Plan

### Contract tests

- valid legacy configs still parse into expected model types.
- unknown fields fail with clear messages.

### Validator tests

- each toggle dependency rule fails/passes correctly.
- run mode consistency rules produce targeted errors.

### Profile tests

- profile selection + precedence merge behavior.
- CLI override dominance over profile defaults.

### Regression tests

- run core PC tests that rely on config loading unchanged.

---

## Open Decisions

1. Whether to support temporary non-strict loader mode and for how long.
2. Whether renamed class aliases (`SystemConfig`) should coexist with existing
   names (`system_config`) permanently or only during migration.
3. How strict to be on deferred-path existence checks by execution stage.
