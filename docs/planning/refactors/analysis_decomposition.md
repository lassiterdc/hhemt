---
impact: Medium
urgency: Low
loe: High
risk: Medium
priority: 0.67
priority-label: "Low value"
created: 2026-03-16
description: "Decompose TRITONSWMM_analysis: extract sensitivity delegation (RC-4), status aggregation (RC-1), and status table builder (RC-2) into focused modules."
---

<!-- Written: 2026-03-16 -->

# Decompose TRITONSWMM_analysis

## Task Understanding

### Requirements

1. **RC-4 — Sensitivity delegation**: Eliminate the 21 `if self.cfg_analysis.toggle_sensitivity_analysis: return self.sensitivity.X` pattern instances by introducing a property-factory or delegator mechanism.
2. **RC-1 — Status aggregation**: Extract `_update_log()` (~75 lines, lines 647–722) into a `LogStatusSnapshot` dataclass returned by a standalone function, reducing the method to a thin setter.
3. **RC-2 — Status table builder**: Extract the `df_status` property (~130 lines, lines 2098–2228) into a `StatusTableBuilder` class that is independently testable.
4. All existing tests must continue to pass without modification — this is a pure refactor with no behavioral changes.
5. No new dependencies.

### Assumptions

- `TRITONSWMM_sensitivity_analysis` already implements every property and method that `TRITONSWMM_analysis` delegates to it — the interface is implicit but complete.
- Tests in `test_PC_04`, `test_PC_05`, and `test_UVA_03` exercise `df_status` and sensitivity delegation but require HPC simulation outputs — local validation is limited to import checks and non-HPC test paths.
- The sensitivity toggle is set at `__init__` time and never changes during an analysis object's lifetime.

### Success Criteria

- `analysis.py` line count reduced by ~200+ lines.
- Zero `if self.cfg_analysis.toggle_sensitivity_analysis: return self.sensitivity.X` one-liner delegation patterns remain in `analysis.py` (complex methods like `submit_workflow`, `_print_resume_status`, and `__init__` that do more than just delegate are excluded from this criterion).
- `_update_log()` body reduced to ~10 lines (call snapshot function, write results to log).
- `df_status` property body reduced to ~5 lines (instantiate builder, call build, return).
- All existing tests pass unchanged.

---

## Evidence from Codebase

- **`analysis.py`** (2554 lines): the central orchestration class. Contains 21 sensitivity delegation sites across 17 methods/properties.
- **Delegation site classification**:
  - **Simple one-liner delegates** (11 sites — RC-4 targets): `scenarios_not_created`, `scenarios_not_run`, `classify_incomplete_sim_failures`, `all_scenarios_created`, `all_sims_run`, `all_TRITONSWMM_performance_timeseries_processed`, `TRITONSWMM_performance_time_series_not_processed`, `all_SWMM_timeseries_processed`, `TRITON_time_series_not_processed`, `SWMM_time_series_not_processed`, `all_TRITON_timeseries_processed`.
  - **Simple multiplier delegates** (2 sites): `n_scenarios`, `n_sims` — use `len(self.sensitivity.df_setup)` as a multiplier, not a direct delegation.
  - **Complex method delegates** (4 sites): `_update_log`, `df_status`, `df_snakemake_allocations`, `submit_workflow` — have distinct sensitivity vs non-sensitivity branches with different logic.
  - **Init/print delegates** (4 sites): `__init__` (creates sensitivity object), `_print_resume_status` (2 sites — flag globbing and node recommendation), `_retrieve_snakemake_allocations`.
- **`sensitivity_analysis.py`**: `TRITONSWMM_sensitivity_analysis` class — aggregates multiple sub-`TRITONSWMM_analysis` instances. Implements every property the simple delegates forward to.
- **`_update_log()`** (lines 647–722): 75 lines of boolean accumulator logic across 7 status dimensions, with a sensitivity branch that reads from `self.sensitivity` and a non-sensitivity branch that iterates scenarios. Both branches write to the same 7 log fields.
- **`df_status`** (lines 2098–2228): 130 lines mixing row assembly, log file I/O (`parse_triton_log_file`, `retrieve_swmm_performance_stats_from_rpt`), Snakemake allocation joining, and validation. Sensitivity branch delegates to `self.sensitivity.df_status` then joins allocations; non-sensitivity branch builds rows from scratch.
- **Tests**: `_update_log` called in `test_PC_01`, `test_PC_02`. `df_status` tested in `test_PC_04`, `test_PC_05`. These require simulation outputs not available locally.

---

## Implementation Strategy

### Chosen approach: Property factory for delegation + standalone extraction for _update_log and df_status

**Phase 1 — RC-4 (sensitivity delegation)**: Create a `_delegate_to_sensitivity(attr_name)` property factory function. For each of the 11 simple one-liner delegation properties, replace the manual `if toggle: return self.sensitivity.X / else: <body>` with a factory-generated property that handles the delegation automatically. The non-sensitivity body moves into a private method (`_<name>_impl`). This is explicit (each property is still declared in the class body), avoids `__getattr__` pitfalls (typo-silent delegation), and preserves IDE discoverability.

The 2 multiplier delegates (`n_scenarios`, `n_sims`) and 4 complex delegates (`_update_log`, `df_status`, `df_snakemake_allocations`, `submit_workflow`) stay as manual if/else — they have branch-specific logic that doesn't fit the factory pattern.

**Phase 2 — RC-1 (_update_log)**: Extract a `collect_log_status(analysis) -> LogStatusSnapshot` function into a new `src/TRITON_SWMM_toolkit/status.py` module. `LogStatusSnapshot` is a `@dataclass` with 7 boolean fields matching the log dimensions. `_update_log()` becomes: call `collect_log_status(self)`, write each field to `self.log`.

**Phase 3 — RC-2 (df_status)**: Extract a `StatusTableBuilder` class into the same `status.py` module. Constructor takes the analysis instance. `build() -> pd.DataFrame` method contains the current row-assembly, log-parsing, and Snakemake-join logic. The `df_status` property becomes: `return StatusTableBuilder(self).build()`.

### Alternatives considered

- **`__getattr__`-based delegation**: Rejected. Silent delegation of any attribute to `self.sensitivity` would mask typos and make the class harder to reason about. The property factory is equally concise but explicit.
- **Protocol/ABC for sensitivity interface**: Rejected as premature. Adding a formal `Protocol` class would be the right move if there were multiple delegation targets, but with exactly one (`TRITONSWMM_sensitivity_analysis`), it adds ceremony without value.
- **Move status methods to `sensitivity_analysis.py`**: Rejected. The status aggregation logic is about the analysis, not about sensitivity — it belongs with or near the analysis class.

### Trade-offs

- The property factory adds a small layer of indirection — each delegated property is now a factory call rather than an inline `if/else`. This is offset by eliminating 11 identical boilerplate patterns.
- Extracting `df_status` into `StatusTableBuilder` means a new class with a reference back to the analysis. This circular reference is acceptable (it's the same pattern `TRITONSWMM_analysis_post_processing` already uses).

---

## File-by-File Change Plan

### New files

| File | Purpose |
|------|---------|
| `src/TRITON_SWMM_toolkit/status.py` | `LogStatusSnapshot` dataclass, `collect_log_status()` function, `StatusTableBuilder` class. Houses all status aggregation and table-building logic extracted from `analysis.py`. |

### Modified files

| File | Change |
|------|--------|
| `src/TRITON_SWMM_toolkit/analysis.py` | **Phase 1**: Add `_delegate_to_sensitivity()` property factory. Replace 11 one-liner delegation properties with factory calls. Move each property's non-sensitivity body to a private `_<name>_impl` method. **Phase 2**: Replace `_update_log()` body with call to `collect_log_status()` + log field writes. **Phase 3**: Replace `df_status` body with `StatusTableBuilder(self).build()`. Remove imports no longer needed at the `analysis.py` level (e.g., `parse_triton_log_file` if only used by `df_status`). |

### Import sites

- `analysis.py` imports `LogStatusSnapshot`, `collect_log_status`, `StatusTableBuilder` from `status.py`.
- `status.py` uses `TYPE_CHECKING` guard for `from .analysis import TRITONSWMM_analysis` to avoid circular imports (same pattern as `sensitivity_analysis.py`).
- `status.py` imports `parse_triton_log_file`, `retrieve_swmm_performance_stats_from_rpt`, `TRITONSWMM_scenario`, and `pd` directly.

---

## Risks and Edge Cases

- **Circular import between `status.py` and `analysis.py`**: Mitigated by `TYPE_CHECKING` guard in `status.py`, same pattern used by `sensitivity_analysis.py` (line 14).
- **Property factory obscuring method bodies**: Each factory call is one line in the class, but the implementation moves to a `_<name>_impl` method in the same class. IDE "go to definition" still works.
- **HPC test regression**: `df_status` reads `log.out` files and `.rpt` files that only exist after HPC runs. Extraction must preserve exact behavior — no changes to logic, only to where the code lives. Validation requires running HPC tests (`test_PC_04`, `test_PC_05`).
- **Sensitivity property name coupling**: The property factory assumes `self.sensitivity.<attr_name>` exists for each delegated attribute. If `TRITONSWMM_sensitivity_analysis` ever drops or renames a property, this will raise `AttributeError` at runtime — same behavior as today, just surfaced via the factory instead of inline code.

---

## Validation Plan

### Local

```bash
# 1. Verify imports
conda run -n triton_swmm_toolkit python -c "
from TRITON_SWMM_toolkit.status import LogStatusSnapshot, collect_log_status, StatusTableBuilder
print('imports OK')
"

# 2. Verify analysis class still loads
conda run -n triton_swmm_toolkit python -c "
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
print('analysis import OK')
"

# 3. Run non-HPC tests
conda run -n triton_swmm_toolkit pytest tests/test_PC_01_singlesim.py -x -q

# 4. Syntax check
conda run -n triton_swmm_toolkit python -m py_compile src/TRITON_SWMM_toolkit/status.py
conda run -n triton_swmm_toolkit python -m py_compile src/TRITON_SWMM_toolkit/analysis.py
```

### HPC (paste results back)

```bash
# 5. Run multisim with Snakemake test (exercises df_status)
pytest tests/test_PC_04_multisim_with_snakemake.py -x -q
```

```
[paste output here]
```

```bash
# 6. Run sensitivity analysis test (exercises delegation + df_status)
pytest tests/test_PC_05_sensitivity_analysis_with_snakemake.py -x -q
```

```
[paste output here]
```

---

## Documentation and Tracker Updates

- Update `architecture.md`: add `status.py` to the Key Modules table with description "Status aggregation (LogStatusSnapshot) and status table builder (StatusTableBuilder) — extracted from analysis.py".
- Update `ideas.md`: remove Idea 3 entry after plan completion (see DoD).

---

## Decisions Needed from User

None — all design decisions resolved during planning.

---

## Definition of Done

- [ ] `src/TRITON_SWMM_toolkit/status.py` created with `LogStatusSnapshot`, `collect_log_status()`, `StatusTableBuilder`
- [ ] 11 one-liner sensitivity delegation properties in `analysis.py` replaced with `_delegate_to_sensitivity()` factory
- [ ] `_update_log()` body replaced with call to `collect_log_status()` + log field writes
- [ ] `df_status` property body replaced with `StatusTableBuilder(self).build()`
- [ ] Local smoke tests pass (import checks, `test_PC_01`)
- [ ] HPC tests pass (`test_PC_04`, `test_PC_05`)
- [ ] Update the workspace's architecture instruction file if module structure changed
- [ ] If any performance or memory risks were surfaced, entries added to `docs/planning/tech_debt_known_risks.md`
- [ ] Copy originating `ideas.md` Idea 3 entry verbatim into `## Appendix: Originating Idea`, then remove it from `docs/planning/ideas.md`
- [ ] When a plan moves to `completed/`, update any active plan `dependencies:` entries that reference the old path
- [ ] Before moving to `completed/`, run the pre-completion accuracy check from `prompts/instructions/protocols/plan-accuracy-gate.md`
- [ ] Set `completed: true` in this plan's YAML frontmatter, then move to `docs/planning/refactors/completed/` and run `scripts/generate_planning_tables.py --planning-dir docs/planning`

---

## Appendix: Originating Idea

```
## Idea 3: Decompose TRITONSWMM_analysis — extract status aggregation, status table, and sensitivity strategy

**Surfaced**: 2026-03-14
**Priority**: Medium
**Requires**: /plan-implementation (large refactor, regression risk)
**Description**: Three related structural problems in `analysis.py` identified by SE specialist audit:
- **RC-1**: `_update_log` (~100 lines, `analysis.py:~478–576`) contains nested boolean accumulators across six status dimensions duplicated between sensitivity and non-sensitivity branches. Extract to a `LogStatusSnapshot` dataclass returned by each branch — reduces method to ~15 lines.
- **RC-2**: `df_status` (~130+ lines, `analysis.py:~2100–2231`) mixes data assembly, I/O (log file parsing), and validation across three model types and two execution paths. Extract to a `StatusTableBuilder` class to make it independently testable.
- **RC-4**: ~10 properties repeat `if toggle_sensitivity_analysis: return self.sensitivity.X` pattern. A Strategy pattern or delegator helper would eliminate the repeated conditional dispatch and make each provider independently testable.
**Approach notes**: These three should be tackled together in a single plan since they all reduce the same root problem (analysis class doing too much). Run /plan-implementation with SE specialist consultation before starting.
**Related ideas**: none
```
