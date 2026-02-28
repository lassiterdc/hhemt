# Comprehensive Cruft Cleanup Plan

## Purpose

Create a structured, repo-wide cleanup plan for accumulated cruft across:

- main source code (`src/TRITON_SWMM_toolkit`)
- test suite (`tests`)
- documentation (`docs`)

This plan explicitly aligns with `CLAUDE.md` development philosophy.

---

## Governing Philosophy (from `CLAUDE.md`)

1. **Backward compatibility is generally not a priority**
   - Treat API backward-compatibility shims/aliases as **new cruft** unless explicitly justified.
   - Remove obsolete code instead of preserving legacy paths.
   - Do not add deprecation shims/aliases unless there is a compelling reason.
   - Update all call sites when introducing cleaner structures.

2. **Backward compatibility for code APIs is currently undesirable**
   - Compatibility-preserving patterns in runtime/library code increase branching, ambiguity,
     and maintenance burden.
   - Prefer clean replacement and immediate call-site migration.
   - During cleanup, PRs that add compatibility layers should be treated as regressions unless
     explicitly approved.

3. **Configuration formats are the exception**
   - Keep config format compatibility where practical.
   - Prefer explicit migration paths if stricter validation is introduced.

4. **Prefer log-based completion checks over file-existence checks**
   - Keep and strengthen `_already_written()`-style status semantics.
   - Avoid regressions to weak existence-only completion logic.

---

## Current Cruft Inventory (Findings)

## A) Source Code Cruft

### A1. Large mixed-concern modules (high priority)
- `config.py`: schema definitions, validation, path checks, loading, and display logic are all interleaved.
- `analysis.py`, `workflow.py`, `process_simulation.py`, `sensitivity_analysis.py`: high method counts and repeated orchestration patterns.

### A2. Legacy/obsolete residue (high priority)
- Explicit `_obsolete_*` methods and legacy fallback paths in runtime code (`run_simulation.py`, path fallback comments, compatibility aliases).
- Backward-compatibility aliases in paths/system layers that are contrary to current philosophy for code APIs.

### A3. Inconsistent execution/logging/error style (high priority)
- Extensive `print(...)` usage in library paths instead of structured logging.
- Mixed error style (`raise`, `warnings.warn`, silent returns in some paths, broad exception handlers).
- Command construction and subprocess patterns duplicated in multiple places.

### A4. Known-bug workaround sprawl (medium-high priority)
- `TODO(TRITON-OUTPUT-PATH-BUG)` appears in multiple modules.
- Necessary for now, but cross-cutting workarounds increase complexity and branching.

### A5. Partial/placeholder modules (medium priority)
- e.g., `swmm_full_model.py` contains TODO scaffolding and incomplete/dead design tracks.

---

## B) Test Suite Cruft

### B1. Duplication and verbosity (high priority)
- Similar test structures repeated across PC/UVA/Frontier variants.
- Large repeated fixture wiring in `tests/conftest.py` and fixture catalogs.

### B2. Diagnostic noise in tests/utilities (medium-high priority)
- Significant use of `print(...)` in tests and test utilities, including large diagnostic dumps in routine success paths.

### B3. Placeholder/skip accumulation (medium priority)
- Multiple `pytest.skip(...)` placeholders and tests with partially implemented intent.

### B4. Assertion-style inconsistency (medium priority)
- Mix of direct file existence checks and higher-level completion semantics.
- Some helpers align with log-based completion philosophy; others still skew filesystem-first.

---

## C) Documentation Cruft (newly expanded scope)

### C1. Plan/summary drift and stale implementation docs (high priority)
- `docs/implementation/` contains both active references and one-off “completed” snapshots that may now be stale.
- Several implementation docs contain contradictory status narratives (e.g., "complete" at top but
  "not started" in phase sections), creating contributor confusion.

### C2. Prompt-oriented internal workflow docs mixed with user docs (medium-high priority)
- `docs/prompts/` includes internal phase prompts/checklists that may not belong in long-term user-facing docs.

### C3. Potentially obsolete planning artifacts (medium priority)
- Multiple overlapping plans/roadmaps with partial completion markers and draft notes.
- Risk: contradictory guidance for contributors.

---

## Cleanup Strategy (Prioritized Sequence)

## Phase 0 — Baseline & Safety Rails

Goal: prevent regressions while cleanup is underway.

Tasks:
- Define a focused “cleanup regression suite” (fast subset of core tests).
- Add a simple inventory tracker for touched modules/files per phase.
- Establish rule: no net-new legacy aliases unless explicitly approved.

Required smoke-test sequence (run in this exact order):
1. `tests/test_PC_01_singlesim.py`
2. `tests/test_PC_02_multisim.py`
3. `tests/test_PC_04_multisim_with_snakemake.py`
4. `tests/test_PC_05_sensitivity_analysis_with_snakemake.py`

Exit criteria:
- Baseline tests green before and after each phase.

---

## Phase 1 — Configuration Layer Refactor (`config.py` first)

Goal: reduce complexity and strictify validation architecture.

Tasks:
- Split `config.py` into focused modules (schema/validation/loading/display).
- Replace dynamic `toggle_tests` registry with explicit validators.
- Enforce strict unknown-key behavior in primary path (`extra="forbid"` equivalent policy).
- Preserve config compatibility where practical; document any migration points.

Exit criteria:
- Existing config-driven workflows still run.
- Validation errors become more explicit/actionable.

---

## Phase 2 — Remove Legacy/Obsolete Runtime Paths

Goal: align runtime code with “clean replacement over compatibility shims.”

Tasks:
- Remove `_obsolete_*` methods and dead launch paths.
- Remove legacy API aliases in code paths where not needed.
- Collapse fallback branches that only exist for retired structures.
- Reject/avoid new compatibility shims that preserve retired APIs.

Exit criteria:
- No `_obsolete` code remains in runtime modules.
- Fewer ambiguous execution branches.
- No net-new compatibility alias/shim paths introduced.

---

## Phase 3 — Orchestration Deduplication

Goal: consolidate repeated flow across analysis/workflow/sensitivity orchestration.

Tasks:
- Extract shared command-building and subprocess-launch patterns.
- Unify model-routing logic currently repeated across multiple modules.
- Normalize workflow submission contract and status payloads.

Exit criteria:
- Reduced duplicated code paths for launch/process/consolidate flow.
- Equivalent behavior for local/slurm/sensitivity execution.

---

## Phase 4 — Logging & Error Contract Normalization

Goal: consistent operational behavior and easier debugging.

Tasks:
- Replace ad-hoc `print(...)` in library/runtime paths with structured logging.
- Standardize exception boundaries and message format.
- Remove silent returns for failure states in critical pipelines.
- Keep log-based completion checks as canonical.

Exit criteria:
- Consistent log/error behavior across runners and orchestration layers.

---

## Phase 5 — Workaround Containment and Retirement Plan

Goal: isolate bug-workaround cruft and define clean removal trigger.

Tasks:
- Centralize all `TODO(TRITON-OUTPUT-PATH-BUG)` logic behind minimal interfaces.
- Maintain one canonical doc for workaround rationale and removal conditions.
- Prepare a single removal PR template for when upstream behavior is fixed.

Exit criteria:
- Workaround logic is traceable and non-duplicative.

---

## Phase 6 — Test Suite Cleanup

Goal: make tests easier to maintain and less noisy.

Tasks:
- Parametrize repeated platform test patterns where practical.
- Consolidate fixture factories and reduce redundant fixtures.
- Reduce unconditional diagnostic prints; keep verbose diagnostics opt-in.
- Standardize assertions around completion semantics.

Exit criteria:
- Lower duplication and clearer failure signals.

---

## Phase 7 — Documentation Cleanup & Archival Policy

Goal: remove doc cruft and keep only current, authoritative guidance.

Tasks:
- Classify all docs into: **Authoritative**, **Active Plan**, **Historical Archive**, **Delete**.
- Move stale implementation snapshots to an explicit archive location (or consolidate).
- Prune `docs/prompts/` content that is no longer relevant to current workflow.
- Remove/replace references to external private plan files and stale status claims.
- Add a “doc freshness” checklist for future PRs touching architecture/workflow.

Proposed classification rubric:
- **Authoritative**: currently used by contributors/users; kept near top-level docs.
- **Active Plan**: not yet implemented; has owner, date, and current status.
- **Historical Archive**: completed historical context, not actionable for current dev.
- **Delete**: obsolete, duplicated, or misleading.

Exit criteria:
- No stale “active” docs remain untagged.
- Main docs point to a single authoritative source per topic.

---

## Documentation Triage Backlog (Initial Candidates)

## Candidate: immediate refresh (high confidence drift)
- `docs/implementation/multi_model_integration.md` (top-level complete status conflicts with
  phase-by-phase "not started" markers).
- `docs/implementation/multi_model_output_processing_plan.md` (top-level complete status conflicts
  with unfinished "Known Issues / Next Steps / Debugging Plan" sections).

## Candidate: likely archive or consolidate
- `docs/implementation/*_summary.md` files with completed one-off status narratives.
- `docs/prompts/refactor_prompt.md`, `docs/prompts/test_refactor_plan.md` (if no longer active workflow).

## Candidate: keep as authoritative implementation notes
- `docs/implementation/triton_output_path_bug.md` (until upstream fix lands).
- Current architecture/spec docs in `docs/planning/` that still drive active changes.

## Candidate: refresh required
- Roadmap/spec docs with draft/open-decision markers that may be outdated after implementation progress.

---

## PR Slicing Recommendation

Keep cleanup incremental and reviewable:

1. `config` structural split + validator cleanup
2. runtime obsolete-path removal
3. orchestration dedup extraction
4. logging/error normalization
5. test refactor and fixture consolidation
6. docs triage/archive pass

Each PR should include:
- scope-bound changes,
- targeted regression tests,
- explicit list of deleted cruft,
- doc updates for any changed behavior.

---

## Acceptance Criteria for “Cruft Cleanup Complete”

1. No obsolete runtime paths (`_obsolete`, legacy aliases) remain without explicit justification.
2. Core orchestration paths are deduplicated and easier to reason about.
3. Logging/error behavior is consistent and actionable.
4. Test suite has materially reduced duplication/noise with preserved coverage intent.
5. Documentation set is triaged, archived where appropriate, and internally consistent.
6. Cleanup decisions are aligned with `CLAUDE.md` philosophy (especially no compatibility-shim creep).

---

## Progress Notes (2026-02-07)

**Phase 7 (Documentation Cleanup):** Partial progress. Completed doc audit and
archival pass — 12 documents moved to `docs/archived/`, stale statuses updated
in remaining implementation docs. See `docs/planning/priorities.md` for current
priority ordering.

**Related completed work:**
- Model-specific logs (commit d0e7b7a) addressed items in A2 (legacy residue)
  and A3 (logging style) **partially** by eliminating shared `log.json` and introducing
  clean per-model logging. Deprecated simlog structures/usages still remain for follow-up cleanup.
- Examples refactor (Phases 1-3) addressed B1 (duplication) partially.

## Immediate Next Actions

1. Start Phase 1 (`config.py` split + strict validator redesign) using `docs/planning/refactors/config_py_refactor_plan.md` as seed input.
2. Open first cleanup PR with an explicit "removed cruft" changelog section.
