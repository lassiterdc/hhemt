# Prompt & CLAUDE.md Cleanup Plan

**Written**: 2026-02-27
**Last Edit**: 2026-02-27 — Phase 3 completed; comparison analysis done; all .prompts files updated; CLAUDE.md freshness fixes applied; agent_files_audit.md expanded with migration tracking and architecture.md idea

---

## Overview

The goal is to bring the best ideas from the `ss-fha` project's prompt/documentation system into this repo, while trimming cruft that has accumulated in `CLAUDE.md` and `docs/prompts/`. This is a documentation and workflow improvement initiative, not a code change.

**Baseline direction**: ss-fha prompt documents are the authoritative baseline. The strategy is to use ss-fha prompts as the starting point and make deliberate decisions about what TRITON-SWMM-specific content to merge in — not the other way around. This prevents the natural tendency to preserve legacy content by default.

**Existing TRITON-SWMM toolkit prompts**: Because these prompts have rarely been used in practice, they will be staged in `.prompts/_to_verify/` rather than deleted outright. Each will be reviewed individually and either kept, merged, or discarded (see Table 3).

The user should use the decision tables below to mark which elements to carry over. A recommendation column (`Rec.`) is provided.

---

## What's Good in Each Repo

### ss-fha Strengths

| # | Strength | Notes |
|---|----------|-------|
| 1 | **`conventions.md` is a standalone file** | Separates "how we work" from "what the project does". CLAUDE.md stays focused on architecture; philosophy is invoked explicitly when needed. Also surfaces as a living document AI can be asked to check against. |
| 2 | **`proceed_with_implementation.md`** | A two-phase preflight + execution gate. First call: asks whether to use Opus subagent, runs freshness check on planning docs, checks alignment with philosophy, and returns a structured "preflight report" before any code is touched. Second call: final check + explicit approval + implementation. This replaces ad-hoc planning patterns. |
| 3 | **`qaqc_and_commit.md`** | Structured post-implementation review: verifies success criteria, checks philosophy alignment, produces a structured report with an "input needed" section, and handles commit coordination. Currently this repo has only `commit_phase.md` which covers only the commit mechanics, not the QA review. |
| 4 | **`implementation_plan.md` includes plan self-check** | After drafting the plan, the AI performs a header/body alignment check, section necessity check, and philosophy alignment check. This version also has a `#user:` comment convention for marking items that must be addressed before implementation. |
| 5 | **`implementation_plan.md` includes a header with edit datetime** | Each plan tracks when it was written and last edited, with a summary of the edit. Helps with stale plan detection. |
| 6 | **Terminology section in `conventions.md`** | Precise vocabulary definitions prevent AI/human miscommunication. E.g., "compound" vs. "combined", `event_iloc`, `ss` vs. `ensemble`. TRITON-SWMM toolkit has no equivalent. |
| 7 | **Philosophy items not in TRITON-SWMM CLAUDE.md** | Several philosophy rules exist only in ss-fha: "most function arguments should not have defaults", "no shims for poorly formatted inputs", "avoid aliases", "all constants in constants.py", "all variables/imports must be used", "functions have docstrings/type hints". |
| 8 | **Utility package candidates tracking** | `docs/planning/utility_package_candidates.md` tracks project-agnostic functions that could be shared between repos. Smart forward-looking practice. |
| 9 | **`'#user:' comment convention in planning docs**` | Developer comments in planning docs prefixed with `#user:` must ALL be addressed before implementation. Forces AI to treat developer feedback as blocking. |
| 10 | **`proceed_with_implementation.md` asks about Opus delegation** | Before heavy implementation, prompts user to decide if task warrants a more capable subagent. Prevents expensive context from being wasted on the wrong model. |

---

### TRITON-SWMM Toolkit Strengths

| # | Strength | Notes |
|---|----------|-------|
| A | **`debugging_hpc_analysis.md`** | Comprehensive HPC debugging protocol — 9-step systematic workflow covering log file locations, SLURM error patterns, srun resource mismatch diagnosis, report writing procedure, and empirical testing scaffolding. Highly evolved from operational experience. ss-fha doesn't have an equivalent yet (though it could be adopted). |
| B | **`.claude/agents/` specialist agents** | Subsystem-specific guidance for Snakemake, SLURM, Pydantic config, SWMM model gen, output processing, testing, sensitivity analysis. Keeps CLAUDE.md focused by moving deep domain knowledge to specialized files invoked contextually. |
| C | **Agent documentation update checklist in CLAUDE.md** | Explicit checklist for when to update CLAUDE.md vs. agent files. Prevents documentation drift. |
| D | **`doc_freshness_check.md`** | Prompt to systematically check whether CLAUDE.md and agent docs need updating after code changes. ss-fha doesn't have this. |
| E | **`smoke_tests.md`** | Explicit smoke test sequence with notes (e.g., "do NOT impose artificial timeouts on PC_05"). Small but useful for onboarding AI to test expectations. |
| F | **`validate_and_proceed.md`** | Orchestration prompt that validates tracker/priorities docs, fixes inconsistencies, and moves to next priority. Useful for maintaining momentum during long multi-phase work. |
| G | **`README.md` in `.prompts/`** | Index of all available prompts with brief usage notes. Low-cost but high-value for discoverability. |

---

## Current State: What Needs Cleanup

### `docs/prompts/` — Historical Artifacts (Not Active Prompts)

These are **completed planning documents** that were placed in `docs/prompts/` during the early phase of the project. They are no longer active working documents and are causing confusion because they live alongside the active `.prompts/` directory. They belong in `docs/planning/` (either under `completed/` or `active/`).

| File | Status | Recommendation |
|------|--------|----------------|
| `refactoring_plan.md` | Complete (Jan 27, 2026) | Move to `docs/planning/completed/` |
| `swmm_output_parser_optimization_plan.md` | Phase 5 complete | Move to `docs/planning/completed/` |
| `test_refactor_plan.md` | Phase 1 complete, Phase 2 future | Move to `docs/planning/active/refactors/` |
| `one_job_many_srun_tasks_plan.md` | Partially complete, ongoing | Move to `docs/planning/active/features/` |
| `refactor_prompt.md` | One-time use senior engineer review prompt | Move to `.prompts/` if it could be reused; otherwise archive |
| `phase13_folder_structure_analysis.md` | Complete analysis, conclusion: no action | Move to `docs/planning/completed/` |

### `CLAUDE.md` — Sections to Trim or Restructure

| Section | Issue | Recommendation |
|---------|-------|----------------|
| "Backward Compatibility" (with code example) | Good philosophy content, but wrong location. Should live in `conventions.md`. The code example adds length without much value for an AI that already understands the concept. | Move to `conventions.md`; keep a one-line summary in CLAUDE.md |
| "Completion Status: Log-Based Checks..." | Philosophy content, not architectural reference | Move to `conventions.md` |
| "Logging & Error Handling" — Logging Patterns subsection | Duplicates patterns better discovered from code. The `print([NAMESPACE])` pattern is clear enough in code. | Trim to exception hierarchy only; move logging philosophy to `conventions.md` |
| "Gotcha #5" (TRITON output path bug) | Already marked as resolved 2/9/2026. | Delete this gotcha entirely |
| Multi-model integration section | Accurate and useful reference | Keep as-is |
| Three-layer hierarchy / Key Modules table | Core architectural reference | Keep as-is |
| Test assertion patterns | Long (~30 lines) and detailed. The agent file carries the full detail, but CLAUDE.md should retain enough for Claude to know the helpers exist in ordinary (non-agent) conversations. | Replace full listing with a 3–4 line summary of key helper names and what they assert, plus a pointer to `triton-test-suite.md` for full reference. |
| "When to Update Agent Documentation" | Good process, but very long | Keep condensed; remove per-agent bullet lists since CLAUDE.md already has the checklist |

---

## Decision Tables

### Table 1: ss-fha Elements to Adopt (All Adopted ✅)

All items below were adopted by the developer. Notes capture any nuance in how they should be applied.

| # | Element | Where it will live | Notes |
|---|---------|---------------------|-------|
| 1 | `conventions.md` as standalone file | `.prompts/conventions.md` | Use ss-fha version as base; replace ss-fha-specific terminology with TRITON-SWMM equivalents. Merge in content extracted from CLAUDE.md (Table 2, item H). Avoid redundancy — when an existing section is related to something in ss-fha's philosophy, decide case-by-case whether to add, clarify, or skip. |
| 2 | `proceed_with_implementation.md` | `.prompts/proceed_with_implementation.md` | Use ss-fha version as base; remove ss-fha-specific references (e.g., `full_codebase_refactor.md`). |
| 3 | `qaqc_and_commit.md` | `.prompts/qaqc_and_commit.md` | Use ss-fha version as base. Decision: absorb `commit_phase.md` mechanics into this file as a subsection; retire `commit_phase.md`. See "Decision" section below. |
| 4 & 5 | `implementation_plan.md` — ss-fha version as base | `.prompts/implementation_plan.md` | Use ss-fha version as baseline. Merge in any TRITON-SWMM-specific content (e.g., smoke test references, agent file mentions) as additions. Includes: plan self-check, edit datetime header, `#user:` convention (items 4, 5, 12, 15). |
| 6 | Terminology section | Inside `.prompts/conventions.md` | Replace ss-fha terminology with TRITON-SWMM equivalents. Confirmed terms to include: `model_type` (`triton`, `tritonswmm`, `swmm`) vs. `run_mode` (`serial`, `openmp`, `mpi`, `gpu`, `hybrid`) — these are frequently confused; `multi_sim_run_method` vs. `run_mode` — different axes of configuration; `event_iloc` — flat integer index (already in CLAUDE.md, migrates here); `in_slurm` — derived boolean, not just the env var. Scope to terms where misuse causes real errors, as in ss-fha. |
| 7 | "Most arguments should not have defaults" | Inside `.prompts/conventions.md` | Add as new section. |
| 8 | ~~"No shims for poorly formatted inputs"~~ | ~~Inside `.prompts/conventions.md`~~ | **Excluded** — no direct TRITON-SWMM equivalent at this time. The ss-fha version is specifically about HydroShare data formatting; this repo has no analogous concept. Revisit if a relevant pattern emerges. |
| 9 | "Avoid aliases" | Inside `.prompts/conventions.md` | Add as clarification/extension to existing no-backward-compatibility section rather than a separate entry, unless the concept is meaningfully distinct. |
| 10 | "All variables/imports/args must be used" | Inside `.prompts/conventions.md` | Add with investigation protocol intact. |
| 11 | "Functions have docstrings, type hints" | Inside `.prompts/conventions.md` | Add as a rule. Apply organically going forward: `qaqc_and_commit.md` already prompts checking touched scripts against philosophy, so adherence will accumulate naturally without a global retrofit pass. |
| 12 | `#user:` blocking comment convention | `.prompts/implementation_plan.md` + `conventions.md` | Bundled with items 4 & 5. |
| 13 | Opus delegation in `proceed_with_implementation.md` | Bundled with item 2 | No separate action needed. |
| 14 | Utility package candidates tracking | `docs/planning/utility_package_candidates.md` | Create as a stub. Low-effort, low-risk — merging across repos later will be trivial. |
| 15 | Philosophy check in `implementation_plan.md` self-check | Bundled with item 4 | No separate action needed. |

---

### Table 2: Cleanup Actions in This Repo

Mark `x` in "Do it?" for cleanup actions to take.

| # | Action | What changes | Impact | Rec. | Do it? |
|---|--------|-------------|--------|------|--------|
| A | Move `docs/prompts/refactoring_plan.md` to `docs/planning/completed/` | File location only | Reduces confusion | ✅ Yes |x |
| B | Move `docs/prompts/swmm_output_parser_optimization_plan.md` to `docs/planning/completed/` | File location only | Reduces confusion | ✅ Yes |x |
| C | Move `docs/prompts/phase13_folder_structure_analysis.md` to `docs/planning/completed/` | File location only | Reduces confusion | ✅ Yes |x |
| D | Move `docs/prompts/test_refactor_plan.md` to `docs/planning/active/refactors/` | File location only | Reduces confusion | ✅ Yes |x |
| E | Move `docs/prompts/one_job_many_srun_tasks_plan.md` to `docs/planning/active/features/` | File location only | Reduces confusion | ✅ Yes |x |
| F | Move `docs/prompts/refactor_prompt.md` to `.prompts/` (rename to `architect_review.md`) | File location + rename | Makes it discoverable as a reusable prompt | ✅ Yes |x |
| G | Delete empty `docs/prompts/` directory | Directory removal | Removes confusing shadow of `.prompts/` | ✅ Yes |x|
| H | Extract philosophy content from CLAUDE.md into `conventions.md` | CLAUDE.md shrinks; new `.prompts/conventions.md` created | Cleaner separation of concerns | ✅ Yes (contingent on item 1 in Table 1) |x |
| I | Delete resolved Gotcha #5 from CLAUDE.md | CLAUDE.md edit | Removes stale content | ✅ Yes |x |
| J | Trim test assertion patterns section in CLAUDE.md | CLAUDE.md edit | CLAUDE.md is ~30 lines shorter; refer to agent file | ✅ Yes | x|
| K | Upgrade `.prompts/implementation_plan.md` with ss-fha improvements | Merge changes | Better planning quality | ✅ Yes (contingent on adopted items from Table 1) | x|
| L | Update `.prompts/README.md` to reflect new prompts | `.prompts/README.md` edit | Keeps index current | ✅ Yes |x|
| M | Add "philosophy check" step to `doc_freshness_check.md` | `.prompts/doc_freshness_check.md` edit | Keeps conventions.md current after changes | ✅ Yes (if conventions.md adopted) |x |

---

### Table 3: Existing `.prompts/` Files — Staging for Review

All existing TRITON-SWMM toolkit prompt files will be moved to `.prompts/_to_verify/` during the initial migration. They are not deleted — they wait there for individual review. For each file, the outcome column will be filled in as decisions are made.

**Outcome options**: `keep-as-is` | `merge-into:<target>` | `edit-then-keep` | `discard`

| File | What it does | Initial assessment | Outcome | Notes |
|------|-------------|-------------------|---------|-------|
| `implementation_plan.md` | Planning prompt | Superseded by ss-fha version (items 4+5 above) | `merge-into:implementation_plan.md` | TRITON-SWMM-specific guardrails (smoke test references, agent file mentions) should be extracted and merged into the new ss-fha-based version before this is discarded |
| `commit_phase.md` | Commit message mechanics | Superseded by `qaqc_and_commit.md` | `merge-into:qaqc_and_commit.md` | Extract commit message format/types section; absorb into `qaqc_and_commit.md`; then discard |
| `smoke_tests.md` | Smoke test sequence with caveats | No ss-fha equivalent; high practical value | TBD | |
| `validate_and_proceed.md` | Validate trackers + proceed to next priority | No ss-fha equivalent; useful for long multi-phase work | TBD | |
| `next_priority.md` | Identify next task from priorities.md | No ss-fha equivalent; lightweight but handy | TBD | |
| `doc_freshness_check.md` | Check if CLAUDE.md / agent docs need updates | No ss-fha equivalent; especially valuable given `.claude/agents/` system | TBD | Will need `conventions.md` added to checklist (Table 2, item M) |
| `import_audit.md` | Verify imports after refactor | No ss-fha equivalent; only needed occasionally | TBD | |
| `debugging_hpc_analysis.md` | HPC debugging protocol | No ss-fha equivalent; highly evolved, operationally critical | TBD — likely `keep-as-is` | Strong candidate for adoption in ss-fha too |
| `update_tracker.md` | Update tracker docs after phase completion | Workflow maintenance prompt | TBD | |
| `architect_review.md` | Whole-codebase senior engineer review prompt | Written for the completed 13-phase refactor; references `docs/refactoring_plan.md` (gone). Core concept is reusable but needs updating. | TBD | References stale paths; assess whether the structured output format is still useful |
| `README.md` | Index of all prompts | Will need full rewrite regardless | `edit-then-keep` | Rewrite once final `.prompts/` structure is settled |

---

## Philosophy Content to Extract from CLAUDE.md

If `conventions.md` is adopted (Table 1, item 1), these CLAUDE.md sections would move there:

| CLAUDE.md Section | Content to Move |
|-------------------|----------------|
| "Backward Compatibility" | Full section + rationale, minus code example |
| "Completion Status: Log-Based Checks" | Full section |
| "Logging Patterns" subsection of "Logging & Error Handling" | The `print([NAMESPACE])` guidance |
| "Error Handling" subsection | The fail-fast / preserve context / raise custom exceptions rules |
| "Development Philosophy" heading | Entire section (currently only contains Backward Compatibility + Log-Based Checks) |

CLAUDE.md would retain only the architecture/reference content: three-layer hierarchy, key modules, runner scripts, multi-model integration, configuration system, HPC/SLURM integration, workflow phases, conda environments, testing, code style, gotchas, and agent documentation guidance.

---

## Proposed New `.prompts/` Structure

After the initial migration, `.prompts/` will have this structure. Files in `_to_verify/` are awaiting individual review (Table 3).

```
.prompts/
├── README.md                         # REWRITE: updated index once structure settled
├── conventions.md                     # NEW: ss-fha base + TRITON-SWMM content
├── implementation_plan.md            # NEW: ss-fha base + TRITON-SWMM additions
├── proceed_with_implementation.md    # NEW: ss-fha base, ss-fha refs removed
├── qaqc_and_commit.md               # NEW: ss-fha base + commit_phase.md mechanics absorbed
│
└── _to_verify/                       # Existing prompts staged for review
    ├── architect_review.md           # MOVED from docs/prompts/refactor_prompt.md — needs triage
    ├── smoke_tests.md
    ├── validate_and_proceed.md
    ├── next_priority.md
    ├── doc_freshness_check.md
    ├── import_audit.md
    ├── debugging_hpc_analysis.md
    ├── update_tracker.md
    ├── [implementation_plan.md]      # Original — extract TRITON-SWMM content, then discard
    └── [commit_phase.md]             # Mechanics absorbed into qaqc_and_commit.md, then discard
```

**Decision: `commit_phase.md` → absorbed into `qaqc_and_commit.md`**. The commit message format and type conventions from `commit_phase.md` become a subsection of `qaqc_and_commit.md`. The original `commit_phase.md` is staged in `_to_verify/` for reference during the merge, then discarded.

**Note on `qaqc_and_commit.md` and `conventions.md` staying in sync**: No manual sync is needed. `qaqc_and_commit.md` instructs the AI to *read* `conventions.md` at review time, so new philosophy rules are picked up automatically without any changes to `qaqc_and_commit.md`.

**Note on `.claude/agents/` files**: Agent file cleanup is tracked in a separate plan (`docs/planning/active/refactors/agent_files_audit.md`). The test assertion patterns decision (keep a short summary in CLAUDE.md, full detail in the agent file) is intentionally modest until agent files are audited and trustworthy enough to invoke confidently.

---

## Definition of Done

### Phase 1 — Migration & Cleanup (no content authoring)
- [x] `docs/prompts/` files relocated per Table 2 (A–G)
- [x] `docs/prompts/` directory removed
- [x] Existing `.prompts/` files moved to `.prompts/_to_verify/`
- [x] `docs/planning/utility_package_candidates.md` stub created (item 14)

### Phase 2 — New prompt authoring (ss-fha as base)
- [x] `conventions.md` written: ss-fha base + TRITON-SWMM terminology + CLAUDE.md philosophy content extracted
- [x] `implementation_plan.md` written: ss-fha base + TRITON-SWMM additions merged in
- [x] `proceed_with_implementation.md` written: ss-fha base, ss-fha-specific refs removed
- [x] `qaqc_and_commit.md` written: ss-fha base + `commit_phase.md` commit mechanics absorbed

### Phase 3 — CLAUDE.md cleanup
- [x] Philosophy content removed from CLAUDE.md (Table 2, item H) with pointer to `conventions.md`
- [x] Gotcha #5 deleted (Table 2, item I)
- [x] Test assertion patterns trimmed (Table 2, item J)
- [x] `architect_review.md` moved to `_to_verify/` for triage (Table 2, item F — revised: staged rather than promoted directly)
- [x] `conventions.md` added to Documentation Update Checklist in CLAUDE.md (Table 2, item M)
- [x] `conventions.md` elevated to mandatory read: CLAUDE.md now instructs "Read `.prompts/conventions.md` before beginning any task"
- [x] CLAUDE.md freshness fixes: broken paths fixed (`frontend_validation_checklist`, `priorities.md`), Gotcha #1 corrected (`hpc_max_simultaneous_sims` is `batch_job` mode, not `1_job_many_srun_tasks`)
- [x] CLAUDE.md trimmed: Preflight Validation, Toggle-Based Validation, 1-Job-Many-srun-Tasks, Conda Architecture, Workflow Phases "why" bullets, Development Priorities subsection
- [x] Agent files obsolete notice added to CLAUDE.md "Specialized Agent Documentation"
- [x] CLAUDE.md freshness check added to `qaqc_and_commit.md` Step 2b
- [x] All conventions.md additions: plan-deviation nuance, utility candidates rule, pyright/Pylance section, GIS data type preference, testing philosophy section
- [x] "Confirm before spawning subagents" rule removed from `conventions.md` (lives in CLAUDE.md only)
- [x] `implementation_plan.md` guardrail restored to ss-fha phrasing ("don't assume user is expert")
- [x] `proceed_with_implementation.md` preamble added
- [x] `qaqc_and_commit.md` philosophy sections explicitly named (all 5 sections listed)

### Phase 4 — `_to_verify/` triage (ongoing, one file at a time)
- [x] Each file in `_to_verify/` reviewed; Table 3 "Outcome" column filled in
- [x] Accepted files moved to `.prompts/`; discarded files deleted (`_to_verify/` directory removed)
- [x] `.prompts/README.md` — decided no README needed; 6 self-evident files, CLAUDE.md dispatches to directory
- [x] `doc_freshness_check.md` — discarded (file was in `_to_verify/`, not promoted); `conventions.md` referenced directly in `qaqc_and_commit.md` Step 2 instead

### Out of scope (tracked separately)
- Agent file audit and cleanup → `docs/planning/refactors/agent_files_audit.md`
