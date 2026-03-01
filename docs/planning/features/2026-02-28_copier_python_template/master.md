# Copier Template System: Master Plan

**Written**: 2026-02-28
**Last edited**: 2026-02-28 — Phase 1 complete; repo at github.com/lassiterdc/copier-python-template tagged v1.0.0

---

## Overview

This is a four-phase plan for building and adopting a Copier-based Python project template system named `copier-python-template`.

| Phase | Goal | Scope | Status | Doc |
|-------|------|-------|--------|-----|
| **Phase 1** | Build the `copier-python-template` repo | New repo; no changes to existing projects | ✅ Complete | `implemented/1_build_template.md` |
| **Phase 2** | Build the `copier-specialist` agent | New agent in `claude-workspace`; no changes to existing projects | Pending | `2_build_copier_specialist.md` |
| **Phase 3** | Spin up `multidriver-swg` via the copier-specialist; verify ReadTheDocs; walk through the update tutorial | New repo; template repo may receive minor fixes | Pending | `3_spin_up_multidriver.md` |
| **Phase 4** | Formalize the `copier update` workflow as a reference document; fully initialize copier-specialist | Reference doc written in template repo; agent startup reads updated | Pending | `4_update_workflow_reference.md` |
| **Phase 5** | Retroactively adopt Copier in the TRITON-SWMM toolkit | Modifies existing toolkit repo | Pending | `5_toolkit_adoption.md` |

Phases must be completed in order — each depends on the previous. Use `@.prompts/proceed_with_implementation.md` with the relevant phase doc at the start of each phase. From Phase 3 onward, use the `copier-specialist` agent to perform Copier operations.

---

## Dependencies (Resolved)

### Subagent Refactor (`2026-02-28_system-level-subagents-in-git-repo.md`) — COMPLETE

The refactor promoting project-level agents to user-level (`~/.claude/agents/`) tracked in `~/dev/claude-workspace/` is complete. All three touch points originally gated on this refactor are now resolved:

| Touch point | Resolution |
|-------------|------------|
| **Phase 1 — `template/.claude/agents/` contents** | Generate `README.md` redirect only (no agent stub file). Agent example pattern documented in `claude-workspace/README.md` instead. |
| **Phase 1 — agent frontmatter pattern** | No `skills:` frontmatter. Agents are project-agnostic; context passed explicitly per-invocation via `@` references as documented in `.prompts/conventions.md`. |
| **Phase 4 — pre-migration audit table** | Toolkit's `.claude/agents/` already contains only `README.md`. Template generates same; resolution is trivially "accept template's". |

**Also relevant**: once `multidriver-swg` has meaningful source code, it may need an entry in `claude-workspace/README.md`. Out of scope for all phases here — natural follow-on task.

---

## Cross-Phase Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Template engine | Copier (not Cookiecutter) | `copier update` propagation mechanism is decisive advantage |
| Repo name | `copier-python-template` | Accurate to engine; generic enough for reuse |
| Docs framework | MkDocs + Material (not Sphinx) | Lower barrier, Markdown consistency, better fit for general template |
| Agent stub location | `claude-workspace/README.md` only | `setup.sh` globs all `agents/*.md` — any stub file would be symlinked as a real agent |
| `template/.claude/agents/` | `README.md` redirect only | Matches toolkit pattern exactly |
| Part III of `conventions.md` | Fully portable verbatim | After deletion of agents_archive reference, no toolkit-specific content remains |
| `conventions.md` Code style | Strip `cfgBaseModel` and `Literal` bullets | Project-specific to TRITON-SWMM Pydantic config system |
| QA gate per phase | Baked into `proceed_with_implementation.md` step 5 | Cleaner than duplicating in each phase DoD |
