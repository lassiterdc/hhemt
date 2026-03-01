# Phase 2: Build the Copier-Specialist Agent

**Part of**: `master.md` — Copier Template System
**Written**: 2026-02-28
**Last edited**: 2026-02-28 — Phase 2 complete

> **⚠ Staleness notice (2026-02-28)**: The startup reads reference to a future `copier_update_workflow.md` (line 37) reflects the original Phase 4 plan. Phase 4 has since been merged into Phase 3 — the reference doc will be `COPIER_UPDATE_GUIDE.md` at the template repo root, not under `docs/planning/reference/`. Update the agent's startup reads when Phase 3 completes.

---

## What Was Built

A `copier-specialist` agent in `~/dev/claude-workspace/agents/` symlinked to `~/.claude/agents/`. The agent assists with all Copier-related tasks across the project ecosystem.

### Files created/modified

| File | Action |
|------|--------|
| `~/dev/claude-workspace/agents/copier-specialist.md` | Created — agent file with frontmatter (including `<example>` blocks), startup reads, six capability areas, gotchas, conflict resolution, version tracking |
| `~/.claude/agents/copier-specialist.md` | Symlink to above |
| `~/dev/claude-workspace/README.md` | Updated — added `claude-specialist` and `copier-specialist` rows to Available Agents table |

### Agent Scope

Six core capability areas documented in the agent body:

| Capability | Key command(s) |
|-----------|---------------|
| **New repo from template** | `copier copy <template> <dest>` |
| **Push template changes to downstream repos** | `copier update --skip-tasks` |
| **Retrofit existing repos** | `copier copy <template> . --overwrite` |
| **Nuclear recopy** | `copier recopy` |
| **Template variable migrations** | `_migrations` in `copier.yml` |
| **Conflict resolution** | inline markers / `--conflict rej` |

### Startup reads

- `/home/***REMOVED***/dev/copier-python-template/README.md` (absolute path, not tilde)
- Future: `/home/***REMOVED***/dev/copier-python-template/docs/planning/reference/copier_update_workflow.md` (after Phase 4)

### Key decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Path style in startup reads | Absolute paths | Tilde paths are not reliably resolved by all tools |
| Example blocks | `<example>` with `<commentary>` tags | Matches existing agents (snakemake-specialist, triton-specialist, slurm-specialist) |
| README table | Added both `claude-specialist` and `copier-specialist` | `claude-specialist` was missing from the table |
| Table ordering | Alphabetical | Consistent with standard table conventions |

---

## Definition of Done

- [x] `~/dev/claude-workspace/agents/copier-specialist.md` created with correct frontmatter, startup reads, and domain knowledge body
- [x] `~/.claude/agents/copier-specialist.md` symlinked and verified
- [ ] Agent confirmed via self-report: reads `~/dev/copier-python-template/README.md` on startup — **requires manual verification via `/agents` in a Claude Code session**
- [x] `claude-workspace/README.md` Available Agents table updated
- [x] Changes committed and pushed to `claude-workspace` remote
- [x] `master.md` "four-phase" corrected to "five-phase"
