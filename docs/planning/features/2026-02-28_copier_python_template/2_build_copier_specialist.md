# Phase 2: Build the Copier-Specialist Agent

**Part of**: `master.md` — Copier Template System
**Written**: 2026-02-28
**Last edited**: 2026-02-28 — initial extraction from master plan

---

## Goal

Create a `copier-specialist` agent in `~/dev/claude-workspace/agents/` and symlink it to `~/.claude/agents/`. The agent's role is to assist with all Copier-related tasks across the project ecosystem: generating new repos from the template, pushing template improvements to downstream projects, retrofitting existing repos, and handling edge cases like broken updates and variable migrations.

This agent is initialized with the template `README.md` as its primary startup read. After Phase 4 (spin up multidriver + hands-on update tutorial), the agent's startup reads will be updated to include the `copier update` workflow reference doc produced during that phase.

---

## Agent Scope

The copier-specialist handles six core capability areas:

| Capability | Key command(s) | Notes |
|-----------|---------------|-------|
| **New repo from template** | `copier copy <template> <dest>` | Walks through questions, post-gen setup, git init, GitHub push |
| **Push template changes to downstream repos** | `copier update --skip-tasks` | Multi-project propagation; tagging template releases |
| **Retrofit existing repos** | `copier copy <template> . --overwrite` | Conflict resolution guided by pre-migration audit |
| **Nuclear recopy** | `copier recopy` | When to use vs. update; discards local evolution |
| **Template variable migrations** | `_migrations` in `copier.yml` | When template variables are renamed between versions |
| **Conflict resolution** | inline markers / `--conflict rej` | Including safe abort: `git reset && git checkout . && git clean` |

### Key gotchas the agent must know

- **Never manually edit `.copier-answers.yml`** — breaks the smart diff algorithm that powers `copier update`
- **`--skip-tasks` for updates** — post-generation hooks (e.g., `git init`, `pre-commit install`) are not idempotent; always skip them on update
- **`--conflict rej`** — alternative to inline conflict markers; creates `.rej` files instead, useful when inline markers interfere with file syntax
- **Aborting a broken update** — `git reset && git checkout . && git clean` (NOT plain `git checkout`, which does not clean untracked files)
- **`_skip_if_exists`** — template setting for files that should be generated once but never overwritten on update (e.g., project-specific `architecture.md`)
- **Version tracking** — `cat .copier-answers.yml | grep _commit` shows which template version a project is pinned to

---

## Implementation

### Step 2.1 — Write the agent file

Create `~/dev/claude-workspace/agents/copier-specialist.md` with:

**Frontmatter:**
```yaml
---
name: copier-specialist
description: |
  Use this agent for any task involving the copier-python-template ecosystem. Invoke when:

  - Generating a new repo from the copier-python-template
  - Pushing a template improvement to one or more downstream projects via copier update
  - Retrofitting an existing repo to use the template (copier copy into existing directory)
  - Deciding between copier update vs copier recopy
  - Resolving copier update conflicts (inline markers or .rej files)
  - Migrating template variables between versions (copier.yml _migrations)
  - Diagnosing a broken update and aborting cleanly

  Examples:
  - "Set up multidriver-swg from the copier-python-template"
  - "I improved conventions.md in the template — propagate it to all downstream repos"
  - "I want to bring my existing repo under the copier template system"
  - "My copier update left conflict markers in three files — help me resolve them"
model: sonnet
---
```

**Body — startup reads:**
```markdown
## Startup Reads

Before doing anything else, read this file:

- `~/dev/copier-python-template/README.md` — template structure, available variables,
  copier copy and copier update usage, Claude Agent Setup section

After Phase 4 of the Copier Template System plan, also read:
- `~/dev/copier-python-template/docs/planning/reference/copier_update_workflow.md`
  — hands-on reference for making template changes, propagating to downstream projects,
  conflict scenarios, and version tracking

Until that file exists, rely on the startup read above and your training knowledge of Copier.
```

**Body — domain knowledge sections:**

The agent body should document the six capability areas from the scope table above, including:
- Standard command sequences for each workflow (copy, update, recopy, retrofit)
- The gotchas list
- Conflict resolution patterns (inline vs. rej, aborting)
- Version tracking commands
- The downstream project registry pattern (grep across `~/dev/*/\.copier-answers.yml`)

### Step 2.2 — Symlink and verify

```bash
cd ~/dev/claude-workspace
ln -s ~/dev/claude-workspace/agents/copier-specialist.md ~/.claude/agents/copier-specialist.md
```

Verify via `/agents` in a Claude Code session — confirm `copier-specialist` appears in the agent list with the correct description.

### Step 2.3 — Push to claude-workspace remote

```bash
cd ~/dev/claude-workspace
git add agents/copier-specialist.md
git commit -m "feat: add copier-specialist agent"
git push
```

### Step 2.4 — Update claude-workspace README

Add `copier-specialist` to the Available Agents table in `~/dev/claude-workspace/README.md`:

```markdown
| `copier-specialist` | Copier template ecosystem: new repo generation, template update propagation, existing repo retrofit, conflict resolution |
```

---

## Definition of Done

- [ ] `~/dev/claude-workspace/agents/copier-specialist.md` created with correct frontmatter, startup reads, and domain knowledge body
- [ ] `~/.claude/agents/copier-specialist.md` symlinked and verified via `/agents`
- [ ] Agent confirmed via self-report: reads `~/dev/copier-python-template/README.md` on startup
- [ ] `claude-workspace/README.md` Available Agents table updated
- [ ] Changes committed and pushed to `claude-workspace` remote
- [ ] `@.prompts/qaqc_and_commit.md` completed and findings reported to developer
