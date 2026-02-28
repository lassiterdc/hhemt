# System-Level Subagents in a Git Repo with Auto-Loaded Project Context

**Written**: 2026-02-28
**Last edited**: 2026-02-28 — removed triton-execution-debugger from scope (deleted as orphaned); added invocation validation to DoD; post-implementation design change: removed skill-based context injection in favor of explicit context passing (see Post-Implementation Note below)

---

## Task Understanding

### Requirements

1. **Promote project-level agents to user-level (system-wide)** — the four specialist agents currently in `.claude/agents/` should be available in all Claude Code sessions, not just within this project.
2. **Keep agents in a git-tracked repo** — agents should be versioned and backed up, similar to the toolkit itself.
3. **Auto-load project conventions into agents** — agents should receive `.prompts/conventions.md` and `.prompts/architecture.md` at startup without per-agent boilerplate.
4. **Codify the convention** — `conventions.md` should document how to pass context to system-level agents.
5. **Context isolation** — adding skills for future repos must have zero effect on existing agents. Each agent loads only what it explicitly declares.

### Assumptions

- "System-level" = stored in `~/.claude/agents/` so they are available across all Claude Code sessions.
- Private git repo: `~/dev/claude-workspace`.
- The three agents to promote: `snakemake-specialist`, `triton-specialist`, `slurm-specialist`.
- All three are project-level only (`.claude/agents/`). `~/.claude/agents/` is now empty (the orphaned `triton-execution-debugger.md` was deleted as a prerequisite — it had no planning record, no startup reads, and had not been used).

### Success Criteria

- All three agents tracked in `~/dev/claude-workspace`, symlinked to `~/.claude/agents/`.
- All three agents available in any Claude Code session.
- Each agent automatically receives TRITON-SWMM context at startup via the `skills:` frontmatter field — no per-agent startup-read boilerplate.
- Adding a skill for a future repo has zero effect on existing TRITON-SWMM agents.
- `.claude/agents/` in this project contains no agent files; replaced with a README.
- `conventions.md` documents the system-level agent and skill pattern.

---

## Evidence from Codebase

- **`.claude/agents/`**: three agents present at project-level: `snakemake-specialist.md`, `triton-specialist.md`, `slurm-specialist.md`. `~/.claude/agents/` is empty (orphaned `triton-execution-debugger.md` deleted as prerequisite).

- **Claude Code agent precedence** (from official docs):
  1. `--agents` CLI flag (session only)
  2. `.claude/agents/` (project) — highest persistent priority
  3. `~/.claude/agents/` (user)
  4. Plugin `agents/` directories

- **No native custom-directory support**: Claude Code has no path override for agent or skill directories. The only user-level locations are `~/.claude/agents/` and `~/.claude/skills/`. Symlinks in those directories are read transparently, enabling a git-tracked source of truth elsewhere.

- **Skills are directories, not flat files**: User-level skills live at `~/.claude/skills/<skill-name>/SKILL.md`. This is a directory per skill, with `SKILL.md` as the required entrypoint. Skills are symlinked at the directory level, not the file level.

- **`skills:` frontmatter on subagents**: Lists skills by name; the full content of each named skill's `SKILL.md` is injected into the subagent's context at startup, before the agent does any work. Skills are NOT inherited from the parent conversation — they must be declared explicitly in each agent's frontmatter.

- **Skills are per-agent, not global**: An agent only loads the skills it declares. Adding a new skill for a different project has no effect on agents that don't list it. This is the key isolation guarantee.

- **Current agent startup reads**: Each agent body manually lists conventions and architecture as the first thing to read (e.g., `snakemake-specialist.md` lines 9–13). This is boilerplate that the `skills:` field replaces cleanly.

- **`snakemake-specialist` reads a third file**: `snakemake-workspace/CLAUDE.md` — this is domain-specific and will be handled by a separate `snakemake-context` skill in the future, not the `triton-swmm-context` skill. For now it stays in the agent body.

- **`conventions.md` AI Working Norms section** (lines 249–280): Documents subagent invocation rules — the right place to add the system-level agent convention.

---

## Implementation Strategy

### Chosen Approach: Symlink + Skills

**Phase A** — Create `~/dev/claude-workspace` as a private git repo with `agents/` and `skills/` directories and a `setup.sh` symlink script.

**Phase B** — Create the `triton-swmm-context` skill as a directory (`skills/triton-swmm-context/SKILL.md`) in `claude-workspace`. Symlink it to `~/.claude/skills/triton-swmm-context`.

**Phase C** — Move all four agent files from `.claude/agents/` into `claude-workspace/agents/`. Update each agent's frontmatter to declare `skills: [triton-swmm-context]` and remove the manual startup-read block from the agent body. Symlink each agent file to `~/.claude/agents/`.

**Phase D** — Replace `.claude/agents/` content with a `README.md` pointing to `claude-workspace`. Update `.prompts/architecture.md` and `.prompts/conventions.md`.

**Phase E** — Update all remaining in-repo references to agent files or `.claude/agents/` paths, then delete the three project-level agent `.md` files.

### Alternatives Considered

- **Pure symlink (no skills)**: Keep per-agent startup-read boilerplate but symlink the files. No single place to update context instructions; adding a new prompt file requires touching all agents.
- **Duplicate files (no symlinks)**: Copy agents into `~/.claude/agents/` and accept drift from the project copy. Not viable — no single source of truth.
- **Plugin distribution**: Overkill for a single developer.

### Trade-offs

| Approach | Version control | Cross-machine | Auto-context | Context isolation | Complexity |
|----------|----------------|---------------|--------------|-------------------|------------|
| Symlink + skills | ✅ | ✅ clone + re-symlink | ✅ via skills field | ✅ explicit per agent | Low-medium |
| Pure symlink | ✅ | ✅ | ❌ manual per agent | ✅ | Low |
| Duplicate | ❌ | ❌ | ❌ | ✅ | Low |

**Winner: Symlink + skills** — cleanest long-term; skills eliminate boilerplate and allow context to be updated in one place as the project evolves.

---

## File-by-File Change Plan

### New repo: `~/dev/claude-workspace/`

```
claude-workspace/
├── README.md
├── setup.sh                          ← automates symlink creation on a fresh machine
├── agents/
│   ├── snakemake-specialist.md       ← flat .md files (agents are flat)
│   ├── triton-specialist.md
│   └── slurm-specialist.md
└── skills/
    └── triton-swmm-context/          ← skills are DIRECTORIES
        └── SKILL.md                  ← required entrypoint
```

**Important**: agents are flat `.md` files; skills are directories containing `SKILL.md`. These are different structures — symlinks reflect this (file symlinks for agents, directory symlinks for skills).

#### Initializing the repo and first push to GitHub

This repo does not exist yet — it must be created locally and then pushed to a new private GitHub remote. Past workspace repos have hit authentication failures on first push due to a protocol mismatch: `gh auth login` defaults to SSH as the git operations protocol, but push authentication is wired through `gh`'s credential helper which only works over **HTTPS**. The fix is always to ensure the remote URL is HTTPS.

**Step 1 — Initialize the repo locally:**
```bash
mkdir -p ~/dev/claude-workspace
cd ~/dev/claude-workspace
git init
mkdir -p agents skills
```

**Step 2 — Create the GitHub repo (private) and add the HTTPS remote:**
```bash
gh repo create claude-workspace --private --source=. --remote=origin
```
`gh repo create --source=.` creates the GitHub repo and sets `origin` to the HTTPS URL automatically. This is the preferred path — it avoids the SSH/HTTPS mismatch entirely.

**If that command is unavailable or fails, do it manually:**
```bash
# Create the repo on GitHub (no local clone)
gh repo create lassiterdc/claude-workspace --private

# Add the HTTPS remote explicitly (NOT the SSH form git@github.com:...)
git remote add origin https://github.com/lassiterdc/claude-workspace.git

# Verify the remote is HTTPS before pushing
git remote -v
```

**Step 3 — Ensure the gh credential helper is registered:**
```bash
gh auth setup-git
```
This registers `gh` as the credential helper for HTTPS remotes. It is idempotent — safe to run even if already configured. The working workspace repos (snakemake-workspace, triton-workspace) both use this setup.

**Step 4 — First commit and push:**
```bash
git add .
git commit -m "feat: initialize claude-workspace with agents, skills, and setup.sh"
git push -u origin main
```

**If push fails with a permission or authentication error:**
```bash
# Confirm the remote is HTTPS (not SSH)
git remote -v
# If SSH form (git@github.com:...), fix it:
git remote set-url origin https://github.com/lassiterdc/claude-workspace.git
# Re-register credential helper and retry
gh auth setup-git
git push -u origin main
```

---

### `~/dev/claude-workspace/skills/triton-swmm-context/SKILL.md` — NEW

Instruction-only: tells the agent what to read and why. Never duplicates file content (stale-copy risk). Includes all categories defined in the expansion spec: mandatory reads, conditional reads, planning doc convention, and write-target rule.

```markdown
---
name: triton-swmm-context
description: TRITON-SWMM toolkit project context — conventions, architecture, and planning patterns
user-invocable: false
disable-model-invocation: true
---

## Required Startup Reads

Before doing anything else, read all of the following files in full:

1. `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.prompts/conventions.md`
   — Project terminology, design philosophy, code standards, testing approach,
     AI working norms, and subagent invocation rules. All recommendations
     you make must align with this document.

2. `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.prompts/architecture.md`
   — Three-layer class hierarchy, key modules, workflow phases, configuration
     system, HPC integration patterns, and known gotchas.

## Conditional Reads

If your task involves HPC debugging, also read:
- `/home/***REMOVED***/dev/TRITON-SWMM_toolkit/.prompts/debugging_hpc_analysis.md`
  — Full log taxonomy, log file locations, and the empirical testing protocol.

## Planning Document Convention

When recording findings, follow the lifecycle in `conventions.md §Planning document lifecycle`:
- Active bugs: `docs/planning/bugs/YYYY-MM-DD_<topic>.md`
- Active features: `docs/planning/features/YYYY-MM-DD_<topic>.md`
- Completed: move to `completed/` subdirectory within the same type directory

## Write-Target Rule

Every subagent invocation should have a designated planning document to write
findings into. Before answering, confirm where your findings will be recorded.
If no write target is specified, ask the main agent for one.
```

`user-invocable: false` and `disable-model-invocation: true` prevent this skill from appearing as a slash command or being auto-triggered — it exists solely to be preloaded by agents.

---

### `~/dev/claude-workspace/agents/*.md` — MOVED + MODIFIED

Move each agent from `.claude/agents/` to `claude-workspace/agents/`. For each:

1. Add `skills: [triton-swmm-context]` to the YAML frontmatter.
2. Remove the manual `## Startup Reads` block from the agent body (now handled by the skill).

Example frontmatter diff for `snakemake-specialist.md`:
```yaml
# Before:
name: snakemake-specialist
model: sonnet

# After:
name: snakemake-specialist
model: sonnet
skills:
  - triton-swmm-context
```

**Special case — `snakemake-specialist`**: currently reads three files at startup, including `snakemake-workspace/CLAUDE.md`. The TRITON-SWMM reads move to the skill. The `snakemake-workspace/CLAUDE.md` read and the conditional FAQ reads stay in the agent body for now, until a `snakemake-context` skill is created in a future iteration.

---

### `~/.claude/agents/` — FILE SYMLINKS

```bash
ln -s ~/dev/claude-workspace/agents/snakemake-specialist.md ~/.claude/agents/snakemake-specialist.md
ln -s ~/dev/claude-workspace/agents/triton-specialist.md ~/.claude/agents/triton-specialist.md
ln -s ~/dev/claude-workspace/agents/slurm-specialist.md ~/.claude/agents/slurm-specialist.md
```

---

### `~/.claude/skills/` — DIRECTORY SYMLINK

```bash
mkdir -p ~/.claude/skills
ln -s ~/dev/claude-workspace/skills/triton-swmm-context ~/.claude/skills/triton-swmm-context
```

Note: this symlinks the **directory** (`triton-swmm-context/`), not the file (`SKILL.md`). Claude Code resolves the `SKILL.md` entrypoint inside the symlinked directory.

---

### `claude-workspace/setup.sh` — NEW

```bash
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$(cd "$(dirname "$0")" && pwd)"

echo "Setting up claude-workspace symlinks from $WORKSPACE..."

# Agents
mkdir -p ~/.claude/agents
for agent in "$WORKSPACE"/agents/*.md; do
    name="$(basename "$agent")"
    target="$HOME/.claude/agents/$name"
    if [ -L "$target" ]; then
        echo "  [skip] ~/.claude/agents/$name already symlinked"
    elif [ -e "$target" ]; then
        echo "  [warn] ~/.claude/agents/$name exists but is not a symlink — remove it manually"
    else
        ln -s "$agent" "$target"
        echo "  [link] ~/.claude/agents/$name -> $agent"
    fi
done

# Skills (directory symlinks)
mkdir -p ~/.claude/skills
for skill_dir in "$WORKSPACE"/skills/*/; do
    name="$(basename "$skill_dir")"
    target="$HOME/.claude/skills/$name"
    if [ -L "$target" ]; then
        echo "  [skip] ~/.claude/skills/$name already symlinked"
    elif [ -e "$target" ]; then
        echo "  [warn] ~/.claude/skills/$name exists but is not a symlink — remove it manually"
    else
        ln -s "$skill_dir" "$target"
        echo "  [link] ~/.claude/skills/$name -> $skill_dir"
    fi
done

echo "Done."
```

---

### `claude-workspace/README.md` — NEW

Full content specified in the Documentation section below.

---

### `.claude/agents/` in TRITON-SWMM_toolkit — CLEAR + README

Delete all four `.md` agent files. Add `.claude/agents/README.md`:

```markdown
# agents/

Agents for this project have been promoted to user-level and are tracked in:

    ~/dev/claude-workspace/agents/

They are symlinked to ~/.claude/agents/ and available in all Claude Code sessions.
See claude-workspace/README.md for setup instructions.
```

---

### `.prompts/architecture.md` — UPDATE

Update the "Specialist Agents" section (lines 153–166) to:
- Remove the reference to `.claude/agents/` as the agent location.
- State that agents are user-level, tracked in `~/dev/claude-workspace/agents/`, symlinked to `~/.claude/agents/`.
- Note that project context is loaded via the `triton-swmm-context` skill.

---

### `.prompts/conventions.md` — UPDATE

Add a new subsection under "Spawning subagents" (Part III, AI Working Norms) titled **"System-level agents and skills"** covering:

1. User-level agents are tracked in `~/dev/claude-workspace/agents/` and symlinked to `~/.claude/agents/`.
2. Project context is injected via the `triton-swmm-context` skill declared in each agent's frontmatter — never via hardcoded startup reads in the agent body.
3. Skills load only for agents that explicitly declare them. Adding a skill for a new repo has no effect on existing agents.
4. When creating a new TRITON-SWMM specialist: add `skills: [triton-swmm-context]` to frontmatter; do not add a manual startup-reads block.
5. On a fresh machine: clone `claude-workspace`, run `setup.sh`, update absolute paths in skill files if the repo location differs.

---

### Phase E: Update In-Repo References to Agent Files

A search of the repository found **five files** outside the agent files themselves that reference agent names or `.claude/agents/`. Each requires a different treatment.

---

#### `scripts/README.md` — REMOVE agent rows from file-mapping table (lines 25–26)

**Current content:**
```
| `workflow.py`                        | `snakemake-specialist.md` |
| `execution.py`, `resource_management.py` | `snakemake-specialist.md` |
```

**Problem**: this table maps source files to "documentation". The agent is a *tool*, not documentation — it reads the source files, it doesn't document them. This is a pre-existing inaccuracy unrelated to the migration, but the migration is the right moment to fix it.

**Change**: remove both rows. The table's remaining rows (`system.py`, `analysis.py`, `scenario.py` → `architecture.md`; `config/` → `architecture.md`) are accurate and stay.

---

#### `.prompts/architecture.md` — REWRITE "Specialist Agents" section (lines 153–166)

**Current content** refers to agents as living in `.claude/agents/` and lists the archive note.

**Replacement:**
```markdown
## Specialist Agents

Active agents are tracked in `~/dev/claude-workspace/agents/` and symlinked to
`~/.claude/agents/` (user-level — available in all Claude Code sessions).

- `snakemake-specialist` — Snakemake internals, SLURM executor plugin, workflow debugging, HPC job resource mapping
- `triton-specialist` — TRITON build system, Kokkos backends, SWMM coupling mechanics, compute config selection
- `slurm-specialist` — SLURM scheduler internals, job lifecycle, srun step creation, site-specific configs for Frontier and UVA

Project context (conventions, architecture, HPC debugging protocol) is loaded into each agent automatically
via the `triton-swmm-context` skill declared in each agent's frontmatter. See `claude-workspace/README.md`
for the full pattern.

**When to invoke a specialist** (subject to the "always confirm before spawning subagents" rule in `conventions.md`):
- The task requires deep knowledge of Snakemake DAG scheduling, SLURM executor internals, or TRITON build/Kokkos/coupling mechanics — areas where the specialist's curated startup reads and domain knowledge outperform inline research.
- The investigation would require reading large source files (`workflow.py` is ~3400 lines; TRITON headers are extensive) that would flood the main context.
- The task is pure research with no code to write, and parallelizing the investigation with a subagent is more efficient.

Eight previous agents are archived in `.claude/agents_archive/`. They are not active. See `docs/planning/refactors/agent_files_audit.md` for context.
```

---

#### `.prompts/debugging_hpc_analysis.md` — UPDATE agent references (lines 372–374)

**Current content** uses backtick names (`` `snakemake-specialist` ``) with no path. The names are still accurate — agent names do not change in this migration. The only update needed is consistency: if the surrounding text implies these are files in `.claude/agents/`, that implication should be removed. A quick check of context (line 368–376 above) shows the text says "specialist agent" with no path reference — no change is needed here. The names remain valid identifiers and the text is accurate as-is.

**Change**: none required.

---

#### `docs/planning/bugs/2026-02-27_diagnose-cpu-bind-cores-step-scheduling-latency.md` — NO CHANGE

References are **historical attribution**: "from `snakemake-specialist`", "from `slurm-specialist` SLURM 24.11.5 source analysis". These record which agent produced a finding during an investigation. Agent names do not change; these references remain accurate regardless of where the agent files live.

**Change**: none.

---

#### `docs/planning/bugs/completed/2026-02-27_add-kill-on-bad-exit-to-srun.md` — NO CHANGE

Same rationale: "Confirmed by: `triton-specialist` analysis" is historical attribution.

**Change**: none.

---

#### `.claude/agents/snakemake-specialist.md`, `triton-specialist.md`, `slurm-specialist.md` — DELETE

After all other Phase E changes are applied and all symlinks are confirmed working, delete these three files.

---

## Risks and Edge Cases

| Risk | Mitigation |
|------|------------|
| **Broken symlinks** if `claude-workspace` is moved or not cloned | `setup.sh` warns on non-symlink conflicts; easy to recreate |
| **Double-read** if manual startup block is not removed when adding skill | Explicitly called out in per-agent migration step |
| **Directory vs file symlink confusion** — agents are file symlinks, skills are directory symlinks | Documented in `setup.sh` and README; `setup.sh` handles both automatically |
| **Machine-specific paths in skill** | Acceptable for single developer; `setup.sh` README section documents update procedure on fresh machines |
| **`snakemake-specialist` partial migration** — `snakemake-workspace/CLAUDE.md` read stays in agent body | Explicitly noted; will be cleaned up when `snakemake-context` skill is created |

---

## Validation Plan

No Python code changes — validation is manual:

1. Run `setup.sh` (or create symlinks manually) and verify no warnings.
2. Run `/agents` in Claude Code — confirm all four agents appear at user-level scope.
3. Invoke a specialist (e.g., ask Claude to use `snakemake-specialist`) — confirm it reads conventions and architecture at startup via skill injection.
4. Confirm `.claude/agents/` contains only `README.md`, no `.md` agent files.
5. Confirm `.prompts/architecture.md` "Specialist Agents" section accurately describes the new user-level location.

---

## Documentation and Tracker Updates

| Document | Update |
|----------|--------|
| `claude-workspace/README.md` | New file — full content specified above |
| `.claude/agents/README.md` | New file — redirects to `claude-workspace` |
| `.prompts/architecture.md` | Rewrite "Specialist Agents" section: user-level location, `triton-swmm-context` skill mention |
| `.prompts/conventions.md` | Add "System-level agents and skills" subsection under AI Working Norms |
| `scripts/README.md` | Remove two rows that incorrectly list `snakemake-specialist.md` as documentation |
| `.prompts/debugging_hpc_analysis.md` | No change — agent name references are accurate as-is |
| Planning docs in `docs/planning/bugs/` | No change — references are historical attribution, not file paths |

### `claude-workspace/README.md` — full content

```markdown
# claude-workspace

Private repository for user-level Claude Code configuration: subagents and skills
available across all projects.

## Structure

    agents/    — User-level subagent definitions (.md files)
    skills/    — User-level skills (directories containing SKILL.md)
    setup.sh   — Automates symlink creation on a fresh machine

## Context isolation: agents only load what they declare

Skills are NOT loaded globally. Each agent only loads the skills explicitly
listed in its frontmatter. An agent covering project A never sees project B's
context, even if both skills exist in ~/.claude/skills/.

- Adding a new project's skill has zero effect on existing agents.
- To cover a new repo, create its skill and update only the agents that work with it.
- An agent spanning two repos declares both skills; all others remain unaffected.

There is no ambient context loading based on your working directory.

## How it works

Claude Code loads agents from ~/.claude/agents/ and skills from ~/.claude/skills/.
Since Claude Code reads symlinks transparently, canonical files live here (in git)
while symlinks in those directories point to them.

- **Agents** are flat .md files — symlinked at the file level.
- **Skills** are directories containing SKILL.md — symlinked at the directory level.

## Setup on a fresh machine

    git clone https://github.com/lassiterdc/claude-workspace.git ~/dev/claude-workspace
    cd ~/dev/claude-workspace
    bash setup.sh

Then update absolute paths in any skill files if your repo locations differ
from the originals (e.g., /home/***REMOVED***/dev/ → your actual path).

**Authentication for pushing changes**: push uses HTTPS with `gh` as the credential
helper. If push fails with a permission error:

    git remote set-url origin https://github.com/lassiterdc/claude-workspace.git
    gh auth setup-git

The remote must be HTTPS (not SSH) even though `gh auth status` may show SSH as
the default git protocol. See snakemake-workspace CLAUDE.md for background.

## How skills are preloaded into agents

Skills listed in a subagent's frontmatter are injected as full content at the
subagent's startup — before the agent does any work. They are NOT slash commands
in this usage; they are preloaded context.

Example — a TRITON-SWMM specialist agent:

    ---
    name: snakemake-specialist
    model: sonnet
    skills:
      - triton-swmm-context
    ---

Every time this agent is invoked, it automatically receives the full
triton-swmm-context/SKILL.md content, which instructs it to read
conventions.md and architecture.md before answering.

Skills compose: an agent covering multiple repos declares all relevant skills:

    skills:
      - triton-swmm-context
      - snakemake-context

## Per-repo skill pattern

Each repository gets its own skill directory. The skill contains:
- Mandatory startup reads (with annotations explaining each file's role)
- Conditional reads (e.g., "if debugging HPC, also read X")
- Planning document conventions (where to record findings, naming patterns)
- Write-target rule (agent must confirm where findings go before answering)

Skills never duplicate file content — they tell agents where to read, not what
the files say. This eliminates stale-copy risk as project docs evolve.

## Adding a new agent

1. Create agents/<agent-name>.md with standard Claude Code frontmatter.
2. Add `skills: [<repo>-context]` for each repo the agent covers.
3. Run setup.sh (or symlink manually):
       ln -s ~/dev/claude-workspace/agents/<agent-name>.md ~/.claude/agents/<agent-name>.md

## Adding a new repo skill

1. Create skills/<repo>-context/SKILL.md (use triton-swmm-context as template).
2. Run setup.sh (or symlink manually):
       ln -s ~/dev/claude-workspace/skills/<repo>-context ~/.claude/skills/<repo>-context
3. Declare the skill in any agent that works with that repo:
       skills:
         - <repo>-context
   Agents that don't list it are unaffected.

## Machine-specific paths

Skill files contain absolute paths to repo locations. On a fresh machine,
update each affected SKILL.md with the correct local paths. Paths are
intentionally explicit — a clear "file not found" error is easier to diagnose
than an obscure path resolution failure.
```

---

## Decisions — All Resolved

1. **Repo name/location**: `~/dev/claude-workspace`. ✅
2. **Path portability**: One skill per repo, hardcoded absolute paths. ✅
3. **Skill content scope**: Instruction-only — agent reads the files; skill never duplicates content. ✅
4. **Empty `.claude/agents/`**: Keep with a short README redirecting to `claude-workspace`. ✅

---

## Definition of Done

**Prerequisite (complete)**
- [x] `~/.claude/agents/triton-execution-debugger.md` deleted — orphaned, untracked, no startup reads

**Phase A–B: claude-workspace repo and skill**
- [x] `~/dev/claude-workspace` git repo created and initialized with `agents/`, `skills/`, `setup.sh`, `README.md`
- [x] GitHub remote created as private repo with HTTPS URL (not SSH)
- [x] `gh auth setup-git` confirmed; first push to `origin main` succeeds
- [x] `skills/triton-swmm-context/SKILL.md` created then deleted (design change — skill approach abandoned)
- [x] `~/.claude/skills/triton-swmm-context` symlink created then removed (no active skills)

**Phase C: agent migration and symlinks**
- [x] All three agent files moved from `.claude/agents/` to `claude-workspace/agents/`
- [x] `skills: [triton-swmm-context]` added then removed (design change — agents are project-agnostic)
- [x] Each agent retains only its domain-specific startup read (`*-workspace/CLAUDE.md`)
- [x] `setup.sh` run successfully; all three agent symlinks created without warnings
- [ ] `/agents` in Claude Code shows all three agents at user-level scope — **confirm in fresh session**

**Phase D: .claude/agents cleanup and prompt docs**
- [x] `.claude/agents/` contains only `README.md` — no agent `.md` files
- [x] `.prompts/architecture.md` "Specialist Agents" section rewritten
- [x] `.prompts/conventions.md` "System-level agents and skills" subsection added

**Phase E: in-repo reference cleanup and final deletion**
- [x] `scripts/README.md` two `snakemake-specialist.md` rows removed from file-mapping table
- [x] `.prompts/debugging_hpc_analysis.md` confirmed accurate — no changes needed
- [x] Planning bug docs confirmed accurate — no changes needed
- [x] Three project-level agent files deleted: `.claude/agents/snakemake-specialist.md`, `triton-specialist.md`, `slurm-specialist.md`

**Invocation validation and developer tutorial (within TRITON-SWMM_toolkit session)**

> **Post-implementation design change**: During the invocation tutorial, the skills-based context injection approach was replaced with explicit context passing. Agents are now project-agnostic; project context is passed in the invocation prompt using `@` references. The `triton-swmm-context` skill and all `skills:` frontmatter fields have been removed. The DoD items below have been updated to reflect this.

*Before starting*: open Claude Code in the TRITON-SWMM_toolkit directory. Run `/agents` — confirm all three specialists appear at user-level scope with no project-level copies.

**Step 1 — Confirm agents are present**
- [x] `/agents` shows all three specialists at user-level scope

**Step 2 — Invoke `snakemake-specialist`**

Confirmed: agent reads only `snakemake-workspace/CLAUDE.md` at startup (agent body). No project context is auto-loaded. Context is passed explicitly when needed.
- [x] Agent self-report confirmed correct startup context

**Step 3 — Invoke `triton-specialist`**

Confirmed: agent reads only `triton-workspace/CLAUDE.md` at startup. No TRITON-SWMM project context auto-loaded.
- [x] Agent self-report confirmed correct startup context

**Step 4 — Invoke `slurm-specialist`**

Confirmed: agent reads only `slurm-workspace/CLAUDE.md` at startup. No project context auto-loaded.
- [x] Agent self-report confirmed correct startup context

**Step 5 — Confirm context isolation**

Design change resolved the isolation concern entirely: agents carry no project context by default. Adding skills for new repos has zero effect on existing agents because no agent declares any skill.
- [x] Developer understands isolation guarantee

**Step 6 — Developer sign-off**

- [x] Developer confirms they understand: explicit invocation — *"Use the snakemake-specialist (passing @.prompts/conventions.md and @.prompts/architecture.md) to investigate X"*
- [x] Developer confirmed: agents are project-agnostic; context is passed explicitly, not injected at startup
- [x] Developer confirmed: to add a new specialist, create agent file in `claude-workspace/agents/`, run `setup.sh`
- [x] Developer confirmed: to pass project context, use `@` file references in the invocation prompt

---

## Self-Check Results

1. **Header/body alignment**: All section headers match content. ✅
2. **Section necessity**: All sections are load-bearing; Decisions section retained as a resolved record; Phase E justified by explicit developer requirement. ✅
3. **Conventions alignment**: Approach respects backward-compatibility policy (delete, don't alias), fail-fast principle (symlink errors are immediate and obvious), system-agnostic principle (paths documented as machine-specific with a clear update procedure). Recording the `scripts/README.md` fix aligns with the out-of-scope observations convention — the inaccuracy was noticed during this migration work. ✅
4. **Task-relevance**: No speculative content; `snakemake-context` future skill is noted only where relevant to the current migration. ✅
