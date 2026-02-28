# System-Level Subagents in a Git Repo with Explicit Context Passing

**Written**: 2026-02-28
**Last edited**: 2026-02-28

---

## What Was Done

Promoted three specialist agents (`snakemake-specialist`, `triton-specialist`, `slurm-specialist`) from project-level (`.claude/agents/`) to user-level (`~/.claude/agents/`), tracked in a new private git repo at `~/dev/claude-workspace`.

Agents are **project-agnostic** — they carry no TRITON-SWMM context by default. Project context is passed explicitly in the invocation prompt using `@` file references:

```
Use the snakemake-specialist (passing @.prompts/conventions.md and
@.prompts/architecture.md) to investigate why rule X is failing.
```

This design was chosen over skill-based auto-injection because it scales to N projects without modifying agent files, and makes context boundaries visible at invocation time.

---

## What Was Built

### `~/dev/claude-workspace/` — new private git repo

```
claude-workspace/
├── README.md          — setup instructions, auth notes, context-passing pattern
├── setup.sh           — creates symlinks on a fresh machine
├── agents/
│   ├── snakemake-specialist.md
│   ├── triton-specialist.md
│   └── slurm-specialist.md
└── skills/            — empty; reserved for future use
```

- GitHub remote at `https://github.com/lassiterdc/claude-workspace` (private)
- Remote must be HTTPS (not SSH); `gh auth setup-git` registers the credential helper

### Symlinks

- `~/.claude/agents/snakemake-specialist.md` → `~/dev/claude-workspace/agents/snakemake-specialist.md`
- `~/.claude/agents/triton-specialist.md` → `~/dev/claude-workspace/agents/triton-specialist.md`
- `~/.claude/agents/slurm-specialist.md` → `~/dev/claude-workspace/agents/slurm-specialist.md`

### Agent changes

Each agent retains its domain-specific startup read (`*-workspace/CLAUDE.md`) in the agent body. No `skills:` frontmatter — project context is passed per-invocation.

### Toolkit changes

| File | Change |
|------|--------|
| `.claude/agents/*.md` | Three agent files deleted |
| `.claude/agents/README.md` | New — redirects to `claude-workspace` |
| `.prompts/architecture.md` | Specialist Agents section rewritten |
| `.prompts/conventions.md` | `#### System-level agents and skills` subsection added |
| `scripts/README.md` | Two rows removed (incorrectly listed agents as documentation) |

---

## Key Decision: Why Explicit Passing Over Skills

A `triton-swmm-context` skill was implemented and validated before being removed. The skills mechanism works, but it doesn't scale: with N projects, every agent that needs to work across projects would need to declare all N skills, loading all N project contexts on every invocation. Explicit passing avoids this — the developer decides what context is relevant for each specific task.

---

## Definition of Done

- [x] `~/.claude/agents/triton-execution-debugger.md` deleted (orphaned, no startup reads, never used)
- [x] `~/dev/claude-workspace` repo created, pushed to GitHub with HTTPS remote
- [x] Three agents symlinked to `~/.claude/agents/` via `setup.sh`
- [x] `.claude/agents/` contains only `README.md`
- [x] `architecture.md` and `conventions.md` updated
- [x] `scripts/README.md` stale rows removed
- [x] `/agents` confirms all three specialists at user-level scope
- [x] All three agents confirmed via self-report: each reads only its own `*-workspace/CLAUDE.md`; no TRITON-SWMM project context auto-loaded
- [x] Developer confirmed: explicit invocation pattern, project-agnostic agents, how to add new specialists
