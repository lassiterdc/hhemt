# Copier Template System: Master Plan

**Written**: 2026-02-28
**Last edited**: 2026-02-28 — merged Phase 4 (update workflow reference) into Phase 3; renumbered to 4 phases; refreshed decisions log; added Future Considerations section; updated all stale `.prompts/` references to match actual template structure

---

## Overview

This is a four-phase plan for building and adopting a Copier-based Python project template system named `copier-python-template`.

| Phase | Goal | Scope | Status | Doc |
|-------|------|-------|--------|-----|
| **Phase 1** | Build the `copier-python-template` repo | New repo; no changes to existing projects | ✅ Complete | `implemented/1_build_template.md` |
| **Phase 2** | Build the `copier-specialist` agent | New agent in `claude-workspace`; no changes to existing projects | ✅ Complete | `implemented/2_build_copier_specialist.md` |
| **Phase 3** | Spin up `multidriver-swg` via the copier-specialist; verify ReadTheDocs; walk through the update tutorial; write the copier update reference guide | New repo; template repo may receive minor fixes | Pending | `3_spin_up_multidriver.md` |
| **Phase 4** | Retroactively adopt Copier in the TRITON-SWMM toolkit | Modifies existing toolkit repo | Pending | `4_toolkit_adoption.md` |

Phases must be completed in order — each depends on the previous. Use `/proceed-with-implementation` with the relevant phase doc at the start of each phase. From Phase 3 onward, use the `copier-specialist` agent to perform Copier operations.

---

## Dependencies (Resolved)

### Subagent Refactor (`2026-02-28_system-level-subagents-in-git-repo.md`) — COMPLETE

The refactor promoting project-level agents to user-level (`~/.claude/agents/`) tracked in `~/dev/claude-workspace/` is complete. All three touch points originally gated on this refactor are now resolved:

| Touch point | Resolution |
|-------------|------------|
| **Phase 1 — `template/.claude/agents/` contents** | Generate `README.md` redirect only (no agent stub file). Agent example pattern documented in `claude-workspace/README.md` instead. |
| **Phase 1 — agent frontmatter pattern** | No `skills:` frontmatter. Agents are project-agnostic; context passed explicitly per-invocation via `@` references. |
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
| Template instructional doc layout | `CLAUDE.md` (inline norms + planning lifecycle + code style) + `CONTRIBUTING.md` (development principles) + root `architecture.md` | Differs from toolkit's `.prompts/` convention — reconciliation deferred to Phase 4 |
| QA gate per phase | Baked into `/proceed-with-implementation` step 5 | Cleaner than duplicating in each phase DoD |
| Phase 4 merge into Phase 3 | Update workflow reference guide written as Phase 3 Step 3.5, not a separate phase | Eliminates awkward "done during Phase 3" coupling from original Phase 4 |
| Copier update reference location | `COPIER_UPDATE_GUIDE.md` at template repo root | Non-rendered reference alongside README.md; not in `docs/` (not project documentation) |
| `.scratch/` convention | Template-level (in `.gitignore` of every generated project) | Lightweight; useful for subagent transcripts and temporary working files in any project |

---

## Template File Inventory (Ground Truth)

The actual template structure as of Phase 3 preflight (supersedes references in implemented Phase 1 doc):

```
copier-python-template/
├── copier.yml
├── README.md
├── .copier-tasks.py
└── template/
    ├── CLAUDE.md                          ← AI norms, planning lifecycle, code style (inline)
    ├── CONTRIBUTING.md                    ← Development principles (Part I equivalent)
    ├── architecture.md                    ← Generic stub (at root, not .prompts/)
    ├── README.md
    ├── HISTORY.md
    ├── .copier-answers.yml
    ├── .gitignore                         ← Standard Python + .scratch/ + scientific data
    ├── .pre-commit-config.yaml
    ├── .readthedocs.yaml
    ├── .claude/
    │   ├── settings.local.json
    │   └── agents/README.md
    ├── pyproject.toml
    ├── mkdocs.yml                         ← With pymdownx.superfences + Mermaid config
    ├── scripts/check_doc_freshness.py
    ├── docs/
    │   ├── index.md                       ← Mermaid flowchart
    │   ├── installation.md                ← Admonition block
    │   ├── usage.md                       ← Mermaid sequence diagram
    │   ├── api.md                         ← mkdocstrings autodoc
    │   └── planning/
    │       ├── README.md
    │       ├── bugs/completed/.gitkeep
    │       ├── features/completed/.gitkeep
    │       └── refactors/completed/.gitkeep
    ├── src/[[package_name]]/__init__.py
    └── tests/__init__.py
```

---

## Future Considerations

### Create-repo-from-template Claude skill

After Phase 4 is complete, consider building a Claude skill (`/create-repo`) that:
1. Walks the user through the repo creation process interactively
2. Runs `copier copy` with appropriate `--data` flags
3. Handles `git init`, `git add`, `git commit`, and `gh repo create`
4. Writes a full command transcript to `.scratch/creation_log.md` for developer review
5. Provides manual startup instructions (ReadTheDocs wiring, what to populate in `CLAUDE.md`/`architecture.md`)

This would formalize the Phase 3 workflow as a repeatable, hands-off process. The `.scratch/` convention (added to template `.gitignore` during Phase 3 preflight) supports this pattern.

### Toolkit CONTRIBUTING.md modernization

The toolkit's `CONTRIBUTING.md` is severely outdated (references flake8, tox, virtualenvwrapper, `setup.py develop`, Make commands). Phase 4 (toolkit adoption) is the natural place to address this — the pre-migration audit table already notes this conflict. The toolkit's `CONTRIBUTING.md` should be rebuilt to align with the template's structure while preserving TRITON-SWMM-specific content from `.prompts/conventions.md`.
