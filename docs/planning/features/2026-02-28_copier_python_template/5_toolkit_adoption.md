# Phase 5: Retroactively Adopt Copier in TRITON-SWMM Toolkit

**Part of**: `master.md` — Copier Template System
**Written**: 2026-02-28
**Last edited**: 2026-02-28 — renumbered from Phase 4 to Phase 5; use copier-specialist agent to guide adoption

---

## Goal

Bring the existing TRITON-SWMM toolkit under the Copier template system so that future `conventions.md` and `.prompts/` improvements can be propagated to it alongside `multidriver-swg` and any other downstream projects.

**Important**: This phase modifies the existing `TRITON-SWMM_toolkit` repo. It does not change any source code — only the Claude/prompts infrastructure and project scaffolding files.

---

## What "Adopting Copier" Means for an Existing Repo

The toolkit was not generated from the template — it predates it. Copier supports this via `copier copy` run into an existing directory. Copier will:
1. Ask all the `copier.yml` questions (pre-filled with defaults)
2. Generate the template files into the existing directory
3. Show conflicts where template files differ from existing files
4. You resolve conflicts, keeping the TRITON-SWMM-specific content and accepting the template's structure where appropriate
5. After accepting, `.copier-answers.yml` is written — from this point forward `copier update` works normally

---

## Step 4.1 — Pre-migration audit

Before running Copier, identify every file the template will touch and note which version should win each conflict:

| Template file | Toolkit equivalent | Expected resolution |
|--------------|-------------------|---------------------|
| `template/.prompts/conventions.md` | `.prompts/conventions.md` | Keep toolkit's — it has the full TRITON-SWMM Part II. Accept template's Part I/III updates only. |
| `template/.prompts/architecture.md` | `.prompts/architecture.md` | Keep toolkit's entirely — it's fully project-specific. |
| `template/.prompts/implementation_plan.md` | `.prompts/implementation_plan.md` | Accept template's — they should be identical (verbatim copy). |
| `template/.prompts/qaqc_and_commit.md` | `.prompts/qaqc_and_commit.md` | Accept template's — verbatim copy. |
| `template/.prompts/proceed_with_implementation.md` | `.prompts/proceed_with_implementation.md` | Accept template's — verbatim copy. |
| `template/CLAUDE.md` | `CLAUDE.md` | Accept template's — pattern is identical. |
| `template/.claude/settings.local.json` | `.claude/settings.local.json` | Keep toolkit's — has TRITON-SWMM-specific allow list entries. |
| `template/.claude/agents/README.md` | `.claude/agents/README.md` | Accept template's — they'll be identical in spirit. |
| `template/pyproject.toml` | `pyproject.toml` | Keep toolkit's — has TRITON-SWMM deps, ruff config, pytest config. |
| `template/mkdocs.yml` | *(does not exist — toolkit uses Sphinx)* | Accept template's — adds MkDocs config alongside existing Sphinx. Toolkit can keep both. |
| `template/.readthedocs.yaml` | `.readthedocs.yaml` | Conflict — see Open Decision below. Recommended: exclude from Copier via `.copierignore`. |
| `template/.gitignore` | `.gitignore` | Accept template's (standard Python gitignore) — toolkit's custom additions will need to be re-added. |
| `template/.pre-commit-config.yaml` | `.pre-commit-config.yaml` | Keep toolkit's — has the project-specific `check-claude-docs` hook. |
| `template/CONTRIBUTING.md` | `CONTRIBUTING.md` | Keep toolkit's — has TRITON-SWMM-specific content. |

---

## Step 4.2 — Run Copier into the existing repo

```bash
cd ~/dev/TRITON-SWMM_toolkit
copier copy ~/dev/copier-python-template . --overwrite
```

The `--overwrite` flag tells Copier to write files even when they already exist (conflicts will still be shown). Work through each conflict using the resolution guide from Step 4.1.

---

## Step 4.3 — Verify and commit

After resolving all conflicts:

1. Run `ruff check .` — confirm no new linting errors introduced
2. Run `pytest tests/test_PC_01_singlesim.py -v` — confirm single-sim smoke test still passes
3. Confirm `.copier-answers.yml` is present and correct
4. Commit: `chore: adopt Copier template system (copier-python-template v1.0.0)`

---

## Step 4.4 — Propagate a real template change to all downstream projects

Now that both `multidriver-swg` and the toolkit are under Copier management, exercise the full multi-project propagation workflow end-to-end.

**The scenario**: Make a small real improvement to a template file (e.g., add a sentence to Part I of `conventions.md`), tag a new template version, and propagate it to both downstream projects.

**Steps**:

1. In `copier-python-template`, edit `template/.prompts/conventions.md` — make a small meaningful improvement to Part I (Universal Principles)
2. Commit and tag: `git tag v1.0.1 && git push --tags`
3. In `multidriver-swg`, run `copier update --skip-tasks` — verify the change appears with no conflict
4. In `TRITON-SWMM_toolkit`, run `copier update --skip-tasks` — verify the change appears with no conflict
5. Commit the update in both downstream projects: `chore: apply template update v1.0.1`

**Completion signal**: Both projects show the improvement in their `conventions.md`. Confirm with Claude before marking done.

---

## Open Decision for Phase 4

**`.readthedocs.yaml` conflict** — two options:

| Option | Description | Recommendation |
|--------|-------------|----------------|
| Switch toolkit to MkDocs | Replace Sphinx with MkDocs + Material; update docs from RST to Markdown | Consistent with template; better long-term; significant docs migration effort |
| Keep Sphinx; exclude from Copier | Add `.readthedocs.yaml` to `.copierignore` for this project | Avoids docs migration now; toolkit docs stay in RST; Copier won't touch this file |

**Recommended**: Exclude `.readthedocs.yaml` from Copier management for now (add to the project's `.copierignore`). Migrate to MkDocs as a separate task when the toolkit docs need a refresh. This keeps Phase 4 narrowly scoped.

---

## Definition of Done

- [ ] `copier copy` completed into `TRITON-SWMM_toolkit` without errors
- [ ] All file conflicts resolved using the pre-migration audit table
- [ ] `.copier-answers.yml` present in toolkit root
- [ ] `.readthedocs.yaml` conflict resolved (either excluded via `.copierignore` or migrated)
- [ ] `ruff check .` passes
- [ ] PC_01 smoke test passes
- [ ] Changes committed with chore commit message referencing template version
- [ ] `copier update --skip-tasks` runs successfully (no errors, even if no changes to apply yet)
- [ ] Toolkit appears alongside `multidriver-swg` when running the version-check grep from Phase 3
- [ ] Step 4.4 completed: template improvement propagated to both `multidriver-swg` and the toolkit via `copier update`
- [ ] `@.prompts/qaqc_and_commit.md` completed and findings reported to developer
