# Phase 4: Retroactively Adopt Copier in TRITON-SWMM Toolkit

**Part of**: `master.md` — Copier Template System
**Written**: 2026-02-28
**Last edited**: 2026-02-28 — renumbered from Phase 5 to Phase 4 (old Phase 4 merged into Phase 3); rebuilt pre-migration audit table from actual template file inventory; updated all stale `.prompts/` references

---

## Goal

Bring the existing TRITON-SWMM toolkit under the Copier template system so that future template improvements can be propagated to it alongside `multidriver-swg` and any other downstream projects.

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

Before running Copier, identify every file the template will generate and note which version should win each conflict.

> **Note on structural difference**: The template's instructional document layout differs significantly from the toolkit's current layout. The template uses `CLAUDE.md` (with inline AI norms, planning lifecycle, code style), `CONTRIBUTING.md` (development principles), and root-level `architecture.md`. The toolkit uses `CLAUDE.md` (minimal — just pointers to `.prompts/`), `.prompts/conventions.md` (comprehensive — all norms + principles + code style), and `.prompts/architecture.md`. Reconciling these structures is a key decision for this phase. For now, the audit table notes each conflict without pre-deciding — resolution will be determined during execution when we can evaluate the trade-offs hands-on.

| Template file | Toolkit equivalent | Expected resolution |
|--------------|-------------------|---------------------|
| `CLAUDE.md` | `CLAUDE.md` | **Conflict** — template version has inline AI norms, planning lifecycle, code style, terminology/architecture placeholders. Toolkit version is minimal (points to `.prompts/` docs). See structural note above — resolution deferred to execution. |
| `CONTRIBUTING.md` | `CONTRIBUTING.md` | **Conflict** — template version has clean development principles matching the toolkit's `.prompts/conventions.md` Part I. Toolkit version is severely outdated (references flake8, tox, virtualenvwrapper). The toolkit's CONTRIBUTING.md should be rebuilt during this phase to align with the template's structure while preserving any TRITON-SWMM-specific content. |
| `architecture.md` | `.prompts/architecture.md` | **Conflict** — template places this at repo root; toolkit has it at `.prompts/architecture.md`. Keep toolkit's content entirely (project-specific), but consider relocating to root to match template convention. |
| `.claude/settings.local.json` | `.claude/settings.local.json` | Keep toolkit's — has TRITON-SWMM-specific allow list entries. |
| `.claude/agents/README.md` | `.claude/agents/README.md` | Accept template's — they should be identical in spirit. |
| `pyproject.toml` | `pyproject.toml` | Keep toolkit's — has TRITON-SWMM deps, ruff config, pytest config. |
| `mkdocs.yml` | *(does not exist — toolkit uses Sphinx)* | Accept template's — adds MkDocs config alongside existing Sphinx. Toolkit can keep both until a full docs migration. |
| `.readthedocs.yaml` | `.readthedocs.yaml` | **Conflict** — template version targets MkDocs; toolkit version targets Sphinx. Recommended: exclude from Copier via `.copierignore`. |
| `.gitignore` | `.gitignore` | Accept template's (standard Python gitignore + `.scratch/` + scientific data formats). Toolkit's custom additions (TRITON-SWMM-specific patterns) will need to be re-added. |
| `.pre-commit-config.yaml` | `.pre-commit-config.yaml` | Keep toolkit's — has the project-specific `check-claude-docs` hook. |
| `.copier-answers.yml` | *(does not exist)* | Accept — this is the Copier anchor file. |
| `README.md` | `README.md` | Keep toolkit's — project-specific. |
| `HISTORY.md` | *(does not exist)* | Accept template's — changelog stub. |
| `scripts/check_doc_freshness.py` | `scripts/check_doc_freshness.py` | Keep toolkit's — has project-specific file mappings. |
| `docs/index.md` | *(toolkit uses RST)* | Accept template's — will coexist with Sphinx docs until migration. |
| `docs/installation.md` | *(toolkit uses RST)* | Accept template's. |
| `docs/usage.md` | *(toolkit uses RST)* | Accept template's. |
| `docs/api.md` | *(toolkit uses RST)* | Accept template's. |
| `docs/planning/README.md` | `docs/planning/README.md` | Merge — compare contents and keep the more complete version. |
| `docs/planning/{bugs,features,refactors}/completed/.gitkeep` | Already exists | No conflict expected. |
| `src/[[package_name]]/__init__.py` | `src/TRITON_SWMM_toolkit/__init__.py` | Keep toolkit's — has real content. |
| `tests/__init__.py` | `tests/__init__.py` | Keep toolkit's. |

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
4. Commit: `chore: adopt Copier template system (copier-python-template v1.0.1)`

---

## Step 4.4 — Propagate a real template change to both downstream projects

Now that both `multidriver-swg` and the toolkit are under Copier management, exercise the full multi-project propagation workflow end-to-end.

**The scenario**: Make a small real improvement to a template file (e.g., add a sentence to the "Development Principles" section of `CONTRIBUTING.md`), tag a new template version, and propagate it to both downstream projects.

**Steps**:

1. In `copier-python-template`, edit `template/CONTRIBUTING.md` — make a small meaningful improvement
2. Commit and tag: `git tag v1.0.2 && git push --tags`
3. In `multidriver-swg`, run `copier update --skip-tasks` — verify the change appears with no conflict
4. In `TRITON-SWMM_toolkit`, run `copier update --skip-tasks` — verify the change appears with no conflict
5. Commit the update in both downstream projects: `chore: apply template update v1.0.2`

**Completion signal**: Both projects show the improvement. Confirm with Claude before marking done.

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
