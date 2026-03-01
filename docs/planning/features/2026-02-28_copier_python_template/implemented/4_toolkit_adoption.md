# Phase 4: Retroactively Adopt Copier in TRITON-SWMM Toolkit

**Part of**: `master.md` — Copier Template System
**Written**: 2026-02-28
**Last edited**: 2026-03-01 — incorporated all preflight decisions; added comprehensive content preservation audit with line-level verification; added reference fixup items for `debugging_hpc_analysis.md`, `check_doc_freshness.py`, and `architecture.md`

---

## Goal

Bring the existing TRITON-SWMM toolkit under the Copier template system so that future template improvements can be propagated to it alongside `multidriver-swg` and any other downstream projects.

**Important**: This phase modifies the existing `TRITON-SWMM_toolkit` repo. It does not change any source code — only the Claude/prompts infrastructure, project scaffolding, and documentation files.

---

## Step 4.1 — Pre-migration audit

Before running Copier, identify every file the template will generate and note which version should win each conflict.

### Structural migration: `.prompts/` → template layout

The toolkit's `.prompts/` directory will be eliminated (except `debugging_hpc_analysis.md`). Content migrates as follows:

| Source | Destination | Action |
|--------|-------------|--------|
| `.prompts/conventions.md` Part I (universal principles) | `CONTRIBUTING.md` "Development Principles" | **Drop** — template's CONTRIBUTING.md already has identical content |
| `.prompts/conventions.md` Part II (project-specific rules) | `CLAUDE.md` (Terminology, Architecture Patterns, Testing sections) | **Add** — populate template's placeholder sections with toolkit-specific content |
| `.prompts/conventions.md` Part III (AI working norms) | `CLAUDE.md` → pointer to `~/dev/claude-workspace/specialist_agent_docs/ai-working-norms.md` | **Drop** — shared doc is equivalent but fresher |
| `.prompts/conventions.md` "Recording out-of-scope observations" | `~/dev/claude-workspace/specialist_agent_docs/planning-document-lifecycle.md` | **Update shared doc** — this convention is universally valuable and missing from the shared lifecycle doc |
| `.prompts/architecture.md` | Root `architecture.md` | **Move** — keep all content, relocate to match template convention |
| `.prompts/implementation_plan.md` | `/implementation-plan` skill | **Delete** — superseded by skill |
| `.prompts/proceed_with_implementation.md` | `/proceed-with-implementation` skill | **Delete** — superseded by skill |
| `.prompts/qaqc_and_commit.md` | `/qaqc-and-commit` skill | **Delete** — superseded by skill |
| `.prompts/debugging_hpc_analysis.md` | `.prompts/debugging_hpc_analysis.md` (stays) | **Keep** — hyper project-specific, no template equivalent |

### File conflict audit table

| Template file | Toolkit equivalent | Resolution |
|--------------|-------------------|------------|
| `CLAUDE.md` | `CLAUDE.md` | **Rebuild** — use template structure, populate with Part II content (terminology, exception hierarchy, logging patterns, architecture patterns, GIS data, testing conventions) |
| `CONTRIBUTING.md` | `CONTRIBUTING.md` | **Accept template's** — no modifications needed |
| `architecture.md` | `.prompts/architecture.md` | **Move** toolkit's content to root `architecture.md` |
| `.claude/settings.local.json` | `.claude/settings.local.json` | **Keep toolkit's** — has TRITON-SWMM-specific allow list |
| `.claude/agents/README.md` | `.claude/agents/README.md` | **Accept template's** |
| `pyproject.toml` | `pyproject.toml` | **Keep toolkit's** — has TRITON-SWMM deps, ruff config, pytest config |
| `mkdocs.yml` | *(does not exist)* | **Accept template's** — toolkit migrating from Sphinx to MkDocs |
| `.readthedocs.yaml` | `.readthedocs.yaml` | **Accept template's** — migrating to MkDocs (template version targets MkDocs) |
| `.gitignore` | `.gitignore` | **Accept template's** — re-add any TRITON-SWMM-specific patterns |
| `.pre-commit-config.yaml` | `.pre-commit-config.yaml` | **Keep toolkit's** — remove stale `check-claude-docs` hook (script deleted) |
| `.copier-answers.yml` | *(does not exist)* | **Accept** — Copier anchor file |
| `README.md` | `README.md` | **Keep toolkit's** — project-specific |
| `HISTORY.md` | `HISTORY.md` | **Keep toolkit's** — already exists |
| `requirements.txt` | `requirements.txt` | **Keep toolkit's** — different content |
| `.github/workflows/publish.yml` | *(does not exist)* | **Accept template's** — adopting automated tag-based PyPI publishing |
| `scripts/check_doc_freshness.py` | `scripts/check_doc_freshness.py` | **Delete** — never used; remove from template too (Step 4.6) |
| `docs/index.md` | *(toolkit uses RST)* | **Accept template's** — MkDocs migration |
| `docs/installation.md` | *(toolkit uses RST)* | **Accept template's** — MkDocs migration |
| `docs/usage.md` | *(toolkit uses RST)* | **Accept template's** — MkDocs migration |
| `docs/api.md` | *(toolkit uses RST)* | **Accept template's** — MkDocs migration |
| `docs/planning/README.md` | `docs/planning/README.md` | **Keep toolkit's** — more complete |
| `docs/planning/{bugs,features,refactors}/completed/.gitkeep` | Already exists | No conflict |
| `src/[[package_name]]/__init__.py` | `src/TRITON_SWMM_toolkit/__init__.py` | **Keep toolkit's** — has real content |
| `tests/__init__.py` | `tests/__init__.py` | **Keep toolkit's** |

---

## Step 4.2 — Prepare content before running Copier

Before running `copier copy`, prepare the files that need manual content work so that the Copier step is a clean overwrite-and-resolve:

### 4.2a — Build new `CLAUDE.md`

Use the template's structure as the skeleton. Populate with project-specific content from `conventions.md` Part II. The content preservation audit below ensures nothing is lost.

**Startup reads**: `CONTRIBUTING.md`, `architecture.md`

**Section-by-section content map**:

- `## Terminology` — verbatim from conventions.md lines 87–99: full terminology table (model_type, run_mode, multi_sim_run_method, event_iloc, in_slurm) with the confusion Rule
- `## Code Style` — template's 3 bullets (Python version, ruff, Pyright) PLUS 2 toolkit extras from conventions.md lines 223–224: `cfgBaseModel` inheritance and `Literal` types convention
- `## Architecture Patterns` — from conventions.md Part II:
  - Custom exception hierarchy (lines 101–120) — full hierarchy with code block
  - `_already_written()` log-based check (lines 122–124)
  - Logging patterns (lines 126–131) — `[NAMESPACE]` print, `getLogger`, runner convention
  - Pydantic/cfgBaseModel config flow (lines 135–136)
  - Runner script subprocess rules (lines 138–155) — **must include the fork bomb warning code block verbatim**
  - Snakemake wildcard convention (lines 157–158)
  - `platform_configs.py` rule (lines 160–161)
  - Utility candidate tracking location (lines 163–164)
  - GIS data preference (lines 227–231) — prefer GeoJSON over Shapefile
  - HPC empirical testing protocol (lines 260–267) — moved here from Part III so `debugging_hpc_analysis.md` can reference CLAUDE.md instead of deleted conventions.md
- `## Testing` (new section) — from conventions.md lines 233–250:
  - Platform-organized tests, `GetTS_TestCases` factories, standardized assertions
  - Full smoke test table (PC_01, PC_02, PC_04, PC_05) with commands and "run when" guidance
  - SLURM testing note, HPC debugging reference to `.prompts/debugging_hpc_analysis.md`
  - Jupyter notebook note: `tests/dev/` notebooks are scratchpads, never commit (from Part III line 282, project-specific detail not in shared AI norms doc)
- `## AI Working Norms` — pointer to `~/dev/claude-workspace/specialist_agent_docs/ai-working-norms.md`
- `## Planning Document Lifecycle` — pointer to `~/dev/claude-workspace/specialist_agent_docs/planning-document-lifecycle.md`

### 4.2b — Build new `CONTRIBUTING.md`

Accept template's `CONTRIBUTING.md` wholesale. No modifications needed.

### 4.2c — Move `architecture.md`

Move `.prompts/architecture.md` → root `architecture.md` with full content preserved. Update internal reference on line 2 from *"load this alongside `.prompts/conventions.md`"* to *"load this alongside `CONTRIBUTING.md`"*.

### 4.2d — Update shared planning lifecycle doc

The "Recording out-of-scope observations" section (conventions.md lines 209–217) is already present in the shared `planning-document-lifecycle.md` (lines 44–52). **No update needed** — verified during audit.

### 4.2e — MkDocs migration

- Accept template's `mkdocs.yml` — update project name and nav to reflect toolkit
- Accept template's `.readthedocs.yaml`
- Accept template's `docs/*.md` files as starting points
- Remove or archive old Sphinx RST files (`docs/source/`, `docs/Makefile`, `docs/make.bat`, `conf.py` etc.)

---

## Step 4.3 — Run Copier into the existing repo

```bash
cd ~/dev/TRITON-SWMM_toolkit
copier copy ~/dev/copier-python-template . --overwrite
```

Answer the Copier questions:
- `project_name`: `TRITON-SWMM Toolkit`
- `project_slug`: `TRITON-SWMM_toolkit`
- `package_name`: `TRITON_SWMM_toolkit`
- `description`: `Orchestrates coupled TRITON (2D hydrodynamic) and SWMM (stormwater management) simulations across local machines and HPC clusters`
- Other fields: accept defaults

Work through each conflict using the resolution guide from Step 4.1. Since content was prepared in Step 4.2, most conflicts should resolve by keeping the prepared version.

---

## Step 4.4 — Clean up `.prompts/` and fix stale references

After Copier runs:

### Delete superseded files
1. Delete `.prompts/conventions.md` — content migrated to CLAUDE.md, CONTRIBUTING.md, and shared docs
2. Delete `.prompts/architecture.md` — moved to root
3. Delete `.prompts/implementation_plan.md` — superseded by `/implementation-plan` skill
4. Delete `.prompts/proceed_with_implementation.md` — superseded by `/proceed-with-implementation` skill
5. Delete `.prompts/qaqc_and_commit.md` — superseded by `/qaqc-and-commit` skill
6. Keep `.prompts/debugging_hpc_analysis.md` — project-specific, no template equivalent

### Fix stale references in surviving files
7. `.prompts/debugging_hpc_analysis.md` line 364 — references `conventions.md` for empirical testing protocol → update to reference `CLAUDE.md`
8. `.prompts/debugging_hpc_analysis.md` line 365 — references `.prompts/conventions.md § Spawning subagents` → update to reference shared `ai-working-norms.md`
9. `.prompts/debugging_hpc_analysis.md` line 370 — same as above
10. Delete `scripts/check_doc_freshness.py` and `scripts/README.md` — never used; remove entire `scripts/` directory
11. Remove the `check-claude-docs` hook from `.pre-commit-config.yaml` that referenced the deleted script
12. Delete `CODE_OF_CONDUCT.md` — cookie-cutter cruft, no longer referenced

---

## Step 4.5 — Verify and commit

After resolving all conflicts:

1. Run `ruff check .` — confirm no new linting errors
2. Run `pytest tests/test_PC_01_singlesim.py -v` — confirm smoke test passes
3. Confirm `.copier-answers.yml` is present and correct
4. Confirm `copier update --skip-tasks` runs successfully
5. Commit: `chore: adopt Copier template system (copier-python-template v1.1.1)`

---

## Step 4.6 — Propagate a real template change to both downstream projects

Now that both `multidriver-swg` and the toolkit are under Copier management, exercise the full multi-project propagation workflow end-to-end.

**The scenario**: Remove the dead `scripts/check_doc_freshness.py` stub and `scripts/README.md` from the template — this script was never populated or activated in any downstream project, and its purpose (doc freshness checking) is handled by the `/qaqc-and-commit` skill's `/reflect` step.

**Steps**:

1. In `copier-python-template`, delete `template/scripts/check_doc_freshness.py` and `template/scripts/README.md`
2. Commit and tag (next version after v1.1.1)
3. In `multidriver-swg`, run `copier update --trust --skip-tasks` — verify the change appears
4. In `TRITON-SWMM_toolkit`, run `copier update --trust --skip-tasks` — verify the change appears
5. Commit the update in both downstream projects

---

## Content Preservation Audit

Line-level verification that every substantive content block from eliminated files has a destination. See preflight conversation for full details.

### `.prompts/conventions.md` — verified complete

| Part | Lines | Disposition | Verified |
|------|-------|-------------|----------|
| Part I (universal principles) | 7–80 | Template `CONTRIBUTING.md` — **verbatim identical** | Yes |
| Part II: Terminology | 87–99 | New `CLAUDE.md` `## Terminology` | Yes |
| Part II: Exception hierarchy | 101–120 | New `CLAUDE.md` `## Architecture Patterns` | Yes |
| Part II: `_already_written()` | 122–124 | New `CLAUDE.md` `## Architecture Patterns` | Yes |
| Part II: Logging patterns | 126–131 | New `CLAUDE.md` `## Architecture Patterns` | Yes |
| Part II: Architecture patterns (Pydantic, runners, fork bomb, Snakemake, platform_configs, utility tracking) | 133–164 | New `CLAUDE.md` `## Architecture Patterns` | Yes |
| Part II: Planning lifecycle | 166–207 | Shared `planning-document-lifecycle.md` — **already present** | Yes |
| Part II: Out-of-scope observations | 209–217 | Shared `planning-document-lifecycle.md` — **already present** | Yes |
| Part II: Code style | 219–225 | New `CLAUDE.md` `## Code Style` (template 3 bullets + 2 toolkit extras: `cfgBaseModel`, `Literal`) | Yes |
| Part II: GIS data | 227–231 | New `CLAUDE.md` `## Architecture Patterns` | Yes |
| Part II: Testing conventions + smoke table | 233–250 | New `CLAUDE.md` `## Testing` | Yes |
| Part II: HPC debugging protocol | 252–267 | New `CLAUDE.md` `## Architecture Patterns` (empirical testing steps) + `.prompts/debugging_hpc_analysis.md` (log taxonomy, stays) | Yes |
| Part III (AI working norms) | 270–328 | Shared `ai-working-norms.md` — **verified equivalent, shared is fresher** | Yes |
| Part III: `tests/dev/` notebook note | 282 | New `CLAUDE.md` `## Testing` — project-specific detail not in shared doc | Yes |

### `.prompts/architecture.md` — verified complete

Entire file (153 lines) moves to root `architecture.md`. Zero content loss. Internal reference on line 2 updated.

### `.prompts/implementation_plan.md` — verified superseded

Entire file (145 lines) superseded by `/implementation-plan` skill. Skill built from this file.

### `.prompts/proceed_with_implementation.md` — verified superseded

Entire file (47 lines) superseded by `/proceed-with-implementation` skill. Skill is strictly a superset (adds `/reflect` step).

### `.prompts/qaqc_and_commit.md` — verified superseded

Entire file (80 lines) superseded by `/qaqc-and-commit` skill. Skill is strictly a superset (adds `/reflect` step, updated co-author tag).

### `CONTRIBUTING.md` — verified no unique valuable content

All unique content is either stale cruft (virtualenvwrapper, flake8, tox, bump2version) or redundant with shared `ai-working-norms.md` (the "Maintaining AI Context Documentation" section).

---

## Definition of Done

- [ ] New `CLAUDE.md` built with template structure + Part II content (per content preservation audit above)
- [ ] `CONTRIBUTING.md` replaced with template's version
- [ ] `architecture.md` at repo root with full toolkit content; internal reference updated
- [ ] `.prompts/` cleaned up (only `debugging_hpc_analysis.md` remains)
- [ ] Stale references fixed in `debugging_hpc_analysis.md`, `check_doc_freshness.py`
- [ ] MkDocs migration complete (mkdocs.yml, .readthedocs.yaml, docs/*.md; old Sphinx files removed)
- [ ] `copier copy` completed into `TRITON-SWMM_toolkit` without errors
- [ ] `.copier-answers.yml` present in toolkit root
- [ ] `.github/workflows/publish.yml` present (tag-based PyPI publishing)
- [ ] `ruff check .` passes
- [ ] PC_01 smoke test passes
- [ ] `copier update --skip-tasks` runs successfully
- [ ] Toolkit appears alongside `multidriver-swg` when running version-check grep
- [ ] Step 4.6 completed: template improvement propagated to both downstream projects
