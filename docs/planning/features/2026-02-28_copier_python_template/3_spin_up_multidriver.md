# Phase 3: Spin Up `multidriver-swg`, Verify the Template, and Write the Update Reference

**Part of**: `master.md` — Copier Template System
**Written**: 2026-02-28
**Last edited**: 2026-02-28 — refreshed all stale `.prompts/` references to match actual template structure; renumbered steps 2.x→3.x; merged Phase 4 (update workflow reference) into Step 3.5; added `.scratch/` creation log objective; added Mermaid rendering note

---

## Goal

Use the newly built template to create the `multidriver-swg` repo from scratch, confirm that ReadTheDocs builds and renders correctly, walk through the full update workflow, and formalize the update process as a reusable reference document.

**`multidriver-swg`**: Stochastic weather generator producing correlated compound forcing (rainfall fields + storm surge + tidal phase) by resampling and rescaling historic events to match randomly generated event statistics.

---

## Step 3.1 — Generate the repo

```bash
cd ~/dev
copier copy ~/dev/copier-python-template multidriver-swg
```

Copier will prompt through each question. Expected answers:
- `project_name`: `multidriver-swg`
- `project_slug`: `multidriver-swg` (accept default)
- `package_name`: `multidriver_swg` (accept default)
- `author_name`: `Daniel Lassiter` (accept default)
- `author_email`: `daniel.lassiter@outlook.com` (accept default)
- `github_username`: `lassiterdc` (accept default)
- `description`: `Stochastic weather generator producing correlated compound forcing (rainfall fields + storm surge + tidal phase) by resampling and rescaling historic events to match randomly generated event statistics.`
- `python_version`: `3.12` (accept default)

After generation, verify that `.copier-answers.yml` exists at the project root and contains the correct values.

**Creation log**: Save the full command transcript (commands run and their outputs) to `multidriver-swg/.scratch/creation_log.md` for developer review.

---

## Step 3.2 — Initialize git and push to GitHub

```bash
cd ~/dev/multidriver-swg
git init
git add .
git commit -m "chore: initial project scaffold from copier-python-template"
gh repo create multidriver-swg --private --source=. --remote=origin --push
```

Confirm the repo appears at `https://github.com/lassiterdc/multidriver-swg`.

---

## Step 3.3 — Wire ReadTheDocs

1. Log in to [readthedocs.org](https://readthedocs.org)
2. Click **Import a Project** → find `multidriver-swg`
3. Trigger a build and confirm it succeeds
4. Verify the following render correctly in the built docs:
   - **Mermaid flowchart** on `index.md` (the `copier update` workflow diagram)
   - **Admonition block** on `installation.md` (the `!!! note` block)
   - **Mermaid sequence diagram** on `usage.md`
   - **mkdocstrings API block** on `api.md` — confirm `hello()` function appears with its docstring and type signature

If any of the four fail, diagnose and fix in the template repo before proceeding to Step 3.4.

> **Note — expected Mermaid rendering**: The template's `mkdocs.yml` was updated during preflight to include `pymdownx.superfences` with Mermaid fence configuration and `admonition` + `pymdownx.details` extensions. If Mermaid diagrams still fail to render, check that the `pymdownx` package is installed (it ships with `mkdocs-material`) and that the fence configuration is correct. The `pyproject.toml` must include `pymdownx-extensions` or rely on Material's bundled version.

---

## Step 3.4 — Update tutorial walkthrough

This is an interactive walkthrough to confirm you understand the update mechanism before relying on it. Claude will guide you through each step.

**The scenario**: You make a small deliberate improvement to the template — adding a new sentence to `CONTRIBUTING.md` under "Fail-fast" — then propagate it to `multidriver-swg` using `copier update`.

**Steps** (Claude will walk you through each one):

1. In the template repo, open `template/CONTRIBUTING.md`
2. Under the "Fail-fast" heading (line 79), add a new sentence: *"Include the offending value in the exception message wherever possible — abstract errors like 'invalid input' are harder to act on than 'expected int, got str for field X'."*
3. Commit and tag the template: `git tag v1.0.1`
4. In `multidriver-swg`, run `copier update --skip-tasks`
5. Observe the diff — the new sentence should appear in your `CONTRIBUTING.md` automatically with no conflict
6. Confirm to Claude that the update applied correctly

**Completion signal**: You describe what happened (what changed, where, how you resolved it) and Claude confirms your understanding is correct before proceeding to Step 3.5.

---

## Step 3.5 — Write the copier update reference guide

After completing the update tutorial, formalize what was learned as a reusable reference document. Write `COPIER_UPDATE_GUIDE.md` at the **template repo root** (non-rendered reference alongside `README.md`).

The guide should cover:

### Part 1: Making a Template Change

1. Edit the relevant file(s) in `template/`
2. Test locally: `copier copy --defaults . /tmp/test-update-project`
3. Verify the change looks correct in the generated output
4. Commit to the template repo
5. Tag the release: `git tag vX.Y.Z && git push --tags`

### Part 2: Propagating to a Downstream Project

1. `cd` into the downstream project
2. Run: `copier update --skip-tasks`
3. Review all changed files (`git diff`)
4. Resolve any conflict markers manually
5. Commit: `git add . && git commit -m "chore: apply template update vX.Y.Z"`

### Part 3: Conflict Scenarios and Resolutions

| Scenario | What you see | What to do |
|----------|-------------|------------|
| Template changed a section; you only touched different sections | File updated automatically, no markers | Just commit |
| Template changed a section; you also changed the same section | Conflict markers in the file | Keep the version you want; remove markers; commit |
| Template added a new file you don't have | New file appears in your project | Review and commit |
| Template changed a file you deleted | File reappears | Delete it again; commit |
| Template renamed a variable in `copier.yml` | `.copier-answers.yml` may have stale key | Run `copier update --defaults` and re-answer changed questions |

### Part 4: Keeping Track of Which Version Each Project Is On

```bash
# Check which template version a project is pinned to:
cat .copier-answers.yml | grep _commit

# Check all downstream projects at once (if organized under ~/dev/):
grep -r "_commit:" ~/dev/*/.copier-answers.yml 2>/dev/null
```

After writing the guide:
1. Commit and push to the template repo
2. Update the `copier-specialist` agent's startup reads to include `COPIER_UPDATE_GUIDE.md` (replace the placeholder reference to `docs/planning/reference/copier_update_workflow.md`)
3. Verify the agent reads both `README.md` and `COPIER_UPDATE_GUIDE.md` on startup

---

## Definition of Done

- [ ] `~/dev/multidriver-swg` exists and was generated from the template
- [ ] `.copier-answers.yml` present with correct values
- [ ] `.scratch/creation_log.md` contains full command transcript
- [ ] `multidriver-swg` repo pushed to GitHub as private
- [ ] ReadTheDocs build succeeds and all four verification points pass (Mermaid, admonition, sequence diagram, mkdocstrings)
- [ ] Update tutorial completed: template improvement propagated to `multidriver-swg` via `copier update`
- [ ] You have described the update outcome and Claude has confirmed your understanding
- [ ] `COPIER_UPDATE_GUIDE.md` written at template repo root and committed
- [ ] `copier-specialist` agent startup reads updated to include `COPIER_UPDATE_GUIDE.md`
- [ ] Agent confirmed via self-report: reads both `README.md` and `COPIER_UPDATE_GUIDE.md` on startup
