# Phase 2: Spin Up `multidriver-swg` and Verify the Template

**Part of**: `master.md` — Copier Template System
**Written**: 2026-02-28
**Last edited**: 2026-02-28 — extracted from master plan

---

## Goal

Use the newly built template to create the `multidriver-swg` repo from scratch, confirm that ReadTheDocs builds and renders correctly, and walk through the full update workflow so you understand how to propagate future template changes.

**`multidriver-swg`**: Stochastic weather generator producing correlated compound forcing (rainfall fields + storm surge + tidal phase) by resampling and rescaling historic events to match randomly generated event statistics.

---

## Step 2.1 — Generate the repo

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

---

## Step 2.2 — Initialize git and push to GitHub

```bash
cd ~/dev/multidriver-swg
git init
git add .
git commit -m "chore: initial project scaffold from copier-python-template"
gh repo create multidriver-swg --private --source=. --remote=origin --push
```

Confirm the repo appears at `https://github.com/lassiterdc/multidriver-swg`.

---

## Step 2.3 — Wire ReadTheDocs

1. Log in to [readthedocs.org](https://readthedocs.org)
2. Click **Import a Project** → find `multidriver-swg`
3. Trigger a build and confirm it succeeds
4. Verify the following render correctly in the built docs:
   - **Mermaid flowchart** on `index.md` (the `copier update` workflow diagram)
   - **Admonition block** on `installation.md` (the `!!! note` block)
   - **Mermaid sequence diagram** on `usage.md`
   - **mkdocstrings API block** on `api.md` — confirm `hello()` function appears with its docstring and type signature

If any of the four fail, diagnose and fix in the template repo before proceeding to Phase 3.

---

## Step 2.4 — Update tutorial walkthrough

This is an interactive walkthrough to confirm you understand the update mechanism before relying on it. Claude will guide you through each step.

**The scenario**: You make a small deliberate improvement to the template — adding a new universal principle to Part I of `conventions.md` — then propagate it to `multidriver-swg` using `copier update`.

**Steps** (Claude will walk you through each one):

1. In the template repo, open `template/.prompts/conventions.md`
2. Add one new sentence to Part I under "Fail-fast": *"Include the offending value in the exception message wherever possible — abstract errors like 'invalid input' are harder to act on than 'expected int, got str for field X'.\"*
3. Commit and tag the template: `git tag v1.0.1`
4. In `multidriver-swg`, run `copier update --skip-tasks`
5. Observe the diff — the new sentence should appear in your `conventions.md` automatically with no conflict
6. Confirm to Claude that the update applied correctly

**Completion signal**: You describe what happened (what changed, where, how you resolved it) and Claude confirms your understanding is correct before marking this phase done.

---

## Definition of Done

- [ ] `~/dev/multidriver-swg` exists and was generated from the template
- [ ] `.copier-answers.yml` present with correct values
- [ ] `multidriver-swg` repo pushed to GitHub as private
- [ ] ReadTheDocs build succeeds and all four verification points pass (Mermaid, admonition, sequence diagram, mkdocstrings)
- [ ] Update tutorial completed: template improvement propagated to `multidriver-swg` via `copier update`
- [ ] You have described the update outcome and Claude has confirmed your understanding
- [ ] `@.prompts/qaqc_and_commit.md` completed and findings reported to developer
