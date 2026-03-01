# Phase 3: Spin Up `multidriver-swg`, Verify the Template, and Write the Update Reference

**Part of**: `master.md` — Copier Template System
**Written**: 2026-02-28
**Last edited**: 2026-03-01 — Phase 3 complete

---

## What Was Built

Used the copier-python-template to create `multidriver-swg`, wired it to ReadTheDocs and PyPI, walked through the `copier update` workflow with real template changes, and wrote a comprehensive setup/update reference guide.

**`multidriver-swg`**: Stochastic weather generator producing multi-driver flood forcing (rainfall + storm surge + tidal phase) by resampling and rescaling historic events to match randomly generated event statistics.

### Repos touched

| Repo | Changes |
|------|---------|
| `copier-python-template` | Bugfixes (markdown_extensions, requirements.txt, description period convention), python_version default → 3.11, publish workflow, `COPIER_UPDATE_GUIDE.md`; tags v1.0.1–v1.1.1 |
| `multidriver-swg` | Generated from template, pushed to GitHub (public), wired to ReadTheDocs, published to PyPI and TestPyPI |
| `claude-workspace` | Updated copier-specialist agent startup reads to include `COPIER_UPDATE_GUIDE.md` |

### Template bugfixes discovered during execution

| Bug | Cause | Fix | Tag |
|-----|-------|-----|-----|
| `markdown_extensions` block missing from generated `mkdocs.yml` | Block was added post-`v1.0.0` tag; Copier uses latest tag, not HEAD | Already in HEAD; tagged v1.0.1 to include it | v1.0.1 |
| `requirements.txt` missing from generated project | Same post-tag issue | Already in HEAD | v1.0.1 |
| Description double period (`statistics..`) | User entered description with trailing period; template appended another in some contexts | Added `(no trailing period)` to copier.yml help text; established convention: description stored without period, template appends `.` in prose contexts only | v1.0.1, v1.1.1 |

### Copier update tutorial results (v1.0.0 → v1.0.2)

- **17 files changed** (204 insertions, 526 deletions)
- **1 conflict** in `src/multidriver_swg/__init__.py` — description docstring diverged between local manual fix and template update; resolved by keeping the "after updating" version
- Key changes: `.prompts/` directory removed, `CLAUDE.md` restructured to reference shared `claude-workspace` docs, docs updated to use skill syntax, publish workflow added, `.gitignore` cleaned

### COPIER_UPDATE_GUIDE.md structure

Written at the template repo root (non-rendered reference). Covers:
- **Part 0**: New Project Setup (generation → git → ReadTheDocs → PyPI trusted publishers → test publish)
- **Part 1**: Making a Template Change
- **Part 2**: Propagating to a Downstream Project
- **Part 3**: Conflict Scenarios and Resolutions
- **Part 4**: Keeping Track of Versions
- **Part 5**: PyPI Publishing Setup

### PyPI publishing

- GitHub Actions workflow at `.github/workflows/publish.yml`
- Uses `pypa/gh-action-pypi-publish@release/v1` with OIDC trusted publishers (no API tokens)
- Flow: push `v*` tag → build → publish to TestPyPI → publish to PyPI
- GitHub Environments (`testpypi`, `pypi`) configured with default settings
- Successfully test-published `multidriver-swg` v0.1.0 to both registries

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Repo visibility | Public (plan said private) | Developer preference |
| Python version | 3.11 (plan said 3.12) | Developer's current target; template default updated to match |
| Description period convention | Stored without period; template appends `.` in prose | Prevents double-period bug; metadata fields (pyproject.toml, mkdocs.yml) don't need periods |
| Step 3.4 tutorial content | Real accumulated template changes, not synthetic edit | More realistic test; exercised conflict resolution |
| PyPI publishing | Trusted publishers (OIDC), not API tokens | Modern best practice; no secrets management needed |
| TestPyPI trigger | All version tags (not just pre-release) | Simpler; TestPyPI serves as a gate before PyPI |
| `build` package | Installed in CI runner, not as project dependency | Build tool, not runtime dependency |

---

## Definition of Done

- [x] `~/dev/multidriver-swg` exists and was generated from the template
- [x] `.copier-answers.yml` present with correct values
- [x] `.scratch/creation_log.md` contains full command transcript
- [x] `multidriver-swg` repo pushed to GitHub as public
- [x] ReadTheDocs build succeeds and all four verification points pass (Mermaid, admonition, sequence diagram, mkdocstrings)
- [x] Update tutorial completed: template improvements propagated to `multidriver-swg` via `copier update` (v1.0.0 → v1.0.2, one conflict resolved)
- [x] Developer described update outcome; Claude confirmed understanding
- [x] `COPIER_UPDATE_GUIDE.md` written at template repo root and committed (Parts 0–5)
- [x] `copier-specialist` agent startup reads updated to include `COPIER_UPDATE_GUIDE.md`
- [x] Agent startup reads verified by direct file inspection (deferred live self-report, consistent with Phase 2)
- [x] GitHub Actions publish workflow added to template (`template/.github/workflows/publish.yml`)
- [x] Template tagged through v1.1.1 and pushed
- [x] `multidriver-swg` updated via `copier update --skip-tasks` through v1.1.1
- [x] Trusted publishers configured on PyPI and TestPyPI for `multidriver-swg`
- [x] Test publish to both TestPyPI and PyPI succeeds (v0.1.0)
