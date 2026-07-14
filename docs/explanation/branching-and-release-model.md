# Branching and release model

This project uses a **gitflow-lite** model: a long-lived development branch (`develop`) that is the GitHub default, and a release-only `main` that advances only through validated releases.

## The two branches

- **`develop`** — the GitHub **default branch** and the primary local checkout. All day-to-day work happens here. Feature/worktree branches are created *from* `develop` and merged *back* into it. Read-the-Docs "latest" builds `develop`.
- **`main`** — **release-only**. `main` advances *only* via a `develop` → `main` release pull request that passes the release gate (all tests green, docs accurate and complete). Every release merge is tagged `vX.Y.Z`. Read-the-Docs "stable" builds the latest tag, so public visitors land on released docs.

A GitHub **ruleset** on `main` enforces this: pull-request-required-before-merge, linear history, `squash`/`rebase` merges only, and blocked force-pushes/deletions. The required status checks that gate a release PR (full test suite, docs build, CITATION.cff validation, identifier-blocklist guard) are configured separately as part of the release gate. The LAYOUT_VERSION check is NOT a required status check: it runs only under pre-commit (against `HEAD~1`), so layout-version discipline is enforced at `develop`-commit time and inherited by the release (see "Two independent version axes").

## Worktree workflow (unchanged)

The per-branch worktree workflow is unaffected by this model. Worktrees branch from the primary checkout's `HEAD`, so with the primary checkout on `develop`, new worktrees branch from `develop` automatically. Staging discipline and merge-back are branch-name-agnostic.

## Cutting a release

Releases are executed with the `/release-project` workflow, which opens the `develop` → `main` PR, verifies every required status check is green, merges, and tags the post-merge `main` commit `vX.Y.Z` (the tag — not `develop`'s tip — is what fires the PyPI publish and the Zenodo DOI mint). Never tag `develop`'s tip: that would publish an un-gated commit.

## Read-the-Docs "stable" default (operator runbook)

Read-the-Docs maps versions to branches/tags in the **project dashboard**, not in `.readthedocs.yaml`. Once the first `vX.Y.Z` tag exists, set the default version to **stable** so public visitors land on released docs:

1. Read-the-Docs dashboard → the `hhemt` project → **Admin → Settings**.
2. Set **Default version** to `stable`.
3. Confirm the `stable` version is **Active** under **Admin → Versions** (it is auto-created from the highest `vX.Y.Z` tag).

Until the first tag exists, `stable` has no build target — leave the default at `latest` (which tracks `develop`). This flip is performed at the first public release.

## Two independent version axes

Do not conflate these:

- **On-disk layout version** — `LAYOUT_VERSION` (`src/hhemt/version_migration/constants.py`), a monotonic integer governing on-disk analysis-tree/system-directory compatibility. Bumping it requires a migration module + golden fixtures (CI Check A/B, enforced at commit time via pre-commit, not in GitHub Actions).
- **Software release version** — the SemVer in `pyproject.toml` and the `vX.Y.Z` git tag, governing the PyPI/release artifact.

A release tag never touches `LAYOUT_VERSION`; a `LAYOUT_VERSION` bump never touches the SemVer. Because `check_layout_version.py` runs only under pre-commit (against `HEAD~1`) and is not wired into any GitHub Actions workflow, a release merge does not re-trigger the layout checks — the release inherits whatever `develop` already validated.
