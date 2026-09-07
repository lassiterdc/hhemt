# Branching and release model

This project uses a **gitflow-lite** model: a long-lived development branch (`develop`) that is the GitHub default, and a release-only `main` that advances only through validated releases.

## The two branches

- **`develop`**: the GitHub **default branch** and the primary local checkout. All day-to-day work happens here. Feature/worktree branches are created *from* `develop` and merged *back* into it. Read-the-Docs "latest" builds `develop`.
- **`main`**: **release-only**. `main` advances *only* via a `develop` → `main` release pull request that passes the release gate (all tests green, docs accurate and complete). Every release merge is tagged `vX.Y.Z`. Read-the-Docs "stable" builds the latest tag, so public visitors land on released docs.

A GitHub **ruleset** on `main` enforces this: pull-request-required-before-merge, linear history, `squash`/`rebase` merges only, and blocked force-pushes/deletions. The required status checks that gate a release PR (full test suite, docs build, CITATION.cff validation, identifier-blocklist guard) are configured separately as part of the release gate. The LAYOUT_VERSION check is not a required status check: it runs only from local pre-commit hooks -- at the pre-commit stage against `HEAD~1`..worktree, and at the pre-push stage against the commit range `HEAD~1`..`HEAD` under `--range` -- so layout-version discipline is enforced locally at `develop`-commit and `develop`-push time and inherited by the release (see "Two independent version axes").

## Branching for contributors

Feature branches are created from `develop` and merged back into it; `main` is
never a branch target for day-to-day work. See [Contributing](../contributing.md)
for the contribution process.

## What a release is

A release is a `develop` → `main` pull request that passes the release gate,
merged and then tagged `vX.Y.Z`. **The tag, not `develop`'s tip, is what fires
the PyPI publish and the Zenodo DOI mint**, which is why `develop` is never
tagged directly: doing so would publish a commit the release gate never saw.

This is the reason the two documentation versions differ. `latest` tracks
`develop` and shows unreleased content; `stable` tracks the newest tag and is
what a visitor arriving without a version in the URL should see.

## Tests the suite harness cannot evaluate

Some tests cannot run inside a SLURM allocation and are therefore **UNEVALUATED** by a
harness array run rather than passed. They are reported by node id in every run's
`summary.md`, under `## Complement`; read the list there rather than from a copy here,
because a hand-maintained copy drifts and the generated one cannot. At the time of
writing the list is 4 node ids across 2 modules.

They compile and run a solver, so running them needs a venue permitted to do that, which
a login node is not.

**Any red they produce, whenever and wherever they are run, must be fixed. Being
unevaluated is not an exemption from the zero-red bar.**

To make such a red visible rather than lost: save the run's `--junitxml` into a suite run
directory under the existing `chunk-NN.junit.xml` convention, beside a
`chunk-NN.status.json`, and add the chunk to the manifest's chunk list. The aggregator
then reports the red on the same surface as any array red. This is recording, not
blocking. Nothing refuses and nothing is computed as a disposition. The cost is low but
not zero: the minimum `status.json` shape is not yet specified.

## Two independent version axes

Do not conflate these:

- **On-disk layout version**: `LAYOUT_VERSION` (`src/hhemt/version_migration/constants.py`), a monotonic integer governing on-disk analysis-tree/system-directory compatibility. Bumping it requires a migration module + golden fixtures (Check A/B, enforced locally by pre-commit at both the pre-commit and pre-push stages, not in GitHub Actions).
- **Software release version**: the SemVer in `pyproject.toml` and the `vX.Y.Z` git tag, governing the PyPI/release artifact.

A release tag never touches `LAYOUT_VERSION`; a `LAYOUT_VERSION` bump never touches the SemVer. Because `check_layout_version.py` runs only from local pre-commit hooks -- at the pre-commit stage against `HEAD~1`..worktree, and at the pre-push stage against the commit range `HEAD~1`..`HEAD` under `--range` -- and is not wired into any GitHub Actions workflow, a release merge does not re-trigger the layout checks; the release inherits whatever `develop` already validated.
