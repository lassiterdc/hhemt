# Branching and release model

This project uses a **gitflow-lite** model: a long-lived development branch (`develop`) that is the GitHub default, and a release-only `main` that advances only through validated releases.

## The two branches

- **`develop`**: the GitHub **default branch** and the primary local checkout. All day-to-day work happens here. Feature/worktree branches are created *from* `develop` and merged *back* into it. Read-the-Docs "latest" builds `develop`.
- **`main`**: **release-only**. `main` advances *only* via a `develop` → `main` release pull request that passes the release gate (all tests green, docs accurate and complete). Every release merge is tagged `vX.Y.Z`. Read-the-Docs "stable" builds the latest tag, so public visitors land on released docs.

A GitHub **ruleset** on `main` enforces this: pull-request-required-before-merge, linear history, `squash`/`rebase` merges only, and blocked force-pushes/deletions. The status checks a release PR must pass (the test workflow, the anonymization guard, the docs build, CITATION.cff validation, and the compile-bearing synth render gate) are required by the same ruleset. The LAYOUT_VERSION check is not a required status check: it runs only from local pre-commit hooks, at the pre-commit stage against `HEAD`..worktree and at the pre-push stage against the commit range `HEAD~1`..`HEAD` under `--range`. Layout-version discipline is therefore enforced locally at `develop`-commit and `develop`-push time and inherited by the release (see "Two independent version axes").

## Branching for contributors

Feature branches are created from `develop` and merged back into it; `main` is
never a branch target for day-to-day work. See [Contributing](../contributing.md)
for the contribution process.

## What a release is

A release is a `develop` → `main` pull request that passes the release gate,
merged and then tagged `vX.Y.Z`. **The tag, not `develop`'s tip, is what fires
the PyPI publish and the Zenodo DOI mint**, which is why `develop` is never
tagged directly: doing so would publish a commit the release gate never saw.

Because the tag fires the mint, two fields in `CITATION.cff` have to be correct
*before* it is cut, not after: `version` must match the version in
`pyproject.toml` that the tag is named from, and `date-released` must be the day
the tag is created. `just tag` checks both and refuses to tag if either is wrong,
so this is a step you are stopped at rather than one you have to remember.

The reason it cannot be fixed afterwards is that the tag freezes an archive. The
`v0.1.0` archive carries no `date-released` at all, because that field was added
in a commit that is not an ancestor of the tag; a later correction reaches the
repository but never the tarball. What is checked here is agreement and dating
only. Which DOI each surface carries is a recorded convention rather than a
checked one, and is described under "DOI kind convention" in `architecture.md`.

This is the reason the two documentation versions differ. `latest` tracks
`develop` and shows unreleased content; `stable` tracks the newest tag and is
what a visitor arriving without a version in the URL should see.

## Tests carrying a structural gate

A small set of tests declares a `skipif` on `on_scheduler_node()` (`tests/utils_for_testing.py`),
which is true whenever a scheduler job variable such as `SLURM_JOB_ID` is set. Every chunk of a
harness array run is a SLURM job, so inside the array the gate fires: the array declines the test
rather than attempting it, and a declined test is not a passed one. Membership is fixed at plan
time, not at run time: a test belongs to the set when any `skipif` reason it declares is one the
aggregator lists as structural (`STRUCTURAL_SKIP_REASONS` in `suite/aggregate.py`), even when a
different gate is the one pytest reports. Every run's `summary.md` reports the set in the section
whose heading begins `## Complement`: the heading carries the size of the declared set, the line
under it counts how many of those tests this run did not attempt, and the members not attempted are
listed by node id. Read the list there rather than from a copy here, because a hand-maintained copy
drifts and the generated one cannot.

The gate is a declaration, not an impossibility. Off a scheduler node it does not fire, and a
test carrying it runs like any other, provided the venue is one the project permits to build and
execute the solver (`HHEMT_COMPILE_VENUE=toolchain`; see [Contributing](../contributing.md)).

**Any red these tests produce, whenever and wherever they are run, must be fixed. Being declined by
the array is not an exemption from the zero-red bar.**

To make such a red visible rather than lost: save the run's `--junitxml` into a suite run
directory under the existing `chunk-NN.junit.xml` convention, beside a
`chunk-NN.status.json`, and add the chunk to the manifest's chunk list. The aggregator
then reports the red on the same surface as any array red. This is recording, not
blocking: nothing refuses. What the aggregator computes from it is coverage, not a
verdict. The complement section counts how many declared tests the run did not attempt.
A `--scope union` aggregation stands only when some manifest chunk entry carries
`covers: complement` and every declared test was attempted; otherwise it is downgraded to
`array` (see the [CLI reference](../reference/cli.md#hhemt-test-toolkit)). The cost is low
but not zero: the minimum `status.json` shape is not yet specified.

## Two independent version axes

Do not conflate these:

- **On-disk layout version**: `LAYOUT_VERSION` (`src/hhemt/version_migration/constants.py`), a monotonic integer governing on-disk analysis-tree/system-directory compatibility. Bumping it requires a migration module + golden fixtures (Check A/B, enforced locally by pre-commit at both the pre-commit and pre-push stages, not in GitHub Actions).
- **Software release version**: the SemVer in `pyproject.toml` and the `vX.Y.Z` git tag, governing the PyPI/release artifact.

A release tag never touches `LAYOUT_VERSION`; a `LAYOUT_VERSION` bump never touches the SemVer. `check_layout_version.py` runs only from local pre-commit hooks (at the pre-commit stage against `HEAD`..worktree, and at the pre-push stage against the commit range `HEAD~1`..`HEAD` under `--range`) and is not wired into any GitHub Actions workflow, so a release merge does not re-trigger the layout checks; the release inherits whatever `develop` already validated.
