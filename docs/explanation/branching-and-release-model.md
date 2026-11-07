# Branching and release model

The project uses a **gitflow-lite** model, with a long-lived development branch, `develop`, as the GitHub default and a release-only `main` that advances only through validated releases.

## The branches

- **`develop`** is the GitHub **default branch**. All day-to-day work happens here. Feature and worktree branches are created *from* `develop` and merged *back* into `develop`. Read-the-Docs "latest" builds `develop`.
- **`main`** is **release-only**. `main` advances *only* through a `develop` → `main` release pull request that passes the release gate. The gate requires all tests green and docs accurate and complete. Every release merge is tagged `vX.Y.Z`. Read-the-Docs "stable" builds the latest tag, so public visitors land on released docs.

A GitHub **ruleset** on `main` enforces the model. The ruleset requires a pull request before merge and linear history. Only `squash` and `rebase` merges are allowed. Force-pushes and deletions are blocked. The same ruleset requires a release PR to pass the test workflow, the anonymization guard, the docs build, the CITATION.cff validation, and the compile-bearing synth render gate. The LAYOUT_VERSION check runs from local pre-commit hooks, as the section [Layout version and release version](#layout-version-and-release-version) describes.

## Branching for contributors

Feature branches are created from `develop` and merged back into `develop`. See
[Contributing](../contributing.md) for the contribution process.

## Release definition

A release is a `develop` → `main` pull request that passes the release gate. The merge is then tagged `vX.Y.Z`.
**The tag fires the PyPI publish and the Zenodo DOI mint.**
Tag the release merge on `main`. A tag on `develop` would publish a commit the
release gate never saw.

The tag also separates the documentation versions. `latest` tracks
`develop` and shows unreleased content, and `stable` tracks the newest tag and is
the version a visitor sees without a version in the URL.

### The release gate runs on a cleared cache, in a compiling venue

**Both halves are conditions of the gate.** A cached run reuses analysis trees and
compiled builds, so a cached run shows only that the cached artifacts satisfied the tests.
The run leaves open whether the sources of this commit build and pass, a weaker claim
than the "all tests green" claim of the gate. Clear the caches
before the gate runs:

```bash
hhemt cache clear --apply
```

Run the command on the machine that will run the gate, immediately before submitting. The
[CLI reference](../reference/cli.md#development) describes the listing that the bare
`hhemt cache clear` prints, the `--apply` flag, and the exit status.

**The venue declaration is the other half.** A session that is unable to compile skips the compile tier
and leaves the tier unfailed, and a skip is not a pass. The release gate needs both exports,
`HHEMT_COMPILE_VENUE=toolchain` and `HHEMT_REQUIRE_COMPILE_TIER=1`.
[Contributing](../contributing.md#workflow) lists the effect of each export and the conditions that make a run
count as a gate result. [`hhemt test toolkit`](../reference/cli.md#hhemt-test-toolkit)
documents the record that a run leaves of the exports.


## Tests carrying a structural gate

A small set of tests declares a `skipif` on `on_scheduler_node()`, defined in `tests/utils_for_testing.py`.
The function returns true whenever a scheduler job variable such as `SLURM_JOB_ID` is set. Every chunk of a
harness array run is a SLURM job, so the gate fires inside the array. The array declines the test
and attempts no run of the test, and a declined test is not a passed test. Membership is fixed at plan
time. A test belongs to the set when any `skipif` reason that the test declares is a reason the
aggregator lists as structural, `STRUCTURAL_SKIP_REASONS` in `src/hhemt/suite/aggregate.py`, even when a
different gate is the gate pytest reports. Every run's `summary.md` reports the set in the section
whose heading begins `## Complement`. The heading carries the size of the declared set, the line
under the heading counts the tests of the set that the run did not attempt, and the members not attempted are
listed by node id. Read the list in `summary.md` and skip any copy elsewhere, because a hand-maintained copy
drifts and the generated list stays current.

The gate is a declaration. Off a scheduler node, a test carrying the gate runs like any other test, provided the venue is one the project permits to build and
execute the solver. [Contributing](../contributing.md) covers `HHEMT_COMPILE_VENUE=toolchain`.

**Fix any red that a test of the set produces. The obligation holds at any time and on any machine. The zero-red bar applies to a
test that the array declines.** [`hhemt test toolkit`](../reference/cli.md#hhemt-test-toolkit) describes the reporting of a run's coverage of the set.

## Layout version and release version

The versions change independently:

- **On-disk layout version.** `LAYOUT_VERSION` in `src/hhemt/version_migration/constants.py` is a monotonic integer governing on-disk analysis-tree and system-directory compatibility. Bumping the integer requires a migration module and golden fixtures. Check A grades a bump, and Check B grades a change to a layout-relevant file that carries no bump. Both run from local hooks, and neither runs in GitHub Actions.
- **Software release version.** The SemVer in `pyproject.toml` and the `vX.Y.Z` git tag govern the PyPI release artifact.

A release tag never touches `LAYOUT_VERSION`, and a `LAYOUT_VERSION` bump never touches the SemVer. `check_layout_version.py` runs only from local pre-commit hooks, and no GitHub Actions workflow invokes the script, so a release merge re-triggers no layout check. The release inherits whatever `develop` already validated.

At the pre-commit stage, Check A grades the commit being created, comparing `HEAD` with the working tree. At the pre-push stage, Check A grades `HEAD~1`..`HEAD` and Check B grades each commit being pushed separately. When the push names a remote, Check B skips any commit already reachable from that remote's tracking refs, so a correct merge of `develop` is not graded again.
