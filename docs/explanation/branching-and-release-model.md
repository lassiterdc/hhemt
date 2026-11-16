# Branching and release model

The project uses a **gitflow-lite** model, with a long-lived development branch, `develop`, as the GitHub default and a release-only `main` that advances only through validated releases.

## The branches

- **`develop`** is the GitHub **default branch**. All day-to-day work happens here. Feature and worktree branches are created *from* `develop` and merged *back* into `develop`. Read-the-Docs "latest" builds `develop`.
- **`main`** is **release-only**. `main` advances *only* through a `develop` → `main` release pull request that passes the release gate. The gate requires all tests green and docs accurate and complete. Every release merge is tagged `vX.Y.Z`. Read-the-Docs "stable" builds the latest tag, so public visitors land on released docs.

A GitHub **ruleset** on `main` enforces the model. The ruleset requires a pull request before merge, and only the `merge` method is allowed, so `main` is always `develop` plus merge commits. Force-pushes and deletions are blocked. The same ruleset requires a release PR to pass the test workflow, the anonymization guard, the docs build, the CITATION.cff validation, the compile-bearing synth render gate, and the `release-pr-tree-identity` check. The LAYOUT_VERSION check runs from local pre-commit hooks, as the section [Layout version and release version](#layout-version-and-release-version) describes.

## Branching for contributors

Feature branches are created from `develop` and merged back into `develop`. See
[Contributing](../contributing.md) for the contribution process.

## Release definition

A release is a `develop` → `main` pull request that passes the release gate. The merge is then tagged `vX.Y.Z`.
**The tag fires the PyPI publish.** A separate step after the tag, publishing a GitHub release for the tag, creates the Zenodo archive and the version DOI.
Tag the release merge on `main`. A tag on `develop` would publish a commit the
release gate never saw.

The tag also separates the documentation versions. `latest` tracks
`develop` and shows unreleased content, and `stable` tracks the newest tag and is
the version a visitor sees without a version in the URL.

### Each check runs once per commit

Every heavy check except the compile gate runs on a push to `develop` and on a pull request into `develop`. The compile gate runs on a push to `develop` and on manual dispatch. No heavy check runs on a pull request into `main` or on a push to `main`. A release PR reads each check from the `develop` push run on the head commit, so the checks that certify the release run once per commit and not twice.

The develop push run certifies the head commit, while the merge lands the merge result. The `release-pr-tree-identity` job closes the difference. The job is cheap and runs only on a pull request into `main`. The job fails when the tree of the merge result differs from the tree of the PR head. The usual cause is content on `main` that `develop` lacks. Merge `main` into `develop` and wait for the checks on the new head.

The job compares the merge result that GitHub computed when the PR was opened or last pushed. A later commit to `main` leaves that result stale, because the job runs again only on a new push to the PR.

The compile gate reuses a pass record. The job `compile-synth-render-gate` runs on every push to `develop`, and `scripts/compile_gate_key.py` prints the key of a one-file record for the commit. The key hashes every tracked path except `docs/`, `.vale/`, and 6 root-level files of prose and release metadata. The 6 files are `HISTORY.md`, `CONTRIBUTING.md`, `CITATION.cff`, `.vale.ini`, `mkdocs.yml`, and `.readthedocs.yaml`. The key also carries the runner operating system and the ISO year and week, so a record expires with the calendar week of the run that wrote the record.

A hit skips every step after the lookup. The `Reuse pass record` step prints the commit, run id, attempt, and date of the full run that wrote the record. The step prints to the log and to the job summary. The job then concludes success. A miss runs the gate in full and saves the record after a pass. A push that changes only inert paths reuses a pass. A push that changes `src/`, `tests/`, `README.md`, `environment.yaml`, the workflow, or `pyproject.toml` runs in full, and a version bump is such a change. A manual dispatch with the `force_full_run` input set to true skips the lookup and runs the gate in full.

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

**The venue declaration is the other half.** A session with no way to compile skips the compile tier
and leaves the tier with no failure, and a skip is not a pass. The release gate needs both exports,
`HHEMT_COMPILE_VENUE=toolchain` and `HHEMT_REQUIRE_COMPILE_TIER=1`.
[Contributing](../contributing.md#workflow) lists the effect of each export and the conditions that make a run
count as a gate result. [`hhemt test toolkit`](../reference/cli.md#hhemt-test-toolkit)
documents the record that a run leaves of the exports.

**A release head's compile context may be a reuse of an identical compile-relevant tree.** When the commits since the last full run changed only inert paths, the pass record of that earlier run is the gate result for the release head. The earlier run was a full run on a fresh runner with both exports set, over a tree identical in every compile-relevant path, and the `Reuse pass record` step names the earlier run. The release walk cites that run in the release record.

### Release step venues

Each command of the release walk has a venue and a tool. A machine without the documentation dependencies or a compiler runs only some of the steps, so the other steps run in CI or in a compiling venue. A named result stands in for a step on a machine that runs the step elsewhere.

| Step | Venue | Tool required | Stand-in |
|---|---|---|---|
| `bash scripts/docs_gates.sh` | CI `docs-build` job | The `docs` extra and Vale | The `docs-build` result on the pushed head |
| `hhemt cache clear --apply` | A compiling venue | `hhemt`, with the `src` directory of the checkout on `PYTHONPATH` or `HHEMT_ALLOW_INSTALLED=1` set when the shell stands in a git worktree | None. On a machine with no permitted compile venue the command exits 2 by design, because the compile tier stays in place. |
| The release-gate test run with `HHEMT_COMPILE_VENUE=toolchain` and `HHEMT_REQUIRE_COMPILE_TIER=1` | A compiling venue | `hhemt` and a compiler | The `compile-synth-render-gate` pass record |
| `python scripts/check_release_metadata.py` | Any machine with the tool | Python | None |
| `python scripts/check_ruleset_sync.py --static --live` | Any machine with the tool | Python and an authenticated `gh` | None |
| `git tag -a vX.Y.Z -m "..."` and `git push origin vX.Y.Z` | Any machine with the tool | `git` | None |
| `just tag` | Any machine with `just` | `just` | The commands of the 3 preceding rows, run in that order after `git diff --quiet HEAD -- CITATION.cff pyproject.toml HISTORY.md` shows no change |

The CI `docs-build` job runs `scripts/docs_gates.sh` bare, so the result of the job on the pushed head stands for the docs gates on a machine without the docs dependencies. The release record cites that run.

`hhemt cache clear --apply` removes the 2 cheap cache tiers and leaves the compiled-build tier in place on a machine with no permitted compile venue. The command then exits 2, as the [CLI reference](../reference/cli.md#development) records. The exit is intended, because a machine that lacks the means to rebuild the tier keeps the tier.

Run from a git worktree, an `hhemt` command exits 99 when the shared conda environment holds an editable install of another checkout. Put the `src` directory of the checkout on `PYTHONPATH`, as in `PYTHONPATH=$PWD/src hhemt cache clear`, or set `HHEMT_ALLOW_INSTALLED=1` to run the installed code on purpose.

A release needs no `just`. The `just tag` row lists the commands of the recipe, so a machine without `just` runs the commands of the 3 preceding rows by name, in the order given.


## Tests carrying a structural gate

A small set of tests declares a `skipif` on `on_scheduler_node()`, defined in `tests/utils_for_testing.py`.
The function returns true whenever a scheduler job variable such as `SLURM_JOB_ID` is set. Every chunk of a
harness array run is a SLURM job, so the gate fires inside the array. The array declines the test
and attempts no run of the test, and a declined test is not a passed test.

Membership is fixed at plan
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
