<!-- hhemt:personal-voice exempt=prose -->
# Contributing

## Licensing of contributions

The project source is licensed under AGPL-3.0-or-later. The `LICENSE` file and the [License section](index.md#license) of the Home page give the full statement.

By submitting a contribution to this project, whether a pull request, a patch, or other work, you agree that:

1. **Inbound = outbound.** Your contribution is provided under the same
   license as the project, AGPL-3.0-or-later, and
   you have the right to submit the contribution under that license.

2. **Forward relicensing.** You grant the project maintainer, the owner of the repository, permission to
   license your contribution under any future license the project
   adopts, including a permissive OSI-approved open-source license
   such as Apache License 2.0. The grant lets the project relicense
   in the future without sign-off from every past contributor.

If you do not agree to the terms for a particular contribution, open an issue to discuss the contribution before submitting.

## Development setup

1. Fork and clone the repository.
2. Create the supported conda environment by following
   [Installation](how-to/installation.md#option-a-the-recommended-full-conda-environment),
   including both `--no-deps` flags. The installation page gives the reason for each flag.
3. Install the `[docs]` extra to build the documentation with `pip install -e ".[docs]"`.
   `environment.yaml` ships no docs tooling, so this step follows step 2.
   Add no `--no-deps` flag here, because the docs tooling has no
   conda-resolved counterpart and the flag would install none of the tooling.
   An editable install names a single `src`, so running the command from a worktree re-points
   `import hhemt` in the shared conda environment at that worktree until the next install.
   With several checkouts, `scripts/hook_env.sh --python 3.12 mkdocs serve` needs no install step and
   binds to the checkout that runs the command.
4. Install `uv` from https://docs.astral.sh/uv/. Every pre-commit hook and every uv-hosted `just` recipe in this
   repository runs through `scripts/hook_env.sh`, so `git commit` fails without `uv`. The wrapper stores dependencies
   in shared environments under `~/.cache/hhemt/hook_envs/`, keyed by the lock file, the Python version, and the
   extras, and binds each run to the checkout that runs the wrapper. The wrapper creates no `.venv/` in a tree, and
   environments unused for 14 days are removed automatically. Set `$HHEMT_HOOK_ENV_ROOT` or `$XDG_CACHE_HOME` to move
   the root. The shared environments stay separate from the conda environment that step 2 creates, so both exist on
   the machine. To point an editor at the shared environment, use the `bin/python` of the path that
   `scripts/hook_env.sh --print-env --python 3.12` prints. Inside an activated virtualenv, the wrapper clears
   `VIRTUAL_ENV` before the `uv` call, so the hooks print no warning.
5. Install the pre-commit hooks with `pre-commit install`. The `pre-commit` package ships
   only with the conda environment defined in `environment.yaml`, and no
   `pyproject.toml` extra declares `pre-commit`. After an install through Option B of
   `docs/how-to/installation.md`, run `pip install pre-commit` first.
   **Rerun `pre-commit install` after pulling when hooks are already installed.**
   The repository declares `default_install_hook_types: [pre-commit, pre-push, commit-msg]`,
   and git reads that key only at install time, so a clone that installed before the
   key existed keeps `.git/hooks/pre-commit` alone and never fires the pre-push and commit-msg
   guards. Re-running is idempotent. The command leaves existing hook files byte-identical
   and adds the missing ones.

## Workflow

- Open an issue before you build when a requirement is unclear or the sources disagree. State the intended change and how a reviewer can tell that the change works in the issue or the pull request description. Update the description when the approach, the scope, or the risk changes.
- Create a feature branch from `develop`.
- Add a test with each change. A bug fix carries a test that fails before the fix, and new behaviour carries a test.
- Run `just qa` before opening a pull request. The `qa` recipe formats, lints, type-checks, runs
  `scripts/check_du_sentinel_sites.py` and `scripts/check_vocabulary_freeze.py`, and runs
  the fast test tier. To gate the compile tier explicitly, run `just test-gated`. The `test-gated` recipe
  invokes pytest under the conda env with `HHEMT_REQUIRE_COMPILE_TIER=1`, so a
  compile-tier skip becomes a hard failure. Neither `environment.yaml` nor any
  `pyproject.toml` extra installs `just`. Install `just` from https://github.com/casey/just, or run the commands under the `qa:`
  recipe in `justfile` yourself, in the order they appear there.
- The suite has [a fast tier and a compile tier](explanation/testing-the-toolkit.md#tiers-and-machine-independence).
  `just qa` runs the fast tier. Gate A and Gate B cover the tiers.
    - **Gate A, `just test-fast`.** Gate A runs the fast tier without the tests marked `slow`. Run `just test-fast` before opening a pull request.
      CI differs. `.github/workflows/test.yml` runs bare `pytest` with no
      marker filter, and `compile-tests.yml` must be a required status
      check before a release PR. `just qa` skips the `test-fast` recipe and runs a
      `pytest` call that carries the same `-m` expression, so both apply the same marker
      filtering. The collected populations can still differ, because the dependency sets differ. `qa` runs `scripts/hook_env.sh --python 3.12`, and
      `test-fast` uses the conda `hhemt` environment. `environment.yaml` pins that
      environment to Python 3.11. A module-level import present in one resolution and
      absent from the other changes the collected set before pytest evaluates any marker.
    - **Gate B, `just test-gated`.** Gate B runs the whole suite with no marker filter, compile
      tier included. Gate B requires the `hhemt` conda environment for `cmake` and `mpic++`, takes
      longer than Gate A, and is the invocation that gates a release.
      Run Gate B detached, with `nohup … &`, `tmux`, or
      your cluster's batch scheduler.
    - **The compile venue.** The guard is armed by default. A pytest session must declare a
      permitted *venue* before the session builds the solver or executes SWMM. `just test-gated`
      sets `HHEMT_COMPILE_VENUE=toolchain` for you, so following the recipe needs no
      environment work. Set the variable by hand only when you invoke `pytest` directly and
      want the compile tier. Neither export implies the other. A run with the venue declared
      and the requirement unset is green even when no compile is possible. A run with the requirement
      set and no venue declared refuses every compile. A run with neither export recorded
      is not a gate result.
    - **Venue of compile-tier runs.** The venue token names a capability, and no code names a cluster, so a
      machine with the solver's toolchain can run the compile tier. `compile-tests.yml` runs a named subset of the
      compile tier after a change merges into `develop`, and Gate B runs the whole tier before a release, so you need
      not run the compile tier yourself. Tests that compile no solver and
      execute no simulation run on any machine. A suite-level green claim requires `scope=union`.
      The `hhemt test toolkit` CLI produces no `scope=union` result, as the
      [CLI reference](reference/cli.md#hhemt-test-toolkit) records, and
      [the testing page](explanation/testing-the-toolkit.md#scope-of-a-green-claim)
      gives the reason.
      The guard shape is explained under
      [the compile venue token](explanation/testing-the-toolkit.md#the-compile-venue-token).
  Every machine gets the same tier from `just qa`, as
  [Tiers and machine independence](explanation/testing-the-toolkit.md#tiers-and-machine-independence) explains.
  The first run of either gate downloads the example data from HydroShare
  when `test_data/norfolk_coastal_flooding` is absent. The download costs time once
  per machine and can dominate the wall clock on a slow link.
- Open a pull request into `develop`. Name in the description each change that is visible when someone runs the toolkit. The `HISTORY.md` entry is written at release.

## Documentation

Build the documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

The install line re-points the shared conda environment's editable install at the
checkout you run the install from, as step 3 of the development setup explains. From a second
checkout, `scripts/hook_env.sh --python 3.12 mkdocs serve` builds the same site with no
install step.

## Writing documentation pages

**State the invocation without the flag list.** A page that documents a command that takes configs shows the config-driven form, `hhemt {verb} --system-config CFG_SYS --analysis-config CFG_ANA` or the `python -m hhemt.{module}` equivalent, and names the configs the command reads. Describe a command by the configs the command takes. `hhemt {command} --help` is the authoritative list of flags, and prose that repeats the list goes stale.

**An interactive session belongs on a tutorial, and only there.** A page that drives `analysis.run()`, `analysis.reprocess()`, or `analysis.render_report()` from Python is a tutorial page and states that the form is a Python session.

**Add no verb count or verb list.** `hhemt --help` enumerates the verbs. The verb tables on the [CLI reference](reference/cli.md) are the subject of that page, so this rule binds new prose only. A sentence that counts or lists the commands is a second copy that goes stale.

**Open every how-to page with the same sections.** Each page under `docs/how-to/` opens with the sections Goal, Prerequisites, and Key references, in that order. `docs/how-to/index.md` is the only exception.

The tests under `tests/docs/` check the stylistic choices and other requirements that keep the documentation consistent. Most checks accept an exception listed in an allowlist file.

## Documentation gates

CI checks every push to `main` or `develop` and every pull request into either
branch against the gates in `scripts/docs_gates.sh`:

- the strict build, including the htmlproofer internal-link check
- public-API autodoc coverage
- docs content hygiene
- docstring dialect
- published-surface fidelity
- the Vale style check, run with the `vale` version that `.github/workflows/docs-build.yml` installs

The Vale gate needs the `vale` executable on `PATH`. Install the version that CI pins with `pip install vale==3.26.0.1`.

Run the same list locally before opening a pull request:

```bash
RUN="scripts/hook_env.sh --python 3.12" bash scripts/docs_gates.sh
```

With `just` installed, `just docs-check` runs the same line. `scripts/hook_env.sh` binds `import hhemt` to the
checkout that runs the command, so the autodoc gate grades that checkout and the `population root:` line of the
gate names the tree the gate enumerated. Avoid `uv run --active` here. Under a foreign `VIRTUAL_ENV`, the command
re-points the editable install of another checkout.

Separately from the gates, `python scripts/check_anonymization.py` fails when a tracked file contains an identifier from the project's private-identifier blocklist. CI runs the script as the `anonymization-guard` job, and the pre-commit configuration runs the script at both the commit and push stages.

## Documentation gate exemptions

A page may declare an exemption from part of the documentation content gate.
The declaration is an HTML comment on the first line of the page. The comment names a marker and
the class groups the marker exempts. The marker table is generated from the constants of the gate at
documentation build time and always agrees with them.

<!-- hhemt:marker-governance -->

An exemption names a ground, and the ground decides whether the exemption may be
withdrawn. A ground is a recorded ruling or the word `incidental`. An
incidental exemption exists because the page needed relief from a single class and
took the whole group along. Anyone who removes that need may delete the
declaration. A ruling-backed exemption stays until the ruling is reopened.
Add a marker only when the page fails the class that the marker
escapes. A marker states a fact about the file and silences no finding.

## Branching and releases

Read [Branching and release model](explanation/branching-and-release-model.md) for the branches, contributor branching, the release definition, the tests that carry a structural gate, the layout version, and the release version.

## Code conventions

`just qa` and the pre-commit hooks check formatting, import order, and the `E`, `W`, `F`, `I`, `B`, and `UP` lint rule families that `[tool.ruff.lint]` in `pyproject.toml` selects, and `just qa` also type-checks with `ty`. Review checks the conventions that no tool checks, and the rest of this section states them.

### Unused code

The `F` family flags an unused import with `F401` and an unused local variable with `F841`. Both `just qa` and the pre-commit hook run `ruff check --fix`. Outside an `__init__.py` file, the fixer deletes an unused import without asking and leaves an unused local variable in place. In an `__init__.py` file, the fixer deletes no import. No selected rule flags an unused function argument, so review removes the argument unless an interface fixes the signature.

An import whose only job is a side effect, such as `import rioxarray` registering the `.rio` accessor, needs `# noqa: F401` on the line and a comment that names the side effect, because the fixer deletes the line otherwise. The form `import x as _` has no compliant alternative, because the alias still trips `F401`. To bind a name only to trigger a property that raises, write `_ = system.analysis`, a form that `F841` accepts.

### Suppressing lint findings

Comply with a finding by default. Suppress a finding only when the finding is wrong about this code, when complying would change behaviour, or when complying would delete information such as a column alignment or an assertion. Satisfying a finding by weakening a check is not compliance. Writing `strict=False` to satisfy `B905` on `zip(a, b)` keeps the silent truncation that `strict=True` turns into a `ValueError`.

Write the reason as a fact that a reader can check. The reason in `# noqa: B905 - the lists are appended in separate statements in a background thread` points at a fact, and a word such as "intentional" asserts only that someone made the choice. When several sites share one ground, such as a Python floor or a dependency version, write the ground once and point the other sites at that statement. The configuration selects no rule that reports a stale `# noqa`, so no tool notices a reason that has gone stale.

Put the suppression at the smallest scope where the reason is true. The choices are a `# noqa: XXX` on the line, a `per-file-ignores` entry in `pyproject.toml` with the reason in a comment above the entry, and a global `ignore`. No suppression has earned a global `ignore`. The `B008` entries for `src/hhemt/cli.py` and `src/hhemt/version_migration/__main__.py` and the `E501` entry for `scripts/rollin_status.py` show the form. Open an issue for a real defect that the pull request leaves unfixed, and cite the issue in the reason.

### Public names and compatibility

Rename or remove a Python name in place. Delete the old name and update every call site in the same pull request. Keep no alias, shim, or warning for the old name, because git history records the name. A migration covers each case that outlives the code:

- A retired or renamed key in a system, analysis, or report configuration file still loads, with a `DeprecationWarning` that says where the key went.
- A change to the on-disk layout bumps `LAYOUT_VERSION` and adds a migration under `src/hhemt/version_migration/`. At commit, `scripts/check_layout_version.py` checks that a bump carries a migration and fixtures. At push, the script checks that a change to a layout-relevant file carries a bump or an allowlist entry.

### Design conventions

- Give a parameter a default only when one value is almost always correct, as with `verbose=True`. A configuration field that a person populates takes the most care, because that person makes an intentional choice about every input.
- Raise an exception on failure in a critical path. Never return `False` or `None` silently. Put the file paths, return codes, and log locations in the message, so that the failure is actionable.
- Check that a step completed through the log of the step. A file can exist and be corrupt, incomplete, or left over from a failed earlier run. Checking that an input file exists before reading the file is the exception.
- Put machine-specific information in the configuration file that each deployment defines, and write no path or machine constant into core code.

### Docstrings and type hints

Write type hints and a NumPy-style docstring, with the `Parameters`, `Returns`, and `Raises` sections of the `numpydoc` convention, for every function you write or modify. The documentation site renders docstrings with the `mkdocstrings` Python handler configured with `docstring_style: numpy`. In a touched script, apply the standard to existing code as the script changes, and skip a global retrofit pass.

## Repository map

- `docs/contributing.md` is this file, the contributor guide.
- `CONTRIBUTING.md` is a short file at the repository root that points here.
- `architecture.md` covers project structure, key modules, and data flow.
- `docs/` holds the source for the published documentation site, built with MkDocs.

