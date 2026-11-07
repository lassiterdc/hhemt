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

If you are unable to agree to the terms for a particular contribution, open an issue to discuss the contribution before submitting.

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
   `scripts/hook_env.sh --print-env --python 3.12` prints. Inside an activated virtualenv, the wrapper unsets
   `VIRTUAL_ENV` before the `uv` call, so the hooks print no warning.
5. Install the pre-commit hooks with `pre-commit install`. The `pre-commit` package ships
   only with the conda environment defined in `environment.yaml`, and no
   `pyproject.toml` extra declares `pre-commit`. After an install through Option B of
   `docs/how-to/installation.md`, run `pip install pre-commit` first.
   **Re-run `pre-commit install` after pulling when hooks are already installed.**
   The repository declares `default_install_hook_types: [pre-commit, pre-push, commit-msg]`,
   and git reads that key only at install time, so a clone that installed before the
   key existed keeps `.git/hooks/pre-commit` alone and never fires the pre-push and commit-msg
   guards. Re-running is idempotent. The command leaves existing hook files byte-identical
   and adds the missing ones.

## Workflow

- Create a feature branch from `develop`.
- Make changes with tests.
- Run `just qa` before opening a pull request. The `qa` recipe formats, lints, type-checks, runs
  `scripts/check_du_sentinel_sites.py` and `scripts/check_vocabulary_freeze.py`, and runs
  the fast test tier. To gate the compile tier explicitly, run `just test-gated`. The `test-gated` recipe
  invokes pytest under the conda env with `HHEMT_REQUIRE_COMPILE_TIER=1`, so a
  compile-tier skip becomes a hard failure. Neither `environment.yaml` nor any
  `pyproject.toml` extra installs `just`. Install `just` from https://github.com/casey/just, or run the commands under the `qa:`
  recipe in `justfile` yourself, in the order they appear there.
- The suite has [a fast tier and a compile tier](explanation/testing-the-toolkit.md#tiers-and-machine-independence).
  `just qa` runs the fast tier. Gate A and Gate B cover the tiers:
    - **Gate A, `just test-fast`.** Gate A runs the fast tier without the tests marked `slow`. Run `just test-fast` before opening a pull request.
      CI differs. `.github/workflows/test.yml` runs bare `pytest` with no
      marker filter, and `compile-tests.yml` must be a required status
      check before a release PR. `just qa` skips the `test-fast` recipe and inlines a
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
      sets `HHEMT_COMPILE_VENUE=toolchain` for you, so following the recipe above needs no
      environment work. Set the variable by hand only when you invoke `pytest` directly and
      want the compile tier. Neither export implies the other. A run with the venue declared
      and the requirement unset is green while unable to compile. A run with the requirement
      set and no venue declared refuses every compile. A run with neither export recorded
      is not a gate result.
    - **Venue of compile-tier runs.** The code leaves the venue unenforced. The project rule places test runs that
      compile the solver or execute a simulation on an HPC cluster with the
      solver's toolchain, on the cluster's scratch filesystem. The rule covers
      Gate B and the scheduler-array suite alike. The toolkit imposes no such restriction, because the venue token names a capability and
      no code names the cluster that runs the tests. Tests that compile no solver and
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
- Submit a pull request.

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

**State the invocation without the flag list.** A page that shows how to invoke a command that takes configs shows the config-driven form, `hhemt {verb} --system-config CFG_SYS --analysis-config CFG_ANA` or the `python -m hhemt.{module}` equivalent, and names the configs the command reads. Describe a command by the configs the command takes. `hhemt {command} --help` is the authoritative list of flags, and prose that repeats the list goes stale.

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
- published-surface fidelity

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
the class groups the marker exempts. The table below is generated from the constants of the gate at
documentation build time and always agrees with them.

<!-- hhemt:marker-governance -->

An exemption names a ground, and the ground decides whether the exemption may be
withdrawn. A ground is a recorded ruling or the word `incidental`. An
incidental exemption exists because the page needed relief from a single class and
took the whole group along. Anyone who removes that need may delete the
declaration. A ruling-backed exemption stays until the ruling is reopened.
Add a marker only when the page fails the class that the marker
escapes. A marker states a fact about the file and silences no finding.

The page carries a personal-voice exemption. The page is written in the voice of the author, so the content gate's prose class skips the page. The style test grades the page like
every other page.

## Branching and releases

Read [Branching and release model](explanation/branching-and-release-model.md) for the branches, contributor branching, the release definition, the tests that carry a structural gate, the layout version, and the release version.

---

## Development Principles

### Ask before assuming
When a requirement is unclear or sources disagree, ask in an issue on the repository, or in the pull request description, before you proceed.

### Describe a change before building
State the intended change and the success criteria in the issue or the pull request description before you build. When the work shows that the description needs a change, such as a deviation from the approach, a scope change, or a new risk, update the description before you continue and avoid adapting silently.

### Prefer the correct approach over the cheap approach
- Look for better ways of achieving development goals and raise the ideas
- Raise a concern in the pull request when a design decision diverges from best practices
- Look for opportunities to make code more efficient, such as vectorizing operations and avoiding loops with pandas

### Backward compatibility is NOT a priority

**Rationale.** Clean code matters more than preserved APIs. Git history is the safety net.

When refactoring:
- ❌ No warnings that announce the removal of an old API
- ❌ No old APIs kept "for compatibility"
- ❌ No compatibility shims or aliases
- ✅ Do update all usage sites immediately
- ✅ Do delete obsolete code completely

### Most function arguments should not have defaults

Default function arguments can lead to unexpected behavior that is hard to debug. Avoid default values unless a default is almost always the correct choice, as with `verbose=True`. Configuration fields that a person populates call for the most care, because that person should make an intentional choice about every input.

### Avoid aliases

Do not create aliases for functions, classes, or variables. An alias is a second name for the same object. An alias creates confusion about the authoritative name and is a form of backward-compatibility shim. Rename the item and update all call sites when something needs renaming.

### No cruft/all variables, imports, and function arguments must be used

Unused elements signal that an implementation may be incomplete. Treat unused elements as an investigation trigger and as more than lint to suppress.

If you come across an unused variable, import, or function argument, investigate before removing:
1. Check whether the surrounding implementation is incomplete
2. Search the git history and the issue tracker for changes to that function, and determine whether a later use is intended
3. If the cause is still unclear, post hypotheses about why the element exists in an issue or on the pull request
4. Exception A applies to elements included for work in progress in the same pull request. Mark each with a comment that links the issue tracking that work
5. Exception B applies to an element whose *evaluation* is the point, where the name is unread and the statement is load-bearing. One case is an import whose side effect is registration, as `import rioxarray` registers the `.rio` accessor. Another case is a bind whose only job is to trigger a property that raises. Prefer the compliant form when a compliant form exists and is a no-op. `_ = system.analysis` trips no F841, so use that form and keep the explanation as a plain comment. No compliant form exists for `import x as _`, since the import still trips F401. Suppress with `# noqa` and state the side effect. "Unused" here is the name-liveness verdict of the linter and makes no claim about the behavior of the statement. The verdict and the behavior come apart in this case.

Post your observations, hypotheses, and recommendations in the issue or the pull request.

After investigation, and once the reviewer of the pull request agrees, remove unused code, dead branches, commented-out blocks, and stale imports.

### Suppressing a lint finding: comply, or annotate with a checkable reason

Comply by default. Reach for `# noqa` only when complying would change
behaviour or delete information.

| The finding is… | Do this | Because |
|---|---|---|
| Wrong about this code, a genuine false positive | `# noqa: XXX` + reason | Complying would encode a falsehood. |
| Right, and the compliant form is behaviour-identical | Comply | A suppression here is debt against no offsetting truth. |
| Right, but compliance is unavailable | `# noqa` or `per-file-ignores` + a reason naming the blocker | The blocker is the reason, and blockers expire, so the reason must be checkable. |
| Right, and complying would delete information | `# noqa: XXX` + reason | Column alignment, an assertion, and an intended shape are all information. |
| A real defect nobody is fixing today | Fix the defect, or ignore the finding with an explicit tracked note | A tracked note keeps the defect visible and dated, and a bare suppression hides the defect. |

**The "No cruft" section above governs unused variables, imports, and function arguments, and the section is narrower than this table.** Investigate first. Do not suppress an F401, F841, or ARG finding on the strength of this section alone. Exceptions A and B in that section are the whole of the license.

**Rules about the reason itself.**

1. *State a fact and omit preferences.* A word such as "intentional" asserts only that
   someone made the choice, and a reader can check no part of that claim. A reader can check the reason in `# noqa: B905 - the lists are appended in separate statements in a background thread`, because the reason points at a fact.
2. *Single-source a reason that names a repo-wide fact.* When several sites share one ground, such as a Python floor, a dependency version, or a downstream consumer, write
   the ground once and point the other sites at that statement.
   A change of ground then takes a single edit, and no missed site goes quietly wrong.
   No tool detects a suppression whose reason has gone stale. The rule still fires, so
   the directive is still consumed, and even `RUF100` stays silent.

**Fixes that are not compliance.** Making the finding go away by weakening the assertions of the
code is a behaviour change with the paperwork of a lint fix. A silent truncation shows the case.
`zip(a, b, strict=False)` truncates where `strict=True` raises. Suppress the finding and say why when the
compliant form changes behaviour.

**Scope of the suppression.** Put the suppression at the smallest scope where the reason is
true:

- the reason is about *this line* -> per-site `# noqa: XXX -- reason`
- the reason is about *this file* -> `per-file-ignores` in `pyproject.toml`, with the
  reason as a comment above the entry. The existing Typer/`B008` and
  `rollin_status.py`/`E501` entries are the model.
- the reason is about *the whole project* -> a global `ignore`. No suppression currently
  qualifies, and the bar should stay high.

### Functions have docstrings, type hints, and type checking

Apply this standard to code you write or modify. In touched scripts, apply the standard to existing code as the scripts change, and skip a global retrofit pass.

### Fail-fast

Critical paths raise exceptions and never return `False` or `None` silently on failure.

### Preserve context in exceptions

Exceptions should include file paths, return codes, and log locations for actionable debugging.

### Prefer log-based completion checks over file existence checks

A file may exist but be corrupt, incomplete, or from a previous failed run. File existence checks can mask errors when log checks are available.

- **Exception.** File existence is appropriate for verifying *input* files before reading them.

### Keep system-agnostic software

System-specific information belongs in configuration files that each deployment defines. Avoid hardcoded paths or machine-specific constants in core code.

---

## Repository map

- `docs/contributing.md`: this file, the contributor guide.
- `CONTRIBUTING.md`: a short file at the repository root that points here.
- `architecture.md`: project structure, key modules, and data flow.
- `docs/`: source for the published documentation site, built with MkDocs.

---

## Docstring standard

Public methods and modules use **NumPy-style docstrings**, the `numpydoc` convention of `Parameters`, `Returns`, and `Raises` sections. The documentation site renders the docstrings with the `mkdocstrings` Python handler, configured with `docstring_style: numpy`, so a docstring that follows the convention renders correctly in the API reference with no further annotation.

Contributors adding new public methods include a NumPy-style docstring when writing the method.
