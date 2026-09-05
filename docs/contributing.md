<!-- hhemt:personal-voice -->
# Contributing

## Licensing of contributions

This project is currently released under the
[PolyForm Noncommercial License 1.0.0](https://github.com/lassiterdc/hhemt/blob/main/LICENSE).

By submitting a contribution (a pull request, patch, or any other work)
to this project, you agree that:

1. **Inbound = outbound.** Your contribution is provided under the same
   license as the project (PolyForm Noncommercial License 1.0.0), and
   you have the right to submit it under that license.

2. **Forward relicensing.** You grant the maintainer permission to
   license your contribution under any future license the project
   adopts, including a permissive OSI-approved open-source license
   (for example, Apache License 2.0). This lets the project relicense
   in the future without requiring sign-off from every past contributor.

If you cannot agree to these terms for a particular contribution, please
open an issue to discuss before submitting.

## Development setup

1. Fork and clone the repository.
2. Create the supported conda environment. See the "Installation" section of
   `README.md` for the four commands and for why both `--no-deps` flags are
   required. A plain `pip install -e .` inside a conda environment displaces
   conda-resolved packages such as `numpy` and `pandas`.
3. To build the documentation, install its tooling by name. `environment.yaml`
   does not ship it, and adding `--no-deps` to the `[docs]` extra would install
   none of it:
   `pip install mkdocs mkdocs-material "mkdocstrings[python]" mkdocs-htmlproofer-plugin`
4. Install `uv` (https://docs.astral.sh/uv/). It is a hard prerequisite, not a
   convenience: every pre-commit hook in this repo runs through `uv run --locked`,
   so `git commit` fails without it. `uv` builds and manages its own project
   environment in `.venv/`, separate from the conda environment step 2 creates —
   both will exist on your machine, and the hooks always use `.venv/`. If you work
   inside an activated virtualenv (this does not apply to conda), `uv` prints a
   `VIRTUAL_ENV ... will be ignored` warning on each hook run; that is expected.
5. Install pre-commit hooks: `pre-commit install`. The `pre-commit` package ships
   only with the conda environment (`environment.yaml`); it is declared in no
   `pyproject.toml` extra, so if you installed via Option B of
   `docs/how-to/installation.md` you must `pip install pre-commit` first.

## Workflow

- Create a feature branch from `develop`
- Make changes with tests
- Run `just qa` before opening a pull request. It formats, lints, type-checks, runs
  the two guard scripts listed in that recipe, and runs the test suite. **Whether it
  runs the compile-dependent tests depends on your PATH, not on the recipe**: they skip
  when `cmake` or `mpic++` is absent, which is the usual case under the uv path `just qa`
  uses, and they run when both are present. So a green `just qa` does not by itself mean
  the compile tier passed. To gate it either way, run `just test-gated`, which invokes
  pytest under the conda env with `HHEMT_REQUIRE_COMPILE_TIER=1` so a compile-tier skip
  becomes a hard failure. `just` is
  installed by neither `environment.yaml` nor any `pyproject.toml` extra — install it
  separately (https://github.com/casey/just), or run the commands under the `qa:`
  recipe in `justfile` yourself, in the order they appear there.
- Submit a pull request

## Documentation

Build docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Branching and releases

`develop` is the default branch; all work branches from and merges back into `develop`. `main` is release-only and advances only via a validated `develop` → `main` release pull request (all tests green, docs complete), tagged `vX.Y.Z`. See [Branching and release model](explanation/branching-and-release-model.md) for the two branches, how to branch as a contributor, what counts as a release, and the two independent version axes.

---

## Development Principles

### Raise questions rather than make assumptions
When you encounter uncertainty or discrepancies (especially when implementing a pre-written plan that may have stale components), err on the side of caution and ask the developer how to proceed.

### Plan, then implement
Follow a plan-then-implement strategy. If implementing a plan uncovers a need to change it or its success criteria (including deviations from the planned approach, scope changes, or new risks), raise the discrepancy before continuing rather than adapting silently.

### Let's do things right, even if it takes more effort
- Always be on the lookout for better ways of achieving development goals and raise these ideas
- Raise concerns when you suspect the developer is making design decisions that diverge from best practices
- Look for opportunities to make code more efficient (vectorize operations, avoid loops with pandas, etc.)

### Backward compatibility is NOT a priority

**Rationale**: Single developer codebase. Clean code matters more than preserved APIs. Git history is the safety net.

When refactoring:
- ❌ Don't add deprecation warnings
- ❌ Don't keep old APIs "for compatibility"
- ❌ Don't create compatibility shims or aliases
- ✅ Do update all usage sites immediately
- ✅ Do delete obsolete code completely

### Most function arguments should not have defaults

Default function arguments can lead to difficult-to-debug unexpected behavior. Avoid default values unless a default is almost always the correct choice (e.g., `verbose=True`). This is especially true for configuration fields that users populate. The user should make an intentional choice about every input.

### Avoid aliases

Do not create aliases for functions, classes, or variables. An alias is a second name for the same thing. It creates confusion about which name is authoritative and is a form of backward-compatibility shim. If something needs renaming, rename it and update all call sites.

### No cruft/all variables, imports, and function arguments must be used

Unused elements are a signal that implementation may be incomplete. Treat them as an investigation trigger, not just lint to suppress.

If you come across an unused variable, import, or function argument, investigate before removing:
1. Check whether the surrounding implementation is incomplete
2. Find planning documents that touched that function and determine whether implementation is planned
3. If still uncertain, raise the concern with the developer with hypotheses about why it exists
4. Exception A: elements included for a currently-planned implementation, marked with a comment referencing the planning document
5. Exception B: an element whose *evaluation* is the point, where the name is unread but the statement is load-bearing — an import whose side effect is registration (`import rioxarray` registers the `.rio` accessor), or a bind whose only job is to trigger a property that raises. Prefer the compliant form when one exists and is a no-op: `_ = system.analysis` does not trip F841, so use it and keep the explanation as a plain comment. When no compliant form exists — `import x as _` still trips F401 — suppress with `# noqa` and say what the side effect is. "Unused" here is the linter's name-liveness verdict, not a claim that the statement does nothing, and the two come apart exactly here.

Report your observations, hypotheses, and recommendations to the developer.

After investigation and with approval from the developer, remove unused code, dead branches, commented-out blocks, and stale imports.

### Suppressing a lint finding: comply, or annotate with a checkable reason

Comply by default. Reach for `# noqa` when — and only when — complying would change
behaviour or delete information.

| The finding is… | Do this | Because |
|---|---|---|
| Wrong about this code (a genuine false positive) | `# noqa: XXX` + reason | Complying would encode a falsehood. |
| Right, and the compliant form is behaviour-identical | Comply | A suppression here is debt against no offsetting truth. |
| Right, but compliance is genuinely unavailable | `# noqa` or `per-file-ignores` + a reason naming the blocker | The blocker is the reason — and blockers expire, so it must be checkable. |
| Right, and complying would delete information | `# noqa: XXX` + reason | Column alignment, an assertion, and a deliberate shape are all information. |
| A real defect nobody is fixing today | Fix it, or ignore it with an explicit tracked note | See the `docs/*.ipynb` F403 block in `pyproject.toml`. |

**Unused variables, imports and function arguments are governed by "No cruft" above,
which is narrower than this table.** Investigate first; do not suppress an F401, F841
or ARG finding on the strength of this section alone. Exceptions A and B there are the
whole of the license.

**Two rules about the reason itself.**

1. *State a fact, not a preference.* "deliberate" and "intentional" assert only that
   someone chose it; a reader cannot check them. `# noqa: B905 - the two lists are
   appended in separate statements in a background thread` is a reason, because a
   reader can go and look.
2. *Single-source a reason that names a repo-wide fact.* If three sites are suppressed
   for one ground — a Python floor, a dependency version, a downstream consumer — write
   the ground once and point the other sites at it. When the ground changes, that is one
   edit rather than three, and the two that were missed cannot become quietly wrong.
   Nothing detects a suppression whose reason has gone stale: the rule still fires, so
   the directive is still consumed, so even `RUF100` stays silent.

**What never counts as complying.** Making the finding go away by weakening what the
code asserts is not compliance, it is a behaviour change with a lint fix's paperwork.
`zip(a, b, strict=False)` silently truncates where `strict=True` would raise. If the
compliant form is not a no-op, suppress and say why.

**Where the suppression lives** — put it at the smallest scope at which the reason is
true:

- the reason is about *this line* -> per-site `# noqa: XXX -- reason`;
- the reason is about *this file* -> `per-file-ignores` in `pyproject.toml`, with the
  reason as a comment above the entry (the existing Typer/`B008` and
  `rollin_status.py`/`E501` entries are the model);
- the reason is about *the whole project* -> a global `ignore`. Nothing currently
  qualifies, and the bar should stay high.

### Functions have docstrings, type hints, and type checking

Apply this standard to code you write or modify. For existing code in touched scripts, apply organically (accumulate adherence naturally as scripts are touched rather than doing a global retrofit pass).

### Fail-fast

Critical paths must raise exceptions; never silently return `False` or `None` on failure.

### Preserve context in exceptions

Exceptions should include file paths, return codes, and log locations for actionable debugging.

### Prefer log-based completion checks over file existence checks

A file may exist but be corrupt, incomplete, or from a previous failed run. File existence checks can mask errors when log checks are available.

- **Exception**: File existence is appropriate for verifying *input* files before reading them.

### Keep system-agnostic software

System-specific information belongs in user-defined configuration files. Avoid hardcoded paths or machine-specific constants in core code.

### Track project-agnostic utility candidates

When writing utility functions that could plausibly belong in a shared library (e.g., general-purpose file I/O helpers, generic array operations), note them in a dedicated tracking document. Do not extract them immediately. Track them so they can be evaluated together.

---

## Repository map

- `docs/contributing.md`: this file, the contributor guide.
- `CONTRIBUTING.md`: a short stub at the repository root that points here.
- `architecture.md`: project structure, key modules, and data flow.
- `docs/`: source for the published documentation site, built with MkDocs.

---

## Docstring standard

Public methods and modules use **NumPy-style docstrings** (the `numpydoc`
`Parameters`/`Returns`/`Raises` section convention). The documentation site
(`mkdocs build`) renders these via the `mkdocstrings` Python handler configured
with `docstring_style: numpy`, so a docstring that follows this convention is
rendered correctly in the API reference with no further annotation.

Backfilling docstrings onto currently-undocumented public methods is tracked
separately (it depends on the public-API surface designation); contributors
adding NEW public methods should include a NumPy-style docstring at authoring time.
