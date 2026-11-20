---
type: operations
title: Development checks, suite campaigns, and fixture caches
description: Check recipes, compile and worktree guards, campaign verdict scope, and safe fixture-cache handling, with explicit limits of this source snapshot.
tags: [testing, development, cache, slurm]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-5fd879e8b1a9ca380f062f2d
    resource: repo://conftest.py
  - id: openwiki-source-193053215eb2d471ac4f675e
    resource: repo://docs/contributing.md
  - id: openwiki-source-6431b9f0656094a245085339
    resource: repo://docs/explanation/testing-the-toolkit.md
  - id: openwiki-source-c59fe4336a371ea1052a01dd
    resource: repo://justfile
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-e4e3b234839bf932f2ec7085
    resource: repo://src/hhemt/_compile_venue.py
  - id: openwiki-source-96397097f3d27f30d5183c33
    resource: repo://src/hhemt/cache/clear.py
  - id: openwiki-source-0b9e3143cb7e518a58b03245
    resource: repo://src/hhemt/cache/keys.py
  - id: openwiki-source-a30422db6b64dbc6256e399b
    resource: repo://src/hhemt/cache/markers.py
  - id: openwiki-source-fa85a49b136ccf9b9e0a40f2
    resource: repo://src/hhemt/suite/_runner.py
  - id: openwiki-source-13a1f1ac9a3fefae37c7147d
    resource: repo://src/hhemt/suite/aggregate.py
  - id: openwiki-source-166ee737d9669c432b59eb8b
    resource: repo://src/hhemt/suite/partition.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Development checks, suite campaigns, and fixture caches

## Snapshot boundary

This checkout contains test directories but no Python files matching `tests/**/*.py`. Root `conftest.py`, pytest settings, recipes, cache code, and campaign implementation remain available. They establish intended procedures, **not evidence of present test coverage or a passing suite**. In particular, fixture-closure marker derivation is described by configuration and documentation, but its `tests/conftest.py` implementation is absent here. Root collection guarding returns early if `tests/fixtures/_compile_guard.py` is absent.

## Environments and focused checks

The package supports Python 3.11 and 3.12 (`>=3.11,<3.13`). `docs/contributing.md` recommends the full conda installation and separate docs tooling. Shared hook commands use `scripts/hook_env.sh`, which the contributor guide describes as binding imports to the invoking checkout rather than installing a `.venv` there. Consult the installation guide before recreating the scientific/SWMM environment; a docs-only environment is not a solver toolchain.

Supported recipes, for a checkout with its test sources restored:

| Recipe | Population / consequence |
| --- | --- |
| `just test path/to/test.py` | Passes a narrow pytest selection through the Python 3.12 hook environment. |
| `just test-fast` | Conda-backed Gate A: excludes slow, compile-tier, and Snakemake-subprocess tests; forbids compile. |
| `just test-no-compile-no-subprocess` | Includes non-slow and slow tests outside those two special tiers; xdist uses all cores with `loadgroup`. |
| `just test-snakemake-subprocess` | Runs subprocess tests outside compile tier serially, avoiding nested schedulers under xdist. |
| `just test-gated` | Unfiltered Gate B with permitted compile venue and required compile tier; writes a JSONL ledger. |
| `just qa` | Formatting, fixing lint/import order, type checking, two structural scripts, then Gate A's marker expression in the hook environment. This recipe mutates formatting/lint findings. |
| `just docs-check` | Runs the same `docs_gates.sh` entrypoint as documented for CI. |

The hook-backed QA environment and conda Gate A use the same marker expression, but dependency differences can affect collection before markers run. Do not equate filtered expressions with measured identical populations. Pytest uses `tests` as its discovery root, retains failed temporary directories, and emits a traceback after 300 seconds without killing the test. The gated conda recipe uses `--no-capture-output` so detached diagnostics are not trapped in conda's buffers.

## Two independent compile declarations

`HHEMT_COMPILE_VENUE=toolchain` permits compile/run capability in test sessions. Unknown or absent tokens relax nothing. `HHEMT_REQUIRE_COMPILE_TIER=1` is a separate anti-silence requirement: compile-tier skips must not count as a gated success. Neither implies the other. `just test-gated` sets both.

The common execution predicate refuses solver compilation/SWMM execution during a pytest phase without a permitted venue. It does not prohibit ordinary non-pytest user runs. The root collection hook additionally arms the fixture compile guard when that fixture module is present.

The root worktree guard prepends this checkout's `src` to `sys.path` and exports it at the front of `PYTHONPATH` for children. An unresolved mismatch exits 99; explicit installed-package allowance warns instead. This protects against accidentally testing another checkout's editable install.

## Campaign evidence and scope

The partitioner groups expensive session-fixture consumers so a shared build is not needlessly repeated across sessions, and rejects assignments that fail to cover the collected universe. The suite implementation persists a manifest of node IDs/chunks/source identity and consumes chunk status sidecars plus JUnit XML.

Chunk results are not interchangeable with individual assertions:

- `MISSING`: a declared chunk has no status.
- `VOID`: preconditions or execution provenance failed, such as missing expected session-fixture setup, source drift/dirty tree, inconsistent import resolution, interruption, scheduler kill, or absent JUnit.
- `FAIL`: completed without a voiding cause but returned nonzero.
- `PASS`: completed without a voiding cause and returned zero.

Aggregation separately tracks failed, unevaluated, absent, structurally skipped, and incidentally skipped tests. Incidental skips are not automatically benign: their messages remain evidence for interpretation.

`scope=array` covers the scheduler-evaluable set only. `scope=union` requires complement evidence; absent or unattempted declared complement nodes force scope back to array. Triage reruns prior failed/unevaluated tests and is structurally bound to `scope=triage` with distinct summary output; tier-restricted plans are bound to `scope=tier`. Neither supports a whole-suite green claim.

## Fixture identity and completion

The cache marker `_status/_fixture_complete.json` attests a fixture's required payload, not merely a workflow stage flag. Its state is `absent`, `stale`, `unsatisfied`, or `complete`: missing/unreadable records rebuild; changed provenance is stale; missing payload or a supplied live digest mismatch is unsatisfied; only complete is reusable. Live payload digest checking is optional at the API boundary, while existence checking still runs. `finish_build` refuses missing payload and writes the marker last with measured payload identity.

Builder provenance includes builder sources and a digest over sorted relative paths and bytes of every Python source beneath the checkout's `src/hhemt`. Uncommitted edits and renames therefore change identity; a commit without source changes does not. Solver, configuration, environment, case, and payload terms are supplied by the caller. The existing guide warns that stale/missing-cache rebuilds can be silent, so distinguish cold construction from warm reuse when interpreting observations.

## Clearing caches safely

`hhemt cache clear` separates model artifacts (T0), analysis trees at the current checkout address (T1), compiled builds (T2), and the canonical source checkout (T3). T0/T1 are in scope by default; T2 requires a permitted venue or `--include-compile-tier`; T3 requires `--include-source-clone`. T1 deletion walks children and excludes `_software`, preserving the independent T2 decision.

A compile-tier refusal is **partial apply**, not an all-or-nothing refusal: cheap tiers may be removed while compiled builds remain, and the CLI owes a nonzero gate result. Held per-build compile locks and the canonical provision lock cause busy T2/T3 roots to be skipped. T0/T1 have no liveness lock checks in this module; do not interpret the clear planner as protection for actively used analysis/model trees.

## Neighbors

- [Durable state](../architecture/durable-state.md)
- [Solver builds and containers](../integrations/solver-builds-and-containers.md)
- [HPC execution](hpc-execution.md)
- [Layout migrations](layout-migrations.md)

No test, compile, solver, or scheduler campaign was executed for this page.
