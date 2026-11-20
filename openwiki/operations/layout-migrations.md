---
type: operations
title: Persisted layout versions and migrations
description: How HHEMT identifies persisted layouts, plans forward migrations, records progress, and separates layout migration from render-bundle compatibility.
tags: [migration, persistence, operations, compatibility]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-21b9d8cf4c93042742d6c138
    resource: repo://src/hhemt/analysis.py
  - id: openwiki-source-1e6bf8fb0554c9f030b787ad
    resource: repo://src/hhemt/bundle/__init__.py
  - id: openwiki-source-faeebb5f7c329d37d48cc997
    resource: repo://src/hhemt/bundle/_combine.py
  - id: openwiki-source-2308e05998320ea9be66f713
    resource: repo://src/hhemt/version_migration/__main__.py
  - id: openwiki-source-5625a6ae130efaad51f81621
    resource: repo://src/hhemt/version_migration/constants.py
  - id: openwiki-source-d23d68e541de179e2cc0e56e
    resource: repo://src/hhemt/version_migration/context.py
  - id: openwiki-source-103a6a79e350e163478264c7
    resource: repo://src/hhemt/version_migration/registry.py
  - id: openwiki-source-eb8509ccf04b604568ad78ec
    resource: repo://src/hhemt/version_migration/runner.py
  - id: openwiki-source-5ebd876dc43bf2b09c1c0565
    resource: repo://src/hhemt/version_migration/state.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Persisted layout versions and migrations

## Version ownership and first writes

`LAYOUT_VERSION` is the persisted analysis/system layout contract (currently 34), not the installed package version. Render bundles have separate contracts: member `BUNDLE_SCHEMA_VERSION` is 7 and combined-bundle schema is 8. Do not baseline an analysis to repair a bundle-reader mismatch.

Analysis execution and submission paths call `stamp_new_target` on first materialization; static-plot submission and the cleanup path also gate their work with a stamp. Stamping is not automatic migration:

- A matching record is returned unchanged, preserving history.
- Fresh mode refuses a different existing version and refuses an unstamped tree containing recognized migratable artifacts.
- Construction mode warns and preserves a differing record, but can stamp an absent record without inspecting content.
- Established mode permits content because the migration runner has already inferred its version.

Malformed `_version.json` raises `VersionFileUnreadableError`; it is not treated as an absent record.

## Determine the starting version

The inference ladder first trusts the target's own record, then positive legacy iloc-prefixed simulation-directory evidence (version 0). A nested `members/` or legacy `subanalyses/` target can inherit its primary's record only after that local legacy check. A unified `experiment_datatree.zarr` without a record establishes the lower bound 21, not the current toolkit version. Flat per-mode stores without an analysis tree imply version 1; an analysis tree uses its own metadata heuristic. No recognized signal raises `BaselineRequiredError` in the runner.

Baseline is an explicit assertion, not a conversion. It accepts versions within the supported range, refuses to change an existing record without `--force`, and clears migration history when force-baselining downward.

## Operator sequence

Run these in an environment where HHEMT and its migration dependencies are installed, substituting the actual target directory:

```sh
python -m hhemt.version_migration status /path/to/analysis
python -m hhemt.version_migration migrate /path/to/analysis
python -m hhemt.version_migration migrate /path/to/analysis --apply
python -m hhemt.version_migration verify /path/to/analysis
```

`migrate` defaults to dry-run. Migrations requiring scenario slug reconstruction, notably V0001, need **both** `--system-config` and `--analysis-config`; the CLI only constructs the configuration mapping when both are supplied. The slug-reconstruction helper suppresses constructor log updates and, during dry-run, layout-state materialization.

If the layout cannot be inferred, inspect its artifacts before explicitly asserting a version:

```sh
python -m hhemt.version_migration baseline /path/to/analysis N
```

`verify` only checks that `_version.json` equals current `LAYOUT_VERSION`. It does **not** walk the tree or prove data/schema integrity.

## Chain, commit boundaries, and failure

Migration modules declare consecutive `version_from`/`version_to` edges and an `upgrade(ctx)` callable. Planning rejects downgrades, duplicate edges, and missing edges in the requested slice. Full registry validation separately checks the complete supported chain.

The runner builds a context, calls `upgrade` to accumulate operations, prints the plan, and executes it only under `--apply`. An inferred starting version is stamped before the first applied migration. Each successful migration records its new layout version and a timestamped history entry only after context execution returns.

This is a **per-migration progress record, not a whole-run rollback transaction**: operations execute sequentially, and the runner has no global rollback around them. A failure can leave partial filesystem changes before the version is advanced. Inspect the failed operation and its migration-specific idempotency/conflict behavior before retrying; do not manually advance the version to skip unfinished work. Selected filesystem/argument errors during planning are wrapped as `MigrationConflictError`; programming errors such as `AttributeError` are deliberately allowed to propagate.

Version-file writes use a temporary file plus `os.replace`. Writes and history read-modify-write use the version-file lock; history updates also require the on-disk version to match the expected predecessor. These locks protect version records, not an exclusive lock around the complete migration chain. JSON-log primitives use their own log locks. The Snakemake metadata-clearing primitive renames metadata to a labeled backup and preserves an existing backup on retries.

## Bundle compatibility is separate

Member and combined bundle readers require exact matches against their respective local schema constants. A newer member bundle requires upgrading the toolkit; an older one must be re-emitted from the source analysis. Combined bundles use their own schema guard and must be re-combined when incompatible. Analysis layout migration does not rewrite those reader contracts.

## Related workflows

- [Durable state](../architecture/durable-state.md): state owners outside migration records.
- [Output data model](../concepts/output-data-model.md): stores affected by layout evolution.
- [Experiment reproduction](../workflows/experiment-reproduction.md): portable bundle lifecycle.
- [Testing and development](testing-and-development.md): development and validation recipes.

This page is based on source inspection; no migration or solver execution was performed.
