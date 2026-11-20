---
type: workflow
title: Reprocessing, invalidation, and data retention
description: Choose a rebuild floor, understand mtime and log invalidation, and preserve recovery guarantees when raw or per-scenario data is removed.
tags: [reprocessing, retention, recovery]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-4cbe2a3b27f6672d80e091b7
    resource: repo://docs/how-to/forcing-reruns.md
  - id: openwiki-source-21b9d8cf4c93042742d6c138
    resource: repo://src/hhemt/analysis.py
  - id: openwiki-source-307087f14bd0c7a3cde6ea8d
    resource: repo://src/hhemt/processing_analysis.py
  - id: openwiki-source-a192b2a3fdd9f24a6b6f55d7
    resource: repo://src/hhemt/recompute.py
  - id: openwiki-source-0b4adb1ce413cf639f14defe
    resource: repo://src/hhemt/scenario_records.py
  - id: openwiki-source-3484b6903eff31449250762b
    resource: repo://src/hhemt/wipe_guard.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Reprocessing, invalidation, and data retention

## Choose the right reentry

`hhemt reprocess` never simulates. Its `--start-with` selects `process`, `consolidate`, or `render`; `--regenerate-existing` requests rebuilding existing processed/consolidated outputs rather than simply reusing them. For example:

```bash
hhemt reprocess --system-config my_system.yaml --analysis-config my_analysis.yaml \
  --start-with process --regenerate-existing
```

Render-only reentry leaves figures in place. Earlier stages also invalidate regenerable plots. Preserve EDA figures that have no workflow rule to regenerate them; not every file under `plots/eda/` is necessarily outside the report DAG. A dry run is not filesystem-read-only: flag/report-shell invalidation can occur to expose the prospective DAG, while destructive store, figure, and disk-accounting changes are skipped.

Reprocess uses `Snakefile.reprocess`, `--rerun-triggers mtime`, and `--nolock`. Driver sentinels under `_status/_orchestrator` are the concurrency authority rather than Snakemake's ordinary lock. Queued simulations alone need not block reprocessing, but process regeneration refuses submitted-work sentinels that could represent concurrent output writers. Heavy deletion can route through SLURM; processing-log invalidation still happens in the caller because deleting files without clearing the writer's completion records would skip rebuilding.

## Force a stage and everything downstream

`--override-force-rerun` overrides the configured value for one invocation. Short forms select all, none, event indices, or sensitivity member IDs and default to the `simulate` floor. The explicit form separates subject and stage:

```bash
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml \
  --override-force-rerun '{"subject": "all", "stage": "render"}'
```

The floors are `simulate`, `process`, `consolidate`, and `render`. A render floor is global, not subject-scoped, because aggregate figures span scenarios. Invalidation runs before DAG planning so missing flags/artifacts trigger mtime-driven work. At simulation/process floors, it also clears per-model processing-log output records and chapter sets; flag deletion alone would allow writers to consider stale outputs complete.

Build-stamp reconciliation promotes stale figure builds to a render floor. Report-setting reconciliation compares the recorded settings hash with this invocation's report configuration and promotes a mismatch to global render unless an earlier, stronger floor is already forced. It does nothing without a record or on dry run. These mechanisms repair render reuse; they do not imply automatic rerunning of solver numerics.

## Retention changes recovery options

The consolidated DataTree is canonical, not a disposable convenience copy. Reuse requires presence, a complete log, matching input fingerprint, and matching consolidation build stamp. Per-scenario summary stores can be cleared after every corresponding consolidated slice has been read back equal. A mismatch raises and retains the source stores. Clearing records progress through `pending` to `done`; a resumed pending deletion finishes from verified consolidated state rather than reading partly deleted sources.

Whole scenario-directory removal has stricter conditions: clearing is enabled, `clear_raw` covers every enabled model, every removable artifact class is selected, models completed, their summary/series stores are recorded cleared, and no processed store remains. Failed or incomplete scenarios retain diagnostic material.

Removal writes an atomic `pending` scenario record before deletion, deletes with disk accounting, then marks it `done`. Pending deletion resumes without using the remaining directory as data. Records preserve model logs, disk information, listings, completion verdicts, and status rows. A simulation-stage force reopens a removed event by invalidating flags before changing its record to `superseded`. Removed sensitivity members similarly retain records/config/log information under the master and are exposed as read-only views rather than reconstructed writable analyses.

Once raw output has gone, a scenario-processing correction cannot be recovered by reprocessing alone. The recompute resolver explicitly maps simulation defects to rerun, scenario-processing defects to reprocess only if raw data remains (otherwise rerun), consolidation defects to reconsolidation, and cosmetic defects to rendering. Its scope is explicit, not inferred from changed source filenames. Check available rebuild inputs before discarding the only remaining consolidated copy.

## Resume is not a fresh wipe

A fresh run's wipe guard counts completed run flags, consolidated trees, queued/submitted records, and orchestrator sentinels. Existing work causes refusal unless the deliberate-wipe override is provided, in which case the cost is printed. This is separate from deletion-command confirmation/in-flight guards. To add scenarios, resume; do not use fresh mode as an invalidation shortcut.

## Related pages

- [Durable state](../architecture/durable-state.md)
- [Output data model](../concepts/output-data-model.md)
- [Diagnostics and in-flight control](../operations/diagnostics-and-in-flight-control.md)
- [Reporting and portable bundles](reporting-and-portable-bundles.md)

No reprocessing, deletion, solver execution, or tests were run for this documentation.
