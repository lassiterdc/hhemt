---
type: concept
title: Processed stores and consolidated DataTree
description: Traces solver outputs through scenario datasets into analysis and experiment trees, including restartable publication, metadata, selection, and retained canonical copies.
tags: [outputs, datatree, zarr, netcdf, provenance]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-50e34414937b57290a25c934
    resource: repo://docs/reference/output-data-model.md
  - id: openwiki-source-21b9d8cf4c93042742d6c138
    resource: repo://src/hhemt/analysis.py
  - id: openwiki-source-5b4f508510e1ffb89526853a
    resource: repo://src/hhemt/cf_conventions.py
  - id: openwiki-source-51e3de6ce2768ffa36863efa
    resource: repo://src/hhemt/process_simulation.py
  - id: openwiki-source-307087f14bd0c7a3cde6ea8d
    resource: repo://src/hhemt/processing_analysis.py
  - id: openwiki-source-6f5ee698913e6f2bb99939d1
    resource: repo://src/hhemt/sensitivity_analysis.py
  - id: openwiki-source-babcd90b587c9eb21aef3222
    resource: repo://src/hhemt/swmm_output_parser.py
  - id: openwiki-source-ac6a13e3578aa4f8c67a15aa
    resource: repo://src/hhemt/timeseries_selection.py
  - id: openwiki-source-62130579e62e37fb95f361a6
    resource: repo://src/hhemt/utils.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Processed stores and consolidated DataTree

## Data moves through distinct tiers

```text
TRITON grids/timers + SWMM output/report + optional diagnostic files
  → scenario processed datasets (timeseries, summaries)
  → analysis_datatree.zarr (event-indexed model/artifact branches)
  → experiment_datatree.zarr (sensitivity member branches + parameters)
```

Per-scenario artifacts live under `sims/{event_id}/processed/` and are flat xarray datasets. `target_processed_output_type` selects Zarr or NetCDF for normal model outputs; hydrology capture summaries remain Zarr. Consolidated tiers are hierarchical DataTrees in Zarr. A sensitivity member owns an analysis tree under its member directory; the sensitivity primary owns the experiment tree rather than a second analysis tree.

The system tier, `system_datatree.zarr`, is a separate explicit-call aggregation, not a normal workflow stage. Do not assume it exists after an analysis run.

## Processing boundary

The processing runner produces time series before summaries. The SWMM parser combines binary-output series and report tables; these are different measurements, not interchangeable sources. For example, generic signed series maxima need not equal a report's maximum magnitude. Coupling flux and diagnostic decoding have their own contracts in [coupling diagnostics](coupling-diagnostics.md).

`_write_output()` applies mode-specific CF attributes, attaches producing SHA/version coordinates when `event_iloc` is present, records simulation/output timestamps, and chooses the configured output format. Units and quantity descriptions belong to `cf_conventions.py`, not to variable-name suffixes or a renderer's guesses.

## Navigating an analysis tree

Summary groups follow `{family}/{kind}/summary`:

| Family | Kinds |
| --- | --- |
| `tritonswmm` | `triton`, `swmm_node`, `swmm_link`, `performance` |
| `triton_only` | `triton`, `performance` |
| `swmm_only` | `swmm_node`, `swmm_link`, `system` |

Summaries combine scenarios along `event_iloc`; spatial results retain `x/y`, `node_id`, or `link_id` as appropriate. Standalone system reductions share a scenario source store with node reductions but are selected into a separate group, avoiding duplicate variable placement.

Corresponding `timeseries` branches are appended scenario by scenario. Current source also consolidates per-rank performance series; do not rely on the older reference-page sentence saying performance has no time-series branch. Time series use `elapsed_s` relative to event start, with axes established from configured event windows rather than whichever source files happen to remain. SWMM can use a different report interval; observations have their own axis. Unavailable values after reindexing are not additional measurements.

Prefer the owned reader after consolidation:

```python
tree = analysis.process.open_datatree()
```

It refreshes the log, refuses an unset completion verdict, and delegates to `open_store(..., chunks="auto")`. Current publication rebuilds a consolidated-metadata index and checks its currency. The older reference's blanket statement that stores have no consolidated metadata does not describe this writer.

## Completion and crash recovery

A gridded TRITON series is built from independently flagged chapter stores. Resume uses covered timestep coordinates, not a count of chapters. Before merging, the Zarr publisher requires chapter indexes contiguous from zero; it refuses a missing interior chapter instead of publishing a gapped result. Publication occurs through a temporary store and rename; the `{artifact}.done` marker follows completed publication and authorizes chapter cleanup. A leftover chapter directory after that marker is not by itself incomplete data. The NetCDF arm similarly publishes one file, flags it, then removes chapters.

Analysis consolidation is deliberately **not** a whole-store replacement: append-only series may already be the only remaining copy. It removes the root consolidation stamp, updates summaries, appends series, verifies and clears elected scenario stores, then writes root attributes last. Deferred metadata consolidation rebuilds the index after the write block. An interrupted write lacks current publication evidence; mere directory presence is insufficient. Reuse also depends on log completion, input fingerprint, and producing-build compatibility.

Once scenario summaries have been cleared, their pending/done clearing record identifies the canonical groups and variables. Cleared events cannot simply disappear from the analysis: consolidation refuses a rewrite that would lose their only stored slice. See [reprocessing and retention](../workflows/reprocessing-and-retention.md).

## Selection is downstream of statistics

`timeseries_selection.py` defines a closed selectable domain by kind and model family. `None`, blank, empty list, and `all` keep the full domain; `default` is family-specific, and explicit lists are checked against the families' union then intersected with each family's domain.

Pruning runs after summaries, records removed names before deleting arrays, and never prunes static variables or unrelated dimensions such as observation-point series. Rebuilding summaries from a pruned source is refused rather than silently dropping statistics.

## Experiment trees and heterogeneous inputs

The experiment `parameters` dataset describes every defined member; only completed members have result nodes. `allow_incomplete=True` skips incomplete members as a whole, with diagnostics, rather than assembling partial event subsets under a misleading complete member. Member summaries and series are copied/read back before clearing can remove the source tree; pending/done movement records preserve recovery authority.

Across scenario joins, differing variable sets are recorded as an `event_iloc` coordinate, `variable_set_across_members`. Missing variables are NaN-filled, so cross-event statistics may cover only a subset. Dimensioned evidence survives attribute conflict dropping. Solver-build causality is attributed only to performance modes, not to every heterogeneous variable set.

## Metadata and provenance ownership

`apply_cf_attributes()` resolves mode overrides, base entries, water-quality entries, and scoped EDA entries. Unknown variables get a humanized fallback label, not fabricated units or a standard name. Mixed pollutant units require the per-pollutant unit coordinate and an explanatory comment rather than one false scalar unit.

Consolidation derives advertised variables and descriptors from the assembled datasets, embeds a deterministic RO-Crate core as `ro_crate_metadata`, and writes the co-located sidecar separately. Per-event producing coordinates remain authoritative; a uniform root producing stamp is only a fast path. See [provenance and publication](../integrations/provenance-and-publication.md) and [reporting and portable bundles](../workflows/reporting-and-portable-bundles.md).

This page is source-grounded; no solver, output-file round trip, scheduler query, or test suite was run during authoring. Stage entrypoints are covered by [ensemble lifecycle](../workflows/ensemble-lifecycle.md).
