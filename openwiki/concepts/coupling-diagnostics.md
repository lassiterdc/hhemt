---
type: concept
title: Coupled exchange and hydraulic diagnostics
description: Describes ordered coupling nodes, signed exchange decoding, interval means, optional diagnostic captures, and solver-used manhole metadata.
tags: [coupling, hydraulics, diagnostics, outputs]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-97f7ba4552e0858d5bfe3a6f
    resource: repo://docs/how-to/per-node-manhole-values.md
  - id: openwiki-source-61d64405d5956ae8ea9a2d7c
    resource: repo://src/hhemt/coupling_diagnostics.py
  - id: openwiki-source-fc5d469d05cd2e1bd2f0012b
    resource: repo://src/hhemt/coupling_exchange.py
  - id: openwiki-source-51e3de6ce2768ffa36863efa
    resource: repo://src/hhemt/process_simulation.py
  - id: openwiki-source-307087f14bd0c7a3cde6ea8d
    resource: repo://src/hhemt/processing_analysis.py
  - id: openwiki-source-92b126da4e4928a37957abfc
    resource: repo://src/hhemt/scenario.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Coupled exchange and hydraulic diagnostics

## Identity precedes decoding

Scenario preparation selects coupling nodes and writes them in SWMM model node order. Selection refuses candidates that cannot be placed from their node definition and coordinates. The one-node-per-cell reduction applies only to `legacy` coupling mode; declared `list` and `all_junctions` modes retain their declared coupled set. The exchange binary's slot order is the order of `coupling_nodes.txt`, not the order a plotting consumer happens to request.

See [scenario inputs and SWMM](../integrations/scenario-inputs-and-swmm.md) for preparation and node eligibility.

## Isolated exchange versus SWMM inflow

`coupling_exchange.py` decodes `*_exchange_replay.bin`, the per-surface-step array of exchange flows handed to SWMM. This is the isolated coupling term; SWMM node inflow also includes other sources and is not a substitute.

The accepted binary has a 12-byte little-endian header: magic `SWM2`, node count, and value width (4 or 8 bytes). Each record carries `dt` followed by one flow per coupling node. The decoder rejects the older widthless `SWMM` format rather than guessing precision, and refuses mismatched/duplicate node lists, zero records, and partial trailing records.

On disk, exchange is in cfs. Conversion uses `0.028316847`, preserving the signed convention:

- positive: surface → sewer, drainage;
- negative: sewer → surface, surcharge.

Statistics stream records in blocks, integrating directional and signed volumes with `q * dt`. They retain directional peaks, first peak times, and surcharge duration. Peak times are returned in minutes; surcharge duration is seconds.

## Reporting intervals are volume-weighted

`to_reporting_intervals()` distributes a step across every reporting interval it overlaps, accumulates flow × overlap duration, then divides by interval length. It is not instantaneous sampling. Times must be finite, increasing, and evenly spaced; a single report time requires an explicit interval length. Invalid time axes or insufficient final coverage raise `ProcessingError`.

**Numerical boundary:** final coverage accepts a shortfall of up to `1e-3` of a reporting interval. The source explicitly calls this a guessed floating-point coverage tolerance, not an empirically established solver-error bound. Fractional timestep accumulation can also vary at rounding level with block partitioning. Do not turn these implementation tolerances into physical accuracy guarantees.

Processing maps the decoded series into `coupling_exchange_flow(date_time, node_id)` and eight per-node statistics. Noncoupled nodes receive NaN, while `is_coupling_node` explicitly distinguishes membership. Exactly one exchange replay file is required when flux output is enabled; missing or ambiguous input refuses processing rather than silently omitting the variable.

## Optional hydraulic captures

`coupling_diagnostics.py` decodes SWMM solve-step, path-3 node, link-flow, surface edge-flux, timestep-limiter, domain-balance, and wet-cell captures. Binary readers require the file-specific magic, version 1, eight-byte values, and the appropriate item count; node/link sidecars fix column identity. Text readers require the exact percent-header column names.

SWMM capture volumes/flows are converted from ft³/cfs to SI on read. The lazy per-step path-3 view is an exception: it retains ft³ and carries the conversion factor as metadata. Surface captures are already SI.

Requested captures are written to `processed/coupling_diagnostics/{name}.{type}` and consolidated under `tritonswmm/coupling_diagnostics/{name}` along the event axis. An existing store recorded as written is reused, including after raw reclamation. A requested but missing raw file/sidecar raises; the raw-only per-step path-3 capture is validated and closed, not ingested into a processed store. This separates diagnostic capture from ordinary report-interval node outputs.

## Manhole values are provenance, not a guessed default

A configured per-node manhole CSV needs finite positive diameters and finite nonnegative loss coefficients. Missing coupling-node rows refuse by default; `manhole_table_fill_with_constants: true` fills missing nodes but does not excuse unknown names or malformed rows. Preparation writes a complete `manhole_table.txt` in coupling-node order.

Processing reads that **written** table and attaches `manhole_diameter` and `manhole_loss_coefficient` on `node_id`, with NaN at uncoupled nodes. It does not infer what an earlier run used from the current constants. Native builds must read `swmm_manhole_table`; the current capability guard refuses container mode because its admitted image set for this key is empty.

## Downstream and retention boundary

Exchange and path-3 time-dependent node variables are excluded from generic node summary reduction; their time series remain available through the time-series consolidation branch. Diagnostic captures have separate DataTree groups. Consumers should therefore choose the correct branch rather than deriving an instantaneous interpretation from an interval mean. See [output data model](output-data-model.md).

Replay reclamation is gated so enabled exchange processing can read the sidefile before the coupled node store is written. See [reprocessing and retention](../workflows/reprocessing-and-retention.md) before deleting raw captures, and [diagnostics](../operations/diagnostics-and-in-flight-control.md) for operations. No solver, remote service, or tests were exercised for this page.
