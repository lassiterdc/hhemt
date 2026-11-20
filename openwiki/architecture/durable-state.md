---
type: architecture
title: Durable status, liveness, and recovery
description: Distinguishes persisted logs, dependency flags, worker markers, orchestration claims, and records retained after removal.
tags: [state, recovery, orchestration, slurm]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-b6387de1609803d932d36116
    resource: repo://src/hhemt/log.py
  - id: openwiki-source-cbb6b02e668804cc97a4e3d1
    resource: repo://src/hhemt/run_simulation_runner.py
  - id: openwiki-source-0b4adb1ce413cf639f14defe
    resource: repo://src/hhemt/scenario_records.py
  - id: openwiki-source-25995bd0e5d97754048ca5f4
    resource: repo://src/hhemt/slurm_liveness.py
  - id: openwiki-source-1c065800b97a0edd46afbc16
    resource: repo://src/hhemt/status_flags.py
  - id: openwiki-source-704d4ea412eaff2fb1236809
    resource: repo://src/hhemt/workflow.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Durable status, liveness, and recovery

## Different artifacts answer different questions

| Artifact | Question it answers |
| --- | --- |
| Model, scenario, analysis, and system JSON logs | What application state has been persisted? |
| `_status/*.flag` | Has a workflow dependency output been produced? |
| `*.flag.json` | Which rule and context produced a flag? |
| `_queued`, `_submitted`, `_completed`, `_failed` | Where is a scheduler-backed worker in its handoff? |
| `_status/_orchestrator/*.json` | Is an orchestration driver protecting this analysis? |
| Scenario/member records | What remains authoritative after directories are removed? |

These are not interchangeable completion proofs. In particular, a successful process exit does not replace model-specific completion checking.

## Log ownership and dependency timestamps

`TRITONSWMM_log.write()` takes a per-log file lock, reloads disk state, and overlays the fields changed by that instance since its merge baseline. It publishes through `write_json`; failure to acquire the lock within 30 seconds becomes a `ProcessingError`. `deferred_writes()` batches updates and flushes on exit, including exceptional exit. Worker construction explicitly uses observer mode for shared system and analysis logs rather than rewriting them from every simulation process.

`write_status_flag()` creates a missing bare flag through a temporary file and atomic replacement. An existing bare flag is left untouched: rewriting only diagnostic provenance must not advance its mtime and cascade downstream work. The JSON sidecar is replaced independently. A rebuild therefore needs the dependency flag absent, not merely a refreshed sidecar.

## Worker handoff and actual completion

With `SLURM_JOB_ID` present, the simulation runner writes `_submitted/{rule_token}.json`, then removes the corresponding `_queued` marker. It clears an earlier failed marker when starting a new attempt. Local execution without a SLURM job identity does not take this marker path; container versus native execution does not change that identity gate. Container execution skips the native compilation-state precondition because its binary comes from the image.

The runner records `simulation_completed` from `model_run_completed(model_type)` and checks that predicate again before returning success. Explicit failure paths write `_failed` before returning. Its `finally` writes `_completed` only when neither terminal marker exists, then removes `_submitted`. `--defer-terminal-markers` leaves terminal publication and submitted cleanup to an enclosing owner. Hard termination can bypass `finally`, leaving evidence for the next reconciliation rather than proving a live worker indefinitely.

## Liveness is not absence

`job_is_dead_confirmed()` first asks `squeue`; a live result prevents death classification. It then requires a terminal `sacct` state to confirm death. An empty accounting result is not confirmation and keeps the waiter waiting. The lower-level probe distinguishes an unlaunchable `squeue` (`None`) from an answered absence; a timeout takes the not-live path and still needs corroboration.

Reprocessing acquires an orchestration claim before destructive artifact invalidation. Dry runs do not claim. The gate can refuse a live or indeterminate driver; a claim is retained until adopted/released by its workflow owner. Driver sentinels and worker sentinels protect different scopes, so clearing one does not establish that the other scope is idle. See [diagnostics and in-flight control](../operations/diagnostics-and-in-flight-control.md).

## Records survive directory removal

Scenario deletion persists a `pending` record **before** deletion, finishes deletion, then persists `done`. On resume, `pending` is already authoritative: readers use its captured model logs even if part of the directory still exists. Removal resumes deletion without rebuilding facts from that partial directory.

A run-stage force can set a removed record to `superseded`, returning authority to live files. Its caller must delete dependency flags first; the opposite order could leave a superseded record with completed flags and no directory. Removed sensitivity members similarly retain records under the master; a `done` member is exposed through a read-only recorded view instead of being reconstructed as an active member.

See [reprocessing and retention](../workflows/reprocessing-and-retention.md), [ensemble lifecycle](../workflows/ensemble-lifecycle.md), and [layout migrations](../operations/layout-migrations.md).

## Evidence scope

This page describes source contracts. No solver execution, scheduler query, remote API, or test run was performed while authoring it.
