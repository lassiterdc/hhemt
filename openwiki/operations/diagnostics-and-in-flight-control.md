---
type: operations
title: Diagnosing and controlling analyses
description: Artifact-first triage, structured validation, scheduler and resume evidence, live-driver safety, and qualified disk accounting.
tags: [diagnostics, operations, status, recovery, liveness]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-fcae922be8b4da21a5427bac
    resource: repo://docs/how-to/diagnosing-a-failed-run.md
  - id: openwiki-source-d0aca4d822ef0805992e41ff
    resource: repo://docs/how-to/in-flight-operations.md
  - id: openwiki-source-b9d0ed732962cf4f4ec1584c
    resource: repo://src/hhemt/analysis_validation.py
  - id: openwiki-source-fe9082870bdbfebfb0dfa808
    resource: repo://src/hhemt/du_sentinels.py
  - id: openwiki-source-57c3fde4c7bb08963390ff60
    resource: repo://src/hhemt/export_scenario_status.py
  - id: openwiki-source-daa78a58553c8c43972cd6c7
    resource: repo://src/hhemt/orchestrator_sentinels.py
  - id: openwiki-source-d02496128facc6814b55a337
    resource: repo://src/hhemt/resume_events.py
  - id: openwiki-source-311c55f667b580280c8e1378
    resource: repo://src/hhemt/slurm_job_recovery.py
  - id: openwiki-source-704d4ea412eaff2fb1236809
    resource: repo://src/hhemt/workflow.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Diagnosing and controlling analyses

## Start with the failed boundary

Use `hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml --status`, then inspect the first incomplete phase's artifacts. The operator guide warns that a green solver exit or an exited batch-job tmux session is not scientific completion. TRITON completion uses `Simulation ends`; standalone SWMM uses its completion marker without a same-line error. Per-simulation logs are overwritten on each attempt, so use `.snakemake/slurm_logs/` for attempt history.

The `_status/` ladder is setup → optional separate preparation → model execution → processing → per-scenario consolidation → analysis/member consolidation → experiment consolidation. Do not sort solely by prefix: regular per-scenario `f_consolidate_scenario_evt-*` precedes analysis-level `e_consolidate_*`. Flag JSON sidecars identify rule/model/member/event. Native setup failures may require a separate solver compile; container setup uses the image's solver instead.

Choose the least expensive repair consistent with surviving inputs:

| Evidence | Next investigation/remedy |
|---|---|
| No setup flag | Read setup or setup-target logs; inspect input preprocessing and enabled native builds. |
| Run flag but no process flag | Inspect processing logs and retained raw outputs before choosing process-stage reprocessing. |
| Correct data, wrong report | Start at render, not simulation. |
| Correct simulations, stale aggregation | Start at consolidate/process as appropriate; first check retained artifacts. |
| Suspected correctness defect | Inspect invalidating-fix and producing-solver findings before planning recomputation. |

These are routing rules from the [failed-run guide](../../docs/how-to/diagnosing-a-failed-run.md), not permission to assume raw outputs survive configured retention. See [reprocessing and retention](../workflows/reprocessing-and-retention.md).

## Status and validation are different products

`export_scenario_status_to_csv` serializes the supplied status snapshot, or `analysis.df_status`, to `scenario_status.csv`. `validate_analysis` reuses one status snapshot across applicable checks and combines setup/run/processing, resource, provenance, known-defect, resume, retained-data, EDA, coupling, and consolidated-metadata checks with persisted EDA verdicts.

`persist_validation_report` writes `validation_report.json` via a temporary file replacement, including the producing toolkit stamp. The report renderer consumes this read-model rather than re-inspecting the whole run tree. A rendered finding describes the persisted inspection generation, not continuous monitoring of live jobs. [Durable state](../architecture/durable-state.md) explains artifact ownership; [exploratory analysis](../workflows/exploratory-analysis.md) explains EDA verdicts.

Resource checking distinguishes actual mismatches from informational missing measurements and negative-signature classifications. The summary separately discloses unmeasured GPU binding and SWMM threads. A passing result with missing measurements is not a measured resource certification.

## Resume and allocation evidence

`resume_events` stores per-simulation JSONL under the model log's `_resume_events/` sibling directory. Appends stamp the cumulative attempt count, optional SLURM job ID, and UTC recording time. Append failure returns `write-failed` instead of failing completed simulation work.

The evidence audit compares attempted resumes with captured solver events. A shortfall can mean an aborted resume, changed solver marker, failed ledger write, degraded log read, or an unreached harvest sink; it does not uniquely diagnose which. More recorded rows than the current counter are not automatically a defect: member force-reruns can restart the counter while older ledger rows survive. Non-monotone attempts disclose these eras.

Scheduler recovery unions plugin-report IDs, the job index, IDs already retained in its store, and submission IDs from Snakemake invocation logs. It queries `sacct -D` for job, batch, and numeric step rows, retains partial recovery, and returns explicit coverage counts. The caller's `RunMethod` is essential: numeric steps mean attempts for `batch_job`, but different simulations sharing one allocation for `1_job_many_srun_tasks`. Missing accounting is not evidence of zero cost or successful execution. Consolidated-store failure warns without destroying the already-written recovery CSV.

## In-flight control protects the driver boundary

Orchestrator sentinels live under `_status/_orchestrator/`. Blocking local drivers remove them on return; detached drivers leave them for liveness reclamation. Run/reprocess safety uses this driver authority, not just Snakemake's working-directory lock.

The liveness gate distinguishes alive, dead, and unknown. Alive blocks; dead permits reclamation; unknown blocks until a configured plausible-lifetime age bound. Job-ID probes can answer across hosts. PID/tmux probes are host-local, so an absent process or session on another login node is not death evidence. A negative scheduler answer also needs corroboration; unavailable accounting can leave it unknown.

`--override-live-driver` asserts that one exact named driver is dead and reclaims that sentinel; it does not disable protection against other drivers. Determine death first. For a batch-job tmux driver, cancelling only simulation jobs leaves the driver able to resubmit them. The [in-flight guide](../../docs/how-to/in-flight-operations.md) describes `hhemt cancel`, same-origin-node handling, and single-allocation cancellation.

A regular analysis cannot render partial unconsolidated scenarios; an existing consolidated store shows its older snapshot. Sensitivity consolidation can skip incomplete members. Portable packaging requires a rendered report and is documented as usable while simulations remain in flight.

## Disk totals carry completeness qualifiers

`_status/_du.json` stores live byte totals, breakdowns, reclaimed-byte counters, walk errors, and absent-child counts. Compare-and-write ignores only incidental timestamp changes and preserves mtime when meaningful values match. Parent rollups sum child sentinels and walk their own non-child artifacts; they do not recursively measure children missing sentinels. `_status` trees are excluded, and analysis scope also excludes `_test`.

Treat totals with `walk_errors` or `absent_children` as incomplete accounting. Above eight absent children, the rollup warns and names the scenario sentinel-seeding remedy. These byte counters are not filesystem quota or free-space measurements.

## Verification boundary

No scheduler query, cancellation, solver execution, or live analysis inspection was performed for this page. Commands and diagnostics describe repository-supported operation; actual findings require the affected run's artifacts.
