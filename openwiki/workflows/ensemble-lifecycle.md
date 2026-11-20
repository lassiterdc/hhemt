---
type: workflow
title: Ensemble workflow lifecycle
description: Follow a configured ensemble from CLI or Toolkit through generated Snakemake rules, isolated runners, consolidation, reporting, and launch-versus-completion results.
tags: [workflow, ensemble, snakemake, execution]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-21b9d8cf4c93042742d6c138
    resource: repo://src/hhemt/analysis.py
  - id: openwiki-source-15bc8d0cbe6b4bd1d7c8f314
    resource: repo://src/hhemt/cli.py
  - id: openwiki-source-92e2fc947ca4b9baa7efe321
    resource: repo://src/hhemt/orchestration.py
  - id: openwiki-source-a4d8f719ac710d6d204d0826
    resource: repo://src/hhemt/prepare_scenario_runner.py
  - id: openwiki-source-64fd0a61cb1df574cc076b2d
    resource: repo://src/hhemt/process_timeseries_runner.py
  - id: openwiki-source-cbb6b02e668804cc97a4e3d1
    resource: repo://src/hhemt/run_simulation_runner.py
  - id: openwiki-source-244df0efdde1c09a688f9ce6
    resource: repo://src/hhemt/setup_workflow.py
  - id: openwiki-source-b0507628fb9723a5ed011ffa
    resource: repo://src/hhemt/toolkit.py
  - id: openwiki-source-704d4ea412eaff2fb1236809
    resource: repo://src/hhemt/workflow.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Ensemble workflow lifecycle

## Entry and ownership

A normal CLI invocation is configuration-driven:

```sh
hhemt run --system-config cfg_system.yaml --analysis-config cfg_analysis.yaml
```

The CLI maps fresh/resume to `analysis.run(from_scratch=...)`, carries overwrite on the independent force-rerun axis, and leaves execution-locus resolution to the analysis. `Toolkit.run` validates the same closed mode set and delegates to `analysis.run`; use the analysis directly for options the facade does not expose.

The analysis owns preflight and orchestration policy; the workflow builder emits and launches the DAG; runner modules reload persisted configs and own stage completion checks. Config files, not the parent process's transient Python objects, cross subprocess boundaries.

## Before launch

`analysis.run` refuses invalid inflow/coupling declarations before submission and resolves the report configuration from an explicit override or the required analysis report field. There is no silent default-report fallback. It can return a detached SIF-build-chain receipt before the ensemble launches when the build prestep selects that path.

Fresh mode is destructive: outside dry-run it calls the deliberate-wipe guard before removing the derived analysis directory. Resume keeps existing work. Layout stamping, stale-metadata handling, and settled-marker pruning occur in the run entry path; a dry-run is DAG validation, not a promise of zero filesystem writes (the code still stamps layout state and writes a Snakefile).

The analysis derives processing selection from enabled model types, resolves locus, and defaults to waiting locally but not on SLURM. A pickup override can independently control within-run checkpoint recovery, even after a fresh initial wipe.

## Generated stage graph

The representative regular-ensemble path is:

```text
setup → prepare(event) → run(model,event) → process(model,event)
                                               ↓
                         scenario/analysis consolidation
                                               ↓
                       status export → plots → report targets
```

The generator uses event IDs in wildcarded flags and maps them to integer event indices for runner calls. Processing consumes `c_run_*` flags and emits `d_process_*` flags; analysis consolidation emits `e_consolidate_complete.flag`. Per-scenario consolidation fans in enabled model processing flags. Report targets and active plot inputs are built together; scenario-status CSV and workflow summary are DAG outputs, not late success-hook side effects. An error hook attempts a diagnostic status export when the normal export cannot run.

### Setup and preparation

Generated setup asserts the native compile tier rather than compiling into shared software directories. Build native solver targets before launching; do not expect setup retries to repair a missing build. Container execution skips the native compilation assertion because solver executables live in SIF images.

`prepare_scenario_runner` runs in an isolated process to avoid PySWMM's multi-simulation limitation. It resolves GPU target information from HPC configuration, reconstructs system/analysis/scenario objects without rewriting shared system logs, prepares inputs, then refreshes the scenario log. Only a true `scenario_creation_complete` emits its completion flag; missing configs produce argument failure and exceptions produce a logged traceback and nonzero exit.

### Model execution and processing

The model runner distinguishes native builds from container execution: native models require their compiled backend; container mode does not require a host build. It emits the run flag on successful completion and writes failure markers on failed execution paths. The processing runner refuses models not recorded complete, verifies expected time-series and summary outputs, and performs guarded reclaim before emitting its completion flag. A stage's flag therefore belongs to the runner's postcondition, not merely to the existence of a launched process.

The process-rule template supports a separate processing stage; the generator also supports a configured combined simulation/processing shell. Do not treat Snakemake grouping as synonymous with that combined shell: their retry semantics differ.

### Consolidation and reporting

The canonical analysis-consolidation rule does not opt into incomplete analysis processing; its generator leaves `allow_incomplete=False`. Reprocessing has a separate partial-analysis path. Consolidation and status exports feed the report tail; bundling and publishing are separate workflows rather than implied by a successful ensemble launch.

## Submission families and resume adoption

Before generation, submission reconciles in-flight work so live submissions can become wait rules rather than duplicate run rules.

- `1_job_many_srun_tasks`: generate/validate a Snakefile and submit a single allocation; waiting is optional. Partial-failure sweeping occurs only when the caller actually waits.
- `batch_job`: generate/validate against the batch profile and host orchestration in tmux. Queue sentinels are written on launch, before any optional wait.
- Local family: resolve explicit local/SLURM locus, validate first, then use local Snakemake or detached SLURM execution. Local blocking completion sweeps permanent failures; detached execution has no in-process completion point to sweep.

Auto locus comes from configuration, not ambient `SLURM_JOB_ID`: local/unspecified family resolves local, the two SLURM families resolve SLURM, and explicit locus overrides pass through. Native/container is a separate execution-environment axis, not a synonym for local/cluster.

## Interpreting the return value

`WorkflowResult.success` on an unwaited detached launch certifies launch only. `phases_completed` is derived from requested phase flags, and `events_processed` is the in-scope integer-index list; neither is an observed per-stage/per-event completion ledger. Read durable status and produced outputs for actual progress.

Partial failures and retried-to-success disclosures are collected on local blocking and waited single-allocation paths, not tmux batch runs. Empty `partial_failures` on a detached or tmux receipt is not proof that all rules succeeded. The CLI routes the result through a shared reporter that checks `success` even for dry-run and returns workflow-failure exit 3; exceptions use the class-based exit mapping.

## Adjacent concerns

- [Ownership](../architecture/ownership.md) and [durable state](../architecture/durable-state.md)
- [HPC execution](../operations/hpc-execution.md)
- [Reporting and portable bundles](reporting-and-portable-bundles.md)
- [Reprocessing and retention](reprocessing-and-retention.md)

This trace is source-based; no ensemble, solver, container, or scheduler run was exercised.
