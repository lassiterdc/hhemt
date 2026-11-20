---
type: architecture
title: System, analysis, and scenario ownership
description: Maps the public Toolkit facade to shared system resources, analysis orchestration, event-local scenarios, and cross-process workers.
tags: [architecture, ownership, orchestration]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-21b9d8cf4c93042742d6c138
    resource: repo://src/hhemt/analysis.py
  - id: openwiki-source-e70681fc40af77dfeddc854e
    resource: repo://src/hhemt/execution.py
  - id: openwiki-source-cbb6b02e668804cc97a4e3d1
    resource: repo://src/hhemt/run_simulation_runner.py
  - id: openwiki-source-92b126da4e4928a37957abfc
    resource: repo://src/hhemt/scenario.py
  - id: openwiki-source-e026674a4e4c40d1fa882dd1
    resource: repo://src/hhemt/system.py
  - id: openwiki-source-b0507628fb9723a5ed011ffa
    resource: repo://src/hhemt/toolkit.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# System, analysis, and scenario ownership

## Public facade and object graph

`Toolkit.from_configs()` schema-checks both documents, constructs a `TRITONSWMM_system` and a linked `TRITONSWMM_analysis`, then optionally runs analysis validation. The facade does not replace these objects: `tk.analysis` exposes the detailed API. `Toolkit.run()` validates its public mode, translates `fresh` into `from_scratch=True`, and delegates to `analysis.run()`; fresh mode is therefore potentially destructive, not just a different scheduling preference.

```text
Toolkit
  system: TRITONSWMM_system
    cfg_system, sys_paths, system log, plot, process
    analysis: TRITONSWMM_analysis
      cfg_analysis, optional HPC config, analysis_paths
      event table, workflow builder, analysis post-processing
      optional sensitivity controller
      scenario(event_iloc)
        event identity, scen_paths, prep log, per-model logs
        run, input generator, runoff modeler, full-model builder
```

## Shared system versus analysis state

The system owns processed DEM/Manning inputs, separate CPU/GPU build paths, compilation artifacts, `system_log.json`, and the system DataTree path. GPU hardware, compilation backend, and additional modules are injected into its constructor rather than being inferred from each scenario. Observer construction suppresses shared-log publication; `skip_layout_state` separately suppresses layout stamping.

An analysis chooses an explicit `analysis_dir` or defaults to `system_directory / analysis_id`. It binds `sims/`, analysis logs, event selection, execution strategy, workflow builder, and post-processing. A regular analysis binds `analysis_datatree.zarr`; a sensitivity primary instead binds `experiment_datatree.zarr` and constructs the sensitivity controller. The primary's `nsims` includes members; it is not the enabled-model run count.

Analysis construction checks that the configured toolkit SHA equals the running toolkit and refuses a dirty running checkout before creating its simulation directory. The event table is loaded during construction; the analysis log is lazy for observers. See [configuration and validation](../concepts/configuration-and-validation.md).

## Scenario-local responsibility

A scenario derives its stable event identity from the analysis's weather event indexers and roots its paths under `sims/{event_id}`. System toggles decide which model output paths exist. It owns preparation inputs and helpers, a `TRITONSWMM_run`, and independent per-model JSON logs so concurrent models do not share one model-log document.

Construction is not universally read-only: active scenarios create directories. For an event already represented by a pending/done removal record, construction keeps path arithmetic but avoids directory recreation and binds guarded preparation components. This distinction matters for status/report consumers after retention has removed raw directories; see [durable state](durable-state.md).

## Orchestration versus direct execution

The high-level `analysis.run()` resolves local versus SLURM execution locus, defaults to waiting only outside SLURM, and calls `submit_workflow()`. Submission routes a sensitivity primary to its sensitivity controller and a regular analysis to its `SnakemakeWorkflowBuilder`. Detached drivers retain enriched orchestration sentinels; a blocking local driver removes its sentinel on return.

The direct execution strategies in `execution.py` are a different layer. They consume `(launcher, finalize_sim)` pairs, not a Snakemake DAG. Selection is `SlurmExecutor` for `1_job_many_srun_tasks`, `LocalConcurrentExecutor` for `local`, and `SerialExecutor` otherwise. Serial execution launches/finalizes one pair at a time. Local concurrency uses a thread pool and records caught future errors as `failed`; its automatic parallelism branch refuses desktop GPU execution. The SLURM strategy validates allocated nodes/CPUs/GPUs, then polls a bounded set of running subprocesses. These direct strategies should not be mistaken for the high-level submission router.

## Cross-process boundary

Simulation workers reconstruct system, analysis, and scenario objects from configuration, with shared-log observer flags and `is_main_orchestrator=False`. Native workers check host compilation state; container workers use the image binary and skip that native check. Persisted logs, dependency flags, and sentinels—not Python object identity—carry progress between processes.

Follow [ensemble lifecycle](../workflows/ensemble-lifecycle.md) for stage ordering and [output data model](../concepts/output-data-model.md) for result ownership. Source contracts were inspected; no solver, scheduler, or test execution is asserted here.
