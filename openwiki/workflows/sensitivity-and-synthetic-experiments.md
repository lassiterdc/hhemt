---
type: workflow
title: Sensitivity members and synthetic experiments
description: Trace sensitivity member identities and reusable system targets, primary aggregation, synthetic input generation, and descriptor-driven experiment execution.
tags: [sensitivity, synthetic-models, experiments]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-dd57620d9cc3d063f1be32a1
    resource: repo://docs/how-to/running-an-experiment-definition.md
  - id: openwiki-source-6c14b517bccc5c4f1c8f7644
    resource: repo://docs/how-to/synthetic-compute-sensitivity-experiment.md
  - id: openwiki-source-d09e93f9eb5acb46f8a03c17
    resource: repo://src/hhemt/experiment_definition.py
  - id: openwiki-source-c3394bd8f68a2ad12edd49b3
    resource: repo://src/hhemt/member_identity.py
  - id: openwiki-source-6f5ee698913e6f2bb99939d1
    resource: repo://src/hhemt/sensitivity_analysis.py
  - id: openwiki-source-64bdcc307bf1eb261abe492c
    resource: repo://src/hhemt/synthetic_experiment.py
  - id: openwiki-source-edd6ec14540e8b482b2323f8
    resource: repo://src/hhemt/synthetic_model/_build.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Sensitivity members and synthetic experiments

## One experiment, many member analyses

`TRITONSWMM_sensitivity_analysis` owns the sensitivity table, member construction, primary workflow, and final cross-member consolidation. Each active member is a full analysis under `members/member_{member_id}`. Member configuration inherits the primary analysis, applies nonempty `analysis.*` and supported `hpc.*` cells, disables nested sensitivity analysis, and records the parent experiment configuration.

The input table requires unique wildcard-safe `member_id` values matching `^[A-Za-z0-9_.]+$`. Retired `sa_id` table headers are rejected with a rename instruction; unknown parameter columns are rejected rather than ignored. This differs from artifact readers: `member_identity.py` accepts legacy `sa_id` in already-written output artifacts while preferring canonical `member_id`. Do not use legacy output-reader support as permission to author an old-format sensitivity table.

## System reuse and per-row hardware

A row can supply a separate `system_config_yaml`, use `system.*` overlays, or inherit the primary system. A row cannot combine the first two mechanisms. The implementation groups reusable system targets by DEM resolution, partition-derived GPU hardware/backend, and extra coupled compiler flags. It does not group merely by partition name: distinct partitions with the same relevant hardware can share a target, while different hardware requires distinct targets.

`hpc.partition` and `hpc.setup_partition` alias analysis resource selectors. Each row's ensemble partition determines its GPU target. The main orchestrator owns generated target YAML refresh; runner reconstruction avoids purging that shared directory. Member YAML materialization similarly avoids repeated writes by reader subprocesses.

## Scheduling and aggregation

The primary submits a Snakemake workflow spanning members and final primary consolidation. Execution locus and execution method are different controls: explicit local mode skips SLURM cross-node visibility checks, but `batch_job` and `1_job_many_srun_tasks` launch routing is determined by the method rather than the mode argument. For a local-method configuration, auto resolves locally, not from the presence of SLURM environment variables.

Do not assume the generated setup phase compiles solvers. Native generated setup rules assert the solver tier; container rules rely on the SIF's binary. Compile or build the appropriate solver/container through the separate build workflow before execution.

The primary's `experiment_datatree.zarr` contains `member_{member_id}` subtrees and a root `parameters` dataset; member parameter values also appear as member-node attributes. Summary groups enter the in-memory primary tree, while time series are copied to the persisted master one event at a time. Under clearing, copied member groups are read back equal before member-tree removal. Missing data with no recorded move is not silently accepted as a completed transfer. See [Output data model](../concepts/output-data-model.md) for tree layout and readers.

## Generate synthetic inputs before executing an ensemble

```bash
hhemt synth-experiment --config synth_experiment.yaml --dry-run
hhemt synth-experiment --config synth_experiment.yaml \
  --hpc-system-config hpc_system_config_uva.yaml --dest-dir runs/synth_cc/
```

The synthetic command prepares inputs and a matrix; it is not the full ensemble runner. The repository driver `scripts/experiments/synth_compute_config.py` executes the clean/resume experiments. Compute/model arms are separate sensitivity primaries with identical compute matrices and differing model toggles, because one sensitivity primary supports one enabled model. Combine their completed bundles for cross-arm reporting.

`experiment_matrix_rows` is shared by matrix emission and capacity validation. A DEM-resolution ladder switches the matrix from compute configurations to resolution rows; its construction checks whether in-line coupling nodes collapse onto the same DEM cell before emitting rows. The generic generator passes grid dimensions, cell size, and rainfall peak to `SyntheticModelParams`; richer production event shaping belongs to the repository driver, not implicitly to every generic generation call.

The source-level case builder creates DEM and landuse rasters, lookup CSV, watershed/boundary vectors, SWMM model, raingage mapping, and weather NetCDF, then checks junction rim elevations against the DEM. It is deterministic parameter-to-files generation with a caller-owned destination; cache locking and completion sentinels are not owned by this builder.

## Execute a portable experiment definition

```bash
hhemt run-experiment --experiment experiments/my_experiment --cluster uva --dry-run
hhemt run-experiment --experiment experiments/my_experiment --cluster uva
```

`experiment.yaml` supplies config paths, inputs, per-cluster HPC profile, toolkit pin, and optional container declaration. The directory names the definition; the analysis configuration supplies analysis identity. The runner loads/validates the descriptor, verifies its toolkit source-archive digest when specified, gates descriptor overrides, constructs a Toolkit, and runs it. An overriding noninteractive invocation must explicitly accept the override table; that acceptance is separate from authorizing a fresh wipe.

Before Toolkit construction, system and analysis `${VAR}` placeholders expand into resolved copies under `$SCRATCH_DIR/resolved_configs`, falling back to a temporary directory if scratch is unset. Unresolved placeholders fail. Shared visibility matters for dispatched compute-node rules; the HPC profile itself is passed through without this expansion. Account and declared-container root prerequisites fail early.

Resume is the default. Unless explicitly overridden, execution waits inside a SLURM allocation and detaches on a login node; a detached success reports submission, not completion. No generation command, solver job, remote API, or tests were executed to produce this page.

## Related workflows

- [Solver builds and containers](../integrations/solver-builds-and-containers.md)
- [Ensemble lifecycle](ensemble-lifecycle.md)
- [Exploratory analysis](exploratory-analysis.md)
