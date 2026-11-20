---
type: operations
title: SLURM execution and resource contracts
description: Configuration-selected execution locus, per-rule versus single-allocation dispatch, partition resource emission, and measured GPU binding safeguards.
tags: [hpc, slurm, resources, mpi, gpu]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-285b1969e3dfb9f60551e7ab
    resource: repo://docs/how-to/hpc-profile-setup.md
  - id: openwiki-source-21b9d8cf4c93042742d6c138
    resource: repo://src/hhemt/analysis.py
  - id: openwiki-source-8c5027a2ab5d04823b61b767
    resource: repo://src/hhemt/config/hpc_system.py
  - id: openwiki-source-18752212556e556350263bc4
    resource: repo://src/hhemt/gpu_bind_guard.py
  - id: openwiki-source-92e2fc947ca4b9baa7efe321
    resource: repo://src/hhemt/orchestration.py
  - id: openwiki-source-454953ddf0b30226a6c97b53
    resource: repo://src/hhemt/resource_management.py
  - id: openwiki-source-76507eb2d2ecd573f79f8fd2
    resource: repo://src/hhemt/run_simulation.py
  - id: openwiki-source-5fee4361564019699ffd6a76
    resource: repo://src/hhemt/validation.py
  - id: openwiki-source-704d4ea412eaff2fb1236809
    resource: repo://src/hhemt/workflow.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# SLURM execution and resource contracts

## Separate the three execution decisions

1. **Locus:** `resolve_execution_locus` honors explicit `local`/`slurm`; `auto` or null derives local for a local/null run method and SLURM for `batch_job` or `1_job_many_srun_tasks`. Unknown values raise. Merely inheriting `SLURM_JOB_ID` does not promote a local workflow to SLURM.
2. **Dispatch family:** the workflow builder has dedicated `batch_job` and `1_job_many_srun_tasks` branches. These are not aliases for the local executor.
3. **Binary environment:** `execution_environment` selects native versus container binaries independently of scheduler placement. See [solver builds and containers](../integrations/solver-builds-and-containers.md).

Allocation environment still matters for resource sizing even when it does not decide locus. `ResourceManager` detects an existing SLURM allocation for concurrency sizing; outside one it can bound parallelism by physical cores and available memory. Per-simulation estimates multiply MPI ranks by OpenMP threads, and sensitivity estimates take maxima across member demands.

## Driver and allocation ownership

| Family | Driver and work placement | Important consequence |
|---|---|---|
| `batch_job` | tmux orchestration drives the Snakemake SLURM executor's per-rule jobs | Stopping a simulation job is not stopping its driver. |
| `1_job_many_srun_tasks` | One submitted exclusive-node allocation hosts Snakemake and simulation steps | Concurrency is sized from the allocation rather than treating each step as its own job. |
| Local-family execution | Uses resolved locus; an explicit SLURM override is supported | Do not infer placement solely from the run-method label or ambient allocation. |

Before emitting either HPC-family workflow, the builder reconciles existing submissions so live work can become wait-rules instead of duplicate run-rules. The single-allocation path validates a dry-run DAG before submission, records queued simulation sentinels against the allocation ID, and only sweeps final failures if it actually waited for completion. Returning a detached job ID is not completion evidence.

Use [durable state](../architecture/durable-state.md) for sentinel semantics and [diagnostics and in-flight control](diagnostics-and-in-flight-control.md) for cancellation and unknown-driver handling.

## Cluster profiles describe capability, not discovery

An HPC profile owns account, modules, GPU request flavor, job/CPU concurrency budgets, and partition limits. `PartitionSpec` includes runtime, memory, CPUs/task, nodes, GPU limits, MPI support, and physical topology. The selected partition supplies `gpu_hardware` and `gpu_compilation_backend`; these are not analysis-wide hardware declarations. Simulation and setup/postprocessing partitions can differ.

`executor_profile_extras` is a guarded escape hatch. Config loading refuses overrides of toolkit-emitted per-rule resources such as tasks, MPI flags, GPU requests, CPUs/task, and `hhemt_cpus`. Profile merging also refuses conflicting scalar keys or duplicate named list resources rather than silently replacing the toolkit's constraints.

Use site documentation to author caps and measure optional guard inputs. The [HPC profile guide](../../docs/how-to/hpc-profile-setup.md) warns that a CPU budget only tracks a site ceiling that meters CPU rather than another billing quantity; `statvfs` free space is not necessarily an account quota.

## Resource requests and launch shape

`_build_resource_block` emits partition, runtime, memory, task/thread counts, and nodes, increasing node count if GPU topology requires it. CPU MPI jobs emit `mpi=True` so the executor honors task counts.

GRES multi-GPU jobs deliberately use `tasks=N`, `mpi=True`, and `tasks_per_gpu=0`. Suppressing inherited `--ntasks-per-gpu` avoids multiple copies of the Python runner racing on one scenario. Full-node GPU requests add `--exclusive`; strict GPU subsets do not. The actual per-rank GPU mapping is established in the inner launch, not proven by this allocation request.

When `max_concurrent_cpus` is set, each resource block carries `hhemt_cpus`. CPU rules use tasks × CPUs/task; GPU rules use GPUs × CPUs/task, except full-node exclusive GPU jobs, whose weight is nodes × physical CPUs/node. Missing topology in that exclusive case raises instead of undercounting charged CPUs.

Simulation launches select `srun` for either HPC family or resolved SLURM locus. Serial/OpenMP local runs execute directly; local MPI/hybrid uses `mpirun`. Standalone SWMM uses its input/report/output invocation rather than the TRITON checkpoint launcher.

The batch-job GRES multi-GPU inner launch uses explicit `--ntasks=N --gpus-per-task=1`, removes `--overlap`, and inserts the GPU guard. Other forms intentionally differ: the single-GPU/single-allocation GRES path retains `--ntasks-per-gpu=1`; the `gpus` flavor uses explicit tasks with per-task GPUs. Do not propagate one cluster branch's binding fix to every launch form without evidence.

## Guard layers and limits

### Paths across nodes

Submit and reprocess facades call `assert_configs_visible_cross_node` before materialization. For a SLURM locus it rejects supplied config YAMLs and enumerated config paths beneath the resolved system temporary directory. `HHEMT_ALLOW_NODE_LOCAL_CONFIGS=1` acknowledges/bypasses that check, and local locus skips it. This is a path-location heuristic, **not a compute-node mount probe** and not proof that every non-temporary path is shared. Stage inputs on a genuinely shared filesystem.

### CPU threads and allocation size

Snakemake child commands default unset/empty `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to 1 without overriding user exports or mutating the caller's environment. Solver commands set their own OpenMP count and binding. For purpose-sized jobs, positive allocation readings smaller than configured CPU/GPU needs cause launch refusal. Missing/zero task readings are treated as unavailable evidence, not proof of a zero-sized allocation. These parent-budget checks exclude single-allocation mode because its parent environment describes the pool rather than an individual simulation step.

### GPU binding evidence

The guarded batch-job multi-GPU GRES branch executes a host wrapper once per task before the solver/container command. L1 refuses missing/short per-node GPU counts (exit 99). L2 waits a bounded interval for task records, then checks distinct device identities (exit 98 on sharing). An early watchdog detects known binding-error signatures and terminates the process group.

The consolidated artifact under the scenario's `_status/` preserves records and verdict precedence: `short_step` → `shared_device` → `watchdog_bind_error` → `not_evaluated` → `pass`. Missing measurements are not synthesized passes. Older per-task launches can be classified by a nonempty log's negative signature, but that result is named `pass_by_negative_signature`, never a measured `pass`.

## Verification boundary

No cluster caps, node mounts, scheduler jobs, MPI fabric, or GPU placement were measured for this page. Configuration and source contracts are documented here; actual site correctness requires the generated command, allocation records, and per-task evidence from the target cluster. See [configuration and validation](../concepts/configuration-and-validation.md) before submission.
