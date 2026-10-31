# Norfolk end-to-end: from setup to a rendered report

## Prerequisites

--8<-- "platform-support.md"

- A working install. See [Installation](../how-to/installation.md).
- The [Quickstart](quickstart.md) completed. This tutorial runs the Norfolk-Irene
  case study, and the fully-worked path runs locally with no HPC.

!!! tip
    Run `analysis.test()` first. It runs a strict `_test/` subset (setup → run → process → consolidate → report) as a fast smoke before committing full compute. The setup step checks that the solver is already compiled rather than compiling it.

## The worked path: a local serial run

Start with the one path that is guaranteed to succeed on a laptop: load the Norfolk example, optionally smoke-test it, run it locally, and render the report. Every later section varies this same run by editing config fields. The calls below never change.

This run is the configuration the Norfolk example ships, unchanged: `run_mode: serial`, which is one process, one thread and no GPU; `multi_sim_run_method: local`, which dispatches the ensemble to a local thread pool with no SLURM; and `local_cpu_cores_for_workflow: 4`, the size of that pool. The analysis is a single simulation of Hurricane Irene. Every section after this one starts from this baseline and changes one axis at a time.

In the analysis config these are:

```yaml
run_mode: serial
multi_sim_run_method: local
local_cpu_cores_for_workflow: 4
```

```python
from hhemt.experiments import NorfolkIreneExperiment
# hhemt_sha is the 40-hex commit you are running; `git rev-parse HEAD` prints it.
norfolk = NorfolkIreneExperiment.load(hhemt_sha="REPLACE-WITH-THE-FULL-40-HEX-HHEMT-COMMIT")
norfolk.system.compile_TRITON_SWMM()            # once per machine; skipped when already built
norfolk.analysis.test()                         # optional smoke first
result = norfolk.analysis.run(from_scratch=False, execution_mode="local")
norfolk.analysis.render_report()                # renders the analysis report
```

Here is what each step does:

- **`NorfolkIreneExperiment.load()`** downloads the Norfolk case-study data once ([Quickstart step 2](quickstart.md#2-get-the-norfolk-example-data) says how), builds the system and analysis objects, and hands you back an experiment whose `.analysis` is the orchestrator and whose `.system` holds the DEM/compilation state.
- **`norfolk.system.compile_TRITON_SWMM()`** clones and builds the coupled solver the first time it is called on a machine and skips the build afterwards. Every later step checks for that build and never performs it; [Compile the solver](../how-to/compiling-the-solver.md) covers the cluster form and when a rebuild is due.
- **`norfolk.analysis.test()`** is the optional smoke test. It runs a strict, least-demanding `_test/` subset of the analysis end-to-end (setup → run → process → consolidate → report) under `{analysis_dir}/_test/`, so you find an uncompiled solver or a missing input in minutes instead of hours into the real run.
- **`norfolk.analysis.run(from_scratch=False, execution_mode="local")`** does the real work. `from_scratch=False` resumes any completed work rather than rebuilding, and `execution_mode="local"` forces a local run (no SLURM) using a thread pool sized to your machine. `execution_mode="auto"`, which the [Quickstart](quickstart.md) passes, instead picks the locus from the analysis config's `multi_sim_run_method`: local for `local`, SLURM for `batch_job` or `1_job_many_srun_tasks`.
- **`norfolk.analysis.render_report()`** assembles the self-contained report from the completed outputs.

Outputs land in the analysis directory (under your configured system directory), with per-scenario results beneath `sims/{event_id}/`. [The interactive analysis report](../reference/example-report.md#producing-a-report) says which file `render_report()` writes there and how to open it.

## Scaling up: changing the compute configuration

Two orthogonal axes control how a run executes: the per-sim compute config (`run_mode` + the `n_*` counts) and the ensemble dispatch (`multi_sim_run_method`). The per-sim axis decides how one simulation uses cores/GPUs; the dispatch axis decides how the ensemble of simulations is launched. The baseline for both is the worked path above: `run_mode: serial` and `multi_sim_run_method: local`. The worked path calls `analysis.run(..., execution_mode="local")`, and an explicit `local` overrides the dispatch axis, so holding that call as written varies the per-sim axis only. To vary the ensemble dispatch, pass `execution_mode="auto"` as the [Quickstart](quickstart.md) does, so that `multi_sim_run_method` picks the locus.

???+ example "Per-sim compute-config deltas (analysis config)"
    - **serial**: `run_mode: serial`
    - **openmp**: `run_mode: openmp`, `n_omp_threads: >=2`
    - **mpi**: `run_mode: mpi`, `n_mpi_procs: >=2` (require `n_mpi_procs >= n_nodes`)
    - **hybrid**: `run_mode: hybrid`, `n_mpi_procs: >=2` and `n_omp_threads: >=2`
    - **single-GPU**: `run_mode: gpu`, `n_gpus: 1`
    - **multi-GPU**: `run_mode: gpu`, `n_gpus: >=2` (typically `n_mpi_procs == n_gpus`)

???+ example "Ensemble dispatch (`multi_sim_run_method`)"
    - `local`: ThreadPoolExecutor on this machine (no SLURM).
    - `batch_job`: Snakemake in a login-node tmux session, one sbatch per sim. The analysis config does not load without `hpc_total_job_duration_min`, `hpc_time_min_per_sim`, `hpc_ensemble_partition` and `hpc_setup_and_analysis_processing_partition`. See the [HPC-profile setup guide](../how-to/hpc-profile-setup.md).
    - `1_job_many_srun_tasks`: one sbatch allocation + an srun pool. The analysis config does not load without `hpc_ensemble_partition` and `hpc_setup_and_analysis_processing_partition`, and preflight also requires `hpc_total_nodes` and `hpc_total_job_duration_min`.

!!! warning
    `multi_sim_run_method` changes both the execution strategy and the generated Snakefile structure: it is not just a scheduler flag.

## Running on HPC

To run the ensemble on a cluster, configure a cluster profile first (see the [HPC-profile setup guide](../how-to/hpc-profile-setup.md)), then set `multi_sim_run_method` and pass `--hpc-system-config` / `hpc_system_config=`. Rather than re-running each compute config by hand, let the toolkit do the sweep for you: the benchmarking sensitivity analysis (`benchmarking_uva_minimal.xlsx`, shipped in `test_data/norfolk_coastal_flooding/`) automates the compute-config sweep as a single analysis.

!!! warning
    The full multi-config sweep (serial → OpenMP → MPI → hybrid → GPU under both execution modes) needs an HPC allocation and hours of compute, so the project's continuous integration (CI) runs do not execute it. This tutorial is authored-and-code-checked; run the pieces your hardware supports.

Before the first cluster run, read [Compile the solver: On a cluster](../how-to/compiling-the-solver.md#on-a-cluster), which says what a GPU partition needs that the worked path's Python call does not build.

## Comparison of compute configurations

| Config | `run_mode` | key fields | typical dispatch |
|---|---|---|---|
| serial | serial | none | local |
| OpenMP | openmp | `n_omp_threads>=2` | local / batch_job |
| MPI | mpi | `n_mpi_procs>=2` | batch_job / 1_job_many_srun_tasks |
| hybrid | hybrid | `n_mpi_procs>=2`, `n_omp_threads>=2` | 1_job_many_srun_tasks |
| single-GPU | gpu | `n_gpus=1` | batch_job |
| multi-GPU | gpu | `n_gpus>=2` | 1_job_many_srun_tasks |

## Next steps
- [Config-filling guide](../how-to/config-filling.md)
- [Rerun-trigger FAQ](../explanation/rerun-faq.md): when re-running does and doesn't re-simulate.
