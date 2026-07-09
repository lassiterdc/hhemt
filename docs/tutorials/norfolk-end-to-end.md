# Norfolk end-to-end: from setup to a rendered report

!!! note "Prerequisites"
    Complete the [Quickstart](quickstart.md) (env + install). This tutorial runs the Norfolk-Irene case study; the fully-worked path runs locally with no HPC.

!!! warning
    The full multi-config sweep (serial → OpenMP → MPI → hybrid → GPU under both execution modes) needs an HPC allocation and hours of compute and is NOT runnable in CI. This tutorial is authored-and-code-checked; run the pieces your hardware supports.

!!! tip
    Run `analysis.test()` first — it runs a strict `_test/` subset (compile → run → process → consolidate → report) as a fast smoke before committing full compute.

## The worked path: a local serial run

Start with the one path that is guaranteed to succeed on a laptop: load the Norfolk example, optionally smoke-test it, run it locally, and render the report. Every later section varies this same run by editing config fields — the calls below never change.

```python
from hhemt.examples import NorfolkIreneExample
norfolk = NorfolkIreneExample.load()
norfolk.analysis.test()                         # optional smoke first
result = norfolk.analysis.run(from_scratch=False, execution_mode="local")
norfolk.analysis.render_report()                # renders the analysis report
```

Here is what each step does:

- **`NorfolkIreneExample.load()`** downloads the Norfolk case-study data once (anonymously — no HydroShare account needed), builds the system and analysis objects, and hands you back an example whose `.analysis` is the orchestrator and whose `.system` holds the DEM/compilation state.
- **`norfolk.analysis.test()`** is the optional smoke test. It runs a strict, least-demanding `_test/` subset of the analysis end-to-end — compile → run → process → consolidate → report — under `{analysis_dir}/_test/`, so you find a broken compile or a missing input in minutes instead of hours into the real run.
- **`norfolk.analysis.run(from_scratch=False, execution_mode="local")`** does the real work. `from_scratch=False` resumes any completed work rather than rebuilding, and `execution_mode="local"` forces a local run (no SLURM) using a thread pool sized to your machine.
- **`norfolk.analysis.render_report()`** assembles the self-contained report from the completed outputs.

Outputs land in the analysis directory (under your configured system directory), with per-scenario results beneath `sims/{event_id}/`. `render_report()` writes `analysis_report.zip` there by default — unzip it and open `report.html` in a browser. Pass `render_report(format="html")` if you would rather get a single self-contained `analysis_report.html` (larger, but no unzip step).

!!! warning "Use `analysis.run()` directly"
    Call `norfolk.analysis.run(...)` — NOT `norfolk.run(...)`/`Toolkit.run(mode=...)`. The `Toolkit.run()` facade is not wired for the first release; `analysis.run()` is the working interactive entry point.

## Scaling up: changing the compute configuration

Two orthogonal axes control how a run executes: the per-sim compute config (`run_mode` + the `n_*` counts) and the ensemble dispatch (`multi_sim_run_method`). The per-sim axis decides how ONE simulation uses cores/GPUs; the dispatch axis decides how the ensemble of simulations is launched. Vary the config fields below; the `analysis.run()` / `render_report()` calls from the worked path are unchanged.

??? example "Per-sim compute-config deltas (analysis config)"
    - **serial**: `run_mode: serial`
    - **openmp**: `run_mode: openmp`, `n_omp_threads: >=2`
    - **mpi**: `run_mode: mpi`, `n_mpi_procs: >=2` (require `n_mpi_procs >= n_nodes`)
    - **hybrid**: `run_mode: hybrid`, `n_mpi_procs: >=2` AND `n_omp_threads: >=2`
    - **single-GPU**: `run_mode: gpu`, `n_gpus: 1`
    - **multi-GPU**: `run_mode: gpu`, `n_gpus: >=2` (typically `n_mpi_procs == n_gpus`)

??? example "Ensemble dispatch (`multi_sim_run_method`)"
    - `local` — ThreadPoolExecutor on this machine (no SLURM).
    - `batch_job` — Snakemake in a login-node tmux session, one sbatch per sim (requires `hpc_total_job_duration_min` + `hpc_time_min_per_sim`). See the [HPC-profile setup guide](../how-to/hpc-profile-setup.md).
    - `1_job_many_srun_tasks` — one sbatch allocation + an srun pool (requires `hpc_total_nodes` + `hpc_total_job_duration_min`).

!!! warning
    `multi_sim_run_method` changes BOTH the execution strategy AND the generated Snakefile structure — it is not just a scheduler flag.

## Running on HPC

To run the ensemble on a cluster, configure a cluster profile first — see the [HPC-profile setup guide](../how-to/hpc-profile-setup.md) — then set `multi_sim_run_method` and pass `--hpc-system-config` / `hpc_system_config=`. Rather than re-running each compute config by hand, let the toolkit do the sweep for you: the benchmarking sensitivity analysis (`benchmarking_uva_minimal.xlsx`, shipped in `test_data/norfolk_coastal_flooding/`) automates the compute-config sweep as a single analysis.

## Comparison of compute configurations

| Config | `run_mode` | key fields | typical dispatch |
|---|---|---|---|
| serial | serial | — | local |
| OpenMP | openmp | `n_omp_threads>=2` | local / batch_job |
| MPI | mpi | `n_mpi_procs>=2` | batch_job / 1_job_many_srun_tasks |
| hybrid | hybrid | `n_mpi_procs>=2`, `n_omp_threads>=2` | 1_job_many_srun_tasks |
| single-GPU | gpu | `n_gpus=1` | batch_job |
| multi-GPU | gpu | `n_gpus>=2` | 1_job_many_srun_tasks |

## Next steps
- [Config-filling guide](../how-to/config-filling.md)
- [Rerun-trigger FAQ](../explanation/rerun-faq.md) — when re-running does and doesn't re-simulate.
