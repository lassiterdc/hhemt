# Operating on an analysis while jobs are in flight

## Goal

Resume, re-render, and package an analysis while simulations are in flight. `hhemt run` and `hhemt reprocess` refuse while the previous driver is alive.

## Prerequisites

- An analysis with simulations submitted to a scheduler.

## Key references

- [CLI reference](../reference/cli.md#running-an-analysis) lists the refusal conditions, the ways to name an analysis, and the `hhemt cancel` exit codes.
- [Forcing and suppressing re-runs](forcing-reruns.md) describes the `--start-with` stages.
- The fields in the example below are described in [Config filling](config-filling.md) and [HPC-profile setup](hpc-profile-setup.md).

## Resume with `hhemt run`

Resuming is safe once the previous driver is gone.

Resume mode, the default, sweeps in added scenarios and events and waits on simulations that a previous driver left queued or running, without resubmitting them. The submit checks the driver sentinels under `_status/_orchestrator/` first. Under `batch_job`, only the login node that wrote a sentinel can probe the sentinel. On another node the refusal lasts until the sentinel is older than `hpc_total_job_duration_min` plus 30 minutes. Stop the driver first, as the tip below describes, or resume from the node that submitted the earlier analysis.

## Re-render with `hhemt reprocess --start-with consolidate`

Re-rendering is safe once the previous driver is gone.

The command deletes the report and the plots of the report, then re-renders both against the consolidated store on disk. [Forcing and suppressing re-runs](forcing-reruns.md#see-re-processed-results-without-re-simulating) describes `--regenerate-existing` and the other `--start-with` stages. The command runs with `--nolock` under the same driver check as `hhemt run`.

The re-render shows no partial results of a single analysis. If the analysis has never been consolidated, consolidation reads the summary of every scenario and fails on the first scenario in flight, so no report exists until a later resume completes the analysis and the render stage runs. If an earlier resume already consolidated the analysis, the command re-renders the report from that earlier store and shows none of the in-flight work. A sensitivity experiment is the exception, because consolidation of a sensitivity experiment skips members that are incomplete.

## Package results with `hhemt bundle`

Packaging is safe while simulations are in flight.

The command harvests plots and configs into a portable bundle and writes the archive. The command requires a rendered report. The command deletes only the figures under `plots/` that no plot rule in the current `Snakefile` declares, and `plots/eda/` is exempt.

!!! tip
    To force a fresh sbatch submission, stop the driver first and then resume with `--override-force-rerun all`. The process that serves as the driver depends on [`multi_sim_run_method`](../reference/glossary.md#multi_sim_run_method).

    - Under `batch_job` the driver is a tmux session on the login node of the submission. Running `scancel` on the simulation jobs leaves the driver running, and Snakemake may resubmit the cancelled work. Run `hhemt cancel` from that node. The [CLI reference](../reference/cli.md#running-an-analysis) lists the ways to name the analysis and the exit codes. `--rule-class run_` cancels only the simulation jobs and leaves processing jobs running. Resuming from the same login node reclaims the sentinel of the dead driver under `_status/_orchestrator/` without further action. A resume from another node has no way to probe the session and refuses. Once you confirm the driver is gone, pass the `driver_id` that the refusal names as `--override-live-driver` on `hhemt run`, as the [CLI reference](../reference/cli.md#running-an-analysis) describes.
    - Under `1_job_many_srun_tasks` the driver is the SLURM job, so `scancel JOBID` stops the driver and the simulations together. The sentinel records the job id, and any login node reclaims the sentinel once the job is gone.

When no node can probe the driver of a sentinel, the age limit of the sentinel is `hpc_total_job_duration_min` plus 30 minutes, or `hpc_max_wait_for_inflight_min` when `hpc_total_job_duration_min` is unset. A `batch_job` analysis sets the fields in the example below, with partition names of your own.

```yaml
multi_sim_run_method: batch_job
hpc_total_job_duration_min: 4320
hpc_time_min_per_sim: 480
hpc_ensemble_partition: your-partition
hpc_setup_and_analysis_processing_partition: your-partition
```
