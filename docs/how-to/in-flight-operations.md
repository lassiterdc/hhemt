# Operating on an analysis while jobs are in flight

These operations do not disturb simulations that are already in flight, but two of them refuse while the previous run's driver is still alive:

| Operation | Safe while sims are in flight? | What it does |
|---|---|---|
| `analysis.run(from_scratch=False)` | Yes, once the previous driver is gone; it refuses while that driver is alive | Resume-sweeps added scenarios/events and waits on sims a previous driver left queued or running instead of resubmitting them. Before it submits anything it reads the driver sentinels under `_status/_orchestrator/` and refuses while any recorded driver is alive, or cannot be probed from the node you are on (under `batch_job` a sentinel written from another login node cannot be probed from this one, so the refusal lasts until the sentinel is older than `hpc_total_job_duration_min` plus 30 minutes). Stop the driver first (see the tip below), or re-run from the node that submitted it. |
| `analysis.reprocess(start_with="consolidate")` | Yes, once the previous driver is gone; it refuses while that driver is alive | Deletes the report and its plots and re-renders them; the consolidated store, if there is one, is left in place, so nothing is re-aggregated unless you pass `regenerate_existing=True`. It cannot show partial results of a single analysis: if the analysis has never been consolidated, consolidation reads every scenario's summary and fails on the first scenario still in flight, leaving no report until a later resume completes the campaign and its render stage runs; if an earlier campaign already consolidated it, the report is re-rendered from that earlier store and shows none of the in-flight work. A sensitivity experiment is the exception: its consolidation skips members that are still incomplete. Runs `--nolock`; refuses while any recorded driver is alive, or cannot be probed from the node you are on. |
| `bundle_report_data()` | Yes | Harvests plots/configs into a portable bundle and writes the archive; the one thing it deletes is any figure under `plots/` that no plot rule in the current `Snakefile` declares (`plots/eda/` is exempt). Requires `render_report()` to have run once. |

!!! tip
    To force fresh sbatch submission instead of resuming, stop the driver first, then re-run with `override_force_rerun="all"`. Which process the driver is depends on `multi_sim_run_method`:

    - Under `batch_job` the driver is a tmux session on the login node you submitted from, and `scancel` on the sim jobs does not end it (Snakemake keeps running and may resubmit the cancelled work). Call `analysis.cancel()` from that node instead; it signals Snakemake, cancels the workers, and ends the session. If you then re-run from that same login node, the dead driver's sentinel under `_status/_orchestrator/` is reclaimed on its own. From another node the re-run cannot probe the session and refuses; only once you have confirmed the driver is gone, pass the `driver_id` the refusal names as `override_live_driver` (`--override-live-driver` on `hhemt run`; see the [CLI reference](../reference/cli.md#running-an-analysis)). A wrong id is refused, not ignored.
    - Under `1_job_many_srun_tasks` the driver is the SLURM job itself, so `scancel JOBID` stops the driver and the simulations together. The sentinel records that job id and is reclaimed from any login node once the job is gone.

## See also
- [Config-filling guide](config-filling.md)
- [HPC-profile setup](hpc-profile-setup.md)
