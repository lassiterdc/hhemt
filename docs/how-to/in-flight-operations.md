# Operating on an analysis while jobs are in flight

You do NOT need to cancel a running Snakemake/SLURM workflow to do these:

| Operation | Safe while sims are in flight? | What it does |
|---|---|---|
| `analysis.run(from_scratch=False)` | Yes | Resume-sweeps added scenarios/events; under v2 graceful-rerun it substitutes wait-rules for in-flight sims rather than aborting. |
| `analysis.reprocess(start_with="consolidate")` | Yes | Re-aggregates + re-renders against existing outputs to see partial results; runs `--nolock`; refuses only if a live orchestration driver exists. |
| `bundle_report_data()` | Yes | Read-only harvest of plots/configs into a portable bundle; requires `render_report()` to have run once. |

!!! tip
    To force fresh sbatch submission instead of resuming, `scancel` the jobs and re-run with `override_force_rerun="all"`.

## See also
- [Config-filling guide](config-filling.md)
- [HPC-profile setup](hpc-profile-setup.md)
