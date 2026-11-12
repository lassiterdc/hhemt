# Rerun triggers reference

hhemt drives the workflow with [Snakemake](https://snakemake.readthedocs.io/). `analysis.run()` tells Snakemake to use one rerun trigger, the modification time of the input files of a step, called `mtime`. A step reruns when an input is newer than the output that the step produced, or when the output is missing. The behavior suits most experiments. Edit a config, rerun, and only the affected steps rebuild. [Forcing and suppressing reruns](../how-to/forcing-reruns.md) covers forcing or suppressing a rerun.

The `force_rerun` field of the analysis config overrides the triggers. The shipped template leaves the field at the default.

```yaml
force_rerun: none
```

## Rerun triggers

`analysis.run()` passes Snakemake only the `mtime` trigger, so a step reruns when either condition holds.

1. **An input is newer than the output.** A file that the step declares as an input was modified after the output that the step produced. Editing a DEM, a weather file, or a SWMM template reruns the steps downstream of the file.
2. **The output is missing.** Snakemake has no output to compare and executes the step.

Neither a change to the list of declared inputs of a step nor a change to the code of a step reruns the step by itself. The toolkit narrows the triggers this way, so a resume after a crashed driver never reruns simulations that already finished.

Steady-state re-invocations are cheap. Once a phase is complete the status flag records the success. The per-model log records the success too. A repeat call of `analysis.run()` resumes and rebuilds no step.

## Surprising cases

### Editing the sensitivity spreadsheet

`sensitivity_analysis_definition.csv` is re-derived from the XLSX on **every** `analysis.run()`, and only the touched `member_id` chains rerun. Editing the CSV directly is futile, because the toolkit overwrites the CSV before Snakemake plans. The workflow reports "resuming, N/N complete" although you expected a new row. Always edit the XLSX and never the derived CSV.

### Removing a row from the spreadsheet

Removing a `member_id` leaves an orphan `members/member_*/` directory on disk. The toolkit detects the orphan, and `analysis.run()` **aborts** with a `ConfigurationError`, so no data is deleted silently. Re-invoke with `cleanup_orphans=True` or execute `hhemt cleanup-orphans --system-config CFG_SYS --analysis-config CFG_ANA --apply --force` to delete the orphan artifacts. Deletion is opt-in because the removal is irrecoverable.

### Steps that re-ran after an upgrade

A step reruns after an upgrade only when an input is newer than the output or the output is missing. A changed rule or a changed list of declared inputs never reruns a step alone. A release that writes a new input file or names an output differently causes such a case. List the steps that re-ran. When each has a newer input or a missing output, the rerun is expected, and the next invocation resumes without rebuilding.

### A re-invocation that rebuilt no step

`_status/*.flag` markers plus per-model log success records track completion, and file presence alone never does. A surviving flag marks the step as done and skips the step. A surviving log record skips the step even when the flag is gone. To force a specific step to rebuild whatever the flag says, use the override options that [Forcing and suppressing reruns](../how-to/forcing-reruns.md) describes.
