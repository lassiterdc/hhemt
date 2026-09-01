# CLI reference

Every `hhemt` command, grouped by what you would be doing when you reach for it.

Run `hhemt --help` for the authoritative list, and `hhemt {command} --help` for a
command's own options. This page describes what each verb is *for* and when to
choose it, not every flag.

Most commands take the same two arguments:

```bash
hhemt {command} --system-config CFG_SYS --analysis-config CFG_ANA
```

--8<-- "hpc-system-config-role.md"

Pass it as `--hpc-system-config`. See
[HPC-profile setup](../how-to/hpc-profile-setup.md) to author one.

## Running an analysis

| Command | What it does |
|---|---|
| `run` | Execute the workflow from a system + analysis config. The main entry point. `--dry-run` validates and prints the plan without executing. |
| `run-experiment` | Run a self-describing experiment bundle: a directory whose `experiment.yaml` names its own configs, inputs, and toolkit pin. See [Running an experiment bundle](../how-to/running-an-experiment-bundle.md). |
| `reprocess` | Re-run the downstream stages (process → consolidate → render) against simulation outputs that already exist, without re-running the simulations. This is the command for "the results are fine but the report is wrong". |

## Inspecting and reporting

| Command | What it does |
|---|---|
| `eda` | Run the in-process EDA loop (calc → plots → doc), producing `eda_report/eda_report.html`. |
| `bundle` | Emit a portable render bundle so you can iterate on report renderers locally instead of on the cluster. |
| `report-from-bundle` | Render a report from an existing bundle. |
| `combine` | Combine N completed render bundles into one cross-experiment report plus a standalone combined bundle. See [Combining experiments](../how-to/combining-experiments.md). |
| `static-plots` | Generate publication static figures under `static_plots/`, one Snakemake rule per static-plot ID. |

## Publishing and reproducing

| Command | What it does |
|---|---|
| `ingest` | Fetch a published reprex bundle by DOI or PID, reconstitute it, and print the runnable configs. The consume half of the DOI round-trip. |
| `build-sif` | Build an Apptainer SIF from a definition file, for container-mode execution. |

## Checking correctness

| Command | What it does |
|---|---|
| `recompute-plan` | Print the dry-run recompute plan for a bug-fix commit: what would need re-running given a fix. |
| `check-invalidating-fixes` | Report which known invalidating fixes match this analysis, i.e. whether its outputs are suspect. |

## Cleanup

These operate on an analysis tree's bookkeeping rather than its results. All are
safe to inspect first: each has a listing mode before it has a deleting mode.

| Command | What it does |
|---|---|
| `delete` | Delete an entire analysis tree by dispatching per-scenario Snakemake delete jobs. **Not `rm -rf`-equivalent**: it refuses by default while any simulation is in flight. Use `--dry-run` first. |
| `cleanup-orphans` | List or delete member directories orphaned by an edit to the sensitivity spreadsheet. |
| `cleanup-stale-metadata` | List or delete orphaned `.snakemake/metadata/` records left by past rule-output renames. |
| `cleanup-orphan-delete-sentinels` | Clear known-dead orphan delete sentinels left by killed delete workers. |
| `cleanup-settled-markers` | Prune settled completion/failure markers whose submitted-sentinel is gone. |

## Development

| Command | What it does |
|---|---|
| `synth-experiment` | Load-smoke a synthetic compute-config experiment: validate the config and build the partition-as-axis matrix. `--dry-run` writes nothing. See [Running a synthetic compute-sensitivity experiment](../how-to/synthetic-compute-sensitivity-experiment.md). |

## Exit codes

The CLI uses structured exit codes, so a script can branch on the failure class
rather than parsing stderr:

| Code | Meaning |
|---|---|
| `0` | Success |
| `2` | Argument or configuration validation failure |
| `3` | Workflow or compilation failure |
| `4` | Simulation failure |
| `5` | Processing failure |
| `6` | Bundle schema mismatch |
| `10` | Unexpected error, the catch-all |

A `10` means the underlying exception had no mapped code. If you are scripting
against a specific failure and getting `10`, that is worth reporting rather than
working around.
