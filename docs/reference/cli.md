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
| `run` | Execute the workflow from a system + analysis config. The main entry point. `--dry-run` validates and prints the plan without executing. In native mode the solver must already be compiled; see [Compile the solver](../how-to/compiling-the-solver.md). |
| `run-experiment` | Run a self-describing experiment bundle: a directory whose `experiment.yaml` names its own configs, inputs, and toolkit pin. See [Running an experiment bundle](../how-to/running-an-experiment-bundle.md). |
| `reprocess` | Re-run the downstream stages (process → consolidate → render) against simulation outputs that already exist, without re-running the simulations. This is the command for "the results are fine but the report is wrong". |

## Inspecting and reporting

| Command | What it does |
|---|---|
| `eda` | Run the in-process EDA loop (calc → plots → notebook). Always writes the seeded notebook; the `eda_report/eda_report.html` export is best-effort and is skipped with a warning when the notebook cannot be executed (no `python3` kernel, a cell error, a timeout). The notebook is the source of truth. |
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
| `test` | Run a test campaign against a chosen SUBJECT. Today the only subject is `toolkit`, the toolkit's own pytest suite run as a scheduler array. See the subsection below. |

### `hhemt test toolkit`

The toolkit's own pytest suite, chunked across a scheduler array. Hours, not seconds. This
is not the tier you run while iterating.

| Command | What it does |
|---|---|
| `plan --toolkit {path} --runs-root {path}` | Collect the universe, derive chunk membership, write `manifest.json`, print the run id. **It also compiles: `plan` runs the warm inline, on the invoking node, unless the underlying module is given `--warm-performed-externally`, a flag this CLI does not expose.** Do not run it on a shared login node. A harness that dispatches chunks should invoke `python -m hhemt.suite._runner` with that flag and supply its own awaited warm. |
| `chunk --chunk {n} --run-dir {path}` | Execute one chunk. The exit code is pytest's. |
| `aggregate --run-dir {path}` | Compute the cross-chunk verdict and write the summary pair. `--allow-not-green` exits 0 even when the verdict is not GREEN. |
| `triage --toolkit {path} --runs-root {path}` | **PLANS** a re-run of only the prior run's failed and unevaluated set: it writes a manifest, prints `run_id=`, and returns. **It executes nothing.** Submit that run's single chunk to actually run the set. `--from-run {id}` names the source run. |

**A triage result is not a suite result, and the filenames enforce it.** A triage manifest
declares `scope_intent: triage`, which the aggregator BINDS as the scope regardless of what
the caller passed, so the verdict line carries `scope=triage`. The summary is then written as
`summary.triage.json` and `summary.triage.md` rather than `summary.json` and `summary.md`.
That naming is deliberate, so that no later reader can pick a triage up as a suite summary.
That split is
one of two independent arms guarding it; the other is the `.triage` run-directory suffix.

Only `scope=union`, the array plus its complement, supports a suite-level green claim.
A `scope=array` result does not cover the complement's tests at all.

**Splitting a heavy component.** A single heavy component can dominate the array's wall
clock. `python -m hhemt.suite._runner --heavy-split-budget-min {minutes}` splits each heavy
component into time-balanced parts of about that length. It requires `--durations-from
{run-dir}`, which supplies the per-file duration table from an earlier run, and
`--isolate-trees-per-chunk`, which gives each part its own analysis-tree root. Neither is
exposed by this CLI; both are flags on the underlying module, reached the same way
`--warm-performed-externally` is.

**Isolation is a requirement rather than an option, and omitting it raises.** Parts of one
component would otherwise share a single analysis tree and race it, which presents as an
intermittent red in an unrelated test rather than as a partition error.

**Two costs to price before asking for the split.** Isolation turns one shared fixture build
into one build per part, paid in the same wall clock the split is buying. And a first run at
any new provenance is not comparable to an earlier one, which is a standing property of the
fixture cache rather than a property of these flags: see
[the fixture cache, and why a rebuild is silent](../explanation/testing-the-toolkit.md#the-fixture-cache-and-why-a-rebuild-is-silent).

## Exit codes

The CLI uses structured exit codes, so a script can branch on the failure class
rather than parsing stderr. This table is the contract for `run` and `run-experiment`,
whose codes are read from one shared map; some other verbs still exit `1` on a failure
the table maps to `3`, `4` or `5`:

| Code | Meaning |
|---|---|
| `0` | Success |
| `2` | Argument or configuration validation failure |
| `3` | Workflow or compilation failure, including a workflow that ran and reported failure |
| `4` | Simulation failure |
| `5` | Processing failure |
| `6` | Bundle schema mismatch |
| `10` | Unexpected error, the catch-all |

A `10` means the underlying exception had no mapped code. If you are scripting
against a specific failure and getting `10`, that is worth reporting rather than
working around.
