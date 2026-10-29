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

**State the invocation, not the flag list.** A page that shows a reader how to invoke the toolkit shows the **config-driven** form: the `hhemt {verb} --system-config … --analysis-config …` line, or its `python -m hhemt.{module}` equivalent, naming the configs it reads. New prose speaks of commands by the configs they take, not by the flags they accept: `--help` is the authoritative list of both, and prose that repeats it goes stale.

**An interactive session belongs on a tutorial, and only there.** A page that drives `analysis.run()`, `analysis.reprocess()` or `analysis.render_report()` from Python is a tutorial page, and it names that session as its form.

**New prose adds no verb count and no verb list.** `hhemt --help` is the enumeration. Where a page already carries verb tables (`docs/reference/cli.md` carries six, 27 command rows over 23 top-level verbs, and names 35 distinct flags), those tables ARE its subject and are not this rule's object; the rule binds what is added, and an addition that re-counts or re-lists the surface is the second copy that goes stale.

The worked example of that form is the dry run in [Config filling § 6](../how-to/config-filling.md): `hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml --dry-run`.

## Running an analysis

| Command | What it does |
|---|---|
| `run` | Execute the workflow from a system + analysis config. The main entry point. `--dry-run` validates and prints the plan without executing. In native mode the solver must already be compiled; see [Compile the solver](../how-to/compiling-the-solver.md). In container mode, `--build-sifs` builds any missing image inside this run's driver before submitting; alongside it, `--sif-build-config` names the build venue and resources, `--force-sif-rebuild` rebuilds the planned identities even when present, and `--no-wait` builds and verifies without submitting the experiment. Those three do nothing on their own. See [Building container images](../how-to/building-container-images.md). Resume is the default mode, so an interrupted run picks up where it stopped without any flag; `--from-scratch` clears the run's artifacts and starts fresh, and `--overwrite` re-runs every simulation without clearing anything (shorthand for `--override-force-rerun all`). `--status` prints the workflow's status report and exits without running anything. |
| `run-experiment` | Run a self-describing experiment definition: a directory whose `experiment.yaml` names its own configs, inputs, and toolkit pin. See [Running an experiment definition](../how-to/running-an-experiment-definition.md). |
| `reprocess` | Re-run the downstream stages (process → consolidate → render) against simulation outputs that already exist, without re-running the simulations. This is the command for "the results are fine but the report is wrong". |

**A run refuses to submit while another orchestration driver may still be alive.**
Every submit records a driver sentinel under
`{analysis_dir}/_status/_orchestrator/`, and the next submit refuses with exit `3`
while any recorded driver is alive, or cannot be probed from the host you are on.
Prefer re-running from the sentinel's own origin host, where the probe returns a
measurement instead of a belief. Only if that host is unreachable, and only once
you have established the driver is gone, pass `--override-live-driver DRIVER_ID` to
`hhemt run`, naming the exact `driver_id` you read from a `*.json` file in that
sentinel directory. It is an assertion rather than a force flag: the submit still
refuses while any other driver is alive or indeterminate, and an id matching no
sentinel is refused rather than honoured. `hhemt run-experiment` is gated the same
way and exposes no override.

## Inspecting and reporting

| Command | What it does |
|---|---|
| `status` | Print an experiment's status report: how far its workflow has got, phase by phase, and which simulations failed. `hhemt status --experiment {definition dir}`, with `--cluster` optional when the experiment definition declares one cluster. Reads logs and outputs; submits nothing and changes nothing. |
| `eda` | Run the in-process EDA loop (calc → plots → notebook). Always writes the seeded notebook; the `eda_report/eda_report.html` export is best-effort and is skipped with a warning when the notebook cannot be executed (no `python3` kernel, a cell error, a timeout). The notebook is the source of truth. |
| `bundle` | Emit a portable render bundle so you can iterate on report renderers locally instead of on the cluster. Not a workflow rule, so it has no toolkit walltime: runtime scales with the number of files the figures declared (a fully-declared zarr store contributes every chunk file; a metadata-only store contributes only its metadata documents). Prefer a login-node `tmux` session over an `sbatch` with a guessed time limit. |
| `report-from-bundle` | Render a report from an existing bundle. |
| `combine` | Combine N completed render bundles into one cross-experiment report plus a standalone combined bundle. See [Combining experiments](../how-to/combining-experiments.md). |
| `static-plots` | Generate publication static figures under `static_plots/`, one Snakemake rule per static-plot ID. |

## Publishing and reproducing

| Command | What it does |
|---|---|
| `publish` | Deposit an experiment's outputs to a DOI-minting repository: `hhemt publish --experiment {definition dir} --target {zenodo or hydroshare}`. By default the deposit is the analysis-directory set (consolidated results, provenance record and the two configs); `--reprex-bundle` deposits the round-trippable reprex bundle instead, and only that form accepts `--exclude-config` and `--container-def`. The dataset license is read from the analysis config. |
| `reprex` | Two subcommands. `reprex emit --experiment {definition dir}` writes the experiment's reprex bundle and prints the directory it wrote. `reprex check --bundle {reprex bundle dir} --reprex-config {file} --hpc-system-config {file}` checks whether a reprex bundle can run on your cluster and prints what to amend; your cluster's HPC settings come from `--hpc-system-config`, which is required. `reprex check` exits 0 when the bundle fits your cluster, 1 when it does not, and 2 or above when the check could not run. |
| `ingest` | Fetch a published reprex bundle by DOI or PID, reconstitute it, and print the runnable configs. The consume half of the DOI round-trip. |
| `build-sifs` | Build every Apptainer SIF an experiment's matrix requires, addressed by identity (toolkit commit, TRITON pin, family recipe, base image); `--dry-run` prints the identity table without building. |

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
| `reset` | Delete an experiment's whole analysis tree so its next run starts from nothing: `hhemt reset --experiment {definition dir} --yes`, with `--cluster` optional when the experiment definition declares one cluster and required when it declares none or several. It runs the same distributed delete as `delete`, so it refuses while simulations are in flight. Without `--yes` it asks for confirmation. `--dry-run` prints the same per-scenario disk preview as `delete --dry-run` and deletes nothing. A reset is a command and never a config field. |
| `cleanup-orphans` | List or delete member directories orphaned by an edit to the sensitivity spreadsheet. |
| `cleanup-stale-metadata` | List or delete orphaned `.snakemake/metadata/` records left by past rule-output renames. |
| `cleanup-orphan-delete-sentinels` | Clear known-dead orphan delete sentinels left by killed delete workers. |
| `cleanup-settled-markers` | Prune settled completion/failure markers whose submitted-sentinel is gone. |

## Development

| Command | What it does |
|---|---|
| `synth-experiment` | Load-smoke a synthetic compute-config experiment: validate the config and build the partition-as-axis matrix. `--dry-run` writes nothing. See [Running a synthetic compute-sensitivity experiment](../how-to/synthetic-compute-sensitivity-experiment.md). |
| `test` | Run a test campaign against a chosen SUBJECT. Two subjects: `toolkit`, the toolkit's own pytest suite run as a scheduler array (see the subsection below), and `analysis`, one end-to-end smoke run of YOUR experiment on its own cluster: `hhemt test analysis --experiment {definition dir}`, with `--cluster` optional when the experiment definition declares one cluster. `test analysis` submits real work, exits `3` when any representative reports failure, and takes no further subcommand. |

### `hhemt test toolkit`

The toolkit's own pytest suite, chunked across a scheduler array. Hours, not seconds. This
is not the tier you run while iterating.

| Command | What it does |
|---|---|
| `plan --toolkit {path} --runs-root {path}` | Collect the universe, derive chunk membership, write `manifest.json`, print the run id. **It also compiles: `plan` runs the warm inline, on the invoking node, unless the underlying module is given `--warm-performed-externally`, a flag this CLI does not expose.** The warm is the suite's own `tests/test_synth_00_compile_models.py`, so it is a different build from the workstation recipe in [Compile the solver](../how-to/compiling-the-solver.md). Do not run it on a shared login node. A harness that dispatches chunks should invoke `python -m hhemt.suite._runner` with that flag and supply its own awaited warm. |
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

`aggregate` exposes no `--scope`, so a verdict produced through this CLI is
`scope=array`, or `scope=triage` when the run's manifest declares a triage intent.
Union is reached only through the underlying module, the same way
`--warm-performed-externally` is:

```bash
python -m hhemt.suite.aggregate --run-dir RUN_DIR --scope union
```

Even there the aggregator downgrades the scope back to `array`, and records why,
unless some chunk reported `covers=complement` and every structurally gated test
was attempted.

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

**Price the split before asking for it.** What isolation costs, and what a first run at a
new provenance cannot tell you, are set out under
[the fixture cache, and why a rebuild is silent](../explanation/testing-the-toolkit.md#the-fixture-cache-and-why-a-rebuild-is-silent).

## Exit codes

The CLI uses structured exit codes, so a script can branch on the failure class
rather than parsing stderr. Four verbs read every code from one shared map, so for
those four the table is exact: `run`, `run-experiment`, `build-sifs` and `ingest`.

`reset`, `publish`, `reprex` and `status` read their failure codes from the same map, with two additions: `reset` exits `1` when you decline its confirmation or the analysis directory is still on disk afterwards, and `reprex check` exits `1` when the check ran and the bundle does not fit your cluster.

Four verbs predate that map and still exit `1` where the table says `3`, `4` or `5`:
`reprocess`, `recompute-plan`, `check-invalidating-fixes` and `delete`. `delete`
also exits `1` for two outcomes that are not failures of the run at all: you
declined its confirmation prompt, or it finished with the analysis directory still
on disk.

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

For `run-experiment`, the codes cover these cases:

| Code | What it covers for `run-experiment` |
|---|---|
| `0` | success, or a `--dry-run` whose plan built cleanly |
| `2` | a bad `experiment.yaml`, a schema-invalid `system.yaml` or `analysis.yaml`, an unset `${VAR}`, a missing or placeholder `default_account` or `container.sif_root`, a declined override gate, an unknown `--mode` value, or `--mode fresh` refused by the wipe guard |
| `3` | Snakemake exited non-zero, `sbatch` refused the submission, a rule failed permanently, or a `--dry-run` whose plan failed; the console output printed just before the exit names the cause |
