# CLI reference

Run `hhemt --help` for the authoritative list of commands and `hhemt {command} --help` for the options of one command. Each table row gives the purpose of a verb and the occasion to use the verb.

Most commands take the same arguments.

```bash
hhemt {command} --system-config CFG_SYS --analysis-config CFG_ANA
```

--8<-- "hpc-system-config-role.md"

Pass the HPC system config as `--hpc-system-config`. Author the config as [HPC-profile setup](../how-to/hpc-profile-setup.md) describes.

A command that takes configs runs as `hhemt {verb} --system-config CFG_SYS --analysis-config CFG_ANA` or as the equivalent `python -m hhemt.{module}` call with the same configs. The dry run in [Config filling § 6](../how-to/config-filling.md) is the worked example.

```bash
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml --dry-run
```

[Tutorials](../tutorials/index.md) call `analysis.run()`, `analysis.reprocess()`, and `analysis.render_report()` from a Python session.

## Running an analysis

| Command | Purpose |
|---|---|
| `run` | Execute the workflow from the system and analysis configs. Use the command when you hold one system config and one analysis config, and use `run-experiment` when an experiment definition names the configs. `--dry-run` validates and prints the plan without executing. In native mode, compile the solver first with [Compile the solver](../how-to/compiling-the-solver.md). In container mode, `--build-sifs` builds any missing image inside the driver before submitting. Alongside `--build-sifs`, `--sif-build-config` names the build venue and resources. `--force-sif-rebuild` rebuilds every identity that the experiment requires even when the image is present. `--no-wait` builds and verifies without submitting the experiment. The flags `--sif-build-config`, `--force-sif-rebuild`, and `--no-wait` need `--build-sifs`. See [Building container images](../how-to/building-container-images.md). Resume is the default mode, so an interrupted workflow continues without a flag. `--mode fresh` clears the artifacts and starts fresh, and `--mode resume` names the default. A `--dry-run` skips the wipe guard that `--mode fresh` applies, so a dry run shows the plan and leaves open whether the real run is refused. `--overwrite` equals `--override-force-rerun all` and re-executes every simulation without clearing anything. `--status` prints the status report of the workflow and exits without executing anything. |
| `run-experiment` | Execute an [experiment definition](glossary.md#experiment-definition). The `experiment.yaml` names the configs, the inputs, and the toolkit pin. See [Running an experiment definition](../how-to/running-an-experiment-definition.md). |
| `reprocess` | Re-execute the downstream stages of processing, consolidation, and rendering against simulation outputs that already exist, without repeating the simulations. The command fits correct results with a wrong report. See [Forcing and suppressing reruns](../how-to/forcing-reruns.md). |
| `cancel` | Cancel a running workflow. The command signals Snakemake, cancels the workers, ends the driver session, and prints the cancellation record that `analysis.cancel()` returns. Name the analysis the way you started the workflow. `--experiment {definition dir}` takes an optional `--cluster` when the definition declares one cluster. `--system-config {file} --analysis-config {file}` takes an optional `--hpc-system-config`. The command refuses when given neither form or both forms. `--rule-class {prefix}` cancels only the jobs of that rule class, may be repeated, and leaves other jobs running. `--wait-timeout {seconds}` bounds the wait for the workflow process to exit. The command exits `0` when the `success` of the record is true and `1` otherwise. A `jobs_were_running` of `None` means the command could not establish whether jobs were running, as when the session is gone and the command checked only the driver. |

**A submission refuses while another orchestration driver may be alive.** Every submission records a driver sentinel under `{analysis_dir}/_status/_orchestrator/`. The next submission refuses with exit `3` while any recorded driver is alive or the host lacks a probe for the driver. Re-submit from the origin host of the sentinel, where the probe returns a measurement and no belief.

Pass `--override-live-driver DRIVER_ID` to `hhemt run` only when that host is unreachable and you have established that the driver is gone. Name the exact `driver_id` from a `*.json` file in the sentinel directory. The flag is an assertion and no force flag. The submission still refuses while any other driver is alive or indeterminate, and the command refuses an id that matches no sentinel. `hhemt run-experiment` has the same gate and exposes no override.

## Inspecting and reporting

| Command | Purpose |
|---|---|
| `status` | Print the status report of an experiment, with the progress of the workflow phase by phase and the failed simulations. Use `hhemt status --experiment {definition dir}` with an optional `--cluster` when the definition declares one cluster. The command reads logs and outputs, submits no job, and changes no file. |
| `eda` | Execute the in-process EDA loop of calculation, plots, and notebook. The command always writes the seeded notebook. [The seeded notebook](../tutorials/exploratory-analysis.md#the-seeded-notebook) covers the `eda_report/eda_report.html` export. |
| `bundle` | Emit a portable render bundle for local iteration on report renderers. The command is no workflow rule and has no toolkit walltime. The duration scales with the number of files that the figures declare. A fully declared zarr store contributes every chunk file, and a metadata-only store contributes only the metadata documents. Prefer a login-node `tmux` session to an `sbatch` with a guessed time limit. `--omit-output-zarrs` leaves out the chunk files of every output zarr store and carries the metadata documents. `combine` skips a same-role bundle whose data stores hold only metadata and records the reason `consolidated tree carried metadata-only`. The command refuses a clean-versus-resume pair that would read such a bundle. A bundle emitted this way refuses to re-execute a data figure, and `report-from-bundle` renders from the bundled figures. `bundle` refuses a sensitivity analysis that holds figures of members absent from the setup table. Re-render with `--override-force-rerun '{"subject": "all", "stage": "render"}'` to clear the figures. Pass `--allow-stale-figures` to bundle the figures as they stand and list them under `stale_figures_included` in the bundle manifest. |
| `report-from-bundle` | Render a report from an existing bundle that is single-analysis or combined by `hhemt combine`. The bundle is a directory or a zip. `--format {html,zip}` chooses the container, and [Producing a report](example-report.md#producing-a-report) states the default. `--declare-stale-plots` renders although the build stamp of the figures differs from the running build. |
| `combine` | Combine completed render bundles into one cross-experiment report plus a standalone combined bundle written as a zip. The input bundles are zips or directories. `--unzipped` writes a folder, and `--workdir` sets the location of the working folder. See [Combining experiments](../how-to/combining-experiments.md). |
| `static-plots` | Generate publication static figures under `static_plots/` with one Snakemake rule per static-plot ID. |

## Publishing and reproducing

| Command | Purpose |
|---|---|
| `publish` | Deposit the outputs of an experiment to a DOI-minting repository with `hhemt publish --experiment {definition dir} --target {zenodo or hydroshare}`. The default deposit is the analysis-directory set of consolidated results, the provenance record, the system config, and the analysis config. `--reprex-bundle` deposits the round-trippable reprex bundle instead, and only that form accepts `--exclude-config` and `--container-def`. The command credits a Zenodo deposit to the organization and takes no creators, and `analysis.publish(creators=...)` credits named people. The analysis config sets the dataset license, and the [FAIR scope table](fair-scope-table.md) lists the identifiers. |
| `reprex` | Subcommands for the reprex bundle. `reprex emit --experiment {definition dir}` writes the reprex bundle of the experiment and prints the bundle directory. `reprex check --bundle {reprex bundle dir} --reprex-config {file} --hpc-system-config {file}` checks whether a reprex bundle can execute on your cluster and prints the amendments. The required `--hpc-system-config` supplies the HPC settings of your cluster. `reprex check` exits `0` when the bundle fits your cluster and `1` when the cluster is unsuited to the bundle. A code of `2` or higher means the check could not run. `reprex config --bundle {reprex bundle dir}` writes the runnable `system_config.yaml` of the bundle, with the bundle-relative paths resolved to absolute paths under the bundle directory, and prints the path of that file. `--output {file}` writes the file in place of `system_config.yaml` in the bundle directory. `--software-dir {dir}` points both build directories at the `swmm` and `tritonswmm` subdirectories, as the experiment needs. |
| `ingest` | Fetch a published reprex bundle by DOI or PID, reconstitute the bundle, and print the runnable configs. The command is the consume half of the DOI round-trip. |
| `build-sifs` | Build every Apptainer SIF that the matrix of an experiment requires, addressed by identity from the toolkit commit, the TRITON pin, the family recipe, and the base image. `--dry-run` prints the identity table without building. See [Building container images](../how-to/building-container-images.md). |

## Checking correctness

| Command | Purpose |
|---|---|
| `recompute-plan` | Print the dry-run recompute plan for a bug-fix commit, with the work that the fix would require to re-execute. |
| `check-invalidating-fixes` | Report the known invalidating fixes that match the analysis and show whether the outputs are suspect. |

## Cleanup

The commands delete an analysis tree or the bookkeeping inside the tree. Each lists the deletion targets before deleting. The `cleanup-*` commands list by default and delete only with `--apply`. `delete` and `reset` take `--dry-run`.

| Command | Purpose |
|---|---|
| `delete` | Delete an entire analysis tree by dispatching per-scenario Snakemake delete jobs. The command refuses by default while any simulation is in flight. Run `--dry-run` first. |
| `reset` | Delete the whole analysis tree of an experiment so the next execution starts from scratch, with `hhemt reset --experiment {definition dir} --yes`. `--cluster` is optional when the definition declares one cluster and required when the definition declares none or several. `reset` executes the same distributed delete as `delete` and refuses while simulations are in flight. Without `--yes`, `reset` asks for confirmation. `--dry-run` prints the per-scenario disk preview of `delete --dry-run` and deletes no file. A reset exists as a command only. |
| `cleanup-orphans` | List or delete member directories orphaned by an edit to the sensitivity spreadsheet. |
| `cleanup-stale-metadata` | List or delete orphaned `.snakemake/metadata/` records left by past rule-output renames. |
| `cleanup-orphan-delete-sentinels` | Clear known-dead orphan delete sentinels left by killed delete workers. |
| `cleanup-settled-markers` | Prune settled completion and failure markers whose submitted-sentinel is gone. |

## Development

| Command | Purpose |
|---|---|
| `cache` | Report the on-disk caches that the synthetic test tier reuses and the tier address that keys the caches. `cache status` prints each root with a presence flag and the address that this checkout resolves to. `cache clear` lists the roots that a clear would remove and removes them only with `--apply`. Without a permitted compile venue, `cache clear` leaves the compiled-build tier in place and exits `2`, because the machine lacks the means to rebuild the tier. See [Branching and release](../explanation/branching-and-release-model.md). |
| `synth-experiment` | Load-smoke a synthetic experiment by validating the config and building the matrix. An empty `dem_resolution_ladder` selects the compute-config matrix with the partition as an axis, and a set ladder selects the DEM-resolution matrix. `--dry-run` writes no file. Without the flag the command writes the matrix CSV and generates the synthetic model. See [Running a synthetic compute-sensitivity experiment](../how-to/synthetic-compute-sensitivity-experiment.md). |
| `test` | Execute a test campaign against a chosen SUBJECT. The subject `toolkit` is the pytest suite of the toolkit as a scheduler array, described in the `hhemt test toolkit` subsection. The subject `analysis` is one end-to-end smoke test of YOUR experiment on the cluster of the experiment. Start the test with `hhemt test analysis --experiment {definition dir}`. `--cluster` is optional when the definition declares one cluster. `test analysis` submits real work, exits `3` when any representative reports failure, and takes no further subcommand. |

??? note "The toolkit test suite as a scheduler array"

    ### `hhemt test toolkit`

    The pytest suite of the toolkit, chunked across a scheduler array. A complete execution takes hours, so use a lighter tier while iterating.

    | Command | Purpose |
    |---|---|
    | `plan --toolkit {path} --runs-root {path}` | Collect the universe, derive the chunk membership, write `manifest.json`, and print the run id. **`plan` also compiles.** The command executes the warm inline on the invoking node unless the underlying module receives `--warm-performed-externally`, a flag that this CLI omits. The warm is the suite file `tests/test_synth_00_compile_models.py`, so the warm produces a different build than the workstation recipe in [Compile the solver](../how-to/compiling-the-solver.md). Do not run `plan` on a shared login node. A harness that dispatches chunks should invoke `python -m hhemt.suite._runner` with the flag and supply an awaited warm. |
    | `chunk --chunk {n} --run-dir {path}` | Execute one chunk. The exit code is the exit code of pytest. |
    | `aggregate --run-dir {path}` | Compute the cross-chunk verdict and write the summary pair. `--allow-not-green` exits 0 even when the verdict is not GREEN. |
    | `triage --toolkit {path} --runs-root {path}` | **Plans** a repeat of the failed and unevaluated tests of an earlier suite execution. `triage` writes a manifest, prints `run_id=`, and returns. **`triage` executes no test.** Submit the single chunk of the plan to execute the tests. `--from-run {id}` names the source. |

    **The filenames separate a triage result from a suite result.** A triage manifest declares `scope_intent: triage`. The aggregator binds the declaration as the scope whatever the caller passed, so the verdict line carries `scope=triage`. The summary then goes to `summary.triage.json` and `summary.triage.md` in place of `summary.json` and `summary.md`. The naming keeps a later reader from mistaking a triage for a suite summary. The filenames are one guard, the `.triage` run-directory suffix is the other guard, and each works alone.

    Only a `scope=union` verdict supports a suite-level green claim, and [Scope of a green claim](../explanation/testing-the-toolkit.md#scope-of-a-green-claim) gives the reason.

    `aggregate` exposes no `--scope`. A verdict from this CLI is therefore `scope=array`, or `scope=triage` when the manifest declares a triage intent. Only the underlying module reaches union, as with `--warm-performed-externally`.

    ```bash
    python -m hhemt.suite.aggregate --run-dir RUN_DIR --scope union
    ```

    The aggregator still downgrades the scope to `array` and records the reason unless some chunk reported `covers=complement` and every structurally gated test was attempted.

    **Splitting a heavy component.** A single heavy component can dominate the wall clock of the array. `python -m hhemt.suite._runner --heavy-split-budget-min {minutes}` splits each heavy component into time-balanced parts of about that length. The split requires `--durations-from {run-dir}`, the per-file duration table of an earlier execution, and `--isolate-trees-per-chunk`, a separate analysis-tree root for each part. The CLI exposes neither flag. Both flags belong to the underlying module, reached as `--warm-performed-externally` is.

    **Isolation is required, and omitting `--isolate-trees-per-chunk` raises an error.** Parts of one component share a single analysis tree without isolation and race on the tree. The race presents as an intermittent red in an unrelated test and never as a partition error.

    **Price the split before requesting the split.** The cost of isolation and the facts that a first execution at a new provenance fails to show are set out under [the fixture cache, and why a rebuild is silent](../explanation/testing-the-toolkit.md#the-fixture-cache-and-why-a-rebuild-is-silent).

## Exit codes

The CLI uses structured exit codes, so a script can branch on the failure class and skip the parsing of stderr. The verbs `run`, `run-experiment`, `build-sifs`, and `ingest` read every code from a shared map, and the table is exact for them.

`reset`, `publish`, `reprex`, and `status` read their failure codes from the same map with additions. `reset` exits `1` when you decline the confirmation or the analysis directory is still on disk afterwards. `reprex check` exits `1` when the check ran and the cluster is unsuited to the bundle.

The verbs `reprocess`, `recompute-plan`, `check-invalidating-fixes`, and `delete` predate the map and exit `1` where the table says `3`, `4`, or `5`. `delete` also exits `1` when you decline the confirmation prompt or the command finishes with the analysis directory still on disk. Neither outcome is a failure of the workflow.

| Code | Meaning |
|---|---|
| `0` | Success |
| `2` | Argument or configuration validation failure |
| `3` | Workflow or compilation failure, including a workflow that executed and reported failure |
| `4` | Simulation failure |
| `5` | Processing failure |
| `6` | Bundle schema mismatch |
| `10` | Unexpected error, the catch-all |

A `10` means that the underlying exception had no mapped code. When a script targets a specific failure and receives `10`, open an issue on the repository with the command and the output.

The codes of `run-experiment` cover the following cases.

| Code | Cases for `run-experiment` |
|---|---|
| `0` | Success, or a `--dry-run` whose plan built cleanly. A `--dry-run` skips the wipe guard, so exit `0` from a dry run of `--mode fresh` leaves open whether the real run is refused with exit `2`. |
| `2` | A bad `experiment.yaml`. A schema-invalid `system.yaml` or `analysis.yaml`. An unset `${VAR}`. A missing or placeholder `default_account` or `container.sif_root`. A declined override gate. An unknown `--mode` value. `--mode fresh` refused by the wipe guard. |
| `3` | Snakemake exited non-zero, `sbatch` refused the submission, a rule failed permanently, or a `--dry-run` whose plan failed. The console output printed just before the exit names the cause. |
