# Reprex round-trip: reproduce an analysis on a foreign HPC system

## Goal

Emit a **reprex bundle** from your analysis and run the round-trip check of a bundle on a foreign HPC system. A reprex bundle is a render bundle upgraded into a round-trippable Workflow-Run-Crate. A second person, given only the bundle, validates whether the analysis runs on *their* cluster and gets an exact list of the values to supply or amend. Reprex bundles and publishing are both opt-in. `analysis.run()` and `submit_workflow()` never emit a bundle.

## Prerequisites

- A working [installation](installation.md).
- A **rendered** analysis to emit a bundle.
- The bundle directory, a `reprex_config.yaml`, and the HPC system config of the target cluster to check a bundle.

## Key references

- [DOI round-trip for operators](doi-roundtrip-e2e.md) covers fetching and running the bundle on another machine.
- [HPC-profile setup](hpc-profile-setup.md) shows an example profile for the target cluster.
- The [CLI reference](../reference/cli.md#publishing-and-reproducing) lists the `reprex` subcommands.

## Emit a reprex bundle

Emit from a **rendered** analysis. The bundle harvests the sources declared during the report render, so render first.

```bash
hhemt reprex emit --experiment {experiment-definition-dir} --cluster {cluster}
```

The command writes the bundle and prints the **directory** that the command wrote. The command emits the bundle zip and extracts the zip to a sibling directory, so the round-trip below consumes the directory directly. A sensitivity analysis is the main reprex surface, because the problem-pair report per `member_id` and column is a sensitivity concept.

A **container-mode** analysis, set by `execution_environment: container` in place of the default, needs one Apptainer `.def` per distinct architecture in the matrix. Pass each file with a separate `--container-def`.

```bash
hhemt reprex emit --experiment {experiment-definition-dir} --cluster {cluster} \
    --container-def recipes/openmpi-cuda.def
```

Emitting without `--container-def` raises `ConfigurationError`. No field of the analysis or HPC-system config names a `.def`, since `ContainerSpec` has none, so the recipe is an emit-time operator input. The `container:` block of an experiment definition, `ContainerRef`, names no recipe either, as [The experiment definition layout](running-an-experiment-definition.md#the-experiment-definition-layout) describes. The family `.def` files ship as package data under `src/hhemt/sif/recipes/`. Take the `.def` file that your architecture needs and pass the path of the file. A native analysis ignores the argument.

The emitted bundle root carries the minimal runnable set.

- `cfg_system.yaml` and `cfg_analysis.yaml` hold path-scrubbed configs. The scrub rewrites Path fields relative to the bundle, nulls the software directories, and keeps experiment fields as written, except the `report` section of the analysis config. When a render record exists, that section holds the settings the report was rendered with, and a comment on the first line of the file names the source.
- `reprex_config.yaml` is a **template** that the reproducer fills with *their* values.
- `hpc_system_config.template.yaml` is a scrubbed HPC profile with `{your-allocation}` placeholders and never your real account.
- `Snakefile.source` is the generated workflow, typed as the `mainEntity` of the crate.
- `ro-crate-metadata.json` holds the Workflow-Run-Crate metadata. Heavy inputs, and the SIF of a container analysis, are recorded **by reference** with their `sha256`.

## Distribute by DOI

Mint a **runnable-DOI** in place of handing someone the bundle directory. Publish the reprex bundle to a DOI-minting repository, and a reproducer fetches, reconstitutes, and runs the bundle from the DOI alone. [The DOI round-trip](doi-roundtrip-e2e.md) walks through the deposit, the fetch, the credentials, the sandbox, excluded inputs, and the HydroShare manual-DOI caveat. `hhemt ingest` reads the crate `mainEntity` to reconstitute the runnable config pair, and the round-trip check below then applies on the cluster of the reproducer.

## Reproducer inputs

The reproducer fills the inputs below with the values of *their own* system.

1. A `reprex_config.yaml`, the minimal host-local field set.

   ```yaml
   default_account: my-alloc                # your HPC allocation
   sif_root: /scratch/my-alloc/sifs         # the fetched SIF and its .manifest.json sit
                                            # at an identity path under here
   target_ensemble_partition: gpu-a100      # your partition for the ensemble simulations
   # login_node / scratch_dir / target_setup_and_analysis_processing_partition optional
   ```

2. Their own HPC-system config for the target machine.

    --8<-- "hpc-system-config-role.md"

    [HPC-profile setup](hpc-profile-setup.md) describes the file.

## Run the round-trip check

```bash
hhemt reprex check --bundle {reprex-bundle-dir} --reprex-config reprex_config.yaml \
    --hpc-system-config {your-cluster-hpc-system-config.yaml}
```

The check runs the steps below in order.

1. **Verify the identity of the SIF.** When the crate references a SIF, as a container analysis does, the check requires a **mandatory, fail-closed** `sha256` match between the digest of the crate and the image that the check finds by that digest under `reprex_config.sif_root`. The match establishes that the bytes of your image file hash to the digest that the crate records. A mismatch, or a missing manifest under `sif_root` with that digest, raises `ProcessingError` before any validation runs. The match leaves open whether the digest of the crate is the digest that the producer ran, and whether your host, including the MPI, the drivers, and the GPU, matches the host of the producer, so a matching image alone promises no matching results. When the crate carries no SIF entity, the check is skipped. **The toolkit signs no SIF with PGP, and the digest is the identity carrier.** The `sif_signature_ok` field of the printed `ReprexResult` is retained for schema stability and is always `None`, because no `apptainer verify` runs. A bundle whose crate carries no SIF entity prints `sif_reference_present=False`. The value alone does **not** show a native analysis, because a container-mode bundle whose `bundle_manifest.json` carries no `sif_manifests` list, or whose manifest is missing or unreadable, lands there too. No field of the printed `ReprexResult` reliably separates a native bundle from such a container bundle today. The `sif_verified` field is informative in one direction only. `None` proves a container bundle with no digest to verify against, and `True` is consistent with both a genuinely native analysis and an unverifiable container analysis.

2. **Re-aim preflight at your profile.** The check re-runs validation with your partition selectors overlaid, so the report reflects the caps of *your* cluster.

3. **Report problems and amendments.** The command prints a `ReprexResult` with the fields below.

   | Field | Meaning |
   |---|---|
   | `runnable` | `True` when no sensitivity row exceeds a target partition cap |
   | `problem_pairs` | One `ValidationIssue` per member and column that exceeds a cap, naming the exact rows and resources to reduce |
   | `amendments` | Per-field experiment amendments, each labelled `validated` when a deterministic target-partition lookup pins the value, or `advisory` when you must decide, with a named reason |
   | `sif_reference_present` | `True` when the crate carries a SIF entity. `False` never implies a native analysis, as step 1 explains |
   | `sif_verified` | `True` when the digest matched. The field also prints `True`, ambiguously, when no SIF entity is present and the bundle announces no container mode. A genuinely native analysis **and** a container bundle whose `bundle_manifest.json` carries no `sif_manifests` list, is missing, or is unreadable both land there. `None` is the one unambiguous state. The bundle announces container mode and carries no digest, so the check verified no digest. `False` is unreachable, because a mismatch raises and never returns |
   | `sif_signature_ok` | **Always `None`.** Retained for schema stability. The toolkit signs no SIFs and runs no `apptainer verify`, so the field never carries a result and distinguishes no case |
   | `zero_user_info_leaks` | Informational. The field is empty or holds a combined entry whose text names every match of the generic patterns or of your private list in the bundle you hold. When `$HHEMT_REPREX_BLOCKLIST` is set to an unusable value, the entry is the diagnosis of the gate itself, as the section on producer information below describes |

Read `runnable` first. A false `runnable` means that `problem_pairs` names each exceeding member and column and `amendments` lists each field to change with the status and the reason.

Write a runnable `system_config.yaml`, with bundle-relative paths resolved to the inputs you fetched into the bundle, with the command below. The command prints the path of the file written.

```bash
hhemt reprex config --bundle path/to/bundle
```

The [`reprex` row of the CLI reference](../reference/cli.md#publishing-and-reproducing) lists `--output` and `--software-dir`. Without `--software-dir` both build directories stay null. A null value suits rendering and exploratory analysis from the bundle.

## Producer information in the bundle

An emit refuses to write a bundle that names the producing machine or person. Before the archive moves into place, the emit scans every text member with the layers below. A match in any member raises `ProcessingError` and leaves no archive at the output path. The message names each offending member, the layer that fired, and either a class label or the position of the matched value in the list. The message omits the matched value. The manifest records the layers that ran.

The emit rewrites each text member first, so that the scan runs on the text written into the archive. The rewrite replaces the directories, interpreter, home directory, login name, and SLURM account of the producer with a closed set of tokens:

- `{hhemt_analysis_dir}`
- `{hhemt_system_dir}`
- `{hhemt_toolkit_dir}`
- `{hhemt_package_dir}`
- `{hhemt_python}`
- `{hhemt_conda_env}`
- `{hhemt_home}`
- `{hhemt_user}`
- `{hhemt_slurm_account}`
- `{hhemt_producer_path}`

The last stands for the whole of a producer path, either an absolute path under the login segment that no other token covers, or a relative path that climbs 3 or more levels. A shorter relative path stays in place. The rewrite replaces the login name only where the name stands as a whole path segment, the local part of an address, or the value after `--user`, `-u`, or `user=`. The rewrite leaves a value shorter than 4 characters in place. The rewrite leaves declared inputs unchanged, so every recorded digest of an input stays true. The scan reads declared inputs like any other text member, and an input that carries a producer value refuses the emit.

The layers are below.

1. **Generic patterns.** An email address, and an absolute path under a home-directory or project root such as `/home/`, `/Users/`, `/scratch/`, `/project/`, or `/lustre/orion/`. The pattern also covers the Windows home directory in the backslash, forward-slash, and JSON-escaped forms, and the other cluster roots that the pattern lists. A brace-led segment such as `/scratch/{your-allocation}/sifs` passes the scan. The scan exempts the published author address of the toolkit and a closed list of addresses that vendored JavaScript puts in the license headers of the baseline report page. Any other address refuses the emit.
2. **Values read from the analysis being emitted.** The layer reads every absolute path held by the system, analysis, and HPC-system configs of the analysis being emitted, when the configs are present. The layer also reads the directory directly under a home-directory root in each config. The layer reads the `default_account` and `login_node` of the HPC config as well, and the running login name in the contexts above. The layer reads the same configs as the rewrite.
3. **Your optional private list.** Specific identifiers that no pattern describes, such as the allocation name of an institution, go in a plain text file in your own private storage, never in a public repository and never in a package. Point `$HHEMT_REPREX_BLOCKLIST` at the file. The file holds one entry per line, matched as a whole word without regard to case. The scan ignores blank lines and `#` comment lines. A `# min-tokens: N` line sets the number of distinct entries that the file must hold, and the number is 4 when the line is absent. The emit reads the list from the variable alone, with no default location, no copy inside the toolkit, and no fallback. A variable that names a missing file, a directory, an unreadable file, or a file below the entry count refuses the emit, because a mistyped path would otherwise pass as an absent list.

`bundle_manifest.json` records the layers that ran under `zero_user_info`. `pattern_rules` and `config_values` read `checked`. `private_list` reads `checked` when the variable named a usable list, `not configured` when the variable is unset, and `set but empty` when the variable is set to an empty string. Without a list the emit runs layers 1 and 2 and logs a warning.

A reproducer rehydrates a rewritten member with `hhemt.experiment_definition.rehydrate_placeholders(path, values=None, *, dest_dir=None)`. The function puts the reproducer's own values back for the tokens and writes a resolved copy named `rehydrated_{name}` under `dest_dir`, or under the scratch or temporary directory when no `dest_dir` is given. The defaults are listed below.

- `{hhemt_analysis_dir}` and `{hhemt_system_dir}` take the bundle root, meaning the nearest directory above the member that holds `bundle_manifest.json`.
- `{hhemt_python}` takes `sys.executable`.
- `{hhemt_home}` takes the home directory of the reproducer.
- `{hhemt_package_dir}` takes the installed package directory.
- `{hhemt_toolkit_dir}` has a default only when the installed package sits under `src/`.

The function refuses a token with neither a supplied value nor a default, such as a directory token on a member outside any bundle, and leaves no such token in place. `{hhemt_conda_env}`, `{hhemt_user}`, `{hhemt_slurm_account}`, and `{hhemt_producer_path}` stay as the token unless you supply a value.

??? note "Coverage limits of the producer-information check"
    - **The check reads text members only.** The emit copies zarr chunks and declared binary inputs through unchanged. Binary inputs include a DEM, a land-use raster, and a weather NetCDF. A producer value inside a binary file survives. The baseline report ships only as `bundle_baseline/analysis_report.html`, because the compressed entries of an archive version of the report carried producer values.
    - **The second layer shares a source with the rewrite.** Both read the same configs, so only your private list catches a value that is in neither the configs nor a generic pattern.
    - **The private list bounds the third layer.** The check finds only the entries that the list holds, so pointing the variable at a list you wrote yourself certifies no value beyond the list.
    - **History keeps earlier values.** The public repository keeps any value committed before the check existed.

    On the consume side, `hhemt reprex check` repeats the generic patterns and your private list over the bundle you hold. The check reports the matches in the `zero_user_info_leaks` field of the printed result as a combined entry whose text names every match, and the check fails on no match. A consumer holds no producer configs, so the check skips the second layer there. When the variable is set to an unusable value, the entry is the diagnosis and no match, so read the entry before treating a non-empty field as a list of leaks.
