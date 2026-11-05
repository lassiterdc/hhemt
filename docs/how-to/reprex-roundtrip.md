# Reproduce an analysis on a foreign HPC system (reprex round-trip)

A **reprex bundle** is a render bundle upgraded into a round-trippable
Workflow-Run-Crate: a second person, given only the bundle, can validate whether your
analysis will run on *their* cluster and get an exact list of what they must supply or
amend. This guide covers both ends: **emitting** a reprex bundle, and **running the
round-trip check** on a foreign system.

Like publishing, this is opt-in: a reprex bundle is never emitted by `analysis.run()`
or `submit_workflow()`.

## Emit a reprex bundle

Emit from a completed, **rendered** analysis (the bundle harvests the sources declared
during `render_report()`, so run it first):

```python
# Single analysis
bundle_dir = analysis.reprex_bundle()

# Sensitivity primary (the main reprex surface, because the per-(member_id, column)
# problem-pair report is intrinsically a sensitivity concept)
bundle_dir = sensitivity.reprex_bundle()
```

`reprex_bundle()` returns a **directory** (it emits the bundle zip, then extracts it to a
sibling directory) so the round-trip below can consume it directly. Pass
`output_path=Path(...)` to control where the bundle is written.

For a **container-mode** analysis (`execution_environment: container`, which is not the
default) you must also pass one Apptainer `.def` per distinct architecture in the matrix.
Both surfaces take it:

```python
from pathlib import Path

defs = [Path("recipes/openmpi-cuda.def")]

bundle_dir = analysis.reprex_bundle(container_defs=defs)
bundle_dir = sensitivity.reprex_bundle(container_defs=defs)
```

Emitting without it raises `ConfigurationError`: no field of the analysis or HPC-system
config names a `.def` (`ContainerSpec` has none), so it is an emit-time operator input.
An experiment definition is no longer an exception: its `container:` block (`ContainerRef`)
names no recipe either, as [The experiment definition layout](running-an-experiment-definition.md#the-experiment-definition-layout)
describes. The family
recipes now ship as package data under `src/hhemt/sif/recipes/`; copy the one your
architecture needs and pass its path. A native analysis ignores the argument.

The emitted bundle root carries the minimal runnable set:

- `cfg_system.yaml` + `cfg_analysis.yaml`: path-scrubbed configs (their Path fields are
  rewritten relative to the bundle, the two software directories are nulled, and
  experiment fields are kept as written).
- `reprex_config.yaml`: a **template** the reproducer fills with *their* values.
- `hpc_system_config.template.yaml`: a scrubbed HPC profile with `{your-allocation}`
  placeholders (never your real account).
- `Snakefile.source`: the generated workflow, typed as the crate's `mainEntity`.
- `ro-crate-metadata.json`: the Workflow-Run-Crate metadata; heavy inputs and (for a
  container run) the SIF are recorded **by reference** with their `sha256`.

## Distribute by DOI (publish → ingest → run)

Instead of handing someone the bundle directory, you can mint a **runnable-DOI**: publish the
reprex bundle to a DOI-minting repository, and a reproducer fetches, reconstitutes, and runs
it from the DOI alone. [The DOI round-trip runbook](doi-roundtrip-e2e.md) is the walkthrough:
the deposit, the fetch, the credentials, the sandbox, excluded inputs and the HydroShare
manual-DOI caveat. `hhemt ingest` reads the crate `mainEntity` to reconstitute the runnable
config pair, and the round-trip check below then applies on the reproducer's cluster.

The deposit is made with `hhemt publish --experiment {experiment-definition-dir} --cluster {cluster} --target zenodo --reprex-bundle`, and the fetch with
`hhemt ingest --doi {minted-doi} --host zenodo --target-dir {target} --hpc-system-config {your-cluster-hpc-system-config.yaml}`;
the last option is needed because the reconstituted analysis dispatches with `batch_job`.

!!! example "Verified against a live deposit (Zenodo sandbox)"
    The command forms on this page were run against a Zenodo **sandbox** deposit of a
    15-configuration CPU sensitivity analysis: one data deposit and one runnable
    reprex-bundle deposit. The reprex-bundle deposit is far larger because it carries the
    declared inputs. The `10.5072` DOI prefix marks the sandbox: a sandbox record is
    discardable and is not a citable DOI, and production Zenodo mints `10.5281`.
    `HHEMT_ZENODO_BASE_URL=https://sandbox.zenodo.org` together with a sandbox token
    selects the sandbox. Not verified: the 15-member sweep was not re-run from the
    deposit; only `hhemt run --dry-run` on the reconstituted configs was.

## What the reproducer supplies

The reproducer fills two things with *their own* system's values:

1. A `reprex_config`, the minimal host-local field set:

   ```python
   from hhemt.config.reprex_config import reprex_config

   my_reprex = reprex_config(
       default_account="my-alloc",              # your HPC allocation
       sif_root="/scratch/my-alloc/sifs",       # the fetched SIF and its .manifest.json sit
                                                # at an identity path under here
       target_ensemble_partition="gpu-a100",    # your partition for the ensemble sims
       # login_node / scratch_dir / target_setup_and_analysis_processing_partition optional
   )
   ```

   The same fields written as a `reprex_config.yaml`:

   ```yaml
   default_account: my-alloc
   sif_root: /scratch/my-alloc/sifs
   target_ensemble_partition: gpu-a100
   ```

2. Their own HPC-system config for the target machine.

    --8<-- "hpc-system-config-role.md"

    See [HPC-profile setup](hpc-profile-setup.md) to author one.

## Run the round-trip check

```python
from hhemt.bundle import Bundle

result = Bundle.from_directory(bundle_dir).reprex(my_reprex, my_hpc_profile)
```

`reprex()` does three things, in order:

1. **Verify the SIF's identity.** If the crate references a SIF (a container run), the digest
   is a **mandatory, fail-closed** `sha256` match against the image found by that digest
   under `reprex_config.sif_root`, establishing that your image file's bytes hash to the
   digest the crate records. A mismatch, or no manifest under `sif_root` carrying that
   digest, raises `ProcessingError` before any validation runs. This does not establish
   that the crate's digest is the one the producer ran, or that your host (MPI, drivers,
   GPU) matches the producer's, so a matching image does not by itself promise matching
   results. When the crate carries no SIF entity, this check does not run. **The toolkit does not PGP-sign
   SIFs: the digest is the identity carrier.** `result.sif_signature_ok` is retained on
   `ReprexResult` for schema stability and is permanently `None`; no `apptainer verify`
   runs, so the field reports nothing and cannot. A bundle whose
   crate carries no SIF entity reports `result.sif_reference_present = False`. That alone
   does **not** mean the run was native: a container-mode bundle emitted before the
   `sif_manifests` manifest list existed, or one whose manifest is missing or
   unreadable, lands there too. No field on `ReprexResult` reliably tells those apart
   today. `result.sif_verified` is informative in one direction only: `None` proves a
   container bundle with nothing to verify against, while `True` is consistent with both
   a genuinely native run and an unverifiable container one.

2. **Re-aim preflight at your profile.** Validation is re-run with your partition
   selectors overlaid, so the report reflects *your* cluster's caps.

3. **Report problems and amendments.** `result` is a `ReprexResult`:

   | Field | Meaning |
   |---|---|
   | `runnable` | `True` when no sensitivity row exceeds a target partition cap |
   | `problem_pairs` | one `ValidationIssue` per `(member_id, column)` that exceeds a cap, naming the exact rows/resources to reduce |
   | `amendments` | per-field experiment amendments, each labelled `validated` (a deterministic target-partition lookup pins the value) or `advisory` (you must decide, with a named reason) |
   | `sif_reference_present` | `True` when the crate carries a SIF entity. `False` does **not** imply a native run (see step 1) |
   | `sif_verified` | `True` when the digest matched. Also `True`, ambiguously, when no SIF entity is present and the bundle does not announce container mode: that covers a genuinely native run **and** a container bundle whose `bundle_manifest.json` carries no `sif_manifests` list, is missing, or is unreadable. `None` is the one unambiguous state: the bundle does announce container mode and carries no digest, so nothing was checked. `False` is unreachable, because a mismatch raises rather than returns |
   | `sif_signature_ok` | **always `None`.** Retained for schema stability; the toolkit does not sign SIFs and runs no `apptainer verify`, so this field never carries a result and its value distinguishes nothing |
   | `zero_user_info_leaks` | informational. At most one entry, whose text names every match of the generic patterns or of your private list in the bundle you hold; when `$HHEMT_REPREX_BLOCKLIST` is set to something unusable the one entry is the gate's own diagnosis (see below) |

   ```python
   if not result.runnable:
       for issue in result.problem_pairs:
           print(issue)               # e.g. row[3].n_gpus requests 8, cap is 4
   for a in result.amendments:
       print(a.status, a.field_name, "->", a.to_value, f"({a.reason})")
   ```

Reconstitute a runnable `system_config.yaml` (bundle-relative paths resolved to the
inputs you fetched into the bundle) with
`hhemt.bundle._emit.reconstitute_runnable_config(bundle_dir)`.

## Zero-user-info status

An emit refuses to write a bundle that still names the machine and person that produced it. Before the
archive is moved into place, every text member is scanned with three layers, and a match in any member
raises `ProcessingError` and leaves no archive at the output path. The message names each offending
member, the layer that fired and either a class label or the position of the matched value in its list.
It never prints the value itself. A bundle that was emitted has therefore passed the layers that ran,
which its manifest records; the notes at the end of this section say what that does not cover.

Each text member is first rewritten, so that the scan runs on what is actually written into the archive.
The producer's directories, interpreter, home directory, login name and SLURM account are replaced by a
closed set of tokens: `{hhemt_analysis_dir}`, `{hhemt_system_dir}`, `{hhemt_toolkit_dir}`,
`{hhemt_package_dir}`, `{hhemt_python}`, `{hhemt_conda_env}`, `{hhemt_home}`, `{hhemt_user}`,
`{hhemt_slurm_account}` and `{hhemt_producer_path}`. The last stands for the whole of a producer path: an
absolute path under the login segment that no other token covers, or a relative path that climbs three or
more levels (one or two levels are kept). The login name is replaced only where it is plainly a user
name: a whole path segment, the local part of an address, or the value after `--user`, `-u` or `user=`. A
value shorter than four characters is never replaced. Declared inputs are not rewritten, so every
recorded digest of an input stays true; they are scanned like any other text member, and one that
carries a producer value refuses the emit.

The three layers:

1. **Generic patterns.** An email address, and an absolute path under a per-user or per-project root
   (`/home/`, `/Users/`, `/scratch/`, `/project/`, `/lustre/orion/`, the Windows home directory in its
   backslash, forward-slash and JSON-escaped forms, and the other cluster roots the pattern lists). A
   brace-led segment such as `/scratch/{your-allocation}/sifs` names no user and does not fire. The
   toolkit's own published author address is exempt, as is a closed list of nine addresses that
   vendored JavaScript puts in the license headers of the baseline report page; any other address
   refuses.
2. **Values read from what you are publishing.** Every absolute path held by the system, analysis and
   (when present) HPC-system configs of the analysis being emitted, the directory directly under a
   per-user root in each of them, the HPC config's `default_account` and `login_node`, and the running
   login name in the contexts above. This layer reads the same configs the rewrite reads.
3. **Your private list (optional).** Specific identifiers that no pattern can describe, such as an
   institution's allocation name, go in a plain text file that you keep in your own private storage,
   never in a public repository and never in a package. Point `$HHEMT_REPREX_BLOCKLIST` at it. It holds
   one entry per line, matched as a whole word without regard to case; blank lines and `#` comment lines
   are ignored; a `# min-tokens: N` line sets how many distinct entries the file must hold (four when it
   is absent). There is no default location, no copy inside the toolkit and no fallback: a variable that
   names a missing file, a directory, an unreadable file or a file below its entry count refuses the
   emit, because a mistyped path must not read as "no list".

`bundle_manifest.json` records what ran under `zero_user_info`: `pattern_rules` and `config_values` are
`checked`, and `private_list` is `checked` when the variable named a usable list, `not configured` when
it is unset, and `set but empty` when it is set to an empty string. With no list, the emit still runs the
first two layers and warns once. A reproducer rehydrates a rewritten member with
`hhemt.experiment_definition.rehydrate_placeholders(path, values=None, *, dest_dir=None)`, which puts the
reproducer's own values back for the tokens and writes a resolved copy named `rehydrated_{name}` under `dest_dir`, or under the scratch or temporary directory when none is given. The defaults
are the bundle root (the nearest directory above the member that holds `bundle_manifest.json`) for the
analysis and system directory tokens, `sys.executable` for `{hhemt_python}`, the reproducer's home
directory for `{hhemt_home}` and the installed package directory for `{hhemt_package_dir}`.
`{hhemt_toolkit_dir}` has a default only when the installed package sits under `src/`. A token with
neither a supplied value nor a default (a directory token on a member outside any bundle, say) is refused
rather than left in place. `{hhemt_conda_env}`, `{hhemt_user}`, `{hhemt_slurm_account}` and
`{hhemt_producer_path}` stay as the token unless you supply a value.

??? note "What the zero-user-info check does not cover"
    - **Binary members are neither rewritten nor scanned.** The emit copies zarr chunks and declared
      binary inputs (a DEM, a land-use raster, a weather NetCDF) through unchanged, so a producer value
      inside one survives. No other binary member is carried: the baseline report is kept only as
      `bundle_baseline/analysis_report.html`, and the archive version of it that used to nest inside the
      bundle, and carried producer values in its compressed entries, is no longer written.
    - **The second layer shares its source with the rewrite.** Both read the same configs, so a value
      that is in neither those configs nor a generic pattern is caught only by your private list.
    - **A private list is only as good as its owner.** The check can find only the entries it is given,
      so pointing the variable at a list you wrote yourself certifies nothing beyond that list.
    - **History is not rewritten.** The public repository keeps any value that was committed before the
      check existed.

    On the consume side, `reprex()` repeats the generic patterns and your private list over the bundle
    you hold and reports them in `result.zero_user_info_leaks` as one entry whose text names every match, without failing. A consumer holds
    no producer configs, so the second layer does not run there. When the variable is set to something
    unusable, the entry is the diagnosis rather than a match, so read the entries before treating a
    non-empty list as a list of leaks.
