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

# Sensitivity master (the primary reprex surface, because the per-(member_id, column)
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
An experiment bundle is no longer an exception — `ContainerRef.def_recipe` was retired,
and a descriptor still carrying that key is refused by name when it loads. The family
recipes now ship as package data under `src/hhemt/sif/recipes/`; copy the one your
architecture needs and pass its path. A native analysis ignores the argument.

The emitted bundle root carries the minimal runnable set:

- `cfg_system.yaml` + `cfg_analysis.yaml`: path-scrubbed configs (your machine-local
  paths are nulled; experiment fields are preserved verbatim).
- `reprex_config.yaml`: a **template** the reproducer fills with *their* values.
- `hpc_system_config.template.yaml`: a scrubbed HPC profile with `{your-allocation}`
  placeholders (never your real account).
- `Snakefile.source`: the generated workflow, typed as the crate's `mainEntity`.
- `ro-crate-metadata.json`: the Workflow-Run-Crate metadata; heavy inputs and (for a
  container run) the SIF are recorded **by reference** with their `sha256`.

## Distribute by DOI (publish → ingest → run)

Instead of handing someone the bundle directory, you can mint a **runnable-DOI**: publish the
reprex bundle to a DOI-minting repository, and a reproducer fetches, reconstitutes, and runs
it from the DOI alone.

```python
# Producer: deposit the bundle and mint the DOI (Zenodo shown; see publishing.md for credentials)
result = analysis.publish_reprex_bundle(target="zenodo")
print(result["data_doi"])          # the runnable-DOI
```

```bash
# Reproducer: fetch + reconstitute by DOI, then run the round-trip check below
hhemt ingest --doi {minted-doi} --host zenodo --sha256 {digest}
```

`hhemt ingest` fetches the bundle, schema-guards it (an exact `BUNDLE_SCHEMA_VERSION` match),
reads the crate `mainEntity`, and reconstitutes the runnable config pair. The round-trip
check below then applies on the reproducer's cluster. For the full operator walkthrough
(credentials, the sandbox, excluded inputs, the HydroShare manual-DOI caveat) see
[the DOI round-trip runbook](doi-roundtrip-e2e.md).

> A sandbox-minted DOI requires `HHEMT_ZENODO_BASE_URL=https://sandbox.zenodo.org` on the
> reproducer side too. The fetch resolves the Zenodo host from the same env var as the deposit,
> so a sandbox record resolves against the sandbox rather than 404ing on production.

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
   under `reprex_config.sif_root`, establishing that your image file is byte-identical to
   the producer's. A mismatch — or no manifest under `sif_root` carrying that digest —
   raises `ProcessingError` before any validation runs. **The toolkit does not PGP-sign
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
   | `zero_user_info_leaks` | informational. One entry per producer token still present in the bundle; on an install where no blocklist carrier is reachable it instead carries the gate's own diagnosis (see below) |

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

**Treat a reprex bundle as not-yet-guaranteed-private before sharing it widely.**
The design goal is that a shared bundle carries **zero** producer-specific
information, but that is not yet enforced at emit time.

??? note "What the zero-user-info gate does and does not currently catch"
    The property is proven by a positive blocklist scan over the emitted tree, and
    that gate currently runs **consume-side and informationally**: `reprex()`
    populates `result.zero_user_info_leaks` but does not fail. Hard emit-time
    enforcement is deferred to an emit-hardening pass. The producer's absolute paths
    still leak through `bundle_manifest.json`, harvested SWMM `.inp` `FILE`
    references, and `validation_report.json`, all surfaces the config-field scrub
    does not yet cover.

    Four further things about the gate. Its blocklist carrier comes from
    `$HHEMT_REPREX_BLOCKLIST` when that variable is set, and from the source checkout's
    `scripts/reprex_blocklist.txt` only when it is not. The override is terminal rather
    than the first step of a fallback chain: if it is set and does not point at an
    existing file, resolution stops there and the source-checkout copy is never
    consulted. There is deliberately no packaged-data copy, because a wheel could ship
    either the real private tokens or synthetic ones that would certify a pass it never
    performed. When no carrier is reachable (the usual case for a PyPI install), the gate
    raises a diagnosed `ProcessingError` which `reprex()` catches and records: the
    diagnosis lands in `result.zero_user_info_leaks` as a finding rather than crashing
    the round-trip. So a non-empty `zero_user_info_leaks` is not automatically a list of
    leaks. Read the entries. Note also that the gate is producer-local by design. It can
    only find tokens it was given, so pointing the override at a list you wrote yourself
    buys a pass the gate never earned.
