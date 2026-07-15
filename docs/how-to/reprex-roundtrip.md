# Reproduce an analysis on a foreign HPC system (reprex round-trip)

A **reprex bundle** is a render bundle upgraded into a round-trippable
Workflow-Run-Crate: a second person, given only the bundle, can validate whether your
analysis will run on *their* cluster and get an exact list of what they must supply or
amend. This guide covers both ends: **emitting** a reprex bundle, and **running the
round-trip check** on a foreign system.

Like publishing, this is opt-in — a reprex bundle is never emitted by `analysis.run()`
or `submit_workflow()`.

## Emit a reprex bundle

Emit from a completed, **rendered** analysis (the bundle harvests the sources declared
during `render_report()`, so run it first):

```python
# Single analysis
bundle_dir = analysis.reprex_bundle()

# Sensitivity master (the primary reprex surface — the per-(sa_id, column)
# problem-pair report is intrinsically a sensitivity concept)
bundle_dir = sensitivity.reprex_bundle()
```

`reprex_bundle()` returns a **directory** (it emits the bundle zip, then extracts it to a
sibling directory) so the round-trip below can consume it directly. Pass
`output_path=Path(...)` to control where the bundle is written.

The emitted bundle root carries the minimal runnable set:

- `cfg_system.yaml` + `cfg_analysis.yaml` — path-scrubbed configs (your machine-local
  paths are nulled; experiment fields are preserved verbatim).
- `reprex_config.yaml` — a **template** the reproducer fills with *their* values.
- `hpc_system_config.template.yaml` — a scrubbed HPC profile with `{your-allocation}`
  placeholders (never your real account).
- `Snakefile.source` — the generated workflow, typed as the crate's `mainEntity`.
- `ro-crate-metadata.json` — the Workflow-Run-Crate metadata; heavy inputs and (for a
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
reads the crate `mainEntity`, and reconstitutes the runnable config pair — the round-trip
check below then applies on the reproducer's cluster. For the full operator walkthrough
(credentials, the sandbox, excluded inputs, the HydroShare manual-DOI caveat) see
[the DOI round-trip runbook](doi-roundtrip-e2e.md).

> A sandbox-minted DOI requires `HHEMT_ZENODO_BASE_URL=https://sandbox.zenodo.org` on the
> reproducer side too — the fetch resolves the Zenodo host from the same env var as the deposit,
> so a sandbox record resolves against the sandbox rather than 404ing on production.

## What the reproducer supplies

The reproducer fills two things with *their own* system's values:

1. A `reprex_config` — the minimal host-local field set:

   ```python
   from hhemt.config.reprex_config import reprex_config

   my_reprex = reprex_config(
       default_account="my-alloc",              # your HPC allocation
       sif_path="/scratch/my-alloc/tritonswmm.sif",  # where YOU fetched the SIF
       target_ensemble_partition="gpu-a100",    # your partition for the ensemble sims
       # login_node / scratch_dir / target_setup_and_analysis_processing_partition optional
   )
   ```

2. Their own `hpc_system_config` (the target profile — partition caps, container spec).
   See [HPC-profile setup](hpc-profile-setup.md).

## Run the round-trip check

```python
from hhemt.bundle import Bundle

result = Bundle.from_directory(bundle_dir).reprex(my_reprex, my_hpc_profile)
```

`reprex()` does three things, in order:

1. **Verify the SIF.** If the crate references a SIF (a container run), the digest is a
   **mandatory, fail-closed** `sha256` match against `reprex_config.sif_path` — a mismatch
   raises `ProcessingError` before any validation runs. A best-effort `apptainer verify`
   PGP check runs too (`result.sif_signature_ok` is `None` when `apptainer` or the
   producer key is unavailable — a warning, not a failure). A **native run** records no
   SIF in its crate, so `result.sif_reference_present` is `False` and verification is a
   vacuous pass.

2. **Re-aim preflight at your profile.** Validation is re-run with your partition
   selectors overlaid, so the report reflects *your* cluster's caps.

3. **Report problems and amendments.** `result` is a `ReprexResult`:

   | Field | Meaning |
   |---|---|
   | `runnable` | `True` when no sensitivity row exceeds a target partition cap |
   | `problem_pairs` | one `ValidationIssue` per `(sa_id, column)` that exceeds a cap — the exact rows/resources to reduce |
   | `amendments` | per-field experiment amendments, each labelled `validated` (a deterministic target-partition lookup pins the value) or `advisory` (you must decide, with a named reason) |
   | `sif_reference_present` / `sif_verified` / `sif_signature_ok` | SIF verification outcome (see step 1) |
   | `zero_user_info_leaks` | informational: producer tokens still present in the bundle (see below) |

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

The reprex design goal is that a shared bundle carries **zero** producer-specific
information — proven by a positive blocklist scan over the emitted tree. That gate
currently runs **consume-side and informationally**: `reprex()` populates
`result.zero_user_info_leaks` but does not fail. Hard emit-time enforcement is deferred
to an emit-hardening pass (the producer's absolute paths still leak through
`bundle_manifest.json`, harvested SWMM `.inp` `FILE` references, and
`validation_report.json` — surfaces the config-field scrub does not yet cover). Until
that lands, treat a reprex bundle as not-yet-guaranteed-private before sharing it widely.
