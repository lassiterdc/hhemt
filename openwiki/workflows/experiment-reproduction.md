---
type: workflow
title: Case studies and DOI reproduction
description: Distinguish template-based case acquisition from runnable-bundle ingest, including input materialization, schema checks, external HPC profiles, image rebuilding, and reprex validation limits.
tags: [reproducibility, doi, case-study, bundle, container]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-7a81d87b7263dea117027463
    resource: repo://docs/how-to/doi-roundtrip-e2e.md
  - id: openwiki-source-a894331b70fe7231b86aa5ca
    resource: repo://src/hhemt/bundle/_archive.py
  - id: openwiki-source-2a8f0de7a696992562ec0be3
    resource: repo://src/hhemt/bundle/_reprex_gate.py
  - id: openwiki-source-e94703da1784a74c27edf7bb
    resource: repo://src/hhemt/bundle/_reprex.py
  - id: openwiki-source-4acf236050b6b0f83a798bc5
    resource: repo://src/hhemt/experiments.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Case studies and DOI reproduction

## Two acquisition paths

`TRITON_SWMM_experiment.from_case_study` starts from case metadata and system/analysis templates. It resolves data paths, acquires the case's HydroShare/Zenodo payload, fills the configurations, and constructs an experiment. The caller supplies a required `hhemt_sha` for the generated analysis; case selection is not restoration of a previously rendered bundle. Existing case data can be reused unless re-download is requested.

`from_doi` instead fetches a published runnable reprex bundle, extracts it, reconstitutes its config pair and declared inputs, resolves target image requirements, and constructs an experiment. `from_bundle_zip` joins the same post-extraction path for a local archive. Ingest does not itself submit the ensemble, although missing container images can trigger a foreground build DAG.

## DOI/PID ingest sequence

The API requires a DOI or PID and explicitly accepts only `zenodo` or `hydroshare`. It uses a supplied target directory or creates a temporary one. The fetch path requires a single candidate bundle ZIP and optionally verifies `expected_sha256` before extraction.

The post-extraction gate runs in this order:

1. `Bundle.from_directory` checks the manifest and exact local bundle schema compatibility.
2. The crate must identify a runnable workflow through `mainEntity`.
3. System and analysis configurations are reconstituted with bundle-relative inputs rebased to the extracted root. Software defaults to a writable `software/` below that root unless overridden.
4. Declared input paths must materialize before model construction. Missing by-reference inputs with `contentUrl` are downloaded and hash-checked; restricted/unfetchable entries fail with citation and placement instructions rather than substituting data.
5. An explicitly supplied reproducer HPC config is resolved; it is never inferred from an environment variable.
6. Container images are resolved/rebuilt and a derived HPC config points at the resolved SIF root.
7. The experiment is constructed and, when validation is enabled, `analysis.validate().raise_if_invalid()` runs.

These are ingest guards, not a claim that every low-level archive or bundle read materializes missing inputs. A render-only bundle reader has a different responsibility.

Operator forms, with real identifiers and local profile substituted:

```sh
hhemt ingest --doi DOI --host zenodo --target-dir TARGET --hpc-system-config YOUR_HPC.yaml
hhemt run --system-config RECONSTITUTED_SYSTEM.yaml --analysis-config RECONSTITUTED_ANALYSIS.yaml --hpc-system-config YOUR_HPC.yaml
```

For sandbox deposits, use the same `HHEMT_ZENODO_BASE_URL` on deposit and fetch. HydroShare DOI minting is a web-UI step after upload. The operator guide notes that ingest's printed run suggestion omits the HPC-config flag; carry it explicitly for batch execution.

## Trust and extraction limits

Only ingest deposits you trust: later execution derives shell commands from fetched configurations. A matching SHA-256 pins transferred bytes, not the authority or safety of those bytes. Obtain a pin through an independent trusted channel when integrity matters.

The reprex extractor replaces an existing extraction destination, but refuses a destination containing the ZIP, the user's home, or the current working directory/its ancestors. The underlying threaded extractor uses `ZipFile.extract` sanitization, groups archive members that sanitize to the same target to preserve archive order, and propagates extraction failures. This is not a sandbox or authenticity check. Its optional missing-or-resized optimization trusts file size and preserves same-size edits; do not confuse that with digest verification.

## The reproducer owns the HPC profile

A real `hpc_system_config` is external target-local input. A scrubbed template is guidance, not the producer's allocation or a usable substitute for the target cluster's partitions. Container ingest refuses a missing profile or absent `sif_manifests` and checks that carried `(mpi_family, accel, gpu_hardware)` identities cover every required matrix partition before attempting a build.

For each carried image, ingest retains a file whose adjacent manifest records the carried digest. Otherwise it plans a rebuild at the carried identity. That rebuild requires the running checkout's commit to match the carried `hhemt_sha`, an explicit SIF build config, and no live producer sentinel for the identity. It runs the standard build DAG in the foreground, refuses nonzero exit or missing resulting manifests, and writes a derived HPC configuration without changing the user's original file.

**Presence/manifest reuse is weaker than byte verification:** the ingest reuse branch compares the manifest digest, whereas reprex's SIF check hashes the image itself. Matching image bytes still do not establish host MPI/driver/hardware equivalence or numerical identity.

## What reprex check proves

```sh
hhemt reprex check --bundle BUNDLE_DIR --reprex-config YOUR_REPREX.yaml --hpc-system-config YOUR_HPC.yaml
```

The check finds a crate SIF reference and requires its digest to match a locally resolved image. Missing or mismatched referenced images raise `ProcessingError`; no signature check runs (`sif_signature_ok` is always `None`). No reference is not sufficient to infer native execution: a manifest announcing container mode without a reference produces `sif_verified=None`; absent/unreadable container indicators can still give the vacuous `True` branch.

Next, the check reconstitutes configs, optionally redirects only output `analysis_dir` to scratch, overlays target partition selectors, and runs preflight. If full preflight raises, it falls back to isolated per-member row-cap checking. `runnable` means **no extracted sensitivity row-cap problem pairs**, not successful simulation, complete validation, or identical outputs. Partition amendments are `validated` only when a deterministic target lookup exists; resource replacements remain advisory decisions.

The consume-side producer-information scan is informational. Emit-time scanning is a hard gate using generic patterns, producer config/login values, and an optional private blocklist. Current scanner code handles text, decoded text arrays, and supported binary/container scanning; do not rely on older guide prose claiming that all binary members are outside the scan. An unusable configured private list fails the gate; an absent list means that layer was not checked. No scan establishes safety for executing a fetched workflow.

## Related pages

- [Reporting and portable bundles](reporting-and-portable-bundles.md): emission and render-only use.
- [Solver builds and containers](../integrations/solver-builds-and-containers.md): image identity and build machinery.
- [Provenance and publication](../integrations/provenance-and-publication.md): deposit and evidence ownership.
- [Quickstart](../quickstart.md): task routing.

No deposit, remote acquisition, image build, reprex check, or solver execution was performed for this page.
