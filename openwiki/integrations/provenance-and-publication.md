---
type: integration
title: Provenance, licenses, publication, and transfers
description: How consolidated analyses acquire RO-Crate provenance, how publication consumes archived licensing, and how Globus transports results.
tags: [provenance, ro-crate, publication, licensing, globus]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-20T07:08:41.116Z
sources:
  - id: openwiki-source-36d65f12e05a99a16e66e8db
    resource: repo://src/hhemt/globus_transfer.py
  - id: openwiki-source-cdb06fae94a673beaa807df4
    resource: repo://src/hhemt/metadata.py
  - id: openwiki-source-307087f14bd0c7a3cde6ea8d
    resource: repo://src/hhemt/processing_analysis.py
  - id: openwiki-source-5b9f5518b8c292c67f0c089e
    resource: repo://src/hhemt/provenance.py
  - id: openwiki-source-769d371100f2111b21c29fd2
    resource: repo://src/hhemt/publishing.py
generated: { by: "omp", at: "2026-11-20T07:08:41.116Z" }
---

# Provenance, licenses, publication, and transfers

## Ownership and artifact flow

Consolidation owns provenance emission; publication consumes it rather than retroactively changing the archive. The analysis consolidation path gathers the variables and attributes actually emitted, calls `emit_provenance`, embeds its core on the DataTree, writes the summary store, and writes the full graph beside it as `ro-crate-metadata.json`. This keeps variable descriptions aligned with the stored products, rather than merely the static vocabulary. See [output data model](../concepts/output-data-model.md).

`emit_provenance` reads the analysis log as its source of record. It creates per-member/event/model `CreateAction` entities linking input objects, output results, the toolkit, and an optional SIF identity. Submission time and node come from persisted log fields, not a fresh emission timestamp. The function returns two serializations: an allowlisted embedded core and the full sidecar graph. The SIF is optional; the emitter does not imply that every analysis ran in a container.

Canonical JSON-LD fixes context, entity ordering, and key ordering. The core excludes volatile provenance properties. Sidecar writes compare parsed content and atomically replace only changed content, preserving bytes and mtime on an unchanged re-emission. This is deterministic serialization of recorded inputs, not a claim that different executions have identical provenance.

## License is an archived contract

`publish_analysis` requires a recognizable license already declared by the crate. It normalizes configured and archived licenses to records, including SPDX vocabulary licenses and custom details-file licenses, and refuses mismatches before depositing. `override_dataset_license` asserts the expected archived SPDX identifier; it is not a rewrite operation.

If the configuration was accidentally edited, restore the terms the archive already carries. For an intentional license change, set the new terms and regenerate consolidation with `reprocess(start_with='consolidate', regenerate_existing=True)` before publishing. The failure message explicitly requires regeneration rather than assuming that a normal consolidation revisit rewrites an existing store.

## Deposit units and target boundaries

`publish_analysis` selects the analysis-directory deposit set by default. It also accepts a freshly emitted reprex-bundle ZIP or an explicit artifact `Path`. Staging lives under `TMPDIR` when supplied, otherwise under the analysis directory. Size validation reports documented platform limits and warns without blocking; a subsequent storage-related `PublishError` can be reframed using the measured size. These are documented limits, not a live quota query.

| Target | Implemented sequence | Result and limits |
|---|---|---|
| Zenodo | Create draft → set native InvenioRDM metadata → initialize/upload/commit each file → publish → read minted DOI | Uses `HHEMT_ZENODO_TOKEN`; `HHEMT_ZENODO_BASE_URL` selects another host such as the sandbox. Directories are zipped for upload. |
| HydroShare | Create resource → upload → set rights and optional software relation → save → make public | Uses `HHEMT_HYDROSHARE_USERNAME` and `HHEMT_HYDROSHARE_PASSWORD`; returns `data_doi: null` and instructions to mint a DOI in the web UI. |

Zenodo can record the data-to-software relation and attempts a reciprocal software-record update after publication. That update is advisory, not a condition for the already-published data record. HydroShare creates a new resource on every call: do not rerun publication to finish DOI minting or add a relation. Supplied creators are applied by Zenodo; HydroShare warns that it does not map that argument and uses account author metadata instead.

These are implementation contracts, not live-service verification. No remote deposit, authentication, DOI registration, or solver execution was exercised while preparing this page. Consult the maintained [publishing guide](../../docs/how-to/publishing.md) for operator command forms and [experiment reproduction](../workflows/experiment-reproduction.md) for runnable deposits.

## Globus is transport, not publication

`GlobusTransferManager.transfer` puts all configured source/destination items into one task, with optional file and directory exclusions. A policy-related authorization error triggers one forced reauthentication using the domains required by the endpoint, then one resubmission. Other transfer API errors propagate.

`wait` polls until success; failed or cancelled tasks raise `GlobusTransferError`, and an optional local timeout raises `TimeoutError`. OAuth tokens are stored in `~/.globus_tokens.json`, with refresh-token support and mapped-collection data-access scopes. Initial authentication prints an authorize URL and requests the returned code. Transport completion does not mint a DOI or establish scientific reproducibility.

For the archive transported or deposited, see [reporting and portable bundles](../workflows/reporting-and-portable-bundles.md).

## Source map

- `src/hhemt/processing_analysis.py`: consolidation emission and persistence boundary.
- `src/hhemt/provenance.py`, `src/hhemt/metadata.py`: graph construction, partitioning, canonicalization, and sidecar writes.
- `src/hhemt/publishing.py`: license refusal, deposit selection, target adapters, and size reporting.
- `src/hhemt/globus_transfer.py`, `src/hhemt/config/globus.py`: transfer task lifecycle and configuration.
