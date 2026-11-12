# DOI round-trip for operators

## Goal

Mint a DOI for a runnable reprex bundle. Then fetch and run the bundle on another machine as the end-to-end reproducibility round-trip. The bundle is **self-contained by default** and carries every input that the experiment declares, so a consumer can run the bundle from scratch. [Excluding an input](#excluding-an-input) covers the governed opt-out for data you may not redistribute.

## Prerequisites

- A working [installation](installation.md).
- A completed analysis with a rendered report. Render the report at least once, so that the provenance sidecars the bundle is built from exist.
- **Deposit credentials** in the environment variables that [Provide credentials as environment variables](publishing.md#provide-credentials-as-environment-variables) names.

## Key references

- [Publish a dataset and fetch inputs by DOI/PID](publishing.md) describes the deposit units and the credentials.
- [Reprex round-trip](reprex-roundtrip.md) describes the bundle and the check on another cluster.

## 1. Deposit the reprex bundle to mint the DOI

```bash
export HHEMT_ZENODO_TOKEN={your-token}
```

Deposit the bundle from the experiment definition. The command prints the deposit record with the minted DOI under `data_doi` and the record address under `record_url`.

```bash
hhemt publish --experiment {experiment-definition-dir} --cluster {cluster} --target zenodo --reprex-bundle
```

??? note "Bundle larger than the target limit"
    Before uploading, a size validator compares the bundle with the **documented** limits of the target, Zenodo 50 GB per record and HydroShare 20 GB default account quota, and warns on overflow. The warning states the exact overflow and a menu of remedies. The validator never blocks, because neither platform exposes a queryable quota. The deposit is attempted, and the toolkit catches a live storage rejection and restates the rejection with the real numbers.

## 2. Fetch and reconstitute on another machine

```bash
hhemt ingest --doi {minted-doi} --host zenodo --target-dir {target} --hpc-system-config {your-cluster-hpc-system-config.yaml}
```

A reconstituted analysis that dispatches with `batch_job` needs the `hpc_system_config` of your own cluster, so pass the config here. Without the config the ingest stops at preflight with a configuration error, after the download has happened.

Deposits made to the **sandbox** in step 1 need the same base URL on the fetch side. [Choose a deposit unit](publishing.md#choose-a-deposit-unit-data-doi-or-runnable-doi) explains the sandbox and the behavior of `hhemt ingest` with the variable.

```bash
export HHEMT_ZENODO_BASE_URL=https://sandbox.zenodo.org
hhemt ingest --doi {minted-doi} --host zenodo --target-dir {target} --hpc-system-config {your-cluster-hpc-system-config.yaml}
```

## 3. Run

```bash
hhemt run --system-config {reconstituted-system.yaml} --analysis-config {reconstituted-analysis.yaml} --hpc-system-config {your-cluster-hpc-system-config.yaml}
```

The `Run it with:` line that ingest prints omits `--hpc-system-config`. Add the flag. Without the flag the analysis is refused for the same reason.

!!! example "Verified against a live Zenodo sandbox deposit"
    The command forms on this page were run against a Zenodo **sandbox** deposit of a CPU sensitivity analysis of compute configurations, with one data deposit and one runnable reprex-bundle deposit. The reprex-bundle deposit is larger because the bundle carries the declared inputs. [Choose a deposit unit](publishing.md#choose-a-deposit-unit-data-doi-or-runnable-doi) explains the sandbox and the sandbox DOI prefix. The check did not rerun the sweep from the deposit and ran only `hhemt run --dry-run` on the reconstituted configs.

## Excluding an input

Use the exclusion when an input is too large to deposit, or when you may **not redistribute the input** because the data is licensed, proprietary, or otherwise restricted. The bundle carries the excluded input *by reference*. The bundle records the origin of the input, the hash of the input, and the way to obtain the input.

Follow the steps in order. Each excluded input needs a durable record before you author the exclude-config. The toolkit offers no per-file deposit helper to mint a record.

### Step 1: give the input a durable record

- **Data you own.** Deposit the data as a separate Zenodo or HydroShare resource, and note the direct-download URL and the DOI.
- **Third-party or licensed data.** Locate the original source. **The toolkit must not redeposit the data**, and neither should you.

### Step 2: author the exclude-config

`hhemt bundle --list-excludable` prints every input you may opt out of, with the reproducibility cost of each. The command needs no configs and runs before you write any file.

```yaml
# bundle_exclude.yaml
exclusions:
  DEM_fullres:
    citation: "City of Norfolk (2019). 1 m LiDAR DEM. Deposited by the analysis operator."
    contentUrl: "https://zenodo.org/api/records/7654321/files/dem.tif/content"
    identifier: "10.5281/zenodo.7654321"
    url: "https://doi.org/10.5281/zenodo.7654321"

  SWMM_hydraulics:
    # No contentUrl: this network is licensed and may not be redistributed.
    citation: >-
      Licensed municipal sewer network. Not redistributable. Request from the County GIS
      office at the URL below; place the received file at the path the error message names.
    url: "https://gis.example-county.gov/data-request"
```

`citation` is **required** for every excluded input. `contentUrl` decides the experience of a consumer.

- **`contentUrl` present.** `hhemt ingest` downloads the file automatically and verifies the file against the recorded sha256.
- **`contentUrl` omitted.** Ingest **fails closed** and prints your citation, the landing page, the expected sha256, and the path to place the file at. **Failing closed is the correct outcome for restricted data.** The bundle still carries complete, public, machine-readable metadata about the input, and the license requires only that the bytes stay withheld.

### Step 3: emit and deposit

```bash
hhemt publish --experiment {experiment-definition-dir} --cluster {cluster} --target zenodo --reprex-bundle --exclude-config bundle_exclude.yaml
```

## Notes and caveats

- **HydroShare mints the DOI in the web UI.** Mint the DOI there before the ingest step. `hhemt publish` stops after the upload, as [Publish to HydroShare](publishing.md#publish-to-hydroshare) describes.
- **Deposit record ids belong to you.** Supply them through the environment variables `HHEMT_E2E_ZENODO_RECORD` and `HHEMT_E2E_HYDROSHARE_RESOURCE`, and never commit them.
- **Match the toolkit version.** Install the version named in `SoftwareApplication.softwareVersion` of the crate. A different version fails with a `BundleSchemaError`. [Reproducibility boundaries](../explanation/limitations.md#reproducibility-boundaries) explains the error.
- **Ingesting a DOI and running the ingested analysis executes shell derived from the fetched config.** Ingest only deposits you trust. `hhemt ingest --sha256 {digest}` checks the fetched bundle zip against the SHA-256 and refuses a mismatch. The check verifies transfer integrity and authenticates no deposit, so a match leaves an untrusted deposit unsafe to run. Take the digest from the depositor by a channel other than the record, or compute the digest with `sha256sum` on a copy you inspected. Zenodo shows an `md5:` checksum for each file and no SHA-256, so the record page supplies no digest.
