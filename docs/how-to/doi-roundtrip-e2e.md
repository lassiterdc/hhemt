# Reproduce an experiment by DOI (operator round-trip)

This runbook mints a DOI for a runnable reprex bundle, then fetches and runs it on another
machine — the end-to-end reproducibility round-trip.

The bundle is **self-contained by default**: it carries every input the experiment declares,
so a consumer can run it from scratch. Excluding an input is a governed opt-out for data you
cannot redistribute — see [Excluding an input](#excluding-an-input-the-governed-opt-out).

## Prerequisites

- A completed analysis with a rendered report (`analysis.render_report()` must have run at
  least once, so the provenance sidecars the bundle is built from exist).
- **Deposit credentials, under exactly these names.** These are the variables the code reads
  (`publishing.py::_require_env`); a wrong name fails with
  `PublishError: missing required credential env var …`:

  | Target | Environment variables |
  |---|---|
  | Zenodo | `HHEMT_ZENODO_TOKEN` (optionally `HHEMT_ZENODO_BASE_URL` to target the sandbox) |
  | HydroShare | `HHEMT_HYDROSHARE_USERNAME` and `HHEMT_HYDROSHARE_PASSWORD` |

## 1. Deposit the reprex bundle (mint the DOI)

```bash
export HHEMT_ZENODO_TOKEN={your-token}
```

```python
from hhemt import Toolkit

tk = Toolkit.from_configs("system.yaml", "analysis.yaml")
result = tk.analysis.publish_reprex_bundle(target="zenodo")
print(result["data_doi"], result["record_url"])
```

Before uploading, a size validator measures the bundle against the target's **documented**
limits (Zenodo 50 GB/record; HydroShare 20 GB default account quota) and warns — with the
exact overflow and a remediation menu — if it will not fit. It does not block: neither
platform exposes a queryable quota, so the deposit is attempted and a live storage rejection
is caught and reframed with the real numbers.

## 2. Fetch and reconstitute on another machine

```bash
hhemt ingest --doi {minted-doi} --host zenodo
```

If you minted against the **sandbox** in step 1, set the same base URL on the fetch side —
the ingest resolves the Zenodo host from `HHEMT_ZENODO_BASE_URL`, so a sandbox DOI resolves
against the sandbox instead of 404ing on production:

```bash
export HHEMT_ZENODO_BASE_URL=https://sandbox.zenodo.org
hhemt ingest --doi {minted-doi} --host zenodo
```

## 3. Run

```bash
hhemt run --system-config {reconstituted-system.yaml} --analysis-config {reconstituted-analysis.yaml}
```

## Excluding an input (the governed opt-out)

Use this when an input is too large to deposit, or when you are **not permitted to
redistribute it** (licensed, proprietary, or otherwise restricted data). The excluded input
is carried *by reference* instead: the bundle records where it came from, what it hashes to,
and how to obtain it.

**This is a three-step sequence, and the order matters.** The excluded input must already
have a durable record before you author the exclude-config — the toolkit has no per-file
deposit helper and cannot mint one for you.

### Step 1 — Give the input a durable record

- **Data you own**: deposit it as its own Zenodo/HydroShare resource. Note its
  direct-download URL and its DOI.
- **Third-party or licensed data**: locate its original source. **The toolkit must not
  redeposit it**, and you should not either.

### Step 2 — Author the exclude-config

`hhemt bundle --list-excludable` prints every input you may opt out of, with the
reproducibility cost of each. It needs no configs — run it before you have written anything.

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

`citation` is **required** for every excluded input. `contentUrl` is what decides what a
consumer experiences:

- **`contentUrl` present** — `hhemt ingest` downloads the file automatically and verifies it
  against the recorded sha256.
- **`contentUrl` omitted** — ingest **fails closed** and prints your citation, the landing
  page, the expected sha256, and the exact path to place the file at. **This is the correct
  outcome for restricted data, not a degraded one.** The bundle still carries complete,
  public, machine-readable metadata about the input; only the bytes are withheld, which is
  what the license requires.

### Step 3 — Emit and deposit

```python
tk.analysis.publish_reprex_bundle(target="zenodo", exclude_config="bundle_exclude.yaml")
```

## Notes and caveats

- **A green Zenodo run is not a green HydroShare run.** `hsclient` v1.1.6 cannot mint a DOI
  programmatically, so the HydroShare path publishes the resource and then *stops*, printing
  a manual web-UI DOI-mint instruction. The "fetch back by DOI" leg is therefore **not** a
  single automated round-trip on that host — mint the DOI in the web UI first, then ingest.
- **Deposit record ids are yours, not the repo's.** Supply them via environment variables
  (`HHEMT_E2E_ZENODO_RECORD` / `HHEMT_E2E_HYDROSHARE_RESOURCE`); never commit them.
- **Match the toolkit version.** The bundle-schema guard is exact-match: install the version
  named in the crate's `SoftwareApplication.softwareVersion`, or `hhemt ingest` fails with a
  `BundleSchemaError`. Re-emit from source if you cannot.
- **Ingesting a DOI and running it executes shell derived from the fetched config.** Ingest
  only deposits you trust. `hhemt ingest --sha256 {digest}` pins the fetched bundle's
  integrity.
