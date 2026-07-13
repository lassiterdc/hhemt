# Publish a dataset and fetch inputs by DOI/PID

This guide covers the two ends of the reproducibility loop: **publishing** a consolidated
analysis to a DOI-minting repository (Zenodo or HydroShare), and **fetching** a case
study's heavy inputs back by DOI/PID. Both are opt-in — publishing is never triggered by
`analysis.run()` or `submit_workflow()`.

## Before you publish

`analysis.publish()` deposits the *analysis-directory set* — the consolidated
`analysis_datatree.zarr`, the co-located `ro-crate-metadata.json` provenance sidecar, and
the two configs (`cfg_analysis.yaml` + `cfg_system.yaml`). So the analysis must already be
**consolidated** (run through `reprocess(start_with="consolidate")` or a full `run()`), or
publish has nothing to deposit and no crate to read the license from.

The dataset license is read back from the crate sidecar — it is baked in at consolidation
(default `CC0-1.0`; set `analysis_config.dataset_license: CC-BY-NC-4.0` before consolidating
to choose the other vocab entry). Publishing does **not** re-stamp the archived license.

## Provide credentials (environment variables)

Live deposits require host credentials, supplied via environment variables so they never
land in a config file:

```bash
# Zenodo (or the sandbox: set HHEMT_ZENODO_BASE_URL=https://sandbox.zenodo.org)
export HHEMT_ZENODO_TOKEN=<your-zenodo-personal-access-token>

# HydroShare
export HHEMT_HYDROSHARE_USERNAME=<your-hydroshare-username>
export HHEMT_HYDROSHARE_PASSWORD=<your-hydroshare-password>
```

## Publish to Zenodo

Zenodo mints the DOI programmatically (reserve-DOI → deposit → publish):

```python
result = analysis.publish(
    target="zenodo",
    software_doi="10.5281/zenodo.SOFTWARE",   # optional: links data -> software (IsCompiledBy)
)
print(result["data_doi"], result["record_url"])
# {"target": "zenodo", "data_doi": ..., "software_doi": ..., "record_url": ...}
```

`software_doi` is optional; when given, the deposit records a DataCite `IsCompiledBy`
`relatedIdentifier` (data → the software that produced it) and backfills the reciprocal
edge onto the software record.

To assert (not re-stamp) the license you expect, pass `override_dataset_license`. If it
disagrees with the license baked into the crate, publish raises `PublishError` and directs
you to set `analysis_config.dataset_license` and re-consolidate — it will not silently
publish a mismatched license:

```python
analysis.publish(target="zenodo", override_dataset_license="CC0-1.0")
```

## Publish to HydroShare

HydroShare is a two-step flow — hsclient (v1.1.6) has no programmatic DOI mint. `publish()`
creates the resource, uploads the deposit set, sets it public, then **stops and returns a
manual instruction**:

```python
result = analysis.publish(target="hydroshare")
print(result["manual_step"])   # open result["record_url"] and use 'Publish' in the web UI
```

Open `result["record_url"]`, click **Publish** in the HydroShare web UI to mint the DOI,
then re-run `publish` with the minted `software_doi` if you want the reciprocal edge.

## Publish a sensitivity analysis

A sensitivity analysis deposits its **master** tree the same way:

```python
sensitivity.publish(target="zenodo")   # deposits sensitivity_datatree.zarr + master sidecar
```

## Fetch a case study's inputs by DOI/PID

A case study's `case.yaml` is a provenance descriptor of *remote* heavy inputs. Point it at
a durable deposit with `host` + `doi` (or `pid`):

```yaml
# test_data/<case_name>/case.yaml
case_name: norfolk_coastal_flooding
res_identifier: <32-hex-hydroshare-resource-id>
host: zenodo                       # or: hydroshare
doi: '10.5281/zenodo.1234567'      # host='zenodo' requires a doi OR pid
```

Then load the case study — the toolkit dispatches on `host`, fetches over anonymous-first
HTTPS, and verifies every file against the `manifest` sha256 map:

```python
from hhemt.experiments import TRITON_SWMM_experiment

experiment = TRITON_SWMM_experiment.from_case_study(
    case_name="norfolk_coastal_flooding",
    download_if_exists=False,   # set True to re-download even if the data is already local
)
```

The fetch is host-agnostic on verification: the streaming 1 MiB-chunk sha256 check is
byte-identical regardless of `host`, and Globus (when used) stays transport-only.
