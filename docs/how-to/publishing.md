# Publish a dataset and fetch inputs by DOI/PID

This guide covers the two ends of the reproducibility loop: **publishing** a consolidated
analysis to a DOI-minting repository (Zenodo or HydroShare), and **fetching** a case
study's heavy inputs back by DOI/PID. Both are opt-in: publishing is never triggered by
`analysis.run()` or `submit_workflow()`.

## Which deposit unit? data-DOI vs runnable-DOI

The toolkit can publish two different units, minting two different kinds of DOI. Pick by what
you want the DOI to *do*:

| Deposit unit | Command | The DOI mints a… | Use when |
|---|---|---|---|
| **analysis-directory set** (consolidated zarr + `ro-crate-metadata.json` + the two configs) | `analysis.publish(target=…)` | **data-DOI**: the archived, citable analysis outputs + provenance (reproducible from the configs and crate) | you're archiving results for citation / a data-availability statement |
| **reprex bundle** (the round-trippable bundle that runs an experiment from scratch) | `analysis.publish_reprex_bundle(target=…)` | **runnable-DOI**: `hhemt ingest --doi {DOI}` fetches and reconstitutes it, then prints the `hhemt run` command that runs it | you want a reproducible experiment that anyone can fetch by DOI (see [the DOI round-trip runbook](doi-roundtrip-e2e.md)) |

Both go through the same `target` seam and the same credentials below; they differ only in
what bytes are deposited. The rest of this guide uses `analysis.publish()` (the data-DOI); the
runnable-DOI path is identical with `publish_reprex_bundle()` in place of `publish()`.

## Before you publish

`analysis.publish()` deposits the *analysis-directory set*: the consolidated
`analysis_datatree.zarr`, the co-located `ro-crate-metadata.json` provenance sidecar, and
the two configs (`cfg_analysis.yaml` + `cfg_system.yaml`). So the analysis must already be
**consolidated** (run through `reprocess(start_with="consolidate")` or a full `run()`), or
publish has nothing to deposit and no crate to read the license from.

The dataset license is read back from the crate sidecar: it is baked in at consolidation
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

Zenodo mints the DOI on publish:

??? note "What the toolkit does behind `publish(target='zenodo')`"
    It creates a draft, embeds the record metadata, uploads the deposit, publishes,
    and reads the minted, DataCite-registered DOI back from the published record.
    No DOI is reserved up front, so the identifier does not exist until the deposit
    is published.

```python
result = analysis.publish(
    target="zenodo",
    software_doi="10.5281/zenodo.SOFTWARE",   # optional: links data -> software (IsCompiledBy)
)
print(result["data_doi"], result["record_url"])
# {"target": "zenodo", "data_doi": ..., "software_doi": ..., "record_url": ...}
```

`software_doi` is optional; when given, the deposit records a DataCite `IsCompiledBy`
`relatedIdentifier` (data → the software that produced it). When that software DOI is itself
a Zenodo record, `publish` also tries to backfill the reciprocal edge onto the software
record; a backfill failure is never raised, because the data record is already published.

Publishing does not re-stamp the archived license, so `publish` refuses to deposit a crate
whose license disagrees with your config. This check runs before any override is read: if
`analysis_config.dataset_license` differs from the license in `ro-crate-metadata.json`,
`publish` raises `PublishError` and tells you to re-emit the crate. Re-emitting requires
`regenerate_existing=True`. A plain `reprocess(start_with="consolidate")` leaves an intact
consolidated zarr in place and never rewrites the crate. Set `dataset_license` in the analysis
config file and rebuild the analysis from that file (a value set on the in-memory config never
reaches the consolidation subprocess), then:

```python
analysis.reprocess(start_with="consolidate", regenerate_existing=True)
analysis.publish(target="zenodo")
```

To additionally assert the license you expect at publish time, pass
`override_dataset_license`. It is compared against the crate only after the config check has
passed, and a disagreement raises `PublishError` rather than publishing a mismatched license:

```python
analysis.publish(target="zenodo", override_dataset_license="CC-BY-NC-4.0")
```

## Publish to HydroShare

HydroShare is a two-step flow, because hsclient (v1.1.6) has no programmatic DOI mint. `publish()`
creates the resource, uploads the deposit set, sets it public, then **stops and returns a
manual instruction**:

```python
result = analysis.publish(target="hydroshare")
print(result["manual_step"])   # open result["record_url"] and use 'Publish' in the web UI
```

Open `result["record_url"]` and click **Publish** in the HydroShare web UI to mint the DOI. If
you want the data-to-software relation recorded on the resource, pass `software_doi` on that
first `publish(target="hydroshare")` call, as in the Zenodo example above: the toolkit writes
it into the resource's relations before the resource goes public. Do not re-run `publish` to
add it afterwards. Every HydroShare `publish` call creates a new resource, so a second call
deposits a second copy and leaves the first untouched. For HydroShare deposits, the toolkit
writes no reciprocal edge (software to data, which needs the minted DOI) on the software
record.

## Publish a sensitivity analysis

A sensitivity analysis deposits its **master** tree the same way:

```python
sensitivity.publish(target="zenodo")   # deposits experiment_datatree.zarr + master sidecar
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

Then load the case study. The toolkit dispatches on `host`, fetches over anonymous-first
HTTPS, and verifies every file against the `manifest` sha256 map:

```python
from hhemt.experiments import TRITON_SWMM_experiment

experiment = TRITON_SWMM_experiment.from_case_study(
    case_name="norfolk_coastal_flooding",
    system_config_template="template_system_config.yaml",
    analysis_config_template="template_analysis_config.yaml",
    case_config_filename="case.yaml",
    weather_events_to_simulate="hurricane_irene_event_index.csv",
    analysis_description="Single Simulation of Hurricane Irene 8-27-2011",
    download_if_exists=False,   # set True to re-download even if the data is already local
)
```

The first six arguments are required and `download_if_exists` is optional. The two template
names and `case.yaml` live under `test_data/{case_name}/`. For the shipped Norfolk case,
`NorfolkIreneExperiment.load()` wraps this call.

The fetch is host-agnostic on verification: the streaming 1 MiB-chunk sha256 check is
byte-identical regardless of `host`, and Globus (when used) stays transport-only.
