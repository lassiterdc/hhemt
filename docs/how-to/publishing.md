# Publish a dataset and fetch inputs by DOI/PID

## Goal

Publish a consolidated analysis to Zenodo or HydroShare and fetch the heavy inputs of a case study back by DOI or PID. Both directions are opt-in. `analysis.run()` and `submit_workflow()` never trigger publishing.

## Prerequisites

- A [consolidated analysis](#before-you-publish).
- [Host credentials in environment variables](#provide-credentials-as-environment-variables).

## Key references

- The [FAIR scope table](../reference/fair-scope-table.md) lists the license identifiers.
- [DOI round-trip for operators](doi-roundtrip-e2e.md) walks through the runnable-DOI round-trip.
- [Config filling](config-filling.md#4-point-the-analysis-config-at-your-events) describes `hhemt_sha`.
- The [configuration schema](../reference/config-schema.md) describes `dataset_license`.

## Choose a deposit unit: data-DOI or runnable-DOI

| Deposit unit | Command | DOI type | Use case |
|---|---|---|---|
| **Analysis-directory set**, the consolidated zarr with `ro-crate-metadata.json` and the system and analysis configs | `hhemt publish --target {zenodo or hydroshare}` | **Data-DOI** for the archived, citable analysis outputs with provenance, reproducible from the configs and the crate | Archiving results for citation or a data-availability statement |
| **Reprex bundle**, the round-trippable bundle that runs an experiment from scratch | `hhemt publish --target {zenodo or hydroshare} --reprex-bundle` | **Runnable-DOI**. `hhemt ingest --doi {DOI} --hpc-system-config {your-cluster-hpc-system-config.yaml}` fetches and reconstitutes the bundle, then prints the `hhemt run` command that runs the bundle | A reproducible experiment that anyone can fetch by DOI, as [the DOI round-trip](doi-roundtrip-e2e.md) shows |

Both deposits use the same `target` seam and the same credentials, and differ only in the deposited bytes. The rest of this guide describes the data-DOI deposit, and the runnable-DOI path is identical with `--reprex-bundle` added.

Both deposits are made from an experiment definition with the config-driven form.

```bash
hhemt publish --experiment {experiment-definition-dir} --cluster {cluster} --target zenodo
hhemt publish --experiment {experiment-definition-dir} --cluster {cluster} --target zenodo --reprex-bundle
```

To try a deposit without minting a citable DOI, use the Zenodo sandbox. `HHEMT_ZENODO_BASE_URL=https://sandbox.zenodo.org` together with a sandbox token selects the sandbox. The `10.5072` DOI prefix marks a record that you can discard, and production Zenodo mints `10.5281`. `hhemt ingest` resolves the Zenodo host from the same variable, so a sandbox DOI resolves against the sandbox and never against production. [The DOI round-trip](doi-roundtrip-e2e.md) names the deposit that verified the command forms and the parts left unverified.

## Before you publish

`hhemt publish` deposits the *analysis-directory set*, meaning the consolidated `analysis_datatree.zarr`, the co-located `ro-crate-metadata.json` provenance sidecar, `cfg_analysis.yaml`, and `cfg_system.yaml`, with paths relative to the deposit folder. The analysis must therefore be **consolidated**. Run `hhemt reprocess --start-with consolidate` or a full `hhemt run` first. Otherwise publish has no output to deposit and no crate to read the license from.

Publishing reads the dataset license back from the crate sidecar. Set `analysis_config.dataset_license` before consolidating, because consolidation bakes the license into the crate. Publishing keeps the archived license that consolidation stamped.

## Provide credentials as environment variables

Live deposits require host credentials, supplied through environment variables so that the credentials never land in a config file. A missing or misnamed variable fails with a `PublishError` that names the variable.

```bash
# Zenodo (or the sandbox: set HHEMT_ZENODO_BASE_URL=https://sandbox.zenodo.org)
export HHEMT_ZENODO_TOKEN=<your-zenodo-personal-access-token>

# HydroShare
export HHEMT_HYDROSHARE_USERNAME=<your-hydroshare-username>
export HHEMT_HYDROSHARE_PASSWORD=<your-hydroshare-password>
```

## Publish to Zenodo

??? note "Toolkit steps behind `publish(target='zenodo')`"
    The call creates a draft, embeds the record metadata, uploads the deposit, publishes, and reads the minted DataCite-registered DOI back from the published record. The toolkit reserves no DOI up front, so the identifier exists only after the deposit is published.

```bash
hhemt publish --experiment {experiment-definition-dir} --cluster {cluster} --target zenodo \
    --software-doi 10.5281/zenodo.SOFTWARE    # optional: links data -> software (IsCompiledBy)
```

`hhemt publish` prints the deposit record with `data_doi` as the minted DOI and `record_url` as the address of the record.

`--software-doi` is optional. When given, the deposit records a DataCite `IsCompiledBy` `relatedIdentifier` from the data to the software that produced the data. When the software DOI is a Zenodo record, `publish` also tries to backfill the reciprocal edge onto the software record, and a backfill failure never raises, because the data record is already published.

Publishing leaves the archived license as stamped, so `publish` refuses to deposit a crate whose license disagrees with your config. The check runs before any override is read. When `analysis_config.dataset_license` differs from the license in `ro-crate-metadata.json`, `publish` raises `PublishError`. The error tells you to re-emit the crate. Re-emitting requires `--regenerate-existing`, and a plain `hhemt reprocess --start-with consolidate` leaves an intact consolidated zarr in place and never rewrites the crate. Set `dataset_license` in the analysis config file, run the consolidation again from that file, and then deposit.

```bash
hhemt reprocess --system-config my_system.yaml --analysis-config my_analysis.yaml \
    --start-with consolidate --regenerate-existing
hhemt publish --experiment {experiment-definition-dir} --cluster {cluster} --target zenodo
```

## Publish to HydroShare

HydroShare needs a manual step, because hsclient has no programmatic DOI mint. `hhemt publish` creates the resource, uploads the deposit set, and sets the resource public. The command then **stops and prints a manual instruction**.

```bash
hhemt publish --experiment {experiment-definition-dir} --cluster {cluster} --target hydroshare
```

Open the `record_url` that the command prints and click **Publish** in the HydroShare web UI to mint the DOI. To record the data-to-software relation on the resource, pass `--software-doi` on that first `hhemt publish --target hydroshare` call, as the Zenodo example shows. The toolkit writes the software DOI into the relations of the resource before the resource goes public. Do not rerun `publish` to add the DOI afterwards. Every HydroShare `publish` call creates a new resource, so a second call deposits a second copy and leaves the first untouched. For HydroShare deposits, the toolkit writes no reciprocal edge on the software record, from software to data, because that edge needs the minted DOI.

## Publish a sensitivity analysis

The deposit of a sensitivity analysis uses the same command and covers the **primary** tree, so the consolidated zarr of the analysis-directory set is `experiment_datatree.zarr`.

## Fetch a case study's inputs by DOI/PID

The `case.yaml` of a case study is a provenance descriptor of *remote* heavy inputs. Point the descriptor at a durable deposit with `host` and `doi`, or `pid`.

```yaml
# test_data/<case_name>/case.yaml
case_name: norfolk_coastal_flooding
res_identifier: <32-hex-hydroshare-resource-id>
host: zenodo                       # or: hydroshare
doi: '10.5281/zenodo.1234567'      # host='zenodo' requires a doi OR pid
```

Load the case study from Python, because the case-study fetch has no command-line form. The toolkit dispatches on `host`, fetches over anonymous-first HTTPS, and verifies every file against the `manifest` sha256 map.

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
    # hhemt_sha is the commit you run: `git rev-parse HEAD`
    hhemt_sha="REPLACE-WITH-THE-FULL-40-HEX-HHEMT-COMMIT",
)
```

The call requires every argument shown except `download_if_exists`. `hhemt_sha` is keyword-only and holds the commit of hhemt that you run, as [Config filling](config-filling.md#4-point-the-analysis-config-at-your-events) describes. The template names and `case.yaml` live under `test_data/{case_name}/`. For the shipped Norfolk case, `NorfolkIreneExperiment.load(hhemt_sha=...)` wraps this call.

Verification is host-agnostic. The streaming 1 MiB-chunk sha256 check is byte-identical regardless of `host`, and Globus, when used, stays transport-only.
