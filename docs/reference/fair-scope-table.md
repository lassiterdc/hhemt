# FAIR scope of an HHEMT reproducibility dataset

FAIR applies as a continuum over the whole recreation set of inputs, configs, captured environment, provenance, and outputs. Outputs alone are not the scope. The **dataset** license, described below, is chosen separately from the software license. [The Home page](../index.md#license) states the license of the toolkit. The dataset has no license until you choose one, and publishing requires a license.

| Artifact class | Findable | Accessible | Interoperable | Reusable |
|---|---|---|---|---|
| Heavy inputs such as the DEM, the weather files, and the SWMM `.inp` files | DataCite DOI and PID in `case.yaml` | Anonymous-first HTTPS fetch with sha256 verification | CF-1.13 and documented formats | The toolkit grants no license here. The data host named in `case.yaml` supplies the data and the provenance. The host is HydroShare by `res_identifier` or Zenodo by `doi` or `pid`. |
| Configs, meaning the system, analysis, and case YAML files | Bundled by reference in the crate | Path-scrubbed and scanned for producer-specific values | Pydantic-validated YAML | Same dataset license |
| Captured environment as a SIF image | By-reference DOI and SHA-256 | The image is re-hashed against the SHA-256 of the crate. A missing image or a mismatch is refused. A container bundle whose crate carries no digest is reported as unverified and is not refused. | OCI and SIF | Software license and build definition |
| Provenance in RO-Crate and PROV | Embedded in the DataTree with a co-located sidecar | Deterministic JSON-LD | RO-Crate 1.2, PROV-O, CodeMeta, DataCite, and CF-1.13 | Reusable metadata |
| Consolidated outputs as a zarr DataTree | DataCite DOI through `analysis.publish` | Published, version-pinned, and DOI-registered | Hierarchical zarr and CF-1.13 | Dataset license `rightsList` and `IsCompiledBy` → software |

Choose the dataset license with `analysis_config.dataset_license`. The value is a Creative Commons SPDX identifier from `CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0`, `CC-BY-ND-4.0`, and `CC-BY-NC-ND-4.0`, and the field has no default. A configuration without the license loads, executes, and consolidates. `publish()` refuses to deposit until the license is set and the outputs are consolidated again. `CC0-1.0` waives copyright, database rights, and attribution, and the DOI, DataCite, and CITATION.cff carry citability. CC NonCommercial is broader than education-only use and applies to every reuser.

Set the license in the analysis config.

```yaml
dataset_license: CC-BY-4.0
```
