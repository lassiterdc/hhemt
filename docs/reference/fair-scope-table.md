# FAIR scope of an HHEMT reproducibility dataset

FAIR is applied as a continuum over the whole recreation set (inputs, configs, captured
environment, provenance, and outputs), not outputs alone. The **dataset** license (below) is
chosen separately from the software license (the toolkit's source is AGPL-3.0-or-later;
the dataset has no license until you choose one, and publishing requires one).

| Artifact class | Findable | Accessible | Interoperable | Reusable |
|---|---|---|---|---|
| Heavy inputs (DEM, weather, .inp) | DataCite DOI + PID in `case.yaml` | anonymous-first HTTPS fetch + sha256 verify | CF-1.13 / documented formats | not licensed by the toolkit: fetched from the data host `case.yaml` names (HydroShare by `res_identifier`, or Zenodo by `doi` or `pid`) + provenance |
| Configs (system/analysis/case yaml) | bundled by-reference in the crate | path-scrubbed, zero user info | Pydantic-validated YAML | same dataset license |
| Captured environment (SIF) | by-reference DOI + SHA-256 | immutable SIF, digest-verified fail-closed | OCI/SIF | software license + build def |
| Provenance (RO-Crate + PROV) | embedded in the DataTree + co-located sidecar | deterministic JSON-LD | RO-Crate 1.2 + PROV-O + CodeMeta + DataCite + CF-1.13 | reusable metadata |
| Consolidated outputs (zarr DataTree) | DataCite DOI (via `analysis.publish`) | published + version-pinned + DOI'd | hierarchical zarr + CF-1.13 | dataset license `rightsList` + `IsCompiledBy` → software |

Dataset license: chosen with `analysis_config.dataset_license`, one of seven Creative Commons SPDX
identifiers (`CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-NC-4.0`, `CC-BY-NC-SA-4.0`, `CC-BY-ND-4.0`,
`CC-BY-NC-ND-4.0`), with no default. A configuration without it loads, runs and consolidates;
`publish()` refuses to deposit until it is set and the outputs are consolidated again. `CC0-1.0` is
totally open, waiving copyright + DB rights + attribution, with citability carried by the DOI +
DataCite + CITATION.cff; note CC NonCommercial is broader than "education only" and does not turn
on user type.
