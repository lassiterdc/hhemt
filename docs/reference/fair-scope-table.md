# FAIR scope of an HHEMT reproducibility dataset

FAIR is applied as a continuum over the WHOLE recreation set — inputs, configs, captured
environment, provenance, and outputs — not outputs alone. The DATASET license (below) is
chosen SEPARATELY from the software license (the toolkit is PolyForm-NC source-available;
the dataset defaults to CC0-1.0, an open public good — a recognized, coherent pattern).

| Artifact class | Findable | Accessible | Interoperable | Reusable |
|---|---|---|---|---|
| Heavy inputs (DEM, weather, .inp) | DataCite DOI + PID in `case.yaml` | anonymous-first HTTPS fetch + sha256 verify | CF-1.13 / documented formats | `CC0-1.0` (default) rightsList + provenance |
| Configs (system/analysis/case yaml) | bundled by-reference in the crate | path-scrubbed, zero user info | Pydantic-validated YAML | same dataset license |
| Captured environment (SIF) | by-reference DOI + SHA-256 | signed immutable SIF | OCI/SIF | software license + build def |
| Provenance (RO-Crate + PROV) | embedded in the DataTree + co-located sidecar | deterministic JSON-LD | RO-Crate 1.2 + PROV-O + CodeMeta + DataCite + CF-1.13 | reusable metadata |
| Consolidated outputs (zarr DataTree) | DataCite DOI (via `analysis.publish`) | published + version-pinned + DOI'd | hierarchical zarr + CF-1.13 | dataset license `rightsList` + `IsCompiledBy` → software |

Dataset license: `CC0-1.0` (default; totally open — waives copyright + DB rights + attribution;
citability carried by the DOI + DataCite + CITATION.cff) or `CC-BY-NC-4.0` (research/education-leaning;
note CC NonCommercial is broader than "education only" and does not turn on user type).
