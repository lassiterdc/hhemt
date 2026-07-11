"""RO-Crate metadata layer (C3) for the reproducibility-system (ADR-5/6/7).

Builds the RO-Crate @graph for one HHEMT analysis, serializes it byte-deterministically
(canonical post-process over crate.metadata.generate() — NEVER crate.write(), which
copies multi-GB data bytes), partitions the deterministic core from volatile provenance,
and compare-and-write-emits the co-located ro-crate-metadata.json sidecar.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from rocrate.model.contextentity import ContextEntity
from rocrate.rocrate import ROCrate

from hhemt.cf_conventions import _CF_VARIABLE_MAP

if TYPE_CHECKING:  # type-only edge — no runtime metadata->config coupling / import cycle
    from hhemt.config.invalidating_fixes import InvalidatingFix

_CF_PROFILE_ID = "https://cfconventions.org/cf-conventions/cf-conventions.html"
# Frozen, pinned context (RO-Crate 1.2). Pinning makes the serialized @context
# byte-deterministic even if add_workflow()/extra_terms would otherwise mutate it.
# Confirm the exact profile URIs against the live w3id registry at impl time (Follow-up Ideas).
_CANONICAL_CONTEXT = "https://w3id.org/ro/crate/1.2/context"

# Root-level profile set for the round-trip Workflow-Run-Crate (ADR-9 primary,
# D1/NQ-7 resolved: a GENERATED Snakefile is a valid mainEntity ComputationalWorkflow).
# Profiles go on the ROOT per the RO-Crate spec (NOT the descriptor, where ro-crate-py's
# add_workflow puts WORKFLOW_PROFILE); the descriptor keeps only the version permalink.
_WFRUN_ROOT_PROFILES: tuple[str, ...] = (
    "https://w3id.org/workflowhub/workflow-ro-crate/1.0",  # structural Workflow-RO-Crate
    "https://w3id.org/ro/wfrun/process/0.5",
    "https://w3id.org/ro/wfrun/workflow/0.5",
    "https://w3id.org/ro/wfrun/provenance/0.5",
)
# ro-crate-py's built-in Snakemake ComputerLanguage @id (rocrate/model/computerlanguage.py::snakemake).
_SNAKEMAKE_LANG_ID = "https://w3id.org/workflowhub/workflow-ro-crate#snakemake"

# Single source of truth for the frozen 2-entry dataset-license vocab (ADR-8). Feeds BOTH
# the RO-Crate root Dataset.license CreativeWork entity (here) AND the DataCite rightsList
# (publishing.py, Phase 4) — the two serializations can never drift.
_SPDX_LICENSE_TABLE: dict[str, dict[str, str]] = {
    "CC0-1.0": {
        "name": "Creative Commons Zero v1.0 Universal",
        "uri": "https://spdx.org/licenses/CC0-1.0",
        "scheme_uri": "https://spdx.org/licenses/",
    },
    "CC-BY-NC-4.0": {
        "name": "Creative Commons Attribution Non Commercial 4.0 International",
        "uri": "https://spdx.org/licenses/CC-BY-NC-4.0",
        "scheme_uri": "https://spdx.org/licenses/",
    },
}

# Single source of truth for what may live in the embedded core (CI grep-guard in
# test_metadata.py asserts no other key reaches tree.attrs["ro_crate_metadata"]).
_EMBEDDED_PROV_KEYS: frozenset[str] = frozenset(
    {
        "@context",
        "@id",
        "@type",
        "name",
        "analysis_id",
        "system_id",
        "schemaVersion",
        "conformsTo",
        "variableMeasured",
        "encodingFormat",
        "contentSize",
        "sha256",
        "version",
        "softwareVersion",
        "downloadUrl",
        "isBasedOn",
        "hasPart",
        "instrument",
        "object",
        "result",
        "description",
        "unitText",
        "propertyID",
        "measurementTechnique",
        "wasGeneratedBy",
        "license",
        # C8 Workflow-Run-Crate mainEntity (deterministic; bundle-side upgrade only,
        # NOT emitted at consolidation — see upgrade_doc_to_workflow_run_crate):
        "programmingLanguage",  # mainEntity ComputationalWorkflow runtime ref
        "mainEntity",           # Root focus -> the workflow
        "url",                  # ComputerLanguage.url (fixed vocab URI)
    }
)
# Keys that MUST never appear in the embedded core (wall-clock/host/jobid/run-ordinal).
_VOLATILE_PROV_KEYS: frozenset[str] = frozenset(
    {
        "startTime",
        "endTime",
        "duration",
        "agent",
        "datePublished",
        "dateModified",
        "actionStatus",
        "identifier",
    }
)
# Volatile keys excluded from the sidecar compare-and-write. Empty by Option-B payload
# discipline (the co-located sidecar carries the deterministic core only); keep in
# lockstep with the graph schema if a timestamp is ever added to the sidecar.
_VOLATILE_GRAPH_KEYS: set[str] = set()


def build_analysis_crate(
    *,
    analysis_id: str,
    system_id: str | None,
    layout_version: int,
    toolkit_git_sha: str,
    code_repository: str,
    cfg_case,  # CaseManifest
    dataset_license: str = "CC0-1.0",  # frozen 2-entry SPDX id (ADR-8 D1); strategy owned by reprex-specialist
    sif_spec,  # {@id, softwareVersion, sha256, downloadUrl} | None (native run)
    consolidated_zarr_relpath: str,  # e.g. "analysis_datatree.zarr"
    input_parts: list[dict],  # [{"@id", "sha256", "contentSize", "encodingFormat"}]
    sub_dataset_relpaths: list[str] | None = None,  # D5: master hasPart-refs each sub Dataset (FLAT)
) -> ROCrate:
    crate = ROCrate()  # seeds Root Dataset (./) + Metadata descriptor
    root = crate.root_dataset
    root["name"] = cfg_case.case_name
    root["description"] = cfg_case.description or ""
    root["analysis_id"] = analysis_id
    if system_id is not None:
        root["system_id"] = system_id
    root["schemaVersion"] = str(layout_version)

    _lic = _SPDX_LICENSE_TABLE[dataset_license]
    crate.add(
        ContextEntity(
            crate,
            _lic["uri"],
            properties={"@type": "CreativeWork", "name": _lic["name"]},
        )
    )
    root["license"] = {"@id": _lic["uri"]}

    toolkit_src = crate.add(
        ContextEntity(
            crate,
            "#hhemt-toolkit-src",
            properties={
                "@type": "SoftwareSourceCode",
                "name": "H&H Ensemble Modeling Toolkit",
                "codeRepository": code_repository,
                "version": toolkit_git_sha,
            },
        )
    )
    crate.add(
        ContextEntity(
            crate,
            "#hhemt-app",
            properties={
                "@type": "SoftwareApplication",
                "name": "hhemt",
                "softwareVersion": toolkit_git_sha,
                "isBasedOn": {"@id": toolkit_src.id},
            },
        )
    )

    if sif_spec is not None:  # native run path: sif_spec is None -> no SIF entity
        crate.add(
            ContextEntity(
                crate,
                sif_spec["@id"],
                properties={
                    "@type": "SoftwareApplication",
                    "name": sif_spec.get("name", "TRITON-SWMM Apptainer container"),
                    "softwareVersion": sif_spec["softwareVersion"],
                    "sha256": sif_spec["sha256"],
                    "downloadUrl": sif_spec["downloadUrl"],
                },
            )
        )

    for part in input_parts:  # by-reference File parts (bag-relative @id is unique)
        crate.add_file(
            source=None,
            dest_path=part["@id"],
            fetch_remote=False,
            validate_url=False,
            properties={
                "@type": "File",
                "sha256": part["sha256"],
                "contentSize": part.get("contentSize"),
                "encodingFormat": part.get("encodingFormat"),
            },
        )

    var_refs = []  # CF crosswalk -> variableMeasured PropertyValues
    for var, attrs in _CF_VARIABLE_MAP.items():
        pv = crate.add(
            ContextEntity(
                crate,
                f"#var-{var}",
                properties={
                    "@type": "PropertyValue",
                    "name": var,
                    "description": attrs["long_name"],
                    "unitText": attrs["units"],
                    **({"propertyID": attrs["standard_name"]} if attrs["standard_name"] else {}),
                    **({"measurementTechnique": attrs["cell_methods"]} if attrs["cell_methods"] else {}),
                },
            )
        )
        var_refs.append({"@id": pv.id})

    crate.add_dataset(  # the consolidated zarr is a DIRECTORY -> Dataset, not File
        source=None,
        dest_path=consolidated_zarr_relpath,
        properties={
            "@type": "Dataset",
            "name": "Consolidated analysis DataTree (zarr)",
            "conformsTo": {"@id": _CF_PROFILE_ID},
            "encodingFormat": "application/x-zarr",
            "variableMeasured": var_refs,
        },
    )

    if sub_dataset_relpaths:  # D5: master crate hasPart-references each sub Dataset (FLAT, not Subcrate)
        existing = root.get("hasPart") or []
        root["hasPart"] = list(existing) + [{"@id": rel} for rel in sub_dataset_relpaths]
    return crate


def canonical_jsonld_from_doc(doc: dict, *, context: str = _CANONICAL_CONTEXT) -> str:
    """Byte-reproducible JSON-LD string over a {"@context","@graph"} doc.

    ro-crate-py already sorts per-entity keys (Metadata.stream sort_keys=True); the single
    residual non-determinism is graph-element order + the @context shape. Closes both. No
    data bytes are copied (callers pass crate.metadata.generate() output, never crate.write()).
    """

    def _rank(entity: dict) -> tuple:
        eid = entity.get("@id", "")
        head = {"ro-crate-metadata.json": 0, "./": 1}.get(eid, 2)
        return (head, eid)

    out = {"@context": context, "@graph": sorted(doc["@graph"], key=_rank)}
    return json.dumps(out, sort_keys=True, ensure_ascii=False, separators=(",", ": "), indent=2) + "\n"


def canonical_jsonld(crate, *, context: str = _CANONICAL_CONTEXT) -> str:
    return canonical_jsonld_from_doc(crate.metadata.generate(), context=context)


def upgrade_doc_to_workflow_run_crate(doc: dict, *, workflow_relpath: str) -> dict:
    """Upgrade a built RO-Crate ``{"@context","@graph"}`` doc to a Workflow-Run-Crate.

    D1/NQ-7 (data-management FQ1, Option B): types the (already-bundled, generated)
    Snakefile as the crate ``mainEntity`` ``ComputationalWorkflow`` with a Snakemake
    ``programmingLanguage``, and appends the ``wfrun`` profiles to the ROOT
    ``conformsTo`` (spec-correct placement — profiles attach to the Root Data Entity,
    NOT the descriptor). A GENERATED Snakefile is a spec-valid ``mainEntity``; "generated
    vs hand-written" is not a spec-visible property of the file.

    This is a pure doc-level transform (mutates and returns ``doc``) so callers on the
    hand-emit + :func:`canonical_jsonld_from_doc` byte-deterministic path — the
    ``bundle/_combine.py`` pattern — can patch a copied ``ro-crate-metadata.json`` in
    place, reusing the by-reference SIF + ``input_parts`` already present without
    reconstructing ``sif_spec`` (sidesteps the ``_case_manifest`` gap). It is NOT wired
    into the consolidation-time :func:`build_analysis_crate` emit, so the on-disk
    embedded core stays byte-unchanged. Idempotent.
    """
    graph = doc["@graph"]
    # Snakemake ComputerLanguage context entity (fixed vocab term; add once).
    if not any(e.get("@id") == _SNAKEMAKE_LANG_ID for e in graph):
        graph.append(
            {
                "@id": _SNAKEMAKE_LANG_ID,
                "@type": "ComputerLanguage",
                "name": "Snakemake",
                "url": {"@id": "https://snakemake.readthedocs.io"},
            }
        )
    # The generated Snakefile as the mainEntity ComputationalWorkflow (upgrade if the
    # bundle already carries a plain File node for it; else add the node by-reference).
    wf_node = next((e for e in graph if e.get("@id") == workflow_relpath), None)
    if wf_node is None:
        wf_node = {"@id": workflow_relpath}
        graph.append(wf_node)
    wf_node["@type"] = ["File", "SoftwareSourceCode", "ComputationalWorkflow"]
    wf_node["name"] = "TRITON-SWMM ensemble Snakemake workflow (toolkit-generated)"
    wf_node["programmingLanguage"] = {"@id": _SNAKEMAKE_LANG_ID}
    # Root: mainEntity focus + the wfrun profiles (spec-correct: profiles on the ROOT).
    root = next(e for e in graph if e.get("@id") == "./")
    root["mainEntity"] = {"@id": workflow_relpath}
    existing = root.get("conformsTo") or []
    if isinstance(existing, dict):
        existing = [existing]
    existing_ids = {c.get("@id") for c in existing if isinstance(c, dict)}
    root["conformsTo"] = list(existing) + [
        {"@id": p} for p in _WFRUN_ROOT_PROFILES if p not in existing_ids
    ]
    return doc


def partition_core_vs_sidecar(full_doc: dict) -> dict:
    """Return the deterministic-core subset of a full JSON-LD doc (sidecar = full_doc)."""
    core_graph = []
    for entity in full_doc["@graph"]:
        kept = {k: v for k, v in entity.items() if k in _EMBEDDED_PROV_KEYS}
        assert not (_VOLATILE_PROV_KEYS & kept.keys()), kept.keys() & _VOLATILE_PROV_KEYS
        core_graph.append(kept)
    return {"@context": full_doc["@context"], "@graph": core_graph}


def _strip_volatile(graph_obj: dict) -> dict:
    if not _VOLATILE_GRAPH_KEYS:
        return graph_obj
    return {k: v for k, v in graph_obj.items() if k not in _VOLATILE_GRAPH_KEYS}


def write_rocrate_sidecar(analysis_dir: Path, *, graph_json: str) -> bool:
    """Compare-and-write ro-crate-metadata.json at the crate root (analysis_dir).

    Returns True if (re)written, False if skipped (mtime + bytes preserved). Mirrors
    du_sentinels.write_du_sentinel so an idempotent re-consolidation does NOT bump the
    file's mtime NOR perturb the Gotcha-38 analysis-scope DU own-files walk.
    """
    sidecar = Path(analysis_dir) / "ro-crate-metadata.json"
    new_obj = json.loads(graph_json)
    existing = None
    if sidecar.exists():
        try:
            existing = sidecar.read_text()
        except (OSError, UnicodeDecodeError):
            existing = None
    if existing is not None:
        try:
            if _strip_volatile(json.loads(existing)) == _strip_volatile(new_obj):
                return False
        except (json.JSONDecodeError, TypeError):
            pass
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.rocrate.", suffix=".json", dir=sidecar.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(graph_json)
        os.replace(tmp, sidecar)
    except Exception:
        Path(tmp).unlink(missing_ok=True)  # EXEMPT-DU: transient-intermediate
        raise
    return True


# --------------------------------------------------------------------------- #
# ADR-17 invalidating-fix quality-annotation projection (D7 — pure, deferred).
#
# These functions are the data-management-owned registry -> schema.org/DataCite
# crosswalk. They are SIDE-EFFECT-FREE and are deliberately NOT wired into
# build_analysis_crate / the consolidation-time emit path: a known-issue annotation
# is registry-state-dependent and time-varying (a bug can be registered AFTER an
# analysis's final consolidation), so baking it would either trip the
# _EMBEDDED_PROV_KEYS byte-determinism grep-guard (if in the embedded core) or churn
# the compare-and-write sidecar + strand post-consolidation bugs (if in the sidecar).
# Crate materialization + DOI relations + the report page defer to
# reproducibility-system_metadata-report-section (ADR-14). This plan ships the pure
# mapping + unit tests only, keeping the projection importable without pulling in the
# recompute resolver. Adding these adds ZERO keys to _EMBEDDED_PROV_KEYS.
# --------------------------------------------------------------------------- #
def project_invalidating_fix(entry: InvalidatingFix, affected_part_id: str) -> dict:
    """Pure registry-entry -> ``schema:Comment`` quality-annotation (ADR-17 / D7).

    Returns the known-issue annotation node for ``affected_part_id`` (an existing
    File/Dataset ``@id`` in a crate @graph). The annotation is a ``schema:Comment``
    (subset of ``CreativeWork``) attached via ``Comment.about``, carrying
    severity / recommended-action / version as nested ``schema:PropertyValue``
    machine-readable fields (severity as a controlled VALUE token, D1 — forward-
    compatible with a future ``{info, warning, error}`` widening). ``additionalProperty``
    on a ``CreativeWork`` is used as an RO-Crate extra term (RO-Crate tolerates extra
    terms; every field stays a schema.org-native ``PropertyValue``).
    """
    description = (entry.description or "").rstrip()
    significance = (entry.significance or "").rstrip()
    text = f"{description}\n\n{significance}" if significance else description
    return {
        "@id": f"#known-issue-{entry.commit_id[:7]}",
        "@type": "Comment",
        "about": {"@id": affected_part_id},
        "text": text,
        "identifier": entry.commit_id,
        "additionalProperty": [
            {
                "@type": "PropertyValue",
                "propertyID": "invalidating-fix-severity",
                "value": entry.severity,
            },
            {
                "@type": "PropertyValue",
                "propertyID": "recommended-recompute-action",
                "value": entry.recommended_action.value,
            },
            {
                "@type": "PropertyValue",
                "propertyID": "introduced-in-version",
                "value": entry.introduced_in_version,
            },
            {
                "@type": "PropertyValue",
                "propertyID": "affected-version-range",
                "value": entry.affected_version_range,
            },
        ],
    }


def datacite_supersession_descriptor(*, recomputed_doi: str, affected_doi: str) -> dict:
    """Pure DataCite ``IsNewVersionOf`` / ``IsPreviousVersionOf`` pair (ADR-17 / D-Q4).

    DOI-granularity supersession — the DATASET-level counterpart of the part-level
    ``Comment.about`` back-reference in :func:`project_invalidating_fix`. The
    re-computed dataset's record carries ``isNewVersionOf`` -> the affected DOI; the
    affected dataset's record carries the inverse ``isPreviousVersionOf`` -> the
    re-computed DOI. Verified against DataCite Metadata Schema 4.6: the controlled
    ``relationType`` list contains ``IsNewVersionOf`` / ``IsPreviousVersionOf`` (the
    ADR-17-named baseline pair) and NO ``IsCorrectedBy`` / ``Corrects``.
    """
    return {
        "isNewVersionOf": {
            "relatedIdentifier": affected_doi,
            "relatedIdentifierType": "DOI",
            "relationType": "IsNewVersionOf",
        },
        "isPreviousVersionOf": {
            "relatedIdentifier": recomputed_doi,
            "relatedIdentifierType": "DOI",
            "relationType": "IsPreviousVersionOf",
        },
    }
