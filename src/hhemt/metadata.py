"""Machine-readable provenance for a consolidated analysis, written as an RO-Crate.

A consolidated analysis output is reusable by someone else only if it says what produced
it: which toolkit commit, which container image, which input files, and which variables
the deposited store contains. This module writes that description in the RO-Crate JSON-LD
format, so a repository or a reproducer can read it without this toolkit installed.

The description has to be byte-deterministic. Its core is embedded in the consolidated
tree's attributes, and the co-located ro-crate-metadata.json sidecar is emitted at every
consolidation, so a byte that varied between two consolidations of the same run would
rewrite the sidecar and bump its mtime on a re-consolidation that changed nothing. So the
module builds the @graph for one HHEMT analysis, serializes it canonically (a
post-process over crate.metadata.generate() — NEVER crate.write(), which copies the
multi-GB data bytes), partitions the deterministic core from volatile provenance, and
writes the sidecar only when its content changed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from rocrate.model.contextentity import ContextEntity
from rocrate.rocrate import ROCrate

from hhemt.cf_conventions import _ADVERTISABLE_VARIABLES, _CF_VARIABLE_MAP

if TYPE_CHECKING:  # type-only edge — no runtime metadata->config coupling / import cycle
    from hhemt.config.invalidating_fixes import InvalidatingFix

_CF_PROFILE_ID = "https://cfconventions.org/cf-conventions/cf-conventions.html"
# Frozen, pinned context (RO-Crate 1.2). Pinning makes the serialized @context
# byte-deterministic even if add_workflow()/extra_terms would otherwise mutate it.
# Not established here: that this URI and the profile URIs below resolve in the live w3id
# registry. Nothing in this module fetches them, so a wrong URI would surface only in a
# consumer that dereferences it.
_CANONICAL_CONTEXT = "https://w3id.org/ro/crate/1.2/context"

# Root-level profile set for the round-trip Workflow-Run-Crate. A GENERATED Snakefile is a
# valid mainEntity ComputationalWorkflow: whether a workflow file was written by hand or by
# a program is not visible to the spec, so the toolkit-generated one qualifies.
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

# Single source of truth for the dataset-license vocabulary, keyed by SPDX id. Feeds BOTH
# the RO-Crate root Dataset.license CreativeWork entity (here) AND the DataCite rightsList
# that publishing.py builds, and bundle/_emit.py checks a bundled crate's license URI
# against it — so the serializations of one dataset's license cannot name different terms.
_SPDX_LICENSE_TABLE: dict[str, dict[str, str]] = {
    "CC0-1.0": {
        "name": "Creative Commons Zero v1.0 Universal",
        "uri": "https://spdx.org/licenses/CC0-1.0",
        "scheme_uri": "https://spdx.org/licenses/",
    },
    "CC-BY-4.0": {
        "name": "Creative Commons Attribution 4.0 International",
        "uri": "https://spdx.org/licenses/CC-BY-4.0",
        "scheme_uri": "https://spdx.org/licenses/",
    },
    "CC-BY-SA-4.0": {
        "name": "Creative Commons Attribution Share Alike 4.0 International",
        "uri": "https://spdx.org/licenses/CC-BY-SA-4.0",
        "scheme_uri": "https://spdx.org/licenses/",
    },
    "CC-BY-NC-4.0": {
        "name": "Creative Commons Attribution Non Commercial 4.0 International",
        "uri": "https://spdx.org/licenses/CC-BY-NC-4.0",
        "scheme_uri": "https://spdx.org/licenses/",
    },
    "CC-BY-NC-SA-4.0": {
        "name": "Creative Commons Attribution Non Commercial Share Alike 4.0 International",
        "uri": "https://spdx.org/licenses/CC-BY-NC-SA-4.0",
        "scheme_uri": "https://spdx.org/licenses/",
    },
    "CC-BY-ND-4.0": {
        "name": "Creative Commons Attribution No Derivatives 4.0 International",
        "uri": "https://spdx.org/licenses/CC-BY-ND-4.0",
        "scheme_uri": "https://spdx.org/licenses/",
    },
    "CC-BY-NC-ND-4.0": {
        "name": "Creative Commons Attribution Non Commercial No Derivatives 4.0 International",
        "uri": "https://spdx.org/licenses/CC-BY-NC-ND-4.0",
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
        # Workflow-Run-Crate mainEntity keys (deterministic). Only the bundle-side
        # upgrade_doc_to_workflow_run_crate adds them; consolidation never emits them:
        "programmingLanguage",  # mainEntity ComputationalWorkflow runtime ref
        "mainEntity",  # Root focus -> the workflow
        "url",  # ComputerLanguage.url (fixed vocab URI)
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
# PER-TIER STRIP SETS. The two sets below are byte-equal to each other and to
# _VOLATILE_PROV_KEYS above TODAY. That equality is a coincidence of timing, not a shared
# rule. Do NOT re-merge them and do NOT alias either one to the set above.
#
# The reason is a STANDARDS reason, and it outlives the change that introduced the split.
# `datePublished` is a MUST on the RO-Crate Root Data Entity. The two tiers owe that
# property under DIFFERENT rules: the bundle tier TRANSPORTS an existing crate and must
# preserve the description it carries, while the combined tier emits a FIRST publication
# of the combination and must author its own value. One denylist cannot carry two rules.
#
# The bundle tier in particular should never have applied a MUST-stripping denylist at
# all. The staging directory carries a ro-crate-metadata.json, which makes it an ATTACHED
# PACKAGE, and an Attached Package's Root Data Entity MUST carry datePublished. Stripping
# it produces a non-conformant crate on its own terms, independently of any sequencing and
# independently of what the source crate was doing.
#
# Scheduling note, which EXPIRES once both tiers emit correctly and is therefore not the
# justification above: while the bundle tier still strips the key and the combined tier
# does not yet author one, no single edit to one shared constant satisfies both tiers,
# because each tier's correct behaviour depends on the other's already being in place.
# The split is what makes the two changes independently landable.
_BUNDLE_TRANSFER_STRIP_KEYS: frozenset[str] = frozenset(
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
_COMBINED_EMIT_STRIP_KEYS: frozenset[str] = frozenset(
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
# Volatile keys excluded from the sidecar compare-and-write.
#
# CORRECTED: this previously read "the co-located sidecar carries the deterministic core
# only". That is FALSE and the error is load-bearing, because it inverts the analysis of
# whether a new graph key can perturb the sidecar. `emit_provenance` returns
# `canonical_jsonld(crate)` -- the FULL graph -- as the sidecar, and the CreateAction
# entities it builds carry `startTime` and `agent`, both members of _VOLATILE_PROV_KEYS.
# The core-only document is the OTHER return value, produced by partition_core_vs_sidecar.
#
# So this set being EMPTY makes `_strip_volatile` a no-op and the compare-and-write below
# run over the full document, volatile keys included. That is currently SAFE, and the
# reason is worth stating because it is a property rather than an accident: every value in
# the graph is a stable function of the RUN'S LOG, not of emit time -- `startTime` reads
# the persisted `workflow_submission_time` LogField and `agent` reads
# `workflow_submission_node`, so an idempotent re-consolidation regenerates byte-identical
# content and the compare-and-write skips, leaving the sidecar's bytes and mtime unchanged.
# Add ANY emit-time clock read to the graph and every re-consolidation rewrites the sidecar
# immediately; the remedy then is to name that key here, which is what this set is for.
_VOLATILE_GRAPH_KEYS: set[str] = set()


def build_analysis_crate(
    *,
    analysis_id: str,
    system_id: str | None,
    layout_version: int,
    toolkit_git_sha: str,
    code_repository: str,
    cfg_case,  # CaseManifest
    dataset_license: str | None = None,  # one of the SPDX ids in _SPDX_LICENSE_TABLE, or None: no license written
    sif_spec,  # {@id, softwareVersion, sha256, downloadUrl} | None (native run)
    consolidated_zarr_relpath: str,  # e.g. "analysis_datatree.zarr"
    input_parts: list[dict],  # [{"@id", "sha256", "contentSize", "encodingFormat"}]
    emitted_vars: set[str] | None = None,  # actual data_vars of the deposited store
    # {name: attrs} read off the DEPOSITED store, so each advertised variable is described
    # exactly as the store describes it. A precomputed MAPPING rather than the datasets
    # themselves, deliberately: both callers already iterate the tree to build `emitted_vars`
    # and can fill this in the same pass, and this module then gains no xarray dependency.
    emitted_attrs: dict[str, dict[str, str | None]] | None = None,
    sub_dataset_relpaths: list[str] | None = None,  # experiment crate hasPart-refs each member Dataset (FLAT)
) -> ROCrate:
    crate = ROCrate()  # seeds Root Dataset (./) + Metadata descriptor
    root = crate.root_dataset
    root["name"] = cfg_case.case_name
    root["description"] = cfg_case.description or ""
    root["analysis_id"] = analysis_id
    if system_id is not None:
        root["system_id"] = system_id
    root["schemaVersion"] = str(layout_version)

    # No license is a legitimate state: a dataset license is required only to PUBLISH, so an
    # unlicensed analysis consolidates with no root `license` and no license entity, and
    # publish refuses it until one is chosen. Writing a placeholder license here instead
    # would make an unpublished dataset assert terms its owner never chose.
    if dataset_license is not None:
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
                    # Besides the entity's @id, sha256 is the only required field (both are
                    # indexed unguarded, so a spec missing either raises KeyError), because
                    # the digest and the URL split apart: a digest is emittable today from
                    # the image the run used, while a downloadUrl requires a deposit target
                    # that does not yet exist. Requiring the URL would have made the digest
                    # unemittable — `sif_spec["downloadUrl"]` raised KeyError on a
                    # digest-only spec, so the two halves of "referenced by DOI + SHA-256"
                    # were welded into one all-or-nothing feature and neither shipped.
                    #
                    # softwareVersion is likewise optional: reading it means an
                    # `apptainer inspect` label read, and NO consumer requires it —
                    # `_reprex._find_sif_entity` selects on `sha256` and `_verify_sif`
                    # reads only `sha256`. Taking a binary dependency for an unread field
                    # would be cost without a consumer.
                    "sha256": sif_spec["sha256"],
                    **({"softwareVersion": sif_spec["softwareVersion"]} if sif_spec.get("softwareVersion") else {}),
                    **({"downloadUrl": sif_spec["downloadUrl"]} if sif_spec.get("downloadUrl") else {}),
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

    # variableMeasured is a claim about the DEPOSITED store, so advertise only what the
    # store contains. Iterating _CF_VARIABLE_MAP alone published every map key as a claim:
    # measured 2026-07-21 on a real sensitivity crate, 18 advertised / 11 absent from the
    # zarr. Passing emitted_vars=None preserves the legacy whole-map behavior for callers
    # that cannot see the dataset; every in-tree caller passes the real set.
    # The union is scoped to the GATED branch only. Advertising performance names on
    # the emitted_vars=None path would re-open the 2026-07-21 over-claim defect, since
    # variableMeasured is a claim about the DEPOSITED store and a non-performance store
    # contains none of the thirteen.
    # `_ADVERTISABLE_VARIABLES` is IMPORTED rather than rebuilt here, so that the
    # descriptor drift guard asserting coverage of this union reads the same object. A
    # union rebuilt locally would let a widening here pass a guard that recomputed its own.
    _advertised = (
        _CF_VARIABLE_MAP.items()
        if emitted_vars is None
        else [(v, a) for v, a in _ADVERTISABLE_VARIABLES.items() if v in emitted_vars]
    )

    # THE DESCRIPTOR IS SOURCED FROM THE DEPOSITED STORE'S OWN ATTRS WHEN THE CALLER SUPPLIES
    # THEM, which is what makes the advertisement-set invariant structural rather than policed:
    # "advertise exactly the variables the deposited store contains, described exactly as the
    # store describes them." The membership half is the `emitted_vars` gate above; this is the
    # description half. You cannot describe a variable you did not read, and the descriptor
    # cannot diverge from the stamp because it IS the stamp.
    #
    # UNTIL THIS LANDED, THE CRATE WAS OVERRIDE-BLIND, and the recorded consequence is kept here
    # rather than deleted with the defect. Before it landed, this module read `_CF_VARIABLE_MAP`
    # and `_CF_PERFORMANCE_VARIABLES` and read `_CF_VARIABLE_OVERRIDES_BY_MODE` NOT AT ALL
    # (`grep -c '_CF_VARIABLE_OVERRIDES_BY_MODE' src/hhemt/metadata.py` was 0 against 2 in
    # `cf_conventions.py`), so a link-mode deposit published a `long_name` and a `cell_methods`
    # that its own store contradicted. Two override modes carry that entry and they are
    # byte-identical -- `tritonswmm_swmm_link` and `swmm_only_link`, the coupled model's and the
    # standalone model's link outputs. After the base entries were respelled to the conformant
    # `time:` spelling the `cell_methods` half agrees BY COINCIDENCE OF TWO CONVERGING VALUES,
    # not by derivation; the surviving `long_name` divergence is what proved the crate was not
    # store-sourced. Reading the store closes both halves at once.
    #
    # THE FALLBACK IS CONSERVATIVE AND ITS PREDICATE IS `long_name`. A store node that was never
    # CF-stamped carries no `long_name`, and a tree may legitimately contain such a node (the
    # opt-in per-scenario timeseries groups are not stamped), so an advertised name resolving to
    # an unstamped node falls back to the static entry rather than publishing an empty
    # descriptor. That makes this change MONOTONE: it can only replace a static descriptor with
    # a store descriptor that actually exists.
    def _descriptor(var: str, static: dict) -> dict[str, str | None]:
        stored = (emitted_attrs or {}).get(var)
        src = stored if stored and stored.get("long_name") else static
        return {k: src.get(k) for k in ("long_name", "units", "standard_name", "cell_methods")}

    var_refs = []  # CF crosswalk -> variableMeasured PropertyValues
    for var, attrs in _advertised:
        attrs = _descriptor(var, attrs)
        pv = crate.add(
            ContextEntity(
                crate,
                f"#var-{var}",
                properties={
                    "@type": "PropertyValue",
                    "name": var,
                    "description": attrs["long_name"],
                    # GUARDED SYMMETRICALLY WITH ITS TWO SIBLINGS, and this is prophylactic
                    # rather than a repair of a live false claim. Every `_ADVERTISABLE_VARIABLES`
                    # entry is typed `str | None`, so `None` is admissible for `units` exactly as
                    # for the other two; measured, `units is None` in zero of the 25, and a real
                    # deposited crate carried 9 PropertyValues with 9 non-null `unitText`. The
                    # value of the guard is that a FUTURE entry legitimately carrying no units --
                    # or a store-sourced descriptor read off a node that declares none -- cannot
                    # silently publish a `unitText` null into the deposited record.
                    **({"unitText": attrs["units"]} if attrs["units"] else {}),
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

    if sub_dataset_relpaths:  # experiment crate hasPart-references each member Dataset (FLAT, not Subcrate)
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

    Types the (already-bundled, generated) Snakefile as the crate ``mainEntity``
    ``ComputationalWorkflow`` with a Snakemake ``programmingLanguage``, and appends the
    ``wfrun`` profiles to the ROOT ``conformsTo`` (spec-correct placement — profiles
    attach to the Root Data Entity, NOT the descriptor). A GENERATED Snakefile is a
    spec-valid ``mainEntity``; "generated vs hand-written" is not a spec-visible
    property of the file.

    This is a pure doc-level transform (mutates and returns ``doc``) so callers on the
    hand-emit + :func:`canonical_jsonld_from_doc` byte-deterministic path — the
    ``bundle/_combine.py`` pattern — can patch a copied ``ro-crate-metadata.json`` in
    place, reusing the by-reference SIF + ``input_parts`` already present without
    reconstructing ``sif_spec``. It is NOT wired into the consolidation-time
    :func:`build_analysis_crate` emit, so the on-disk embedded core stays
    byte-unchanged. Idempotent.
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
    root["conformsTo"] = list(existing) + [{"@id": p} for p in _WFRUN_ROOT_PROFILES if p not in existing_ids]
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
    file's mtime. The file sits directly under analysis_dir, where the analysis-scope
    disk-usage walk counts it, so a skipped write also leaves that walk's input unchanged.
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
# Invalidating-fix quality annotations: registry entry -> schema.org / DataCite.
#
# When a registered fix invalidates results an analysis already produced, a published
# record should be able to say so. These functions are the crosswalk from one entry of the
# invalidating-fix registry to that statement. They are SIDE-EFFECT-FREE and are
# deliberately NOT wired into build_analysis_crate / the consolidation-time emit path: a
# known-issue annotation is registry-state-dependent and time-varying (a bug can be
# registered AFTER an analysis's final consolidation), so baking it would either trip the
# _EMBEDDED_PROV_KEYS byte-determinism grep-guard (if in the embedded core) or churn the
# compare-and-write sidecar + strand post-consolidation bugs (if in the sidecar).
# Materializing them into a crate, emitting the DOI relations, and rendering them in a
# report are not done in this module: it holds only the pure mapping, which keeps the
# projection importable without pulling in the recompute resolver. These functions add
# ZERO keys to _EMBEDDED_PROV_KEYS.
# --------------------------------------------------------------------------- #
def project_invalidating_fix(entry: InvalidatingFix, affected_part_id: str) -> dict:
    """Pure registry-entry -> ``schema:Comment`` quality-annotation.

    Returns the known-issue annotation node for ``affected_part_id`` (an existing
    File/Dataset ``@id`` in a crate @graph). The annotation is a ``schema:Comment``
    (subset of ``CreativeWork``) attached via ``Comment.about``, carrying
    severity / recommended-action / version as nested ``schema:PropertyValue``
    machine-readable fields. Severity is a controlled VALUE token rather than a property
    of its own, so widening its vocabulary later (to ``{info, warning, error}``, say)
    changes the value and not the record's shape. ``additionalProperty``
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
    """Pure DataCite ``IsNewVersionOf`` / ``IsPreviousVersionOf`` pair.

    DOI-granularity supersession — the DATASET-level counterpart of the part-level
    ``Comment.about`` back-reference in :func:`project_invalidating_fix`. The
    re-computed dataset's record carries ``isNewVersionOf`` -> the affected DOI; the
    affected dataset's record carries the inverse ``isPreviousVersionOf`` -> the
    re-computed DOI. Verified against DataCite Metadata Schema 4.6: the controlled
    ``relationType`` list contains ``IsNewVersionOf`` / ``IsPreviousVersionOf`` and NO
    ``IsCorrectedBy`` / ``Corrects``, which is why a re-computation is recorded as a new
    version of the affected dataset rather than as a correction of it.
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
