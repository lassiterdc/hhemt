"""Unit tests for the C3 metadata layer (pure; no consolidation wiring)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from hhemt import metadata


def _fake_case():
    return SimpleNamespace(case_name="norfolk", description="test", manifest={})


def _build(**over):
    kw = dict(
        analysis_id="a1",
        system_id="s1",
        layout_version=16,
        toolkit_git_sha="deadbeef",
        code_repository="https://example/repo",
        cfg_case=_fake_case(),
        sif_spec=None,
        consolidated_zarr_relpath="analysis_datatree.zarr",
        input_parts=[{"@id": "inputs/dem.tif", "sha256": "ab" * 32, "contentSize": 10, "encodingFormat": "image/tiff"}],
    )
    kw.update(over)
    return metadata.build_analysis_crate(**kw)


def test_serializer_is_byte_deterministic():
    # R10: two emits of identical-content crates are byte-identical.
    assert metadata.canonical_jsonld(_build()) == metadata.canonical_jsonld(_build())


def test_jsonld_roundtrips_valid():
    # R10 validity: the doc parses and re-canonicalizes to the same string (idempotent).
    s = metadata.canonical_jsonld(_build())
    doc = json.loads(s)
    assert set(doc) == {"@context", "@graph"}
    assert metadata.canonical_jsonld_from_doc(doc) == s


def test_native_path_has_no_sif_entity():
    # R9: sif_spec=None -> no SoftwareApplication SIF node.
    doc = json.loads(metadata.canonical_jsonld(_build(sif_spec=None)))
    assert not any(e.get("@type") == "SoftwareApplication" and "downloadUrl" in e for e in doc["@graph"])


def test_partition_strips_volatile_and_guards_allowlist():
    # R3/R7: every key in the core is on the embedded allowlist; no volatile key survives.
    full = _build().metadata.generate()
    full["@graph"][0]["startTime"] = "2026-01-01T00:00:00"  # inject a volatile field
    core = metadata.partition_core_vs_sidecar(full)
    for entity in core["@graph"]:
        assert set(entity) <= metadata._EMBEDDED_PROV_KEYS
        assert not (metadata._VOLATILE_PROV_KEYS & set(entity))


def test_dataset_license_creativework_entity_survives_into_core():
    # R6/R7: dataset_license threads into a root `license` reference plus a CreativeWork
    # contextual entity, and "license" survives partition into the embedded deterministic core.
    crate = _build(dataset_license="CC-BY-NC-4.0")
    doc = json.loads(metadata.canonical_jsonld(crate))
    root = next(e for e in doc["@graph"] if e.get("@id") == "./")
    assert root["license"] == {"@id": "https://spdx.org/licenses/CC-BY-NC-4.0"}
    assert any(
        e.get("@id") == "https://spdx.org/licenses/CC-BY-NC-4.0" and e.get("@type") == "CreativeWork"
        for e in doc["@graph"]
    )
    # R7: "license" is on the embedded allowlist, so it lands in the deterministic core (not the sidecar-only set).
    assert "license" in metadata._EMBEDDED_PROV_KEYS
    core = metadata.partition_core_vs_sidecar(crate.metadata.generate())
    core_root = next(e for e in core["@graph"] if e.get("@id") == "./")
    assert core_root["license"] == {"@id": "https://spdx.org/licenses/CC-BY-NC-4.0"}


def test_default_dataset_license_is_cc0():
    # R5/R6: the default build (no override) carries CC0-1.0 as the root license.
    doc = json.loads(metadata.canonical_jsonld(_build()))
    root = next(e for e in doc["@graph"] if e.get("@id") == "./")
    assert root["license"] == {"@id": "https://spdx.org/licenses/CC0-1.0"}


def test_upgrade_doc_to_workflow_run_crate():
    # C8 D1/NQ-7 (Option C shared helper): a GENERATED Snakefile is typed as the crate
    # mainEntity ComputationalWorkflow with a Snakemake programmingLanguage, and the
    # wfrun profiles land on the ROOT (spec-correct placement, NOT the descriptor).
    doc = json.loads(metadata.canonical_jsonld(_build()))
    metadata.upgrade_doc_to_workflow_run_crate(doc, workflow_relpath="Snakefile.source")
    root = next(e for e in doc["@graph"] if e.get("@id") == "./")
    assert root["mainEntity"] == {"@id": "Snakefile.source"}
    wf = next(e for e in doc["@graph"] if e.get("@id") == "Snakefile.source")
    assert wf["@type"] == ["File", "SoftwareSourceCode", "ComputationalWorkflow"]
    assert wf["programmingLanguage"] == {"@id": metadata._SNAKEMAKE_LANG_ID}
    lang = next(e for e in doc["@graph"] if e.get("@id") == metadata._SNAKEMAKE_LANG_ID)
    assert lang["@type"] == "ComputerLanguage"
    assert lang["url"] == {"@id": "https://snakemake.readthedocs.io"}
    profile_ids = {c["@id"] for c in root["conformsTo"]}
    assert set(metadata._WFRUN_ROOT_PROFILES) <= profile_ids  # all four wfrun profiles
    # R10: byte-deterministic reserialize + idempotent re-upgrade.
    s1 = metadata.canonical_jsonld_from_doc(doc)
    doc2 = json.loads(s1)
    metadata.upgrade_doc_to_workflow_run_crate(doc2, workflow_relpath="Snakefile.source")
    assert metadata.canonical_jsonld_from_doc(doc2) == s1


def test_bundle_schema_version_bumped_to_v3():
    # C8: round-trippable Workflow-Run-Crate + reprex carriage = bundle schema v3.
    from hhemt.version_migration.constants import BUNDLE_SCHEMA_VERSION

    assert BUNDLE_SCHEMA_VERSION == 3


def test_sidecar_compare_and_write_idempotent(tmp_path):
    # R4: a second write of identical content returns False (no rewrite / mtime bump).
    g = metadata.canonical_jsonld(_build())
    assert metadata.write_rocrate_sidecar(tmp_path, graph_json=g) is True
    mtime1 = (tmp_path / "ro-crate-metadata.json").stat().st_mtime_ns
    assert metadata.write_rocrate_sidecar(tmp_path, graph_json=g) is False
    assert (tmp_path / "ro-crate-metadata.json").stat().st_mtime_ns == mtime1


def _advertised_var_names(crate) -> set[str]:
    """The variable names the deposited-store Dataset advertises via variableMeasured."""
    doc = json.loads(metadata.canonical_jsonld(crate))
    by_id = {e["@id"]: e for e in doc["@graph"]}
    ds = next(e for e in doc["@graph"] if e.get("encodingFormat") == "application/x-zarr")
    return {by_id[ref["@id"]]["name"] for ref in ds.get("variableMeasured", [])}


def test_variable_measured_advertises_only_emitted_vars():
    # RELEASE-BLOCKING (VMS-2, 2026-07-21): variableMeasured is a claim about the
    # DEPOSITED store, so build_analysis_crate must advertise only the variables the
    # store actually contains. A prior whole-map iteration published every _CF_VARIABLE_MAP
    # key as a claim (measured: 18 advertised / 11 absent on a real sensitivity crate).
    # This guards the emitter fix directly; test_no_phantom_swmm_keys_in_cf_variable_map
    # guards the MAP, not the emitter's filtering — the two are distinct regressions.
    from hhemt.cf_conventions import _CF_VARIABLE_MAP

    # emitted_vars=None -> legacy whole-map behavior preserved (callers that cannot see
    # the dataset, e.g. a doc-level upgrade path, still advertise the full crosswalk).
    whole = _advertised_var_names(_build())
    assert whole == set(_CF_VARIABLE_MAP)

    # A real emitted set advertises exactly those variables — nothing the store lacks.
    subset = {"max_wlevel_m", "max_flow_cms"}
    filtered = _advertised_var_names(_build(emitted_vars=subset))
    assert filtered == subset
    assert filtered < whole  # strictly fewer than the whole map: the over-claim is gone

    # A name not present in the crosswalk is silently dropped (advertise ⊆ crosswalk ∩ emitted).
    over = _advertised_var_names(_build(emitted_vars={"max_wlevel_m", "not_a_real_var"}))
    assert over == {"max_wlevel_m"}
