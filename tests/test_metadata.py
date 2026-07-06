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


def test_sidecar_compare_and_write_idempotent(tmp_path):
    # R4: a second write of identical content returns False (no rewrite / mtime bump).
    g = metadata.canonical_jsonld(_build())
    assert metadata.write_rocrate_sidecar(tmp_path, graph_json=g) is True
    mtime1 = (tmp_path / "ro-crate-metadata.json").stat().st_mtime_ns
    assert metadata.write_rocrate_sidecar(tmp_path, graph_json=g) is False
    assert (tmp_path / "ro-crate-metadata.json").stat().st_mtime_ns == mtime1
