"""ADR-17 Phase 3 — pure invalidating-fix RO-Crate projection unit tests (D7).

Covers the ``schema:Comment`` annotation shape, the DataCite supersession
descriptor, side-effect-freeness, and — the determinism guard — that Phase 3 adds
ZERO keys to ``metadata.py::_EMBEDDED_PROV_KEYS`` (R9 / data-management OE-1). The
projection is deliberately NOT wired into the consolidation-time crate emit, so a
byte-unchanged embedded-core frozenset is the checkable invariant that the RO-Crate
byte-determinism grep-guard still holds.
"""

from __future__ import annotations

from hhemt import metadata
from hhemt.config.invalidating_fixes import InvalidatingFix
from hhemt.metadata import datacite_supersession_descriptor, project_invalidating_fix

_FULL_SHA = "0123456789abcdef0123456789abcdef01234567"
_PART_ID = "sims/evt_0001/processed/depth.zarr"


def _entry(**overrides) -> InvalidatingFix:
    base = {
        "commit_id": _FULL_SHA,
        "severity": "error",
        "affected_scope": "scenario",
        "recommended_action": "re-run",
        "affected_version_range": ">=0.8.0,<0.9.3",
        "introduced_in_version": "0.9.3",
        "description": "The manning raster was read with the wrong band index.",
        "significance": "Affects peak depth for all scenarios; non-cosmetic.",
    }
    base.update(overrides)
    return InvalidatingFix.model_validate(base)


# --------------------------------------------------------------------------- #
# project_invalidating_fix — schema:Comment annotation shape.
# --------------------------------------------------------------------------- #
def test_projection_comment_shape():
    node = project_invalidating_fix(_entry(), _PART_ID)
    assert node["@type"] == "Comment"
    assert node["@id"] == "#known-issue-0123456"  # first 7 chars of the sha
    assert node["about"] == {"@id": _PART_ID}
    assert node["identifier"] == _FULL_SHA  # FULL sha, not the truncated @id
    assert "manning raster" in node["text"]
    assert "peak depth" in node["text"]  # description + significance both present


def test_projection_property_values_are_machine_readable():
    props = {p["propertyID"]: p["value"] for p in project_invalidating_fix(_entry(), _PART_ID)["additionalProperty"]}
    assert props["invalidating-fix-severity"] == "error"
    assert props["recommended-recompute-action"] == "re-run"  # the enum VALUE, not repr
    assert props["introduced-in-version"] == "0.9.3"
    assert props["affected-version-range"] == ">=0.8.0,<0.9.3"
    # All four are schema.org-native PropertyValue nodes.
    for p in project_invalidating_fix(_entry(), _PART_ID)["additionalProperty"]:
        assert p["@type"] == "PropertyValue"


def test_projection_is_side_effect_free_and_deterministic():
    entry = _entry()
    first = project_invalidating_fix(entry, _PART_ID)
    second = project_invalidating_fix(entry, _PART_ID)
    assert first == second  # pure — identical output, no mutation of the entry


# --------------------------------------------------------------------------- #
# datacite_supersession_descriptor — IsNewVersionOf / IsPreviousVersionOf pair.
# --------------------------------------------------------------------------- #
def test_datacite_supersession_pair():
    desc = datacite_supersession_descriptor(recomputed_doi="10.5281/zenodo.NEW", affected_doi="10.5281/zenodo.OLD")
    # The re-computed dataset carries IsNewVersionOf -> the AFFECTED doi.
    assert desc["isNewVersionOf"] == {
        "relatedIdentifier": "10.5281/zenodo.OLD",
        "relatedIdentifierType": "DOI",
        "relationType": "IsNewVersionOf",
    }
    # The affected dataset carries the inverse IsPreviousVersionOf -> the RE-COMPUTED doi.
    assert desc["isPreviousVersionOf"] == {
        "relatedIdentifier": "10.5281/zenodo.NEW",
        "relatedIdentifierType": "DOI",
        "relationType": "IsPreviousVersionOf",
    }


# --------------------------------------------------------------------------- #
# Determinism guard (R9 / data-management OE-1): Phase 3 adds ZERO embedded-core keys.
# --------------------------------------------------------------------------- #
_EXPECTED_EMBEDDED_PROV_KEYS = frozenset(
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
        # reproducibility-publishing-and-durable-storage Phase 3 (R6): the frozen dataset
        # license is deterministic, so it lands in the embedded core (not the volatile sidecar).
        "license",
    }
)


def test_embedded_prov_keys_byte_unchanged_after_phase3():
    """The RO-Crate byte-determinism invariant: the projection must NOT add any key
    to the embedded core. If this fails, the projection was wrongly wired into the
    consolidation-time emit path (or _EMBEDDED_PROV_KEYS was otherwise widened)."""
    assert metadata._EMBEDDED_PROV_KEYS == _EXPECTED_EMBEDDED_PROV_KEYS
