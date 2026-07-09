"""Mock-HTTP unit tests for the C6 publishing subsystem (ADR-11).

The Zenodo/HydroShare live deposits are behind the ``publish_e2e`` marker; these tests
mock the HTTP layer and pin the two-phase control flow + payload shapes so live-API field
drift (plan Assumption A4 / Risk X4) cannot silently break the contract.
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from hhemt import publishing
from hhemt.exceptions import PublishError

# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class _Model:
    """Stand-in pydantic-like config exposing model_dump(mode=...)."""

    def __init__(self, payload=None):
        self._payload = payload or {}

    def model_dump(self, mode="json"):
        return dict(self._payload)


class _FakeAnalysis:
    def __init__(self, analysis_dir):
        self.analysis_paths = SimpleNamespace(analysis_dir=analysis_dir)
        self.cfg_analysis = _Model({"analysis_id": "a1"})
        self._system = SimpleNamespace(cfg_system=_Model({"system_id": "s1"}))


def _write_sidecar(analysis_dir, spdx="CC0-1.0"):
    uri = publishing._SPDX_LICENSE_TABLE[spdx]["uri"]
    doc = {
        "@context": "https://w3id.org/ro/crate/1.1/context",
        "@graph": [
            {"@id": "ro-crate-metadata.json", "@type": "CreativeWork"},
            {"@id": "./", "@type": "Dataset", "license": {"@id": uri}},
        ],
    }
    (analysis_dir / "ro-crate-metadata.json").write_text(json.dumps(doc))


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _RecordingSession:
    """Records (method, url, json) and returns canned InvenioRDM-shaped responses."""

    RESERVED_DOI = "10.5281/zenodo.12345"

    def __init__(self):
        self.headers = {}
        self.calls = []

    def post(self, url, json=None, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self._respond(url)

    def put(self, url, json=None, data=None, headers=None, timeout=None):
        self.calls.append(("PUT", url, json))
        return self._respond(url)

    def _respond(self, url):
        if url.endswith("/api/records"):
            return _Resp(201, {"id": "12345", "links": {"self_html": "https://zenodo.org/records/12345"}})
        if url.endswith("/draft/pids/doi"):
            return _Resp(200, {"pids": {"doi": {"identifier": self.RESERVED_DOI}}})
        if url.endswith("/actions/publish"):
            return _Resp(202, {"id": "12345", "links": {"self_html": "https://zenodo.org/records/12345"}})
        return _Resp(200, {})


def _patch_zenodo_session(monkeypatch):
    sess = _RecordingSession()
    monkeypatch.setattr(publishing.requests, "Session", lambda: sess)
    monkeypatch.setenv("HHEMT_ZENODO_TOKEN", "tok")
    monkeypatch.delenv("HHEMT_ZENODO_BASE_URL", raising=False)
    return sess


# --------------------------------------------------------------------------------------
# Shared DataCite builders
# --------------------------------------------------------------------------------------


def test_rightslist_is_five_field_spdx_entry():
    rights = publishing.build_datacite_rightslist("CC0-1.0")
    assert len(rights) == 1
    entry = rights[0]
    assert set(entry) == {"rights", "rightsUri", "rightsIdentifier", "rightsIdentifierScheme", "schemeUri"}
    assert entry["rightsIdentifier"] == "CC0-1.0"
    assert entry["rightsIdentifierScheme"] == "SPDX"
    assert entry["rightsUri"] == "https://spdx.org/licenses/CC0-1.0"


def test_related_uses_iscompiledby_for_software():
    edges = publishing.build_datacite_related(software_doi="10.x/soft")
    assert len(edges) == 1
    assert edges[0]["relationType"] == "IsCompiledBy"
    assert edges[0]["relatedIdentifier"] == "10.x/soft"
    assert edges[0]["relatedIdentifierType"] == "DOI"


def test_related_adds_supplement_edge_for_paper():
    edges = publishing.build_datacite_related(software_doi="10.x/soft", paper_doi="10.x/paper")
    rels = {e["relationType"]: e["relatedIdentifier"] for e in edges}
    assert rels == {"IsCompiledBy": "10.x/soft", "IsSupplementTo": "10.x/paper"}


def test_related_empty_when_no_dois():
    assert publishing.build_datacite_related(software_doi=None) == []


# --------------------------------------------------------------------------------------
# Sidecar license read-back
# --------------------------------------------------------------------------------------


def test_read_license_from_sidecar_round_trips(tmp_path):
    _write_sidecar(tmp_path, "CC-BY-NC-4.0")
    assert publishing._read_license_from_sidecar(tmp_path) == "CC-BY-NC-4.0"


def test_read_license_from_sidecar_raises_on_unknown(tmp_path):
    doc = {"@graph": [{"@id": "./", "@type": "Dataset", "license": {"@id": "https://example/nope"}}]}
    (tmp_path / "ro-crate-metadata.json").write_text(json.dumps(doc))
    with pytest.raises(PublishError):
        publishing._read_license_from_sidecar(tmp_path)


# --------------------------------------------------------------------------------------
# Deposit set
# --------------------------------------------------------------------------------------


def test_deposit_set_contains_configs_sidecar_and_zarr(tmp_path):
    _write_sidecar(tmp_path)
    (tmp_path / "analysis_datatree.zarr").mkdir()
    analysis = _FakeAnalysis(tmp_path)
    deposit = publishing._deposit_set(analysis, "analysis_datatree.zarr")
    names = {p.name for p in deposit}
    assert names == {"ro-crate-metadata.json", "analysis_datatree.zarr", "cfg_analysis.yaml", "cfg_system.yaml"}
    # Configs are materialized from the config models (eda() precedent).
    assert (tmp_path / "cfg_analysis.yaml").exists()
    assert (tmp_path / "cfg_system.yaml").exists()


# --------------------------------------------------------------------------------------
# publish_analysis controller — override mismatch
# --------------------------------------------------------------------------------------


def test_override_license_mismatch_raises(tmp_path):
    _write_sidecar(tmp_path, "CC0-1.0")
    analysis = _FakeAnalysis(tmp_path)
    with pytest.raises(PublishError) as ei:
        publishing.publish_analysis(analysis, target="zenodo", override_dataset_license="CC-BY-NC-4.0")
    assert "reprocess" in ei.value.status


def test_override_license_match_is_accepted(tmp_path, monkeypatch):
    _write_sidecar(tmp_path, "CC0-1.0")
    (tmp_path / "analysis_datatree.zarr").mkdir()
    analysis = _FakeAnalysis(tmp_path)
    _patch_zenodo_session(monkeypatch)
    # Matching override must NOT raise; adapter runs against the mocked session.
    result = publishing.publish_analysis(analysis, target="zenodo", override_dataset_license="CC0-1.0")
    assert result["data_doi"] == _RecordingSession.RESERVED_DOI


# --------------------------------------------------------------------------------------
# Zenodo two-phase control-flow + payload-shape test (master FQ3)
# --------------------------------------------------------------------------------------


def test_zenodo_two_phase_call_order_and_payloads(tmp_path, monkeypatch):
    sess = _patch_zenodo_session(monkeypatch)
    f1 = tmp_path / "ro-crate-metadata.json"
    f1.write_text("{}")
    result = publishing._ZenodoTarget().publish(
        deposit=[f1],
        license_spdx="CC0-1.0",
        software_doi="10.5281/zenodo.999",
        analysis_dir=tmp_path,
    )

    # Milestone call order (subsequence over the recorded calls).
    seq = [(m, u) for (m, u, _b) in sess.calls]

    def idx(pred):
        return next(i for i, (m, u) in enumerate(seq) if pred(m, u))

    i_create = idx(lambda m, u: m == "POST" and u.endswith("/api/records"))
    i_reserve = idx(lambda m, u: m == "POST" and u.endswith("/draft/pids/doi"))
    i_embed = idx(lambda m, u: m == "PUT" and u.endswith("/records/12345/draft"))
    i_upload = idx(lambda m, u: m == "PUT" and u.endswith("/content"))
    i_publish = idx(lambda m, u: m == "POST" and u.endswith("/actions/publish"))
    i_backfill = idx(lambda m, u: m == "PUT" and u.endswith("/records/999/draft"))
    assert i_create < i_reserve < i_embed < i_upload < i_publish < i_backfill

    # Embed payload carries the reserved data-DOI, IsCompiledBy edge, and 5-field rightsList.
    embed_body = next(b for (m, u, b) in sess.calls if m == "PUT" and u.endswith("/records/12345/draft"))
    assert embed_body["pids"]["doi"]["identifier"] == _RecordingSession.RESERVED_DOI
    rl = embed_body["metadata"]["rightsList"]
    assert len(rl[0]) == 5 and rl[0]["rightsIdentifierScheme"] == "SPDX"
    rel = embed_body["metadata"]["relatedIdentifiers"]
    assert rel[0]["relationType"] == "IsCompiledBy" and rel[0]["relatedIdentifier"] == "10.5281/zenodo.999"

    # Backfill payload writes the reciprocal IsSourceOf edge onto the software record.
    backfill_body = next(b for (m, u, b) in sess.calls if m == "PUT" and u.endswith("/records/999/draft"))
    assert backfill_body["metadata"]["relatedIdentifiers"][0]["relationType"] == "IsSourceOf"

    assert result == {
        "target": "zenodo",
        "data_doi": _RecordingSession.RESERVED_DOI,
        "software_doi": "10.5281/zenodo.999",
        "record_url": "https://zenodo.org/records/12345",
    }


def test_zenodo_missing_token_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("HHEMT_ZENODO_TOKEN", raising=False)
    with pytest.raises(PublishError) as ei:
        publishing._ZenodoTarget().publish(
            deposit=[], license_spdx="CC0-1.0", software_doi=None, analysis_dir=tmp_path
        )
    assert "HHEMT_ZENODO_TOKEN" in ei.value.status


def test_zenodo_http_error_raises_publisherror(tmp_path, monkeypatch):
    class _FailSession(_RecordingSession):
        def _respond(self, url):
            return _Resp(500, {}, text="boom")

    monkeypatch.setattr(publishing.requests, "Session", lambda: _FailSession())
    monkeypatch.setenv("HHEMT_ZENODO_TOKEN", "tok")
    with pytest.raises(PublishError) as ei:
        publishing._ZenodoTarget().publish(
            deposit=[], license_spdx="CC0-1.0", software_doi=None, analysis_dir=tmp_path
        )
    assert "create draft failed" in ei.value.status


# --------------------------------------------------------------------------------------
# HydroShare two-invocation adapter — stops before DOI mint
# --------------------------------------------------------------------------------------


class _FakeResource:
    def __init__(self):
        self.resource_id = "abc123"
        self.metadata = SimpleNamespace(rights=None, relations=None)
        self.uploaded = []
        self.saved = False
        self.public = None

    def file_upload(self, path):
        self.uploaded.append(path)

    def save(self):
        self.saved = True

    def set_sharing_status(self, public):
        self.public = public


def _install_fake_hsclient(monkeypatch, resource):
    class _FakeHS:
        def __init__(self, username=None, password=None):
            self.username = username

        def create(self):
            return resource

    monkeypatch.setitem(sys.modules, "hsclient", SimpleNamespace(HydroShare=_FakeHS))


def test_hydroshare_stops_before_doi_and_returns_manual_step(tmp_path, monkeypatch):
    resource = _FakeResource()
    _install_fake_hsclient(monkeypatch, resource)
    monkeypatch.setenv("HHEMT_HYDROSHARE_USERNAME", "u")
    monkeypatch.setenv("HHEMT_HYDROSHARE_PASSWORD", "p")
    f1 = tmp_path / "ro-crate-metadata.json"
    f1.write_text("{}")

    result = publishing._HydroShareTarget().publish(
        deposit=[f1], license_spdx="CC0-1.0", software_doi="10.5281/zenodo.999", analysis_dir=tmp_path
    )

    assert result["data_doi"] is None
    assert result["target"] == "hydroshare"
    assert "abc123" in result["record_url"]
    assert "manual_step" in result and "web UI" in result["manual_step"]
    # Resource was made public but NOT programmatically DOI-minted.
    assert resource.public is True
    assert resource.saved is True
    assert resource.uploaded == [str(f1)]


def test_hydroshare_missing_credentials_raises(tmp_path, monkeypatch):
    _install_fake_hsclient(monkeypatch, _FakeResource())
    monkeypatch.delenv("HHEMT_HYDROSHARE_USERNAME", raising=False)
    with pytest.raises(PublishError) as ei:
        publishing._HydroShareTarget().publish(
            deposit=[], license_spdx="CC0-1.0", software_doi=None, analysis_dir=tmp_path
        )
    assert "HHEMT_HYDROSHARE_USERNAME" in ei.value.status


# --------------------------------------------------------------------------------------
# e2e skeleton (skipped without HHEMT_PUBLISH_E2E=1)
# --------------------------------------------------------------------------------------


@pytest.mark.publish_e2e
@pytest.mark.skipif(
    os.environ.get("HHEMT_PUBLISH_E2E") != "1",
    reason="requires HHEMT_PUBLISH_E2E=1 and live Zenodo sandbox credentials",
)
def test_zenodo_sandbox_e2e():  # pragma: no cover - live-credential path
    # Skeleton: exercise the Zenodo sandbox against a real deposit set.
    # Set HHEMT_ZENODO_BASE_URL=https://sandbox.zenodo.org and HHEMT_ZENODO_TOKEN.
    pytest.skip("e2e skeleton — provide a real analysis_dir + sandbox credentials to exercise")
