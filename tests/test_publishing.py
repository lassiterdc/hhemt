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
            {"@id": "./", "@type": "Dataset", "name": "Test Analysis", "license": {"@id": uri}},
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

    MINTED_DOI = "10.5281/zenodo.12345"

    def __init__(self):
        self.headers = {}
        self.calls = []

    def post(self, url, json=None, data=None, headers=None, timeout=None):
        self.calls.append(("POST", url, json))
        return self._respond(url)

    def put(self, url, json=None, data=None, headers=None, timeout=None):
        self.calls.append(("PUT", url, json))
        return self._respond(url)

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None))
        return self._respond(url)

    def _respond(self, url):
        if url.endswith("/api/records"):
            return _Resp(201, {"id": "12345", "links": {"self_html": "https://zenodo.org/records/12345"}})
        if url.endswith("/api/records/12345"):
            # Record GET after publish. Verified sandbox shape (record 565599,
            # 2026-07-15): the minted DOI is at top-level ``doi`` / ``metadata.doi``,
            # ``pids`` is null. This is what the round-trip reads the DOI from.
            return _Resp(
                200,
                {
                    "id": "12345",
                    "doi": self.MINTED_DOI,
                    "metadata": {"doi": self.MINTED_DOI},
                    "pids": None,
                    "links": {"self_html": "https://zenodo.org/records/12345"},
                },
            )
        if url.endswith("/actions/publish"):
            # Sandbox publish-action response does NOT carry an extractable DOI
            # (``pids`` null); the DOI is authoritative on the record GET above.
            return _Resp(
                202,
                {
                    "id": "12345",
                    "pids": None,
                    "links": {"self_html": "https://zenodo.org/records/12345"},
                },
            )
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


def test_inveniordm_rights_is_lowercased_spdx_vocab_id():
    assert publishing.build_inveniordm_rights("CC0-1.0") == [{"id": "cc0-1.0"}]
    assert publishing.build_inveniordm_rights("CC-BY-NC-4.0") == [{"id": "cc-by-nc-4.0"}]


def test_inveniordm_related_uses_lowercase_relation_vocab():
    edges = publishing.build_inveniordm_related(software_doi="10.x/soft")
    assert edges == [
        {"identifier": "10.x/soft", "scheme": "doi", "relation_type": {"id": "iscompiledby"}}
    ]
    assert publishing.build_inveniordm_related(software_doi=None) == []


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
    assert result["data_doi"] == _RecordingSession.MINTED_DOI


def test_publish_analysis_reprex_bundle_threads_container_defs(tmp_path, monkeypatch):
    """The deposit path threads ``container_defs`` into ``emit_bundle`` (ADR-19 multi-SIF).

    Regression for the deposit-facade gap: ``publish_analysis(deposit_source="reprex_bundle")``
    (and the ``publish_reprex_bundle`` facades that route through it) dropped ``container_defs``,
    so ``emit_bundle``'s container branch fail-closed on any container-mode analysis. The spy
    on ``emit_bundle`` captures the kwarg; the mocked session runs the rest of the deposit.
    """
    _write_sidecar(tmp_path, "CC0-1.0")
    real_zip = tmp_path / "bundle.zip"
    real_zip.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty-zip EOCD record
    analysis = _FakeAnalysis(tmp_path)
    captured: dict = {}

    def _spy_emit(a, *, container_defs=None, **kw):
        captured["container_defs"] = container_defs
        return real_zip

    monkeypatch.setattr("hhemt.bundle.emit_bundle", _spy_emit)
    _patch_zenodo_session(monkeypatch)

    defs = [tmp_path / "uva-cuda.def", tmp_path / "uva-cuda-a6000.def", tmp_path / "uva-cpu.def"]
    result = publishing.publish_analysis(
        analysis, target="zenodo", deposit_source="reprex_bundle", container_defs=defs
    )
    assert captured["container_defs"] == defs
    assert result["data_doi"] == _RecordingSession.MINTED_DOI


def test_publish_analysis_reprex_bundle_defaults_container_defs_none(tmp_path, monkeypatch):
    """Native-mode regression: omitting ``container_defs`` passes ``None`` (byte-identical
    to the pre-fix behavior — the deposit path is default-preserving)."""
    _write_sidecar(tmp_path, "CC0-1.0")
    real_zip = tmp_path / "bundle.zip"
    real_zip.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    analysis = _FakeAnalysis(tmp_path)
    captured: dict = {}

    def _spy_emit(a, *, container_defs=None, **kw):
        captured["container_defs"] = container_defs
        return real_zip

    monkeypatch.setattr("hhemt.bundle.emit_bundle", _spy_emit)
    _patch_zenodo_session(monkeypatch)

    publishing.publish_analysis(analysis, target="zenodo", deposit_source="reprex_bundle")
    assert captured["container_defs"] is None


# --------------------------------------------------------------------------------------
# Zenodo two-phase control-flow + payload-shape test (master FQ3)
# --------------------------------------------------------------------------------------


def test_zenodo_lets_zenodo_mint_and_embeds_native_metadata(tmp_path, monkeypatch):
    sess = _patch_zenodo_session(monkeypatch)
    # Sidecar root name -> metadata.title (InvenioRDM-mandatory).
    _write_sidecar(tmp_path, "CC0-1.0")
    result = publishing._ZenodoTarget().publish(
        deposit=[tmp_path / "ro-crate-metadata.json"],
        license_spdx="CC0-1.0",
        software_doi="10.5281/zenodo.999",
        analysis_dir=tmp_path,
    )

    # Milestone call order (subsequence over the recorded calls). NO reserve step.
    seq = [(m, u) for (m, u, _b) in sess.calls]

    def idx(pred):
        return next(i for i, (m, u) in enumerate(seq) if pred(m, u))

    i_create = idx(lambda m, u: m == "POST" and u.endswith("/api/records"))
    i_embed = idx(lambda m, u: m == "PUT" and u.endswith("/records/12345/draft"))
    i_upload = idx(lambda m, u: m == "PUT" and u.endswith("/content"))
    i_publish = idx(lambda m, u: m == "POST" and u.endswith("/actions/publish"))
    i_backfill = idx(lambda m, u: m == "PUT" and u.endswith("/records/999/draft"))
    assert i_create < i_embed < i_upload < i_publish < i_backfill

    # NO reserve-DOI call: Zenodo mints on publish (self-supplying pids caused the 400).
    assert not any(u.endswith("/draft/pids/doi") for (_m, u) in seq)

    # Embed carries the four InvenioRDM-mandatory fields + native rights/related, NO pids.
    embed_body = next(b for (m, u, b) in sess.calls if m == "PUT" and u.endswith("/records/12345/draft"))
    md = embed_body["metadata"]
    assert md["title"] == "Test Analysis"
    assert md["creators"][0]["person_or_org"]["type"] in ("personal", "organizational")
    assert md["publication_date"]  # EDTF Level 0 date string
    assert md["resource_type"] == {"id": "dataset"}
    assert md["publisher"] == "Zenodo"  # DataCite-mandatory for DOI registration
    assert md["rights"] == [{"id": "cc0-1.0"}]
    assert md["related_identifiers"] == [
        {"identifier": "10.5281/zenodo.999", "scheme": "doi", "relation_type": {"id": "iscompiledby"}}
    ]
    assert "pids" not in embed_body

    # Backfill writes the reciprocal issourceof edge (native shape) onto the software record.
    backfill_body = next(b for (m, u, b) in sess.calls if m == "PUT" and u.endswith("/records/999/draft"))
    assert backfill_body["metadata"]["related_identifiers"][0]["relation_type"] == {"id": "issourceof"}

    # The minted DOI is read from the PUBLISH response.
    assert result == {
        "target": "zenodo",
        "data_doi": _RecordingSession.MINTED_DOI,
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


# ---------------------------------------------------------------------------
# ADR-20 — Option-B pre-upload size validator + attempt-and-surface backstop.
# ---------------------------------------------------------------------------


def test_size_validator_sums_directory_trees_not_inode_size(tmp_path):
    """OE-2 (a green-check-that-certifies-nothing defect, found at Phase-3 preflight).

    ``Path.stat().st_size`` on a DIRECTORY returns the ~4 KB inode size. The deposit set's
    heavyweight member is the consolidated zarr STORE — a directory. Stat-ing it would
    report a 20 GB deposit as fitting comfortably inside HydroShare's 20 GB quota, which is
    precisely the "CI check whose green is misleading" class.
    """
    from hhemt.publishing import _path_size_bytes, validate_deposit_size

    store = tmp_path / "analysis_datatree.zarr"
    (store / "chunks").mkdir(parents=True)
    for i in range(4):
        (store / "chunks" / f"c{i}").write_bytes(b"x" * 1000)

    assert _path_size_bytes(store) == 4000  # summed, not stat()'d
    assert store.stat().st_size != 4000  # the trap this guards

    report = validate_deposit_size([store], "hydroshare")
    assert report["total_bytes"] == 4000
    assert report["fits"] is True


def test_size_validator_reports_overflow_and_remediation(tmp_path):
    from hhemt.publishing import _TARGET_LIMITS, validate_deposit_size

    big = tmp_path / "big.zip"
    big.write_bytes(b"x" * 16)
    # Shrink the documented limit rather than writing 20 GB to disk.
    original = _TARGET_LIMITS["hydroshare"]["max_total_bytes"]
    _TARGET_LIMITS["hydroshare"]["max_total_bytes"] = 8
    try:
        report = validate_deposit_size([big], "hydroshare")
    finally:
        _TARGET_LIMITS["hydroshare"]["max_total_bytes"] = original

    assert report["fits"] is False
    assert report["overflow_bytes"] == 8
    assert report["as_of"]  # the constant carries its verification date
    assert any("exclude" in r for r in report["remediation"])


def test_classify_storage_error_reframes_quota_signal_and_passes_others_through():
    """G2 backstop: a quota rejection is reframed with the emit-side delta; a NON-storage
    failure is passed through (returns None) so the caller re-raises it verbatim — a
    mislabelled auth error would send the operator hunting a disk-space problem."""
    from hhemt.publishing import _classify_storage_error

    quota = _classify_storage_error(
        "Request Entity Too Large: user quota exceeded", 30 * 1000**3, 20 * 1000**3
    )
    assert quota is not None
    assert "STORAGE/QUOTA" in quota
    assert "over by 10.00 GB" in quota

    assert _classify_storage_error("401 Unauthorized: bad token", 1, 100) is None


def test_check_body_window_is_wide_enough_for_a_quota_signal():
    """OE-3: ``_check`` truncates the server body, and ``_classify_storage_error``
    string-matches that already-captured body. A quota message sitting past the old 300-char
    window was invisible, silently degrading the backstop to a verbatim passthrough."""
    import inspect

    from hhemt.publishing import _check

    assert "resp.text[:1000]" in inspect.getsource(_check)
