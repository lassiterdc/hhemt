"""Unit tests for the CaseManifest schema and the case-manifest helper (ADR-12)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from hhemt import experiments as ex
from hhemt import generate_case_manifest as gcm
from hhemt.config.case_manifest import CaseManifest


def test_casemanifest_minimal_backcompat():
    # R2: the legacy 2-field case.yaml validates unchanged.
    m = CaseManifest.model_validate(
        {"case_name": "norfolk case study", "res_identifier": "a4aace329b8c401a93e94ce2a761fe1b"}
    )
    assert m.host == "hydroshare"
    assert m.manifest == {}
    assert m.resource_version is None


def test_casemanifest_full():
    # R1: all fields validate.
    m = CaseManifest.model_validate(
        {
            "case_name": "x",
            "res_identifier": "y",
            "resource_version": "v1",
            "description": "d",
            "citation": "c",
            "manifest": {"data/contents/dem.dem": "ab" * 32},
            "host": "hydroshare",
        }
    )
    assert m.manifest["data/contents/dem.dem"] == "ab" * 32


def test_casemanifest_rejects_unknown_key():
    # R1: extra="forbid".
    with pytest.raises(Exception):
        CaseManifest.model_validate({"case_name": "x", "res_identifier": "y", "bogus": 1})


def test_compute_manifest(tmp_path: Path):
    # R5: per-file sha256, posix-relative keys.
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_bytes(b"hello")
    (tmp_path / "sub" / "b.txt").write_bytes(b"world")
    manifest = gcm.compute_manifest(tmp_path)
    assert manifest == {
        "a.txt": hashlib.sha256(b"hello").hexdigest(),
        "sub/b.txt": hashlib.sha256(b"world").hexdigest(),
    }


def test_compute_manifest_streaming_digest_parity(tmp_path: Path):
    # Phase-1 guard: the streaming 1 MiB-chunk sha256 in compute_manifest must be
    # byte-identical to a whole-file hashlib.sha256(data) digest for a >1 MiB file
    # (exercises the chunk loop across multiple reads; proves no digest regression).
    data = b"x" * (2 << 20)  # 2 MiB -> spans two full 1 MiB chunks
    (tmp_path / "big.bin").write_bytes(data)
    manifest = gcm.compute_manifest(tmp_path)
    assert manifest == {"big.bin": hashlib.sha256(data).hexdigest()}


def test_populate_case_yaml_roundtrips(tmp_path: Path):
    # R5: the CLI core writes a schema-valid case.yaml with the computed manifest.
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "f.txt").write_bytes(b"data")
    case_yaml = tmp_path / "case.yaml"
    case_yaml.write_text(yaml.safe_dump({"case_name": "x", "res_identifier": "y"}))
    updated = gcm.populate_case_yaml(bag, case_yaml)
    assert updated.manifest == {"f.txt": hashlib.sha256(b"data").hexdigest()}
    reloaded = CaseManifest.model_validate(yaml.safe_load(case_yaml.read_text()))
    assert reloaded.manifest == updated.manifest


def test_connect_anonymous_first(monkeypatch):
    # R3: anonymous read succeeds -> sign_in NOT called.
    calls = {"resource": 0, "sign_in": 0}

    class FakeHS:
        def resource(self, rid, validate=True):
            calls["resource"] += 1
            return object()

        def sign_in(self):
            calls["sign_in"] += 1

    monkeypatch.setattr(ex, "HydroShare", lambda: FakeHS())
    ex.TRITON_SWMM_experiment._connect_to_hydroshare("res123")
    assert calls["resource"] == 1
    assert calls["sign_in"] == 0


def test_connect_falls_back_to_sign_in(monkeypatch):
    # R3 tier 3: anonymous read raises AND no env credentials -> interactive sign_in.
    # delenv makes this deterministic regardless of the ambient shell (a developer who
    # sourced e2e.env would otherwise take the new tier-2 credentialed path).
    monkeypatch.delenv("HHEMT_HYDROSHARE_USERNAME", raising=False)
    monkeypatch.delenv("HHEMT_HYDROSHARE_PASSWORD", raising=False)
    calls = {"sign_in": 0}

    class FakeHS:
        def resource(self, rid, validate=True):
            raise RuntimeError("403 not anonymously accessible")

        def sign_in(self):
            calls["sign_in"] += 1

    monkeypatch.setattr(ex, "HydroShare", lambda: FakeHS())
    ex.TRITON_SWMM_experiment._connect_to_hydroshare("res123")
    assert calls["sign_in"] == 1


def test_connect_uses_env_credentials_for_private_resource(monkeypatch):
    """Tier 2: anonymous read fails on a PRIVATE resource, but HHEMT_HYDROSHARE_* are set
    -> authenticate non-interactively and DO NOT prompt. This is what makes private-resource
    retrieval work headlessly (batch job / HPC node / automated test)."""
    monkeypatch.setenv("HHEMT_HYDROSHARE_USERNAME", "someuser")
    monkeypatch.setenv("HHEMT_HYDROSHARE_PASSWORD", "somepass")
    seen = {"anon_ctor": 0, "auth_ctor_creds": None, "sign_in": 0}

    class AnonHS:
        def resource(self, rid, validate=True):
            raise RuntimeError("403 not anonymously accessible")

        def sign_in(self):
            seen["sign_in"] += 1  # must never be called on the credentialed path

    class AuthHS:
        def resource(self, rid, validate=True):
            return object()  # authenticated read succeeds

    def fake_hydroshare(*args, **kwargs):
        if kwargs.get("username"):
            seen["auth_ctor_creds"] = (kwargs["username"], kwargs["password"])
            return AuthHS()
        seen["anon_ctor"] += 1
        return AnonHS()

    monkeypatch.setattr(ex, "HydroShare", fake_hydroshare)
    ex.TRITON_SWMM_experiment._connect_to_hydroshare("res123")

    assert seen["auth_ctor_creds"] == ("someuser", "somepass")  # env creds were used
    assert seen["sign_in"] == 0  # interactive prompt was NOT reached


def test_verify_manifest_raises_on_mismatch(tmp_path):
    # R4: a non-empty manifest with a wrong sha256 raises ProcessingError.
    from hhemt.exceptions import ProcessingError
    bag = tmp_path / "bagroot"
    (bag / "data" / "contents").mkdir(parents=True)
    f = bag / "data" / "contents" / "a.txt"
    f.write_bytes(b"hello")
    good = {"data/contents/a.txt": hashlib.sha256(b"hello").hexdigest()}
    bad = {"data/contents/a.txt": "00" * 32}
    # matching manifest: no raise
    ex.TRITON_SWMM_experiment._verify_manifest(bag, good)
    # mismatched manifest: raises
    with pytest.raises(ProcessingError):
        ex.TRITON_SWMM_experiment._verify_manifest(bag, bad)


def test_verify_manifest_raises_on_absent_file(tmp_path):
    # R4: a manifest entry naming an absent file raises ProcessingError.
    from hhemt.exceptions import ProcessingError
    bag = tmp_path / "bagroot"
    bag.mkdir()
    with pytest.raises(ProcessingError):
        ex.TRITON_SWMM_experiment._verify_manifest(bag, {"data/contents/missing.txt": "ab" * 32})


def test_verify_manifest_empty_is_noop(tmp_path):
    # R4: an empty manifest skips the sha256 check (no raise).
    bag = tmp_path / "bagroot"
    bag.mkdir()
    ex.TRITON_SWMM_experiment._verify_manifest(bag, {})


def test_manifest_generation_verification_parity(tmp_path):
    # Flag-1 guard: compute_manifest's keys must verify against the same bag root.
    bag = tmp_path / "bagroot"
    (bag / "data" / "contents").mkdir(parents=True)
    (bag / "data" / "contents" / "a.txt").write_bytes(b"x")
    (bag / "bagit.txt").write_bytes(b"BagIt-Version: 0.97")
    manifest = gcm.compute_manifest(bag)            # keys relative to bag root
    ex.TRITON_SWMM_experiment._verify_manifest(bag, manifest)  # must not raise


def test_casemanifest_zenodo_requires_doi_or_pid():
    # R3: host="zenodo" with neither doi nor pid fails Pydantic validation.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CaseManifest.model_validate(
            {"case_name": "x", "res_identifier": "y", "host": "zenodo"}
        )


def test_casemanifest_zenodo_with_doi_or_pid_validates():
    # R3: host="zenodo" validates when a doi (or a pid alone) is present.
    by_doi = CaseManifest.model_validate(
        {"case_name": "x", "res_identifier": "y", "host": "zenodo", "doi": "10.5281/zenodo.1"}
    )
    assert by_doi.host == "zenodo"
    assert by_doi.doi == "10.5281/zenodo.1"
    by_pid = CaseManifest.model_validate(
        {"case_name": "x", "res_identifier": "y", "host": "zenodo", "pid": "1234567"}
    )
    assert by_pid.pid == "1234567"


def test_download_data_from_zenodo_writes_and_verifies(tmp_path, monkeypatch):
    # R4: host="zenodo" fetch resolves the record id from the DOI, downloads each
    # file, then runs the host-agnostic _verify_manifest (passes on sha256 match).
    import requests

    content = b"zenodo-bag-bytes"
    sha = hashlib.sha256(content).hexdigest()

    class FakeStreamResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=1):
            yield content

    class FakeMetaResp:
        status_code = 200

        def json(self):
            return {
                "files": [
                    {
                        "key": "data/contents/a.txt",
                        "links": {"self": "https://zenodo.org/api/files/abc/a.txt"},
                    }
                ]
            }

    def fake_get(url, *args, **kwargs):
        if "/api/records/" in url:
            return FakeMetaResp()
        return FakeStreamResp()

    monkeypatch.setattr(requests, "get", fake_get)

    cm = CaseManifest.model_validate(
        {
            "case_name": "z",
            "res_identifier": "unused",
            "host": "zenodo",
            "doi": "10.5281/zenodo.123456",
            "manifest": {"data/contents/a.txt": sha},
        }
    )
    target = tmp_path / "bag"
    ex.TRITON_SWMM_experiment._download_data_from_zenodo(
        cm, target, download_if_exists=False, expected_manifest=cm.manifest
    )
    assert (target / "data" / "contents" / "a.txt").read_bytes() == content


def test_download_data_from_zenodo_honors_base_url_env(tmp_path, monkeypatch):
    # R5/R8 round-trip symmetry: the fetch-back host must be resolved from
    # HHEMT_ZENODO_BASE_URL (mirroring publishing._ZenodoTarget.publish), so a
    # sandbox-minted record resolves against the sandbox instead of 404ing on
    # production. Locks against a future refactor re-hardcoding the host.
    import requests

    monkeypatch.setenv("HHEMT_ZENODO_BASE_URL", "https://sandbox.zenodo.org")

    captured = []

    class FakeMetaResp:
        status_code = 200

        def json(self):
            return {"files": []}

    def fake_get(url, *args, **kwargs):
        captured.append(url)
        return FakeMetaResp()

    monkeypatch.setattr(requests, "get", fake_get)

    cm = CaseManifest.model_validate(
        {
            "case_name": "z",
            "res_identifier": "unused",
            "host": "zenodo",
            "doi": "10.5072/zenodo.565605",
            "manifest": {},
        }
    )
    ex.TRITON_SWMM_experiment._download_data_from_zenodo(
        cm, tmp_path / "bag", download_if_exists=False, expected_manifest={}
    )
    assert captured == ["https://sandbox.zenodo.org/api/records/565605"]


def test_download_data_from_zenodo_unresolvable_recid_raises(tmp_path):
    # R4: host="zenodo" with a non-zenodo DOI and no pid cannot resolve a record id.
    from hhemt.exceptions import ProcessingError

    cm = CaseManifest.model_construct(
        case_name="z", res_identifier="unused", host="zenodo", doi=None, pid=None, manifest={}
    )
    with pytest.raises(ProcessingError):
        ex.TRITON_SWMM_experiment._download_data_from_zenodo(cm, tmp_path / "bag")
