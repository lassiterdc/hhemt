"""Unit tests for the CaseManifest schema and the case-manifest helper (ADR-12)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from hhemt import examples as ex
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
    ex.TRITON_SWMM_example._connect_to_hydroshare("res123")
    assert calls["resource"] == 1
    assert calls["sign_in"] == 0


def test_connect_falls_back_to_sign_in(monkeypatch):
    # R3: anonymous read raises -> sign_in called.
    calls = {"sign_in": 0}

    class FakeHS:
        def resource(self, rid, validate=True):
            raise RuntimeError("403 not anonymously accessible")

        def sign_in(self):
            calls["sign_in"] += 1

    monkeypatch.setattr(ex, "HydroShare", lambda: FakeHS())
    ex.TRITON_SWMM_example._connect_to_hydroshare("res123")
    assert calls["sign_in"] == 1


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
    ex.TRITON_SWMM_example._verify_manifest(bag, good)
    # mismatched manifest: raises
    with pytest.raises(ProcessingError):
        ex.TRITON_SWMM_example._verify_manifest(bag, bad)


def test_verify_manifest_raises_on_absent_file(tmp_path):
    # R4: a manifest entry naming an absent file raises ProcessingError.
    from hhemt.exceptions import ProcessingError
    bag = tmp_path / "bagroot"
    bag.mkdir()
    with pytest.raises(ProcessingError):
        ex.TRITON_SWMM_example._verify_manifest(bag, {"data/contents/missing.txt": "ab" * 32})


def test_verify_manifest_empty_is_noop(tmp_path):
    # R4: an empty manifest skips the sha256 check (no raise).
    bag = tmp_path / "bagroot"
    bag.mkdir()
    ex.TRITON_SWMM_example._verify_manifest(bag, {})


def test_manifest_generation_verification_parity(tmp_path):
    # Flag-1 guard: compute_manifest's keys must verify against the same bag root.
    bag = tmp_path / "bagroot"
    (bag / "data" / "contents").mkdir(parents=True)
    (bag / "data" / "contents" / "a.txt").write_bytes(b"x")
    (bag / "bagit.txt").write_bytes(b"BagIt-Version: 0.97")
    manifest = gcm.compute_manifest(bag)            # keys relative to bag root
    ex.TRITON_SWMM_example._verify_manifest(bag, manifest)  # must not raise
