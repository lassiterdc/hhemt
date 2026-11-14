"""`_bundle_is_container_mode` keys on the manifest key the emitter actually writes (ADR-21).

Spec 32 retired `bundle_manifest.json["container_build"]` for `["sif_manifests"]`; this pins the
reprex-side reader to the same key so a container bundle is never read as native (which would
silently turn a missing digest into a vacuous pass — `sif_verified = None if container else True`).
"""

from __future__ import annotations

import json
from pathlib import Path

from hhemt.bundle._reprex import _bundle_is_container_mode


def _write_manifest(root: Path, payload: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "bundle_manifest.json").write_text(json.dumps(payload))
    return root


def test_container_bundle_is_recognised_by_sif_manifests(tmp_path):
    payload = {"bundle_schema_version": 6, "sif_manifests": [{"sha256": "0" * 64, "identity": {}}]}
    root = _write_manifest(tmp_path / "container", payload)
    assert _bundle_is_container_mode(root) is True


def test_native_bundle_reads_false(tmp_path):
    root = _write_manifest(tmp_path / "native", {"bundle_schema_version": 6})
    assert _bundle_is_container_mode(root) is False


def test_legacy_container_build_key_no_longer_counts(tmp_path):
    # The retired key is NOT a container marker any more: a manifest carrying only it is
    # pre-ADR-21 (rejected on schema upstream) and must not be read as a container bundle here.
    root = _write_manifest(tmp_path / "legacy", {"bundle_schema_version": 5, "container_build": {"defs": []}})
    assert _bundle_is_container_mode(root) is False


def test_absent_manifest_reads_false(tmp_path):
    assert _bundle_is_container_mode(tmp_path / "nowhere") is False
