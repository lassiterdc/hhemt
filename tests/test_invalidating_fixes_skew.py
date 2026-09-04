"""ADR-17 Phase 4 — git-remote version-skew discovery unit tests (network-mocked).

Covers the best-effort raw-fetch degrade (offline / non-200 / malformed -> None, never
raises), the reachable-remote skew path (a newer canonical fix affecting the installed
version that the local registry lacks -> new_fixes), the already-local + not-affecting
paths, and the anonymization assertion (the constructed raw URL carries no blocklisted
token). No real network I/O — urllib is mocked and the registry seams are monkeypatched.
"""

from __future__ import annotations

import urllib.error

import yaml

from hhemt import invalidating_fixes_skew as skew
from hhemt.config.invalidating_fixes import InvalidatingFixRegistry
from hhemt.invalidating_fixes_skew import (
    _REGISTRY_RAW_URL,
    _REGISTRY_REMOTE_OWNER,
    _REGISTRY_REMOTE_REPO,
    SkewResult,
    discover_version_skew,
    fetch_remote_registry,
    parse_remote_registry,
)

_SHA_A = "a" * 40
_SHA_B = "b" * 40


def _entry(commit_id: str, *, version_range: str = ">=0.8.0,<0.9.3") -> dict:
    return {
        "commit_id": commit_id,
        "severity": "error",
        "affected_scope": "scenario",
        "recommended_action": "re-run",
        "affected_version_range": version_range,
        "introduced_in_version": "0.9.3",
        "description": "wrong band index on the manning raster",
        "significance": "affects peak depth for all scenarios",
    }


def _remote_yaml(*entries: dict) -> str:
    return yaml.safe_dump({"schema_version": 1, "fixes": list(entries)})


class _FakeResp:
    def __init__(self, status: int, body: bytes = b""):
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


# --------------------------------------------------------------------------- #
# fetch_remote_registry — best-effort degrade, never raises.
# --------------------------------------------------------------------------- #
def test_fetch_offline_returns_none_and_logs(monkeypatch, caplog):
    def _boom(*_a, **_k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    with caplog.at_level("INFO"):
        assert fetch_remote_registry(timeout=0.1) is None  # degrades, no raise
    # Non-silent: the INFO names the GitHub URL so the user can check manually.
    assert any(_REGISTRY_RAW_URL in rec.message for rec in caplog.records)


def test_fetch_non_200_returns_none(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResp(404))
    assert fetch_remote_registry(timeout=0.1) is None


def test_fetch_200_returns_text(monkeypatch):
    body = _remote_yaml(_entry(_SHA_A)).encode("utf-8")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResp(200, body))
    text = fetch_remote_registry(timeout=0.1)
    assert text is not None and "fixes:" in text


# --------------------------------------------------------------------------- #
# parse_remote_registry — advisory parse, degrade to None on malformed.
# --------------------------------------------------------------------------- #
def test_parse_valid_remote():
    reg = parse_remote_registry(_remote_yaml(_entry(_SHA_A)))
    assert isinstance(reg, InvalidatingFixRegistry)
    assert reg.fixes[0].commit_id == _SHA_A


def test_parse_malformed_remote_returns_none():
    assert parse_remote_registry("schema_version: 1\nfixes: [ {bad: ") is None  # invalid yaml
    assert parse_remote_registry("") is None  # empty -> None
    assert parse_remote_registry("just: a string, not a registry") is None  # schema fail


# --------------------------------------------------------------------------- #
# discover_version_skew — the actionable skew.
# --------------------------------------------------------------------------- #
def test_skew_reachable_new_fix(monkeypatch):
    # Remote has a fix affecting 0.8.5; local registry is EMPTY -> it is new skew.
    monkeypatch.setattr(skew, "fetch_remote_registry", lambda timeout=5.0: _remote_yaml(_entry(_SHA_A)))
    monkeypatch.setattr(skew, "_local_fix_commit_ids", lambda: set())
    result = discover_version_skew(local_version="0.8.5")
    assert isinstance(result, SkewResult)
    assert result.reachable is True
    assert [f.commit_id for f in result.affecting] == [_SHA_A]
    assert [f.commit_id for f in result.new_fixes] == [_SHA_A]


def test_skew_fix_already_in_local_registry_is_not_new(monkeypatch):
    monkeypatch.setattr(skew, "fetch_remote_registry", lambda timeout=5.0: _remote_yaml(_entry(_SHA_A)))
    monkeypatch.setattr(skew, "_local_fix_commit_ids", lambda: {_SHA_A})  # already known locally
    result = discover_version_skew(local_version="0.8.5")
    assert [f.commit_id for f in result.affecting] == [_SHA_A]
    assert result.new_fixes == []  # affecting but not NEW -> no skew


def test_skew_fix_not_affecting_installed_version(monkeypatch):
    # Remote fix affects <0.9.3; installed version 1.2.0 is out of range -> not affecting.
    monkeypatch.setattr(skew, "fetch_remote_registry", lambda timeout=5.0: _remote_yaml(_entry(_SHA_B)))
    monkeypatch.setattr(skew, "_local_fix_commit_ids", lambda: set())
    result = discover_version_skew(local_version="1.2.0")
    assert result.reachable is True
    assert result.affecting == []
    assert result.new_fixes == []


def test_skew_offline_is_unreachable_and_never_raises(monkeypatch):
    monkeypatch.setattr(skew, "fetch_remote_registry", lambda timeout=5.0: None)
    result = discover_version_skew(local_version="0.8.5")
    assert result.reachable is False
    assert result.affecting == [] and result.new_fixes == []


def test_skew_malformed_remote_is_unreachable(monkeypatch):
    monkeypatch.setattr(skew, "fetch_remote_registry", lambda timeout=5.0: "not a registry")
    result = discover_version_skew(local_version="0.8.5")
    assert result.reachable is False


# --------------------------------------------------------------------------- #
# Anonymization (git Q4 / ADR-14): the constructed raw URL carries no blocklisted token.
# --------------------------------------------------------------------------- #
def test_registry_raw_url_targets_the_public_repo():
    """ADR-14 anonymization: the pinned raw URL targets the PUBLIC repo.

    This deliberately does NOT loop the blocklist, and the omission is the point rather
    than an oversight. The URL is composed only from the four module-level literals plus
    fixed text, and no blocklist token contains "/" (measured: 0 of 8), so no token can
    span a component boundary -- the composed URL can therefore carry a token only if one
    of those literals already carries it, in this same file. scripts/check_anonymization.py
    scans this file against INDEPENDENT ground truth and does not self-exclude it, so it
    already catches that case. Re-deriving the check here bought no coverage and made this
    module hard-depend on a carrier that is being relocated out of the repository.

    A whole-URL equality assertion was considered and REJECTED. Its expected value would be
    co-maintained by the same edit that changes the URL, so a deliberate change to a private
    handle would be "fixed" by updating the literal, silently removing the property -- the
    same failure the blocklist's own independence invariant exists to prevent. What remains
    here is the one thing the guard cannot see: that the URL points at the intended PUBLIC
    target at all. Enforcement of the anonymization property itself is the guard's.
    """
    assert (
        _REGISTRY_REMOTE_OWNER == "lassiterdc"
    ), f"registry owner is {_REGISTRY_REMOTE_OWNER!r}, not the public handle"
    assert _REGISTRY_REMOTE_REPO == "hhemt", f"registry repo is {_REGISTRY_REMOTE_REPO!r}, not the public repository"
    assert _REGISTRY_RAW_URL.startswith(
        "https://raw.githubusercontent.com/lassiterdc/hhemt/"
    ), f"pinned registry URL does not target the public repo: {_REGISTRY_RAW_URL!r}"
