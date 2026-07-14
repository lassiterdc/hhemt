"""ADR-17 Phase 3 — invalidating-fix registry loader + report-check unit tests.

Covers the Pydantic loader (valid load, fail-loud on a malformed entry, duplicate-
commit_id rejection, the REQUIRED SpecifierSet ``affected_version_range`` + its
malformed-range raise + the descriptive ``introduced_in_version`` soft inconsistency
warn), the graceful-absent Phase-2 consuming seam, and the ``check_invalidating_fixes``
report surface (match / no-match). No real zarr trees are needed — the report check is
exercised through a monkeypatched resolver so the CheckResult mapping is pinned.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hhemt import recompute
from hhemt.analysis_validation import check_invalidating_fixes
from hhemt.config.invalidating_fixes import (
    InvalidatingFixRegistry,
    load_invalidating_fixes,
)
from hhemt.recompute import RecomputeAction, RegistryMatch

_FULL_SHA = "0123456789abcdef0123456789abcdef01234567"  # 40-char full sha


def _entry(**overrides) -> dict:
    """A valid registry-entry dict; override individual fields per test."""
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
    return base


def _registry(*entries) -> dict:
    return {"schema_version": 1, "fixes": list(entries)}


# --------------------------------------------------------------------------- #
# The shipped package-data registry.
# --------------------------------------------------------------------------- #
def test_shipped_registry_loads_and_is_empty():
    """The package-data registry ships valid and EMPTY (zero matches until a real
    fix is registered)."""
    reg = load_invalidating_fixes()
    assert isinstance(reg, InvalidatingFixRegistry)
    assert reg.schema_version == 1
    assert reg.fixes == []


def test_load_from_explicit_path(tmp_path):
    import yaml

    p = tmp_path / "reg.yaml"
    p.write_text(yaml.safe_dump(_registry(_entry())))
    reg = load_invalidating_fixes(registry_path=p)
    assert len(reg.fixes) == 1
    assert reg.fixes[0].commit_id == _FULL_SHA


def test_empty_top_level_yaml_raises(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")  # parses to None
    with pytest.raises(ValueError, match="parsed to None"):
        load_invalidating_fixes(registry_path=p)


# --------------------------------------------------------------------------- #
# Valid entry + enum coercion.
# --------------------------------------------------------------------------- #
def test_valid_entry_loads_and_coerces_action_to_enum():
    reg = InvalidatingFixRegistry.model_validate(_registry(_entry()))
    fix = reg.fixes[0]
    assert isinstance(fix.recommended_action, RecomputeAction)
    assert fix.recommended_action is RecomputeAction.RE_RUN
    assert fix.recommended_action.value == "re-run"


# --------------------------------------------------------------------------- #
# Fail-loud on a malformed entry.
# --------------------------------------------------------------------------- #
def test_missing_required_field_raises():
    bad = _entry()
    del bad["severity"]
    with pytest.raises(ValidationError):
        InvalidatingFixRegistry.model_validate(_registry(bad))


def test_unknown_recommended_action_raises():
    with pytest.raises(ValidationError):
        InvalidatingFixRegistry.model_validate(_registry(_entry(recommended_action="upgrade")))


# --------------------------------------------------------------------------- #
# Duplicate commit_id rejection (ambiguity-refusal, C10).
# --------------------------------------------------------------------------- #
def test_duplicate_commit_id_raises():
    with pytest.raises(ValidationError, match="duplicate commit_id"):
        InvalidatingFixRegistry.model_validate(_registry(_entry(), _entry()))


def test_distinct_commit_ids_ok():
    other = "f" * 40
    reg = InvalidatingFixRegistry.model_validate(_registry(_entry(), _entry(commit_id=other)))
    assert len(reg.fixes) == 2


# --------------------------------------------------------------------------- #
# affected_version_range — REQUIRED + SpecifierSet-validated (T1 Option C).
# --------------------------------------------------------------------------- #
def test_affected_version_range_is_required():
    bad = _entry()
    del bad["affected_version_range"]
    with pytest.raises(ValidationError):
        InvalidatingFixRegistry.model_validate(_registry(bad))


def test_malformed_affected_version_range_raises():
    with pytest.raises(ValidationError, match="SpecifierSet"):
        InvalidatingFixRegistry.model_validate(_registry(_entry(affected_version_range="not-a-range")))


# --------------------------------------------------------------------------- #
# introduced_in_version — descriptive; SOFT inconsistency warn (never raises).
# --------------------------------------------------------------------------- #
def test_inconsistent_version_pair_warns_but_loads():
    # upper bound (<0.9.3) disagrees with introduced_in_version (1.0.0) -> warn.
    with pytest.warns(UserWarning, match="introduced_in_version"):
        reg = InvalidatingFixRegistry.model_validate(_registry(_entry(introduced_in_version="1.0.0")))
    assert reg.fixes[0].introduced_in_version == "1.0.0"  # loaded, not rejected


def test_consistent_version_pair_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any of OUR UserWarnings would fail here
        reg = InvalidatingFixRegistry.model_validate(_registry(_entry()))
    assert reg.fixes[0].introduced_in_version == "0.9.3"


# --------------------------------------------------------------------------- #
# Phase-2 consuming seam — graceful-absent degrade.
# --------------------------------------------------------------------------- #
def test_seam_returns_empty_for_shipped_empty_registry():
    # _load_invalidating_fix_registry wires load_invalidating_fixes().fixes.
    assert recompute._load_invalidating_fix_registry(SimpleNamespace()) == []


def test_seam_degrades_malformed_registry_to_empty(monkeypatch):
    def _boom():
        raise ValueError("malformed registry")

    monkeypatch.setattr("hhemt.config.invalidating_fixes.load_invalidating_fixes", _boom)
    # A malformed registry that raises at load must NOT crash construction (degrade to []).
    assert recompute._load_invalidating_fix_registry(SimpleNamespace()) == []


# --------------------------------------------------------------------------- #
# check_invalidating_fixes — report read-model surface (non-blocking).
# --------------------------------------------------------------------------- #
def _match(severity: str) -> RegistryMatch:
    return RegistryMatch(
        commit_id=_FULL_SHA,
        severity=severity,
        recommended_action=RecomputeAction.RE_RUN,
        affected_scope="analysis-datatree",
        summary="Scope predates invalidating fix.",
    )


def test_check_no_match_passes(monkeypatch):
    monkeypatch.setattr(recompute, "match_registry_against_stamps", lambda a: [])
    result = check_invalidating_fixes(SimpleNamespace())
    assert result.passed is True
    assert result.level == "aggregate"
    assert result.details == []
    assert "No registered" in result.summary


def test_check_error_match_fails_but_does_not_raise(monkeypatch):
    monkeypatch.setattr(recompute, "match_registry_against_stamps", lambda a: [_match("error")])
    result = check_invalidating_fixes(SimpleNamespace())
    assert result.passed is False  # error -> failing row
    assert len(result.details) == 1
    assert result.details[0]["recommended_action"] == "re-run"


def test_check_warning_only_match_keeps_passed_true(monkeypatch):
    monkeypatch.setattr(recompute, "match_registry_against_stamps", lambda a: [_match("warning")])
    result = check_invalidating_fixes(SimpleNamespace())
    assert result.passed is True  # warning-only does not fail the check
    assert len(result.details) == 1
