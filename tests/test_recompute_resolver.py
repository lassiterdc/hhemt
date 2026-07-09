"""ADR-16 Phase 2 — recompute RESOLVER unit tests.

Covers the frozen enums + 8-cell action table, the git-framed pre-fix ancestry
predicate (mocked git; polarity is load-bearing — a hand-merge could invert it),
the semver fallback, the scoped ``override_force_rerun`` emission shape, the D6
unstamped-scope INFO, and the empty-registry consuming interface. No real git
history and no real zarr trees are needed — the git predicate is exercised through
monkeypatched helpers so the exit-code -> verdict mapping is pinned deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hhemt import recompute
from hhemt.recompute import (
    RecomputeAction,
    RecomputeScope,
    RegistryMatch,
    _classify_unstamped_scope,
    _emit_re_run_instruction,
    _is_scope_affected_by_semver,
    _is_scope_pre_fix,
    classify_unstamped_scopes,
    match_registry_against_stamps,
    resolve_recompute_action,
)

_SCOPE_SHA = "a1b2c3d4e5f6"  # 12-char short sha, as the ADR-15 stamp records
_FIX_SHA = "0f1e2d3c4b5a6978" * 2 + "aaaaaaaa"  # 40-char full sha, as the registry records


# --------------------------------------------------------------------------- #
# Import safety (DoD: imports on py3.10 -- no StrEnum) + string identity.
# --------------------------------------------------------------------------- #
def test_enums_are_str_enum_not_strenum():
    # (str, Enum) mix-in gives string identity + serialization while staying
    # import-safe on 3.10 (StrEnum landed in 3.11).
    assert isinstance(RecomputeAction.RE_RUN, str)
    assert isinstance(RecomputeScope.CONSOLIDATION, str)
    assert RecomputeAction.RE_RUN == "re-run"
    assert RecomputeAction.REPROCESS_SCENARIO == "reprocess-scenario"
    assert RecomputeAction.RE_CONSOLIDATE == "re-consolidate"
    assert RecomputeAction.NONE_COSMETIC == "none-cosmetic"
    assert RecomputeScope.SCENARIO_PROCESSING == "scenario-processing"


# --------------------------------------------------------------------------- #
# The exhaustive 8-cell scope x clear_raw -> action table.
# --------------------------------------------------------------------------- #
_EXPECTED_TABLE = {
    (RecomputeScope.SIMULATION, False): RecomputeAction.RE_RUN,
    (RecomputeScope.SIMULATION, True): RecomputeAction.RE_RUN,
    (RecomputeScope.SCENARIO_PROCESSING, False): RecomputeAction.REPROCESS_SCENARIO,
    (RecomputeScope.SCENARIO_PROCESSING, True): RecomputeAction.RE_RUN,
    (RecomputeScope.CONSOLIDATION, False): RecomputeAction.RE_CONSOLIDATE,
    (RecomputeScope.CONSOLIDATION, True): RecomputeAction.RE_CONSOLIDATE,
    (RecomputeScope.COSMETIC, False): RecomputeAction.NONE_COSMETIC,
    (RecomputeScope.COSMETIC, True): RecomputeAction.NONE_COSMETIC,
}


@pytest.mark.parametrize(("key", "expected"), list(_EXPECTED_TABLE.items()))
def test_scope_clear_raw_action_table_all_8_cells(key, expected):
    scope, clear_raw = key
    assert resolve_recompute_action(scope, clear_raw) is expected


def test_load_bearing_cell_scenario_processing_raw_cleared_escalates_to_re_run():
    # When raw outputs are cleared, reprocess-from-raw is impossible, so a
    # scenario-processing bug must escalate to a full RE_RUN (encoded IN the table).
    assert (
        resolve_recompute_action(RecomputeScope.SCENARIO_PROCESSING, True)
        is RecomputeAction.RE_RUN
    )
    assert (
        resolve_recompute_action(RecomputeScope.SCENARIO_PROCESSING, False)
        is RecomputeAction.REPROCESS_SCENARIO
    )


# --------------------------------------------------------------------------- #
# The git-framed ancestry predicate — polarity is load-bearing (mocked git).
# --------------------------------------------------------------------------- #
def _mock_ancestry(
    monkeypatch,
    *,
    rc=None,
    in_checkout=True,
    shallow=False,
    shas_exist=True,
):
    monkeypatch.setattr(recompute, "_in_git_checkout", lambda: in_checkout)
    monkeypatch.setattr(recompute, "_repo_is_shallow", lambda: shallow)
    monkeypatch.setattr(recompute, "_sha_exists", lambda sha: shas_exist)
    if rc is not None:
        monkeypatch.setattr(
            recompute.subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(returncode=rc),
        )


def test_ancestry_exit0_is_post_fix_false(monkeypatch):
    # exit 0 -> fix IS an ancestor of scope -> scope HAS the fix -> False (post-fix)
    _mock_ancestry(monkeypatch, rc=0)
    assert _is_scope_pre_fix(_SCOPE_SHA, _FIX_SHA) is False


def test_ancestry_exit1_is_pre_fix_true(monkeypatch):
    # exit 1 -> fix NOT an ancestor -> scope LACKS the fix -> True (pre-fix / affected)
    _mock_ancestry(monkeypatch, rc=1)
    assert _is_scope_pre_fix(_SCOPE_SHA, _FIX_SHA) is True


def test_ancestry_exit128_is_indeterminate_none(monkeypatch):
    # 128 (bogus/unfetched sha) must NEVER be collapsed into pre-fix -> None.
    _mock_ancestry(monkeypatch, rc=128)
    assert _is_scope_pre_fix(_SCOPE_SHA, _FIX_SHA) is None


def test_ancestry_shallow_repo_is_indeterminate_none(monkeypatch):
    # A shallow clone can report a silently-wrong not-ancestor -> refuse -> None.
    _mock_ancestry(monkeypatch, rc=1, shallow=True)
    assert _is_scope_pre_fix(_SCOPE_SHA, _FIX_SHA) is None


def test_ancestry_off_checkout_is_indeterminate_none(monkeypatch):
    # Installed wheel: no object DB -> short-circuit to None (semver fallback).
    _mock_ancestry(monkeypatch, rc=0, in_checkout=False)
    assert _is_scope_pre_fix(_SCOPE_SHA, _FIX_SHA) is None


def test_ancestry_missing_operand_is_indeterminate_none(monkeypatch):
    # Either operand absent from local history -> None, never a false pre-fix.
    _mock_ancestry(monkeypatch, rc=1, shas_exist=False)
    assert _is_scope_pre_fix(_SCOPE_SHA, _FIX_SHA) is None


def test_ancestry_unstamped_scope_is_none(monkeypatch):
    _mock_ancestry(monkeypatch, rc=1)
    assert _is_scope_pre_fix(None, _FIX_SHA) is None
    assert _is_scope_pre_fix("unknown", _FIX_SHA) is None


def test_ancestry_reflexive_scope_equals_fix_uses_git_not_string_equality(monkeypatch):
    # A scope produced AT the fix commit HAS the fix. Real git returns exit 0 for
    # the reflexive `--is-ancestor A A`, so the predicate returns False via git
    # object resolution -- NOT via a Python string-equality guard (which is
    # forbidden by the C3 identifier-length invariant: 12-char stamp vs 40-char
    # registry sha would never string-match anyway).
    _mock_ancestry(monkeypatch, rc=0)
    assert _is_scope_pre_fix(_FIX_SHA, _FIX_SHA) is False


# --------------------------------------------------------------------------- #
# Semver fallback (shares ONE SpecifierSet(affected_version_range) evaluator).
# --------------------------------------------------------------------------- #
def test_semver_fallback_in_range_is_affected():
    assert _is_scope_affected_by_semver("0.9.2", ">=0.9.0,<1.0.0") is True


def test_semver_fallback_out_of_range_is_not_affected():
    assert _is_scope_affected_by_semver("1.5.0", ">=0.9.0,<1.0.0") is False


def test_semver_fallback_absent_or_unknown_is_none():
    assert _is_scope_affected_by_semver(None, ">=0.9.0,<1.0.0") is None
    assert _is_scope_affected_by_semver("unknown", ">=0.9.0,<1.0.0") is None
    assert _is_scope_affected_by_semver("0+unknown", ">=0.9.0,<1.0.0") is None


def test_semver_fallback_invalid_specifier_is_none():
    assert _is_scope_affected_by_semver("0.9.2", "not-a-valid-range") is None


# --------------------------------------------------------------------------- #
# Scoped override_force_rerun emission shape (D4 — surgical, never from_scratch).
# --------------------------------------------------------------------------- #
def _fake_analysis(*, sensitivity: bool):
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(toggle_sensitivity_analysis=sensitivity)
    )


def test_emit_re_run_non_sensitivity_uses_event_iloc_ints():
    instr = _emit_re_run_instruction(_fake_analysis(sensitivity=False), {7, 3, 1})
    assert instr["action"] == "re-run"
    assert instr["call"] == "analysis.run"
    assert instr["kwargs"]["override_force_rerun"] == {"event_iloc": [1, 3, 7]}
    # NEVER from_scratch — the whole point of scoped force-rerun is drift preservation.
    assert "from_scratch" not in instr["kwargs"]


def test_emit_re_run_sensitivity_uses_sa_id_strings():
    instr = _emit_re_run_instruction(_fake_analysis(sensitivity=True), {5, 22, 0})
    assert instr["kwargs"]["override_force_rerun"] == {"sa_id": ["0", "22", "5"]}


def test_emit_re_run_targets_only_the_given_prefix_scopes():
    instr = _emit_re_run_instruction(_fake_analysis(sensitivity=False), {4})
    assert instr["kwargs"]["override_force_rerun"] == {"event_iloc": [4]}


# --------------------------------------------------------------------------- #
# D6 unstamped-scope INFO — a non-blocking record with NO verdict.
# --------------------------------------------------------------------------- #
def test_classify_unstamped_scope_is_info_with_no_action():
    match = _classify_unstamped_scope("sa_5")
    assert isinstance(match, RegistryMatch)
    assert match.severity == "info"
    assert match.recommended_action is None  # the AttributeError-avoidance property
    assert match.commit_id is None
    assert match.affected_scope == "sa_5"
    assert "unknown" in match.summary.lower()


# --------------------------------------------------------------------------- #
# Consuming interface — Phase-2 stubs an empty registry (graceful-absent).
# --------------------------------------------------------------------------- #
def test_match_registry_against_stamps_empty_registry_returns_empty(monkeypatch):
    monkeypatch.setattr(recompute, "_load_invalidating_fix_registry", lambda analysis: [])
    fake = SimpleNamespace()  # never touched when the registry is empty
    assert match_registry_against_stamps(fake) == []
    assert classify_unstamped_scopes(fake) == []
