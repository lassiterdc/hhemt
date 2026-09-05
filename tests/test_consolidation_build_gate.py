"""The consolidation build gate: does the store reuse decision see the producing sha?

Pure-predicate and source-level. No fixture, no solver, no cached tree — every store
here is written under tmp_path by the test that reads it.

The property under test is an ASYMMETRY, and it is the one a reviewer is most likely
to soften: absent is never equal, including absent on both sides. A store with no
`hhemt_producing_sha` must be a MISMATCH, because reading absence as "no objection"
silently reuses every tree built before the ADR-15 capture site existed — which is
exactly the population most likely to be stale.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.provenance import store_build_mismatch


def _store(tmp_path: Path, attrs: dict | None, *, v2: bool = False) -> Path:
    d = tmp_path / "s.zarr"
    d.mkdir(parents=True)
    if attrs is not None:
        (d / (".zattrs" if v2 else "zarr.json")).write_text(
            json.dumps(attrs if v2 else {"attributes": attrs}), encoding="utf-8"
        )
    return d


def test_equal_sha_is_no_objection(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hhemt.provenance.producing_stamp", lambda: {"hhemt_sha": "abc123def456", "hhemt_dirty": "false"}
    )
    assert store_build_mismatch(_store(tmp_path, {"hhemt_producing_sha": "abc123def456"})) is None


def test_differing_sha_reports_both_shas_and_the_remedy(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hhemt.provenance.producing_stamp",
        lambda: {"hhemt_sha": "c1176f281e7b", "hhemt_dirty": "false"},
    )
    why = store_build_mismatch(_store(tmp_path, {"hhemt_producing_sha": "b62cc6d0ae86"}))
    assert why is not None
    assert "b62cc6d0ae86" in why and "c1176f281e7b" in why
    assert '"stage": "consolidate"' in why, "the remedy must name the exact invocation"


@pytest.mark.parametrize(
    "attrs, running",
    [
        ({"analysis_id": "x"}, "c1176f281e7b"),
        ({"hhemt_producing_sha": ""}, "c1176f281e7b"),
        ({"hhemt_producing_sha": "b62cc6d0ae86"}, ""),
        ({}, ""),
        ({"hhemt_producing_sha": "unknown"}, "c1176f281e7b"),
        ({"hhemt_producing_sha": "b62cc6d0ae86"}, "unknown"),
        ({"hhemt_producing_sha": "unknown"}, "unknown"),
    ],
    ids=[
        "store-unstamped",
        "store-empty-sha",
        "running-unresolvable",
        "absent-on-both-sides",
        "store-sentinel-unknown",
        "running-sentinel-unknown",
        "sentinel-on-BOTH-sides",
    ],
)
def test_absent_is_never_equal(tmp_path, monkeypatch, attrs, running):
    """The load-bearing asymmetry, and ABSENCE INCLUDES A SENTINEL.

    `sentinel-on-BOTH-sides` is the case this module was extended for and it is the one
    a reader will be tempted to drop as redundant. It is not redundant: `_toolkit_git_sha`
    returns the literal "unknown" on a wheel install, "unknown" is TRUTHY, so a predicate
    guarding on falsiness compares two placeholders EQUAL and reuses a store built by a
    different toolkit with no output at all. The empty-string cases do NOT cover it --
    they exercise a different branch and pass against the defective predicate.
    """
    monkeypatch.setattr("hhemt.provenance.producing_stamp", lambda: {"hhemt_sha": running, "hhemt_dirty": "false"})
    assert store_build_mismatch(_store(tmp_path, attrs)) is not None


def test_dirty_running_checkout_is_a_mismatch_when_the_shas_agree(tmp_path, monkeypatch):
    """`git rev-parse` succeeds on a dirty tree and returns the COMMITTED sha, so a
    developer editing a consolidation module and re-running gets an UNCHANGED sha. That
    is the workflow this gate exists to serve; without this arm the gate is decorative
    for its own use case."""
    monkeypatch.setattr(
        "hhemt.provenance.producing_stamp",
        lambda: {"hhemt_sha": "b62cc6d0ae86", "hhemt_dirty": "true"},
    )
    why = store_build_mismatch(_store(tmp_path, {"hhemt_producing_sha": "b62cc6d0ae86"}))
    assert why is not None and "DIRTY" in why


def test_dirty_is_not_consulted_when_the_shas_already_differ(tmp_path, monkeypatch):
    """The narrowing that keeps the arm honest: when the shas differ the store already
    rebuilds, so dirty must not be the REASON reported -- a reader debugging the rebuild
    would otherwise be pointed at their working tree instead of at the build gap."""
    monkeypatch.setattr(
        "hhemt.provenance.producing_stamp",
        lambda: {"hhemt_sha": "c1176f281e7b", "hhemt_dirty": "true"},
    )
    why = store_build_mismatch(_store(tmp_path, {"hhemt_producing_sha": "b62cc6d0ae86"}))
    assert why is not None and "DIRTY" not in why
    assert "b62cc6d0ae86" in why and "c1176f281e7b" in why


def test_absent_store_is_a_mismatch_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hhemt.provenance.producing_stamp", lambda: {"hhemt_sha": "c1176f281e7b", "hhemt_dirty": "false"}
    )
    assert store_build_mismatch(tmp_path / "does_not_exist.zarr") is not None


def test_zarr_v2_zattrs_layout_is_read(tmp_path, monkeypatch):
    """A pre-V0021 store may be zarr v2; the gate must not report a v2 store unstamped."""
    monkeypatch.setattr(
        "hhemt.provenance.producing_stamp", lambda: {"hhemt_sha": "deadbeefcafe", "hhemt_dirty": "false"}
    )
    assert store_build_mismatch(_store(tmp_path, {"hhemt_producing_sha": "deadbeefcafe"}, v2=True)) is None


def test_escape_prints_and_records_rather_than_bypassing_silently(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "hhemt.provenance.producing_stamp", lambda: {"hhemt_sha": "c1176f281e7b", "hhemt_dirty": "false"}
    )
    monkeypatch.setenv("HHEMT_DECLARE_STALE_BUILD", "1")
    assert store_build_mismatch(_store(tmp_path, {"hhemt_producing_sha": "b62cc6d0ae86"})) is None
    out = capsys.readouterr().out
    assert "DECLARED STALE BUILD" in out and "b62cc6d0ae86" in out


def test_both_gates_consume_the_build_mismatch_term():
    """Source-level, and the reason this module is more than an import check.

    Gate B's early return is the FIRST branch, so Spec 6 applied without Spec 5 leaves
    a predicate that is computed and never consulted. Reading the source is the only way
    to catch that without building a full analysis fixture.
    """
    import hhemt.processing_analysis as pa
    import hhemt.sensitivity_analysis as sa

    a = Path(sa.__file__).read_text(encoding="utf-8")
    b = Path(pa.__file__).read_text(encoding="utf-8")
    assert "(_subs_stale or _build_mismatch)" in a, "Gate A rebuild condition not wired"
    assert "_inputs_match and not _build_mismatch" in b, "Gate B early return not wired"
    assert "(not _inputs_match or _build_mismatch)" in b, "Gate B rebuild condition not wired"
