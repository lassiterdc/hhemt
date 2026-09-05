"""Pins conformance-to-a-declaration, and specifically that it is NOT peer agreement."""

from __future__ import annotations

import json

import pytest

from hhemt.suite import (
    VERSION_EXPECTATION,
    verify_version_conformance,
    write_version_expectation,
)


def test_match_when_the_resolved_tree_is_the_declared_one(tmp_path):
    """SKIPS under a non-editable wheel install rather than failing.

    `resolved_tree()` returns None for a non-`src` layout, which would make the expectation
    record the literal string "None" and the arm fail -- making the suite's own greenness
    depend on how the toolkit was installed rather than on the mechanism under test.
    """
    from hhemt.suite import resolved_tree

    mine = resolved_tree()
    if mine is None:
        pytest.skip("source-tree install required; this arm cannot construct a MATCH otherwise")
    write_version_expectation(tmp_path, sha="0" * 40, tree=str(mine))
    status, rec = verify_version_conformance(tmp_path, site="drive")
    assert status == "MATCH", rec


def test_unresolvable_is_not_reported_as_mismatch(monkeypatch, tmp_path):
    """ "Could not determine my tree" and "my tree is wrong" need different operator actions."""
    import hhemt.suite as S

    monkeypatch.setattr(S, "resolved_tree", lambda: None)
    write_version_expectation(tmp_path, sha="0" * 40, tree="/some/tree")
    status, rec = S.verify_version_conformance(tmp_path, site="drive")
    assert status == "UNRESOLVABLE", rec


def test_mismatch_when_the_declaration_names_another_tree(tmp_path):
    write_version_expectation(tmp_path, sha="0" * 40, tree="/nonexistent/other/tree")
    status, rec = verify_version_conformance(tmp_path, site="verdict")
    assert status == "MISMATCH", rec
    assert rec["expected_tree"] == "/nonexistent/other/tree"


def test_absent_declaration_is_distinguishable_from_agreement(tmp_path):
    """A pre-floor run dir must read NO_EXPECTATION, never MATCH.

    This is the arm that makes the mechanism a declaration rather than peer agreement:
    a site with nothing to compare against must not report success.
    """
    status, rec = verify_version_conformance(tmp_path, site="verdict")
    assert status == "NO_EXPECTATION", rec
    assert rec["expected_tree"] is None


def test_a_silent_site_is_observable_as_a_gap(tmp_path):
    """Calls THE ROSTER FUNCTION rather than recomposing its set subtraction.

    Two arms of this test have now been refuted for the same generative reason: an arm that
    reconstructs the logic under test stays green when that logic is deleted. This one
    imports `missing_conformance_sites`, so removing the roster from `aggregate.main()`
    leaves it green but removing the FUNCTION breaks it at import.

    IT DOES NOT PIN THE WIRING, and the sentence that used to stand here -- "the U2-4 call
    site is a one-line consumer that a reader can check by eye" -- was the argument this
    floor itself rejects when made about a participant: a site that did not run leaves
    nothing behind, so its absence must be observed by something other than a reader.
    Applying that standard to the detector's own wiring is the same standard, not pedantry.
    test_the_roster_is_wired_into_main_not_merely_importable below is what closes it.
    """
    from hhemt.suite import missing_conformance_sites

    write_version_expectation(tmp_path, sha="0" * 40, tree="/x")
    for site in ("drive", "chunk-00"):
        verify_version_conformance(tmp_path, site=site)
    assert missing_conformance_sites(tmp_path, chunk_count=1) == ["verdict"]
    # And the satisfying arm: with every implied site present, nothing is reported.
    verify_version_conformance(tmp_path, site="verdict")
    assert missing_conformance_sites(tmp_path, chunk_count=1) == []
    assert json.loads((tmp_path / VERSION_EXPECTATION).read_text())["sha"] == "0" * 40


def test_the_roster_is_wired_into_main_not_merely_importable(tmp_path):
    """EXECUTION arm. Deleting the roster block from aggregate.main() must redden something.

    The chain the floor depends on is three links: main() calls missing_conformance_sites,
    which returns silent sites, which set NOT-GREEN. The arm above pins link two only, and
    deleting link one is a two-line edit that leaves every other arm in this file green.
    A source-text `"missing_conformance_sites" in aggregate.py` assertion was considered and
    rejected: it observes REFERENCE rather than execution and survives a call moved behind a
    dead branch. This arm runs main() against a synthetic run dir that the sibling module
    already knows how to build, so it observes the BEHAVIOUR.

    The assertion is on the PROBLEM TEXT, not on the verdict token, and that is deliberate:
    this run dir also produces a verdict-site MISMATCH (the declaration names a tree nothing
    resolves to), so an assertion on `verdict == "NOT-GREEN"` alone would be satisfied by the
    wrong mechanism -- the shape five arms in this session have already died of.
    """
    from hhemt.suite import aggregate as A
    from tests.test_verdict_partition import _build

    _build(tmp_path, chunk_entries=[[], []])
    write_version_expectation(tmp_path, sha="0" * 40, tree="/nonexistent/declared/tree")
    for site in ("drive", "chunk-00"):
        verify_version_conformance(tmp_path, site=site)
    A.main(["--run-dir", str(tmp_path), "--allow-not-green"])
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "NOT-GREEN", summary["problems"]
    assert any("left no version-conformance record" in p and "chunk-01" in p for p in summary["problems"]), summary[
        "problems"
    ]
    # The printed/summary.md line must carry the flip too -- see the verdict_line arm below.
    assert summary["verdict_line"].startswith("verdict=NOT-GREEN "), summary["verdict_line"]


def test_an_unresolvable_verdict_job_is_reported_rather_than_waved_through(monkeypatch, tmp_path):
    """The site the floor names as unverifiable, in the state it cannot answer for itself.

    Under the MISMATCH-only form of this branch an UNRESOLVABLE verdict job wrote its
    record, SATISFIED the roster (which checks presence, and the record is present),
    appended nothing, and the summary read GREEN. Measured on a copy 2026-09-05.
    """
    import hhemt.suite as S
    from hhemt.suite import aggregate as A
    from tests.test_verdict_partition import _build

    _build(tmp_path, chunk_entries=[[], []])
    write_version_expectation(tmp_path, sha="0" * 40, tree="/declared/tree")
    for site in ("drive", "chunk-00", "chunk-01"):
        verify_version_conformance(tmp_path, site=site)
    monkeypatch.setattr(S, "resolved_tree", lambda: None)
    A.main(["--run-dir", str(tmp_path), "--allow-not-green"])
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "NOT-GREEN", summary["problems"]
    assert any("cannot determine which source tree" in p for p in summary["problems"]), summary["problems"]
    assert summary["verdict_line"].startswith("verdict=NOT-GREEN "), summary["verdict_line"]


def test_the_verdict_line_composer_agrees_with_aggregates_own(tmp_path):
    """Pins the two-composer duplicate that render_verdict_line's docstring declines to remove.

    aggregate() builds verdict_line inline from LOCALS; render_verdict_line rebuilds it from
    `result`. Add an interpolant to one and not the other and a conformance-only NOT-GREEN
    run silently loses it from the operator's line -- the rare path, which is why the
    duplicate needs a comparison rather than an eye. No declaration is written here, so the
    run dir is pre-floor and aggregate() is exercised unmutated.
    """
    from hhemt.suite import aggregate as A
    from tests.test_verdict_partition import _build

    _build(tmp_path, chunk_entries=[[], []])
    result = A.aggregate(tmp_path, scope="array")
    assert A.render_verdict_line(result) == result["verdict_line"]
