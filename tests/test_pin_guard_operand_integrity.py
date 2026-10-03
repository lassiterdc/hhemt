"""The in-session mutation tripwire `test_check_live_pin_registered.py` advertises.

WHAT IS WRONG AND WHY IT IS INVISIBLE. `tests/test_check_live_pin_registered.py` states a
mutation-discipline contract in its module docstring -- "`test_the_real_tree_is_unmutated`
re-reads the two tracked operands' sha256 afterwards, so a test that accidentally wrote to
the real tree is caught here rather than in someone's later `git status`" -- and that test's
own docstring adds "Digests recorded 2026-10-03 against the committed content; a legitimate
edit to either file updates them in the same change". The implementation computes the two
digests and then asserts:

    assert all(len(d) == 64 for d in digests.values())

`hashlib.sha256(...).hexdigest()` is 64 characters for every possible input, so that
predicate is CONSTANT and cannot separate an unmutated tree from a mutated one. No digests
are recorded anywhere in the file. Positive-controlled at `5c2ef5a3`: the predicate returns
True on the real tree, True on a copy with arbitrary text appended to both operands, and
True on a copy with both operands EMPTIED. The two non-vacuous assertions beside it --
`guard.main([]) == 0` and `guard.read_pin(_REPO) == TRITON_PIN` -- catch a mutation that
BREAKS the pin/registry agreement or moves the pin, and are blind to any mutation that
leaves the agreement intact (a comment, a docstring, a reordered set, an added defect entry
whose `known_absent_in` already contains the live pin).

WHY THIS MODULE AND NOT A RECORDED-DIGEST LIST. A list of literal digests compared against
the live files reds on every LEGITIMATE edit too, and `src/hhemt/model_defects.py` is edited
on every pin registration -- so that form would red most often at exactly the moments the
tree is being changed on purpose, and would be routed around within two bumps. The stated
PURPOSE is narrower than the stated MECHANISM: catching an in-session accidental write. The
instrument that matches the purpose is a SNAPSHOT taken at module import -- which pytest
performs during COLLECTION, strictly before any test in the session executes -- compared
against the live bytes inside a test. That has zero false reds across legitimate commits and
fires on exactly the event the contract names.

SCOPE, stated so a green here is not over-read. This catches a write performed by the
PYTEST SESSION. It is not a git-cleanliness check and it does not replace one.

COMPILE-FREE AND ANALYSIS-FREE BY CONSTRUCTION: no `*_compiled` fixture, no
`TRITONSWMM_analysis`. Screened by fixture closure, never by module text.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

#: The same two tracked operands `scripts/check_live_pin_registered.py` audits.
_OPERANDS = ("tests/fixtures/_triton_source_cache.py", "src/hhemt/model_defects.py")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot() -> dict[str, str]:
    return {rel: _digest(_REPO / rel) for rel in _OPERANDS}


#: Taken at IMPORT, i.e. during pytest's collection phase, which precedes every test in the
#: session including the mutating ones in `test_check_live_pin_registered.py`.
_AT_COLLECTION = _snapshot()


def test_no_test_in_this_session_wrote_to_the_guard_operands():
    """The tripwire itself: live bytes must equal the collection-time snapshot.

    A red here names the file and is actionable immediately; the alternative is discovering
    it in a later `git status`, attributed to nothing.
    """
    now = _snapshot()
    drifted = {rel: (_AT_COLLECTION[rel], now[rel]) for rel in _OPERANDS if _AT_COLLECTION[rel] != now[rel]}
    assert not drifted, "a test in this session wrote to a tracked guard operand: " + "; ".join(
        f"{rel}: {was[:12]} -> {is_[:12]}" for rel, (was, is_) in drifted.items()
    )


def test_the_snapshot_is_populated_and_distinct():
    """Instrument check. An empty or collapsed snapshot would make the tripwire vacuous.

    This is the assertion the length-only form should have been: it ranges over the
    snapshot's STRUCTURE, which is what a constant-length predicate cannot establish.
    """
    assert set(_AT_COLLECTION) == set(_OPERANDS)
    assert len(set(_AT_COLLECTION.values())) == len(_OPERANDS), "two operands hashed to one digest"
    for rel, d in _AT_COLLECTION.items():
        assert (_REPO / rel).is_file(), rel
        assert len(d) == 64 and int(d, 16) >= 0, rel  # well-formedness only; the COMPARISON is the tripwire above


def test_a_length_only_digest_predicate_cannot_discriminate(tmp_path):
    """Why this module exists, as a measurement rather than a reading.

    Evaluates the landed length-only predicate and this module's snapshot predicate against
    the SAME pair of trees -- one faithful, one mutated. The length-only predicate returns
    the same verdict on both; the snapshot predicate separates them. A test that flips this
    result has either repaired the sibling or broken this one, and either way should be read
    here before anywhere else.
    """

    def length_only(root: Path) -> bool:
        digests = {rel: _digest(root / rel) for rel in _OPERANDS}
        return all(len(d) == 64 for d in digests.values())

    def against_snapshot(root: Path, snap: dict[str, str]) -> bool:
        return all(_digest(root / rel) == snap[rel] for rel in _OPERANDS)

    faithful = tmp_path / "faithful"
    mutated = tmp_path / "mutated"
    for root in (faithful, mutated):
        for rel in _OPERANDS:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_REPO / rel, root / rel)
    snap = {rel: _digest(faithful / rel) for rel in _OPERANDS}
    for rel in _OPERANDS:
        (mutated / rel).write_text((mutated / rel).read_text() + "\n# in-session accidental write\n")

    assert length_only(faithful) is True
    assert length_only(mutated) is True, "the length-only predicate has become discriminating"
    assert against_snapshot(faithful, snap) is True
    assert against_snapshot(mutated, snap) is False, "the snapshot predicate has stopped discriminating"
