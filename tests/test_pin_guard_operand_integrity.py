"""The in-session mutation tripwire `test_check_live_pin_registered.py` advertises and imports.

WHAT WAS WRONG AND WHY IT WAS INVISIBLE -- REPAIRED BY `944364b9`. This paragraph is a
HISTORICAL RECORD, not a live finding, and it is retained rather than deleted because a
reader arriving with the old claim in hand needs to find it here and learn which version
they hold. Until that commit, `tests/test_check_live_pin_registered.py` stated a
mutation-discipline contract its implementation did not keep. Its module docstring read
"`test_the_real_tree_is_unmutated` re-reads the two tracked operands' sha256 afterwards, so
a test that accidentally wrote to the real tree is caught here rather than in someone's
later `git status`", and that test's own docstring added "Digests recorded 2026-10-03
against the committed content; a legitimate edit to either file updates them in the same
change". BOTH SENTENCES ARE NOW GONE FROM THERE -- `944364b9` rewrote each one, so a search
for either returns nothing and that is the expected result rather than evidence this
paragraph has drifted. Verify with whitespace normalization, never a single-line literal
grep: the sibling line-wraps that prose, so a line-scoped search returns a zero that looks
identical to genuine absence. The implementation computed the two digests and then asserted:

    assert all(len(d) == 64 for d in digests.values())

`hashlib.sha256(...).hexdigest()` is 64 characters for every possible input, so that
predicate is CONSTANT and cannot separate an unmutated tree from a mutated one. That is a
property of the predicate rather than of any one commit, which is why the control below
still measures it on a locally re-implemented copy. No literal digests were recorded
anywhere in that file, and none are recorded there now either -- post-repair that is the
DESIGN rather than the defect, because the sibling imports this module's single snapshot
instead of keeping a second copy. Positive-controlled at `5c2ef5a3`: the predicate returns
True on the real tree, True on a copy with arbitrary text appended to both operands, and
True on a copy with both operands EMPTIED. The two non-vacuous assertions that stood beside
it -- `guard.main([]) == 0` and `guard.read_pin(_REPO) == TRITON_PIN` -- SURVIVE in the
repaired sibling: they catch a mutation that BREAKS the pin/registry agreement or moves the
pin, and are blind to any mutation that leaves the agreement intact (a comment, a docstring,
a reordered set, an added defect entry whose `known_absent_in` already contains the live
pin). That residual blindness is what the snapshot comparison closes.

WHY THIS MODULE AND NOT A RECORDED-DIGEST LIST. A list of literal digests compared against
the live files reds on every LEGITIMATE edit too, and `src/hhemt/model_defects.py` is edited
on every pin registration -- so that form would red most often at exactly the moments the
tree is being changed on purpose, and would be routed around within two bumps. Pre-repair
the sibling's stated PURPOSE was narrower than its stated MECHANISM: catching an in-session
accidental write. The two now AGREE there, and this paragraph is why they agree on the
snapshot rather than on a digest list -- the sibling's own docstring reproduces the
argument, so changing it here without changing it there splits one rationale across two
sources. The instrument that matches the purpose is a SNAPSHOT taken at module import --
which pytest performs during COLLECTION, strictly before any test in the session executes --
compared against the live bytes inside a test. That has zero false reds across legitimate
commits and fires on exactly the event the contract names.

SCOPE, stated so a green here is not over-read. This catches a write performed BETWEEN
THIS MODULE'S IMPORT AND THIS TEST'S EXECUTION, IN THIS PROCESS -- which is narrower than
"a write performed by the session" in two measured ways. A COLLECTION-TIME write (a
module-level statement in a module imported before this one) precedes the snapshot and is
NOT caught: measured, a probe moving its write from a test body to module level took the
module set from 2 failed to 17 passed on a mutated tree. And under xdist the session spans
processes while this comparison does not, so a write in another worker after this check has
run is likewise outside it. It is not a git-cleanliness check and it does not replace one;
for both residuals above, the git check is the covering instrument.

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

    Evaluates the RETIRED length-only predicate -- re-implemented locally below, because
    `944364b9` removed it from the sibling -- and this module's snapshot predicate against
    the SAME pair of trees, one faithful and one mutated. The length-only predicate returns
    the same verdict on both; the snapshot predicate separates them.

    BOTH predicates are LOCAL to this test and neither reads the sibling, so a flip here
    says nothing about the sibling's state: it means this module's own control has changed,
    and it should be read here before anywhere else. The test outliving the defect is the
    point rather than an oversight -- it is the standing proof that the two predicates
    differ, which is what the sibling's `test_the_real_tree_is_unmutated` docstring now
    cites when it explains why it imports this snapshot instead of re-taking or pinning one.
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
