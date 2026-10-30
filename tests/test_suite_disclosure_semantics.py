"""The disclosure's compile-tier split must not state a definite zero it cannot know.

Property: ``compile_tier_outcome["adopted_from_cache"]`` is never a definite ``0`` on a run
whose own ``build_dirs_reused`` records that at least one compiled build predates the
session. The ninth disclosure key exists so that a green whose compile tier stood on a
cached build is distinguishable from one that built everything; a split that answers
"built" for every passing node as soon as ANY build was new answers that question wrongly,
in the reassuring direction, and the reader has nothing to compare against.

Class: NEW CAPABILITY. The plausible wrong implementation is the all-or-nothing branch in
``build_disclosure`` -- ``presented and reused >= presented`` -- which reports every passing
compile-tier node as built whenever a single build was produced during the session. Under
it this test is red; there is no prior implementation to regress against.

Why it is required: ``build_dirs_reused.dirs[]`` already carries per-build ``reused``
flags, so the information the split needs is present in the same disclosure record and is
discarded by the branch. A reader holding both keys can see the contradiction; only a test
holds the producer to it.

What kills it: any change to the split that either attributes some passing compile-tier
nodes to ``adopted_from_cache`` when the reuse is partial, or reports the case as ``None``
/ an explicit mixed marker instead of a definite zero.

A second correct implementation under which it still passes: the split could enumerate
build dirs and attribute nodes per dir (``adopted_from_cache`` > 0), OR it could decline to
split a mixed tier at all (``adopted_from_cache`` is ``None``). The assertion turns on the
definite-zero-with-evidence-to-the-contrary shape, which neither form produces.
"""

from __future__ import annotations

#: Three compile-tier nodes is the shape's own stated build-dir count (R2b: 2 fixtures over
#: 3 build dirs), so the population this test ranges over is the one the run actually holds.
_COMPILE_NODES = ("n1", "n2", "n3")


def _manifest() -> dict:
    """A plan-time manifest carrying the compile record the ninth key is read against."""
    return {
        "run_id": "reviewer-disclosure-fixture",
        "tier_address": "reviewer-address",
        "compile_precondition": {
            "compile_tier_nodes": list(_COMPILE_NODES),
            "compile_tier_node_count": len(_COMPILE_NODES),
            "compile_bearing": True,
            "compile_venue_permitted": True,
            "compile_tier_required": True,
        },
    }


def _merged(*, present: int, reused: int) -> dict:
    """The folded session inventory: ``reused`` of ``present`` builds predate the session."""
    return {
        "entries": {},
        "states_observed": {},
        "build_dirs": {
            "expected": present,
            "present": present,
            "reused": reused,
            "dirs": [{"build_dir": f"build_{i}", "reused": i < reused} for i in range(present)],
        },
        "triton_clone": {"borrowed": False, "tree": "/nonexistent"},
        "roots_rebuilt": {"T0": False, "T1": False, "T2": False, "T3": False},
        "tier_address": "reviewer-address",
        "sessions": 1,
    }


def _disclosure(*, present: int, reused: int) -> dict:
    """One run's disclosure block, with every compile-tier node passing."""
    from hhemt.suite.aggregate import build_disclosure

    return build_disclosure(
        _manifest(),
        scope="union",
        passed=list(_COMPILE_NODES),
        uneval=[],
        structural=[],
        incidental=[],
        absent_list=[],
        declared_not_attempted=[],
        merged=_merged(present=present, reused=reused),
    )


def test_partial_build_reuse_never_reports_a_definite_zero_for_adopted_compile_nodes():
    """A two-of-three reuse is not a zero-adoption compile tier.

    The nine-key contract states a "split into built / adopted-from-cache", and this run
    satisfies the preconditions for every one of those buckets being meaningful: the plan
    recorded its compile nodes, three builds are present, and two of them predate the
    session. A record that then says nothing was adopted states a fact the same record
    contradicts two keys later.
    """
    disclosure = _disclosure(present=3, reused=2)
    outcome = disclosure["compile_tier_outcome"]
    reused = disclosure["build_dirs_reused"]["reused"]

    assert reused == 2, "fixture precondition: the folded inventory must record the reuse"
    assert not (outcome["adopted_from_cache"] == 0 and reused > 0), (
        "compile_tier_outcome reports adopted_from_cache=0 while build_dirs_reused records "
        f"{reused} predating build(s); outcome={outcome}"
    )
