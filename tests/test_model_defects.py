from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from hhemt.model_defects import (
    REGISTRY_BY_ID,
    SHA_DEPTH_SCATTER_FIX,
    SHA_EXTBC_GHOST_RING_FIX,
    ModelDefect,
    resolve,
    resolve_for_tree_attrs,
)

#: A TRITON clone containing the pre-fix history, for the LIVE-ancestry tests only. Every
#: other test in this module is clone-free by construction -- that is the point of the cached
#: sets, and it is why the registry works on a render bundle. Overridable so the live arm can
#: run wherever a clone exists rather than only on the maintainer's box.
#: The default is derived from the running user's home rather than hardcoded — an
#: absolute path under a named developer's home is a private identifier in a public
#: repository, which the anonymization guard blocks. Set HHEMT_TRITON_CLONE to point
#: the live arm at a clone anywhere; when neither resolves, the live-ancestry tests
#: skip, which is their existing clone-absent behaviour.
_CLONE = os.environ.get("HHEMT_TRITON_CLONE", str(Path.home() / "dev" / "triton-workspace" / "triton"))


def _clone_has(sha: str) -> bool:
    r = subprocess.run(["git", "-C", _CLONE, "cat-file", "-t", sha], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "commit"


#: A real commit in NO cached set, so resolving it EXERCISES the live-ancestry path. Picking a
#: sha that VMS 32 later cached would silently route the "live" assertion through the override
#: rule instead -- which is what happened to this test and is why the sha is chosen for its
#: ABSENCE from the sets rather than for its position in history.
_PREFIX_SHA = "1642a7d47b1460253c9ad7626004453c633ce846"
#: b3820a4 itself, for the two tests NAMED for its position in history. Kept separate from
#: _PREFIX_SHA: that one is chosen for its ABSENCE from the cached sets, this one for where it
#: sits in the history, and collapsing them made the live-path test silently take the override rule.
_SHA_B3820A4 = "b3820a448f304b3f732f4b6fac5564adf86ac333"
#: The live arm needs BOTH the pre-fix sha and the two fix shas resolvable in one object DB.
_LIVE = pytest.mark.skipif(
    not (_clone_has(_PREFIX_SHA) and _clone_has(SHA_DEPTH_SCATTER_FIX)),
    reason=f"no TRITON clone with the pre-fix history at {_CLONE} (set HHEMT_TRITON_CLONE)",
)


def _git_is_ancestor(candidate, descendant):
    r = subprocess.run(
        ["git", "-C", _CLONE, "merge-base", "--is-ancestor", candidate, descendant],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None


def _v(did, sha, **kw):
    return resolve(REGISTRY_BY_ID[did], sha, **kw)


# ---- the four current bundles: producing sha 9db367d, NO clone available ----


def test_campaign_sha_classifies_without_a_clone():
    assert _v("TRITON-COUPLED-RESUME-REPLAY", SHA_DEPTH_SCATTER_FIX).status == "absent"
    assert _v("TRITON-RESUME-DEPTH-SCATTER", SHA_DEPTH_SCATTER_FIX).status == "absent"
    ghost = _v("TRITON-RESUME-EXTBC-GHOST-RING", SHA_DEPTH_SCATTER_FIX)
    assert ghost.status == "present"
    assert ghost.rule == "also_present_set"


def test_split_pin_resume_sha_is_clean_on_all_three():
    for did in REGISTRY_BY_ID:
        v = _v(did, SHA_EXTBC_GHOST_RING_FIX)
        assert v.status == "absent", (did, v)
        assert v.rule == "known_absent_set"


def test_the_false_stamp_is_contradicted():
    """The retired flag said scatter-fix ABSENT on every arm; the registry says the defect is ABSENT."""
    assert _v("TRITON-RESUME-DEPTH-SCATTER", SHA_DEPTH_SCATTER_FIX).status == "absent"


# ---- rule recording ----


@_LIVE
def test_rule_names_distinguish_cached_from_live_from_override():
    assert _v("TRITON-RESUME-DEPTH-SCATTER", SHA_DEPTH_SCATTER_FIX).rule == "known_absent_set"
    # Cached CANNOT answer this sha (it is in no set); live ancestry CAN. Asserting both is
    # what proves the three rules are distinguishable rather than merely that the status is right.
    cached = _v("TRITON-RESUME-DEPTH-SCATTER", _PREFIX_SHA)
    assert cached.rule == "ancestry_unresolvable" and cached.status == "indeterminate"
    live = _v("TRITON-RESUME-DEPTH-SCATTER", _PREFIX_SHA, is_ancestor=_git_is_ancestor)
    assert live.rule == "default_present" and live.status == "present"
    assert _v("TRITON-RESUME-EXTBC-GHOST-RING", SHA_DEPTH_SCATTER_FIX).rule == "also_present_set"


@_LIVE
def test_registry_reproduces_the_user_ruled_mapping_at_b3820a4():
    """Independent check: LIVE ancestry against the real clone must reproduce the ruled mapping.

    Ruling: bug 1 (node-depth scatter) applies to b3820a4 and ancestors, because 9db367d fixes
    it; bug 2 (extbc ghost ring) applies to 9db367d and ancestors. b3820a4 (2026-07-25) already
    POSTDATES the original coupled-resume replay fix (3a832f7d), so that one must read absent.
    Nothing here is asserted from the registry's cached sets — the shas are resolved live.
    """
    got = {
        did: resolve(
            ModelDefect(d.defect_id, d.title, fixed_in=d.fixed_in, trigger=d.trigger),
            _SHA_B3820A4,
            is_ancestor=_git_is_ancestor,
        ).status
        for did, d in REGISTRY_BY_ID.items()
    }
    assert got["TRITON-COUPLED-RESUME-REPLAY"] == "absent"
    assert got["TRITON-RESUME-DEPTH-SCATTER"] == "present"
    # INVARIANT, not a position: a fix sha absent from THIS clone's object DB is honestly
    # unanswerable live; one that is present must classify PRESENT at b3820a4, which
    # predates it. Asserting `indeterminate` unconditionally encoded the ORNL-tracked
    # clone's contents, so fetching the fork into HHEMT_TRITON_CLONE — the obvious way to
    # make ancestry answerable — reddened this test with no diagnostic.
    expected_ghost = "indeterminate" if not _clone_has(SHA_EXTBC_GHOST_RING_FIX) else "present"
    assert got["TRITON-RESUME-EXTBC-GHOST-RING"] == expected_ghost
    # ...and the shipped registry answers it from the cached set instead.
    assert _v("TRITON-RESUME-EXTBC-GHOST-RING", SHA_DEPTH_SCATTER_FIX).status == "present"


# ---- live ancestry against the real clone ----


@_LIVE
def test_live_ancestry_agrees_with_the_cached_set():
    d = REGISTRY_BY_ID["TRITON-COUPLED-RESUME-REPLAY"]
    bare = ModelDefect(d.defect_id, d.title, fixed_in=d.fixed_in, trigger=d.trigger)
    v = resolve(bare, SHA_DEPTH_SCATTER_FIX, is_ancestor=_git_is_ancestor)
    assert v.status == "absent" and v.rule == "fixed_in_ancestor"


@_LIVE
def test_prefix_sha_is_prefix_matched_and_classified_present():
    d = REGISTRY_BY_ID["TRITON-RESUME-DEPTH-SCATTER"]
    bare = ModelDefect(d.defect_id, d.title, fixed_in=d.fixed_in, trigger=d.trigger)
    v = resolve(bare, _SHA_B3820A4, is_ancestor=_git_is_ancestor)
    assert v.status == "present" and v.rule == "default_present"


# ---- indeterminate paths ----


def test_no_producing_sha_is_indeterminate():
    v = _v("TRITON-RESUME-DEPTH-SCATTER", None)
    assert v.status == "indeterminate" and v.rule == "no_producing_sha"


def test_unresolvable_ancestry_is_indeterminate_never_a_warn():
    d = REGISTRY_BY_ID["TRITON-RESUME-EXTBC-GHOST-RING"]
    bare = ModelDefect(d.defect_id, d.title, fixed_in=d.fixed_in, trigger=d.trigger)
    v = resolve(bare, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", is_ancestor=lambda a, b: None)
    assert v.status == "indeterminate" and v.rule == "ancestry_unresolvable"
    v2 = resolve(bare, "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    assert v2.status == "indeterminate"


# ---- the revert case: ancestry says fixed, the set overrules ----


def test_set_override_beats_ancestry_for_a_revert():
    """A build pinned at the REVERT commit is classified FIXED by ancestry and is not.

    Between 9db367d and 5d2ad1e8 the ghost-ring fix landed (e4cae7c), was reverted (936874c),
    and was re-implemented (5d2ad1e). Ancestry cannot see a revert; only the explicit set can.
    """
    reverted = "936874c936874c936874c936874c936874c9368"
    d = ModelDefect(
        "X",
        "x",
        fixed_in="e4cae7ce4cae7ce4cae7ce4cae7ce4cae7ce4cae",
        also_present_in=frozenset({reverted}),
    )
    v = resolve(d, reverted, is_ancestor=lambda a, b: True)
    assert v.status == "present" and v.rule == "also_present_set", v


# ---- tree-attr entry point ----


def test_resolve_for_tree_attrs_reads_only_the_stamped_sha():
    got = {
        v.defect_id: v.status
        for v in resolve_for_tree_attrs(
            {"triton_producing_sha": SHA_DEPTH_SCATTER_FIX, "triton_has_swmm_depth_scatter_fix": False}
        )
    }
    assert got == {
        "TRITON-COUPLED-RESUME-REPLAY": "absent",
        "TRITON-RESUME-DEPTH-SCATTER": "absent",
        "TRITON-RESUME-EXTBC-GHOST-RING": "present",
    }
    assert all(v.status == "indeterminate" for v in resolve_for_tree_attrs({}))


def test_triggers_are_the_population_selectors():
    assert REGISTRY_BY_ID["TRITON-RESUME-EXTBC-GHOST-RING"].trigger == "resumed_any"
    assert REGISTRY_BY_ID["TRITON-RESUME-DEPTH-SCATTER"].trigger == "resumed_coupled"


def test_ghost_ring_entry_carries_the_provenance_note():
    note = REGISTRY_BY_ID["TRITON-RESUME-EXTBC-GHOST-RING"].provenance_note
    assert "TRITON_EXTBC_PROBE" in note and "instrumented/" in note


def test_introduced_in_arm_excludes_a_build_that_predates_the_defect():
    """The `introduced_in` arm — the ONLY branch of resolve() no shipped entry exercises.

    All three registered defects are present from the repo root, so `introduced_in` is None on
    every one of them and this arm is dead against the shipped registry. It is NOT dead code:
    it is the half of the two-sided predicate that a defect INTRODUCED partway through history
    needs, and shipping it unreached is how a branch rots before its first real user. Covered
    here with a synthetic defect so the arm has a reaching input class.
    """
    introduced, fix = "aaaaaaa1111111", "bbbbbbb2222222"
    predates = "ccccccc3333333"

    def _anc(candidate, descendant):
        # `introduced` is NOT an ancestor of a build that predates the defect; the fix is not
        # an ancestor either, so without the introduced_in arm this would read PRESENT.
        return False

    d = ModelDefect("X", "x", fixed_in=fix, introduced_in=introduced)
    v = resolve(d, predates, is_ancestor=_anc)
    assert v.status == "absent"
    assert v.rule == "introduced_in_not_ancestor", v

    # ...and the same defect against a build that DOES contain the introduction reads present.
    v2 = resolve(d, "ddddddd4444444", is_ancestor=lambda c, _d: c == introduced)
    assert v2.status == "present" and v2.rule == "default_present", v2


def test_a_sha_shorter_than_gits_abbreviation_floor_never_matches():
    """The `_MIN_SHA_PREFIX` guard. Below git's own 7-char abbreviation floor a prefix match is
    not evidence of identity, so a 4-char value must NOT collide with a cached set entry."""
    d = ModelDefect("X", "x", fixed_in=None, known_absent_in=frozenset({SHA_DEPTH_SCATTER_FIX}))
    v = resolve(d, "9db3")
    assert v.status == "present", v
    assert v.rule == "default_present", v
    # the full sha DOES match the same set
    assert resolve(d, SHA_DEPTH_SCATTER_FIX).rule == "known_absent_set"
