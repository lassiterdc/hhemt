from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hhemt.model_defects import (
    REGISTRY_BY_ID,
    SHA_DEPTH_SCATTER_FIX,
    SHA_EXTBC_GHOST_RING_FIX,
    SHA_ORNL_GHOST_RING_AND_GPU,
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
    for did in [i for i, d in REGISTRY_BY_ID.items() if d.trigger != "coupled"]:
        v = _v(did, SHA_EXTBC_GHOST_RING_FIX)
        assert v.status == "absent", (did, v)
        assert v.rule == "known_absent_set"


def test_ornl_ghost_ring_sha_is_clean_on_all_three():
    """a38338b0 (ORNL upstream `triton-swmm`) must resolve ABSENT on every defect, WITHOUT a clone.

    Why this needs its own test rather than riding on the fork sha's: a38338b0 reaches `absent` on
    two DIFFERENT bases -- cached ancestry for the coupled-resume and depth-scatter defects
    (3a832f7d and 9db367dd really are its ancestors), and an explicit independent-implementation
    override for the ghost-ring one, where 5d2ad1e8 is on the other remote and ancestry cannot
    answer at all. All three must nonetheless report the SAME rule, because the read path has no
    clone and `known_absent_set` is the only rule reachable there. A regression that dropped this
    sha from one set would surface as `ancestry_unresolvable`, which is the exact abstention this
    entry exists to remove -- and which is easy to mistake for a pass, since `indeterminate` never
    fails a check.
    """
    for did in [i for i, d in REGISTRY_BY_ID.items() if d.trigger != "coupled"]:
        v = _v(did, SHA_ORNL_GHOST_RING_AND_GPU)
        assert v.status == "absent", (did, v)
        assert v.rule == "known_absent_set", (did, v)


def test_ornl_entry_records_the_weaker_evidentiary_basis():
    """The note must SAY the basis is not a content comparison, and must name the PARENT as where
    the fix landed -- a38338b0 is the GPU-emission commit, not the ghost-ring one. A silent
    certification is the failure mode the ghost-ring entry's existing note was written to prevent."""
    note = REGISTRY_BY_ID["TRITON-RESUME-EXTBC-GHOST-RING"].provenance_note
    assert "a38338b0" in note
    assert "0cf5faff" in note
    assert "weaker" in note


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
    """Property: the tree-attr entry point derives every verdict from `triton_producing_sha`
    alone, and reports one verdict per registered defect.

    Class: REGRESSION. The expected mapping was hardcoded at three entries, so the fourth
    registry entry red it with `Left contains 1 more item` and no instruction about what to add.
    The guard below turns that into a named diagnostic.

    What kills it: making this entry point read a second tree attr (the stamped booleans it
    replaced), or dropping a defect from its output.

    A second correct implementation under which it still passes: one that returns the verdicts
    in any order, or as any iterable -- the assertion is on the id-to-status mapping.
    """
    got = {
        v.defect_id: v.status
        for v in resolve_for_tree_attrs(
            {"triton_producing_sha": SHA_DEPTH_SCATTER_FIX, "triton_has_swmm_depth_scatter_fix": False}
        )
    }
    #: Measured at SHA_DEPTH_SCATTER_FIX, per defect. The snapshot-traversal entry is ABSENT
    #: here because that sha predates the snapshot payload -- its tree carries no
    #: external/swmm/src/solver/snapshot.c -- so there is no guard present to be wrong.
    expected = {
        "TRITON-COUPLED-RESUME-REPLAY": "absent",
        "TRITON-RESUME-DEPTH-SCATTER": "absent",
        "TRITON-RESUME-EXTBC-GHOST-RING": "present",
        "TRITON-SNAPSHOT-TRAVERSAL-GUARD": "absent",
        # 9db367dd predates the snapshot payload too, so it has no snapshot to leave behind.
        "TRITON-SNAPSHOT-STALE-AT-CLEAN-START": "absent",
        # The three every-coupled-run entries predate their fixes here (is-ancestor rc 1).
        "TRITON-COUPLED-RIM-BELOW-CROWN": "present",
        "TRITON-COUPLED-US-UNIT-RIM": "present",
        "TRITON-COUPLED-LIST-MISALIGNED": "present",
    }
    # BIDIRECTIONAL set-equality against the live registry, so a fifth entry fails HERE, by
    # name, instead of surfacing as a dict-inequality diff. A len() comparison is deliberately
    # not used: equal counts with one swapped id pass it and leave a defect unasserted.
    unlisted = set(REGISTRY_BY_ID) - set(expected)
    stale = set(expected) - set(REGISTRY_BY_ID)
    assert not unlisted, (
        f"registry gained {sorted(unlisted)} and this table does not expect them. Add each "
        f"one's verdict at {SHA_DEPTH_SCATTER_FIX[:12]} -- measure it, do not assume."
    )
    assert not stale, f"this table expects {sorted(stale)}, which REGISTRY no longer carries."

    assert got == expected
    assert all(v.status == "indeterminate" for v in resolve_for_tree_attrs({}))


def test_triggers_are_the_population_selectors():
    assert REGISTRY_BY_ID["TRITON-RESUME-EXTBC-GHOST-RING"].trigger == "resumed_any"
    assert REGISTRY_BY_ID["TRITON-RESUME-DEPTH-SCATTER"].trigger == "resumed_coupled"


def test_ghost_ring_entry_carries_the_provenance_note():
    note = REGISTRY_BY_ID["TRITON-RESUME-EXTBC-GHOST-RING"].provenance_note
    assert "TRITON_EXTBC_PROBE" in note and "instrumented/" in note


def test_introduced_in_arm_excludes_a_build_that_predates_the_defect():
    """The `introduced_in` arm — the ONLY branch of resolve() no shipped entry exercises.

    THREE of the four registered defects are present from the repo root, so `introduced_in` is
    None on those three. It is NOT None on all of them any more: TRITON-SNAPSHOT-TRAVERSAL-GUARD
    sets it, because the snapshot payload its guard sits in did not exist at the repo root -- so
    this arm is NO LONGER dead against the shipped registry, and the sentence that said it was
    invited deleting a branch that now has a real user.

    What the arm is still worth covering synthetically: the shipped entry resolves through its
    EXPLICIT sets on the read path, which has no clone, so the ancestry arm this test exercises
    is not the one that fires in production for it. The synthetic defect below is what gives the
    arm a reaching input class with ancestry available.
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


def test_the_clean_start_stale_snapshot_entry_resolves_by_content_at_every_registered_sha():
    """The clean-start stale-snapshot entry is present exactly where a snapshot payload exists
    without the disposal.

    Class: every registered solver sha, read without a clone (the read path's only rule is the
    explicit sets). Why: a clean start that leaves an earlier run's SWMM snapshots in place lets
    a resume after a failed or refused save load the earlier run's state, which is possible only
    where snapshots are written and the clean-start disposal (`cf357e4b`) is missing. What kills
    it: listing the pre-fix pin `c7b2fd1d` as absent, or a no-payload sha as present. Second
    correct implementation: one table per sha built from the two content probes in the entry's
    comment; the assertions read only status and rule.
    """
    from hhemt.model_defects import (
        SHA_CURATED_CLEAN_START_FIX,
        SHA_CURATED_MAIN_PIN,
        SHA_DEPTH_SCATTER_FIX,
        SHA_SNAPSHOT_CLEAN_START_FIX,
        SHA_SNAPSHOT_GUARD_PRIOR_PIN,
        SHA_WP1_SNAPSHOT_GUARD_AND_TESTS,
    )

    entry = REGISTRY_BY_ID["TRITON-SNAPSHOT-STALE-AT-CLEAN-START"]
    assert entry.title == (
        "A coupled run's clean start left an earlier run's SWMM snapshots in place, so after a "
        "failed or refused snapshot save a resume could load the earlier run's SWMM state"
    )
    assert entry.trigger == "resumed_coupled"
    assert entry.fixed_in == SHA_CURATED_CLEAN_START_FIX
    for sha, status in (
        (SHA_WP1_SNAPSHOT_GUARD_AND_TESTS, "present"),
        (SHA_SNAPSHOT_CLEAN_START_FIX, "absent"),
        (SHA_SNAPSHOT_GUARD_PRIOR_PIN, "absent"),
        (SHA_CURATED_MAIN_PIN, "absent"),
        (SHA_DEPTH_SCATTER_FIX, "absent"),
    ):
        v = resolve(entry, sha)
        assert v.status == status, (sha, v)
        assert v.rule == ("known_absent_set" if status == "absent" else "also_present_set"), (sha, v)


def test_every_registered_defect_is_absent_at_the_live_pin():
    """Property: the live solver pin resolves ABSENT on every defect from the explicit sets.

    Class: every REGISTRY entry at `a146fb6`, the curated fork `main` pin. Why: an entry missing the
    pin reads `ancestry_unresolvable` on the read path, which never fails a check and so hides a
    regression. What kills it: dropping the pin from one entry's `known_absent_in`. Second correct
    implementation: deriving the membership from the pin file at import; the assertion reads the
    verdicts only.
    """
    from hhemt.model_defects import SHA_CURATED_MAIN_PIN

    for did, d in REGISTRY_BY_ID.items():
        v = resolve(d, SHA_CURATED_MAIN_PIN)
        assert (v.status, v.rule) == ("absent", "known_absent_set"), (did, v)


#: The three every-coupled-run entries and the title each must carry, word for word.
_COUPLED_ENTRIES = (
    (
        "TRITON-COUPLED-RIM-BELOW-CROWN",
        "coupled junction recorded below SWMM's crown-raised depth (a recorded depth of 0 is one case)",
    ),
    (
        "TRITON-COUPLED-US-UNIT-RIM",
        "A US-unit coupled model's node rims (and, with the DEM-rim option, its inverts) reached "
        "TRITON in feet and were used as metres",
    ),
    (
        "TRITON-COUPLED-LIST-MISALIGNED",
        "TRITON's coupled node list could silently disagree with SWMM's FLOW-inflow set: a header "
        "line skipped a node, or out-of-order lines swapped exchanges between nodes",
    ),
)


@pytest.mark.parametrize(("did", "title"), _COUPLED_ENTRIES)
def test_each_every_coupled_run_entry_is_present_before_its_fix_and_absent_at_it_and_at_the_pin(did, title):
    """Each `coupled` entry is present at `c7b2fd1d` and absent at its fix and at both pins.

    Class: the three every-coupled-run registry entries, read without a clone. Why: the analysis
    check reports them from these explicit sets alone, and the title is what a report reader sees.
    What kills it: the pre-fix pin missing from `also_present_in`, the fix or the pin missing from
    `known_absent_in`, a trigger other than `coupled`, or a reworded title. Second correct
    implementation: a per-sha table; the assertions read status, rule, trigger and title.
    """
    from hhemt.model_defects import SHA_CURATED_MAIN_PIN, SHA_SNAPSHOT_GUARD_PRIOR_PIN, SHA_WP1_SNAPSHOT_GUARD_AND_TESTS

    d = REGISTRY_BY_ID[did]
    assert d.defect_id == did
    assert d.trigger == "coupled"
    assert d.title == title
    v = resolve(d, SHA_WP1_SNAPSHOT_GUARD_AND_TESTS)
    assert (v.status, v.rule) == ("present", "also_present_set"), v
    for sha in (d.fixed_in, SHA_SNAPSHOT_GUARD_PRIOR_PIN, SHA_CURATED_MAIN_PIN):
        v = resolve(d, sha)
        assert (v.status, v.rule) == ("absent", "known_absent_set"), (sha, v)


@pytest.mark.parametrize("did", [did for did, _ in _COUPLED_ENTRIES])
def test_live_ancestry_agrees_with_each_every_coupled_run_entry_cached_sets(did):
    """The cached verdicts at `c7b2fd1d` and at `a146fb6` are what ancestry gives, for each `coupled` entry.

    Class: the three entries, re-resolved from their fix sha alone against a real clone. Why: the
    cached sets were resolved once by hand, and the read path trusts them without a clone, so a
    wrong set would report a defect a build does not carry, or hide one it does. What kills it: a
    `fixed_in` that is an ancestor of `c7b2fd1d`, or one that is not an ancestor of the pin. Second
    correct implementation: comparing every registered sha's cached verdict with its live one; the
    assertions read status and rule. Skips where the clone lacks the fork's fix commits.
    """
    from hhemt.model_defects import SHA_CURATED_MAIN_PIN, SHA_WP1_SNAPSHOT_GUARD_AND_TESTS

    d = REGISTRY_BY_ID[did]
    needed = (d.fixed_in, SHA_WP1_SNAPSHOT_GUARD_AND_TESTS, SHA_CURATED_MAIN_PIN)
    if not all(_clone_has(s) for s in needed):
        pytest.skip(f"no TRITON clone holding {[s[:8] for s in needed]} at {_CLONE} (set HHEMT_TRITON_CLONE)")
    bare = ModelDefect(d.defect_id, d.title, fixed_in=d.fixed_in, trigger=d.trigger)
    v = resolve(bare, SHA_WP1_SNAPSHOT_GUARD_AND_TESTS, is_ancestor=_git_is_ancestor)
    assert (v.status, v.rule) == ("present", "default_present"), v
    v = resolve(bare, SHA_CURATED_MAIN_PIN, is_ancestor=_git_is_ancestor)
    assert (v.status, v.rule) == ("absent", "fixed_in_ancestor"), v


def test_every_entry_registers_its_own_fix_sha():
    """Without a clone, every defect resolves ABSENT at its own `fixed_in`.

    The invariant the registry has held for every entry: `fixed_in` sits in `known_absent_in`, so a tree
    stamped at the fix commit itself resolves on the read path, which has no ancestry. A re-key of
    `fixed_in` that does not register the new sha breaks it silently, which is how the re-pin to the
    curated fork first broke it.
    """
    for d in REGISTRY_BY_ID.values():
        assert d.fixed_in in d.known_absent_in, d.defect_id
        v = resolve(d, d.fixed_in)
        assert v.status == "absent" and v.rule == "known_absent_set", (d.defect_id, v)
