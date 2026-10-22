"""Declarative registry of KNOWN MODEL (TRITON / TRITON-SWMM) DEFECTS, and their
applicability to a given producing build.

Replaces the ad-hoc per-defect boolean flags (``triton_has_coupled_resume_fix``,
``triton_has_swmm_depth_scatter_fix``) with one entry per defect. Two properties motivate
the change, both measured rather than anticipated:

FOSSIL CLASS. The retired flags are booleans DERIVED FROM A TOOLKIT CONSTANT and frozen
into durable state at run time, so a later toolkit change cannot correct them. Dated
instance: ``_PINNED_TRITON_SWMM_DEPTH_SCATTER_FIX_SHA`` was introduced as ``None`` in
toolkit commit ``463ab9e`` (2026-08-02 19:05:37) together with an ``is None -> set(False)``
branch; the synth_cc campaign ran 2026-08-04, inside that window, and every arm was stamped
``triton_has_swmm_depth_scatter_fix=False``. The constant received its real value at
``7b7b573`` (2026-08-05 17:00:59), by which time the sims were on disk. The stamp is
refuted by direct measurement -- the producing sha IS the fix commit
(``git log -1 --format=%s 9db367d`` -> "restore per-rank new_depth after hotstart-exchange
replay") -- yet no re-consolidation can correct it, because ``_stamp_triton_provenance``
re-reads the same system log and the flag is re-derived ONLY on the compile path
(``system.py:665`` and ``:1441``, the only two call sites of
``_capture_tritonswmm_provenance``). Deriving applicability from the recorded sha at read
time removes the class: nothing durable is written that a later registry edit can
contradict.

ANCESTRY IS NOT ALWAYS COMPUTABLE, AND IS NOT ALWAYS RIGHT.
  * Not computable: ``merge-base --is-ancestor`` needs both shas in one object DB. The
    availability of a given sha depends on WHICH remote a given clone tracks: ORNL upstream
    (``code.ornl.gov/hydro/triton.git``) and the fork (``github.com/lassiterdc/triton.git``)
    are both named ``triton.git`` and carry different histories, so a sha can be resolvable
    in one clone and absent in another on the same machine. (The synthetic-test canonical
    tracks the fork as of the pin re-point; the maintainer clone at ``HHEMT_TRITON_CLONE``
    and the container ``.def`` builds still track ORNL.) And a render BUNDLE carries
    zarr stores and figures, never a git repo, so ``report-from-bundle`` on any other
    machine has no clone at all — that half of the argument is remote-independent and is
    what makes the cached sets necessary rather than merely convenient. Hence
    ``also_present_in`` / ``known_absent_in``: shas whose
    ancestry was resolved ONCE at registry-authoring time, on a machine that had both, and
    cached here so the read path needs no clone.
  * Not right: ancestry answers "did this history contain that commit", never "does this
    tree contain that change", so it cannot see a REVERT. This is not hypothetical in the
    branch being pinned -- between ``9db367d`` and ``5d2ad1e8`` the ghost-ring fix landed at
    ``e4cae7c``, was reverted at ``936874c``, and was re-implemented at ``5d2ad1e``. The end
    state is correct, so no entry here needs an override for it today; but a build pinned at
    ``936874c`` would be classified FIXED by ancestry and is not. The explicit sets are the
    only expression of that, which is why they outrank ancestry in the resolution order.

Every verdict records WHICH RULE decided it (``DefectVerdict.rule``), so a reader can tell a
cached-ancestry verdict from a live-ancestry one from a genuine override.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

#: Minimum characters compared when matching a registry sha against a recorded one. Git's own
#: abbreviation floor is 7; below that a prefix match is not evidence of identity.
_MIN_SHA_PREFIX = 7

DefectStatus = Literal["present", "absent", "indeterminate"]

#: Which sims could have EXERCISED a defect. This is deliberately a POPULATION SELECTOR, not
#: a second boolean gate: the selected rows ARE the `examined` population of the validation
#: check, so an empty selection flows into the existing examined-zero gate and renders N/A
#: rather than a warning. A version-only registry would warn on a clean arm that carries the
#: affected build but never resumed -- the P7 violation the examined-zero work removed.
TriggerKind = Literal["always", "resumed_any", "resumed_coupled"]


@dataclass(frozen=True)
class DefectVerdict:
    defect_id: str
    status: DefectStatus
    rule: str
    detail: str = ""


@dataclass(frozen=True)
class ModelDefect:
    """One known model defect and the builds/runs it applies to."""

    defect_id: str
    title: str
    #: Sha that REMOVES the defect. None = no fix exists yet (every build is affected).
    fixed_in: str | None
    #: Sha that INTRODUCES it. None = present from the repo root.
    introduced_in: str | None = None
    #: Authoring-time-resolved ancestry, and genuine overrides where ancestry is wrong.
    known_absent_in: frozenset[str] = field(default_factory=frozenset)
    also_present_in: frozenset[str] = field(default_factory=frozenset)
    trigger: TriggerKind = "always"
    remedy: str = ""
    bug_report: str | None = None
    #: Free text surfaced beside any verdict on this defect. Used where the build's
    #: provenance needs stating rather than assuming -- see TRITON-RESUME-EXTBC-GHOST-RING.
    provenance_note: str = ""


def _sha_eq(a: str, b: str) -> bool:
    """Prefix-tolerant sha equality, mirroring git's own abbreviation semantics."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if len(a) < _MIN_SHA_PREFIX or len(b) < _MIN_SHA_PREFIX:
        return False
    n = min(len(a), len(b))
    return a[:n] == b[:n]


def _in_set(sha: str, shas) -> bool:
    return any(_sha_eq(sha, s) for s in shas)


def resolve(
    defect: ModelDefect,
    producing_sha: str | None,
    *,
    is_ancestor: Callable[[str, str], bool | None] | None = None,
) -> DefectVerdict:
    """Classify `defect` against `producing_sha`, recording which rule decided.

    `is_ancestor(candidate, descendant)` returns True/False, or None when the question
    cannot be answered in this environment (sha absent from the object DB, no clone, no
    git). Passing `is_ancestor=None` means "no ancestry available", which is the normal
    case on the read path -- the cached sets are expected to carry it.

    Order is deliberate: explicit sets OUTRANK ancestry, because a set is the only way to
    express a revert (see module docstring) and the only form that works without a clone.
    """
    did = defect.defect_id
    if not producing_sha:
        return DefectVerdict(did, "indeterminate", "no_producing_sha", "no triton_producing_sha recorded on this tree")
    if _in_set(producing_sha, defect.known_absent_in):
        return DefectVerdict(did, "absent", "known_absent_set", "producing sha is explicitly recorded as unaffected")
    if _in_set(producing_sha, defect.also_present_in):
        return DefectVerdict(did, "present", "also_present_set", "producing sha is explicitly recorded as affected")

    unresolved: list[str] = []

    if defect.fixed_in is not None:
        anc = is_ancestor(defect.fixed_in, producing_sha) if is_ancestor else None
        if anc is None:
            unresolved.append(f"fixed_in={defect.fixed_in[:12]}")
        elif anc:
            return DefectVerdict(
                did, "absent", "fixed_in_ancestor", f"fix {defect.fixed_in[:12]} is an ancestor of the producing sha"
            )

    if defect.introduced_in is not None:
        anc = is_ancestor(defect.introduced_in, producing_sha) if is_ancestor else None
        if anc is None:
            unresolved.append(f"introduced_in={defect.introduced_in[:12]}")
        elif not anc:
            return DefectVerdict(
                did,
                "absent",
                "introduced_in_not_ancestor",
                f"defect postdates this build ({defect.introduced_in[:12]} not an ancestor)",
            )

    if unresolved:
        return DefectVerdict(
            did, "indeterminate", "ancestry_unresolvable", "could not resolve ancestry for: " + ", ".join(unresolved)
        )
    return DefectVerdict(did, "present", "default_present", "no fix is an ancestor of the producing sha")


#: Full shas, kept as module constants so a reader can grep one place for "which build".
SHA_COUPLED_RESUME_FIX = "3a832f7d5eedd96aaee0dfe9181da5774adfb9f4"
SHA_DEPTH_SCATTER_FIX = "9db367ddc79f86c7f708686d1dd805dc992fb0a4"
SHA_EXTBC_GHOST_RING_FIX = "5d2ad1e8adf9a85d7df14e885b76e59a10f9a98b"
#: `main` on the fork: the ghost-ring fix CHERRY-PICKED onto 9db367dd (probes excluded), plus the
#: RUN INFO GPU-device-name emission. NOT a descendant of 5d2ad1e8 -- a cherry-pick mints a new
#: sha, so `merge-base --is-ancestor 5d2ad1e8 21e666d6` answers 1 while the change IS present.
#: Ancestry cannot express a cherry-pick any more than it can express a revert, which is why this
#: sha must appear in the explicit sets below. Content-verified: src/ghost_ring.h is blob-identical
#: to 5d2ad1e8, and src/triton.h is whitespace-normalised identical once the probe blocks are
#: stripped.
SHA_MAIN_GHOST_RING_AND_GPU = "21e666d6e0efc3383344813853386aaba1474785"
#: ORNL UPSTREAM (`code.ornl.gov/hydro/triton.git`), branch `triton-swmm`, NOT the fork -- and the
#: remote is load-bearing rather than decorative, because the two repositories are both named
#: `triton.git` and carry different histories (module docstring). The sha is stored bare, as every
#: entry here is: `_sha_eq` compares 40-hex identifiers whose collision probability is the reason
#: git uses them, so the SET needs no qualifier and the COMMENT is where a reader learns which
#: clone resolves it.
#:
#: THE CLAIM IS ABOUT THE TREE, NOT ABOUT THIS COMMIT, and the distinction is the whole reason this
#: comment is worded carefully. a38338b0 is `emit the GPU device name in the TRITON RUN INFO block`;
#: the ghost-ring fix landed at its PARENT 0cf5faff (`TRITON: persist and restore the perimeter
#: ghost ring at checkpoints`). The TREE at a38338b0 carries that fix, plus 9db367dd, b3820a44 and
#: 3a832f7d among its ancestors. Evidence that the solver content is unchanged between parent and
#: child: `git diff --stat 0cf5faff a38338b0` reports ONE file, `src/output.h`, +15/-0 -- a
#: WHOLE-TREE diff, which is what makes it sound. (A path-targeted diff naming src/ghost_ring.h
#: would be vacuous corroboration: ORNL implements the ring inline and that path does not exist
#: upstream, so `git ls-tree` returns 0 entries for it at BOTH shas against a control needle of 1
#: for src/triton.h. Cite the whole-tree diff.)
#:
#: Same structural position as 21e666d6 above -- ghost-ring fix PLUS the GPU-device-name emission,
#: one repository each -- and ONE EVIDENTIARY STEP WEAKER: 21e666d6 is a cherry-pick that was
#: content-verified blob-identical to 5d2ad1e8, while this is an INDEPENDENT upstream
#: implementation of the same intent, attested by the parent commit's title and by ancestry rather
#: than by a diff against the fork. That comparison cannot be performed on any single clone:
#: 5d2ad1e8 is absent from ORNL history (`git cat-file -t 5d2ad1e8` fails in an ORNL clone), so
#: `merge-base --is-ancestor 5d2ad1e8 a38338b0` cannot answer at all.
#:
#: APPLICABILITY, recorded from upstream's own commit message rather than inferred: the ghost-ring
#: defect "only manifests when open_boundaries=1 and/or external BCs are active; with both inactive
#: nothing writes the ring mid-step and resumes were already bit-exact (verified: the fix is inert
#: there, byte-identical rasters)."
#:
#: WHAT WOULD FALSIFY THIS ENTRY: the four synthetic hydraulics_vs_compute_config experiments are
#: the measurement. If the upstream implementation is subtly wrong, their clean-vs-resume b4b
#: comparison disagrees while this set says `absent`. Content-verify against 5d2ad1e8 when a clone
#: carrying both remotes exists, and downgrade this entry if the check fails.
SHA_ORNL_GHOST_RING_AND_GPU = "a38338b09e62e57c936f51516bbdbe495d89a546"

#: Historical PRODUCING shas whose ancestry was resolved ONCE here, at registry-authoring time,
#: against a clone that had them — the read path has no clone (see module docstring) and would
#: otherwise return INDETERMINATE for every sha not already named above, leaving the pre-fix arms
#: unreachable in production. These are project-relevant, not illustrative: 15eb18a5 is the pin the
#: container-validation suite names, and b3820a4 is the sha the defect-to-version ruling keys on.
SHA_PRE_COUPLED_RESUME = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"
SHA_PRE_DEPTH_SCATTER = "b3820a448f304b3f732f4b6fac5564adf86ac333"

#: The SOLVER-WORK CAMPAIGN BRANCH TIP: `solver-work/replay-precision-accounting` on the fork,
#: the build that introduces the full-precision SWMM state SNAPSHOT as a second resume
#: mechanism. It is recorded here because the read path has no clone and would otherwise return
#: INDETERMINATE for it -- and an INDETERMINATE replay verdict does two bad things at once
#: (`check_coupled_resume_validity` early-returns "resume validity NOT verified" before any
#: marker arm runs, and `check_known_resume_defects` selects on `status == "present"`, so
#: INDETERMINATE reads as ABSENT and returns an affirmative clean bill).
#:
#: ALL THREE RESUME registry defects resolve ABSENT at this sha; the snapshot-traversal entry
#: resolves PRESENT here and its basis is its own `also_present_in` record. Measured 2026-09-25
#: against the clone at `HHEMT_TRITON_CLONE` with HEAD = 658a7a37:
#:
#:   * `merge-base --is-ancestor 3a832f7d 658a7a37` -> rc 0   (coupled-resume fix: ancestor)
#:   * `merge-base --is-ancestor 9db367dd 658a7a37` -> rc 0   (depth-scatter fix: ancestor)
#:   * `merge-base --is-ancestor 5d2ad1e8 658a7a37` -> rc 1   (ghost-ring fix: NOT an ancestor)
#:   * `git ls-tree 658a7a37 -- src/ghost_ring.h`   -> 0 entries (control needle src/triton.h -> 1)
#:
#: Read alone, the last two say the ghost-ring defect is PRESENT. Both are artifacts of asking
#: the fork's question of an ORNL-descended tree, which is the trap this module's docstring
#: already names for the a38338b0 entry -- "ORNL implements the ring inline and that path does
#: not exist upstream", so a `src/ghost_ring.h` probe is vacuous against it. The deciding
#: commands are the UPSTREAM ancestry and the SYMBOLS:
#:
#:   * `merge-base --is-ancestor 0cf5faff 658a7a37` -> rc 0  (ORNL's ghost-ring fix commit)
#:   * `merge-base --is-ancestor a38338b0 658a7a37` -> rc 0  (the sha already in the sets below)
#:   * `git grep -c "write_output_ghost_ring\|read_output_ghost_ring" 658a7a37 -- src/output.h
#:      src/triton.h` -> output.h 5, triton.h 1, and the resume-path call site is wired at
#:      `src/triton.h`'s `if (arglist.checkpoint_id > 0) { out.read_output_ghost_ring(...) }`.
#:
#: So this sha inherits a38338b0's absence verdict by ORDINARY CACHED ANCESTRY, and it is
#: recorded at the same evidentiary strength as that entry -- no stronger. Whatever would
#: falsify a38338b0 (see its constant's comment) falsifies this one.
SHA_SOLVER_WORK_REPLAY_PRECISION = "658a7a37032a95842e1662fe330190da24ffd4e8"

#: The SAME branch, ADVANCED: `solver-work/replay-precision-accounting` tip after WP-1C landed
#: `feat: preserve and append the SWMM output series across a resume`. It is a DESCENDANT of the
#: constant above and its ONLY descendant as of this writing (`git log --oneline 658a7a37..01e95a7`
#: -> one commit), so it is registered for the ordinary reason a moved tip must be: the read path
#: has no clone, `resolve` falls through to `ancestry_unresolvable`, and an unregistered tip
#: re-opens BOTH consequences the constant above was added to close.
#:
#: MEASURED, not inherited. A descendant does NOT automatically carry an ancestor's verdict here
#: -- explicit sets outrank ancestry precisely because a set is the only form that can express a
#: REVERT (module docstring) -- so every probe was re-run rather than argued from the parent's
#: row. Measured 2026-09-28 against a fork clone (`HHEMT_TRITON_CLONE`), HEAD = 01e95a7:
#:
#:   * `merge-base --is-ancestor 658a7a37 01e95a7` -> rc 0   (the parent tip: ancestor)
#:   * `merge-base --is-ancestor 3a832f7d 01e95a7` -> rc 0   (coupled-resume fix: ancestor)
#:   * `merge-base --is-ancestor 9db367dd 01e95a7` -> rc 0   (depth-scatter fix: ancestor)
#:   * `merge-base --is-ancestor 5d2ad1e8 01e95a7` -> rc 1   (ghost-ring fix: NOT an ancestor)
#:   * `git ls-tree 01e95a7 -- src/ghost_ring.h`   -> 0 entries (control needle src/triton.h -> 1)
#:
#: The last two are the SAME two-probe trap the parent's comment documents, and they mislead here
#: for the same reason: they ask the fork's question of an ORNL-descended tree. The deciding
#: commands agree with the parent's, value for value:
#:
#:   * `merge-base --is-ancestor 0cf5faff 01e95a7` -> rc 0  (ORNL's ghost-ring fix commit)
#:   * `merge-base --is-ancestor a38338b0 01e95a7` -> rc 0  (the sha already in the sets below)
#:   * `git grep -c "write_output_ghost_ring\|read_output_ghost_ring" 01e95a7 -- src/output.h
#:      src/triton.h` -> output.h 5, triton.h 1 (identical to the parent's counts).
#:
#: And the delta itself touches no ghost-ring path: `git diff --name-only 658a7a37 01e95a7` lists
#: `src/swmm_triton.h`, `src/triton.h`, three `external/swmm/src/solver/` files and six `test/`
#: files -- no `src/output.h`, so the symbol counts above could not have moved.
#:
#: Evidentiary strength is the parent's, no stronger: this sha inherits a38338b0's absence verdict
#: by ordinary cached ancestry, and whatever falsifies a38338b0 falsifies this one too.
SHA_WP1C_SWMM_OUTPUT_APPEND = "01e95a76ba4b7015d13b20423eb0e72ec1fb495c"

#: The SAME branch, ADVANCED AGAIN: `solver-work/replay-precision-accounting` tip after the WP-1C
#: linkage repair `fix: give output_end C linkage at its C++ call site` (fea07ee) and its
#: characterization test. It is registered for the SAME ordinary reason the constant above was:
#: the read path has no clone, `resolve` falls through to `ancestry_unresolvable`, and an
#: UNREGISTERED TIP RE-OPENS BOTH CONSEQUENCES. It is the pin the synthetic tier builds as of the
#: 2026-09-29 repin of `tests/fixtures/_triton_source_cache.py::TRITON_PIN`, which is what made
#: registering it owed -- 01e95a76 alone does NOT LINK without fea07ee, so the tier cannot pin there.
#: Re-measured against THIS sha rather than inherited (each `merge-base --is-ancestor X e53c2fa`):
#:   * a38338b0 -> rc 0    * 0cf5faff -> rc 0    * 3a832f7d -> rc 0    * 9db367dd -> rc 0
#: Submodule gitlinks are UNCHANGED across the move (`git ls-tree {pin} external/kokkos
#: external/yaml-cpp` -> dc1bf738 and a83cd315 at BOTH 658a7a37 and e53c2fa, measured), so the
#: borrow's `submodule.alternateLocation=superproject` path needs no submodule-side network work.
#: Evidentiary strength is the parent's, no stronger: whatever falsifies a38338b0 falsifies this.
SHA_WP1C_LINKAGE_REPAIR_AND_TEST = "e53c2fa01a64583fb57bc58082245fd687882b8f"

#: `main` ON THE FORK, and the move to it is a CROSS-BRANCH move whose merge-base is a38338b0.
#: The 46 commits `solver-work/replay-precision-accounting` carries and `main` does not are the
#: AUTHORING HISTORY of work `main` re-lands as 10 squashed commits on top of a38338b0, plus the
#: two snapshot cfg keys (`swmm_snapshot_disable`, `swmm_snapshot_keep_all`) the branch does not
#: have. Measured, not argued: `git diff --name-status e53c2fa 031ea42` returns M and A only --
#: ZERO D -- and the removed-non-comment-line census over `src` + `external/swmm` returns nine
#: lines, every one an old signature replaced by its two-argument-wider form in the same hunk.
#: It is the pin BOTH live UVA benchmarking arms declare in their committed
#: `system_config_*.yaml`, which is what made registering it owed.
#:
#: RE-MEASURED AGAINST THIS SHA, NOT INHERITED -- explicit sets outrank ancestry precisely so a
#: set can express a revert, so a descendant does not inherit its parent's row. Each probe is
#: `git merge-base --is-ancestor X 031ea42b` in a fork clone:
#:   * a38338b0 -> rc 0    * 0cf5faff -> rc 0    * 3a832f7d -> rc 0    * 9db367dd -> rc 0
#:   * e53c2fa  -> rc 1, and 031ea42 -> e53c2fa is ALSO rc 1: the two DIVERGED. `rev-list --count`
#:     gives 10 on main not on the branch and 46 the other way; `merge-base` is a38338b0.
#: Submodule gitlinks are UNCHANGED (`git ls-tree 031ea42b external/kokkos external/yaml-cpp` ->
#: dc1bf738 and a83cd315, identical at e53c2fa AND at 658a7a37), so the borrow's
#: `submodule.alternateLocation=superproject` path needs no submodule-side network work.
#:
#: THE SAME TWO-PROBE TRAP recurs here and is recorded rather than left for the next reader:
#: `is-ancestor 5d2ad1e8 031ea42b` -> rc 1 and `ls-tree 031ea42b -- src/ghost_ring.h` -> 0 entries
#: (control needle src/triton.h -> 1). Read alone those say the ghost-ring defect is PRESENT. Both
#: ask the FORK's question of an ORNL-descended tree. The deciding commands agree with the parent
#: entries value for value: `git grep -c "write_output_ghost_ring\|read_output_ghost_ring"
#: 031ea42b -- src/output.h src/triton.h` -> output.h 5, triton.h 1, identical at e53c2fa.
#: Evidentiary strength is a38338b0's, no stronger: whatever falsifies a38338b0 falsifies this.
SHA_MAIN_SNAPSHOT_AND_COLUMN_SPLIT = "031ea42b063ce1083e3d965597cdb44da2e748ef"

#: The commit that INTRODUCES the snapshot traversal-guard defect, by adding the SWMM state
#: snapshot's dynamic-wave routing payload: `5fadf167` "extend the state snapshot over dynamic-wave
#: routing state". It is `introduced_in` for TRITON-SNAPSHOT-TRAVERSAL-GUARD, and carrying it is
#: what keeps the entry from claiming a pre-snapshot build is affected.
#: ONE SEARCH TRAP, recorded because taking the other answer registers the defect ABSENT on the
#: pin that carries it: `git log -S` in its `--all` form also surfaces `1a80bcaf`, which is
#: contained ONLY in the unmerged side branch `solver-work/replay-precision-accounting` and is NOT
#: an ancestor of any campaign pin. Measured 2026-10-14 against this sha, not inherited:
#:   * `git merge-base --is-ancestor 5fadf167 031ea42b` -> rc 0
#:   * `git merge-base --is-ancestor 5fadf167 c7b2fd1d` -> rc 0
#:   * `git merge-base --is-ancestor 5fadf167 15eb18a5` -> rc 1 (the defect postdates that build)
SHA_SNAPSHOT_DW_ROUTING_STATE = "5fadf167487a647e9978c2c1aa0bd6ec137e423e"

#: The commit that REMOVES it: `23b75781` "fix(swmm): land work package T1a -- the snapshot
#: traversal guard's wrong predicate". Re-predicates the per-element guard on `s->routeTo >= 0`,
#: repositions it, and selects the write target through a three-way chain so a route-less outfall
#: reaches a placeholder rather than forming an address from NULL.
#: THE FIX'S OWN SIGNATURE, measured rather than described, with the file confirmed to RESOLVE at
#: both refs so a zero is an absence and not a missing path:
#:   * `git grep -c "s->routeTo >= 0" c7b2fd1d -- external/swmm/src/solver/snapshot.c` -> 1
#:   * the same at `031ea42b` -> 0
#:   * `#define SNAPSHOT_VERSION` -> 6 at c7b2fd1d, 5 at 031ea42b, so a post-fix READER refuses a
#:     pre-fix file rather than misreading it. That version bump is why a mixed-pin resume fails
#:     loudly instead of silently.
SHA_SNAPSHOT_TRAVERSAL_GUARD_FIX = "23b757810cde67a83a0aeed003689bb6009f6894"

#: THE LIVE PIN as of this move: `c7b2fd1d` "test(swmm): land work package T2a -- killed-write
#: consultation regression tests", HEAD of `wp1/snapshot-guard` on the fork, 2 commits past
#: `031ea42b` and pushed (`rev-list --left-right --count HEAD...@{upstream}` -> `0 0`).
#: It is the `known_absent_in` member for every defect in REGISTRY, which is what
#: `scripts/check_live_pin_registered.py` asserts at commit time.
#: Measured 2026-10-14 against THIS sha, never inherited from a parent entry:
#:   * `git merge-base --is-ancestor a38338b0 c7b2fd1d` -> rc 0
#:   * `git merge-base --is-ancestor 3a832f7d c7b2fd1d` -> rc 0
#:   * `git merge-base --is-ancestor 9db367dd c7b2fd1d` -> rc 0
#:   * `git merge-base --is-ancestor 031ea42b c7b2fd1d` -> rc 0
#:   * `git merge-base --is-ancestor 23b75781 c7b2fd1d` -> rc 0 (it carries the snapshot fix)
#:   * `git ls-tree c7b2fd1d external/kokkos external/yaml-cpp` -> dc1bf738 a83cd315, BYTE-EQUAL
#:     to `031ea42b`; `external/swmm` is a TREE (mode 040000) rather than a gitlink and it DOES
#:     move (92ccd1b4 -> 4f987437), which is expected because the fix lives inside it.
#: THE TWO-PROBE TRAP RECURS AND IS RECORDED RATHER THAN LEFT FOR THE NEXT READER:
#: `is-ancestor 5d2ad1e8 c7b2fd1d` -> rc 1. Read alone that says the ghost-ring defect is PRESENT
#: here. It asks the FORK's question of an ORNL-descended tree, exactly as it does at `031ea42b`;
#: the deciding basis is `a38338b0` -> rc 0, and this sha's ghost-ring verdict is that entry's,
#: no stronger.
SHA_WP1_SNAPSHOT_GUARD_AND_TESTS = "c7b2fd1da3c2ed48d0d8cc0bf0e4b703b1f86e5e"

#: T4a's landing commit on the fork: a coupled CLEAN START disposes of the earlier run's SWMM
#: snapshot artifacts (the disposal reads the cfg key `swmm_snapshot_keep_previous`), and a
#: refused snapshot save stops the restart. Registered as the fix of
#: TRITON-SNAPSHOT-STALE-AT-CLEAN-START below.
SHA_SNAPSHOT_CLEAN_START_FIX = "cf357e4b4c4cf2cc1bfb2af2ad2210808b75d1a0"

#: The solver pin of package 3a (`3a.1`): the fork head after packages T7a and T8a, which also
#: carries WP1's T1b-T6a (`c7b2fd1d` is its ancestor). Measured against this sha on the fork
#: clone, each `git merge-base --is-ancestor {fix} 320b3313`: 3a832f7d rc 0, 9db367dd rc 0,
#: 23b75781 rc 0, cf357e4b rc 0, a38338b0 rc 0; and 5d2ad1e8 rc 1 -- the same two-probe trap
#: the ghost-ring entry records, so that entry's verdict here rests on a38338b0, as at c7b2fd1d.
SHA_SNAPSHOT_REFUSAL_PIN = "320b3313c99c3e929e77a258ff6e4b041bae2927"

REGISTRY: tuple[ModelDefect, ...] = (
    ModelDefect(
        defect_id="TRITON-COUPLED-RESUME-REPLAY",
        title="Coupled SWMM re-initializes from t=0 on a hotstart resume",
        fixed_in=SHA_COUPLED_RESUME_FIX,
        known_absent_in=frozenset(
            {
                SHA_COUPLED_RESUME_FIX,
                SHA_PRE_DEPTH_SCATTER,
                SHA_DEPTH_SCATTER_FIX,
                SHA_EXTBC_GHOST_RING_FIX,
                SHA_MAIN_GHOST_RING_AND_GPU,
                # 3a832f7d is an ANCESTOR of a38338b0 -- cached ancestry, the ordinary basis.
                SHA_ORNL_GHOST_RING_AND_GPU,
                # 3a832f7d is an ANCESTOR of 658a7a37 -- measured rc 0, same ordinary basis.
                SHA_SOLVER_WORK_REPLAY_PRECISION,
                # 3a832f7d is an ANCESTOR of 01e95a76 -- re-measured rc 0, not inherited.
                SHA_WP1C_SWMM_OUTPUT_APPEND,
                # e53c2fa descends from 01e95a76; re-measured rc 0 against this sha, not inherited.
                SHA_WP1C_LINKAGE_REPAIR_AND_TEST,
                # 3a832f7d is an ANCESTOR of 031ea42b -- re-measured rc 0 against this sha.
                SHA_MAIN_SNAPSHOT_AND_COLUMN_SPLIT,
                # 3a832f7d is an ANCESTOR of c7b2fd1d -- re-measured rc 0 against this sha.
                SHA_WP1_SNAPSHOT_GUARD_AND_TESTS,
                # 3a832f7d is an ANCESTOR of 320b3313 -- measured rc 0 against this sha.
                SHA_SNAPSHOT_REFUSAL_PIN,
            }
        ),
        also_present_in=frozenset({SHA_PRE_COUPLED_RESUME}),
        trigger="resumed_coupled",
        remedy="Re-run these sims at a TRITON carrying the coupled-resume fix.",
    ),
    ModelDefect(
        defect_id="TRITON-RESUME-DEPTH-SCATTER",
        title="Replayed SWMM node depths are never scattered to the per-rank new_depth[]",
        fixed_in=SHA_DEPTH_SCATTER_FIX,
        known_absent_in=frozenset(
            # 9db367dd is an ANCESTOR of a38338b0 -- cached ancestry, the ordinary basis.
            {
                SHA_DEPTH_SCATTER_FIX,
                SHA_EXTBC_GHOST_RING_FIX,
                SHA_MAIN_GHOST_RING_AND_GPU,
                SHA_ORNL_GHOST_RING_AND_GPU,
                # 9db367dd is an ANCESTOR of 658a7a37 -- measured rc 0; it is that branch's base.
                SHA_SOLVER_WORK_REPLAY_PRECISION,
                # 9db367dd is an ANCESTOR of 01e95a76 -- re-measured rc 0, not inherited.
                SHA_WP1C_SWMM_OUTPUT_APPEND,
                # e53c2fa descends from 01e95a76; re-measured rc 0 against this sha, not inherited.
                SHA_WP1C_LINKAGE_REPAIR_AND_TEST,
                # 9db367dd is an ANCESTOR of 031ea42b -- re-measured rc 0 against this sha.
                SHA_MAIN_SNAPSHOT_AND_COLUMN_SPLIT,
                # 9db367dd is an ANCESTOR of c7b2fd1d -- re-measured rc 0 against this sha.
                SHA_WP1_SNAPSHOT_GUARD_AND_TESTS,
                # 9db367dd is an ANCESTOR of 320b3313 -- measured rc 0 against this sha.
                SHA_SNAPSHOT_REFUSAL_PIN,
            }
        ),
        also_present_in=frozenset({SHA_PRE_COUPLED_RESUME, SHA_COUPLED_RESUME_FIX, SHA_PRE_DEPTH_SCATTER}),
        trigger="resumed_coupled",
        remedy="Re-run at a TRITON carrying the node-depth scatter fix (9db367d or a descendant).",
    ),
    ModelDefect(
        defect_id="TRITON-RESUME-EXTBC-GHOST-RING",
        title="The extbc perimeter ghost ring is not restored across a hotstart resume",
        fixed_in=SHA_EXTBC_GHOST_RING_FIX,
        known_absent_in=frozenset(
            # a38338b0's TREE carries an INDEPENDENT upstream implementation of this fix, landed at
            # its parent 0cf5faff; ancestry cannot reach it from 5d2ad1e8, which is on the other
            # remote. See the constant's comment for the basis and for what would falsify it.
            # 658a7a37 descends from a38338b0 (rc 0), so it inherits that entry's verdict by
            # ordinary cached ancestry -- NOT from 5d2ad1e8, which is not its ancestor. See the
            # SHA_SOLVER_WORK_REPLAY_PRECISION comment for the two probes that classify it wrong.
            {
                SHA_EXTBC_GHOST_RING_FIX,
                SHA_MAIN_GHOST_RING_AND_GPU,
                SHA_ORNL_GHOST_RING_AND_GPU,
                SHA_SOLVER_WORK_REPLAY_PRECISION,
                # 01e95a76 descends from a38338b0 (re-measured rc 0) and its delta from 658a7a37
                # touches no ghost-ring path, so it inherits the SAME upstream-implementation
                # verdict on the SAME basis -- and it is listed explicitly because the read path
                # has no clone to derive that from.
                SHA_WP1C_SWMM_OUTPUT_APPEND,
                # e53c2fa descends from 01e95a76; re-measured rc 0 against this sha, not inherited.
                SHA_WP1C_LINKAGE_REPAIR_AND_TEST,
                # 031ea42b descends from a38338b0 (re-measured rc 0) and inherits that entry's
                # independent-upstream-implementation verdict on the SAME basis -- listed
                # explicitly because the read path has no clone to derive it from, and because
                # ancestry from 5d2ad1e8 answers rc 1 for the reason the constant's comment gives.
                SHA_MAIN_SNAPSHOT_AND_COLUMN_SPLIT,
                # c7b2fd1d descends from a38338b0 (re-measured rc 0) and its delta from 031ea42b
                # touches no ghost-ring path, so it inherits the SAME independent-upstream verdict
                # on the SAME basis. `is-ancestor 5d2ad1e8 c7b2fd1d` -> rc 1, the same two-probe
                # trap the parent entries record, which is why this member is listed and not derived.
                SHA_WP1_SNAPSHOT_GUARD_AND_TESTS,
                # 320b3313 descends from c7b2fd1d and from a38338b0 (both measured rc 0) and
                # inherits the SAME independent-upstream verdict on the SAME basis;
                # `is-ancestor 5d2ad1e8 320b3313` -> rc 1, the same trap, so it is listed.
                SHA_SNAPSHOT_REFUSAL_PIN,
            }
        ),
        also_present_in=frozenset(
            {SHA_PRE_COUPLED_RESUME, SHA_COUPLED_RESUME_FIX, SHA_PRE_DEPTH_SCATTER, SHA_DEPTH_SCATTER_FIX}
        ),
        trigger="resumed_any",
        remedy="Re-run resumed sims at 5d2ad1e8 or a descendant.",
        # The report itself lives in a private deployment estate, so only its
        # identifier is carried here — the estate's repository name is a private
        # identifier that the anonymization guard blocks from this public tree.
        # The dated slug is what makes the report findable to someone who has
        # access; naming the repo adds nothing for a reader who does not.
        bug_report=("bug_reports/2026-08-04_triton-swmm_resume-extbc-boundary-perturbation-amplified-by-coupling"),
        provenance_note=(
            "The fix sha 5d2ad1e8 sits on the fork branch `instrumented/extbc-ghost-probe`, whose "
            "name describes the branch's diagnostic commits (fc807d1, 2372793 -- both `diag:`), NOT "
            "the fix commit 5d2ad1e (`fix:`). All instrumentation is guarded by "
            "#ifdef TRITON_EXTBC_PROBE, which defaults OFF, so a default build of this sha is "
            "production-equivalent rather than a diagnostic build. Stated explicitly rather than "
            "left silent, because a registry that certifies a build whose branch name reads "
            "`instrumented/` should say on what basis. "
            "A SECOND build is certified for this defect on a DIFFERENT basis: a38338b0 on ORNL "
            "upstream carries the maintainers' own implementation of the same fix, landed at its "
            "parent 0cf5faff. Its absence verdict rests on that parent's commit and on ancestry, "
            "not on a content comparison with 5d2ad1e8 -- which no single clone can perform, since "
            "the two shas live on repositories that share a name and not a history. That is one "
            "step weaker than 21e666d6's blob-identical cherry-pick, and it is recorded as weaker."
        ),
    ),
    ModelDefect(
        defect_id="TRITON-SNAPSHOT-TRAVERSAL-GUARD",
        title="A route-less outfall aborts the SWMM snapshot traversal, truncating every snapshot written",
        fixed_in=SHA_SNAPSHOT_TRAVERSAL_GUARD_FIX,
        #: Set, rather than left None, because the snapshot payload this guard sits in did not
        #: exist before 5fadf167 on the `main` lineage -- so a pre-snapshot build cannot carry the
        #: defect and the default-present arm would otherwise claim it does. IT IS A
        #: LINEAGE-SPECIFIC FACT and NOT the whole story, which is why the explicit sets below
        #: carry every verdict: the fork side branch `solver-work/replay-precision-accounting`
        #: introduced its OWN snapshot payload, so `is-ancestor 5fadf167 658a7a37` -> rc 1 while
        #: `git ls-tree 658a7a37 -- external/swmm/src/solver/snapshot.c` -> 1 entry. Ancestry from
        #: this constant therefore answers the WRONG LINEAGE's question for three registered shas,
        #: the same two-probe trap the ghost-ring entry above records. The deciding instrument is
        #: CONTENT, and it is recorded per member below.
        introduced_in=SHA_SNAPSHOT_DW_ROUTING_STATE,
        known_absent_in=frozenset(
            {
                SHA_SNAPSHOT_TRAVERSAL_GUARD_FIX,
                # 23b75781 is an ANCESTOR of c7b2fd1d -- measured rc 0 against this sha.
                SHA_WP1_SNAPSHOT_GUARD_AND_TESTS,
                # 23b75781 is an ANCESTOR of 320b3313 -- measured rc 0 against this sha.
                SHA_SNAPSHOT_REFUSAL_PIN,
                # THE SEVEN NO-PAYLOAD SHAS. `external/swmm/src/solver/snapshot.c` does not exist
                # at any of them (`git ls-tree` -> 0 entries each, against a control needle
                # `src/triton.h` -> 1 entry each, so the zero is an absence and not a bad path).
                # There is no guard here to be wrong, so the defect cannot be carried. Listed
                # EXPLICITLY rather than derived, because the read path has no clone and the
                # `introduced_in_not_ancestor` arm needs one.
                SHA_PRE_COUPLED_RESUME,
                SHA_COUPLED_RESUME_FIX,
                SHA_PRE_DEPTH_SCATTER,
                SHA_DEPTH_SCATTER_FIX,
                SHA_EXTBC_GHOST_RING_FIX,
                SHA_MAIN_GHOST_RING_AND_GPU,
                SHA_ORNL_GHOST_RING_AND_GPU,
            }
        ),
        #: THE FOUR PAYLOAD-WITHOUT-FIX SHAS, every one decided by CONTENT rather than by
        #: ancestry, because ancestry reaches only one of the two lineages that introduced the
        #: payload. Measured 2026-10-14 over
        #: `external/swmm/src/solver/snapshot.c`, with the post-fix value as the positive control:
        #:   sha        s->routeTo>=0   SNAPSHOT_VERSION
        #:   658a7a37         0               4
        #:   01e95a76         0               5
        #:   e53c2fa          0               5
        #:   031ea42b         0               5
        #:   23b75781         1               6   <- control: the fix's own signature
        #:   c7b2fd1d         1               6   <- control
        #: A zero in that column IS the defective predicate still in place, and the two 1s are
        #: what establish the matcher reaches this population at all.
        #: 658a7a37 is the one a reader will question: it is on the UNMERGED side branch
        #: `solver-work/replay-precision-accounting` (`git branch -a --contains` names only that
        #: branch) and is NOT an ancestor of the new pin (`is-ancestor 658a7a37 c7b2fd1d` -> rc 1),
        #: so its payload is that branch's own and the fix will never reach it by descent.
        #: OPERATIONAL CONSEQUENCE: every snapshot written at any of these four is short by
        #: 2,287 doubles, and a truncated file passes every pre-payload integrity check because
        #: the header, shape block and manifest are all written before the abort.
        also_present_in=frozenset(
            {
                SHA_MAIN_SNAPSHOT_AND_COLUMN_SPLIT,
                SHA_SOLVER_WORK_REPLAY_PRECISION,
                SHA_WP1C_SWMM_OUTPUT_APPEND,
                SHA_WP1C_LINKAGE_REPAIR_AND_TEST,
            }
        ),
        trigger="resumed_coupled",
        remedy=(
            "Re-run resumed coupled sims at 23b75781 or a descendant. A snapshot written at an "
            "affected build is unusable rather than degraded, so there is no partial-credit arm: "
            "the reader refuses it at the first read past the written extent."
        ),
        provenance_note=(
            "WHAT THE DEFECT IS, because the registry's other rows are resume-path defects and this "
            "one is a WRITE-path defect whose only observable is on resume. A null guard in "
            "external/swmm/src/solver/snapshot.c tested the pollutant count where the array it "
            "protects is allocated on a routing field, so it fired on any model whose outfall has no "
            "routing destination. It sits ABOVE the pollutant loop and the traversal is shared across "
            "manifest, write and read modes, so it aborted the remainder of every traversal in every "
            "mode -- a build that cannot write a usable snapshot also cannot read one. "
            "WHY A TRUNCATED SNAPSHOT PASSED EVERY PRE-PAYLOAD CHECK: the header, the shape block and "
            "the embedded manifest were all written BEFORE the abort, so integrity checks that read "
            "only those were satisfied, and the file was correctly refused only at the first read past "
            "the written extent. A reader looking for a corrupt-file signature will not find one. "
            "WHY 15eb18a5 IS NOT IN also_present_in, stated because the three sibling entries all list "
            "it and a reader will expect a fourth: external/swmm/src/solver/snapshot.c does not exist "
            "at that sha (`git ls-tree` -> 0 entries, against a control needle src/triton.h -> 1), and "
            "`is-ancestor 5fadf167 15eb18a5` -> rc 1. The defect postdates that build, so listing it "
            "would be the laundering the live-pin guard's own remedy text forbids. "
            "ONE CONSEQUENCE FOR MIXED-PIN DATA: SNAPSHOT_VERSION moves 5 -> 6 across the fix, so a "
            "post-fix reader refuses a pre-fix snapshot loudly rather than misreading it. A campaign "
            "holding both pins therefore fails fast on a cross-pin resume instead of producing a "
            "plausible wrong answer."
        ),
    ),
    ModelDefect(
        defect_id="TRITON-SNAPSHOT-STALE-AT-CLEAN-START",
        title=(
            "A coupled run's clean start left an earlier run's SWMM snapshots in place, so after a "
            "failed or refused snapshot save a resume could load the earlier run's SWMM state"
        ),
        fixed_in=SHA_SNAPSHOT_CLEAN_START_FIX,
        #: Set, rather than left None, for the traversal entry's reason: a build with no
        #: snapshot payload writes no snapshot, so it has none to leave behind.
        introduced_in=SHA_SNAPSHOT_DW_ROUTING_STATE,
        #: Every verdict below is decided by CONTENT and listed explicitly, because the read
        #: path has no clone. Measured on the fork clone per registered sha:
        #: `git ls-tree {sha} -- external/swmm/src/solver/snapshot.c` (the payload; control
        #: `src/triton.h` -> 1 entry at every sha) and
        #: `git show {sha}:src/swmm_triton.h | grep -c swmm_snapshot_keep_previous` (the
        #: clean-start disposal's cfg key). No payload -> absent; payload and disposal -> absent;
        #: payload without disposal -> present.
        known_absent_in=frozenset(
            {
                # payload 1, disposal 5 at both (and cf357e4b is an ancestor of 320b3313, rc 0).
                SHA_SNAPSHOT_CLEAN_START_FIX,
                SHA_SNAPSHOT_REFUSAL_PIN,
                # THE SEVEN NO-PAYLOAD SHAS: snapshot.c -> 0 entries at each.
                SHA_PRE_COUPLED_RESUME,
                SHA_COUPLED_RESUME_FIX,
                SHA_PRE_DEPTH_SCATTER,
                SHA_DEPTH_SCATTER_FIX,
                SHA_EXTBC_GHOST_RING_FIX,
                SHA_MAIN_GHOST_RING_AND_GPU,
                SHA_ORNL_GHOST_RING_AND_GPU,
            }
        ),
        #: THE SEVEN PAYLOAD-WITHOUT-DISPOSAL SHAS: snapshot.c -> 1 entry, the disposal key -> 0
        #: lines, and `is-ancestor cf357e4b {sha}` -> rc 1 at each.
        also_present_in=frozenset(
            {
                SHA_SOLVER_WORK_REPLAY_PRECISION,
                SHA_WP1C_SWMM_OUTPUT_APPEND,
                SHA_WP1C_LINKAGE_REPAIR_AND_TEST,
                SHA_MAIN_SNAPSHOT_AND_COLUMN_SPLIT,
                SHA_SNAPSHOT_DW_ROUTING_STATE,
                SHA_SNAPSHOT_TRAVERSAL_GUARD_FIX,
                SHA_WP1_SNAPSHOT_GUARD_AND_TESTS,
            }
        ),
        trigger="resumed_coupled",
        remedy="Re-run resumed coupled sims at cf357e4b or a descendant.",
    ),
)

REGISTRY_BY_ID = {d.defect_id: d for d in REGISTRY}


def resolve_all(producing_sha, *, is_ancestor=None) -> tuple[DefectVerdict, ...]:
    return tuple(resolve(d, producing_sha, is_ancestor=is_ancestor) for d in REGISTRY)


def resolve_for_tree_attrs(attrs, *, is_ancestor=None) -> tuple[DefectVerdict, ...]:
    """Classify every registered defect from a consolidated tree's ROOT ATTRS.

    Reads only ``triton_producing_sha`` -- a value ALREADY stamped by
    ``processing_analysis._stamp_triton_provenance`` (which writes the system log's
    ``triton_head_sha`` under this tree-side name; the rename is why grepping either name
    alone finds only half the chain). Writes nothing: this is the read-side derivation that
    replaces the stamped booleans.
    """
    return resolve_all((attrs or {}).get("triton_producing_sha"), is_ancestor=is_ancestor)
