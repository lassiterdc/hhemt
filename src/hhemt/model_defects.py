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


_TRITON_REPO = "https://github.com/lassiterdc/triton.git"

#: Full shas, kept as module constants so a reader can grep one place for "which build".
SHA_COUPLED_RESUME_FIX = "3a832f7d5eedd96aaee0dfe9181da5774adfb9f4"
SHA_DEPTH_SCATTER_FIX = "9db367ddc79f86c7f708686d1dd805dc992fb0a4"
SHA_EXTBC_GHOST_RING_FIX = "5d2ad1e8adf9a85d7df14e885b76e59a10f9a98b"

#: Historical PRODUCING shas whose ancestry was resolved ONCE here, at registry-authoring time,
#: against a clone that had them — the read path has no clone (see module docstring) and would
#: otherwise return INDETERMINATE for every sha not already named above, leaving the pre-fix arms
#: unreachable in production. These are project-relevant, not illustrative: 15eb18a5 is the pin the
#: container-validation suite names, and b3820a4 is the sha the defect-to-version ruling keys on.
SHA_PRE_COUPLED_RESUME = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"
SHA_PRE_DEPTH_SCATTER = "b3820a448f304b3f732f4b6fac5564adf86ac333"

REGISTRY: tuple[ModelDefect, ...] = (
    ModelDefect(
        defect_id="TRITON-COUPLED-RESUME-REPLAY",
        title="Coupled SWMM re-initializes from t=0 on a hotstart resume",
        fixed_in=SHA_COUPLED_RESUME_FIX,
        known_absent_in=frozenset(
            {SHA_COUPLED_RESUME_FIX, SHA_PRE_DEPTH_SCATTER, SHA_DEPTH_SCATTER_FIX, SHA_EXTBC_GHOST_RING_FIX}
        ),
        also_present_in=frozenset({SHA_PRE_COUPLED_RESUME}),
        trigger="resumed_coupled",
        remedy="Re-run these sims at a TRITON carrying the coupled-resume fix.",
    ),
    ModelDefect(
        defect_id="TRITON-RESUME-DEPTH-SCATTER",
        title="Replayed SWMM node depths are never scattered to the per-rank new_depth[]",
        fixed_in=SHA_DEPTH_SCATTER_FIX,
        known_absent_in=frozenset({SHA_DEPTH_SCATTER_FIX, SHA_EXTBC_GHOST_RING_FIX}),
        also_present_in=frozenset(
            {SHA_PRE_COUPLED_RESUME, SHA_COUPLED_RESUME_FIX, SHA_PRE_DEPTH_SCATTER}
        ),
        trigger="resumed_coupled",
        remedy="Re-run at a TRITON carrying the node-depth scatter fix (9db367d or a descendant).",
    ),
    ModelDefect(
        defect_id="TRITON-RESUME-EXTBC-GHOST-RING",
        title="The extbc perimeter ghost ring is not restored across a hotstart resume",
        fixed_in=SHA_EXTBC_GHOST_RING_FIX,
        known_absent_in=frozenset({SHA_EXTBC_GHOST_RING_FIX}),
        also_present_in=frozenset(
            {SHA_PRE_COUPLED_RESUME, SHA_COUPLED_RESUME_FIX, SHA_PRE_DEPTH_SCATTER, SHA_DEPTH_SCATTER_FIX}
        ),
        trigger="resumed_any",
        remedy="Re-run resumed sims at 5d2ad1e8 or a descendant.",
        bug_report=(
            "hhemt_projects/bug_reports/2026-08-04_triton-swmm_resume-extbc-boundary-perturbation-amplified-by-coupling"
        ),
        provenance_note=(
            "The fix sha 5d2ad1e8 sits on the fork branch `instrumented/extbc-ghost-probe`, whose "
            "name describes the branch's diagnostic commits (fc807d1, 2372793 -- both `diag:`), NOT "
            "the fix commit 5d2ad1e (`fix:`). All instrumentation is guarded by "
            "#ifdef TRITON_EXTBC_PROBE, which defaults OFF, so a default build of this sha is "
            "production-equivalent rather than a diagnostic build. Stated explicitly rather than "
            "left silent, because a registry that certifies a build whose branch name reads "
            "`instrumented/` should say on what basis."
        ),
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
