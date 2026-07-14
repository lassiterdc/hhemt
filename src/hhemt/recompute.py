"""ADR-16 recompute RESOLVER — classify which analysis scopes a bug-fix commit
invalidates, and emit the surgical recompute instruction that repairs them.

Given a bug-fix commit id (+ an explicit ``--scope``), the resolver reads the
ADR-15 per-scope version-provenance stamps to find which scopes are PRE-FIX
(produced by code lacking the fix), and maps each onto a value of the FROZEN
``RecomputeAction`` class. It never exercises LLM judgement, never derives scope
from changed-file paths (that mis-scopes a cosmetic edit as a numerics bug), and
NEVER touches the filesystem — it only EMITS ``reprocess(start_with=)`` /
``run(override_force_rerun=)`` call descriptors for the operator to run.

Layering (ADR-16/17):
    * Phase 2 (this module) owns the frozen ``RecomputeAction`` / ``RecomputeScope``
      enums, the 8-cell scope x clear_raw -> action table, the git-framed ancestry
      predicate + semver fallback, the scoped force-rerun emission, and the
      registry-match consuming interface (``match_registry_against_stamps``).
    * Phase 3 ships the ``invalidating_fixes.yaml`` registry + its Pydantic loader
      (``hhemt.config.invalidating_fixes``) and wires ``check_invalidating_fixes``
      into ``validate_analysis``. This module loads that registry LAZILY and
      degrades to "no matches" when it is not yet present (graceful-absent).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hhemt.analysis import TRITONSWMM_analysis


# --------------------------------------------------------------------------- #
# Frozen action / scope enums (versioning-specialist Q2/Q3).
#
# Both are ``(str, Enum)`` NOT ``StrEnum``: StrEnum landed in Python 3.11, and the
# ``(str, Enum)`` mix-in gives the same string-identity + clean YAML/JSON
# serialization (the members are written into the ``hhemt recompute-plan`` output
# and compared against the registry's fields) while staying import-safe on 3.10.
# ``analysis_validation.IssueLevel`` is a bare ``Enum`` because it is internal-only
# and never serialized; ``RecomputeAction`` IS serialized, so it needs ``str``.
# --------------------------------------------------------------------------- #
class RecomputeAction(str, Enum):  # noqa: UP042 -- (str, Enum) is deliberate; see the comment block above (NOT StrEnum)
    """The four recompute tiers, each a thin dispatch into existing machinery.

    * ``RE_RUN``              -> scoped ``run(override_force_rerun=...)`` per affected
                                 ``sa_id`` / ``event_iloc`` (surgical; NOT a
                                 ``from_scratch`` whole-tree wipe).
    * ``REPROCESS_SCENARIO``  -> ``reprocess(start_with="process", regenerate_existing=True)``.
    * ``RE_CONSOLIDATE``      -> ``reprocess(start_with="consolidate")``.
    * ``NONE_COSMETIC``       -> ``reprocess(start_with="render")`` (re-render only).
    """

    RE_RUN = "re-run"
    REPROCESS_SCENARIO = "reprocess-scenario"
    RE_CONSOLIDATE = "re-consolidate"
    NONE_COSMETIC = "none-cosmetic"


class RecomputeScope(str, Enum):  # noqa: UP042 -- (str, Enum) is deliberate; see the comment block above (NOT StrEnum)
    """Which production tier a bug lives in — designated EXPLICITLY, never inferred
    from changed-file paths (the ADR-16 flip-condition: path->scope auto-derivation
    is fragile and mis-scopes a cosmetic edit as a scenario-processing NUMERICS bug)."""

    SIMULATION = "simulation"                    # solver numerics (run tier)
    SCENARIO_PROCESSING = "scenario-processing"  # per-scenario summary calc (process tier)
    CONSOLIDATION = "consolidation"              # datatree assembly (consolidate tier)
    COSMETIC = "cosmetic"                        # render/label only, no numeric change


# --------------------------------------------------------------------------- #
# The frozen scope x clear_raw -> action mapping (versioning-specialist Q2).
#
# The bool key is ``clear_raw`` (True => the raw ``out_*`` outputs were deleted).
# The load-bearing cell is (SCENARIO_PROCESSING, clear_raw=True) -> RE_RUN: once
# the raw outputs are gone, ``reprocess(start_with="process")`` has no rebuild
# source and would raise, so a scenario-processing bug must escalate to a full
# re-run. Encoding the escalation IN the table keeps all 8 cells auditable and
# exhaustively unit-testable in one place.
# --------------------------------------------------------------------------- #
_SCOPE_CLEAR_RAW_ACTIONS: dict[tuple[RecomputeScope, bool], RecomputeAction] = {
    (RecomputeScope.SIMULATION, False): RecomputeAction.RE_RUN,
    (RecomputeScope.SIMULATION, True): RecomputeAction.RE_RUN,
    (RecomputeScope.SCENARIO_PROCESSING, False): RecomputeAction.REPROCESS_SCENARIO,
    (RecomputeScope.SCENARIO_PROCESSING, True): RecomputeAction.RE_RUN,  # raw gone -> reprocess cannot rebuild
    (RecomputeScope.CONSOLIDATION, False): RecomputeAction.RE_CONSOLIDATE,
    (RecomputeScope.CONSOLIDATION, True): RecomputeAction.RE_CONSOLIDATE,
    (RecomputeScope.COSMETIC, False): RecomputeAction.NONE_COSMETIC,
    (RecomputeScope.COSMETIC, True): RecomputeAction.NONE_COSMETIC,
}


def resolve_recompute_action(scope: RecomputeScope, clear_raw: bool) -> RecomputeAction:
    """Look up the frozen 8-cell (scope, clear_raw) -> RecomputeAction table."""
    return _SCOPE_CLEAR_RAW_ACTIONS[(scope, clear_raw)]


# --------------------------------------------------------------------------- #
# Git-framed pre-fix ancestry predicate (git-specialist Q2/Q3).
# --------------------------------------------------------------------------- #
def _in_git_checkout() -> bool:
    """True iff CWD is inside a git work tree (False off an installed wheel).

    Detect the no-git state cleanly with a single probe rather than
    exception-driven control flow; gate ``_is_scope_pre_fix`` behind it so the
    ``cat-file`` / ``merge-base`` calls are skipped entirely off a wheel.
    """
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _repo_is_shallow() -> bool:
    """True on a shallow clone, where ancestry can be silently WRONG past the frontier."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            capture_output=True,
            text=True,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _sha_exists(sha: str) -> bool:
    """True iff ``sha`` resolves to a commit object in the local object DB.

    Pre-gating both operands turns the dangerous exit 128 (bogus/absent sha,
    e.g. a scope built on another machine whose commit was never fetched) into an
    explicit indeterminate branch rather than a spurious "not-ancestor" -> pre-fix
    -> force-recompute-of-already-fixed-data misclassification.
    """
    try:
        return (
            subprocess.run(
                ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                capture_output=True,
            ).returncode
            == 0
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _is_scope_pre_fix(scope_sha: str | None, fix_sha: str) -> bool | None:
    """Decide whether a scope was produced by code LACKING ``fix_sha``.

    Returns:
        True  -- scope predates the fix (pre-fix / AFFECTED -> recompute)
        False -- scope provably contains the fix (post-fix -> no recompute)
        None  -- ancestry is undecidable (unstamped scope, installed wheel,
                 shallow clone, unfetched/rewritten sha) -> the caller falls back
                 to the semver comparison.

    Ported from the git-specialist ``_classify_scope_against_fix`` VMS into this
    ``(scope_sha, fix_sha)`` signature. Uses
    ``git merge-base --is-ancestor {fix_sha} {scope_sha}`` (empirically confirmed
    on git 2.43.0):

        exit 0   -> fix IS an ancestor of scope -> scope HAS the fix   -> False
        exit 1   -> fix is NOT an ancestor      -> scope LACKS the fix -> True
        128 / shallow / missing operand / no-git -> None (semver fallback)

    C3 identifier-length invariant: ``scope_sha`` is a 12-char short sha (the
    ADR-15 stamp), ``fix_sha`` the registry's 40-char full sha. They are compared
    ONLY via git object resolution (``cat-file`` / ``merge-base``, length-agnostic),
    NEVER by Python string ``==``. The exit-0 branch handles the produced-AT-fix
    reflexive case (``--is-ancestor`` is reflexive), so NO string-equality guard is
    added — adding one would compare a 12-char against a 40-char sha and mis-handle
    the reflexive case the git predicate already resolves correctly.
    """
    if scope_sha is None or scope_sha == "unknown":
        return None  # unstamped / off-checkout -> emit INFO upstream (D6)
    if not _in_git_checkout():
        return None  # installed wheel: no object DB -> semver fallback
    if _repo_is_shallow():
        return None  # ancestry can be silently wrong past the shallow frontier
    if not (_sha_exists(fix_sha) and _sha_exists(scope_sha)):
        return None  # either operand absent from local history -> indeterminate
    try:
        rc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", fix_sha, scope_sha],
            capture_output=True,
        ).returncode
    except (subprocess.SubprocessError, FileNotFoundError):
        return None  # git vanished mid-call -> indeterminate
    if rc == 0:
        return False  # fix is an ancestor of scope -> scope has the fix
    if rc == 1:
        return True   # fix not an ancestor -> scope predates the fix
    return None       # rc >= 2 (128 / unexpected) -> undecidable, never crash


def _is_scope_affected_by_semver(
    scope_semver: str | None, affected_version_range: str
) -> bool | None:
    """Semver fallback for the indeterminate-ancestry branch (T1 -- Option C).

    Evaluate ``scope_semver in SpecifierSet(affected_version_range)``. This is the
    ONE ``SpecifierSet(affected_version_range)`` evaluator the Phase-4 skew check
    also shares; ``introduced_in_version`` is deliberately NOT read here. Returns
    None when the scope semver is absent/unresolvable (caller emits the D6 INFO).
    """
    if scope_semver is None or scope_semver in ("unknown", "0+unknown"):
        return None
    # Imported lazily so a bare ``import hhemt.recompute`` stays cheap and the
    # dependency surface is explicit at the call site.
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        return Version(scope_semver) in SpecifierSet(affected_version_range)
    except (InvalidVersion, InvalidSpecifier):
        return None


# --------------------------------------------------------------------------- #
# Scoped force-rerun emission (hhemt-specialist Q1 / plan D4).
# --------------------------------------------------------------------------- #
def _emit_re_run_instruction(
    analysis: TRITONSWMM_analysis, prefix_scopes: set[str | int]
) -> dict:
    """Map ``RecomputeAction.RE_RUN`` -> a scoped ``run(override_force_rerun=...)`` plan.

    ``prefix_scopes`` is the set of (sa_id | event_iloc) identifiers whose ADR-15
    stamp is a pre-fix ancestor of the bug-fix commit. Uses the surgical scoped
    force-rerun (``analysis.py`` ``_apply_force_rerun``), NEVER
    ``run(from_scratch=True)``: from_scratch wipes the whole ``analysis_dir`` and
    destroys correct post-fix scenarios, defeating ADR-15's per-scope drift
    capture. The key is ``sa_id`` for a sensitivity master, ``event_iloc`` otherwise
    (an off-by-one key raises ``ConfigurationError`` at the API boundary -- a loud,
    safe failure, not silent).
    """
    if analysis.cfg_analysis.toggle_sensitivity_analysis:
        force = {"sa_id": sorted(str(s) for s in prefix_scopes)}
    else:
        force = {"event_iloc": sorted(int(s) for s in prefix_scopes)}
    return {
        "action": "re-run",
        "call": "analysis.run",
        "kwargs": {"override_force_rerun": force},
        "rationale": (
            "scenario-scoped bug with raw outputs cleared -> re-execute the "
            "affected sims; scoped (not from_scratch) preserves correct "
            "post-fix scenarios and the shared system tier"
        ),
    }


# --------------------------------------------------------------------------- #
# Registry-match consuming interface (hhemt Q4/Q5; D6).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegistryMatch:
    """One resolver verdict about a scope against a registered invalidating fix.

    ``recommended_action`` is None ONLY on the INFO channel (an unstamped /
    unclassifiable scope, D6). ``match_registry_against_stamps`` returns ONLY
    actionable ``{warning, error}`` matches (non-None ``recommended_action``);
    the INFO records travel on the SEPARATE ``classify_unstamped_scopes`` channel
    so Phase-3's ``check_invalidating_fixes`` never dereferences
    ``.recommended_action.value`` on a None.
    """

    commit_id: str | None
    severity: str
    recommended_action: RecomputeAction | None
    affected_scope: str
    summary: str


def _classify_unstamped_scope(scope_id: str) -> RegistryMatch:
    """D6: an unstamped scope -> a non-blocking INFO record, NO verdict.

    Returns a ``RegistryMatch`` with ``severity="info"`` and
    ``recommended_action=None`` so the report surfaces it as an informational row,
    not a pass/fail. NEVER a recompute recommendation, NEVER raises, NEVER silently
    skips (mirrors the graceful-absent load pattern + ADR-17's non-silent-degrade
    principle). ``severity="info"`` here is a RESOLVER-INTERNAL status value, kept
    distinct from the registry's frozen ``{warning, error}`` severity enum so the
    two do not couple.
    """
    return RegistryMatch(
        commit_id=None,
        severity="info",
        recommended_action=None,
        affected_scope=scope_id,
        summary=(
            f"Producing toolkit version unknown for scope {scope_id!r} "
            "(pre-stamp tree or off-checkout install); cannot determine "
            "pre/post-fix status. Re-run or reprocess this analysis to stamp "
            "the producing version, after which invalidating-fix checks become precise."
        ),
    )


def _load_invalidating_fix_registry(analysis: TRITONSWMM_analysis) -> list:
    """Lazily load the ADR-17 registry entries; graceful-absent -> ``[]``.

    The registry file + its Pydantic loader ship in Phase 3
    (``hhemt.config.invalidating_fixes``). Until then this returns ``[]`` so the
    resolver's consuming interface is exercisable now (a Phase-2 test stubs an
    empty registry through this seam). A malformed registry that raises at load is
    also degraded to ``[]`` here -- the load-time emission is non-blocking by
    construction (C-NON-BLOCKING-BUGEMIT); it must never crash analysis construction.
    """
    try:
        from hhemt.config.invalidating_fixes import load_invalidating_fixes
    except ImportError:
        return []  # Phase-3 module not present yet -- no registry, no matches
    try:
        return list(load_invalidating_fixes().fixes)
    except Exception:
        return []


def _iter_scope_stamps(analysis: TRITONSWMM_analysis):
    """Yield ``(scope_id, {"sha": str|None, "semver": str|None})`` per scope-zarr.

    The four ADR-15 stamp-carrier locations (hhemt Q4 confirmed set) are:
      1. per-scenario ``sims/{event_id}/processed/*.zarr``
      2. per-sub ``subanalyses/sa_{id}/analysis_datatree.zarr``
      3. the regular ``analysis_datatree.zarr`` (non-sensitivity consolidated tree)
      4. the master ``sensitivity_datatree.zarr`` (sensitivity master tree)
    with the asymmetry that a non-sensitivity analysis has scopes {1,3} and a
    sensitivity master has {1,2,4} (no regular tree at the master level).

    Phase 2 wires the consolidated-tree locations (3 / 4) that
    ``analysis.analysis_paths`` exposes directly; the per-scenario (1) and per-sub
    (2) sweep is completed in Phase 3, when a real registry makes it testable. The
    read uses the ADR-15 root fast-path (``tree.attrs["hhemt_producing_sha"]``,
    uniform common case) and falls back to ``read_producing_stamp`` under
    absence/divergence -- exactly the read model ``cf_conventions`` documents.
    """
    import xarray as xr

    from hhemt.cf_conventions import read_producing_stamp

    paths = analysis.analysis_paths
    if analysis.cfg_analysis.toggle_sensitivity_analysis:
        locations = [("sensitivity-master", getattr(paths, "sensitivity_datatree_zarr", None))]
    else:
        locations = [("analysis-datatree", getattr(paths, "analysis_datatree_zarr", None))]

    for scope_id, zarr_path in locations:
        if zarr_path is None or not zarr_path.exists():
            continue  # graceful-absent: scope not yet consolidated
        try:
            tree = xr.open_datatree(zarr_path, engine="zarr", chunks="auto", consolidated=False)
        except Exception:
            continue  # unreadable tree -> skip this scope (never crash the resolver)
        sha = tree.attrs.get("hhemt_producing_sha")
        semver = tree.attrs.get("hhemt_producing_version")
        if sha is None:  # divergent or absent scalar fast-path -> consult the coordinate
            stamp = read_producing_stamp(tree)
            sha = stamp["uniform"] if stamp else None
        yield scope_id, {"sha": sha, "semver": semver}


# --------------------------------------------------------------------------- #
# Ad-hoc-path resolver (the `hhemt recompute-plan` CLI entry point).
# --------------------------------------------------------------------------- #
def _scope_clear_raw(analysis: TRITONSWMM_analysis) -> bool:
    """Best-effort analysis-level raw-cleared rollup (hhemt Q2).

    The authoritative source is the per-model log booleans
    (``_all_raw_TRITON_outputs_cleared`` / ``_all_raw_SWMM_outputs_cleared``,
    computed-on-read from the persisted processing log -- NEVER a filesystem walk).
    Defaults to False when not determinable. Only the SCENARIO_PROCESSING row of
    the action table depends on this flag.
    """
    for attr in ("_all_raw_TRITON_outputs_cleared", "_all_raw_SWMM_outputs_cleared"):
        try:
            if bool(getattr(analysis, attr)):
                return True
        except Exception:
            continue
    return False


def _action_call_descriptor(
    action: RecomputeAction,
    analysis: TRITONSWMM_analysis,
    pre_fix_scopes: list[str],
) -> dict:
    """Map a resolved ``RecomputeAction`` -> the toolkit call it dispatches into.

    The resolver only ever EMITS these descriptors; it never runs them and never
    touches the filesystem. Each maps onto the existing ``reprocess(start_with=)``
    tier ladder or the scoped ``run(override_force_rerun=)`` machinery.
    """
    if action is RecomputeAction.RE_RUN:
        numeric = [s for s in pre_fix_scopes if str(s).lstrip("-").isdigit()]
        if numeric:
            return _emit_re_run_instruction(analysis, set(numeric))
        # Phase-2 coarse (whole-tree) stamps carry no per-event ids; per-scenario
        # force-rerun targeting resolves from the per-scenario ADR-15 stamps in
        # Phase 3. Surface the degrade rather than silently emitting an empty force.
        return {
            "action": "re-run",
            "call": "analysis.run",
            "kwargs": {"override_force_rerun": {"pre_fix_scopes": pre_fix_scopes}},
            "rationale": (
                "scenario-scoped re-run; per-scenario force-rerun targets resolve "
                "from the per-scenario ADR-15 stamps"
            ),
        }
    if action is RecomputeAction.REPROCESS_SCENARIO:
        return {
            "action": "reprocess-scenario",
            "call": "analysis.reprocess",
            "kwargs": {"start_with": "process", "regenerate_existing": True},
        }
    if action is RecomputeAction.RE_CONSOLIDATE:
        return {
            "action": "re-consolidate",
            "call": "analysis.reprocess",
            "kwargs": {"start_with": "consolidate"},
        }
    return {
        "action": "none-cosmetic",
        "call": "analysis.reprocess",
        "kwargs": {"start_with": "render"},
    }


def plan_recompute(
    analysis: TRITONSWMM_analysis, fix_sha: str, scope: RecomputeScope
) -> dict:
    """Ad-hoc dry-run resolver for ``hhemt recompute-plan``.

    Classifies this analysis's ADR-15-stamped scopes against a single bug-fix
    commit (``--commit``) using the git ancestry predicate, resolves the recompute
    tier from the operator-supplied ``--scope`` (D2 Option A -- scope is NEVER
    inferred from changed-file paths) and the analysis's ``clear_raw`` state, and
    emits the per-scope dry-run plan. Read-only: never submits, never touches the
    filesystem. Indeterminate scopes (unstamped / off-checkout) are reported as D6
    INFO, never as a false verdict.
    """
    clear_raw = _scope_clear_raw(analysis)
    action = resolve_recompute_action(scope, clear_raw)

    pre_fix: list[str] = []
    post_fix: list[str] = []
    indeterminate: list[str] = []
    for scope_id, stamp in _iter_scope_stamps(analysis):
        verdict = _is_scope_pre_fix(stamp.get("sha"), fix_sha)
        if verdict is True:
            pre_fix.append(scope_id)
        elif verdict is False:
            post_fix.append(scope_id)
        else:
            indeterminate.append(scope_id)

    plan = {
        "commit": fix_sha,
        "scope": scope.value,
        "clear_raw": clear_raw,
        "recommended_action": action.value,
        "pre_fix_scopes": pre_fix,
        "post_fix_scopes": post_fix,
        "indeterminate_scopes": indeterminate,
        "instruction": None,
    }
    if pre_fix:
        plan["instruction"] = _action_call_descriptor(action, analysis, pre_fix)
    return plan


def match_registry_against_stamps(
    analysis: TRITONSWMM_analysis,
) -> list[RegistryMatch]:
    """Return the ACTIONABLE ``{warning, error}`` registry matches for this analysis.

    Reads the registry lazily (graceful-absent -> ``[]`` until Phase 3 ships it) and
    iterates the ADR-15 scope stamps, classifying each scope against every
    registered fix via the git ancestry predicate with a semver fallback. Returns
    ONLY matches with a non-None ``recommended_action``; the D6 INFO records for
    unstamped/unclassifiable scopes travel on the separate
    ``classify_unstamped_scopes`` channel.
    """
    actionable, _info = _resolve_registry_matches(analysis)
    return actionable


def classify_unstamped_scopes(
    analysis: TRITONSWMM_analysis,
) -> list[RegistryMatch]:
    """Return the D6 INFO records (severity='info', recommended_action=None) for
    scopes that could not be classified -- the SEPARATE channel from the actionable
    ``match_registry_against_stamps`` result."""
    _actionable, info = _resolve_registry_matches(analysis)
    return info


def _resolve_registry_matches(
    analysis: TRITONSWMM_analysis,
) -> tuple[list[RegistryMatch], list[RegistryMatch]]:
    """Single-pass resolver feeding both the actionable and INFO channels."""
    registry = _load_invalidating_fix_registry(analysis)
    if not registry:
        return [], []  # Phase-2 empty-registry path (graceful-absent)

    actionable: list[RegistryMatch] = []
    info: list[RegistryMatch] = []
    for scope_id, stamp in _iter_scope_stamps(analysis):
        scope_sha = stamp.get("sha")
        scope_semver = stamp.get("semver")
        if scope_sha is None or scope_sha == "unknown":
            info.append(_classify_unstamped_scope(scope_id))
            continue
        for entry in registry:
            affected = _is_scope_pre_fix(scope_sha, entry.commit_id)
            if affected is None:  # indeterminate ancestry -> semver fallback
                affected = _is_scope_affected_by_semver(
                    scope_semver, entry.affected_version_range
                )
            if affected is None:  # neither ancestry nor semver could decide
                info.append(_classify_unstamped_scope(scope_id))
                continue
            if affected:
                actionable.append(
                    RegistryMatch(
                        commit_id=entry.commit_id,
                        severity=entry.severity,
                        recommended_action=entry.recommended_action,
                        affected_scope=scope_id,
                        summary=(
                            f"Scope {scope_id!r} predates invalidating fix "
                            f"{entry.commit_id} ({entry.severity})."
                        ),
                    )
                )
    return actionable, info
