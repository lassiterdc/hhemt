"""V0023: retire the stranded regular-arm store on a tree V0021 left carrying both names.

SCOPE IS ROUTE 2 ONLY, and the scope statement is the whole of this module's safety.
`V0021.upgrade` selects its source with an `elif`, so on a pre-migration root carrying
BOTH retired names it migrates ONE and leaves the other untouched. That stranded
`analysis_datatree.zarr` has no writer, no reader and no role: it is the pre-migration
store of a directory whose CURRENT root store is the unified one.

THIS MIGRATION MUST NOT TOUCH THE OTHER ROUTE. A regular analysis migrated by V0021 and
then re-consolidated carries the SAME pair of names, and there the retired-name store is
the producer's LIVE artifact while the unified one is a stale derived view. Retiring the
retired name there would move the working store aside and leave every root-group consumer
opening a member-shaped tree. The two states are identical in their name set and in their
shape pair, so the discrimination below is not hygiene -- it is the precondition for
acting at all.

THE DISCRIMINATOR IS A CONJUNCTION AND NO MEMBER OF IT IS SUFFICIENT.
  S1  a `members/` container at the analysis root -- the master's own construction
      residue, OUTSIDE every store, one `is_dir()`.
  S2  the MEMBER-group count inside the unified store: exactly one means V0021's regular
      reshape, more than one means a sensitivity master's own tree.
  S3  `toggle_sensitivity_analysis` as the INVOCATION'S analysis config declares it.
A single-signal test on S2 alone REFUSES on the regular arm by construction, because
V0021's regular reshape uses the fixed ordinal `member_0` and therefore always yields
exactly one member group -- a test keyed on a value that DEFINES a population restates
that population rather than discriminating it. S1 alone is wrong on a master whose own
store was never consolidated. S1 and S2 fail TOGETHER on ARM REVERSION -- a directory run
as a master and then re-run as a regular analysis, where `members/` persists because
nothing deletes it -- and that is the one cell a two-member conjunction cannot see,
because a conjunction's whole value is that disagreement announces error. S3 is the only
available signal that is right there, which is why the third member is not redundancy.

S2 COUNTS MEMBER GROUPS BY VOCABULARY, NOT EVERY CHILD, AND THE DIFFERENCE IS THE WHOLE
SIGNAL. `MigrationContext._apply_zarr_unify_to_experiment_tree` demotes the store's
children under the member node and THEN writes a `parameters` group unconditionally, so a
count of every child carrying a group marker returns TWO on a store with ONE member.
Measured on `tests/fixtures/legacy_layouts/v22/experiment_datatree.zarr`: `member_0` and
`parameters` both present. A naive child count is therefore CONSTANT-TRUE for
`count > 1` on every store that primitive produced, which makes the regular-arm branch
below unreachable AND -- the worse direction -- makes a ONE-MEMBER SENSITIVITY EXPERIMENT,
which is an enumerated REFUSAL state, read as a multi-member master and MOVED. Count only
children whose name is in a member vocabulary. An exclusion list would have to be extended
on every new sibling group and fails silently toward acting; a positive membership test
fails toward DISAGREE, and therefore toward refusal, which is the safe direction for a
branch that moves a store.

S2 ALSO REQUIRES A GROUP MARKER, OF EITHER ZARR LAYOUT, AND THE `OF EITHER` IS NOT
DEFENSIVE TIDYING. The marker test is inherited from the primitive being measured:
`_apply_zarr_unify_to_experiment_tree` builds its own child list with
`p.is_dir() and (p / "zarr.json").is_file()`, so a counter that admits a bare directory is
not counting what that primitive produced. But the UNIFIED store is not always that
primitive's output -- on the sensitivity arm V0021 RENAMES the master's own tree rather
than reshaping it, and `V0022._producer_generation` records that population in terms:
"Both zarr layouts are read because a store predating the unification may be v2." A
v3-only marker test therefore counts ZERO on a legitimate multi-member v2 master, refusing
with a message naming three states none of which is "this store is zarr v2". Admit both
markers.

THE COUNT DIVERGES FROM THE ARM CLASSIFIER'S ON EXACTLY ONE SHAPE, AND THE DIVERGENCE IS
CORRECT. The runtime arm classifier counts member NODES by name vocabulary alone, with no
group-marker test, and `V0022._member_group_prefixes` does the same. That is right for
their question -- which vocabulary APPEARS, a presence test that should tolerate noise --
and wrong for this one, which is a CARDINALITY that authorises a MOVE. On a store carrying
member-named directories that are not group nodes at all, the classifier counts them and
this module counts zero. This module fails toward DISAGREE and therefore toward refusal;
the classifier fails toward reporting an arm, which costs nothing. Do not reconcile these
by making either answer the other's question.

S3 TAKES NO ON-ROOT FALLBACK, AND THE REASON IS THE POINT RATHER THAN A DETAIL. A live
analysis root may carry `cfg_analysis.yaml`, but only `eda()` and `publish_analysis()`
ever rewrite it, so on ARM REVERSION -- the one transition S3 exists to detect -- that
file still asserts the PRIOR arm. Reading it would make the conjunction unanimous and
wrong where absence would have left it two-membered and announced: AN ABSENT SIGNAL
LEAVES A CONJUNCTION TWO-MEMBERED AND ANNOUNCED; A STALE SIGNAL MAKES IT THREE-MEMBERED
AND SILENT. Do not add the fallback. It reads as an obvious improvement because it
strictly increases the populations on which S3 is "available", and availability is the
wrong property to grade a source on.

THE MEMBER VOCABULARY IS DEFINED LOCALLY AND DELIBERATELY NOT IMPORTED. `V0022` defines
the same two prefixes locally, so a local definition here keeps the migration series
internally consistent; and a migration's verdict must be a function of the tree and of its
own frozen body, never of a runtime constant that can widen after this module ships and
retroactively change what it counts on a tree it has not yet migrated. Importing would
also make this module's importability depend on another module's contents, and
`registry.discover_migrations` imports EVERY file in `versions/` -- so a missing symbol
would break the whole migration CLI rather than this one migration. The duplication is a
KNOWN RESIDUAL: if a third prefix is introduced and only one site is updated, this module
counts zero and refuses. The closer is a test asserting the sites agree, not an import.

NOTHING IS DELETED. The stranded store is MOVED ASIDE to
`analysis_datatree.zarr.superseded-v0023`, inheriting V0022's retention design verbatim,
including the MEDIAL `.zarr`: `Path("analysis_datatree.zarr.superseded-v0023").suffix` is
`".superseded-v0023"` and `.match("*.zarr")` is False, so every trailing-`.zarr` glob and
every `suffix == ".zarr"` test in this package misses the retained store structurally
rather than by a grep that happened to return zero.

REFUSAL IS THE ANSWER TO DISAGREEMENT, NEVER A GUESS. Where the available signals do not
agree, the root is one of three enumerated states -- a sensitivity master whose own store
was never consolidated, a ONE-MEMBER sensitivity experiment, or an arm reversion -- and
this module raises `MigrationBlockedError` naming all three and moving nothing. A
migration is the venue where an announced refusal reaches an operator; a resolver called
from a rendering path has nowhere to put one.

AND A SECOND REFUSAL IS SCOPED TO THE ACT BRANCH ALONE. When the signals AGREE on the
master arm but S3 was never supplied, the agreement rests on the two signals that fail
together on arm reversion, so the classification authorising the move is the one case
they cannot see. This module refuses THERE and only there: a root carrying at most one
root store still early-returns, and a root classified REGULAR still no-ops. Retention
makes a wrong move RECOVERABLE; it does not make it DISCOVERABLE, and the refusal is what
supplies the second property. The refusal names the two CLI options that make it pass,
because a gate whose remedy names a state rather than a command leaves the operator to
invent the action.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from hhemt.version_migration.context import MigrationContext
from hhemt.version_migration.exceptions import MigrationBlockedError

logger = logging.getLogger(__name__)

version_from: int = 22
version_to: int = 23
description: str = (
    "Retire the stranded regular-arm analysis_datatree.zarr that V0021's arm selection "
    "left beside the unified store, moving it aside to a medial-suffix retained name; "
    "refuse on any root the arm conjunction cannot classify, and refuse on the acting "
    "branch when the invocation supplied no analysis config"
)

_STRANDED = "analysis_datatree.zarr"
_UNIFIED = "experiment_datatree.zarr"
_SUPERSEDED = "analysis_datatree.zarr.superseded-v0023"
_MEMBERS_DIR = "members"
_TOGGLE_KEY = "toggle_sensitivity_analysis"
_MIGRATION_ID = "V0023__retire_stranded_regular_store"

# The member-group vocabularies, taken from V0022, which defines the same two locally:
# `sa_` is the vocabulary V0019 retired and `member_` the current one. Both are admitted
# because an archived tree can still carry the retired form. DEFINED, NOT IMPORTED -- see
# the module docstring under THE MEMBER VOCABULARY IS DEFINED LOCALLY.
_MEMBER_PREFIXES = ("sa_", "member_")

# Group markers for BOTH zarr layouts. A v3 node carries `zarr.json`; a v2 node carries
# `.zgroup`. The unified store is not always v3 -- on the sensitivity arm V0021 renames a
# tree it did not build -- so a v3-only test undercounts a legitimate master.
_GROUP_MARKERS = ("zarr.json", ".zgroup")


def _has_members_container(target_dir: Path) -> bool:
    """S1. The master's construction residue, outside every store."""
    return (target_dir / _MEMBERS_DIR).is_dir()


def _is_group_node(child: Path) -> bool:
    """True when `child` is a zarr group node under either layout."""
    return any((child / marker).is_file() for marker in _GROUP_MARKERS)


def _member_group_count(store: Path) -> int:
    """S2. MEMBER-group count inside the unified store, by directory listing.

    A member group is a child directory that is a group node under either zarr layout
    AND whose name is in a member vocabulary. Neither test is defensive tidying: the
    demotion primitive writes a `parameters` group unconditionally, so counting every
    child returns 2 on a one-member store; and admitting a bare directory would stop
    measuring what that primitive produced. No zarr library is loaded and no chunk is
    read, so this stays decidable on an archived tree.
    """
    if not store.is_dir():
        return 0
    return sum(
        1
        for child in store.iterdir()
        if child.is_dir() and _is_group_node(child) and child.name.startswith(_MEMBER_PREFIXES)
    )


def _declared_arm_is_sensitivity(ctx: MigrationContext) -> bool | None:
    """S3, or None when the INVOCATION cannot answer.

    Reads the invocation-supplied analysis config and NOTHING else -- see the module
    docstring on why an on-root `cfg_analysis.yaml` is not a sanctioned fallback. Every
    unavailable case returns None: no `cfg_paths`, no `analysis` key, an unreadable or
    unparseable document, a document that is not a mapping, or a mapping without the
    toggle. None is not False: False would assert the regular arm on evidence that does
    not exist.
    """
    cfg_paths = getattr(ctx, "cfg_paths", None)
    if not cfg_paths or "analysis" not in cfg_paths:
        return None
    try:
        doc = yaml.safe_load(Path(cfg_paths["analysis"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or _TOGGLE_KEY not in doc:
        return None
    return bool(doc[_TOGGLE_KEY])


def upgrade(ctx: MigrationContext) -> None:
    target_dir = Path(ctx.target_dir)
    stranded = target_dir / _STRANDED
    unified = target_dir / _UNIFIED
    superseded = target_dir / _SUPERSEDED

    if not (stranded.is_dir() and unified.is_dir()):
        logger.info("[V0023] %s does not carry both root store names; nothing to retire", target_dir)
        ctx.record_applied(_MIGRATION_ID)
        return

    if superseded.exists():
        raise MigrationBlockedError(
            f"V0023: {superseded} already exists, so a previous run retained a store and this "
            f"tree carries THREE candidate copies. Refusing to overwrite the retained one. "
            f"Inspect all three, keep the store you trust at {unified}, and remove the other "
            f"two by hand before re-running."
        )

    s1_master = _has_members_container(target_dir)
    s2_master = _member_group_count(unified) > 1
    s3_master = _declared_arm_is_sensitivity(ctx)

    if s3_master is None:
        logger.warning(
            "[V0023] at %s the invocation supplied no analysis config, so the arm "
            "conjunction is running on TWO signals rather than three. The pair is blind to "
            "arm reversion, where both on-disk signals agree and are both wrong. This is a "
            "DEGRADED classification; it is recorded rather than silent, and it BLOCKS the "
            "acting branch below.",
            target_dir,
        )

    votes = [s1_master, s2_master] if s3_master is None else [s1_master, s2_master, s3_master]
    if any(vote != votes[0] for vote in votes):
        raise MigrationBlockedError(
            f"V0023: at {target_dir} the arm signals DISAGREE. members/ container present="
            f"{s1_master}; unified-store MEMBER groups more than one={s2_master}; config "
            f"{_TOGGLE_KEY}={s3_master}. This root is one of three enumerated states and this "
            f"migration will not guess between them: a sensitivity master whose own store was "
            f"never consolidated, a ONE-MEMBER sensitivity experiment, or an ARM REVERSION (a "
            f"directory run as a sensitivity master and then re-run as a regular analysis). "
            f"If the store at {unified} carries member-named directories that are NOT zarr "
            f"group nodes under either layout, this count reads zero by design and that is a "
            f"fourth possibility worth checking before the three above. NOTHING HAS BEEN "
            f"MOVED and nothing has been deleted. Decide which store is current, move the "
            f"other aside by hand, and re-run; or re-run from the tree's real layout version "
            f"(`python -m hhemt.version_migration baseline {target_dir} 20`)."
        )

    if not votes[0]:
        logger.info(
            "[V0023] %s classifies as the REGULAR arm on every available signal, so the store "
            "at %s is the producer's live artifact and %s is a stale derived view of it. This "
            "migration does not act on that state: retiring the live store would leave every "
            "root-group consumer opening a member-shaped tree. Nothing moved.",
            target_dir,
            _STRANDED,
            _UNIFIED,
        )
        ctx.record_applied(_MIGRATION_ID)
        return

    if s3_master is None:
        raise MigrationBlockedError(
            f"V0023: at {target_dir} the two ON-DISK arm signals agree that this root is a "
            f"SENSITIVITY MASTER, which is the branch that MOVES a store -- but the third "
            f"signal is unavailable, because this invocation supplied no analysis config. "
            f"Those two signals fail TOGETHER on ARM REVERSION (a directory run as a "
            f"sensitivity master and then re-run as a regular analysis), so the agreement "
            f"authorising the move is exactly the case they cannot see, and moving here would "
            f"relocate the regular producer's LIVE store. NOTHING HAS BEEN MOVED. Supply the "
            f"analysis config this tree is currently governed by and re-run:\n"
            f"    python -m hhemt.version_migration migrate {target_dir} --apply "
            f"--system-config PATH_TO_CFG_SYSTEM_YAML "
            f"--analysis-config PATH_TO_CFG_ANALYSIS_YAML\n"
            f"BOTH options are required together -- the migration reads a config only when "
            f"both are passed. Do NOT point --analysis-config at a cfg_analysis.yaml sitting "
            f"inside {target_dir}: only `eda()` and `publish_analysis()` rewrite that file, so "
            f"on a reverted tree it still asserts the arm this check exists to doubt."
        )

    ctx.move_dir(stranded, superseded, merge_policy="error")
    logger.warning(
        "[V0023] retired the stranded %s at %s to %s. The retained store is NOT reclaimed "
        "automatically. Verify %s reads as expected, then remove %s.",
        _STRANDED,
        target_dir,
        superseded,
        unified,
        superseded,
    )
    ctx.record_applied(_MIGRATION_ID)
