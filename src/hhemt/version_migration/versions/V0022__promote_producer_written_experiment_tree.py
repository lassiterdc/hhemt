"""V0022: promote the producer-written store when a v21 tree carries BOTH names.

THE THIRD STATE, and why a NAME is not enough to identify it. `V0021` renamed the
consolidated store to `experiment_datatree.zarr` and never touched the PRODUCER, so a
tree that ran V0021 and then re-consolidated carries the migrated store under the
UNIFIED name and a producer-written store under the RETIRED name, both stamped 21.

WHICH ONE IS NEWER IS A THEOREM, NOT A GUESS. `V0021.upgrade` ends in
`ctx.move_dir(store, target_dir / _NEW_EXPERIMENT)`, which CONSUMES the retired path.
So any store sitting at the retired path on a v21 tree was written AFTER V0021 ran.
Newness is established by construction; no mtime read is involved, and none would be
reliable on a directory-shaped store.

NOTHING IS DELETED, AND THAT IS THE WHOLE OF THE DESIGN. Newer is not the same as
SOUND, and this tree has already ruled that soundness is undecidable here:
`utils._publish_store_crash_safe`'s docstring records that "no sound completeness
detector is possible", because zarr omits an all-fill chunk by default, so a
legitimately dry corner of a flood-depth field is byte-identical to a killed write.
Measured directly against a chunk-stripped store: it opens without error, reports the
full expected shape, and returns all-NaN. A completeness gate is therefore not
available at any price, and an earlier draft of this module that removed the incumbent
under `guarded_remove(..., force=True)` was refuted by running it -- that primitive
verifies only that the replacement EXISTS and is a NON-EMPTY DIRECTORY, and a
chunk-stripped store satisfies both, so the sound store was deleted and the empty one
promoted.

The response to an undecidable predicate is not a better detector; it is not needing
one. The incumbent is MOVED ASIDE to `experiment_datatree.zarr.superseded-v0022`, never
removed, so every outcome of this migration is reversible by one rename. The cost is
2.00x peak on the consolidated store until an operator reclaims it -- exactly the cost
`_publish_store_crash_safe` already pays on its own rewrite path, and it names it.

THE RETAINED NAME IS SAFE BY CONSTRUCTION, NOT BY A GREP RETURNING ZERO. Its `.zarr` is
MEDIAL: `Path("experiment_datatree.zarr.superseded-v0022").suffix` is
`".superseded-v0022"` and `.match("*.zarr")` is False, so BOTH families of name-based
reader miss it structurally -- every glob in this package anchors `.zarr` at the end
(`*_summary.zarr`, `chapter_*.zarr`, `*/sensitivity_datatree.zarr`), and
`_figure_emission`'s `p.suffix == ".zarr"` test cannot fire either, nor can its
`.zattrs`/`.zgroup`/`.zarray` fallback, because these stores are zarr v3 and carry only
`zarr.json`. An earlier draft justified this from a three-pattern grep returning 0; that
instrument could not have seen
`report_renderers/cross_experiment_intercomparison_maps.py`'s
`glob("*/sensitivity_datatree.zarr")`, which a wider census does find and which is also
harmless. The structural argument is the load-bearing one.

WHY THE RESIDUAL IS NARROW BUT NOT EMPTY. `write_datatree_zarr` publishes through
`_publish_store_crash_safe`, whose guarantee is that the final path is "either ABSENT or
a COMPLETE store, never an INCOMPLETE one" -- verified here by crashing inside the write
callable, after which the final path did not exist and only a `.tmp` sibling remained.
So a store written at the retired path by a CURRENT toolkit is complete. It is not
provable that every toolkit which could produce a third state had that protection:
measured, the publisher (c34dd6e1) is NOT an ancestor of V0021 (5c64558b), the two
landed 39 minutes apart on divergent lines, and at V0021's own commit
`write_datatree_zarr` did not call the publisher at all. A build inside that window has
V0021 and no publisher. Retention is what makes that unresolvable window survivable.

SCOPE IS THE SENSITIVITY ARM ONLY. A regular analysis's root `analysis_datatree.zarr`
is the FLAT-rooted producer shape, while `experiment_datatree.zarr` is the demoted
member shape; promoting one onto the other would replace an experiment tree with a flat
one. The regular producer also still resolves the retired name after the S8b rename
(`analysis.py` binds `analysis_datatree_zarr` unconditionally), so no stale-return
window opens there and there is nothing to repair.

OP ORDER IS A CORRECTNESS CONSTRAINT, inherited verbatim from V0021: provenance is
PLANNED while the pre-move paths still exist and is APPLIED before the moves. Plans are
executed in append order by `MigrationContext.execute`, so a provenance step ordered
after a move is handed a path the move already consumed, and its failure is a silent
no-op rather than an error.

SLASH TOLERANCE IS LOAD-BEARING. `ro-crate-py` normalizes a Dataset `@id` to a trailing
slash, so a raw `==` against the unslashed relpath matches nothing. V0021's docstring
records that this exact error produced "a silent failure reported as a success"; the
comparison here is the same `rstrip("/")` form.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from hhemt.version_migration.context import MigrationContext
from hhemt.version_migration.exceptions import MigrationBlockedError

logger = logging.getLogger(__name__)

version_from: int = 21
version_to: int = 22
description: str = (
    "Promote the producer-written sensitivity_datatree.zarr onto experiment_datatree.zarr "
    "on a v21 tree that carries both, retaining the superseded store beside it and "
    "re-pointing the RO-Crate sidecar and its embedded twin onto the promoted store at "
    "schemaVersion 22 (S8b stage-5)"
)

_RETIRED_SENSITIVITY = "sensitivity_datatree.zarr"
_UNIFIED_EXPERIMENT = "experiment_datatree.zarr"
_SUPERSEDED = "experiment_datatree.zarr.superseded-v0022"
_SIDECAR = "ro-crate-metadata.json"
_SCHEMA_VERSION = str(version_to)  # the LITERAL terminus, never LAYOUT_VERSION


def _same_store(candidate: object, relpath: str) -> bool:
    """True when `candidate` names `relpath`, ignoring a trailing slash."""
    return isinstance(candidate, str) and candidate.rstrip("/") == relpath.rstrip("/")


def _repoint(doc: dict, old_relpath: str) -> bool:
    """Re-point every reference to the promoted store and re-stamp schemaVersion.

    The return value is NOT a verification signal: assert on the emitted `@id`.

    THE schemaVersion CLAUSE IS NOT DECORATION. On a third-state tree V0021 has
    already run and will never run again, so nothing else can advance the crate's
    declared layout. Measured against a body without this clause: `_version.json`
    read 22 while the sidecar still declared its pre-migration value. The crate is
    DEPOSIT METADATA -- it is published -- so a stale value there is a wrong public
    claim about the tree, and the golden ladder cannot see it: `_walk_relative`
    compares path sets, `_content_projection` is guarded on the zarr store, and no
    committed legacy_layouts fixture carries a `ro-crate-metadata.json` at all.
    """
    new_dir_id = _UNIFIED_EXPERIMENT + "/"
    changed = False
    for entity in doc.get("@graph", []):
        if _same_store(entity.get("@id"), old_relpath):
            entity["@id"] = new_dir_id
            changed = True
        if entity.get("@id") == "./" and entity.get("schemaVersion") != _SCHEMA_VERSION:
            entity["schemaVersion"] = _SCHEMA_VERSION
            changed = True
        parts = entity.get("hasPart")
        if isinstance(parts, list):
            for ref in parts:
                if isinstance(ref, dict) and _same_store(ref.get("@id"), old_relpath):
                    ref["@id"] = new_dir_id
                    changed = True
    return changed


def _plan_sidecar(ctx: MigrationContext, target_dir: Path) -> None:
    sidecar = target_dir / _SIDECAR
    if not sidecar.is_file():
        return
    try:
        doc = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("[V0022] sidecar unreadable, leaving as-is: %s", sidecar)
        return
    if _repoint(doc, _RETIRED_SENSITIVITY):
        ctx.rewrite_text_preserving_mtime(sidecar, json.dumps(doc, indent=2))


def _plan_embedded_core(ctx: MigrationContext, store: Path) -> None:
    """The provenance is DUAL-WRITTEN, so patching only the sidecar leaves a stale
    core INSIDE the promoted store where no sidecar-shaped repair reaches it."""
    zarr_json = store / "zarr.json"
    if not zarr_json.is_file():
        return
    meta = json.loads(zarr_json.read_text(encoding="utf-8"))
    core = (meta.get("attributes") or {}).get("ro_crate_metadata")
    if not core:
        return
    doc = json.loads(core)
    if _repoint(doc, _RETIRED_SENSITIVITY):
        ctx.zarr_set_attrs(
            store,
            "",
            {"ro_crate_metadata": json.dumps(doc, separators=(",", ":"), sort_keys=True)},
            merge=True,
        )


#: Root-attribute key carrying the producing build's `git describe` string, stamped by
#: `cf_conventions.apply_producing_stamp` at consolidation (ADR-15).
_PRODUCER_VERSION_ATTR = "hhemt_producing_version"

#: `{tag}+{N}.g{sha}`. The TAG is captured and is load-bearing: `_describe_version` resolves
#: against the NEAREST REACHABLE TAG, and this repo carries three, so N is a distance from a
#: moving origin rather than a global ordinal. Measured on this repository: two first-parent
#: mainline commits 34 minutes apart carry `pre-public-cut+92` then `0.1.0+89`. `fullmatch`
#: is deliberate -- `_describe_version`'s fallbacks ("0.1.0", "0+unknown") carry no count and
#: must read as NO EVIDENCE rather than as zero.
_DESCRIBE_RE = re.compile(r"(?P<tag>[^+]*)\+(?P<n>\d+)\.g(?P<sha>[0-9a-f]+)")

#: The V0019 vocabulary rename. A group named `sa_*` is the RETIRED form; `member_*` is
#: current. This pair is the only DECIDABLE discriminator available here.
_RETIRED_MEMBER_PREFIX = "sa_"
_CURRENT_MEMBER_PREFIX = "member_"


def _producer_generation(store: Path) -> tuple[str, int] | None:
    """Return `(tag, commit_count)` for `store`'s producing build, or None when unavailable.

    THE COUNT IS NOT AN ORDERING AND THIS FUNCTION DOES NOT PRETEND IT IS. A `git describe`
    count measures distance from the nearest reachable tag along an ancestry walk, so it is
    incommensurable across tag bases and non-monotone across divergent branches on one base.
    Measured on this repository at `8f47c8fb`: 82 of 542 `v0.1.0`-based commits sit in a
    chronological adjacency where the LATER commit carries the LOWER count. The tag is
    therefore returned alongside the count so the caller can refuse to compare across bases,
    and even within one base the caller treats the result as CORROBORATION, never as grounds
    to refuse -- see `_vocabulary_inverted`, which is the decidable test.

    ABSENCE IS NOT ZERO. Every unavailable case returns None -- no attribute, an unparseable
    value, an unreadable or absent metadata file. `apply_producing_stamp` additionally leaves
    this key ABSENT and writes `hhemt_producing_version_divergent` when a store's events came
    from different builds, so a divergent store degrades through the same path with no
    special-casing.

    EQUAL COUNTS ARE TIED, NOT ABSENT, AND THE CALLER TREATS THEM THE SAME DELIBERATELY.
    On one tag base a linear history gives equal counts only for the same commit, so the
    vocabulary-generation gap this gate exists to catch cannot be present. Where the shas
    differ at equal count the two commits are on divergent lines at equal distance and the
    count carries no ordering information at all. Both land on positional, which retention
    makes reversible.

    Both zarr layouts are read because a store predating the unification may be v2.
    """
    for name in ("zarr.json", ".zattrs"):
        meta_path = store / name
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        attrs = (meta.get("attributes") or {}) if name == "zarr.json" else meta
        match = _DESCRIBE_RE.fullmatch(str(attrs.get(_PRODUCER_VERSION_ATTR) or "").strip())
        if match:
            return match.group("tag"), int(match.group("n"))
    return None


def _member_group_prefixes(store: Path) -> set[str]:
    """Which member-naming vocabularies appear as top-level groups in `store`."""
    found: set[str] = set()
    if not store.is_dir():
        return found
    for child in store.iterdir():
        if not child.is_dir():
            continue
        for prefix in (_RETIRED_MEMBER_PREFIX, _CURRENT_MEMBER_PREFIX):
            if child.name.startswith(prefix):
                found.add(prefix)
    return found


def _vocabulary_inverted(producer_written: Path, migrated: Path) -> bool:
    """True when promoting `producer_written` would REGRESS the member vocabulary.

    THIS IS THE DECIDABLE TEST AND IT IS WHY THE COUNT IS NOT THE GATE. The harm this
    migration must not cause is content-shaped and nameable: an `sa_*`-keyed store replacing
    a `member_*`-keyed one, which is a V0019 vocabulary regression. That property is a TOTAL
    FUNCTION OF THE TWO STORES -- it needs no git, no tag, no describe string and no ancestry,
    and it is decidable on a bundle or an archived tree where none of those is available.

    It is strictly NARROWER than a recency test: it detects the vocabulary inversion and
    nothing else, and it will not catch a future content regression that keeps the naming.
    That is the trade, taken deliberately -- a narrow test that is sound today beats a general
    one that is measurably wrong 15% of the time and whose failure mode instructs an operator
    to delete the newer store by hand.
    """
    return _RETIRED_MEMBER_PREFIX in _member_group_prefixes(
        producer_written
    ) and _CURRENT_MEMBER_PREFIX in _member_group_prefixes(migrated)


def upgrade(ctx: MigrationContext) -> None:
    target_dir = Path(ctx.target_dir)
    producer_written = target_dir / _RETIRED_SENSITIVITY
    migrated = target_dir / _UNIFIED_EXPERIMENT
    superseded = target_dir / _SUPERSEDED

    if not producer_written.is_dir():
        logger.info("[V0022] no retired-name sensitivity store at %s; nothing to promote", target_dir)
        ctx.record_applied("V0022__promote_producer_written_experiment_tree")
        return

    if not migrated.is_dir():
        raise MigrationBlockedError(
            f"V0022: {target_dir} carries {_RETIRED_SENSITIVITY} but no {_UNIFIED_EXPERIMENT}, "
            f"so V0021's rename did not run against this tree and this is not the third state. "
            f"Re-run the migration from layout 20 "
            f"(`python -m hhemt.version_migration baseline {target_dir} 20`), or, if this tree "
            f"genuinely predates V0021, migrate it from its real layout version instead of 21."
        )

    # This check sits AFTER the no-op early-return, deliberately: a tree V0022 has
    # already migrated has no retired-name store, returns above, and never reaches here,
    # so re-application stays a clean no-op with the retained store still in place.
    if superseded.exists():
        raise MigrationBlockedError(
            f"V0022: {superseded} already exists, so a previous run retained a superseded store "
            f"and this tree carries THREE candidate copies. Refusing to overwrite the retained "
            f"one. Inspect all three, keep the store you trust at {migrated}, and remove the "
            f"other two by hand before re-running."
        )

    # 0. DIRECTION. Ahead of step 1 because that is where the direction is decided, not
    #    because a later position would be unsafe: MigrationContext only appends to ctx.plan,
    #    and runner.py skips both the plan print and ctx.execute() when upgrade() raises, so
    #    nothing reaches disk from any position in this body.
    if _vocabulary_inverted(producer_written, migrated):
        raise MigrationBlockedError(
            f"V0022: at {target_dir} the store at the RETIRED name carries "
            f"'{_RETIRED_MEMBER_PREFIX}'-prefixed member groups while the store at the UNIFIED "
            f"name carries '{_CURRENT_MEMBER_PREFIX}'-prefixed ones, so promoting the retired "
            f"store would REGRESS the member vocabulary that V0019 renamed. This is not the "
            f"third state this migration promotes. NOTHING HAS BEEN MOVED and nothing has been "
            f"deleted. Inspect both stores; if the unified-name store is the one you want, "
            f"remove the retired-name store or re-run from the tree's real layout version "
            f"(`python -m hhemt.version_migration baseline {target_dir} 20`)."
        )

    # Corroboration only, and deliberately NOT a refusal trigger. The commit count is not an
    # ordering (see _producer_generation), so a disagreement here is logged for a human and
    # never halts: a false refusal would send an operator to delete a store BY HAND, outside
    # the retention guarantee this module's design rests on.
    _retired_gen = _producer_generation(producer_written)
    _unified_gen = _producer_generation(migrated)
    if (
        _retired_gen is not None
        and _unified_gen is not None
        and _retired_gen[0] == _unified_gen[0]
        and _retired_gen[1] < _unified_gen[1]
    ):
        logger.warning(
            "[V0022] at %s the retired-name store reports producing generation %s+%d and the "
            "unified-name store %s+%d, so the retired store MAY be the older of the two. The "
            "commit count is not a reliable ordering and is not being acted on; the promotion "
            "proceeds. Verify %s after this migration.",
            target_dir,
            _retired_gen[0],
            _retired_gen[1],
            _unified_gen[0],
            _unified_gen[1],
            migrated,
        )

    # 1. PROVENANCE FIRST, planned while the pre-move paths still exist.
    _plan_sidecar(ctx, target_dir)
    _plan_embedded_core(ctx, producer_written)

    # 2. RETAIN the incumbent. Never `guarded_remove`: its verification passes on a
    #    chunk-stripped store (measured), and no completeness test can do better.
    ctx.move_dir(migrated, superseded, merge_policy="error")

    # 3. THEN the promotion. merge_policy stays "error": step 2 has vacated the
    #    destination, so a surviving destination here means step 2 did not run.
    ctx.move_dir(producer_written, migrated, merge_policy="error")

    logger.warning(
        "[V0022] promoted %s -> %s at %s; the superseded store is RETAINED at %s and is "
        "not reclaimed automatically. Verify %s reads as expected, then remove %s.",
        _RETIRED_SENSITIVITY,
        _UNIFIED_EXPERIMENT,
        target_dir,
        superseded,
        migrated,
        superseded,
    )
    ctx.record_applied("V0022__promote_producer_written_experiment_tree")
