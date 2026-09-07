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
