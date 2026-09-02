"""V0021: unify the consolidated store into one experiment tree per experiment.

SCOPE, and it is narrower than a reader expects. This migration makes an ARCHIVED
tree READABLE under the unified model. It does NOT reconstruct the per-member
`analysis_datatree.zarr`, which is a LIVE-RUN artifact: the next consolidation
writes it, and would overwrite anything written here. The in-store demotion below
CONSUMES the store, so a migration cannot both demote in place and leave a member
store behind without a full data copy of the largest artifact in the system --
which is exactly the cost the metadata-only demotion exists to avoid.

TWO ARMS, and only one reshapes. A sensitivity master's `sensitivity_datatree.zarr`
is ALREADY the experiment shape (root `parameters` + one node per member, built by
`build_sensitivity_datatree`), so that arm is a pure rename. Only a regular
analysis's `analysis_datatree.zarr` is demoted. The arm is selected by which root
store is present -- an existence test, which is the one place that discrimination is
legitimate, because a migration is the only code that sees both pre-states and it
runs once per tree.

OP ORDER IS A CORRECTNESS CONSTRAINT. Provenance rewrites are PLANNED while the
pre-rename paths still exist and are APPLIED before the rename. `MigrationContext`
plans are read eagerly and executed in append order, so a provenance step ordered
after the rename is handed a path the rename already consumed. Measured, its failure
is a silent NO-OP -- not V0020's resurrection of a retired filename, which leaves an
extra file a walk can see, but nothing at all: the planner finds no source, plans
nothing, and the store keeps a stale `@id` and `schemaVersion` inside it. Do not
reorder the calls in `upgrade`.

PROVENANCE IS SURGICAL, NEVER RE-EMITTED. `emit_provenance` resolves its
`code_repository` through `_default_code_repository()`, which RAISES when installed
package metadata exposes no homepage. A migration calling it would succeed or fail
as a function of the ENVIRONMENT rather than of the tree it is migrating. It would
also freeze today's emitter output into every v21 tree, and would propagate the
graph's existing non-conformance forward-only. Rewrite the two VALUES instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hhemt.version_migration.constants import LAYOUT_VERSION
from hhemt.version_migration.context import MigrationContext

logger = logging.getLogger(__name__)

version_from: int = 20
version_to: int = 21
description: str = (
    "Unify the consolidated store into one experiment_datatree.zarr per experiment: "
    "rename the sensitivity master, demote a regular analysis's root groups under a "
    "member node and mint a root parameters node, and re-point the RO-Crate sidecar "
    "and its embedded twin (S8b stage-4)"
)

_OLD_REGULAR = "analysis_datatree.zarr"
_OLD_SENSITIVITY = "sensitivity_datatree.zarr"
_NEW_EXPERIMENT = "experiment_datatree.zarr"
_SIDECAR = "ro-crate-metadata.json"


def _same_store(candidate: object, relpath: str) -> bool:
    """True when `candidate` names `relpath`, ignoring a trailing slash.

    `ro-crate-py` normalizes a Dataset `@id` to end with `/` (`Dataset.format_id`:
    `identifier.rstrip("/") + "/"`), and the consolidated store is routed through
    `crate.add_dataset(dest_path=consolidated_zarr_relpath, ...)`. So a REAL emitted
    crate carries `analysis_datatree.zarr/` and a raw `==` against the unslashed
    relpath matches NOTHING. Measured: the unslashed comparison left the retired `@id`
    in place in the sidecar AND the embedded core, in the entity AND its `hasPart`
    reference, while `_repoint` still returned True from the `schemaVersion` clause --
    a silent failure reported as a success.
    """
    return isinstance(candidate, str) and candidate.rstrip("/") == relpath.rstrip("/")


def _resolve_member_group(store: Path) -> str:
    """Member node name for a one-member experiment.

    FIXED BY RULING D67, and the reason is a defect rather than a preference.

    An earlier revision derived this from the store's root `analysis_id` attr, on the
    reasoning that it was the non-positional identity the combine layer already keys
    on. That observation is CORRECT and its verdict was WRONG.
    `V0004__cf_conventions_backfill` derives that attr from `ctx.target_dir.name`, so
    the value is a function of WHERE THE TREE WAS when V0004 ran, not of what the tree
    contains. Measured across the golden ladder: identical content entering at v0..v3
    versus v4..v20 yields five distinct values, so an identity-derived node name gives
    one tree five different SHAPES and reddens four ladder pairs.

    The invariant a migration owes is CONFLUENCE: it may know its path, but its path
    may not determine the shape it produces. A fixed ordinal satisfies that
    unconditionally, and it is also the only rule that works on BOTH arms -- a regular
    analysis has no `members/` container, so a container-derived name raises on exactly
    the shape this migration exists to serve.

    IDENTITY IS NOT LOST, it moves off the path: the `parameters` node carries
    `member_id` and `analysis_id`, and the member crate root carries `analysis_id` and
    `name`. What this function returns is an ORDINAL -- the zeroth member of a
    one-member experiment -- and it makes no claim about which analysis this is.

    THE UNDERLYING DEFECT IS DEFERRED, NOT FIXED. `analysis_id` is still
    path-dependent at `V0004__cf_conventions_backfill:27`, in this tree's root attrs and
    in `parameters`. That is a separate change against a LANDED migration; it is pinned
    by `test_v0004_analysis_id_is_still_path_determined` in the golden suite so it cannot
    be forgotten, and excluded by name from the ladder's content projection so it cannot
    be mistaken for new drift.

    LANDING CONSTRAINT: this module matches the layout-relevant glob
    `src/hhemt/version_migration/versions/*.py` and carries no `non_breaking_allowlist`
    entry, and `.pre-commit-config.yaml` invokes `check-b HEAD~1`, whose skip window is
    the bump commit and the one after it. This edit must therefore land in the SAME
    commit as the LAYOUT_VERSION 20->21 bump, or at latest the commit immediately after.
    """
    return "member_0"


def _repoint(doc: dict, old_relpath: str) -> bool:
    """Re-point every reference to the renamed store and re-stamp schemaVersion.

    Returns True when the document changed. Touches three shapes because the crate
    names the store in three places: the Dataset entity's own `@id`, the root
    entity's `schemaVersion`, and any `hasPart` reference to the store.

    The return value is NOT a verification signal and must never be asserted on: it
    is True whenever ANY clause fired, so a `schemaVersion` re-stamp alone sets it
    even when every `@id` went unrepaired. Assert on the emitted `@id`.
    """
    new_dir_id = _NEW_EXPERIMENT + "/"  # the slashed form a fresh emit produces
    changed = False
    for entity in doc.get("@graph", []):
        if _same_store(entity.get("@id"), old_relpath):
            entity["@id"] = new_dir_id
            changed = True
        if entity.get("@id") == "./" and entity.get("schemaVersion") != str(LAYOUT_VERSION):
            entity["schemaVersion"] = str(LAYOUT_VERSION)
            changed = True
        parts = entity.get("hasPart")
        if isinstance(parts, list):
            for ref in parts:
                if isinstance(ref, dict) and _same_store(ref.get("@id"), old_relpath):
                    ref["@id"] = new_dir_id
                    changed = True
    return changed


def _plan_sidecar(ctx: MigrationContext, target_dir: Path, old_relpath: str) -> None:
    sidecar = target_dir / _SIDECAR
    if not sidecar.is_file():
        return
    try:
        doc = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("[V0021] sidecar unreadable, leaving as-is: %s", sidecar)
        return
    if _repoint(doc, old_relpath):
        ctx.rewrite_text_preserving_mtime(sidecar, json.dumps(doc, indent=2))


def _plan_embedded_core(ctx: MigrationContext, store: Path, old_relpath: str) -> None:
    """The provenance is DUAL-WRITTEN: `_EMBEDDED_PROV_KEYS` admits `@id`, `hasPart`
    and `schemaVersion`, so patching only the sidecar leaves a stale core INSIDE the
    renamed store, where no sidecar-shaped repair reaches it."""
    meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    core = (meta.get("attributes") or {}).get("ro_crate_metadata")
    if not core:
        return
    doc = json.loads(core)
    if _repoint(doc, old_relpath):
        ctx.zarr_set_attrs(
            store,
            "",
            {"ro_crate_metadata": json.dumps(doc, separators=(",", ":"), sort_keys=True)},
            merge=True,
        )


def upgrade(ctx: MigrationContext) -> None:
    target_dir = Path(ctx.target_dir)
    sensitivity = target_dir / _OLD_SENSITIVITY
    regular = target_dir / _OLD_REGULAR

    if sensitivity.is_dir():
        old_relpath, store = _OLD_SENSITIVITY, sensitivity
    elif regular.is_dir():
        old_relpath, store = _OLD_REGULAR, regular
    else:
        logger.info("[V0021] no root consolidated store at %s; nothing to unify", target_dir)
        ctx.record_applied("V0021__experiment_tree_unification")
        return

    # 1. PROVENANCE FIRST, planned while the pre-rename paths still exist. Reordering
    #    this below the rename does not fail -- it silently plans nothing.
    _plan_sidecar(ctx, target_dir, old_relpath)
    _plan_embedded_core(ctx, store, old_relpath)

    # 2. Reshape, but ONLY the regular arm. The sensitivity master is already the
    #    experiment shape and a demotion there would wrap the members a second time.
    if old_relpath == _OLD_REGULAR:
        member_group = _resolve_member_group(store)
        meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
        analysis_id = (meta.get("attributes") or {}).get("analysis_id", "")
        ctx.zarr_unify_to_experiment_tree(
            store,
            member_group,
            {"member_id": [member_group.removeprefix("member_")], "analysis_id": [analysis_id]},
            "member_id",
        )

    # 3. THEN the rename.
    ctx.move_dir(store, target_dir / _NEW_EXPERIMENT)

    ctx.record_applied("V0021__experiment_tree_unification")
