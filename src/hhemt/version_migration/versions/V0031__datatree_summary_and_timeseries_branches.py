"""V0031: every group of ``analysis_datatree.zarr`` holds a ``summary`` and a ``timeseries`` branch.

Layouts up to 30 stored each consolidated summary at ``{family}/{kind}`` (for example ``tritonswmm/swmm_node``) of
``analysis_datatree.zarr`` and the consolidated SWMM series in a SEPARATE store, ``analysis_timeseries.zarr``. From
layout 31 the tree is the one store: ``{family}/{kind}/summary`` holds the summary statistics and
``{family}/{kind}/timeseries`` holds the series, and ``analysis_timeseries.zarr`` no longer exists.

WHAT IT TOUCHES. Every ``analysis_datatree.zarr`` under the tree (the analysis itself, each ``members/member_*``
analysis of a sensitivity master, each ``_test/group_*`` subset analysis), the ``analysis_timeseries.zarr`` beside it
when it exists, and the root tree of a sensitivity master (``experiment_datatree.zarr``, or its retired name
``sensitivity_datatree.zarr``), under each of its ``member_*`` nodes. Nothing under ``sims/`` is read or written; in
particular this migration deletes NO per-scenario summary store.

THE SUMMARY MOVE. A group cannot be renamed to a path inside itself (measured: ``os.rename`` of a directory to its own
subdirectory raises ``OSError``), so moving ``{family}/{kind}`` to ``{family}/{kind}/summary`` is three renames and one
group-document write, each step atomic and each state recognisable on a re-run: (1) rename ``{kind}`` to the sibling
``{kind}__v30_moving``; (2) create ``{kind}`` as a plain group, read as done only when ``{kind}/zarr.json`` exists and
never from the directory alone, because a kill between creating the directory and writing its group document leaves a
directory that is not yet a group; (3) rename ``{kind}__v30_moving`` to ``{kind}/summary``. State by what exists: only
``{kind}`` holding arrays: start at (1); ``{kind}__v30_moving`` present: resume at (2) or (3), whichever is not yet
done, never reading the moving directory as data; ``{kind}/summary`` present and no moving directory: done, skipped. No
chunk data is rewritten, and a kill never leaves one group's arrays split across the old and the new path.

THE SERIES FOLD. Each group of ``analysis_timeseries.zarr`` (``tritonswmm/swmm_node_timeseries``,
``tritonswmm/swmm_link_timeseries``) is copied into ``analysis_datatree.zarr`` at ``tritonswmm/swmm_node/timeseries``
and ``tritonswmm/swmm_link/timeseries``, read back equal (NaN-aware, every variable and coordinate), and only then is
``analysis_timeseries.zarr`` removed. The record comes BEFORE the delete: the target's root attribute
``migrated_from_analysis_timeseries: done`` is written after the read-back and before the removal, so a re-run that
finds it removes what is left of a half-deleted source without re-verifying it (a partly deleted source cannot be
compared against). A target WITHOUT the attribute re-runs the copy and the read-back, and refuses a source whose own
root index lists a node whose metadata document is gone: zarr reads a missing chunk as fill values, so a copy of such a
source would pass its own read-back, and chunk presence cannot decide it because zarr writes no chunk equal to the fill
value. When ``analysis_datatree.zarr`` is absent (a reprocess delete removed it) the copy creates the tree holding only
the ``timeseries`` branches; it carries no consolidation stamp, so the next consolidation rewrites every summary group.

THE VERSION-1 CALENDAR GROUPS. An analysis consolidated at version 1 holds ``tritonswmm/swmm_node_timeseries`` and
``tritonswmm/swmm_link_timeseries`` INSIDE ``analysis_datatree.zarr`` on a calendar ``date_time`` axis (the residual
V0030 named). Each is DELETED when a version-2 copy of the same series exists at ``{kind}/timeseries``, and otherwise
kept at ``{kind}/timeseries_legacy_calendar`` with a root attribute saying why. The keep is one directory rename, run
only AFTER that kind's summary move has made ``tritonswmm/{kind}`` a plain group, and skipped on a re-run when the
source is gone and the target present.

THE INDEX. The whole restructure of a tree runs inside ``utils.deferred_consolidation``, which removes the store's
consolidated index on entry and rebuilds it once on a normal exit, so from the first move until the rebuild no reader
can trust a stale index (measured: after step (3) with the pre-move index kept, ``xr.open_datatree`` with the default
consolidated read lists no ``{kind}/summary``, while ``consolidated=False`` lists it). The rebuild runs on EVERY run
that reaches the end, including a re-run in which every move, copy and removal is skipped as already done, and only
then is the layout stamp written by the runner, so a killed migration is recognisable as unfinished by its stamp and
re-runs.

THE SENSITIVITY SERIES FOLD. A sensitivity master's ``sensitivity_timeseries.zarr`` holds ``member_{id}/...`` series
groups, either at the layout-30 name ``{family}/{kind}_timeseries`` or at ``{family}/{kind}/timeseries``. Each is copied
into the master's tree (``experiment_datatree.zarr``, or its retired name ``sensitivity_datatree.zarr``) at
``member_{id}/{family}/{kind}/timeseries``, read back equal, and only then is ``sensitivity_timeseries.zarr`` removed,
by the same record-before-delete order as above with the root attribute ``migrated_from_sensitivity_timeseries: done``.
The fold runs AFTER the master's summary move so a ``{kind}`` is already a plain group when its ``timeseries`` child is
written. It is queued as the op ``sensitivity_series_fold``, whose handler this module attaches to the context: the
shared context primitive carries the analysis fold only.

WHAT IT DOES NOT DO. It does not touch ``system_datatree.zarr`` (built only on request, rewritten at its next build).
``CONSOLIDATION_VERSION`` 3 makes every tree consolidated at version 2 rewrite its summary branches once, and its series
branches not at all.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from hhemt.version_migration.context import (
    MigrationContext,
    PlannedOp,
    _array_dirs,
    _is_group_dir,
    _read_group_equal,
    _source_manifest_complete,
    _write_plain_group,
)

version_from: int = 30
version_to: int = 31
description: str = "Move every datatree summary group under a summary branch and fold analysis_timeseries.zarr into it"

#: Directory names the walk never enters. ``sims`` holds per-scenario stores only; a tree under ``_test/`` IS entered.
_SKIP_DIR_PREFIXES = ("out_",)
_SKIP_DIR_NAMES = frozenset({".snakemake", ".git", "sims"})
_SKIP_DIR_SUFFIXES = (".chapters",)

_ANALYSIS_TREE = "analysis_datatree.zarr"
_SERIES_STORE = "analysis_timeseries.zarr"
_MASTER_TREES = ("experiment_datatree.zarr", "sensitivity_datatree.zarr")
_SENSITIVITY_SERIES_STORE = "sensitivity_timeseries.zarr"
#: Root attribute of the master tree recording that the fold of ``sensitivity_timeseries.zarr`` was read back equal
#: (written BEFORE the source is deleted, as ``migrated_from_analysis_timeseries`` is for an analysis).
SENSITIVITY_SERIES_MIGRATED_ATTR = "migrated_from_sensitivity_timeseries"


def _is_store_root(path: Path) -> bool:
    return (path / "zarr.json").is_file() or (path / ".zgroup").is_file()


def _walk(root: Path):
    """Yield ``(directory, names)`` for each directory holding one of the stores this migration touches.

    ``names`` is the sorted subset of the four store names present in it. Never descends into a ``.zarr`` store.
    """
    wanted = {_ANALYSIS_TREE, _SERIES_STORE, _SENSITIVITY_SERIES_STORE, *_MASTER_TREES}
    for current, dirs, _files in os.walk(root):
        keep = []
        present = set()
        for name in sorted(dirs):
            if name in _SKIP_DIR_NAMES or name.startswith(_SKIP_DIR_PREFIXES) or name.endswith(_SKIP_DIR_SUFFIXES):
                continue
            if name.endswith(".zarr"):
                if name in wanted:
                    present.add(name)
                continue
            keep.append(name)
        dirs[:] = keep
        if present:
            yield Path(current), sorted(present)


def _member_series_groups(source: Path) -> list[tuple[str, str]]:
    """Each series group of ``sensitivity_timeseries.zarr`` as ``(source group, master group)``, sorted.

    A group counts when it sits under a ``member_*`` (or ``sa_*``) node, holds arrays directly, and is named
    ``{kind}_timeseries`` (layout 30) or ``timeseries`` under ``{kind}`` (the layout of the 1d.1 build). The target
    is always ``member_{id}/{family}/{kind}/timeseries``.
    """
    found: list[tuple[str, str]] = []
    for member in sorted(source.iterdir()):
        if not (member.is_dir() and member.name.startswith(("member_", "sa_")) and _is_group_dir(member)):
            continue
        for current, dirs, _files in os.walk(member):
            here = Path(current)
            arrays = {a.name for a in _array_dirs(here)} if _is_group_dir(here) else set()
            dirs[:] = sorted(d for d in dirs if d not in arrays)
            if not arrays:
                continue
            rel = here.relative_to(source).as_posix()
            if rel.endswith("/timeseries"):
                found.append((rel, rel))
            elif rel.endswith("_timeseries"):
                found.append((rel, rel[: -len("_timeseries")] + "/timeseries"))
    return sorted(found)


def _fold_sensitivity_series(store: str, source: str) -> None:
    """Copy each series group of ``source`` into the master tree ``store``, verify it, record it, drop ``source``.

    Mirrors the analysis fold of the shared context primitive: a target already carrying the done attribute only
    finishes removing the source (a partly deleted source cannot be compared); otherwise the source must be complete
    against its own index, every group is copied (or kept when already equal), read back equal, and only then is the
    attribute written and the source removed. A read-back that fails raises before the attribute is written and the
    source is kept.
    """
    from hhemt.utils import append_zarr, deferred_consolidation, open_store, update_zarr_root_attrs, zarr_root_attrs

    master, src = Path(store), Path(source)
    if master.is_dir() and zarr_root_attrs(master).get(SENSITIVITY_SERIES_MIGRATED_ATTR) == "done":
        if src.exists():
            # EXEMPT-DU: migration-primitive
            shutil.rmtree(src)
        return
    if not src.exists():
        return
    complete, why = _source_manifest_complete(src)
    if not complete:
        raise ValueError(
            f"{why}; it is not used as a copy source, because zarr reads a missing chunk as fill values and a copy "
            "of it would pass its own read-back. Restore the store or re-run the consolidation that wrote it."
        )
    if not _is_group_dir(master):
        _write_plain_group(master)
    with deferred_consolidation(master):
        for src_group, dst_group in _member_series_groups(src):
            wanted = open_store(src, kind="zarr", group=src_group, chunks=None, decode_timedelta=False).load()
            existing = None
            if _is_group_dir(master / dst_group):
                existing = open_store(master, kind="zarr", group=dst_group, chunks=None, decode_timedelta=False).load()
            if existing is None or not _read_group_equal(wanted, existing):
                append_zarr(wanted, master, group=dst_group, consolidate=False, mode="w")
            stored = open_store(master, kind="zarr", group=dst_group, chunks=None, decode_timedelta=False).load()
            if not _read_group_equal(wanted, stored):
                raise ValueError(
                    f"the copy of {src_group} in {master.name} at {dst_group} does not read back equal to its "
                    f"source; {src.name} was kept"
                )
        update_zarr_root_attrs(master, {SENSITIVITY_SERIES_MIGRATED_ATTR: "done"})
    # EXEMPT-DU: migration-primitive
    shutil.rmtree(src)


def upgrade(ctx: MigrationContext) -> None:
    """Queue one restructure per consolidated tree under ``ctx.target_dir``, and the sensitivity series fold."""
    for directory, names in _walk(Path(ctx.target_dir)):
        master_present = None
        for master_name in _MASTER_TREES:
            if master_name in names and _is_store_root(directory / master_name):
                ctx.datatree_branch_layout(directory / master_name, master=True)
                master_present = master_present or master_name
        if _SENSITIVITY_SERIES_STORE in names:
            # The op's handler lives in this module (the shared primitive folds an analysis series store only), and
            # it runs after the summary move queued above because the plan executes in order.
            ctx._apply_sensitivity_series_fold = _fold_sensitivity_series  # type: ignore[attr-defined]
            ctx.plan.append(
                PlannedOp(
                    "sensitivity_series_fold",  # type: ignore[arg-type]
                    {
                        "store": str(directory / (master_present or _MASTER_TREES[0])),
                        "source": str(directory / _SENSITIVITY_SERIES_STORE),
                    },
                )
            )
        if _ANALYSIS_TREE in names or _SERIES_STORE in names:
            series = directory / _SERIES_STORE
            ctx.datatree_branch_layout(
                directory / _ANALYSIS_TREE, series_source=series if series.exists() else None, master=False
            )
