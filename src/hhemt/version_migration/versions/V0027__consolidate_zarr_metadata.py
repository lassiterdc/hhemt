"""V0027: give every zarr store under an analysis tree a current consolidated-metadata index.

From layout 27 the toolkit writes every zarr store with an index and opens every store
through it (`utils.open_store`), so a store written before that carries none, or carries
one that no longer matches what is on disk, and would be read wrongly or flagged by the
staleness check (`analysis_validation.check_consolidated_metadata_current`). This migration
rebuilds the index of each store that needs it and changes no data value.

WHAT IT TOUCHES. Every directory under the tree whose name ends in ``.zarr`` and whose root
holds a format 3 ``zarr.json`` or a format 2 ``.zgroup``. A format 2 store (the
``slurm_jobs.zarr`` of the efficiency report) takes its index in the root ``.zmetadata``, which
is how `utils.zarr_store_has_index` already reads it, so the two formats are handled alike.
The walk does not descend into a store, and it skips the raw solver output directories
(``out_*``), the Snakemake state (``.snakemake``) and the transient chapter sets
(``*.chapters``): none holds a store a reader opens by index, and walking raw output costs
millions of directory entries.

THE STALE-INDEX ASSUMPTION, stated because a migration CAN meet it. A legacy store may carry an
index that is STALE, for example a group attribute changed outside the toolkit's write helpers
after the index was built; a read through such an index serves the old value with no error
(measured by the chunk 3a.2 reviewer: `_apply_zarr_rename_variable` kept the stale attributes).
This migration therefore NEVER reads a store through its index. Whether a store needs work is
decided by `check_consolidated_metadata_current`, which compares the index with the metadata
documents on disk; the rebuild reads those same documents. What it cannot repair is a store whose
documents on disk are themselves wrong. Migrations queued BEFORE this one (V0004, V0021 and the
rest) read legacy stores through `open_store`, which trusts an index when one is present. That
is an assumption this module does not verify and cannot enforce from here: it holds only for a
legacy store whose index was not left stale by a write outside the toolkit's helpers (the one
in-toolkit route, a kill inside the unify step, is repaired in the same package).

IDEMPOTENT. A store whose index already equals its documents is skipped, so a second run, or a
run over a tree whose stores are current, plans nothing. The rebuild is queued through the
existing ``zarr_set_attrs`` primitive with an empty attribute merge, whose apply rewrites the
root attributes unchanged and then calls `utils.consolidate_zarr_store`; no write of its own.
"""

from __future__ import annotations

import os
from pathlib import Path

from hhemt.version_migration.context import MigrationContext

version_from: int = 26
version_to: int = 27
description: str = "Rebuild the consolidated-metadata index of every zarr store under the tree"

#: Directory names the walk never enters (see the module docstring).
_SKIP_DIR_PREFIXES = ("out_",)
_SKIP_DIR_NAMES = frozenset({".snakemake", ".git"})
_SKIP_DIR_SUFFIXES = (".chapters",)


def _is_store_root(path: Path) -> bool:
    """A zarr store root: a format 3 ``zarr.json`` or a format 2 ``.zgroup`` directly inside."""
    return (path / "zarr.json").is_file() or (path / ".zgroup").is_file()


def _store_roots(root: Path) -> list[Path]:
    """Every ``*.zarr`` store root under ``root``, sorted; never descends into a store."""
    found: list[Path] = []
    for current, dirs, _files in os.walk(root):
        keep = []
        for name in sorted(dirs):
            if name in _SKIP_DIR_NAMES or name.startswith(_SKIP_DIR_PREFIXES) or name.endswith(_SKIP_DIR_SUFFIXES):
                continue
            path = Path(current) / name
            if name.endswith(".zarr") and _is_store_root(path):
                found.append(path)
                continue
            keep.append(name)
        dirs[:] = keep
    return found


def upgrade(ctx: MigrationContext) -> None:
    """Queue one index rebuild for each store under ``ctx.target_dir`` whose index is absent or stale."""
    # Imported here: the check pulls in xarray, which a plain migration listing need not load.
    from hhemt.analysis_validation import check_consolidated_metadata_current

    for store in _store_roots(Path(ctx.target_dir)):
        if check_consolidated_metadata_current(store, index_required=True).passed:
            continue
        ctx.zarr_set_attrs(store, "", {}, merge=True)
