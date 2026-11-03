"""V0004: backfill CF-1.13 attributes on analysis_datatree.zarr.

Phase 2 of `datatree_result_organization` (consolidated zarr) initially shipped
without CF attributes. The `consolidated outputs carry cf conventions`
stipulation requires every variable to carry standard_name (when matched),
long_name, units, and cell_methods, and the tree root to carry
``Conventions: 'CF-1.13'`` and ``analysis_id``. This migration backfills
these from the canonical source ``cf_conventions.py``.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 3
version_to: int = 4
description: str = "Backfill CF-1.13 attributes on analysis_datatree.zarr per cf_conventions stipulation"


def upgrade(ctx: MigrationContext) -> None:
    store = ctx.target_dir / "analysis_datatree.zarr"
    if not store.exists():
        return  # nothing to stamp (e.g., system_directory pass; or v0 partial migration)

    # Set root-level CF attributes
    analysis_id = ctx.target_dir.name
    ctx.zarr_set_convention(store, conventions="CF-1.13", analysis_id=analysis_id)

    # Per-variable attrs: delegate to the canonical substrate function
    # `cf_conventions.apply_cf_attributes(ds, mode)`. This function is the
    # authority named by the `consolidated outputs carry cf conventions`
    # stipulation; reimplementing it here would silently drift from the
    # canonical CF contract (auto long_name fallback, _CF_VARIABLE_MAP
    # updates, mode-specific overrides). See cf_conventions.py:184.

    from hhemt import cf_conventions as cf
    from hhemt.utils import append_zarr, deferred_consolidation, open_store

    # The three appends run in one deferred block: it removes the store's index first, so an
    # interrupted backfill leaves an unindexed store (read unindexed, every group visible) rather
    # than an index that is stale against the attrs already rewritten, and rebuilds the index once
    # at the end so consolidated=True readers (project default per `datatree primary consolidation
    # format` stipulation) see the updated attrs.
    with deferred_consolidation(store):
        for mode in ("triton", "tritonswmm", "swmm"):
            group_path = f"/{mode}/summary"
            try:
                ds = open_store(store, kind="dataset", group=group_path, chunks={})
            except (KeyError, FileNotFoundError, OSError):
                continue  # mode not present in this analysis
            ds = cf.apply_cf_attributes(ds, mode=mode)
            # mode="a" preserves existing encoding when no encoding= kwarg is passed.
            append_zarr(ds, store, group=group_path, consolidate=False)
