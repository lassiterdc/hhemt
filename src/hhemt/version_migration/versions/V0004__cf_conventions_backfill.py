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
description: str = (
    "Backfill CF-1.13 attributes on analysis_datatree.zarr per cf_conventions stipulation"
)


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
    import xarray as xr
    import zarr

    from hhemt import cf_conventions as cf

    for mode in ("triton", "tritonswmm", "swmm"):
        group_path = f"/{mode}/summary"
        try:
            ds = xr.open_dataset(
                store,
                engine="zarr",
                group=group_path,
                consolidated=False,
                chunks={},
            )
        except (KeyError, FileNotFoundError, OSError):
            continue  # mode not present in this analysis
        ds = cf.apply_cf_attributes(ds, mode=mode)
        # mode="a" preserves existing encoding when no encoding= kwarg is passed.
        ds.to_zarr(store, group=group_path, mode="a", consolidated=False)

    # Reconsolidate after full backfill so consolidated=True readers
    # (project default per `datatree primary consolidation format` stipulation)
    # see the updated attrs.
    zarr.consolidate_metadata(str(store))
