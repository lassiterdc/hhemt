"""V0003: consolidate per-mode flat zarrs into hierarchical analysis_datatree.zarr.

Phase 2 of `datatree_result_organization` moved from per-mode flat zarr stores
(``triton_summary.zarr``, ``tritonswmm_summary.zarr``, ``swmm_summary.zarr``)
to a single hierarchical ``analysis_datatree.zarr`` with /triton/, /tritonswmm/,
/swmm/ groups. This migration consolidates without deleting the source stores
(non-destructive default; a future ``--cleanup`` flag would remove them after
verification).
"""
from __future__ import annotations

from TRITON_SWMM_toolkit.version_migration.context import MigrationContext

version_from: int = 2
version_to: int = 3
description: str = (
    "Consolidate flat per-mode zarrs into hierarchical analysis_datatree.zarr"
)

_FLAT_STORE_NAMES = {
    "triton_summary": "triton_summary.zarr",
    "tritonswmm_summary": "tritonswmm_summary.zarr",
    "swmm_summary": "swmm_summary.zarr",
}

_TREE_SPEC = {
    "/triton/summary": "triton_summary",
    "/tritonswmm/summary": "tritonswmm_summary",
    "/swmm/summary": "swmm_summary",
}


def upgrade(ctx: MigrationContext) -> None:
    input_stores = {
        key: ctx.target_dir / fname
        for key, fname in _FLAT_STORE_NAMES.items()
        if (ctx.target_dir / fname).exists()
    }
    if not input_stores:
        return  # nothing to consolidate (e.g., system_directory pass)
    output_store = ctx.target_dir / "analysis_datatree.zarr"
    if output_store.exists():
        return  # idempotent
    tree_spec = {path: key for path, key in _TREE_SPEC.items() if key in input_stores}
    ctx.zarr_flat_to_datatree(
        input_stores=input_stores,
        output_store=output_store,
        tree_spec=tree_spec,
    )
