"""V0016: embed the deterministic JSON-LD provenance core into the consolidated DataTree root attrs (ADR-5/6/7).

The reproducibility-system metadata-rocrate-core writes a deterministic, mtime-decoupled
JSON-LD provenance string into ``analysis_datatree.zarr`` (and the sensitivity master tree)
root ``tree.attrs["ro_crate_metadata"]`` at consolidation time. This is an ADDITIVE on-disk
surface on NEWLY-consolidated trees — it does NOT transform, rename, or relocate any existing
on-disk tree, and the core cannot be reconstructed for a historical tree without re-running
consolidation (it derives from the toolkit git sha + input provenance available only at
consolidation). This migration is therefore a no-op against persisted state: existing trees
gain the core on their next re-consolidation, never eagerly here. The 15->16 ``_version.json``
bump + ``migration_history`` append are performed by the RUNNER unconditionally after
``upgrade(ctx)`` (the V0011/V0012/V0014/V0015 precedent); this body does nothing.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 15
version_to: int = 16
description: str = (
    "Embed the deterministic JSON-LD provenance core into the consolidated DataTree root attrs (ADR-5/6/7)"
)


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state — prov is written lazily at the next consolidation."""
    return
