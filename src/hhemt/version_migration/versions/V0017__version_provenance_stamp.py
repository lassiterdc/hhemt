"""V0017: introduce the per-scope version-provenance stamp (ADR-15).

The reproducibility-system version-provenance layer stamps the producing HHEMT
git-sha (+ semver) into every per-scenario summary zarr as per-`event_iloc`
COORDINATES and into the consolidated DataTree root `tree.attrs` as a scalar
uniform fast-path, at WRITE time. This is an ADDITIVE on-disk surface on
NEWLY-produced/consolidated artifacts — it does NOT transform, rename, or
relocate any existing on-disk tree, and the producing sha of a historical
artifact is unrecoverable (it derives from the toolkit git sha available only
at production time). This migration is therefore a no-op against persisted
state: existing artifacts gain the stamp on their next re-production /
re-consolidation, never eagerly here (the V0011/V0016 precedent). The 16->17
`_version.json` bump + `migration_history` append are performed by the RUNNER
unconditionally after `upgrade(ctx)`; this body does nothing.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 16
version_to: int = 17
description: str = "Introduce the per-scope version-provenance stamp (ADR-15)"


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state — the stamp is written lazily at the next production/consolidation."""
    return
