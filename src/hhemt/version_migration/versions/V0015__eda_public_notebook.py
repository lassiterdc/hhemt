"""V0015: add the lazily-created {root}/-resident EDA notebook + eda_local/ surfaces (ADR-12/13/14).

analysis.eda() / Bundle.eda() now emit a seeded eda.ipynb notebook and a bundle-adjacent
eda_local/ package under {root}/, created LAZILY on first eda() call — neither transforms,
renames, nor relocates any existing on-disk tree. This migration is therefore a no-op against
persisted state. The 14->15 _version.json bump + migration_history append are performed by the
RUNNER unconditionally after upgrade(ctx) (the V0012/V0014 precedent); this body does nothing.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 14
version_to: int = 15
description: str = (
    "Add the lazily-created {root}/-resident EDA notebook + eda_local/ source-independent surfaces (ADR-12/13/14)"
)


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state — the notebook + eda_local/ are created lazily at first eda()."""
    return
