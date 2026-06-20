"""V0014: add the lazily-created {analysis_dir}/static_plots/ output subdirectory (ADR-8).

static_plots() writes publication figures under a NEW {analysis_dir}/static_plots/
directory, created LAZILY on first render by the static-plot rules — it does NOT
transform, rename, or relocate any existing on-disk tree. This migration is therefore
a no-op against persisted state: static_plots/ is created at first write, never eagerly
here. The 13->14 _version.json bump + migration_history append are performed by the
RUNNER unconditionally after upgrade(ctx) (the V0012 precedent); this body does nothing.
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 13
version_to: int = 14
description: str = "Add the lazily-created {analysis_dir}/static_plots/ publication-figure output directory (ADR-8)"


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state — static_plots/ is created lazily at first render."""
    return
