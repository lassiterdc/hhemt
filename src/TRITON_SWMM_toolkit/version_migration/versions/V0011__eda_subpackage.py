"""V0011: add the lazily-created {analysis_dir}/eda/ output subdirectory (ADR-9).

The ADR-9 EDA data-prep layer writes derived artifacts + verdicts under a NEW
``{analysis_dir}/eda/`` directory, created LAZILY on first write by
``eda.check_cross_sim_identity`` (and the downstream ``analysis.eda()`` facade) — it
does NOT transform, rename, or relocate any existing on-disk tree. This migration is
therefore a no-op against the persisted state: the ``eda/`` directory is created at
first write, never eagerly here (eager mkdir would litter empty dirs on trees that
never run an EDA check and is a git-empty-dir footgun in fixtures).

The 10->11 ``_version.json`` bump and ``migration_history`` append are performed by
the RUNNER unconditionally after ``upgrade(ctx)`` (runner.py:179-182,
state.py:133-167); this migration's body intentionally does nothing.

Layout-relevant change in this bump: a new output directory ``{analysis_dir}/eda/``
(content artifacts: ``<plot_id>.zarr`` + ``<plot_id>.manifest.json`` +
``<plot_id>.verdict.json``). No ``.py`` module is added to
``_layout_relevant_files.yaml`` (that file lists code modules, not output dirs; this
``versions/V0011__*.py`` is covered by the existing ``versions/*.py`` glob).
"""

from __future__ import annotations

from TRITON_SWMM_toolkit.version_migration.context import MigrationContext

version_from: int = 10
version_to: int = 11
description: str = "Add the lazily-created {analysis_dir}/eda/ EDA data-prep output directory (ADR-9)"


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state — the eda/ dir is created lazily at first write.

    The runner performs the 10->11 version stamp + migration_history append after
    this returns; an empty upgrade plans zero ops and the stamp still lands.
    """
    return
