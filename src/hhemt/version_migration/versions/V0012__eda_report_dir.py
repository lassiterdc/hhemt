"""V0012: add the lazily-created {analysis_dir}/eda_report/ output subdirectory (ADR-10).

The ADR-10 EDA loop writes the standalone updateable eda_report.html under a NEW
{analysis_dir}/eda_report/ directory, created LAZILY on first write by
eda.assemble_eda_report (driven by analysis.eda() / Bundle.eda()) — it does NOT
transform, rename, or relocate any existing on-disk tree. This migration is therefore
a no-op against persisted state: eda_report/ is created at first write, never eagerly
here.

The 11->12 _version.json bump and migration_history append are performed by the RUNNER
unconditionally after upgrade(ctx) (the V0011 precedent); this migration's body
intentionally does nothing.

Layout-relevant change in this bump: a new output directory {analysis_dir}/eda_report/
(content: eda_report.html). The plots/eda/ plot emission is NOT a layout change (the
in-process plotter derives its own output_path; no report_plot_ids.py edit).
"""

from __future__ import annotations

from hhemt.version_migration.context import MigrationContext

version_from: int = 11
version_to: int = 12
description: str = "Add the lazily-created {analysis_dir}/eda_report/ EDA-doc output directory (ADR-10)"


def upgrade(ctx: MigrationContext) -> None:
    """No-op against on-disk state — the eda_report/ dir is created lazily at first write."""
    return
