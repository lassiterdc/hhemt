"""V0002: rewrite _status/scenario_{iloc}-{slug}.flag to scenario_{slug}.flag.

The pre-Phase-0 Snakefile keyed scenario flag files on `{event_iloc}-{slug}`.
Phase 0 of `datatree_result_organization` dropped the `{iloc}-` prefix from
the on-disk scenario directory layout (V0001), but legacy `_status/` flag
files keyed on the old naming were not rewritten at the same time. Without
this migration, post-V0001 analyses still carry stale flag files that no
longer correspond to the renamed scenario directories — Snakemake cannot
reliably resume from those flags.

The current Snakefile (see `workflow.py`) emits flags under a different
`{phase}_{...}_evt-{event_id}_complete.flag` scheme; this migration only
touches *legacy* `scenario_*` flags and is a no-op on trees written by the
current Snakefile.
"""
from __future__ import annotations

from TRITON_SWMM_toolkit.version_migration.context import MigrationContext

version_from: int = 1
version_to: int = 2
description: str = "Rewrite _status/scenario_{iloc}-{slug}.flag to scenario_{slug}.flag"


def upgrade(ctx: MigrationContext) -> None:
    ctx.flag_rewrite_paths(
        analysis_dir=ctx.target_dir,
        old_regex=r"^scenario_\d+-(?P<slug>.+)\.flag$",
        new_template="scenario_{slug}.flag",
    )
