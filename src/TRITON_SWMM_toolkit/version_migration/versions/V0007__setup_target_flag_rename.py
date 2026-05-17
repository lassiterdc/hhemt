"""V0007: rewrite _status/a_setup_complete.flag to _status/a_setup_target_0_complete.flag.

The `prefixed_column_config_variation` feature changed the sensitivity-master
Snakefile's setup-rule output filename from a singleton `a_setup_complete.flag`
to per-`UniqueSystemTarget` `a_setup_target_{N}_complete.flag` files. The
master target (compile-relevant tuple = master cfg_system's tuple) is always
assigned `target_id=0`, so the legacy state's `a_setup_complete.flag` is
semantically equivalent to the new `a_setup_target_0_complete.flag`.

This migration applies only to sensitivity-analysis trees (it requires a
`subanalyses/` sibling to fire). Multisim trees are not affected because the
multisim Snakefile (workflow.py:971) still emits the singleton
`a_setup_complete.flag` unchanged.

New targets (`target_id` 1..N introduced by per-sub-analysis system.* overlay
columns) have NO legacy state and intentionally have NO migrated flag — the
setup_target_1, setup_target_2, ... rules must actually fire to compile their
target-specific build artifacts. This migration synthesizes a flag only for
target_0.

mtime preservation: Path.rename preserves the mtime by POSIX semantics
(rename is a metadata-only operation on the same filesystem). No os.utime
call is required.
"""
from __future__ import annotations

from TRITON_SWMM_toolkit.version_migration.context import MigrationContext

version_from: int = 6
version_to: int = 7
description: str = (
    "Rename _status/a_setup_complete.flag to _status/a_setup_target_0_complete.flag "
    "(sensitivity-only; multisim trees no-op)"
)


def upgrade(ctx: MigrationContext) -> None:
    subanalyses = ctx.target_dir / "subanalyses"
    if not subanalyses.is_dir():
        return  # multisim or system_directory pass — no rename
    legacy = ctx.target_dir / "_status" / "a_setup_complete.flag"
    new = ctx.target_dir / "_status" / "a_setup_target_0_complete.flag"
    if new.exists():
        return  # already migrated
    if not legacy.exists():
        return  # nothing to migrate (sensitivity tree that never completed setup)
    ctx.flag_rewrite_paths(
        analysis_dir=ctx.target_dir,
        old_regex=r"^a_setup_complete\.flag$",
        new_template="a_setup_target_0_complete.flag",
    )
