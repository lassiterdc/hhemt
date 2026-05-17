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

Snakemake metadata invalidation: every per-sa_id `prepare_*` rule's
`.snakemake/metadata/` record lists `_status/a_setup_complete.flag` as one of
its inputs. After this rename, the new Snakefile-build declares
`_status/a_setup_target_0_complete.flag` instead. Snakemake's
`--rerun-triggers input` does set-comparison against the metadata record;
the set-change fires reruns of every affected rule, cascading through the
DAG. This migration therefore clears `.snakemake/metadata/` (with a backup
at `.snakemake/metadata.bak.V0007`) so Snakemake falls back to make-style
"outputs exist and are newer than inputs" semantics. The cleared metadata
is regenerated naturally as rules run; the backup is retained for audit.

Order of operations: clear metadata FIRST, then rename the flag. The reverse
order would briefly leave the tree in a half-migrated state where the
filesystem reflects the rename but Snakemake's metadata still references the
old name.
"""
from __future__ import annotations

from TRITON_SWMM_toolkit.version_migration.context import MigrationContext

version_from: int = 6
version_to: int = 7
description: str = (
    "Rename _status/a_setup_complete.flag to _status/a_setup_target_0_complete.flag "
    "and clear .snakemake/metadata/ to invalidate set-change rerun triggers "
    "(sensitivity-only; multisim trees no-op)"
)


def upgrade(ctx: MigrationContext) -> None:
    subanalyses = ctx.target_dir / "subanalyses"
    if not subanalyses.is_dir():
        return  # multisim or system_directory pass — no rename
    legacy = ctx.target_dir / "_status" / "a_setup_complete.flag"
    new = ctx.target_dir / "_status" / "a_setup_target_0_complete.flag"
    # Always clear Snakemake metadata for sensitivity trees: even when the
    # filesystem rename is already done (re-run), the prior metadata records
    # may still reference the old input name. The primitive's existing-backup
    # short-circuit handles repeat invocations without double-backing-up.
    ctx.clear_snakemake_metadata(backup_label="V0007")
    if new.exists():
        return  # rename already done
    if not legacy.exists():
        return  # nothing to rename (sensitivity tree that never completed setup)
    ctx.flag_rewrite_paths(
        analysis_dir=ctx.target_dir,
        old_regex=r"^a_setup_complete\.flag$",
        new_template="a_setup_target_0_complete.flag",
    )
