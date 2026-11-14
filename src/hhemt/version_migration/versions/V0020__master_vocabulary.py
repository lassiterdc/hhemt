"""V0020: rename the on-disk master vocabulary to experiment.

Two on-disk names, and only two. The `master_*` census counts eleven Python identifier
stems; nine have no on-disk footprint (`master_snakefile_path` resolves to the literal
`Snakefile` at `workflow.py::_resolve_snakefile_path`, and `paths.py` carries zero
`master` occurrences, so no directory is named for it).

WHY THE FLAG IS RENAMED AND NOT UNLINKED. V0019 invalidates consolidation by unlinking
the `*consolidate*complete.flag` class, correct THERE because its member-node rename
invalidates the tree's node names. A flag/rule rename changes no tree CONTENT, so
forcing a rebuild of a multi-GB master tree buys nothing. Rename.

WHY THE `.flag.json` SIDECAR NEEDS ITS OWN REGEX. `MigrationContext._apply_flag_rewrite_paths`
iterates `_status` and matches `flag.name`, renaming one FILE per match. It does not
follow a sidecar. Two regexes, not one.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from hhemt.version_migration.context import MigrationContext

logger = logging.getLogger(__name__)

version_from: int = 19
version_to: int = 20
description: str = (
    "Rename the on-disk master vocabulary to experiment: the master consolidation "
    "status flag and its JSON sidecar, and the master_consolidation rule name recorded "
    "as a rule_name VALUE inside every _status/*.flag.json (S8a stage-4)"
)

_OLD_RULE = "master_consolidation"
_NEW_RULE = "experiment_consolidation"

#: (old_regex, new_template). The sidecar is a SEPARATE entry by necessity, not symmetry.
_FLAG_REWRITES: tuple[tuple[str, str], ...] = (
    (r"^f_consolidate_master_complete\.flag$", "f_consolidate_experiment_complete.flag"),
    (r"^f_consolidate_master_complete\.flag\.json$", "f_consolidate_experiment_complete.flag.json"),
)


def _member_dirs(analysis_dir: Path) -> list[Path]:
    """Member roots, mirroring V0019's helper. The per-member reprocess runners write
    into each member's own `_status/`, so the flag pass runs there too."""
    container = Path(analysis_dir) / "members"
    if not container.is_dir():
        return []
    return sorted(p for p in container.iterdir() if p.is_dir())


def _plan_rule_name_rewrites(scope_dir: Path) -> list[tuple[Path, str]]:
    """Read-only scan for `_status/*.flag.json` payloads whose `rule_name` VALUE is the
    retired rule name. A JSON VALUE, not a filename, so no primitive reaches it -- the
    same shape V0019 used for its manifest `plot_id` rewrites."""
    status = scope_dir / "_status"
    if not status.is_dir():
        return []
    out: list[tuple[Path, str]] = []
    for sidecar in sorted(status.glob("*.flag.json")):
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("rule_name") != _OLD_RULE:
            continue
        payload["rule_name"] = _NEW_RULE
        out.append((sidecar, json.dumps(payload, indent=2)))
    return out


def upgrade(ctx: MigrationContext) -> None:
    target_dir = Path(ctx.target_dir)

    # 1. Metadata FIRST (V0007's ordering rationale, which V0019 cites). Snakemake's
    #    metadata is RULE-KEYED, so clearing it is what keeps the rule rename from
    #    firing a spurious NEW_RULE / CODE rerun trigger. Primitive: dry-run-safe.
    ctx.clear_snakemake_metadata("V0020")

    # 2. CONTENT BEFORE FILENAME. This ordering is a CORRECTNESS constraint, not a
    #    preference, and reversing it does not fail -- it SUCCEEDS and resurrects the
    #    retired filename. `MigrationContext.execute` runs `self.plan` in APPEND order
    #    (context.py:128), and `_apply_rewrite_text_preserving_mtime` takes a
    #    CREATE-ANEW branch when its path is absent (context.py:691-693). So a rewrite
    #    planned AFTER the rename holds a pre-rename path, finds nothing at execute
    #    time, and WRITES THE OLD NAME BACK -- measured: 3 files where 2 belong, with
    #    the correctly-named one holding the STALE value and the resurrected one
    #    holding the fresh value. Plan the content rewrite while the path still exists.
    rule_name_rewrites: list[tuple[Path, str]] = []
    for scope_dir in [target_dir, *_member_dirs(target_dir)]:
        rule_name_rewrites.extend(_plan_rule_name_rewrites(scope_dir))
    logger.info("[V0020] planned: %d rule_name sidecar rewrite(s)", len(rule_name_rewrites))
    for sidecar, new_text in rule_name_rewrites:
        ctx.rewrite_text_preserving_mtime(sidecar, new_text)

    # 3. THEN the FILENAME renames, at the analysis root AND every member root.
    for old_regex, new_template in _FLAG_REWRITES:
        ctx.flag_rewrite_paths(target_dir, old_regex, new_template)
        for member in _member_dirs(target_dir):
            ctx.flag_rewrite_paths(member, old_regex, new_template)

    # 4. The one remaining gated write. The reprocess Snakefile is DERIVED and is
    #    overwritten unconditionally on the execution path, so only
    #    render_report(reprocess=True) can read a stale one -- and a stale one names the
    #    pre-rename flag as a rule output, which makes the report engine raise. Unlink
    #    rather than rewrite: a migration that rewrites generated content owes a second
    #    copy of the generator's grammar.
    if not ctx.dry_run:
        (target_dir / "Snakefile.reprocess").unlink(missing_ok=True)  # EXEMPT-DU: migration-primitive

    ctx.record_applied("V0020__master_vocabulary")
