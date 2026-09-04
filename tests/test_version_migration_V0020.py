"""V0020 behavioural tests — the flag/sidecar rename, its content half, and dry-run safety.

The `rename_flag` fixture seeds a `layout_version: 19` tree carrying the retired
`f_consolidate_master_complete` flag class at BOTH scopes the migration walks: the analysis
root and `members/member_0`. Its sidecar's `rule_name` VALUE is the class no filename-matching
primitive reaches — `MigrationContext._apply_flag_rewrite_paths` matches `flag.name` and calls
`rename`, never opening a file — so the content half needs its own assertion.

`test_sidecar_content_follows_the_rename` exists because of a defect the other assertions
could not see. An earlier `upgrade()` planned the renames BEFORE the sidecar rewrite; since
`MigrationContext.execute` runs its plan in append order and
`_apply_rewrite_text_preserving_mtime` create-anew's an absent path, the rewrite resurrected
the retired filename and left the correctly-named file holding the STALE value. Presence and
absence assertions alone pass on a tree whose correctly-named sidecar carries the stale value.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from hhemt.version_migration import runner

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "legacy_layouts" / "v0020_unit_test"

OLD_FLAG = "f_consolidate_master_complete.flag"
NEW_FLAG = "f_consolidate_experiment_complete.flag"
OLD_RULE = "master_consolidation"
NEW_RULE = "experiment_consolidation"


def _copy_variant(name: str, tmp_path: Path) -> Path:
    work = tmp_path / name
    shutil.copytree(FIXTURE_ROOT / name, work)
    return work


def _walk(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def test_flag_and_sidecar_renamed(tmp_path):
    """TWO-ARM differential: the invariant is 'renamed', not 'duplicated' and not 'destroyed'.

    The presence arm alone passes against a migration that COPIED; the absence arm alone
    passes against one that merely DELETED. Only both together match the invariant's
    violation set.
    """
    work = _copy_variant("rename_flag", tmp_path)
    runner.run_migration(work, target=20, apply=True)

    status = work / "_status"
    assert (status / NEW_FLAG).exists()
    assert (status / f"{NEW_FLAG}.json").exists()
    assert not (status / OLD_FLAG).exists()
    assert not (status / f"{OLD_FLAG}.json").exists()


def test_sidecar_content_follows_the_rename(tmp_path):
    """The content half. Asserts the file at the NEW name carries the NEW rule_name —
    the assertion whose absence let the op-order resurrection read as correct."""
    work = _copy_variant("rename_flag", tmp_path)
    runner.run_migration(work, target=20, apply=True)

    sidecar = work / "_status" / f"{NEW_FLAG}.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["rule_name"] == NEW_RULE


def test_member_scope_renamed(tmp_path):
    """V0019's per-member loop shape: the per-member reprocess runners write into each
    member's own `_status/`, so the flag pass must run there too."""
    work = _copy_variant("rename_flag", tmp_path)
    runner.run_migration(work, target=20, apply=True)

    member_status = work / "members" / "member_0" / "_status"
    assert (member_status / NEW_FLAG).exists()
    assert not (member_status / OLD_FLAG).exists()


def test_dry_run_no_mutation(tmp_path):
    """`run_migration` calls `upgrade()` on BOTH paths and gates only `ctx.execute()`, so a
    write escaping the plan would land here and nowhere else."""
    work = _copy_variant("rename_flag", tmp_path)
    before = _walk(work)

    runner.run_migration(work, target=20, apply=False)

    assert _walk(work) == before


def test_idempotent(tmp_path):
    """A second run plans nothing and leaves the stamp byte-identical."""
    work = _copy_variant("rename_flag", tmp_path)
    runner.run_migration(work, target=20, apply=True)

    stamp_after_first = (work / "_version.json").read_bytes()
    second = runner.run_migration(work, target=20, apply=True)

    assert second.migrations_planned == []
    assert (work / "_version.json").read_bytes() == stamp_after_first
