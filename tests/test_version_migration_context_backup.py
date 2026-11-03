"""Tests for ``MigrationContext._apply_clear_snakemake_metadata``: the backup is a rename.

The method moves ``.snakemake/metadata`` to ``.snakemake/metadata.bak.{label}``. Each test builds a
small tree under ``tmp_path`` and uses only the standard library and the context class, so none runs
a solver and all run on a laptop. Run as
``PYTHONPATH=${PWD}/src pytest tests/test_version_migration_context_backup.py``.

Each docstring names its property, class, the wrong implementation that turns it red and a second
correct implementation that stays green. The inode oracle is the discriminator for "renamed, not
copied": a copy gives every file a new inode, a rename keeps them.
"""

from __future__ import annotations

import os
from pathlib import Path

from hhemt.version_migration.context import MigrationContext


def _tree(tmp_path: Path) -> tuple[Path, dict[str, tuple[int, bytes]]]:
    target = tmp_path / "analysis"
    meta = target / ".snakemake" / "metadata"
    files = {"rule_a/rec1": b"one", "rule_a/rec2": b"two", "rule_b/deep/rec3": b"three", "top": b"t"}
    for rel, payload in files.items():
        p = meta / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
    inodes = {rel: (os.stat(meta / rel).st_ino, payload) for rel, payload in files.items()}
    return target, inodes


def _ctx(target: Path) -> MigrationContext:
    return MigrationContext(target_dir=target, dry_run=False, migration_id="V0000__backup_test")


def test_the_records_move_to_the_backup_and_the_metadata_folder_is_gone(tmp_path):
    """Property: after the call the backup holds every file with its bytes and ``metadata/`` is absent.

    Class: REGRESSION guard (the end state of the copy-then-delete form, which the rename must keep;
    green on the tree before the change). Wrong implementation: a rename of the wrong folder or a
    copy that leaves ``metadata/`` behind; red on either assertion. Reachability: a tree with nested
    rule folders. Both-states anchor: bytes are compared with the literals written. Second correct
    implementation: ``shutil.copytree`` then ``shutil.rmtree``.
    """
    target, inodes = _tree(tmp_path)
    _ctx(target)._apply_clear_snakemake_metadata("lbl")
    bak = target / ".snakemake" / "metadata.bak.lbl"
    assert not (target / ".snakemake" / "metadata").exists()
    assert {p.relative_to(bak).as_posix(): p.read_bytes() for p in bak.rglob("*") if p.is_file()} == {
        rel: payload for rel, (_ino, payload) in inodes.items()
    }


def test_the_files_are_renamed_not_copied(tmp_path):
    """Property: every file in the backup has the inode it had in ``metadata/`` before the call.

    Class: NEW CAPABILITY (discriminator: the copy-then-delete form gives each file a new inode, so
    this is red on the tree before the change). Wrong implementation: ``copytree`` plus ``rmtree``;
    red because the inodes differ. Reachability: the inodes are read before the call. Both-states
    anchor: inode numbers are a property of the filesystem, not of any wording. Not established: this
    holds on a filesystem that reuses an inode number at once after a delete, which a tmpfs or ext4
    ``tmp_path`` does not do for files still alive. Second correct implementation:
    ``Path.rename`` instead of ``os.rename``.
    """
    target, inodes = _tree(tmp_path)
    _ctx(target)._apply_clear_snakemake_metadata("lbl")
    bak = target / ".snakemake" / "metadata.bak.lbl"
    assert {rel: os.stat(bak / rel).st_ino for rel in inodes} == {rel: ino for rel, (ino, _p) in inodes.items()}


def test_a_second_call_is_a_no_op_because_the_backup_exists(tmp_path):
    """Property: a second call changes nothing, even when a new ``metadata/`` appears, and keeps the first backup.

    Class: REGRESSION guard. Wrong implementation: overwriting or merging into the backup, or
    renaming onto it; red on the backup bytes or on the surviving new folder. Reachability: after
    the first call a fresh ``metadata/`` with different bytes is created. Both-states anchor:
    bytes and existence. Second correct implementation: checking ``backup_dir.exists()`` first, as
    the code does.
    """
    target, inodes = _tree(tmp_path)
    ctx = _ctx(target)
    ctx._apply_clear_snakemake_metadata("lbl")
    fresh = target / ".snakemake" / "metadata"
    fresh.mkdir()
    (fresh / "new").write_bytes(b"new")
    ctx._apply_clear_snakemake_metadata("lbl")
    bak = target / ".snakemake" / "metadata.bak.lbl"
    assert (fresh / "new").read_bytes() == b"new"
    assert not (bak / "new").exists() and (bak / "top").read_bytes() == b"t"


def test_a_tree_with_no_metadata_folder_is_left_alone(tmp_path):
    """Property: with no ``.snakemake/metadata`` nothing is created.

    Class: REGRESSION guard. Wrong implementation: creating an empty backup; red on the listing.
    Reachability: a never-run analysis. Both-states anchor: the filesystem is listed. Second correct
    implementation: an early return on ``not is_dir()``.
    """
    target = tmp_path / "analysis"
    (target / ".snakemake").mkdir(parents=True)
    _ctx(target)._apply_clear_snakemake_metadata("lbl")
    assert list((target / ".snakemake").iterdir()) == []
