"""Reviewer tests for chunk 2b.4: the single-pass byte walk inside ``fast_rmtree``.

``fast_rmtree`` measures the bytes under a directory with ``_tree_file_bytes`` (one ``os.scandir``
pass) where it used ``Path.rglob`` plus ``is_file`` plus ``stat``. The contract is the same byte count
for every tree shape a caller can present, the same deletion, and the same error behaviour. Each test
compares against the OLD walk, written out below as the oracle, on shapes the coder's file does not build.

SAFETY: every tree is created and deleted under pytest's ``tmp_path``; every symlink target is also
under ``tmp_path``; nothing here points ``fast_rmtree`` at a path outside it. No solver is compiled
or run. Run as ``PYTHONPATH=${PWD}/src pytest tests/test_fast_rmtree_walk_reviewer.py``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import hhemt.utils as utils
from hhemt.utils import _tree_file_bytes, fast_rmtree


def _old_walk(root: Path) -> int:
    """The walk ``fast_rmtree`` used before this change (the oracle)."""
    total = 0
    for f in root.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _w(path: Path, n: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * n)
    return path


def test_hard_links_symlink_loops_and_special_files_count_as_the_old_walk_counts_them(tmp_path):
    """Property: the byte count equals the old walk's for hard links, a symlink loop and a FIFO.

    Input class: two names for one inode (each counted, as before), a directory symlink pointing at its
    own ancestor (a loop that a following walk would never leave), a symlink to a file in another
    directory, and a FIFO (not a regular file). Approved-shape row: section 5.1 row 2b.4, "returning the
    same byte count". Why required: the count feeds the DU sentinel adjustment, so a double or missing
    count drifts the disk-use ledger. The change that kills it is following directory symlinks, counting
    by inode, or counting non-regular entries. A second correct implementation is ``os.walk`` with
    ``followlinks=False`` and ``lstat`` of files plus a followed ``stat`` for file symlinks.
    """
    root = tmp_path / "t"
    a = _w(root / "a" / "f.bin", 1000)
    (root / "b").mkdir()
    os.link(a, root / "b" / "hard.bin")
    (root / "a" / "loop").symlink_to(root, target_is_directory=True)
    (root / "b" / "to_file").symlink_to(a)
    os.mkfifo(root / "b" / "fifo")
    expected = _old_walk(root)
    assert expected == 3000  # a, its hard link, and the file symlink's target
    assert _tree_file_bytes(root) == expected


def test_symlinked_directory_with_files_is_neither_entered_nor_counted(tmp_path):
    """Property: files reachable only through a directory symlink are not counted.

    Input class: a directory symlink to a sibling directory that holds 5 KiB, inside a tree that holds
    1 KiB of its own. Approved-shape row: section 5.1 row 2b.4. Why required: following it would count
    the target twice and, for a link out of the tree, bytes the deletion never frees. Killed by
    ``entry.is_dir()`` with the default ``follow_symlinks=True``. Second correct implementation: an
    ``os.walk`` that does not follow links.
    """
    root = tmp_path / "t"
    _w(root / "own.bin", 1024)
    other = tmp_path / "other"
    _w(other / "big.bin", 5120)
    (root / "link").symlink_to(other, target_is_directory=True)
    assert _old_walk(root) == 1024
    assert _tree_file_bytes(root) == 1024


def test_deep_and_wide_trees_count_the_same_without_recursion_limits(tmp_path):
    """Property: a 300-level chain and a 400-entry directory give the old walk's count.

    Input class: a nesting deeper than the interpreter's default recursion limit divided by the frames a
    recursive walk spends per level, and a wide directory. Approved-shape row: section 5.1 row 2b.4.
    Why required: bundle trees nest zarr groups and a recursive walk would raise ``RecursionError``
    that the old walk did not. Killed by a recursive implementation. Second correct implementation:
    ``os.walk``.
    """
    deep = tmp_path / "t"
    node = deep
    for _ in range(300):
        node = node / "d"
    _w(node / "leaf.bin", 7)
    for i in range(400):
        _w(deep / "wide" / f"f{i}", i)
    assert _tree_file_bytes(deep) == _old_walk(deep) == 7 + sum(range(400))


def test_unreadable_subdirectory_contributes_nothing_and_does_not_raise(tmp_path):
    """Property: a directory that cannot be listed adds nothing, and its siblings are still counted.

    Input class: one subdirectory with mode 000 beside a readable one (skipped when the process can read
    it anyway, e.g. as root). Approved-shape row: section 5.1 row 2b.4, "same byte count" and the old
    walk's tolerance of ``OSError``. Why required: a permissions oddity under a bundle must not abort a
    delete. Killed by dropping the ``except OSError`` around ``scandir``. Second correct implementation:
    ``os.walk`` with its default ``onerror=None``.
    """
    root = tmp_path / "t"
    _w(root / "ok" / "f", 100)
    _w(root / "locked" / "g", 900)
    (root / "locked").chmod(0)
    try:
        if os.access(root / "locked", os.R_OK):
            pytest.skip("directory permissions are not enforced for this user")
        assert _tree_file_bytes(root) == _old_walk(root) == 100
    finally:
        (root / "locked").chmod(0o700)


def test_a_file_that_vanishes_after_it_is_listed_counts_zero_and_siblings_still_count(tmp_path, monkeypatch):
    """Property: an entry deleted between the directory listing and its stat adds nothing and raises nothing.

    Input class: a directory whose listing names ``gone`` and ``kept``, with ``gone`` removed by another
    process after the listing (simulated by deleting it from inside the patched ``scandir`` once the
    listing is built). Approved-shape row: section 5.1 row 2b.4, "files that vanish mid-walk".
    Reachability: a delete racing a running job under the same analysis folder. Why required: the old
    walk caught ``OSError`` per file; a missing guard turns a race into a crash. Killed by moving the
    ``stat`` outside the per-entry ``try``. Second correct implementation: ``os.walk`` plus ``lstat``
    guarded per file.
    """
    root = tmp_path / "t"
    _w(root / "gone", 111)
    _w(root / "kept", 222)
    real_scandir = os.scandir

    def _scandir(path):
        iterator = real_scandir(path)
        if Path(path) == root:
            names = sorted(iterator, key=lambda e: e.name != "gone")  # the vanishing entry is visited first
            iterator.close()
            (root / "gone").unlink()

            class _It:
                def __enter__(self):
                    return iter(names)

                def __exit__(self, *exc):
                    return False

            return _It()
        return iterator

    monkeypatch.setattr(os, "scandir", _scandir)
    assert _tree_file_bytes(root) == 222


def test_the_walk_calls_neither_path_stat_nor_is_file_per_entry(tmp_path, monkeypatch):
    """Property: no second ``stat`` is made through ``pathlib`` for the entries under the tree.

    Input class: 50 files in one directory. Approved-shape row: section 5.1 row 2b.4, "using each
    entry's own stat". Why required: the point of the change is one stat per file; a ``Path.stat`` or
    ``Path.is_file`` call per entry is the old cost back. Killed by re-adding ``Path(entry.path).stat()``.
    Second correct implementation: ``entry.stat(follow_symlinks=False)`` for non-links.
    """
    root = tmp_path / "t"
    for i in range(50):
        _w(root / f"f{i}", 10)
    calls: list[str] = []
    real_stat, real_is_file = Path.stat, Path.is_file

    def _stat(self, *a, **k):
        calls.append(str(self))
        return real_stat(self, *a, **k)

    def _is_file(self, *a, **k):
        calls.append(str(self))
        return real_is_file(self, *a, **k)

    monkeypatch.setattr(Path, "stat", _stat)
    monkeypatch.setattr(Path, "is_file", _is_file)
    assert _tree_file_bytes(root) == 500
    assert calls == []


def test_fast_rmtree_returns_the_old_count_deletes_everything_and_keeps_outside_targets(tmp_path):
    """Property: on a mixed tree ``fast_rmtree`` returns the old walk's count and removes the whole tree.

    Input class: files, an empty directory, a hard link, a symlink to an outside file and to an outside
    directory, both targets inside ``tmp_path`` but outside the tree. Approved-shape row: section 5.1
    row 2b.4 (``rm -rf`` step untouched). Why required: the count is the function's return value and the
    deletion must not follow links. Killed by a change to the delete step or the count. Second correct
    implementation: ``shutil.rmtree`` after the same walk.
    """
    root = tmp_path / "t"
    outside = _w(tmp_path / "outside" / "keep.bin", 4096)
    outside_dir = tmp_path / "outside_dir"
    _w(outside_dir / "k", 8)
    a = _w(root / "a" / "f", 300)
    (root / "empty").mkdir()
    (root / "b").mkdir()
    os.link(a, root / "b" / "h")
    (root / "lf").symlink_to(outside)
    (root / "ld").symlink_to(outside_dir, target_is_directory=True)
    expected = _old_walk(root)
    assert fast_rmtree(root) == expected == 300 + 300 + 4096
    assert not root.exists()
    assert outside.read_bytes() == b"x" * 4096 and (outside_dir / "k").exists()


def test_rm_failure_falls_back_to_shutil_rmtree_with_the_callers_onerror_and_same_count(tmp_path, monkeypatch):
    """Property: when ``rm -rf`` fails the tree is still deleted through ``shutil.rmtree`` and the count is returned.

    Input class: ``subprocess.run`` raising ``CalledProcessError`` for the delete command. Approved-shape
    row: section 5.1 row 2b.4 (the ``rm -rf`` call is untouched; "same error behaviour"). Why required:
    the fallback is the only delete path on hosts without ``rm``. Killed by a count computed after the
    delete or a lost fallback. Second correct implementation: ``os.walk`` bottom-up unlinking.
    """
    root = tmp_path / "t"
    _w(root / "a" / "f", 64)
    seen: list[list[str]] = []

    def _boom(cmd, *a, **k):
        seen.append(list(cmd))
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(utils.subprocess, "run", _boom)
    assert fast_rmtree(root) == 64
    assert seen == [["rm", "-rf", str(root)]]
    assert not root.exists()


def test_absent_path_and_dangling_symlink_follow_missing_ok(tmp_path):
    """Property: an absent path and a dangling symlink return 0, or raise when ``missing_ok`` is False.

    Input class: a path that does not exist and a symlink whose target (inside ``tmp_path``) does not.
    Approved-shape row: section 5.1 row 2b.4 ("same error behaviour"); the docstring says a dangling
    symlink reads as missing and is left in place. Why required: callers rely on 0 for already-clean
    state. Killed by measuring before the existence check. Second correct implementation: a
    ``try/except FileNotFoundError`` around the stat.
    """
    gone = tmp_path / "gone"
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "no_such_target")
    assert fast_rmtree(gone) == 0 and fast_rmtree(dangling) == 0
    assert dangling.is_symlink()
    for p in (gone, dangling):
        with pytest.raises(FileNotFoundError):
            fast_rmtree(p, missing_ok=False)
