"""Tests for the byte count ``fast_rmtree`` returns, now measured by one ``os.scandir`` pass.

The count used to come from ``Path.rglob`` plus ``is_file`` plus ``stat`` (two stat calls per file).
It now comes from ``_tree_file_bytes``, which uses each ``DirEntry``'s own stat. The returned number
must not change for any tree, including symlinks, and the walk must cost one stat per entry.

Every test builds a tree under pytest's ``tmp_path`` and deletes only that tree; none compiles or
runs a solver and all run on a laptop. Run from a worktree as
``PYTHONPATH=${PWD}/src pytest tests/test_fast_rmtree_walk.py``. The expected counts are literals
or come from the reference walk written out in this file (the pre-change algorithm), never from
the function under test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hhemt.utils import _tree_file_bytes, fast_rmtree


def _reference_bytes(root: Path) -> int:
    """The walk ``fast_rmtree`` used before this change, kept here as the oracle."""
    total = 0
    for f in root.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _build(tmp_path: Path) -> tuple[Path, int]:
    """A tree with files, nested dirs, an empty dir, a file symlink, a dir symlink and a dangling symlink.

    Returns the root and the literal byte total the walk must report: 10 + 20 + 30 + 40 + 5 (the
    file symlink counts its 40-byte target, so 40 appears twice) = 145. The directory symlink points at a 1000-byte file
    OUTSIDE the root and must add nothing; the dangling symlink adds nothing.
    """
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"x" * 1000)
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    (root / "empty").mkdir()
    (root / "f1").write_bytes(b"1" * 10)
    (root / "a" / "f2").write_bytes(b"2" * 20)
    (root / "a" / "b" / "f3").write_bytes(b"3" * 30)
    (root / "target40").write_bytes(b"4" * 40)
    (root / "link_to_file").symlink_to(root / "target40")
    (root / "link_to_dir").symlink_to(outside, target_is_directory=True)
    (root / "dangling").symlink_to(root / "does_not_exist")
    (root / "a" / "f5").write_bytes(b"5" * 5)
    return root, 10 + 20 + 30 + 40 + 40 + 5


def test_walk_counts_match_the_literal_and_the_reference_walk(tmp_path):
    """Property: ``_tree_file_bytes`` equals the pre-change walk and the hand-computed literal.

    Class: REGRESSION (holds before and after). The tree mixes every entry kind that could tell two
    walks apart: a file symlink (counts its target, so 40 appears twice), a directory symlink to a
    1000-byte file outside the root (counts nothing) and a dangling symlink (counts nothing). The
    plausible wrong implementations are one that follows directory symlinks, one that skips file
    symlinks, and one that counts a symlink's own length. Why required: callers use the count for
    disk-use accounting. A second correct implementation is ``os.walk`` with ``followlinks=False``
    and a per-file ``os.stat``.
    """
    root, literal = _build(tmp_path)
    assert literal == 145
    assert _reference_bytes(root) == literal
    assert _tree_file_bytes(root) == literal


def test_fast_rmtree_returns_the_same_count_and_removes_only_the_tree(tmp_path):
    """Property: ``fast_rmtree`` returns the literal count, deletes the tree and leaves a symlink's target alone.

    Class: REGRESSION on the count; the satisfying input differs from the test above by going through
    the public function and checking what survives. The directory symlink's target (1000 bytes) must
    still exist. The plausible wrong implementation is a walk that returns a different count, or a
    delete that follows the directory symlink. Why required: ``delete_and_account`` trusts the count.
    A second correct implementation is any deletion order that removes the same paths.
    """
    root, literal = _build(tmp_path)
    assert fast_rmtree(root) == literal
    assert not root.exists()
    assert (tmp_path / "outside" / "big.bin").read_bytes() == b"x" * 1000


def test_walk_makes_one_stat_per_file_entry(tmp_path, monkeypatch):
    """Property: the walk costs one ``stat`` call per non-directory entry, where the old walk cost two per file.

    Class: discriminator against the pre-change walk. A ``DirEntry``'s ``stat`` cannot be patched, so the count
    is taken on ``os.scandir`` itself: every entry
    yielded is wrapped so its ``stat`` calls are tallied. The tree has three regular files and one
    directory; the expected tally is 3 (the directory is classified by ``is_dir`` from the listing and
    never stat-ed). The plausible wrong implementation is ``Path.stat`` per file in addition to the
    entry's own, giving 6 here. Why required: on a shared file system each stat is a metadata round
    trip. A second correct implementation is ``os.walk`` plus ``os.stat``, which also tallies one per
    file when the tally is taken on ``os.stat``; the test pins the scandir-based form only.

    Not established: the tally counts calls through the wrapper, not syscalls the kernel issues.
    """
    root = tmp_path / "root"
    (root / "d").mkdir(parents=True)
    for name in ("a", "b"):
        (root / name).write_bytes(b"x" * 3)
    (root / "d" / "c").write_bytes(b"x" * 3)
    tally = {"entry_stat": 0, "path_stat": 0}
    real_scandir = os.scandir

    class _Entry:
        def __init__(self, entry):
            self._e = entry
            self.path = entry.path
            self.name = entry.name

        def is_dir(self, **kw):
            return self._e.is_dir(**kw)

        def stat(self, **kw):
            tally["entry_stat"] += 1
            return self._e.stat(**kw)

    class _Scan:
        def __init__(self, path):
            self._it = real_scandir(path)

        def __enter__(self):
            return (_Entry(e) for e in self._it.__enter__())

        def __exit__(self, *exc):
            return self._it.__exit__(*exc)

    real_path_stat = Path.stat

    def _path_stat(self, *a, **k):
        tally["path_stat"] += 1
        return real_path_stat(self, *a, **k)

    monkeypatch.setattr(os, "scandir", _Scan)
    monkeypatch.setattr(Path, "stat", _path_stat)
    assert _tree_file_bytes(root) == 9
    assert tally == {"entry_stat": 3, "path_stat": 0}


def test_missing_and_single_file_paths_keep_their_behaviour(tmp_path):
    """Property, differently positioned: a missing path returns 0 (or raises), a file returns its own size.

    Class: REGRESSION. These paths never reach the directory walk, so the plausible wrong
    implementation is a refactor that routes them through it. Why required: callers pass files and
    optional paths. A second correct implementation is checking the type first, as the code does.
    """
    assert fast_rmtree(tmp_path / "nope") == 0
    with pytest.raises(FileNotFoundError):
        fast_rmtree(tmp_path / "nope", missing_ok=False)
    f = tmp_path / "one.bin"
    f.write_bytes(b"x" * 7)
    assert fast_rmtree(f) == 7
    assert not f.exists()
