"""Unit test: _walk_root_and_breakdown issues zero realpath syscalls and
preserves byte-total + per-child breakdown parity.

Plan: library/docs/planning/projects/hhemt/bugs/reprocess du sentinel realpath and dry run mutation.md
Guards R1 (no realpath) + R2 (parity) — the realpath cliff in
du_sentinels._walk_root_and_breakdown is removed (use p.relative_to(root),
not p.resolve().relative_to(root.resolve())), and the no-resolve walk returns
the same (total, breakdown, walk_errors) as an independent os.walk reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hhemt import du_sentinels


def _build_tree(root: Path) -> None:
    (root / "a").mkdir(parents=True)
    (root / "a" / "f1.bin").write_bytes(b"x" * 100)
    (root / "a" / "sub").mkdir()
    (root / "a" / "sub" / "f2.bin").write_bytes(b"y" * 50)
    zarr = root / "analysis_datatree.zarr" / "H" / "0.0.0"
    zarr.parent.mkdir(parents=True)
    zarr.write_bytes(b"z" * 200)
    (root / "analysis_datatree.zarr" / ".zattrs").write_bytes(b"{}")
    status = root / "_status"
    status.mkdir()
    (status / "_du.json").write_bytes(b"{}")  # excluded by the _status filter
    # Clause-6 discriminator: a `_status` BELOW a normal child, not only at the
    # scope root. Without it the walker-agreement assertion in
    # test_walk_root_bytes_parity is falsified by mutating _walk_root_bytes but
    # NOT _walk_root_and_breakdown, whose skip_status_top already excludes a root
    # `_status`. With it, both mutations falsify. Same discriminator S9a added to
    # the parity-oracle fixture.
    deep_status = root / "a" / "_status"
    deep_status.mkdir()
    (deep_status / "deep.flag").write_bytes(b"\x00" * 7)


def _reference_walk(root: Path) -> tuple[int, dict[str, int], int]:
    total = 0
    per_child: dict[str, int] = {}
    for dirpath, dirs, files in os.walk(root):
        # Clause 6: `_status` is excluded at ANY depth, matching
        # _walk_root_and_breakdown's skip_status_any_depth. EXACT equality — a
        # `startswith` prune would also drop `_status_old/`, which production counts.
        dirs[:] = [d for d in dirs if d != "_status"]
        for name in files:
            p = Path(dirpath) / name
            rel = p.relative_to(root)
            top = rel.parts[0]
            if top.startswith("_status"):
                continue
            size = p.stat().st_size
            total += size
            per_child[top] = per_child.get(top, 0) + size
    return total, per_child, 0


def test_walk_breakdown_parity(tmp_path: Path) -> None:
    root = tmp_path / "scope"
    _build_tree(root)
    total, breakdown, walk_errors = du_sentinels._walk_root_and_breakdown(root)
    ref_total, ref_breakdown, ref_errors = _reference_walk(root)
    assert total == ref_total
    assert breakdown == ref_breakdown
    assert walk_errors == ref_errors == 0
    assert "_status" not in breakdown


def test_walk_issues_no_realpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "scope"
    _build_tree(root)
    calls = {"n": 0}
    real_realpath = os.path.realpath

    def _counting_realpath(*args, **kwargs):
        calls["n"] += 1
        return real_realpath(*args, **kwargs)

    # Path.resolve() calls os.path.realpath under the hood in CPython; a stray
    # .resolve() in the walk would increment this counter. The os.scandir
    # rewrite must remain realpath-free (scandir + is_dir(follow_symlinks=False)
    # + entry.stat() never resolve).
    monkeypatch.setattr(os.path, "realpath", _counting_realpath)
    du_sentinels._walk_root_and_breakdown(root)
    assert calls["n"] == 0, f"_walk_root_and_breakdown issued {calls['n']} realpath calls; expected 0"


def _reference_bytes(root: Path) -> tuple[int, int]:
    """Independent os.walk reference for _walk_root_bytes: counts every file
    EXCEPT those under a `_status` directory at any depth (clause 6), recursing
    into real dirs only (matching rglob).

    The prune predicate is EXACT equality. `d.startswith("_status")` is WRONG and
    is corroborated by the wrong instrument: on a tree with `_status_old/f.bin`
    (17 B) + `_status/_du.json` (2 B) + `keep.bin` (3 B), production and this
    reference both read 20, a startswith prune reads 3 -- and
    _walk_root_and_breakdown also reads 3, because it applies a startswith rule at
    the TOP level for its own separate reason.
    """
    total = 0
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "_status"]
        for name in files:
            total += (Path(dirpath) / name).stat().st_size
    return total, 0


def test_walk_root_bytes_parity(tmp_path: Path) -> None:
    """_walk_root_bytes (os.scandir rewrite) is bit-identical to an independent
    os.walk reference implementing clause 6: `_status` directories are excluded at
    ANY depth, so the two production walkers agree on this fixture."""
    root = tmp_path / "scope"
    _build_tree(root)
    total, walk_errors = du_sentinels._walk_root_bytes(root)
    ref_total, ref_errors = _reference_bytes(root)
    assert total == ref_total
    assert walk_errors == ref_errors == 0
    # Clause 6 collapsed the old skip_status_top asymmetry: BOTH walkers exclude
    # every `_status` directory at any depth, so they agree on this fixture. The
    # fixture carries a `_status` at the root AND one under `a/`, which is what
    # makes this a property with content rather than a constant -- it is falsified
    # by removing skip_status_any_depth from EITHER helper (measured on the
    # repaired module: 361 vs 352 for _walk_root_bytes, 352 vs 359 for
    # _walk_root_and_breakdown).
    bd_total, _breakdown, _bd_errors = du_sentinels._walk_root_and_breakdown(root)
    assert total == bd_total


def test_walk_root_bytes_file_root(tmp_path: Path) -> None:
    """_walk_root_bytes handles a file-root (the only caller path that can pass
    a file) — preserved behavior across the rewrite."""
    f = tmp_path / "lonely.bin"
    f.write_bytes(b"q" * 73)
    total, walk_errors = du_sentinels._walk_root_bytes(f)
    assert total == 73
    assert walk_errors == 0


def test_walk_skips_symlinked_dir(tmp_path: Path) -> None:
    """Both walkers recurse into REAL subdirectories only — a symlinked
    directory is not descended (matching rglob's documented non-recursion).
    Guards the R1 parity invariant for the os.scandir rewrite."""
    root = tmp_path / "scope"
    (root / "real").mkdir(parents=True)
    (root / "real" / "f.bin").write_bytes(b"a" * 10)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "hidden.bin").write_bytes(b"b" * 999)
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("platform does not support directory symlinks")
    total, breakdown, walk_errors = du_sentinels._walk_root_and_breakdown(root)
    # The symlinked dir's contents (999 bytes) must NOT be counted.
    assert total == 10
    assert breakdown == {"real": 10}
    assert walk_errors == 0
