"""Unit test: _walk_root_and_breakdown issues zero realpath syscalls and
preserves byte-total + per-child breakdown parity.

Plan: library/docs/planning/projects/TRITON-SWMM_toolkit/bugs/reprocess du sentinel realpath and dry run mutation.md
Guards R1 (no realpath) + R2 (parity) — the realpath cliff in
du_sentinels._walk_root_and_breakdown is removed (use p.relative_to(root),
not p.resolve().relative_to(root.resolve())), and the no-resolve walk returns
the same (total, breakdown, walk_errors) as an independent os.walk reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from TRITON_SWMM_toolkit import du_sentinels


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


def _reference_walk(root: Path) -> tuple[int, dict[str, int], int]:
    total = 0
    per_child: dict[str, int] = {}
    for dirpath, _dirs, files in os.walk(root):
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
    # .resolve() in the walk would increment this counter.
    monkeypatch.setattr(os.path, "realpath", _counting_realpath)
    du_sentinels._walk_root_and_breakdown(root)
    assert calls["n"] == 0, f"_walk_root_and_breakdown issued {calls['n']} realpath calls; expected 0"
