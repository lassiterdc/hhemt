"""V-P1.5 — fast_rmtree restamps parent DU sentinels when analysis_dir kwarg is passed.

Maps to SE F-I Flag 3 contract: passing `analysis_dir=` to `fast_rmtree` triggers
in-utility re-stamping of parent-scope DU sentinels (per the mutation-site stipulation).
The `path == analysis_dir` short-circuit at SE F-I Flag 2 means a root-wipe call does
NOT re-stamp a directory being deleted.

Run:
    conda run -n hhemt python -m pytest tests/test_synth_fast_rmtree_restamps_parents.py -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hhemt.utils import fast_rmtree
from hhemt.du_sentinels import (
    compute_and_write_scope_sentinel,
    read_du_sentinel,
)


def _seed_scope(scope_dir: Path, child_bytes: dict[str, int]) -> None:
    """Create a scope directory populated with named child files of given sizes."""
    scope_dir.mkdir(parents=True, exist_ok=True)
    for name, n_bytes in child_bytes.items():
        (scope_dir / name).write_bytes(b"x" * n_bytes)


def test_fast_rmtree_with_analysis_dir_kwarg_restamps_parent(tmp_path: Path) -> None:
    """fast_rmtree(child, analysis_dir=scope) re-stamps scope's _du.json to exclude
    the removed child's bytes."""
    scope_dir = tmp_path / "analysis"
    _seed_scope(scope_dir, {"a.bin": 1000, "b.bin": 2000})

    # Seed a baseline sentinel reflecting the full state.
    compute_and_write_scope_sentinel(scope_dir, scope="analysis", include_breakdown=True)
    before = read_du_sentinel(scope_dir / "_status" / "_du.json")
    assert before is not None
    assert before["disk_utilization_bytes"] == 3000

    # Remove one of the children with the analysis_dir kwarg passed.
    fast_rmtree(scope_dir / "a.bin", analysis_dir=scope_dir)

    after = read_du_sentinel(scope_dir / "_status" / "_du.json")
    assert after is not None
    assert after["disk_utilization_bytes"] == 2000, (
        "Parent sentinel did not exclude removed child's bytes after fast_rmtree with "
        "analysis_dir kwarg — in-utility re-stamp contract violated"
    )


def test_fast_rmtree_without_analysis_dir_kwarg_does_not_restamp(tmp_path: Path) -> None:
    """fast_rmtree(child) without the kwarg leaves the parent sentinel unchanged
    (stale; the caller is responsible for sentinel accuracy out-of-band)."""
    scope_dir = tmp_path / "analysis"
    _seed_scope(scope_dir, {"a.bin": 1000, "b.bin": 2000})

    compute_and_write_scope_sentinel(scope_dir, scope="analysis", include_breakdown=True)
    sentinel = scope_dir / "_status" / "_du.json"
    before_mtime = sentinel.stat().st_mtime
    before_payload = read_du_sentinel(sentinel)
    assert before_payload is not None
    assert before_payload["disk_utilization_bytes"] == 3000

    time.sleep(1.1)

    # Remove child WITHOUT the kwarg; parent sentinel must stay untouched.
    fast_rmtree(scope_dir / "a.bin")

    assert sentinel.exists(), "Parent sentinel was deleted by fast_rmtree without kwarg"
    assert sentinel.stat().st_mtime == before_mtime, (
        "Parent sentinel mtime advanced even though analysis_dir kwarg was NOT passed"
    )
    after_payload = read_du_sentinel(sentinel)
    assert after_payload is not None
    assert after_payload["disk_utilization_bytes"] == 3000, (
        "Parent sentinel payload changed without the analysis_dir kwarg"
    )


def test_fast_rmtree_root_wipe_short_circuits(tmp_path: Path) -> None:
    """fast_rmtree(scope, analysis_dir=scope) short-circuits the re-stamp (no rglob
    walk on the deleted root) per the EXEMPT-site convention in SE F-I Flag 2."""
    scope_dir = tmp_path / "analysis"
    _seed_scope(scope_dir, {"a.bin": 1000, "b.bin": 2000})
    compute_and_write_scope_sentinel(scope_dir, scope="analysis", include_breakdown=True)

    # Sanity: sentinel exists before delete.
    assert (scope_dir / "_status" / "_du.json").exists()

    # Delete the scope ITSELF, passing analysis_dir=scope_dir — must short-circuit.
    fast_rmtree(scope_dir, analysis_dir=scope_dir)

    # The scope is gone; the EXEMPT short-circuit fired (no attempt to re-stamp a
    # deleted directory, no errors raised).
    assert not scope_dir.exists(), "fast_rmtree did not actually delete the root"
