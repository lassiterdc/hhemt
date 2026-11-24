"""Reviewer probes of the complement runner's refusals and the role derivation (design DR-8, AC-1.3).

None of these reaches pytest-in-a-subprocess or a solver: each calls the runner's own functions over a hand-written
run directory and stops at the refusal, so none is simulation-bearing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hhemt.suite._runner as R


def _run_dir(root: Path, *, declared: list[str], chunks: list[dict] | None = None) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    base_chunks = chunks if chunks is not None else [{"chunk_id": 0, "kind": "cheap", "node_ids": ["t.py::a"]}]
    manifest = {
        "run_id": "rid",
        "chunk_count": len(base_chunks),
        "chunks": base_chunks,
        "declared_complement": declared,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "chunk_count").write_text(f"{len(base_chunks)}\n", encoding="utf-8")
    return run_dir


def _args(run_dir: Path) -> list[str]:
    return ["--complement", "--run-dir", str(run_dir), "--toolkit", str(run_dir.parent)]


def test_an_empty_declared_complement_is_refused_and_appends_nothing(tmp_path):
    """Input class: a run whose plan declared no complement.

    Required: an empty chunk hands pytest no node ids and pytest collects the whole suite, so the mode refuses
    before it writes anything. Killed by a mode that appends the empty chunk.
    """
    run_dir = _run_dir(tmp_path, declared=[])
    before = (run_dir / "manifest.json").read_bytes()
    with pytest.raises(SystemExit) as raised:
        R.main(_args(run_dir))
    assert raised.value.code not in (0, None)
    assert (run_dir / "manifest.json").read_bytes() == before
    assert not (run_dir / "chunks").exists()
    assert (run_dir / "chunk_count").read_text(encoding="utf-8").strip() == "1"


def test_a_directory_without_a_manifest_is_refused(tmp_path):
    """Input class: ``--run-dir`` naming a directory no plan wrote."""
    empty = tmp_path / "unplanned"
    empty.mkdir()
    with pytest.raises(SystemExit) as raised:
        R.main(["--complement", "--run-dir", str(empty), "--toolkit", str(tmp_path)])
    assert raised.value.code not in (0, None)
    assert list(empty.iterdir()) == []


def test_the_mode_without_a_run_dir_is_refused(tmp_path):
    """Input class: ``--complement`` alone, as ``--chunk`` alone is refused."""
    with pytest.raises(SystemExit) as raised:
        R.main(["--complement", "--toolkit", str(tmp_path)])
    assert raised.value.code not in (0, None)


def test_a_run_holding_a_complement_chunk_is_refused_whatever_its_position(tmp_path):
    """Input class: a manifest whose complement chunk is not the last chunk.

    Required: the refusal keys on the ``covers`` field and not on the last chunk, and it names that chunk's id.
    Killed by a refusal that inspects only ``chunks[-1]``.
    """
    chunks = [
        {"chunk_id": 0, "kind": "cheap", "node_ids": ["t.py::a"], "covers": "complement"},
        {"chunk_id": 1, "kind": "cheap", "node_ids": ["t.py::b"], "covers": "tail"},
    ]
    run_dir = _run_dir(tmp_path, declared=["t.py::a"], chunks=chunks)
    before = (run_dir / "manifest.json").read_bytes()
    with pytest.raises(SystemExit) as raised:
        R.main(_args(run_dir))
    message = str(raised.value.code)
    assert "already has its complement chunk 0" in message
    assert "--chunk 0" in message
    assert (run_dir / "manifest.json").read_bytes() == before
