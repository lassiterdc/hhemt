"""The real constructor of the synthetic case discards a reused tree stamped at another layout.

``tests/test_cache_key_layout_term.py`` checks the helper alone and, for the constructor, the order of two source
lines. These tests run the real constructor up to the point where it would provision the solver source: the tier-0
model build, the per-worktree cache root and the source provisioning are replaced by stand-ins, so nothing is built,
cloned or compiled, and a stamped tree under a temporary runs root is the only state the constructor acts on.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt.cache.paths import synthetic_runs_root
from tests.fixtures import test_case_builder as tcb

NAME = "reuse_layout_case"
#: The runs-root directory name, read from its single product definition so this file holds no literal of it.
RUNS_DIR = synthetic_runs_root().name


class _Reached(Exception):
    """Raised by the stand-in for the source provisioning: the constructor got past the reuse branch."""


def _plant(directory: Path, layout_version: int) -> None:
    directory.mkdir(parents=True)
    (directory / "_version.json").write_text(
        json.dumps(
            {"layout_version": layout_version, "toolkit_version": "0", "created_at": "t", "migration_history": []}
        )
    )
    (directory / "inner").mkdir()
    (directory / "inner" / "payload.txt").write_text("x")


def _construct(tmp_path: Path, monkeypatch, *, start_from_scratch: bool) -> Path:
    runs_root = tmp_path / "root"
    monkeypatch.setattr(tcb, "get_or_build_synthetic_case", lambda *a, **k: SimpleNamespace(cache_dir=tmp_path / "t0"))
    monkeypatch.setattr(tcb, "slug_runs_root", lambda slug: tmp_path / "slug")

    def stop(*args, **kwargs):
        raise _Reached

    monkeypatch.setattr(tcb, "provision_borrower", stop)
    with pytest.raises(_Reached):
        tcb.retrieve_synth_TRITON_SWMM_test_case(
            NAME, start_from_scratch=start_from_scratch, runs_root_override=runs_root
        )
    return runs_root / RUNS_DIR / NAME


def test_the_constructor_removes_a_reused_tree_stamped_at_another_layout(tmp_path, monkeypatch):
    """Property: constructing a case with ``start_from_scratch=False`` over a tree stamped at another layout leaves
    no trace of that tree, including its inner directory.

    Class: a planted tree stamped one layout version behind, built through the real constructor.
    Kills it: a helper that is defined and never called; a call placed after the directory is recreated; a call in a
    branch the reuse path does not take.
    """
    tree = tmp_path / "root" / RUNS_DIR / NAME
    _plant(tree, tcb._version_constants.LAYOUT_VERSION - 1)
    _construct(tmp_path, monkeypatch, start_from_scratch=False)
    assert tree.is_dir() and list(tree.iterdir()) == []


def test_the_constructor_keeps_a_reused_tree_stamped_at_the_current_layout(tmp_path, monkeypatch):
    """Control: the same construction over a tree at the current layout keeps the stamp and the inner payload."""
    tree = tmp_path / "root" / RUNS_DIR / NAME
    _plant(tree, tcb._version_constants.LAYOUT_VERSION)
    _construct(tmp_path, monkeypatch, start_from_scratch=False)
    assert (tree / "inner" / "payload.txt").read_text() == "x"
    assert json.loads((tree / "_version.json").read_text())["layout_version"] == tcb._version_constants.LAYOUT_VERSION
