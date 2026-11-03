"""Reviewer tests for chunk 3a.1: the zarr write path, its deferred-index block, the chapter publish and
the lint's spelling assumption.

Written by the reviewing seat, not the coder. Every test builds a few kilobytes of data under
``tmp_path`` or reads source text; none compiles or runs a solver.

Held (green before and after the coder's next turn, each with a named mutation that reds it):
the two ``deferred_consolidation`` block-shape tests, the chapter kill-point test, the
alias-census test. The must-fire control of the census is inside its own test.

Not established here: behaviour on a store larger than a few chunks, on an object store, or
under two processes publishing one store.
"""

from __future__ import annotations

import ast
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import hhemt
from hhemt.utils import (
    append_zarr,
    chapter_flag_for,
    chapter_store_for,
    completed_chapters,
    deferred_consolidation,
    reap_unflagged_chapters,
    verify_and_flag_chapter,
    write_zarr,
)

SRC_ROOT = Path(hhemt.__file__).resolve().parent


def _small(value: float = 1.0, n: int = 3) -> xr.Dataset:
    return xr.Dataset({"a": (("t",), np.full(n, value))}, coords={"t": np.arange(n)})


def _index(store: Path) -> set[str] | None:
    """Node names the root ``zarr.json`` lists in its consolidated index, or None when it carries no index."""
    consolidated = json.loads((store / "zarr.json").read_text()).get("consolidated_metadata")
    return None if consolidated is None else set(consolidated["metadata"])


def _on_disk(store: Path) -> set[str]:
    """Every group or array below the root that has its own ``zarr.json``, as a slash-joined path."""
    return {str(p.parent.relative_to(store)) for p in store.rglob("zarr.json") if p.parent != store}


# ---- 1. the shape of a deferred block, over its body class ------------------------------------------


@pytest.mark.parametrize("body", ["empty", "removes-a-group", "writes-a-new-group"])
def test_a_block_that_exits_normally_leaves_an_index_equal_to_the_store_whatever_its_body(tmp_path, body):
    """After a NORMAL block exit over an existing store the root index lists exactly the groups on disk.

    Input class: the body of a ``with deferred_consolidation(store)`` block over a store that
    already carries an index, in three positions: it does nothing (the system build when every
    analysis is unchanged), it only deletes a group (the build's ``rmtree``), it only adds one.
    Enforces: approved shape D13A, "its normal exit rebuilds the index once"; the coder's round-2
    change that skips the rebuild for an empty block (review item 5). Why: ``open_store`` trusts the
    index whenever the root document carries one, and the staleness check reports an ABSENT index as
    an ERROR, so an empty block that left the index removed (the entry removes it first) would make
    a store that no write touched unreadable-as-indexed. A second correct implementation: a block
    that skips both the removal and the rebuild when its body wrote nothing passes too. Kills it:
    making the exit rebuild conditional on the body having appended anything.
    """
    store = tmp_path / "system.zarr"
    append_zarr(_small(), store, group="g1")
    append_zarr(_small(), store, group="g2")
    assert _index(store) == _on_disk(store), "fixture: the store must start indexed and current"

    with deferred_consolidation(store):
        if body == "removes-a-group":
            import shutil

            shutil.rmtree(store / "g1")
        elif body == "writes-a-new-group":
            append_zarr(_small(), store, group="g3", consolidate=False)

    assert _index(store) is not None, f"block with body {body!r} left the store without an index"
    assert _index(store) == _on_disk(store), f"index {sorted(_index(store))} != disk {sorted(_on_disk(store))}"


def test_a_block_over_a_store_that_does_not_exist_creates_nothing_and_does_not_raise(tmp_path):
    """An empty block over an absent store leaves it absent: the exit has no store to index.

    Input class: the system build when no analysis has a DataTree yet (every analysis skipped by
    ``ValueError``), on a first run. Enforces: the round-2 guard in the helper's exit
    (review item 5). Why: rebuilding an index on a path that holds no store raises
    ``FileNotFoundError`` (measured: ``consolidate_zarr_store`` on an absent path), which would
    turn a legitimate empty build into a crash. Kills it: calling ``consolidate_zarr_store``
    unconditionally at exit (the guard's removal). A second correct implementation: a block that
    only rebuilds when the body appended something passes as well.
    """
    store = tmp_path / "never_written.zarr"
    with deferred_consolidation(store):
        pass
    assert not store.exists()


# ---- 2. a killed chapter write, at each point the publish can be cut ---------------------------------


@pytest.mark.parametrize("kill_point", ["mid-write", "after-write-before-rename", "after-rename-before-flag"])
def test_a_chapter_write_killed_at_any_point_leaves_no_store_a_reader_trusts_and_resume_rewrites_it(
    tmp_path, monkeypatch, kill_point
):
    """Whatever point a chapter write is cut at, reaping leaves the chapters directory with no entry for it.

    Input class: the three points the publish-by-rename can be killed at: during the temporary
    store's write, after it and before the rename, after the rename and before the flag.
    Enforces: the coder's round-2 extension of ``reap_unflagged_chapters`` to ``chapter_*.zarr.tmp``
    (review item 3) and the shape's reader rule "a chapter counts as complete only when its flag
    exists". Why: a reader trusts a chapter by the flag plus a store at the chapter's own path; a
    leftover ``.tmp`` is never at that path, but one that the reaper missed would sit in the
    directory until a later run happened to write the same index, and an unflagged store at the
    final path must still be removed. The resume then writes the same chapter index and flags it.
    Kills it: removing the ``.zarr.tmp`` loop from the reaper (the ``mid-write`` row reds), or
    letting ``verify_and_flag_chapter`` run before the rename. A second correct implementation: a
    writer that sweeps its own ``.tmp`` before writing, with the reaper unchanged, passes too.
    """
    chapters = tmp_path / "tseries.zarr.chapters"
    chapters.mkdir()
    store = chapter_store_for(chapters, 0)
    flag = chapter_flag_for(chapters, 0)
    ds = _small(n=4).rename({"t": "timestep_min"})

    real_to_zarr = xr.Dataset.to_zarr
    real_replace = os.replace

    if kill_point == "mid-write":

        def _partial_then_die(self, store_arg, *args, **kwargs):
            real_to_zarr(self.isel(timestep_min=slice(0, 2)), store_arg, *args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(xr.Dataset, "to_zarr", _partial_then_die)
    elif kill_point == "after-write-before-rename":

        def _die(src, dst, *args, **kwargs):
            # Only the publish's own rename is cut; zarr's local store also uses os.replace for its files.
            if Path(dst) == store:
                raise KeyboardInterrupt
            return real_replace(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "replace", _die)

    with pytest.raises(KeyboardInterrupt):
        write_zarr(ds, store, chunks=None, encoding={})
        if kill_point == "after-rename-before-flag":
            raise KeyboardInterrupt  # killed before verify_and_flag_chapter ran

    monkeypatch.setattr(xr.Dataset, "to_zarr", real_to_zarr)
    monkeypatch.setattr(os, "replace", real_replace)

    assert completed_chapters(chapters) == {}, "an unflagged chapter was reported complete"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reap_unflagged_chapters(chapters, scenario_dir=tmp_path)
    assert sorted(p.name for p in chapters.iterdir()) == [], f"entries survive the reap: {sorted(chapters.iterdir())}"

    write_zarr(ds, store, chunks=None, encoding={})
    verify_and_flag_chapter(store, flag, 4)
    assert list(completed_chapters(chapters)) == [0]


# ---- 3. the lint recognises a route by its spelling, so no spelling it cannot read may exist in src ------

#: Names whose call changes a zarr store's data or metadata (the same open set the lint's own constants name).
_ROUTE_NAMES = frozenset(
    {
        "open",
        "open_group",
        "open_array",
        "group",
        "create",
        "array",
        "create_array",
        "create_group",
        "save",
        "save_array",
        "save_group",
        "consolidate_metadata",
    }
)


def _alias_spellings(source: str) -> list[str]:
    """Spellings of a zarr write route that the lint's ``zarr.<name>(...)`` / ``.to_zarr(...)`` forms cannot see.

    Four: ``from zarr import <route name>``; ``import zarr as <other>``; a ``.to_zarr`` attribute
    that is not the callee of its own call (an alias taken to a name); ``getattr(x, "to_zarr")``.
    """
    found: list[str] = []
    tree = ast.parse(source)
    callee_ids = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "zarr":
            found += [f"from zarr import {a.name}" for a in node.names if a.name in _ROUTE_NAMES]
        elif isinstance(node, ast.Import):
            found += [f"import zarr as {a.asname}" for a in node.names if a.name == "zarr" and a.asname]
        elif isinstance(node, ast.Attribute) and node.attr == "to_zarr" and id(node) not in callee_ids:
            found.append(".to_zarr taken as a value")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "to_zarr"
        ):
            found.append('getattr(..., "to_zarr")')
    return found


def test_no_module_in_src_spells_a_zarr_write_route_in_a_form_the_write_discipline_lint_cannot_read():
    """No file under ``src/hhemt`` aliases a zarr write route; the census itself fires on each alias spelling.

    Input class: the spellings of one route. The write-discipline lint matches the route by the text
    ``zarr.<name>(`` and ``.to_zarr(``; measured on the chunk tree, ``from zarr import open_group``,
    ``import zarr as z``, ``from zarr import consolidate_metadata``, ``w = ds.to_zarr; w(p)`` and
    ``getattr(ds, "to_zarr")(p)`` each returned an EMPTY violation list from ``find_violations``.
    Enforces: approved shape D13 item 2, "stated as a property over the routes, because the routes are
    an open set". Why: the lint is the only thing between the helpers and a later writer, and it is
    blind to exactly these spellings, so the property it states holds only while src contains none.
    Both arms are in this test: the census returns a finding on a synthetic file for every spelling
    (the must-fire control) and none on the real tree. Kills it: adding ``import zarr as z`` to any
    src module and writing through it. A second correct implementation: a lint that resolves aliases,
    with this census removed, satisfies the property by another route.
    """
    controls = {
        "from zarr import open_group\ndef f(p):\n    open_group(p, mode='a')\n": "from zarr import open_group",
        "import zarr as z\ndef f(p):\n    z.open_group(p, mode='a')\n": "import zarr as z",
        "from zarr import consolidate_metadata\n": "from zarr import consolidate_metadata",
        "def f(ds, p):\n    w = ds.to_zarr\n    w(p)\n": ".to_zarr taken as a value",
        "def f(ds, p):\n    getattr(ds, 'to_zarr')(p)\n": 'getattr(..., "to_zarr")',
    }
    for source, spelling in controls.items():
        assert spelling in _alias_spellings(source), f"control: the census did not fire on {spelling!r}"
    assert _alias_spellings("import zarr\ndef f(ds, p):\n    ds.to_zarr(p)\n") == [], "control: a plain spelling fired"

    offenders = {
        str(path.relative_to(SRC_ROOT)): _alias_spellings(path.read_text(encoding="utf-8"))
        for path in sorted(SRC_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    offenders = {name: found for name, found in offenders.items() if found}
    assert offenders == {}, f"aliased zarr write routes the lint cannot read: {offenders}"


# ---- 4. a killed experiment-tree unification must not leave an index readers trust and that is wrong ----


@pytest.mark.parametrize("indexed", [True, False], ids=["root-carries-index", "root-has-no-index"])
def test_a_unify_to_experiment_tree_killed_after_the_moves_leaves_no_wrong_index(tmp_path, monkeypatch, indexed):
    """After `_apply_zarr_unify_to_experiment_tree` is killed between the group moves and the `parameters`
    append, the root index is absent or equals the store's listing.

    Input class: a flat store, with and without a root index, killed at the one point after
    ``shutil.move`` and before the final append (simulated by making ``hhemt.utils.append_zarr``
    raise at entry). Enforces: approved shape D13 end state and D13A's crash-window requirement,
    "an interrupted run must not leave an index that lists fewer (or more) groups than the store
    holds while readers trust it", applied to a route this chunk edits (coordinator re-scope,
    round 2). Why: measured on the chunk tree, the index-carrying store keeps its pre-move node
    list while the groups sit under ``member/``; ``open_datatree(consolidated=True)`` then returns
    ``/sa_a, /sa_b`` where the store holds ``/member/sa_a, /member/sa_b``, with no error, and the
    re-run returns at ``dest.exists()`` without repairing it. REGRESSION: red on the pre-change
    tree too (the window pre-exists), output pasted in the scratch. Kills it: running the move
    and the append inside ``with deferred_consolidation(store)``, which removes the index first.
    A second correct implementation: rebuilding the index before returning on the ``dest.exists()``
    path, with the move outside any block, passes as well.
    """
    import shutil  # noqa: F401  (kept local: the test reads no shutil, the op under test does)

    import zarr

    from hhemt.version_migration.context import MigrationContext

    store = tmp_path / "flat.zarr"
    for group in ("sa_a", "sa_b"):
        _small().to_zarr(store, group=group, mode="a", consolidated=False)
    if indexed:
        zarr.consolidate_metadata(str(store))
        assert _index(store) == _on_disk(store), "fixture: the store must start with a current index"

    def _die(*args, **kwargs):
        raise RuntimeError("killed")

    monkeypatch.setattr("hhemt.utils.append_zarr", _die)
    context = object.__new__(MigrationContext)
    context.migration_id = "reviewer-probe"
    with pytest.raises(RuntimeError, match="killed"):
        context._apply_zarr_unify_to_experiment_tree(str(store), "member", {"p": [1, 2]}, "p")

    assert "member/sa_a" in _on_disk(store), "fixture: the moves must have run before the kill"
    index = _index(store)
    assert index is None or index == _on_disk(store), (
        f"a reader trusting the index sees {sorted(index)} but the store holds {sorted(_on_disk(store))}"
    )
