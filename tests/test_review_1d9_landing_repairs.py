"""Reviewer tests for the landing repairs: the source edits that are executable, and the allowlist row.

Each test names the wrong implementation it kills and carries a control that varies the property under suspicion.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hhemt import utils
from hhemt.utils import (
    _DEFERRED_STORES,
    append_zarr,
    consolidate_zarr_store,
    deferred_consolidation,
    zarr_store_has_index,
)
from hhemt.version_migration import context as migration_context
from hhemt.version_migration.context import (
    _SERIES_GROUP_MAP,
    SERIES_MIGRATED_ATTR,
    _copy_series_into_tree,
)

REPO = Path(__file__).resolve().parents[1]
IDENTITY_FILE = REPO / "src" / "hhemt" / "eda" / "cross_sim_identity.py"


def _piece(value: float) -> xr.Dataset:
    return xr.Dataset({"depth": (("event_iloc", "elapsed_s"), np.full((1, 3), value))}, coords={"event_iloc": [0]})


def _signature(root: Path) -> dict[str, tuple[int, int]]:
    """Every file under ``root`` with its size and nanosecond mtime: a rewrite, a removal or an addition changes it."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_a_nested_deferred_block_leaves_the_outer_blocks_appends_permitted_and_the_index_to_the_outer_exit(tmp_path):
    """Property: the inner ``deferred_consolidation`` on a store an outer block holds is a no-op:
    it neither rebuilds the
    index nor discards the store from the open set, so an append after the inner block is still permitted and the
    index appears once, at the OUTER exit.

    Class: a changed code path (the four append sites now carry their own block and run inside the callers' blocks).
    Killed by: an inner block that rebuilds the index on exit (the index is present before the outer exit), or one
    that discards the store (the append after it raises ``ProcessingError``). Passes under a second correct
    implementation that counts the depth of nesting.
    Must-fire control: two SEQUENTIAL blocks on one store each rebuild the index, so the instrument reading the index
    sees the difference between a rebuild and a no-op; a non-nested append with ``consolidate=False`` is refused.
    """
    store = tmp_path / "s.zarr"
    append_zarr(_piece(1.0), store, group="g", mode="w")
    assert zarr_store_has_index(store), "control: a consolidating write leaves an index"
    with pytest.raises(Exception, match="deferred_consolidation"):
        append_zarr(_piece(9.0), store, group="g", consolidate=False, mode="w")

    with deferred_consolidation(store):
        assert not zarr_store_has_index(store), "the outer block removed the index on entry"
        with deferred_consolidation(store):
            append_zarr(_piece(2.0), store, group="g", consolidate=False, mode="w")
        assert not zarr_store_has_index(store), "the inner exit must not rebuild the index"
        assert Path(store).resolve() in _DEFERRED_STORES, "the inner exit must not discard the store from the open set"
        append_zarr(_piece(3.0), store, group="h", consolidate=False, mode="w")
    assert zarr_store_has_index(store), "the outer exit rebuilds the index once"
    assert Path(store).resolve() not in _DEFERRED_STORES
    assert float(xr.open_zarr(store, group="g", consolidated=True)["depth"].values.max()) == 2.0

    with deferred_consolidation(store):
        pass
    with deferred_consolidation(store):
        assert not zarr_store_has_index(store)
    assert zarr_store_has_index(store), "control: sequential blocks each rebuild, which a nested no-op must not do"


def _series_source(source: Path, *, indexed: bool) -> None:
    for i, group in enumerate(_SERIES_GROUP_MAP):
        xr.Dataset({"depth": (("event_iloc", "elapsed_s"), np.full((1, 3), float(i + 1)))}).to_zarr(
            source, group=group, mode="a", consolidated=False
        )
    if indexed:
        consolidate_zarr_store(source)


def test_the_series_copy_lands_indexed_is_idempotent_and_leaves_a_refused_source_untouched(tmp_path):
    """Property: ``_copy_series_into_tree`` writes each series group at its branch, records the attribute, leaves the
    tree with a current index, removes the source, and a second run changes no byte of the tree.

    Class: a changed code path (reads through ``open_store``, writes through ``append_zarr`` inside the block).
    Killed by: a copy left outside the block (no index after it, or ``append_zarr`` refused), a copy that omits the
    attribute (the second run re-copies and changes the signature), a copy that does not remove the source, and a
    source with no index being used as a copy source (the tree must be left untouched when the source is refused).
    Must-fire control: the tree signature reds an added file, and an UNINDEXED source is refused with the tree
    unchanged.
    """
    store, source = tmp_path / "analysis_datatree.zarr", tmp_path / "analysis_timeseries.zarr"
    xr.Dataset(attrs={"seed": 1}).to_zarr(store, mode="w", consolidated=True)
    _series_source(source, indexed=True)

    _copy_series_into_tree(store, source)

    assert zarr_store_has_index(store), "the copy rebuilds the index at the block exit"
    assert utils.zarr_root_attrs(store).get(SERIES_MIGRATED_ATTR) == "done"
    assert not source.exists(), "the source is removed once the copy is recorded"
    for i, dst in enumerate(_SERIES_GROUP_MAP.values()):
        copied = xr.open_zarr(store, group=dst, consolidated=True)
        assert float(copied["depth"].values.max()) == float(i + 1), dst
    after_first = _signature(store)

    _copy_series_into_tree(store, source)
    assert _signature(store) == after_first, "a second run changes no byte of the tree"
    (store / "stray").write_bytes(b"x")
    assert _signature(store) != after_first, "control: the signature sees an added file"
    (store / "stray").unlink()

    other_store, other_source = tmp_path / "t.zarr", tmp_path / "u.zarr"
    xr.Dataset(attrs={"seed": 2}).to_zarr(other_store, mode="w", consolidated=True)
    _series_source(other_source, indexed=False)
    before = _signature(other_store)
    with pytest.raises(ValueError, match="consolidated index"):
        _copy_series_into_tree(other_store, other_source)
    assert _signature(other_store) == before, "a refused source leaves the tree untouched"
    assert other_source.exists(), "a refused source is kept"


def test_a_copy_whose_read_back_differs_keeps_the_source_and_records_nothing(tmp_path, monkeypatch):
    """Property: when the copy does not read back equal, the source is kept and the done attribute is not written.

    Killed by: a copy that records the attribute or removes the source before the read-back is compared.
    Must-fire control: with the comparison restored the same inputs record the attribute and remove the source.
    """
    store, source = tmp_path / "analysis_datatree.zarr", tmp_path / "analysis_timeseries.zarr"
    xr.Dataset(attrs={"seed": 1}).to_zarr(store, mode="w", consolidated=True)
    _series_source(source, indexed=True)
    real = migration_context._read_group_equal
    calls = {"n": 0}

    def unequal_on_read_back(a, b):
        calls["n"] += 1
        return False if calls["n"] == 1 else real(a, b)

    monkeypatch.setattr(migration_context, "_read_group_equal", unequal_on_read_back)
    with pytest.raises(ValueError, match="does not read back equal"):
        _copy_series_into_tree(store, source)
    assert source.exists(), "the source is kept"
    assert utils.zarr_root_attrs(store).get(SERIES_MIGRATED_ATTR) != "done"

    monkeypatch.setattr(migration_context, "_read_group_equal", real)
    _copy_series_into_tree(store, source)
    assert utils.zarr_root_attrs(store).get(SERIES_MIGRATED_ATTR) == "done", "control: the true comparison records it"
    assert not source.exists()


def _printed_positions(source: str, word: str) -> list[tuple[int, str]]:
    """The ``(line, parent-node type)`` of each non-docstring string constant of ``source`` that holds ``word``, for
    those whose position can be printed: an f-string part, a call argument other than ``Path``, a ``raise`` message."""
    tree = ast.parse(source)
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str) and word in node.value.split("/")):
            continue
        holder = parent[node]
        if isinstance(holder, ast.Expr):
            continue
        is_path_segment = (
            (isinstance(holder, ast.BinOp) and isinstance(holder.op, ast.Div))
            or isinstance(holder, ast.Compare)
            or (isinstance(holder, ast.Call) and isinstance(holder.func, ast.Name) and holder.func.id == "Path")
        )
        if not is_path_segment:
            found.append((node.lineno, type(holder).__name__))
    return found


def test_the_sims_allowlist_row_covers_only_path_segments_that_no_page_prints():
    """Property: every non-docstring ``sims`` constant of ``cross_sim_identity.py`` is a path segment (a ``Path`` join,
    a ``Path(...)`` argument or a comparison), which is the permitted category of a stored directory name.

    Killed by: a message that prints the directory word (``raise ValueError("no sims found")``), which the position
    classifier reports. Must-fire control: the classifier returns the printed position of exactly that synthetic source
    and returns nothing for a path join, while the real file's path segments are found (three, as the row counts).
    """
    printed = _printed_positions('def f():\n    raise ValueError("sims")\n', "sims")
    assert printed == [(2, "Call")], "control: a message argument is reported as a printed position"
    assert _printed_positions('from pathlib import Path\nx = Path("a") / "sims"\n', "sims") == []

    source = IDENTITY_FILE.read_text(encoding="utf-8")
    assert _printed_positions(source, "sims") == [], "no `sims` constant of the file can reach a page"
    tree = ast.parse(source)
    segments = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "sims"]
    assert len(segments) == 3, "control: the three path segments the allowlist row counts exist"
