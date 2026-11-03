"""Behaviour of the toolkit's zarr write helpers and the consolidated-metadata assumption they rest on.

Every zarr store the toolkit writes goes through ``utils.write_zarr``,
``utils.write_datatree_zarr`` or ``utils.append_zarr`` and carries consolidated metadata
(approved shape D13 items 1 and 2, chunk 3a.1). This module pins what those helpers
promise: an append of a NEW group leaves the group in the index, the three helpers pass
``zarr_format`` and ``encoding`` through and shard nothing on their own, an append along a
dimension never extends a sharded array by a partial shard, a chapter set written with
``shards`` merges to a store carrying the same chunk and shard shapes (approved shape
5.7, 3a, retrofit-neutral checks (a) to (d); OI-17), and the write path does not emit the
zarr format 3 consolidated-metadata warning.

No test here compiles or runs a solver. Each builds a few kilobytes of data under
``tmp_path``.

Not established here: behaviour on a store larger than a few chunks, or on an object store;
the index's effect on read cost is measured by the cluster jobs of the approved shape, not
by these tests.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from hhemt.exceptions import ProcessingError
from hhemt.utils import (
    append_zarr,
    chapter_flag_for,
    chapter_store_for,
    consolidate_zarr_store,
    deferred_consolidation,
    invalidate_consolidated_index,
    merge_chapters_to_unified,
    reap_unflagged_chapters,
    write_datatree_zarr,
    write_zarr,
)

_CONSOLIDATED_WARNING = "Consolidated metadata is currently not part in the Zarr format 3 specification"


def _indexed_nodes(store: Path) -> set[str] | None:
    """Node names the root ``zarr.json`` of a format 3 store lists in its consolidated index, or None if it has none."""
    consolidated = json.loads((store / "zarr.json").read_text()).get("consolidated_metadata")
    return None if consolidated is None else set(consolidated["metadata"])


def _on_disk_nodes(store: Path) -> set[str]:
    """Every group or array that has its own ``zarr.json`` below the root, as a slash-joined path."""
    return {str(p.parent.relative_to(store)) for p in store.rglob("zarr.json") if p.parent != store}


def _grid(store: Path, name: str) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """(inner chunk shape, shard shape or None) that the store records for array ``name``."""
    array = zarr.open_group(str(store), mode="r")[name]
    return tuple(array.chunks), None if array.shards is None else tuple(array.shards)


def _small_dataset(n: int = 5, start: int = 0) -> xr.Dataset:
    return xr.Dataset({"a": (("x",), np.arange(float(n)))}, coords={"x": np.arange(start, start + n)})


def test_append_of_a_new_group_through_append_zarr_is_in_the_index(tmp_path):
    """An append of a NEW group through ``append_zarr`` leaves that group in the root index.

    Property: after ``append_zarr`` adds a group, every node on disk is listed in the index.
    Class: NEW CAPABILITY -- the plausible wrong implementation is a writer that appends with
    ``consolidated=False`` and never rebuilds the index; it is red under that implementation
    (control below). Why required: a group missing from the index is invisible to a reader
    that trusts the index, with no error. This is the assumption approved shape D13 rests on.
    Kills it: dropping ``consolidated=True`` from the ``to_zarr`` call and the trailing
    ``consolidate_zarr_store`` call in ``append_zarr``. A second correct implementation:
    an ``append_zarr`` that passes ``consolidated=False`` to xarray and calls
    ``consolidate_zarr_store`` once afterwards still passes.
    """
    store = tmp_path / "system.zarr"
    write_zarr(_small_dataset(), store, chunks=None, encoding={})
    assert _indexed_nodes(store) == {"a", "x"}, (
        "the full write did not carry an index; the append below would prove nothing"
    )

    append_zarr(_small_dataset(), store, group="analysis_1/tritonswmm")
    append_zarr(_small_dataset(), store, group="analysis_2")

    on_disk = _on_disk_nodes(store)
    assert {"analysis_1", "analysis_1/tritonswmm", "analysis_2"} <= on_disk
    assert _indexed_nodes(store) == on_disk, (
        f"index omits {sorted(on_disk - (_indexed_nodes(store) or set()))}; "
        "a reader that trusts the index never sees them"
    )
    tree = xr.open_datatree(store, engine="zarr", consolidated=True)
    assert "analysis_1/tritonswmm" in {p.lstrip("/") for p in tree.groups}

    # Must-fire control, varying the PROPERTY (is a new group indexed) and not the token: the
    # same append through plain xarray with consolidated=False leaves it out of the index.
    control = tmp_path / "control.zarr"
    write_zarr(_small_dataset(), control, chunks=None, encoding={})
    _small_dataset().to_zarr(control, group="analysis_1/tritonswmm", mode="a", consolidated=False)
    assert "analysis_1/tritonswmm" not in (_indexed_nodes(control) or set()), (
        "control: an unconsolidated append was indexed"
    )


def test_append_zarr_indexes_a_store_that_has_no_index_yet(tmp_path):
    """``append_zarr`` onto a store written before consolidated metadata existed produces a complete index.

    Property: a legacy store with no index acquires one that lists every node. Class: NEW
    CAPABILITY -- plausible wrong implementation: an ``append_zarr`` that opens the target with
    ``consolidated=True`` and fails when the store has none (the migration of every legacy
    tree writes through this path). Why: the layout migration must be able to index legacy
    stores. Kills it: passing ``consolidated=True`` where the opener cannot find an index.
    A second correct implementation: building the index with ``consolidate_zarr_store`` only.
    """
    store = tmp_path / "legacy.zarr"
    _small_dataset().to_zarr(store, mode="w", consolidated=False)
    _small_dataset().to_zarr(store, group="g1", mode="a", consolidated=False)
    assert _indexed_nodes(store) is None

    append_zarr(_small_dataset(), store, group="g2")

    assert _indexed_nodes(store) == _on_disk_nodes(store)
    assert {"g1", "g1/a", "g2", "g2/a"} <= _indexed_nodes(store)


def test_append_zarr_replaces_a_group_in_mode_w_and_keeps_the_index_current(tmp_path):
    """``mode="w"`` through ``append_zarr`` replaces one group and the index follows the replacement.

    Property: after replacing group ``g`` the index lists ``g``'s NEW variable and not its old
    one. Class: REGRESSION -- ``version_migration`` renames a variable by rewriting its group
    this way and re-consolidating (the behaviour being preserved). Why: a stale index would
    serve the old variable name. Kills it: skipping the re-consolidation after the write.
    A second correct implementation: ``consolidate_zarr_store`` called by the caller instead.
    """
    store = tmp_path / "s.zarr"
    write_zarr(_small_dataset(), store, chunks=None, encoding={})
    append_zarr(_small_dataset(), store, group="g")
    replacement = xr.Dataset({"b": (("x",), np.arange(3.0))}, coords={"x": np.arange(3)})

    append_zarr(replacement, store, group="g", mode="w")

    indexed = _indexed_nodes(store)
    assert "g/b" in indexed
    assert "g/a" not in indexed
    assert indexed == _on_disk_nodes(store)


def test_append_zarr_refuses_to_be_told_not_to_consolidate(tmp_path):
    """``append_zarr`` raises ``TypeError`` when handed ``consolidated``.

    Property: the helper has no way to write an unindexed store. Class: NEW CAPABILITY --
    plausible wrong implementation: forwarding ``**kw`` so ``consolidated=False`` reaches
    xarray. Why: the single-write-path rule is only as strong as the helper's refusal.
    Kills it: removing the guard. A second correct implementation: dropping ``consolidated``
    from the keyword set altogether, so Python itself raises ``TypeError``.
    """
    with pytest.raises(TypeError, match="consolidated"):
        append_zarr(_small_dataset(), tmp_path / "s.zarr", group="g", consolidated=False)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_helpers_pass_zarr_format_and_encoding_through_and_shard_nothing(tmp_path, zarr_format):
    """The three helpers accept ``zarr_format`` and ``encoding``, index the store, and set no ``shards``.

    Property, for each of ``write_zarr``, ``write_datatree_zarr`` and ``append_zarr``: the
    store's format equals the requested one, the stated chunk shape reaches the array, no array
    is sharded, and the store carries an index. Class: REGRESSION -- today's stores are written
    unsharded (the slurm store is format 2) and must stay so (shape 5.7, 3a check (b)). Why: a
    later sharding retrofit sets shards deliberately; a helper that sharded by default would
    decide it silently. Kills it: a default ``shards`` entry in any helper's encoding, or
    dropping ``zarr_format`` on the way to xarray. A second correct implementation: a helper
    that states ``shards=None`` explicitly passes as well, because None is "not sharded".
    """
    data = xr.Dataset({"v": (("t",), np.arange(8.0))}, coords={"t": np.arange(8)})
    encoding = {"v": {"chunks": (4,)}}

    flat = tmp_path / "flat.zarr"
    write_zarr(data, flat, chunks=None, encoding=encoding, zarr_format=zarr_format)

    tree_store = tmp_path / "tree.zarr"
    write_datatree_zarr(
        xr.DataTree.from_dict({"/n": data}), tree_store, encoding={"/n": encoding}, zarr_format=zarr_format
    )

    appended = tmp_path / "appended.zarr"
    write_zarr(data, appended, chunks=None, encoding=encoding, zarr_format=zarr_format)
    append_zarr(data, appended, group="g", encoding=encoding, zarr_format=zarr_format)

    for store, array_path in ((flat, "v"), (tree_store, "n/v"), (appended, "v"), (appended, "g/v")):
        group = zarr.open_group(str(store), mode="r")
        array = group[array_path]
        assert array.metadata.zarr_format == zarr_format, (store.name, array_path)
        assert tuple(array.chunks) == (4,), (store.name, array_path, array.chunks)
        assert array.shards is None, (store.name, array_path, "sharded without being asked to be")
        marker = store / ("zarr.json" if zarr_format == 3 else ".zmetadata")
        assert marker.exists(), f"{store.name} carries no consolidated index document ({marker.name})"
        if zarr_format == 3:
            assert json.loads(marker.read_text()).get("consolidated_metadata") is not None, store.name


def test_write_helpers_do_not_emit_the_consolidated_metadata_warning(tmp_path):
    """No write helper lets zarr's format 3 consolidated-metadata ``ZarrUserWarning`` out.

    Property: the three writers and ``consolidate_zarr_store`` emit no warning whose message
    names the consolidated-metadata specification gap. Class: NEW CAPABILITY -- plausible
    wrong implementation: passing ``consolidated=True`` with no filter, which warns on every
    write. Why: the index is deliberate and the warning carries no information; left in, it
    buries warnings that do. Must-fire control: a bare ``to_zarr(consolidated=True)`` DOES emit
    it, so the empty result below cannot be a matcher that never matches. Kills it: removing
    ``_silence_zarr_write_warnings`` from a helper. A second correct implementation: a
    ``simplefilter`` installed once at import would pass as well.
    """
    data = _small_dataset()

    def _messages(action) -> list[str]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            action()
        return [str(w.message) for w in caught if _CONSOLIDATED_WARNING in str(w.message)]

    control = tmp_path / "control.zarr"
    assert _messages(lambda: data.to_zarr(control, mode="w", consolidated=True)), (
        "control: bare consolidated=True did not warn, so this test cannot discriminate"
    )

    flat = tmp_path / "flat.zarr"
    tree_store = tmp_path / "tree.zarr"
    assert _messages(lambda: write_zarr(data, flat, chunks=None, encoding={})) == []
    assert _messages(lambda: append_zarr(data, flat, group="g")) == []
    assert _messages(lambda: write_datatree_zarr(xr.DataTree.from_dict({"/n": data}), tree_store, encoding={})) == []
    assert _messages(lambda: consolidate_zarr_store(flat)) == []


def _write_sharded_chapters(root: Path, n_chapters: int = 2, n_t: int = 4, ny: int = 8, nx: int = 10) -> None:
    """Flagged chapter stores whose variable is sharded: inner chunks (2, 4, 5) inside shards (4, 8, 10)."""
    root.mkdir(parents=True, exist_ok=True)
    for index in range(n_chapters):
        first = index * n_t
        ds = xr.Dataset(
            {
                "wlevel_m": (
                    ("timestep_min", "y", "x"),
                    np.random.default_rng(index).random((n_t, ny, nx)).astype("float32"),
                )
            },
            coords={"timestep_min": np.arange(first, first + n_t), "y": np.arange(ny), "x": np.arange(nx)},
        )
        encoding = {"wlevel_m": {"chunks": (2, 4, 5), "shards": (4, 8, 10)}}
        ds.chunk({"timestep_min": n_t, "y": ny, "x": nx}).to_zarr(
            chapter_store_for(root, index), mode="w", encoding=encoding, consolidated=False
        )
        chapter_flag_for(root, index).write_text("ok", encoding="utf-8")


def test_a_chapter_set_written_with_shards_merges_to_a_store_with_the_same_chunks_and_shards(tmp_path):
    """Merging sharded chapters publishes a store whose chunk shape and shard shape equal the chapters'.

    Property: ``(inner chunks, shards)`` of the unified array equals the chapters' pair, and the
    data is byte-for-byte the concatenation. Class: NEW CAPABILITY -- plausible wrong
    implementation: a merge that states no shard setting (so the store is unsharded) or one that
    sizes dask chunks to the inner chunk and raises; red under both (control and characterization
    below). Why: the developer's constraint is that nothing in this run blocks a later sharding
    retrofit (approved shape D14, 5.7 check (a)). Kills it: dropping the ``shards`` key from the
    merge's stated encoding, or the per-variable shard-sized rechunk. A second correct
    implementation: a merge that rechunks every axis to the shard shape for all variables rather
    than only the sharded ones passes too.
    """
    chapters = tmp_path / "tseries.zarr.chapters"
    _write_sharded_chapters(chapters)
    chapter_grid = _grid(chapter_store_for(chapters, 0), "wlevel_m")
    assert chapter_grid == ((2, 4, 5), (4, 8, 10)), "the fixture did not write a sharded chapter"
    final = tmp_path / "tseries.zarr"

    merge_chapters_to_unified(chapters, final, scenario_dir=chapters.parent)

    assert _grid(final, "wlevel_m") == chapter_grid, (
        f"merged grid {_grid(final, 'wlevel_m')} differs from the chapters' {chapter_grid}"
    )
    merged = xr.open_zarr(final, consolidated=True)
    assert merged.sizes["timestep_min"] == 8
    expected = np.concatenate([np.random.default_rng(i).random((4, 8, 10)).astype("float32") for i in range(2)])
    np.testing.assert_array_equal(merged["wlevel_m"].values, expected)
    assert _indexed_nodes(final) == _on_disk_nodes(final)


def test_the_comparison_used_above_detects_a_merge_that_drops_shards(tmp_path):
    """Must-fire control for the sharded merge: a store written without ``shards`` compares unequal.

    Property: the ``(chunks, shards)`` pair separates a sharded store from an unsharded one
    that has the same inner chunk shape. Class: NEW CAPABILITY -- plausible wrong
    implementation: a comparison that looks at ``chunks`` alone, which would pass a merge that
    drops shards. Why: without this, the equality in the previous test could be satisfied by
    the very defect it exists to catch. Also characterizes the failure the explicit encoding
    replaces: written the old way (inherited encoding, dask grid at the inner chunk) the sharded
    chapters raise. Kills it: nothing in production; it kills a weakened comparison. A second
    correct implementation of the control: any unsharded write with chunks (2, 4, 5).
    """
    chapters = tmp_path / "chapters"
    _write_sharded_chapters(chapters)
    chapter_grid = _grid(chapter_store_for(chapters, 0), "wlevel_m")

    chapter = xr.open_zarr(chapter_store_for(chapters, 0), consolidated=False)
    dropped = tmp_path / "dropped.zarr"
    chapter.chunk({"timestep_min": 2, "y": 4, "x": 5}).to_zarr(
        dropped, mode="w", encoding={"wlevel_m": {"chunks": (2, 4, 5)}}, consolidated=False
    )
    assert _grid(dropped, "wlevel_m") == ((2, 4, 5), None)
    assert _grid(dropped, "wlevel_m") != chapter_grid

    old_way = chapter.chunk({"timestep_min": 2})
    with pytest.raises(ValueError, match="overlap multiple Dask chunks"):
        old_way.to_zarr(tmp_path / "old_way.zarr", mode="w", consolidated=False)


def test_merge_of_unsharded_chapters_states_the_chunks_it_inherits(tmp_path):
    """An unsharded merge still publishes the inherited chunk shape, with no shards and a current index.

    Property: for chapters at inner chunks (2, 4, 5) the unified array has exactly that grid
    and ``shards`` is None. Class: REGRESSION -- the explicit encoding the merge now states must
    reproduce what inheritance produced (shape 5.7, 3a check (a)). Why: a merge that changed
    today's unsharded grid would alter every store written from now on. Kills it: stating a
    different default chunk or a shard shape in the merge. A second correct implementation:
    inheriting the encoding through xarray while also stating shards only when present.
    """
    chapters = tmp_path / "tseries.zarr.chapters"
    chapters.mkdir()
    for index in range(2):
        ds = xr.Dataset(
            {"wlevel_m": (("timestep_min", "y", "x"), np.zeros((4, 8, 10), dtype="float32"))},
            coords={"timestep_min": np.arange(index * 4, index * 4 + 4), "y": np.arange(8), "x": np.arange(10)},
        )
        ds["wlevel_m"].encoding["chunks"] = (2, 4, 5)
        ds.to_zarr(chapter_store_for(chapters, index), mode="w", consolidated=False)
        chapter_flag_for(chapters, index).write_text("ok", encoding="utf-8")
    final = tmp_path / "tseries.zarr"

    merge_chapters_to_unified(chapters, final, scenario_dir=chapters.parent)

    assert _grid(final, "wlevel_m") == ((2, 4, 5), None)
    assert _indexed_nodes(final) == _on_disk_nodes(final)


def _sharded_target(tmp_path: Path) -> Path:
    """A store with one array sharded 4 along ``timestep_min`` and four timesteps already written."""
    store = tmp_path / "sharded.zarr"
    ds = xr.Dataset(
        {"v": (("timestep_min", "y"), np.zeros((4, 8), dtype="float32"))},
        coords={"timestep_min": np.arange(4), "y": np.arange(8)},
    )
    write_zarr(
        ds.chunk({"timestep_min": 4, "y": 8}),
        store,
        chunks=None,
        encoding={"v": {"chunks": (2, 4), "shards": (4, 8)}},
    )
    assert _grid(store, "v") == ((2, 4), (4, 8))
    return store


def _next_block(start: int, n: int) -> xr.Dataset:
    return xr.Dataset(
        {"v": (("timestep_min", "y"), np.ones((n, 8), dtype="float32"))},
        coords={"timestep_min": np.arange(start, start + n), "y": np.arange(8)},
    ).chunk({"timestep_min": n, "y": 8})


def test_append_along_a_dimension_into_a_sharded_array_needs_a_whole_number_of_shards(tmp_path):
    """``append_dim`` into a sharded array is refused unless the appended extent is a multiple of the shard length.

    Property: extent 3 against shard length 4 raises ``ProcessingError`` naming the variable and
    leaves the store unchanged; extent 4 appends and the array grows to 8. Class: NEW
    CAPABILITY -- plausible wrong implementation: forwarding ``append_dim`` unchecked, which
    rewrites the last shard's existing bytes in place (shape 5.7, 3a check (c)). Must-fire
    control: the aligned append in the same test succeeds, so the refusal is keyed on
    alignment and not on ``append_dim`` itself. Kills it: removing ``_refuse_unaligned_shard_append``.
    A second correct implementation: refusing any append into a sharded array at all would fail
    the aligned case, so the shard-multiple rule is what is pinned, not "always refuse".
    """
    store = _sharded_target(tmp_path)
    before = _indexed_nodes(store)

    with pytest.raises(ProcessingError, match="'v'.*not a multiple"):
        append_zarr(_next_block(4, 3), store, group=None, append_dim="timestep_min")
    assert zarr.open_group(str(store), mode="r")["v"].shape == (4, 8)
    assert _indexed_nodes(store) == before

    append_zarr(_next_block(4, 4), store, group=None, append_dim="timestep_min")
    assert zarr.open_group(str(store), mode="r")["v"].shape == (8, 8)
    assert _grid(store, "v") == ((2, 4), (4, 8))


def test_append_along_a_dimension_into_an_unsharded_array_is_not_restricted(tmp_path):
    """An unsharded target accepts any appended extent through ``append_dim``.

    Property: extent 3 appended to an unsharded array succeeds. Class: REGRESSION for the shard
    rule's scope -- the refusal must not reach arrays that are not sharded. Why: an over-broad
    refusal would break every future append. Kills it: raising whenever ``append_dim`` is set.
    A second correct implementation: any check that reads ``array.shards`` and acts only on a
    non-None value.
    """
    store = tmp_path / "plain.zarr"
    ds = xr.Dataset({"v": (("t",), np.arange(4.0))}, coords={"t": np.arange(4)})
    write_zarr(ds, store, chunks=None, encoding={"v": {"chunks": (2,)}})

    append_zarr(
        xr.Dataset({"v": (("t",), np.arange(3.0))}, coords={"t": np.arange(4, 7)}),
        store,
        group=None,
        append_dim="t",
    )

    assert zarr.open_group(str(store), mode="r")["v"].shape == (7,)


# ---- the deferred block (approved shape D13A, option B) ------------------------------------------------


def _spy_on_consolidation(monkeypatch) -> list[str]:
    """Record every ``zarr.consolidate_metadata`` call, xarray's and hhemt's alike, while still making it."""
    calls: list[str] = []
    original = zarr.consolidate_metadata

    def spy(store, *args, **kwargs):
        calls.append(str(store))
        return original(store, *args, **kwargs)

    monkeypatch.setattr(zarr, "consolidate_metadata", spy)
    return calls


def _indexed_store(tmp_path: Path, name: str = "system.zarr", groups: tuple[str, ...] = ("g1", "g2")) -> Path:
    store = tmp_path / name
    write_zarr(_small_dataset(), store, chunks=None, encoding={})
    for group in groups:
        append_zarr(_small_dataset(), store, group=group)
    assert _on_disk_nodes(store) <= _indexed_nodes(store)
    return store


def test_n_appends_inside_one_deferred_block_consolidate_once(tmp_path, monkeypatch):
    """N appends inside one ``deferred_consolidation`` block call ``zarr.consolidate_metadata`` exactly once.

    Property: the count of consolidations per block is one, whatever N is. Class: NEW CAPABILITY --
    plausible wrong implementation: a block whose appends still consolidate per call (N calls), the
    quadratic cost approved shape D13A removes. Must-fire control in the same test: the same N appends
    outside a block, each consolidating, call it N times, so the spy is counting what it claims to.
    Why: the cost of N consolidating appends grows with the square of N (the build's laptop-disk
    measurement: 200 appends 1.5 s unconsolidated against 20.7 s consolidating). Kills it: passing
    ``consolidate=True`` inside the block or consolidating in ``append_zarr``. A second correct
    implementation: a block that rebuilds through ``zarr.consolidate_metadata`` directly.
    """
    n = 12
    store = tmp_path / "block.zarr"
    calls = _spy_on_consolidation(monkeypatch)
    with deferred_consolidation(store):
        for i in range(n):
            append_zarr(_small_dataset(), store, group=f"an/leaf{i}", consolidate=False)
    assert len(calls) == 1, f"{len(calls)} consolidations for {n} appends in one block"
    assert _indexed_nodes(store) == _on_disk_nodes(store)

    calls.clear()
    per_call = tmp_path / "per_call.zarr"
    for i in range(n):
        append_zarr(_small_dataset(), per_call, group=f"an/leaf{i}")
    assert len(calls) == n, f"control: {n} consolidating appends made {len(calls)} consolidations"


def test_an_exception_inside_a_block_leaves_no_index_and_an_unindexed_open_lists_every_group(tmp_path):
    """After an interrupted block the root document has no index and an unindexed open sees every group.

    Property: the store left by an exception after k appends has NO ``consolidated_metadata`` and
    ``open_datatree(consolidated=False)`` lists the original groups and the k new ones. Class: NEW
    CAPABILITY -- plausible wrong implementation: a block that skips the per-call consolidation but keeps
    the previous index, which then omits the k new groups while readers trust it. Must-fire control, the
    pattern of V0004 before this change (appends with ``consolidated=False`` and no removal): the same
    interruption leaves an index that lists fewer groups than the store holds. Why: this is the
    crash window D13A closes. Kills it: removing ``invalidate_consolidated_index`` from the block.
    A second correct implementation: a block that deletes the index by unlinking and recreating the root document.
    """
    k = 3
    store = _indexed_store(tmp_path)
    before = _on_disk_nodes(store)
    with pytest.raises(RuntimeError, match="killed"):
        with deferred_consolidation(store):
            for i in range(k):
                append_zarr(_small_dataset(), store, group=f"new{i}", consolidate=False)
            raise RuntimeError("killed mid-block")
    assert _indexed_nodes(store) is None, "an interrupted block left an index"
    tree = xr.open_datatree(store, engine="zarr", consolidated=False)
    listed = {p.strip("/") for p in tree.groups} - {""}
    assert {"g1", "g2", *(f"new{i}" for i in range(k))} <= listed, listed
    assert _on_disk_nodes(store) > before

    control = _indexed_store(tmp_path, name="control.zarr")
    with pytest.raises(RuntimeError, match="killed"):
        for i in range(k):
            _small_dataset().to_zarr(control, group=f"new{i}", mode="a", consolidated=False)
        raise RuntimeError("killed mid-loop")
    assert _indexed_nodes(control) is not None
    assert not {f"new{i}" for i in range(k)} <= _indexed_nodes(control), (
        "control: the old pattern's index already lists the new groups"
    )


def test_a_group_removed_inside_a_block_is_gone_from_the_rebuilt_index(tmp_path):
    """A block whose body removes a group (as the system build's ``rmtree`` does) rebuilds an index without it.

    Property: after the block the index equals the on-disk node set, the removed group is absent and
    the appended one present. Class: NEW CAPABILITY -- plausible wrong implementation: a block that
    keeps the old index and adds to it, which would still list the removed group. Why: the system build
    deletes an analysis group before rewriting it. Kills it: skipping the invalidation or the rebuild.
    A second correct implementation: a rebuild that reads the store listing through zarr's own walk.
    """
    store = _indexed_store(tmp_path)
    assert "g1" in _indexed_nodes(store)
    with deferred_consolidation(store):
        shutil.rmtree(store / "g1")
        append_zarr(_small_dataset(), store, group="g3", consolidate=False)
    indexed = _indexed_nodes(store)
    assert "g1" not in indexed and "g1/a" not in indexed
    assert {"g2", "g3"} <= indexed
    assert indexed == _on_disk_nodes(store)


def test_append_with_consolidate_false_outside_a_block_raises(tmp_path):
    """``append_zarr(..., consolidate=False)`` raises ``ProcessingError`` unless a block is open for THAT store.

    Property: no unindexed append is reachable without the block. Class: NEW CAPABILITY -- plausible
    wrong implementation: honouring ``consolidate=False`` anywhere, which would leave a stale index on
    an interruption. Must-fire: a block open for ANOTHER store does not permit it, and after a block
    closes the permission is gone. Kills it: removing the guard. A second correct implementation:
    keying permission on the store's resolved path or on its string form.
    """
    store = tmp_path / "a.zarr"
    other = tmp_path / "b.zarr"
    with pytest.raises(ProcessingError, match="deferred_consolidation"):
        append_zarr(_small_dataset(), store, group="g", consolidate=False)
    with deferred_consolidation(other):
        with pytest.raises(ProcessingError):
            append_zarr(_small_dataset(), store, group="g", consolidate=False)
    with pytest.raises(ProcessingError):
        append_zarr(_small_dataset(), other, group="g", consolidate=False)
    assert not store.exists()


def test_invalidate_removes_only_the_index_atomically(tmp_path):
    """``invalidate_consolidated_index`` drops ``consolidated_metadata`` from the root document and nothing else.

    Property: the document after equals the document before minus the key, no ``zarr.json.partial``
    is left, and a second call is a no-op. Class: NEW CAPABILITY -- plausible wrong implementation:
    writing the document in place (a kill mid-write leaves a truncated root document that no reader
    can open) or dropping other keys. Kills it: writing without ``os.replace``. A second correct
    implementation: building the new document in memory and replacing through a temporary directory.
    """
    store = _indexed_store(tmp_path)
    before = json.loads((store / "zarr.json").read_text())
    assert "consolidated_metadata" in before
    invalidate_consolidated_index(store)
    after = json.loads((store / "zarr.json").read_text())
    assert after == {k: v for k, v in before.items() if k != "consolidated_metadata"}
    assert not (store / "zarr.json.partial").exists()
    invalidate_consolidated_index(store)
    assert json.loads((store / "zarr.json").read_text()) == after
    invalidate_consolidated_index(tmp_path / "does_not_exist.zarr")


def test_reaping_removes_an_interrupted_chapter_write_and_keeps_a_flagged_chapter(tmp_path):
    """The reap discards a ``chapter_*.zarr.tmp`` left by an interrupted publish and keeps flagged chapters.

    Property: after the reap no ``.zarr.tmp`` sibling remains and the flagged chapter store and its flag
    are untouched. Class: NEW CAPABILITY -- a chapter is now written through ``write_zarr`` (stage a
    ``.tmp`` sibling, rename into place), and the previous reap glob ``chapter_*.zarr`` never matched
    that sibling, so an interrupted write left its bytes until the chapters directory was deleted.
    Plausible wrong implementation: widening the glob to ``chapter_*`` and deleting a flagged chapter's
    store (the SOLE copy after a raw clear); the flagged-chapter assertion is red under it. Kills it:
    removing the ``.zarr.tmp`` loop. A second correct implementation: reaping the ``.tmp`` inside
    ``write_zarr`` before staging.
    """
    chapters = tmp_path / "tseries.zarr.chapters"
    chapters.mkdir()
    flagged = chapter_store_for(chapters, 0)
    write_zarr(_small_dataset(), flagged, chunks=None, encoding={})
    chapter_flag_for(chapters, 0).write_text("ok", encoding="utf-8")
    partial = chapters / "chapter_00001.zarr.tmp"
    write_zarr(_small_dataset(), partial, chunks=None, encoding={})
    assert partial.exists()

    with pytest.warns(UserWarning, match="interrupted chapter write"):
        reap_unflagged_chapters(chapters, scenario_dir=tmp_path)

    assert not partial.exists()
    assert flagged.exists() and chapter_flag_for(chapters, 0).exists()
