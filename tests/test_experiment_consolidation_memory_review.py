"""Reviewer tests for change 1118-mem: the lazy ``_copy_series_group`` against the eager one it replaced.

Chunk test, authored by the reviewing seat; it proves one change (1118-mem) once and guards
``sensitivity_analysis._copy_series_group`` and the master write that calls it. At the landing it is promoted into
the module test of ``sensitivity_analysis`` or deleted with its passing evidence kept in the run record.

The oracle is ``_eager_copy_series_group`` below: the function exactly as it stood before the change (``origin/develop``
``b3b45cfd``), with only the two lines the change replaced restored (``chunks=None`` and ``.load()``). The tests build
small synthetic member stores for every series group the consolidation copies (SWMM node, SWMM link, gridded
TRITON, performance, observation points), run the oracle and the lazy copy side by side, and compare what each STORED
(read back raw from zarr: values, dtypes, dimension names, attributes, fill values, codecs and chunk layout).
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from hhemt import sensitivity_analysis as sa
from hhemt.exceptions import ProcessingError
from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as Post
from hhemt.utils import deferred_consolidation, zarr_store_has_index

# --------------------------------------------------------------------------------------------------------------------
# The oracle: the pre-change function, byte-for-byte except the two restored lines (marked).
# --------------------------------------------------------------------------------------------------------------------


def _eager_copy_series_group(
    src_store, src_group: str, dst_store, dst_group: str, compression_level: int, *, member_id: str
) -> None:
    """The pre-change ``_copy_series_group`` (``origin/develop`` b3b45cfd): loads each event's whole slice."""
    from hhemt.utils import append_zarr

    src = xr.open_zarr(
        src_store, group=src_group, consolidated=zarr_store_has_index(src_store), chunks=None, decode_timedelta=True
    )
    for position, event_iloc in enumerate(int(v) for v in src["event_iloc"].values):
        piece = src.isel(event_iloc=[position]).load().drop_encoding()  # restored: .load()
        state = Post._timeseries_group_state(dst_store, dst_group)
        piece = Post._widen_event_strings(piece, dst_group)
        if state is not None:
            piece = Post._fit_strings(dst_store, dst_group, piece)
        with deferred_consolidation(dst_store):
            if state is not None and event_iloc in state[1]:
                if Post._stored_slice_equals(dst_store, dst_group, piece, event_iloc):
                    continue
                at = state[1].index(event_iloc)
                append_zarr(
                    piece.drop_vars([c for c in piece.coords if "event_iloc" not in piece[c].dims]),
                    dst_store,
                    group=dst_group,
                    consolidate=False,
                    mode="r+",
                    region={"event_iloc": slice(at, at + 1)},
                )
            elif state is not None:
                missing = sorted(set(state[0]) - set(piece.data_vars))
                extra = sorted(set(piece.data_vars) - set(state[0]))
                if missing or extra:
                    raise ProcessingError(
                        "append a member's series slice to the sensitivity master",
                        str(dst_store),
                        f"member {member_id} event_iloc {event_iloc} differs from {dst_group} in its variables: "
                        f"missing {missing}, not in the master {extra}; nothing was written",
                    )
                append_zarr(piece, dst_store, group=dst_group, consolidate=False, mode="a", append_dim="event_iloc")
            else:
                append_zarr(piece, dst_store, group=dst_group, consolidate=False, mode="a")
        if not Post._stored_slice_equals(dst_store, dst_group, piece, event_iloc):
            raise ProcessingError("read back", str(dst_store), f"member {member_id}: slice {event_iloc} unequal")


# --------------------------------------------------------------------------------------------------------------------
# Synthetic member series groups, one builder per kind the consolidation copies, in the shapes measured on the Norfolk
# R2a member (dimension order, string coordinates, datetime coordinate, int8 flag, NaN padding of a shorter event).
# --------------------------------------------------------------------------------------------------------------------

KINDS = ("gridded", "swmm_node", "swmm_link", "performance", "observation")
STEPS = 7  # not a multiple of any member chunk below, so the last block of every time chunk is partial
NY, NX = 11, 13  # not multiples of the member chunk below either


def _labels(ds: xr.Dataset, events: int, seed_shift: float) -> xr.Dataset:
    ds = ds.assign_coords(
        event_iloc=np.arange(events, dtype="int64"),
        event_id=("event_iloc", np.array([f"event_index.{i}" for i in range(events)], dtype="<U255")),
        event_start_datetime=(
            "event_iloc",
            (np.datetime64("2025-08-31T00:00:00", "ns") + np.arange(events) * np.timedelta64(1, "D")),
            {},
        ),
        variable_set_across_members=("event_iloc", np.array(["flood_depth"] * events, dtype="<U2048")),
    )
    ds["event_start_datetime"].encoding.update({"units": "days since 2025-08-31 00:00:00", "dtype": "int64"})
    return ds


def _member_group(kind: str, events: int, *, seed: int = 0) -> tuple[xr.Dataset, dict[str, int]]:
    """One member ``timeseries`` group (as an in-memory Dataset) and the dask chunk shape its store is written with."""
    rng = np.random.default_rng(seed)
    t = np.arange(STEPS, dtype="int64") * 120

    def pad(arr: np.ndarray, time_axis: int) -> np.ndarray:
        # The shorter last event: its last two steps are NaN, as a union time axis pads a shorter event.
        if arr.shape[0] > 1:
            index = [slice(None)] * arr.ndim
            index[0] = slice(-1, None)
            index[time_axis] = slice(-2, None)
            arr[tuple(index)] = np.nan
        return arr

    if kind == "gridded":
        shape = (events, STEPS, NY, NX)
        data = {
            n: (("event_iloc", "elapsed_s", "y", "x"), pad(rng.random(shape).astype("float32"), 1), {"units": u})
            for n, u in (("flood_depth", "m"), ("unit_discharge_x", "m2 s-1"))
        }
        ds = xr.Dataset(data, coords={"elapsed_s": t, "y": np.arange(NY) * 1.1, "x": np.arange(NX) * 1.1})
        chunks = {"event_iloc": 1, "elapsed_s": 3, "y": NY, "x": NX}
    elif kind == "swmm_node":
        n = 9
        ds = xr.Dataset(
            {
                "depth": (
                    ("event_iloc", "node_id", "elapsed_s"),
                    pad(rng.random((events, n, STEPS)), 2),
                    {"units": "m"},
                ),
                "outfall_stage_m": (
                    ("event_iloc", "elapsed_s", "node_id"),
                    pad(rng.random((events, STEPS, n)), 1),
                    {"units": "m"},
                ),
                "is_coupling_node": (
                    ("event_iloc", "node_id"),
                    (rng.random((events, n)) > 0.5).astype("int8"),
                    {"units": "1"},
                ),
            },
            coords={"elapsed_s": t, "node_id": np.array([f"J{i}" for i in range(n)], dtype="<U10")},
            attrs={"flow_units": "cms", "sim_date": ""},
        )
        chunks = {"event_iloc": 1, "node_id": n, "elapsed_s": STEPS}
    elif kind == "swmm_link":
        n = 10
        ds = xr.Dataset(
            {
                "flow": (
                    ("event_iloc", "link_id", "elapsed_s"),
                    pad(rng.random((events, n, STEPS)), 2),
                    {"units": "m3 s-1"},
                ),
                "velocity": (
                    ("event_iloc", "link_id", "elapsed_s"),
                    pad(rng.random((events, n, STEPS)), 2),
                    {"units": "m s-1"},
                ),
            },
            coords={"elapsed_s": t, "link_id": np.array([f"C{i}" for i in range(n)], dtype="<U4")},
            attrs={"flow_units": "cms"},
        )
        chunks = {"event_iloc": 1, "link_id": n, "elapsed_s": STEPS}
    elif kind == "performance":
        ds = xr.Dataset(
            {
                name: (("event_iloc", "elapsed_s", "Rank"), pad(rng.random((events, STEPS, 2)), 1), {"units": "s"})
                for name in ("Compute", "Total")
            },
            coords={"elapsed_s": t, "Rank": np.arange(2, dtype="int64")},
        )
        chunks = {"event_iloc": 1, "elapsed_s": STEPS, "Rank": 2}
    elif kind == "observation":
        ds = xr.Dataset(
            {
                "flood_depth_at_point": (
                    ("event_iloc", "elapsed_s", "point"),
                    pad(rng.random((events, STEPS, 3)).astype("float32"), 1),
                    {"units": "m"},
                )
            },
            coords={"elapsed_s": t, "point": np.array(["p0", "p1", "p2"], dtype="<U10")},
        )
        chunks = {"event_iloc": 1, "elapsed_s": STEPS, "point": 3}
    else:  # pragma: no cover - the parameter list is the closed set
        raise AssertionError(kind)
    return _labels(ds, events, 0.0), chunks


def _write_member(path: Path, kind: str, events: int, *, seed: int = 0, group: str = "fam/timeseries") -> str:
    ds, chunks = _member_group(kind, events, seed=seed)
    ds.chunk({k: v for k, v in chunks.items() if k in ds.dims}).to_zarr(path, group=group, mode="w", consolidated=False)
    return group


# --------------------------------------------------------------------------------------------------------------------
# What a store holds, read back raw from zarr.
# --------------------------------------------------------------------------------------------------------------------


def _snapshot(store: Path, group: str) -> dict:
    """Per array of ``group``: shape, dtype, chunk shape, dimension names, attributes, fill value, codecs, values."""
    root = zarr.open_group(store, mode="r")[group]
    out: dict = {"__group_attrs__": dict(root.attrs)}
    for name, arr in sorted(root.arrays(), key=lambda kv: kv[0]):
        attrs, raw = dict(arr.attrs), np.asarray(arr[...])
        time_encoding = None
        if "since" in str(attrs.get("units", "")):
            # A CF time array: compare the DECODED instants; the raw integers and the units string are the encoding.
            time_encoding = (str(arr.dtype), attrs.pop("units"), attrs.pop("calendar", None), raw.flat[0].item())
            raw = xr.coding.times.decode_cf_datetime(raw, time_encoding[1], time_encoding[2])
        out[name] = {
            "shape": arr.shape,
            "dtype": str(arr.dtype) if time_encoding is None else "datetime",
            "chunks": arr.chunks,
            "dims": arr.metadata.dimension_names,
            "attrs": attrs,
            "fill": repr(arr.fill_value),
            "codecs": repr(arr.compressors),
            "values": raw,
            "time_encoding": time_encoding,
        }
    return out


def _same_values(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _assert_same_store(eager: dict, lazy: dict, *, ignore_chunks: bool = False) -> None:
    assert set(eager) == set(lazy), (sorted(eager), sorted(lazy))
    assert eager["__group_attrs__"] == lazy["__group_attrs__"]
    for name in eager:
        if name == "__group_attrs__":
            continue
        e, lz = eager[name], lazy[name]
        for field in ("shape", "dtype", "dims", "attrs", "fill", "codecs") + (() if ignore_chunks else ("chunks",)):
            assert e[field] == lz[field], (name, field, e[field], lz[field])
        assert _same_values(e["values"], lz["values"]), name


@pytest.fixture(autouse=True)
def _small_blocks(monkeypatch):
    """Blocks and the verification span small enough that every store here spans several of each."""
    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 2_000)
    monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", 6_000)


def _copy_both(tmp_path: Path, kind: str, events: int = 3) -> tuple[Path, Path, str, str]:
    src = tmp_path / "member.zarr"
    src_group = _write_member(src, kind, events)
    dst_group = f"member_a/{src_group}"
    eager, lazy = tmp_path / "eager.zarr", tmp_path / "lazy.zarr"
    for store, fn in ((eager, _eager_copy_series_group), (lazy, sa._copy_series_group)):
        with deferred_consolidation(store):
            fn(src, src_group, store, dst_group, 5, member_id="a")
    return eager, lazy, dst_group, src_group


@pytest.mark.parametrize("kind", KINDS)
def test_the_lazy_copy_stores_every_array_as_the_eager_copy_did_except_the_chunk_layout(tmp_path, kind):
    """Property: for each series-group kind the master group holds the same arrays with the same values (NaN padding
    and the partial last block included), dtypes, dimension names, array and group attributes, fill values and codecs
    as the pre-change copy stored.

    Class: REGRESSION against the oracle (the pre-change function is the reference; a lazy copy that dropped an
    attribute, widened a dtype, mis-encoded the datetime coordinate or transposed a dimension reds here).
    Reachability: the real function over a real zarr member store of 3 events, 7 steps (partial blocks), shapes
    taken from the Norfolk member. Both-states anchor: the stored arrays, a property of the copy and not of its
    wording. Must-fire control: ``test_the_comparison_sees_a_changed_value_and_a_changed_attribute`` perturbs a
    stored value and an attribute and requires this comparison to fail. Killed by: a lazy copy that stores
    ``event_start_datetime`` or a string column under another encoding. Passes under a second correct implementation
    that copies by zarr region reads of one chunk row.
    """
    eager, lazy, dst_group, _ = _copy_both(tmp_path, kind)
    _assert_same_store(_snapshot(eager, dst_group), _snapshot(lazy, dst_group), ignore_chunks=True)


@pytest.mark.parametrize("kind", ("swmm_node", "swmm_link", "performance", "observation"))
def test_the_chunk_layout_of_a_non_gridded_group_is_the_one_the_eager_copy_wrote(tmp_path, kind):
    """Property: for the four non-gridded kinds the master's zarr chunk shape of every array equals the pre-change
    copy's.

    Class: REGRESSION against the oracle. Reachability: same real copy as above. Both-states anchor: chunk shape, a
    property of both stores. Must-fire control: the gridded kind below differs, so the instrument reads layouts.
    Killed by: a lazy copy whose dask chunks are not the member's zarr chunks for these kinds. Passes under a second
    correct implementation that sets the encoding chunks explicitly to the same shape.
    """
    eager, lazy, dst_group, _ = _copy_both(tmp_path, kind)
    e, lz = _snapshot(eager, dst_group), _snapshot(lazy, dst_group)
    assert {n: v["chunks"] for n, v in e.items() if n != "__group_attrs__"} == {
        n: v["chunks"] for n, v in lz.items() if n != "__group_attrs__"
    }


def test_the_gridded_chunk_layout_is_the_members_and_not_the_one_the_eager_copy_wrote(tmp_path):
    """Records, as a passing assertion, the one place the stored layout MOVES: a gridded group is stored with the
    member tree's own chunks (here (1, 3, y, x)) where the eager copy stored zarr's default chunk guess.

    Class: CHARACTERIZATION (green before and after this review; it is not a finding on its own). It exists so a
    later change that restores the old layout, or that keeps this one, is a visible edit. Measured on Rivanna the
    same move takes the Norfolk R2a gridded group from one chunk shape (1, 61, 107, 219) to (1, 3, 1707, 1751), and the
    group's size on disk from under 1.52 GB to 2.10 GB (dst_fixed.zarr triton/timeseries, du -sb). Killed by: the
    layout the eager copy chose. Passes under a second correct implementation that stores the member's layout.
    """
    eager, lazy, dst_group, _ = _copy_both(tmp_path, "gridded")
    e, lz = _snapshot(eager, dst_group), _snapshot(lazy, dst_group)
    assert lz["flood_depth"]["chunks"] == (1, 3, NY, NX)
    assert e["flood_depth"]["chunks"] != lz["flood_depth"]["chunks"]


@pytest.mark.parametrize("kind", ("gridded", "swmm_node"))
def test_a_stale_slice_is_rewritten_and_a_new_event_appended_as_the_eager_copy_did(tmp_path, kind):
    """Property: on a second copy after the member's event 0 changed and a third event was added, the master holds the
    member's current values for events 0..N (the stale slice rewritten in place by a region write, the new one
    appended), and the stored arrays equal what the eager copy stores after the same two steps.

    Class: REGRESSION against the oracle over the two write branches the first-copy test does not enter (region
    rewrite and append along ``event_iloc``). Reachability: the real function twice over one master. Both-states
    anchor: stored values. Must-fire control: the rewritten event's values differ from the first copy's before the
    second copy (asserted). Killed by: a region write that mis-aligns lazy chunks against the stored ones. Passes
    under a second correct implementation that rewrites the whole group.
    """
    src = tmp_path / "member.zarr"
    group = _write_member(src, kind, 2)
    snapshots = {}
    for tag, fn in (("eager", _eager_copy_series_group), ("lazy", sa._copy_series_group)):
        dst = tmp_path / f"{tag}.zarr"
        _write_member(src, kind, 2)
        with deferred_consolidation(dst):
            fn(src, group, dst, f"m/{group}", 5, member_id="a")
        before = _snapshot(dst, f"m/{group}")
        _write_member(src, kind, 3, seed=7)  # every event's values change and a third event appears
        with deferred_consolidation(dst):
            fn(src, group, dst, f"m/{group}", 5, member_id="a")
        after = _snapshot(dst, f"m/{group}")
        first_var = next(n for n in after if n not in ("__group_attrs__",) and after[n]["values"].dtype.kind == "f")
        assert after[first_var]["shape"][0] == 3
        assert not _same_values(before[first_var]["values"][0], after[first_var]["values"][0]), "control: no rewrite"
        snapshots[tag] = after
    _assert_same_store(snapshots["eager"], snapshots["lazy"], ignore_chunks=True)


def test_the_comparison_sees_a_changed_value_and_a_changed_attribute(tmp_path):
    """Must-fire control for the equivalence comparison: a stored value changed by one ulp, and an attribute changed,
    each make ``_assert_same_store`` raise, so the passing rows above are not a comparison that sees nothing.

    Class: CONTROL (a property of the instrument). Killed by: a snapshot that drops values or attributes.
    """
    eager, lazy, dst_group, _ = _copy_both(tmp_path, "swmm_node")
    e, lz = _snapshot(eager, dst_group), _snapshot(lazy, dst_group)
    _assert_same_store(e, lz)  # green first: the chunk layout of this kind is equal too
    bumped = {k: dict(v) if isinstance(v, dict) and "values" in v else v for k, v in lz.items()}
    values = bumped["depth"]["values"].copy()
    values.flat[3] = np.nextafter(values.flat[3], np.inf)
    bumped["depth"]["values"] = values
    with pytest.raises(AssertionError):
        _assert_same_store(e, bumped)
    renamed = {k: dict(v) if isinstance(v, dict) and "values" in v else v for k, v in lz.items()}
    renamed["depth"]["attrs"] = {**renamed["depth"]["attrs"], "units": "ft"}
    with pytest.raises(AssertionError):
        _assert_same_store(e, renamed)


# --------------------------------------------------------------------------------------------------------------------
# A master group laid out differently from the member's (a master written by the toolkit before this change).
# --------------------------------------------------------------------------------------------------------------------


def _tree_digest(root: Path) -> dict:
    """Every file under ``root`` with its size and bytes' hash: what a write would change."""
    import hashlib

    return {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(root.rglob("*"))
        # The root document is excluded: the deferred block rebuilds its consolidated-metadata index on exit, which
        # rewrites it whether or not anything was written; every array document and chunk file is compared.
        if f.is_file() and f.relative_to(root) != Path("zarr.json")
    }


def test_a_master_group_written_with_another_chunk_layout_refuses_by_name_before_anything_is_written(tmp_path):
    """Property: appending a new event to, or rewriting a changed slice in, a master gridded group laid out by the
    earlier toolkit raises ``ProcessingError`` naming the member, the group, the variable, both chunk shapes and the
    master, saying it must be rebuilt, and writes NOTHING (the master's files are byte-identical afterwards).

    Class: NEW CAPABILITY (ruling F5 Form b; plausible wrong implementation: refusing after a partial write, or an
    unnamed xarray ValueError; red under it). Reachability: master from the eager copy, member with 3 events (append),
    member with changed values (rewrite). Both-states anchor: the exception type and the master's bytes. Must-fire
    control: the same member copied into a fresh master succeeds (``..._except_the_chunk_layout`` rows). Passes under
    a second correct implementation with another message wording.
    """
    one = tmp_path / "member1.zarr"
    group = _write_member(one, "gridded", 1)
    master = tmp_path / "master.zarr"
    with deferred_consolidation(master):
        _eager_copy_series_group(one, group, master, f"m/{group}", 5, member_id="a")
    stored = zarr.open_group(master, mode="r")[f"m/{group}"]["flood_depth"].chunks
    assert stored != (1, 3, NY, NX), "control: the master must be laid out differently from the member"
    before = _tree_digest(master)
    three = tmp_path / "member3.zarr"
    _write_member(three, "gridded", 3, seed=5)
    with pytest.raises(ProcessingError) as appended:
        with deferred_consolidation(master):
            sa._copy_series_group(three, group, master, f"m/{group}", 5, member_id="a")
    text = str(appended.value)
    for needle in ("member a", f"m/{group}", str(stored), "(1, 3, ", str(master), "rebuilt"):
        assert needle in text, needle
    assert "flood_depth" in text or "unit_discharge_x" in text  # the first refused variable is named
    assert _tree_digest(master) == before, "the append wrote before refusing"
    changed = tmp_path / "member_changed.zarr"
    _write_member(changed, "gridded", 1, seed=99)
    with pytest.raises(ProcessingError):
        with deferred_consolidation(master):
            sa._copy_series_group(changed, group, master, f"m/{group}", 5, member_id="a")
    assert _tree_digest(master) == before, "the rewrite wrote before refusing"


def test_an_equal_slice_in_an_earlier_layout_master_still_verifies_without_a_refusal(tmp_path):
    """Property: copying a member whose slice already sits, equal, in an earlier-layout master completes without a
    refusal and leaves the master byte-identical.

    Class: REGRESSION guard (the refusal must sit after the equal-slice ``continue``). Reachability: the Phase A case,
    a one-event member re-copied into the master the earlier toolkit wrote. Must-fire control: the layouts differ
    (asserted). Killed by: a refusal placed before the equality check. Passes under a second correct implementation
    that verifies by chunk rows.
    """
    one = tmp_path / "member1.zarr"
    group = _write_member(one, "gridded", 1)
    master = tmp_path / "master.zarr"
    with deferred_consolidation(master):
        _eager_copy_series_group(one, group, master, f"m/{group}", 5, member_id="a")
    assert zarr.open_group(master, mode="r")[f"m/{group}"]["flood_depth"].chunks != (1, 3, NY, NX)
    before = _tree_digest(master)
    with deferred_consolidation(master):
        sa._copy_series_group(one, group, master, f"m/{group}", 5, member_id="a")
    after = {k: v for k, v in _tree_digest(master).items() if not k.endswith("zarr.json") or k in before}
    assert {k: v for k, v in after.items() if k in before and not k.endswith("zarr.json")} == {
        k: v for k, v in before.items() if not k.endswith("zarr.json")
    }, "chunk files changed"


# --------------------------------------------------------------------------------------------------------------------
# Memory over members: one master, N members, one gridded group each, copied one after another as the master does.
# --------------------------------------------------------------------------------------------------------------------

EDGE = 96  # one step of one float32 variable is 36 KiB


def _gridded_member(path: Path, steps: int, events: int = 1, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    ds = xr.Dataset(
        {
            n: (("event_iloc", "elapsed_s", "y", "x"), rng.random((events, steps, EDGE, EDGE)).astype("float32"))
            for n in ("flood_depth", "unit_discharge_x")
        },
        coords={"event_iloc": np.arange(events), "elapsed_s": np.arange(steps, dtype="int64") * 60},
    )
    ds["event_id"] = ("event_iloc", np.array([f"event_index.{i}" for i in range(events)], dtype="<U255"))
    ds = ds.set_coords("event_id")
    ds.chunk({"event_iloc": 1, "elapsed_s": 4}).to_zarr(path, group="fam/timeseries", mode="w", consolidated=False)
    return "fam/timeseries"


def _peak_copying_members(tmp_path: Path, members: int, steps: int, monkeypatch) -> tuple[int, int]:
    """Peak traced bytes while ``members`` members' gridded groups are copied in turn into one master, and the bytes
    of one member's one-event slice."""
    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 1 << 18)
    monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", 1 << 20)
    src = tmp_path / f"src_{members}_{steps}.zarr"
    group = _gridded_member(src, steps)
    master = tmp_path / f"master_{members}_{steps}.zarr"
    slice_bytes = 2 * steps * EDGE * EDGE * 4
    tracemalloc.start()
    try:
        with deferred_consolidation(master):
            tracemalloc.reset_peak()
            for i in range(members):
                sa._copy_series_group(src, group, master, f"member_{i}/{group}", 5, member_id=str(i))
            peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak, slice_bytes


def test_the_peak_of_copying_n_members_one_after_another_does_not_grow_with_n(tmp_path, monkeypatch):
    """Property: copying 8 members' gridded groups one after another into one master peaks within a tenth of one
    member's slice of copying 1, and under half of one member's slice, so the peak is set by one block span and
    not by the member count.

    Class: REGRESSION (red against any copy that holds a member's data past its own turn, or that accumulates verified
    state per member: peak would grow with N; red on the pre-change tree for the second assertion, whose slice-sized
    peak is 3 times the slice). Reachability: the real function, members as separate groups of one master, as
    ``_write_master_in_place`` calls it. Both-states anchor: traced bytes, a property of the copy. Must-fire control:
    the 8-member run's master holds 8 groups (asserted) and its peak is nonzero. Killed by: a module-level cache of
    pieces, or a ``.load()`` of the slice. Passes under a second correct implementation with a different span.
    Not established: tracemalloc sees Python and numpy allocations, not the page cache or a codec's C buffers (see the
    page-cache finding in the review file).
    """
    one, slice_bytes = _peak_copying_members(tmp_path, 1, 384, monkeypatch)
    eight, _ = _peak_copying_members(tmp_path, 8, 384, monkeypatch)
    master = zarr.open_group(tmp_path / "master_8_384.zarr", mode="r")
    assert len([n for n, _ in master.groups()]) == 8 and one > 0 and eight > 0
    # Per added member the peak grows by the metadata of its group (measured 0.26 MB), never by its data (28 MB slice).
    assert eight - one <= 0.1 * slice_bytes, f"peak {one} -> {eight} bytes from 1 to 8 members"
    assert eight < 0.5 * slice_bytes, f"peak {eight} bytes against a {slice_bytes}-byte one-event slice"


# --------------------------------------------------------------------------------------------------------------------
# The rest of the experiment consolidation: building the master tree for members whose tree already moved.
# --------------------------------------------------------------------------------------------------------------------


class _Field:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


SUMMARY = "tritonswmm/triton/summary"
SERIES = "tritonswmm/triton/timeseries"


def _moved_member(root: Path, master_store: Path, member_id: str, *, capture_steps: int, summary_cells: int):
    """A member whose tree already moved into the master: its summary and series groups sit in ``master_store``."""
    summary = xr.Dataset(
        {"max_flood_depth": (("event_iloc", "y", "x"), np.ones((1, summary_cells, summary_cells), dtype="float64"))},
        coords={"event_iloc": [0], "event_id": ("event_iloc", ["e0"])},
    )
    series = xr.Dataset(
        {"flood_depth": (("event_iloc", "elapsed_s", "y", "x"), np.zeros((1, capture_steps, EDGE, EDGE), "float32"))},
        coords={"event_iloc": [0], "elapsed_s": np.arange(capture_steps, dtype="int64")},
    )
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), f"/{SUMMARY}": summary, f"/{SERIES}": series})
    directory = root / "members" / f"member_{member_id}"
    directory.mkdir(parents=True)
    node = f"member_{member_id}"
    mode = "w" if not master_store.exists() else "a"
    for rel, ds in ((SUMMARY, summary), (SERIES, series)):
        ds.chunk({d: (1 if d == "event_iloc" else n) for d, n in ds.sizes.items()}).to_zarr(
            master_store, group=f"{node}/{rel}", mode="a" if master_store.exists() else "w", consolidated=False
        )
    del tree, mode

    member = SimpleNamespace(
        analysis_paths=SimpleNamespace(
            analysis_datatree_zarr=directory / "analysis_datatree.zarr", analysis_dir=directory
        ),
        cfg_analysis=SimpleNamespace(
            toggle_clear_scenario_timeseries_after_consolidation=True, toggle_consolidate_timeseries=True
        ),
        log=SimpleNamespace(
            member_tree_moved_to_master=_Field("done"),
            member_tree_moved_groups=_Field(f'["{SUMMARY}", "{SERIES}"]'),
            datatree_consolidation_complete=_Field(True),
            consolidation_inputs_fingerprint=_Field("fp"),
        ),
    )
    member.process = SimpleNamespace(_consolidation_inputs_fingerprint=lambda: "fp")
    member._refresh_log = lambda: None
    return member


def _build_tree_bytes(tmp_path: Path, members: int, capture_steps: int, summary_cells: int) -> int:
    """Bytes of numpy buffers the master tree build holds for ``members`` moved members (traced peak)."""
    root = tmp_path / f"m{members}_{capture_steps}_{summary_cells}"
    root.mkdir()
    store = root / "experiment_datatree.zarr"
    objs = {
        str(i): _moved_member(root, store, str(i), capture_steps=capture_steps, summary_cells=summary_cells)
        for i in range(members)
    }
    zarr.consolidate_metadata(store)
    master = SimpleNamespace(
        members=objs,
        member_prefix="member_",
        analysis_paths=SimpleNamespace(experiment_datatree_zarr=store),
        experiment=SimpleNamespace(cfg_analysis=SimpleNamespace(analysis_id="exp")),
        df_setup=pd.DataFrame({"run_mode": ["gpu"] * members}, index=pd.Index(list(objs), name="member_id")),
    )
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        sa._build_master_tree(master)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_building_the_master_tree_for_moved_members_holds_summaries_and_never_the_series(tmp_path):
    """Property: ``_build_master_tree`` over members whose tree has already moved (the resume arm) holds each member's
    ``summary`` groups in memory (it reads them out of the master with ``.load()``) and none of the ``timeseries``
    capture: its peak is the same at 6 and 60 times the series size, and grows with the member count at the rate
    of one summary group per member.

    Class: CHARACTERIZATION of the residual this change leaves (green before and after). It names what else the
    experiment consolidation holds: on this arm 32 members hold 32 summary groups at once, 239 MB uncompressed each on
    the Norfolk grid (``tritonswmm/triton/summary``, 13 float64 and float32 layers of 1707 x 1751), which is 7.7 GB
    for the 32-member Phase B bundle, and it does not depend on the capture. Reachability: a master resumed after
    every member moved. Both-states anchor: traced bytes. Must-fire control: the member-count slope is positive
    (asserted), so the instrument sees summary bytes. Killed by: a build that loads a ``timeseries`` group. Passes
    under a second correct implementation that opens the summaries lazily.
    """
    small = _build_tree_bytes(tmp_path, 2, capture_steps=8, summary_cells=64)
    big_capture = _build_tree_bytes(tmp_path, 2, capture_steps=480, summary_cells=64)
    more_members = _build_tree_bytes(tmp_path, 8, capture_steps=8, summary_cells=64)
    assert small > 0 and more_members > small, "control: the build holds summary bytes that grow with members"
    assert big_capture <= 1.25 * small, f"build peak {small} -> {big_capture} bytes at 60 times the series size"


# --------------------------------------------------------------------------------------------------------------------
# Round 2: the repairs of F1 (stored-layout rechunk, stored-side span read in ``_stored_slice_equals``).
# --------------------------------------------------------------------------------------------------------------------


def _old_layout_master(tmp_path: Path, kind: str = "gridded", events: int = 2):
    """A master written by the pre-change copy (its own chunk layout) and the lazy member piece for event 0."""
    src = tmp_path / "member.zarr"
    group = _write_member(src, kind, events)
    master = tmp_path / "old.zarr"
    with deferred_consolidation(master):
        _eager_copy_series_group(src, group, master, f"m/{group}", 5, member_id="a")
    lazy = xr.open_zarr(src, group=group, consolidated=False, chunks={}, decode_timedelta=True)
    return master, f"m/{group}", lazy.isel(event_iloc=[0]).drop_encoding()


def _poke(master: Path, group: str, name: str, index: tuple, value) -> None:
    arr = zarr.open_group(master, mode="r+")[f"{group}/{name}"]
    arr[index] = value


@pytest.mark.parametrize("kind", ("gridded", "swmm_node"))
def test_the_stored_side_read_still_sees_a_changed_last_block_value_and_a_nan_position(tmp_path, kind):
    """Property: ``_stored_slice_equals`` returns True for the stored slice as copied, and False when the stored last
    (partial) time block holds one changed value, when a stored NaN is replaced by a number, and when a stored number
    is replaced by NaN, on a master laid out by the earlier toolkit.

    Class: REGRESSION guard on the round-2 change (the stored side is now persisted per span in its own chunk shape;
    a read that dropped a block or compared NaN positions loosely would return True). Reachability: real stores.
    Both-states anchor: the boolean verdict. Must-fire control: the unperturbed stored slice returns True first.
    Killed by: a stored-side read that skips the last partial span. Passes under a second correct implementation
    that compares by chunk row.
    """
    master, group, piece = _old_layout_master(tmp_path, kind)
    name = "flood_depth" if kind == "gridded" else "depth"
    last_step = STEPS - 1
    index = (0, last_step, 0, 0) if kind == "gridded" else (0, 0, last_step)
    assert Post._stored_slice_equals(master, group, piece, 0), "control: unperturbed slice must verify"
    arr = zarr.open_group(master, mode="r+")[f"{group}/{name}"]
    original = arr[index]
    arr[index] = original + 1.0
    assert not Post._stored_slice_equals(master, group, piece, 0), "changed value in the last partial block"
    arr[index] = original
    assert Post._stored_slice_equals(master, group, piece, 0)
    arr[index] = np.nan
    assert not Post._stored_slice_equals(master, group, piece, 0), "number replaced by NaN"
    arr[index] = original
    # a stored NaN (the padded tail of the shorter last event is event 1; use event 1) replaced by a number
    piece1 = xr.open_zarr(tmp_path / "member.zarr", group=group.split("/", 1)[1], consolidated=False, chunks={})
    piece1 = piece1.isel(event_iloc=[1]).drop_encoding()
    nan_index = (1, last_step, 0, 0) if kind == "gridded" else (1, 0, last_step)
    assert Post._stored_slice_equals(master, group, piece1, 1)
    arr[nan_index] = 3.0
    assert not Post._stored_slice_equals(master, group, piece1, 1), "stored NaN replaced by a number"


def test_the_stored_side_read_still_compares_the_event_identifier(tmp_path):
    """Property: the same values stored under another ``event_id`` do not verify, on an earlier-layout master.

    Class: REGRESSION guard (coordinate comparison of ``_stored_slice_equals``). Must-fire control: the unchanged
    store verifies. Killed by: dropping the event_id check. Passes under a second correct implementation that
    compares ids first.
    """
    master, group, piece = _old_layout_master(tmp_path)
    assert Post._stored_slice_equals(master, group, piece, 0)
    other = piece.assign_coords(event_id=("event_iloc", np.array(["other"], dtype="<U255")))
    assert not Post._stored_slice_equals(master, group, other, 0)
