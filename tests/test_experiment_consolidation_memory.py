"""Copying a member's ``timeseries`` group into the experiment store holds memory bounded by a span, not the slice.

``_copy_series_group`` moves one member's series group into the master one ``event_iloc`` slice at a time and reads
each slice back. It used to ``.load()`` the whole slice, which for a gridded group on the Norfolk grid is about 46 GB
for one event (four variables of 968 steps over 1707 x 1751 cells), so the experiment consolidation of ONE Phase A
member peaked at 89.98 GB (Rivanna job 23133488) and three attempts at 24 GB were killed. Members are copied one
after another, so the peak is set by the size of one event's slice and not by the number of members.

The measured quantity is the Python-allocated peak (``tracemalloc``, numpy buffers included) during the copy, as a
fraction of the event slice's bytes, at two slice sizes with the verification span held fixed.

Chunk test: it proves one change (change 1118-mem) once. It guards ``sensitivity_analysis._copy_series_group``.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import xarray as xr

from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as Post
from hhemt.sensitivity_analysis import _copy_series_group
from hhemt.utils import deferred_consolidation

GROUP = "member_a/tritonswmm/triton/timeseries"
EDGE = 128  # y and x cells; one step of one float32 variable is 64 KiB


def _member_tree(path, steps: int, events: int = 2) -> int:
    """A member tree holding one gridded ``timeseries`` group; returns the bytes of ONE event's slice."""
    rng = np.random.default_rng(3)
    data = {
        name: (("event_iloc", "elapsed_s", "y", "x"), rng.random((events, steps, EDGE, EDGE), dtype="float32"))
        for name in ("flood_depth", "unit_discharge_x")
    }
    ds = xr.Dataset(
        data,
        coords={
            "event_iloc": np.arange(events),
            "elapsed_s": np.arange(steps, dtype="float64") * 60.0,
            "event_id": ("event_iloc", [f"event_index.{i}" for i in range(events)]),
        },
    )
    ds.chunk({"event_iloc": 1, "elapsed_s": 8}).to_zarr(path, group=GROUP.split("/", 1)[1], mode="w")
    return 2 * steps * EDGE * EDGE * 4


def _peak_of_copy(tmp_path, steps: int, monkeypatch) -> tuple[int, int, xr.Dataset, xr.Dataset]:
    """Peak traced bytes while the group is copied, the slice bytes, and the source and master groups."""
    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 1 << 20)
    monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", 4 << 20, raising=False)
    src, dst = tmp_path / f"src{steps}.zarr", tmp_path / f"dst{steps}.zarr"
    slice_bytes = _member_tree(src, steps)
    src_group = GROUP.split("/", 1)[1]
    tracemalloc.start()
    try:
        with deferred_consolidation(dst):
            tracemalloc.reset_peak()
            _copy_series_group(src, src_group, dst, GROUP, 5, member_id="a")
            peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return (
        peak,
        slice_bytes,
        xr.open_zarr(src, group=src_group, consolidated=False).load(),
        xr.open_zarr(dst, group=GROUP, consolidated=False).load(),
    )


def test_copying_a_series_group_holds_a_fraction_of_one_events_slice_and_copies_every_value(tmp_path, monkeypatch):
    """Property: the peak memory of the copy is under 75 percent of one event's slice, and the master's group
    holds every value of the member's group.

    Class: REGRESSION (red on the pre-change tree: the slice is loaded whole, peak 3.0 times the slice measured
    on the synthetic 32 MB event, 4 times larger for a 4 times larger slice; 89.98 GB on the Norfolk
    member). Reachability: the real ``_copy_series_group`` over a real zarr tree of two events with 8-step chunks,
    the span shrunk to 4 MB so the bound is the span and not the slice.
    Both-states anchor: traced bytes and the stored values, properties of the
    copy and not of its wording. Must-fire control: the values compare equal (a copy that skipped the write would
    fail it) and the peak is nonzero. Killed by: ``.load()`` of the slice. Passes under a second correct
    implementation that copies by zarr region reads of one chunk row at a time.
    """
    peak, slice_bytes, member, master = _peak_of_copy(tmp_path, 256, monkeypatch)
    assert peak > 0
    for name in member.data_vars:
        assert np.array_equal(member[name].values, master[name].values), name
    assert peak < 0.75 * slice_bytes, f"peak {peak} bytes against a {slice_bytes}-byte event slice"


def test_the_copy_peak_does_not_grow_when_the_slice_grows_at_a_fixed_span(tmp_path, monkeypatch):
    """Property: quadrupling the event slice (steps 128 to 512) at a fixed verification span leaves the peak within
    1.5 times, so the bound is the span and not the capture's size.

    Class: REGRESSION (red on the pre-change tree: the peak follows the slice, a ratio near 4). Reachability: two
    runs of the real copy, the step count the only varied input. Both-states anchor: the peak ratio, a property of
    the copy. Must-fire control: the larger run's peak is nonzero and its slice is four times the smaller. Killed
    by: any copy that holds the slice. Passes under a second correct implementation that bounds its working set by
    a different constant.
    """
    small, small_bytes, *_ = _peak_of_copy(tmp_path, 128, monkeypatch)
    large, large_bytes, *_ = _peak_of_copy(tmp_path, 512, monkeypatch)
    assert large_bytes == 4 * small_bytes and large > 0
    assert large <= 1.5 * small, (
        f"peak grew {small} -> {large} bytes while the slice grew {small_bytes} -> {large_bytes}"
    )


@pytest.mark.parametrize("events", [3])
def test_a_second_copy_of_an_unchanged_group_finds_every_event_verified(tmp_path, monkeypatch, events):
    """Property: copying the same member group twice leaves the master unchanged and does not raise (the resume
    path compares each stored event with the lazy slice and skips an equal one).

    Class: NEW CAPABILITY (plausible wrong implementation: a lazy slice that the equality check cannot compare,
    which would rewrite or raise on the second pass; red under it). Both-states anchor: the stored values after the
    second pass equal the member's. Must-fire control: the first pass wrote the group. Passes under a second
    correct implementation that records verified events elsewhere.
    """
    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 1 << 20)
    src, dst = tmp_path / "src.zarr", tmp_path / "dst.zarr"
    _member_tree(src, 32, events=events)
    src_group = GROUP.split("/", 1)[1]
    for __ in range(2):
        with deferred_consolidation(dst):
            _copy_series_group(src, src_group, dst, GROUP, 5, member_id="a")
    master = xr.open_zarr(dst, group=GROUP, consolidated=False).load()
    member = xr.open_zarr(src, group=src_group, consolidated=False).load()
    assert master.sizes["event_iloc"] == events
    assert np.array_equal(member["flood_depth"].values, master["flood_depth"].values)


def _master_with_another_layout(tmp_path, steps: int = 32):
    """A member tree, and a master that already holds its event 0 written in a chunk layout the member does not use."""
    src, dst = tmp_path / "src.zarr", tmp_path / "dst.zarr"
    _member_tree(src, steps)
    group = GROUP.split("/", 1)[1]
    older = xr.open_zarr(src, group=group, consolidated=False).isel(event_iloc=[0]).load().drop_encoding()
    older.chunk({"event_iloc": 1, "elapsed_s": 16, "y": 64, "x": 64}).to_zarr(dst, group=GROUP, mode="w")
    return src, dst, group


def test_a_slice_to_append_into_a_master_with_another_chunk_layout_is_refused_by_name_and_writes_nothing(
    tmp_path, monkeypatch
):
    """Property: appending or rewriting a slice into a gridded group the master stores with other chunks raises
    ``ProcessingError`` naming the group, both chunk shapes and the master, and leaves the master as it was.

    Class: NEW CAPABILITY (plausible wrong implementation: rechunking the lazy slice to the stored layout, which
    holds memory that follows the slice, or a bare xarray ``ValueError``; red under both). Reachability: a master
    written by an earlier toolkit and resumed with a new event (append) or an unequal event (rewrite). Both-states
    anchor: the master's shape and values after the refusal. Must-fire control: the equal event is NOT refused
    (the next test). Passes under a second correct implementation that checks the layout from the group metadata.
    """
    import zarr

    from hhemt.exceptions import ProcessingError

    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 1 << 20)
    src, dst, group = _master_with_another_layout(tmp_path)
    before = xr.open_zarr(dst, group=GROUP, consolidated=False).load()
    with xr.open_zarr(src, group=group, consolidated=False) as member:
        assert member.sizes["event_iloc"] == 2, "control: the member holds an event the master lacks"
    with pytest.raises(ProcessingError, match=r"another chunk layout.*stored chunks \(1, 16, 64, 64\)"):
        with deferred_consolidation(dst):
            _copy_series_group(src, group, dst, GROUP, 5, member_id="a")
    after = xr.open_zarr(dst, group=GROUP, consolidated=False).load()
    assert after.sizes == before.sizes and np.array_equal(after["flood_depth"].values, before["flood_depth"].values)
    assert zarr.open_group(dst, mode="r")[GROUP]["flood_depth"].chunks == (1, 16, 64, 64)


def test_an_equal_slice_in_a_master_with_another_chunk_layout_is_left_alone(tmp_path, monkeypatch):
    """Property: an event the master already holds, equal to the member's, is verified and skipped whatever the
    master's chunk layout; no refusal, no write.

    Class: REGRESSION control (holds on both trees; it is the case the refusal must not catch). Both-states anchor:
    the master is unchanged and the call returns. Killed by: a layout check placed before the equality check.
    """
    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 1 << 20)
    src, dst, group = _master_with_another_layout(tmp_path)
    only = tmp_path / "one_event.zarr"
    xr.open_zarr(src, group=group, consolidated=False).isel(event_iloc=[0]).load().drop_encoding().to_zarr(
        only, group=group, mode="w"
    )
    with deferred_consolidation(dst):
        _copy_series_group(only, group, dst, GROUP, 5, member_id="a")
    assert xr.open_zarr(dst, group=GROUP, consolidated=False).sizes["event_iloc"] == 1
