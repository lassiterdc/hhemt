"""Reviewer's tests for change 1118-slow: the verification that reads a consolidated series slice back is still a
comparison, its resident memory is bounded by the span constant, and the coupling-capture chunk halving keeps every
stored value.

Problem these tests guard. ``_stored_slice_equals`` decides whether a consolidated slice is already correct, so a
false "equal" silently ships a wrong or empty slice and a false "unequal" rewrites a correct one. The repair
evaluates the lazy expected side once per span of blocks (``_TIMESERIES_VERIFY_SPAN_BYTES``) instead of once per
block; a span, a persisted copy and a last partial span are new places for a value to slip past the comparison.

Chunk test: it proves change 1118-slow once and is deleted at the landing unless promoted; the module whose behaviour
it guards is ``hhemt.processing_analysis`` (``_stored_slice_equals`` and ``_consolidate_coupling_diagnostics``).
"""

from __future__ import annotations

import tracemalloc
from types import SimpleNamespace

import dask
import numpy as np
import pytest
import xarray as xr
import zarr

import hhemt.processing_analysis as pa
from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as Post

T = 61  # rows along elapsed_s: not a multiple of any span or block below, so the last span and block are partial
GROUP = "g/timeseries"
EVENT = 0
AXIS = np.arange(1, T + 1, dtype="float64") * 60.0
Y, X = 4, 4
PER_STEP = 4 * Y * X  # bytes of one elapsed_s row of the float32 gridded variable

#: span constants, in rows of the gridded variable: one block; not a multiple of the block; the 30-row span that
#: makes three spans (the last holds one row); one span holding the whole slice.
SPAN_ROWS = {"one-block": 2, "odd": 7, "three-spans": 30, "whole": 1 << 20}


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    """Blocks of 2 rows for the gridded variable (128 bytes), against source chunks 20 rows deep."""
    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 128)


def _dataset(nan_at=(), zero_at=()):
    """The expected slice: a gridded float32 variable, a float64 Rank variable and a static int8 variable."""
    rng = np.random.default_rng(11)
    grid = rng.random((1, T, Y, X), dtype="float32") + 1.0
    for row, y, x in nan_at:
        grid[0, row, y, x] = np.nan
    for row, y, x in zero_at:
        grid[0, row, y, x] = 0.0
    perf = rng.random((1, T, 2)) + 1.0
    return xr.Dataset(
        {
            "flood_depth": (("event_iloc", "elapsed_s", "y", "x"), grid),
            "total": (("event_iloc", "elapsed_s", "Rank"), perf),
            "is_coupling_node": (("event_iloc", "node_id"), np.array([[1, 0, 1]], dtype="int8")),
        },
        coords={
            "event_iloc": [EVENT],
            "elapsed_s": AXIS,
            "event_id": ("event_iloc", ["event_index.0"]),
        },
    )


def _case(tmp_path, *, lazy, nan_at=(), zero_at=()):
    """``(expected, store)``: the expected slice (a dask graph over a source store 20 rows deep when ``lazy``) and a
    consolidated group holding exactly its values."""
    eager = _dataset(nan_at, zero_at)
    store = tmp_path / "consolidated.zarr"
    eager.to_zarr(store, group=GROUP, mode="w", consolidated=False)
    if not lazy:
        return eager, store
    source = tmp_path / "source.zarr"
    eager.chunk({"event_iloc": 1, "elapsed_s": 20}).to_zarr(source, mode="w", consolidated=False)
    expected = xr.open_zarr(source, consolidated=False, chunks={"event_iloc": 1, "elapsed_s": 20})
    assert expected["flood_depth"].chunks is not None, "control: the lazy arm must hand the verification a dask graph"
    return expected, store


def _set(store, var, index, value):
    """Overwrite one stored value in place and return the value it held."""
    arr = zarr.open_group(store, path=GROUP, mode="r+")[var]
    old = arr[index]
    arr[index] = value
    return old


def _equal(expected, store):
    return Post._stored_slice_equals(store, GROUP, expected, EVENT)


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy-expected", "eager-expected"])
@pytest.mark.parametrize("layout", list(SPAN_ROWS))
def test_a_single_changed_value_in_any_row_of_any_variable_is_reported_unequal(tmp_path, monkeypatch, layout, lazy):
    """Property: for every row of elapsed_s, changing ONE stored value of the gridded variable, the Rank variable, or
    the static variable makes the verification report unequal, and restoring it makes it report equal again.

    Class: every (row, variable) position of the slice, across four span layouts and both the lazy and the eager
    expected side: REGRESSION for the persisted-span arm (a verification that persists the first span and compares
    the other spans against a stale or empty array is red under it), and a characterisation of behaviour the
    per-block code also had for the eager arm. Reachability: T is 61, so the layouts give blocks of 2 rows against
    spans of 2, 7, 30 and all rows, with a last partial block and a last partial span of one row. Why required: a
    false equal leaves a wrong slice in the consolidated tree and nothing rewrites it. Both-states anchor: the
    verdict on the stored values, not any wording. Must-fire control: the unmodified store reports equal first, and
    each mutated store reports unequal. Killed by: persisting only the first span, or comparing the persisted span
    against itself. Passes under a second correct implementation that compares a whole variable at once or that
    reads source chunks through another cache.
    """
    monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", SPAN_ROWS[layout] * PER_STEP, raising=False)
    expected, store = _case(tmp_path, lazy=lazy)
    assert _equal(expected, store), "control: the untouched store must compare equal"
    missed = []
    for row in range(T):
        for var, index in (
            ("flood_depth", (0, row, row % Y, (row * 3) % X)),
            ("total", (0, row, row % 2)),
        ):
            old = _set(store, var, index, 123456.0)
            if _equal(expected, store):
                missed.append((var, index))
            _set(store, var, index, old)
    old = _set(store, "is_coupling_node", (0, 1), 1)
    if _equal(expected, store):
        missed.append(("is_coupling_node", (0, 1)))
    _set(store, "is_coupling_node", (0, 1), old)
    assert not missed, f"{len(missed)} changed value(s) reported equal: {missed[:8]}"
    assert _equal(expected, store), "control: the restored store must compare equal again"


@pytest.mark.parametrize("lazy", [True, False], ids=["lazy-expected", "eager-expected"])
@pytest.mark.parametrize("layout", ["three-spans", "whole"])
def test_a_nan_in_one_side_only_is_unequal_and_a_nan_in_both_is_equal_at_every_row(tmp_path, monkeypatch, layout, lazy):
    """Property: at every row, a NaN stored where the expected value is finite, and a finite value stored where the
    expected value is NaN, are both reported unequal, while NaN at the same position on both sides is equal.

    Class: every NaN position of the gridded variable (one per row), REGRESSION for the persisted-span arm and a
    characterisation of the NaN-aware rule the per-block code had. Reachability: the rows cross the span edges at
    30 and 60 and the last partial span. Why required: a NaN-blind comparison would treat a missing value as a
    match, and a NaN-only-in-one-side change is the shape of an unfinished write. Both-states anchor: the verdict.
    Must-fire control: the all-equal NaN arm reports equal for every row, so an always-unequal comparison is red.
    Killed by: replacing ``np.array_equal(equal_nan=True)`` by a comparison that drops NaN positions or maps NaN to
    zero (the zero arms: a stored 0.0 against an expected NaN, and a stored NaN against an expected 0.0). Passes under a
    second correct implementation that compares a NaN mask and the finite values separately.
    """
    monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", SPAN_ROWS[layout] * PER_STEP, raising=False)
    positions = [(row, row % Y, (row * 3) % X) for row in range(T)]
    (tmp_path / "finite").mkdir()
    (tmp_path / "nan").mkdir()
    expected, store = _case(tmp_path / "finite", lazy=lazy)
    expected_nan, store_nan = _case(tmp_path / "nan", lazy=lazy, nan_at=positions)
    assert _equal(expected, store), "control: the finite store compares equal"
    assert _equal(expected_nan, store_nan), "control: NaN at the same position on both sides compares equal"
    (tmp_path / "zero").mkdir()
    expected_zero, store_zero = _case(tmp_path / "zero", lazy=lazy, zero_at=positions)
    assert _equal(expected_zero, store_zero), "control: an expected zero stored as zero compares equal"
    wrong = {"stored_nan": [], "expected_nan": [], "stored_nan_vs_zero": [], "stored_zero_vs_nan": []}
    for row, y, x in positions:
        index = (0, row, y, x)
        old = _set(store, "flood_depth", index, np.nan)
        if _equal(expected, store):
            wrong["stored_nan"].append(row)
        _set(store, "flood_depth", index, old)
        old = _set(store_nan, "flood_depth", index, 5.0)
        if _equal(expected_nan, store_nan):
            wrong["expected_nan"].append(row)
        _set(store_nan, "flood_depth", index, 0.0)  # the unwritten-chunk fill a NaN-blind comparison would accept
        if _equal(expected_nan, store_nan):
            wrong["stored_zero_vs_nan"].append(row)
        _set(store_nan, "flood_depth", index, old)
        old = _set(store_zero, "flood_depth", index, np.nan)
        if _equal(expected_zero, store_zero):
            wrong["stored_nan_vs_zero"].append(row)
        _set(store_zero, "flood_depth", index, old)
    assert not any(wrong.values()), {k: v[:8] for k, v in wrong.items() if v}


def test_the_event_id_of_the_stored_slice_is_part_of_the_comparison(tmp_path, monkeypatch):
    """Property: the same values stored under a different event_id are reported unequal, with spans in force.

    Class: the event_id coordinate, the one coordinate the verification compares; CHARACTERISATION (the same on the
    pre-change tree). Reachability: ``event_id`` is a coordinate of every consolidated series group. Why required:
    equal data under a stale id would otherwise be read as verified and never corrected. Both-states anchor: the
    verdict. Killed by: dropping the event_id comparison when the span loop was written. Passes under a second
    correct implementation that compares ids after the values.
    """
    monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", 30 * PER_STEP, raising=False)
    expected, store = _case(tmp_path, lazy=True)
    assert _equal(expected, store)
    zarr.open_group(store, path=GROUP, mode="r+")["event_id"][0] = "event_index.9"
    assert not _equal(expected, store)


def _peak_bytes(expected, store) -> int:
    # The threaded scheduler's worker count follows os.cpu_count(), which a cgroup does not limit, so on a many-core
    # node more chunks decompress at once and the traced peak rises with the host and not with the code. One worker
    # makes the peak a property of the allocations alone.
    with dask.config.set(scheduler="single-threaded"):
        tracemalloc.start()
        try:
            assert _equal(expected, store)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()


def test_resident_memory_of_the_verification_grows_with_the_span_and_not_with_the_slice(tmp_path, monkeypatch):
    """Property: the peak of Python-and-NumPy allocations made by the verification rises by about the span's own
    bytes when the span grows, does not rise when the slice grows at a fixed span, and a span as long as the slice
    peaks near the slice's bytes above the floor.

    Class: every slice length at a fixed span constant and every span length at a fixed slice, NEW CAPABILITY of the
    span arm (the plausible wrong implementation persists the whole lazy variable once, red under the slice-length
    and span-growth arms). Reachability: the real member is about 1 GB per variable and per_step 12 MB against a
    1 GiB span, so it holds one span per variable; this test shrinks slice, span and chunk by the same factor
    (measured on the repaired tree with the single-threaded scheduler: 10.6 / 22.4 / 126.2 MB at spans of 2.0 / 8.2 /
    61 MB, where the pre-change per-block verification is flat; the threaded scheduler gave 22.5 / 29.2 / 100.5 MB on
    the laptop and a higher peak on a node with more cores). Why required: the Phase A
    run peaked at 55.8 GB of 64 GB, and a verification whose peak grew with the slice would exceed the allocation on a
    longer event. Both-states anchor: tracemalloc peak, a property of the allocations. Must-fire control: the
    whole-slice span peaks well above the small span, so the instrument sees span-sized allocations; on the
    pre-change tree, which has no span constant, that control is red. Killed by: persisting the whole variable, or
    holding three spans at once. Passes under a second correct implementation that holds fewer bytes than two spans.
    Not established: the scheduler's transient task memory at a different thread count (the measurement is made on the
    single-threaded scheduler, because the threaded one adds a per-host amount), and the floor of source-chunk
    reads at the real 837 MB chunk rows, which this scaled slice does not reproduce.
    """
    ny = nx = 160
    per_step = 4 * ny * nx
    rng = np.random.default_rng(3)

    def build(base, n_rows):
        base.mkdir()
        grid = rng.random((1, n_rows, ny, nx), dtype="float32")
        eager = xr.Dataset(
            {"flood_depth": (("event_iloc", "elapsed_s", "y", "x"), grid)},
            coords={
                "event_iloc": [EVENT],
                "elapsed_s": np.arange(1, n_rows + 1, dtype="float64"),
                "event_id": ("event_iloc", ["e"]),
            },
        )
        store = base / "consolidated.zarr"
        eager.to_zarr(store, group=GROUP, mode="w", consolidated=False)
        eager.chunk({"event_iloc": 1, "elapsed_s": 25}).to_zarr(base / "source.zarr", mode="w", consolidated=False)
        lazy = xr.open_zarr(base / "source.zarr", consolidated=False, chunks={"event_iloc": 1, "elapsed_s": 25})
        return lazy, store

    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 4 * per_step)
    cases = {}
    for tag, n_rows, span_rows in (
        ("s20", 600, 20),
        ("s80", 600, 80),
        ("whole", 600, 600),
        ("long40", 1200, 40),
        ("s40", 600, 40),
    ):
        monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", span_rows * per_step, raising=False)
        cases[tag] = _peak_bytes(*build(tmp_path / tag, n_rows))
    small = cases["s20"]
    assert cases["whole"] - small > 0.8 * 580 * per_step, f"control: whole-slice span did not peak higher: {cases}"
    # The verification persists the expected side and the stored side of one span, so a span of extra rows costs two
    # copies of its bytes (measured 1.9 times); a third copy held per span (2.9 times) fails this bound.
    assert cases["s80"] - small <= 2.5 * 60 * per_step, f"peak grew by more than the extra span bytes: {cases}"
    # Doubling the slice adds only the metadata and graph of its extra chunks (measured 1.11 to 1.12 times); an
    # implementation holding the slice would peak near 2 times the 126 MB whole-slice peak, about 18 times.
    assert cases["long40"] <= 1.25 * cases["s40"], f"peak grew with the slice length at a fixed span: {cases}"


def _consolidate_coupling(tmp_path, monkeypatch, sources, *, group_chunk, chunk_bytes):
    """Run the real ``_consolidate_coupling_diagnostics`` over ``sources`` ({capture name: scenario store path})."""
    import hhemt.process_simulation as ps

    monkeypatch.setattr(Post, "_COUPLING_DIAGNOSTICS_CHUNK", group_chunk)
    monkeypatch.setattr(Post, "_COUPLING_DIAGNOSTICS_CHUNK_BYTES", chunk_bytes, raising=False)
    proc = object.__new__(Post)
    proc._analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(target_processed_output_type="zarr"), df_sims=SimpleNamespace(index=[EVENT])
    )
    proc._event_slug = lambda iloc: f"event_index.{iloc}"
    proc._coupling_captures = lambda: [SimpleNamespace(name=n, raw_file=f"{n}.bin") for n in sources]
    monkeypatch.setattr(pa, "TRITONSWMM_scenario", lambda ei, analysis: SimpleNamespace(scen_paths=None))
    monkeypatch.setattr(ps, "coupling_capture_store_path", lambda scen_paths, name, out_type: sources[name])
    store = tmp_path / "consolidated.zarr"
    with pa.deferred_consolidation(store):
        proc._consolidate_coupling_diagnostics(store)
    return store


def _open_capture(store, name):
    return xr.open_zarr(store, group=f"{Post._COUPLING_DIAGNOSTICS_GROUP}/{name}", consolidated=False, chunks=None)


@pytest.mark.parametrize(
    "rows, links, dtype, untouched",
    [(10, 5, "float32", 2), (5, 10, "float32", 1), (10, 5, "float64", None), (13, 7, "float32", None)],
    ids=["steps-longer", "links-longer", "float64-cuts-both", "odd-extents-cuts-both"],
)
def test_halving_a_wide_capture_keeps_every_value_its_coordinate_and_stays_under_the_byte_limit(
    tmp_path, monkeypatch, rows, links, dtype, untouched
):
    """Property: a coupling capture over two dimensions that is split to fit the byte limit stores every value of
    every variable and its coordinate unchanged and its chunk bytes are within the limit; where one halving of the
    longer dimension suffices, the other dimension keeps its full chunk length.

    Class: every wide capture (either dimension longer, two dtypes, extents that halve unevenly), REGRESSION for the
    whole-capture arm (red on the pre-change tree: one chunk over the limit). Reachability: the group chunk is 8 and
    the limit 100 bytes, so a 10 x 5 float32 capture (200 bytes at chunk 8 x 5 = 160) halves once or twice, which is
    what 1,153,321 x 1,145 float32 does against 1 GiB. Why required: readers index the capture by label and by
    position, so the stored values are the contract; the repair halves whichever chunk is longest until the bytes fit,
    so a capture whose two dimensions are comparable is cut along both and keeps its values.
    Both-states anchor: values read back and the chunk shape read from the zarr array. Must-fire control: the values
    differ at every position, so a permuted or truncated write is red. Killed by: halving the wrong dimension or
    dropping the odd remainder of ``ceil(n / 2)``. Passes under a second correct implementation that cuts the
    longest dimension by another factor.
    """
    vals = np.arange(rows * links, dtype=dtype).reshape(rows, links) * 3 + 1
    src = tmp_path / "wide.zarr"
    ds = xr.Dataset(
        {"flow": (("step", "link"), vals), "stage": (("step",), np.arange(rows, dtype="float64") + 0.5)},
        coords={"simtime": ("step", np.arange(rows, dtype="float64") * 2.0)},
    )
    ds.to_zarr(src, mode="w", consolidated=False)
    limit = 100
    store = _consolidate_coupling(tmp_path, monkeypatch, {"wide": src}, group_chunk=8, chunk_bytes=limit)
    arr = zarr.open_group(store, mode="r")[f"{Post._COUPLING_DIAGNOSTICS_GROUP}/wide/flow"]
    assert int(np.prod(arr.chunks)) * arr.dtype.itemsize <= limit, f"chunk {arr.chunks}"
    if untouched is not None:  # axis of (event_iloc, step, link) that one halving of the other dimension leaves whole
        assert arr.chunks[untouched] == min(arr.shape[untouched], 8), f"untouched dimension was cut: {arr.chunks}"
    got = _open_capture(store, "wide")
    assert np.array_equal(got["flow"].values[0], vals)
    assert np.array_equal(got["stage"].values[0], np.arange(rows, dtype="float64") + 0.5)
    assert np.array_equal(got["simtime"].values, np.arange(rows, dtype="float64") * 2.0)


@pytest.mark.parametrize("limit, expect_step_chunk", [(128, 8), (127, 4)], ids=["exactly-at-limit", "one-byte-over"])
def test_a_chunk_exactly_at_the_byte_limit_is_not_split_and_one_byte_over_is_split_once(
    tmp_path, monkeypatch, limit, expect_step_chunk
):
    """Property: the split applies when a chunk is strictly larger than the limit, so a chunk of exactly the limit
    keeps its shape and a chunk one byte over is halved once.

    Class: the boundary of the byte limit, NEW CAPABILITY (the plausible wrong implementation splits at ``>=`` and is
    red at the exact-limit arm). Reachability: float32 over 8 steps by 4 links is 128 bytes at the group chunk of 8.
    Why required: a capture that fits must keep the layout it has always had. Both-states anchor: the stored chunk
    shape. Must-fire control: the one-byte-over arm must split, so a limit that never fires is red. Killed by:
    ``>=`` in place of ``>``. Passes under a second correct implementation that computes the limit in elements.
    """
    vals = np.arange(8 * 4, dtype="float32").reshape(8, 4)
    src = tmp_path / "edge.zarr"
    xr.Dataset({"flow": (("step", "link"), vals)}).to_zarr(src, mode="w", consolidated=False)
    store = _consolidate_coupling(tmp_path, monkeypatch, {"edge": src}, group_chunk=8, chunk_bytes=limit)
    arr = zarr.open_group(store, mode="r")[f"{Post._COUPLING_DIAGNOSTICS_GROUP}/edge/flow"]
    assert arr.chunks == (1, expect_step_chunk, 4), arr.chunks
    assert np.array_equal(_open_capture(store, "edge")["flow"].values[0], vals)


def test_a_capture_with_a_shared_dimension_keeps_the_values_of_the_variable_that_did_not_need_the_split(
    tmp_path, monkeypatch
):
    """Property: a 1-D variable sharing the split dimension with a wide variable, and an integer variable, are stored
    with their values unchanged even though the shared dimension's chunk shrank for them too.

    Class: every variable of a capture whose neighbour needed the split. Reachability: the dimension chunk is a
    property of the dimension, so the narrow variable is re-chunked without any need of its own. Why required: the
    split must change the layout and never the stored values of an unrelated variable. Both-states anchor: values.
    Killed by: applying the halved chunk to the wide variable only and leaving the dask chunks of the others
    misaligned with their encoding. Passes under a second correct implementation that chunks each variable alone.
    """
    rows, links = 12, 5
    src = tmp_path / "mixed.zarr"
    xr.Dataset(
        {
            "flow": (("step", "link"), np.arange(rows * links, dtype="float32").reshape(rows, links)),
            "count": (("step",), np.arange(rows, dtype="int32") * 7),
            "level": (("step",), np.linspace(0, 1, rows)),
        },
        coords={"simtime": ("step", np.arange(rows, dtype="float64"))},
    ).to_zarr(src, mode="w", consolidated=False)
    store = _consolidate_coupling(tmp_path, monkeypatch, {"mixed": src}, group_chunk=8, chunk_bytes=100)
    got = _open_capture(store, "mixed")
    assert np.array_equal(got["count"].values[0], np.arange(rows, dtype="int32") * 7)
    assert np.array_equal(got["level"].values[0], np.linspace(0, 1, rows))
    assert np.array_equal(got["flow"].values[0], np.arange(rows * links, dtype="float32").reshape(rows, links))
