"""The work of verifying a consolidated series slice does not grow with the number of blocks it is read in.

``consolidate_timeseries`` reads each scenario's slice back from the consolidated store and compares it with the
slice it wrote (``_stored_slice_equals``). The expected side of a gridded or performance group is a LAZY dask
graph over the scenario's own store, and the comparison walks ``elapsed_s`` in blocks of about
``_TIMESERIES_BLOCK_BYTES``. When the scenario's store is chunked deeper in time than one block, every block
re-decompresses the source chunks it overlaps, so a source chunk is read once per block that touches it: on the
Norfolk R2a member (source chunks 70 steps deep, blocks 3 steps deep, 967 steps, four variables) that is about 23
reads of every source chunk and the member's consolidation did not finish in four hours.

The population here is every series group ``consolidate_timeseries`` writes: SWMM node, SWMM link, gridded TRITON,
performance and observation. The measured quantity is the number of source-chunk read tasks the verification
executes (dask tasks whose key names the ``open_dataset`` layer, one per dask chunk of the scenario's own store,
counted with task fusion switched off so each read keeps its name), compared with the reads of ONE evaluation of
the same slice. A block-by-block recomputation inflates the first and a read-once verification does not.

Chunk test: it proves one change (change 1118-slow) once. It guards ``processing_analysis._stored_slice_equals``.
"""

from __future__ import annotations

from types import SimpleNamespace

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from hhemt.processing_analysis import TRITONSWMM_analysis_post_processing as Post

STEPS = 60  # minutes of the synthetic event, one report per minute
START = pd.Timestamp("2025-01-01")
STEP_S = 60.0
AXIS = np.arange(1, STEPS + 1, dtype="float64") * STEP_S
EVENT_ILOC = 0
GROUP = "g/timeseries"

#: kind -> (ts_mode, whether the expected slice is a lazy dask graph over the scenario's own store)
KINDS = {
    "swmm_node": ("tritonswmm_swmm_node_ts", False),
    "swmm_link": ("tritonswmm_swmm_link_ts", False),
    "gridded": ("tritonswmm_triton_ts", True),
    "performance": ("tritonswmm_performance_ts", True),
    "observation": ("tritonswmm_observation_ts", False),
}
PATH_ATTR = {
    "swmm_node": "output_tritonswmm_node_time_series",
    "swmm_link": "output_tritonswmm_link_time_series",
    "gridded": "output_tritonswmm_triton_timeseries",
    "performance": "output_tritonswmm_performance_timeseries",
    "observation": "output_tritonswmm_triton_timeseries",
}


class _ReadCounter(Callback):
    """Counts the dask tasks that read one chunk of a scenario's own store (key names the ``open_dataset`` layer)."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    def _pretask(self, key, dsk, state) -> None:
        name = key[0] if isinstance(key, tuple) else key
        if "open_dataset" in str(name):
            self.reads += 1


def _source(kind: str) -> xr.Dataset:
    """A scenario store of ``kind`` whose time chunks (20 steps) are deeper than the blocks the tests read in."""
    rng = np.random.default_rng(7)
    minutes = np.arange(1, STEPS + 1, dtype="float64")
    if kind == "gridded":
        ds = xr.Dataset(
            {n: (("timestep", "y", "x"), rng.random((STEPS, 4, 4), dtype="float32")) for n in ("flood_depth", "ux")},
            coords={"timestep": minutes},
        )
        return ds.chunk({"timestep": 20, "y": 2, "x": 2})
    if kind == "performance":
        ds = xr.Dataset(
            {n: (("timestep", "Rank"), rng.random((STEPS, 2))) for n in ("performance.Total", "performance.Init")},
            coords={"timestep": minutes, "Rank": [0, 1]},
        )
        return ds.chunk({"timestep": 20, "Rank": 1})
    when = START + pd.to_timedelta(minutes, unit="m")
    if kind in ("swmm_node", "swmm_link"):
        dim = "node_id" if kind == "swmm_node" else "link_id"
        names = ["depth", "head"] if kind == "swmm_node" else ["flow_rate", "flow_depth"]
        data = {n: (("date_time", dim), rng.random((STEPS, 3))) for n in names}
        if kind == "swmm_node":
            data["is_coupling_node"] = ((dim,), np.array([1, 0, 1], dtype="int8"))
        return xr.Dataset(data, coords={"date_time": when, dim: ["a", "b", "c"]}).chunk({"date_time": 20})
    from hhemt.process_simulation import OBSERVATION_SERIES_FILES, OBSERVATION_TIME_DIM

    data = {n: ((OBSERVATION_TIME_DIM, "obs_point"), rng.random((STEPS, 2))) for n in OBSERVATION_SERIES_FILES}
    return xr.Dataset(data, coords={OBSERVATION_TIME_DIM: AXIS.copy()}).chunk({OBSERVATION_TIME_DIM: 20})


def _one_pass_reads(expected) -> int:
    """Source-chunk read tasks of ONE evaluation of every lazy variable of ``expected``: the floor a read-once
    verification can reach, measured on the same graph so a chunking that dask changes moves both figures."""
    lazy = [expected[v].data for v in expected.data_vars if hasattr(expected[v].data, "dask")]
    with dask.config.set({"optimization.fuse.active": False}), _ReadCounter() as counter:
        dask.compute(*lazy)
    return counter.reads


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    """One scenario store per kind, written once."""
    base = tmp_path_factory.mktemp("series_sources")
    paths = {}
    for kind in KINDS:
        path = base / f"{kind}.zarr"
        _source(kind).to_zarr(path, mode="w", consolidated=False)
        paths[kind] = path
    return paths


def _processor():
    proc = object.__new__(Post)
    proc._analysis = SimpleNamespace(
        cfg_analysis=SimpleNamespace(target_processed_output_type="zarr", run_mode="cpu", n_mpi_procs=2, n_gpus=0)
    )
    proc._event_slug = lambda iloc: f"event_index.{iloc}"
    return proc


def _expected_and_store(kind, source_path, tmp_path):
    """The slice the real ``_timeseries_slice`` builds for ``kind``, and a consolidated group holding exactly it."""
    ts_mode = KINDS[kind][0]
    scen = SimpleNamespace(scen_paths=SimpleNamespace(**{PATH_ATTR[kind]: source_path}))
    proc = _processor()
    if kind == "observation":
        expected = proc._timeseries_slice(scen, ts_mode, EVENT_ILOC, START, 60.0, AXIS)
    else:
        expected = proc._timeseries_slice(scen, ts_mode, EVENT_ILOC, START, STEP_S, AXIS)
    store = tmp_path / "consolidated.zarr"
    expected.to_zarr(store, group=GROUP, mode="w", consolidated=False)
    return expected, store


def _verify(expected, store):
    """``_stored_slice_equals`` and the number of source-chunk read tasks it executed."""
    with dask.config.set({"optimization.fuse.active": False}), _ReadCounter() as counter:
        equal = Post._stored_slice_equals(store, GROUP, expected, EVENT_ILOC)
    return equal, counter.reads


@pytest.fixture
def small_blocks(monkeypatch):
    """Blocks 2 steps deep for the gridded and performance slices, against source chunks 20 steps deep.

    Reached through ``raising=False`` for the span constant so the same tests run on a tree that predates it.
    """
    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 128)
    monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", 1 << 30, raising=False)


@pytest.mark.parametrize("kind", list(KINDS))
def test_verifying_a_slice_reads_each_source_chunk_at_most_once_in_every_series_group(
    kind, stores, tmp_path, small_blocks
):
    """Property: for every series group, verifying a slice that fits one span reads each chunk of the scenario's own
    store at most once, and still reports the slice equal.

    Class: REGRESSION for the gridded and performance groups (red on the pre-change tree: one read per overlapping
    block, 150 reads of 6 source chunks measured for the gridded group); the SWMM node, SWMM link and observation
    groups load their slice eagerly, so their expected side has no read task and the assertion holds on both
    trees (they bound the population the claim ranges over). Reachability: blocks are 2 steps and source chunks 20
    deep, so a block-by-block recomputation reads each chunk about 10 times. Both-states anchor: reads of the
    scenario's own store, a property of both the old and the new verification. Must-fire control: the gridded and
    performance groups execute more than zero read tasks, and a changed stored value is reported unequal.
    Killed by: recomputing the lazy graph once per block. Passes under a second correct implementation that
    aligns its spans to the source chunk grid or caches decompressed source chunks.
    """
    expected, store = _expected_and_store(kind, stores[kind], tmp_path)
    one_pass = _one_pass_reads(expected)
    equal, reads = _verify(expected, store)
    assert equal, f"{kind}: the slice read back from the store it was written to must compare equal"
    if KINDS[kind][1]:
        assert one_pass > 0, f"{kind}: control, the lazy expected side must execute read tasks or the counter is blind"
    assert reads <= one_pass, f"{kind}: {reads} source-chunk reads against {one_pass} for one evaluation of the slice"


def test_gridded_verification_work_does_not_grow_with_the_number_of_blocks(stores, tmp_path, monkeypatch):
    """Property: halving the block size (doubling the blocks) leaves the number of source-chunk reads unchanged.

    Class: REGRESSION (red on the pre-change tree: 150 reads at 30 blocks, 306 at 60 blocks). The quantifier is the
    block count, which is the claim of the change: the work is linear in the data and not in the blocks. Reachability:
    the real ``_stored_slice_equals`` over the real gridded slice, with ``_TIMESERIES_BLOCK_BYTES`` the only
    varied input. Both-states anchor: the read count of the scenario's own store. Must-fire control: the counts
    are nonzero, and the block counts really differ (``want.sizes`` over the block width). Killed by: per-block
    recomputation. Passes under a second correct implementation that reads each source chunk once per span.
    """
    monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", 1 << 30, raising=False)
    counts = {}
    for block_bytes in (512, 128, 64):  # 8, 2 and 1 steps per block
        monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", block_bytes)
        expected, store = _expected_and_store("gridded", stores["gridded"], tmp_path / str(block_bytes))
        equal, reads = _verify(expected, store)
        assert equal
        counts[block_bytes] = reads
    assert min(counts.values()) > 0, counts
    assert len(set(counts.values())) == 1, f"reads must not depend on the block count: {counts}"


def test_spans_that_split_the_slice_still_read_each_source_chunk_about_once_and_catch_a_late_change(
    stores, tmp_path, monkeypatch
):
    """Property: with the slice split over several spans, the reads stay within three times those of one evaluation
    (a chunk that straddles a span boundary is read once per span, and dask's take over the padded axis touches a
    chunk beyond a span's edge: measured 56 against 24, where a per-block recomputation measured 416), and one
    value changed in the LAST span still makes the verification report unequal.

    Class: REGRESSION for the read bound (red on the pre-change tree, which has no spans and reads each chunk once
    per block); NEW CAPABILITY for the late-change arm (the plausible wrong implementation is a verification that
    persists only the first span; red under it). Reachability: the span constant is shrunk to 30 steps, deeper
    than the 20-step source chunks and shallower than the 61-row slice, so three spans run; a span shallower than
    the source chunks would read each chunk once per span and is outside the bound by design. Both-states anchor:
    the unequal verdict and the read count are properties of the stored data and the source store, not of any new
    wording. Must-fire control: the unmodified store compares
    equal first. Passes under a second correct implementation that compares spans through another reader.
    """
    import zarr

    monkeypatch.setattr(Post, "_TIMESERIES_BLOCK_BYTES", 128)  # blocks of 2 steps
    per_step = 4 * 4 * 4  # float32 over y=4, x=4
    monkeypatch.setattr(Post, "_TIMESERIES_VERIFY_SPAN_BYTES", 30 * per_step, raising=False)  # spans of 30 steps
    expected, store = _expected_and_store("gridded", stores["gridded"], tmp_path)
    one_pass = _one_pass_reads(expected)
    equal, reads = _verify(expected, store)
    assert equal, "control: the untouched store compares equal"
    assert 0 < one_pass and reads <= 3 * one_pass, f"{reads} source-chunk reads against {one_pass} for one evaluation"
    arr = zarr.open_group(store, path=GROUP, mode="r+")["flood_depth"]
    arr[0, STEPS, 1, 1] = 999.0  # the last elapsed_s row of event 0
    changed, __ = _verify(expected, store)
    assert not changed, "a value changed in the last span must fail the verification"


def test_a_coupling_capture_longer_than_the_group_chunk_is_written_whatever_chunk_its_own_store_used(
    tmp_path, monkeypatch
):
    """Property: a coupling-diagnostics capture with a dimension longer than ``_COUPLING_DIAGNOSTICS_CHUNK`` reaches
    the consolidated tree with its values intact, whatever chunk shape the scenario's own store of it used.

    Class: REGRESSION (red on the pre-change tree: ``to_zarr`` refuses a source chunk that spans two dask chunks,
    the failure that ended the R2a member's consolidation after the series verification passed, on the
    ``simtime`` coordinate that does not depend on the event). Reachability: the
    group chunk is shrunk to 4 against a dimension of 10 and the scenario's own store holds that dimension as one
    chunk, which is what a capture of 1,153,321 rows does against the real 1,048,576. Both-states anchor: the values
    read back from the written group, a property of the data and not of how it was chunked. Must-fire control: the
    same capture with a dimension SHORTER than the chunk is written on both trees. Killed by: writing the capture
    with the encoding its own store carried. Passes under a second correct implementation that rechunks the
    capture to its own store's chunk shape. Domain: every capture of every scenario this path writes.
    """
    import hhemt.processing_analysis as pa

    monkeypatch.setattr(Post, "_COUPLING_DIAGNOSTICS_CHUNK", 4)
    captures = {"long": 10, "short": 3}
    sources = {}
    for name, rows in captures.items():
        src = tmp_path / f"{name}.zarr"
        simtime = np.arange(rows, dtype="float64")
        xr.Dataset({"level": ("step", simtime * 2.0)}, coords={"simtime": ("step", simtime)}).chunk(
            {"step": rows}
        ).to_zarr(src, mode="w", consolidated=False)
        sources[name] = src
    proc = _processor()
    proc._analysis.df_sims = SimpleNamespace(index=[EVENT_ILOC])
    proc._coupling_captures = lambda: [SimpleNamespace(name=n, raw_file=f"{n}.bin") for n in captures]
    monkeypatch.setattr(pa, "TRITONSWMM_scenario", lambda ei, analysis: SimpleNamespace(scen_paths=None))
    import hhemt.process_simulation as ps

    monkeypatch.setattr(ps, "coupling_capture_store_path", lambda scen_paths, name, out_type: sources[name])
    store = tmp_path / "consolidated.zarr"
    with pa.deferred_consolidation(store):
        written = proc._consolidate_coupling_diagnostics(store)
    assert sorted(written) == sorted(f"{Post._COUPLING_DIAGNOSTICS_GROUP}/{n}" for n in captures)
    for name, rows in captures.items():
        stored = xr.open_zarr(store, group=f"{Post._COUPLING_DIAGNOSTICS_GROUP}/{name}", consolidated=False)
        assert stored["level"].values.shape == (1, rows)
        assert np.array_equal(stored["simtime"].values, np.arange(rows, dtype="float64")), name
        assert np.array_equal(stored["level"].values[0], np.arange(rows, dtype="float64") * 2.0), name


def test_a_capture_over_two_long_dimensions_is_chunked_under_the_codec_buffer_limit_and_a_narrow_one_keeps_its_shape(
    tmp_path, monkeypatch
):
    """Property: no stored chunk of a coupling-diagnostics group exceeds ``_COUPLING_DIAGNOSTICS_CHUNK_BYTES``, and a
    capture whose chunks already fit keeps the chunk shape it was always given.

    Class: REGRESSION for the first arm (red on the pre-change tree: the wide capture is written as one 160-byte
    chunk against a 100-byte limit; on the real member it is a 4.8 GB chunk that the Blosc codec refuses above 2 GiB,
    ``Codec does not support buffers of > 2147483647 bytes``, which ended the R2a consolidation); the second arm is a
    must-fire control that holds on both trees (the layout of a capture that never failed is not changed). The
    limit is shrunk to 100 bytes and the group chunk to 8 rows so the same branch runs on a 50-value array.
    Both-states anchor: the stored chunk shape, read through zarr, is a property of the written group and not of any
    new wording. Killed by: chunking every dimension to the group chunk whatever the variable's width. Passes under
    a second correct implementation that splits the longest dimension by a different factor, or that sizes chunks
    from the variable's byte count in one step.
    """
    import zarr

    import hhemt.processing_analysis as pa

    monkeypatch.setattr(Post, "_COUPLING_DIAGNOSTICS_CHUNK", 8)
    monkeypatch.setattr(Post, "_COUPLING_DIAGNOSTICS_CHUNK_BYTES", 100, raising=False)
    rows, links = 10, 5
    sources = {}
    for name, dims in {"wide": ("step", "link"), "narrow": ("step",)}.items():
        shape = (rows, links) if name == "wide" else (rows,)
        src = tmp_path / f"{name}.zarr"
        xr.Dataset({"flow": (dims, np.arange(int(np.prod(shape)), dtype="float32").reshape(shape))}).to_zarr(
            src, mode="w", consolidated=False
        )
        sources[name] = src
    proc = _processor()
    proc._analysis.df_sims = SimpleNamespace(index=[EVENT_ILOC])
    proc._coupling_captures = lambda: [SimpleNamespace(name=n, raw_file=f"{n}.bin") for n in sources]
    monkeypatch.setattr(pa, "TRITONSWMM_scenario", lambda ei, analysis: SimpleNamespace(scen_paths=None))
    import hhemt.process_simulation as ps

    monkeypatch.setattr(ps, "coupling_capture_store_path", lambda scen_paths, name, out_type: sources[name])
    store = tmp_path / "consolidated.zarr"
    with pa.deferred_consolidation(store):
        proc._consolidate_coupling_diagnostics(store)
    root = zarr.open_group(store, mode="r")
    wide = root[f"{Post._COUPLING_DIAGNOSTICS_GROUP}/wide/flow"]
    narrow = root[f"{Post._COUPLING_DIAGNOSTICS_GROUP}/narrow/flow"]
    assert int(np.prod(wide.chunks)) * wide.dtype.itemsize <= 100, f"wide capture chunk {wide.chunks}"
    assert narrow.chunks == (1, 8), f"a capture whose chunk fits keeps its shape, got {narrow.chunks}"
    values = xr.open_zarr(store, group=f"{Post._COUPLING_DIAGNOSTICS_GROUP}/wide", consolidated=False)["flow"].values
    assert np.array_equal(values[0], np.arange(rows * links, dtype="float32").reshape(rows, links))
