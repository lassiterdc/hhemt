"""Reviewer probes for the gridded series: a damaged gridded slice already in the group, and
a clearing killed before its delete.

Both arms use differently-positioned inputs than the builder's tests: the builder damages a slice during the same pass
that wrote it, and kills the clearing of the gridded kind only after its delete. These start from a group that already
holds a damaged slice (the resume of an interrupted append), and from a recorded clearing whose store is still whole,
for each of the three kinds. No solver, no Snakemake: the committed multi-sim golden tree.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tests.test_triton_timeseries_consolidation import (
    _LINK,
    _NODE,
    _TRI,
    _flag,
    _own,
    _ready,
    _scen,
    _store_files,
)


def _corrupt_last_block(analysis, event_iloc: int) -> None:
    """Overwrite one value of ``flood_depth`` in the LAST ``elapsed_s`` position of the event's slice, on disk."""
    store = analysis.analysis_paths.analysis_datatree_zarr
    group = xr.open_zarr(store, group=_TRI, consolidated=False, chunks=None)
    position = [int(v) for v in group["event_iloc"].values].index(event_iloc)
    last = group.sizes["elapsed_s"] - 1
    patch = group[["flood_depth"]].isel(event_iloc=[position], elapsed_s=[last]).load()
    patch["flood_depth"].values[...] = 987654.0
    patch.drop_vars([c for c in patch.coords if c not in patch.dims]).drop_vars(
        ["elapsed_s", "y", "x"], errors="ignore"
    ).to_zarr(
        store,
        group=_TRI,
        mode="r+",
        consolidated=False,
        region={"event_iloc": slice(position, position + 1), "elapsed_s": slice(last, last + 1)},
    )


def test_a_damaged_gridded_slice_already_in_the_group_is_rewritten_from_the_source_and_then_reads_back_equal(
    golden_extractor,
):
    """Property: a slice found in the group with one wrong value in its last block is detected by the block-wise
    read-back and rewritten from the scenario's own store; a second pass leaves the group byte-identical.

    Class: REGRESSION-shaped on the resume path of the gridded kind -- every event whose slice is present in the group
    with a failing read-back (the interrupted-append state). Reachability: a kill inside ``append_zarr`` leaves the
    shape persisted and chunks partly missing, which is the on-disk state this test writes directly. Both-states
    anchor: the assertion turns on the stored values equalling the source values, not on any message wording.
    Killed by: a plan that returns ``verified`` from metadata (event present) or a rewrite that drops the last block.
    Passes under a second correct implementation that deletes and re-appends the event.
    """
    a = _ready(golden_extractor, clear=False)
    a.process.consolidate_timeseries()
    source = xr.open_zarr(_own(a, 1), consolidated=False).load()
    _corrupt_last_block(a, 1)
    damaged = xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group=_TRI, consolidated=False).load()
    assert float(damaged["flood_depth"].isel(event_iloc=1, elapsed_s=-1).max()) == 987654.0, "probe did not damage"

    a.process.consolidate_timeseries()

    fixed = xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group=_TRI, consolidated=False).load()
    got = fixed["flood_depth"].isel(event_iloc=1).values
    want = source["flood_depth"].values
    # the group carries the extra leading instant 0 (NaN for a store with no initial state): align on the source's
    # own times, in seconds
    seconds = np.rint(source["timestep"].values * 60.0).astype("int64")
    placed = fixed["flood_depth"].isel(event_iloc=1).sel(elapsed_s=seconds).values
    assert np.array_equal(placed, want, equal_nan=True), "the damaged slice was not rewritten from the source"
    assert got.shape[0] == fixed.sizes["elapsed_s"]
    again = _store_files(a.analysis_paths.analysis_datatree_zarr / _TRI)
    a.process.consolidate_timeseries()
    assert _store_files(a.analysis_paths.analysis_datatree_zarr / _TRI) == again, "a third pass changed a clean group"


@pytest.mark.parametrize(
    "mode, group, attr",
    [
        ("tritonswmm_swmm_node_ts", _NODE, "consolidated_node_timeseries"),
        ("tritonswmm_swmm_link_ts", _LINK, "consolidated_link_timeseries"),
        ("tritonswmm_triton_ts", _TRI, "consolidated_triton_timeseries"),
    ],
)
def test_a_clearing_recorded_pending_before_its_delete_is_finished_for_every_kind(golden_extractor, mode, group, attr):
    """Property: for each of the three kinds, a scenario whose clearing is recorded pending while its store is still
    whole (a kill between the record and the delete) resumes to done: the store is removed, the kind's cleared field is
    True and its pending record is gone, and the group's slice is byte-identical to before the resume.

    Class: NEW CAPABILITY over the kinds (the builder's resume tests drive the gridded kind only). Plausible wrong
    implementation: a pending record honoured for the gridded kind only, or one whose names are read from the wrong
    kind's field. Reachability: the record is written with the real field helper over a store the consolidation just
    verified, which is the state a kill leaves. Killed by: either wrong implementation. Passes under a second correct
    implementation that re-verifies a WHOLE source before deleting it.
    """
    a = _ready(golden_extractor, clear=False)
    a.process.consolidate_timeseries()
    cls = type(a.process)
    path = getattr(_scen(a, 1).scen_paths, cls._TIMESERIES_MODE_CONFIG[mode])
    assert path.exists(), "precondition: the kind's per-scenario store is whole"
    names = sorted(xr.open_zarr(a.analysis_paths.analysis_datatree_zarr, group=group, consolidated=False).data_vars)
    log = _scen(a, 1).get_log("tritonswmm")
    cleared_name, pending_name = cls._TIMESERIES_CLEARED_FIELDS[mode]
    assert cleared_name == f"{attr}_cleared" and pending_name == f"{attr}_clearing_pending"
    a.process._log_field(log, pending_name).set(list(names))
    before = _store_files(a.analysis_paths.analysis_datatree_zarr / group)

    a.process.consolidate_timeseries()

    assert not path.exists(), f"the {mode} store survived a recorded clearing"
    assert _flag(a, 1, cleared_name) is True and _flag(a, 1, pending_name) is None
    assert _store_files(a.analysis_paths.analysis_datatree_zarr / group) == before
