"""Reviewer probes of the H7 report step: the coupled SWMM series on their own ``elapsed_s`` axis.

Written by the reviewing seat, not the coder. The coder's arms use one step (5 s beside a 10 s TRITON step, an exact
divisor, finer). These arms vary the property the axis depends on: a step that is NOT a divisor of the TRITON step or
of the event window, a step COARSER than TRITON's, a re-consolidation of a tree that already holds the own axis, and a
consolidation that clears the per-scenario stores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tests.test_triton_timeseries_consolidation import _LINK, _NODE, _TRI, _group, _ready, _rewrite, _scen, _store_files


def _regrid(ds: xr.Dataset, start, step_s: int, window_s: float) -> xr.Dataset:
    """The store as SWMM writes it at ``step_s``: periods at start + k*step for every k inside the window."""
    periods = int(window_s // step_s)
    source = ds.sizes["date_time"]
    picked = ds.isel(date_time=np.arange(periods) % source).load()
    for variable in picked.variables.values():
        variable.encoding = {}
    stamps = np.datetime64(start, "ns") + np.arange(1, periods + 1) * np.timedelta64(step_s, "s")
    return picked.assign_coords(date_time=stamps)


def _regrid_swmm_stores(a, step_s: int) -> float:
    window_s = 0.0
    for event_iloc in a.df_sims.index:
        scen = _scen(a, int(event_iloc))
        start, end = scen._resolve_event_window()
        window_s = (end - start).total_seconds()
        for attr in ("output_tritonswmm_node_time_series", "output_tritonswmm_link_time_series"):
            _rewrite(
                Path(getattr(scen.scen_paths, attr)),
                lambda ds, start=start, window_s=window_s: _regrid(ds, start, step_s, window_s),
            )
    return window_s


@pytest.mark.parametrize("step_s", [4, 7, 9, 20, 30], ids=lambda s: f"{s}s")
def test_the_own_axis_follows_a_step_that_is_not_a_divisor_and_a_step_coarser_than_triton(golden_extractor, step_s):
    """Property: for any report step the coupled SWMM groups carry an axis of exact multiples of THAT step, covering
    every period SWMM writes, while the gridded group keeps the TRITON step.

    Class: NEW CAPABILITY, input class widened beyond the coder's 5 s. Plausible wrong implementations: an axis built
    from ``round`` of a ratio that drops a period, or one that snaps SWMM times to the TRITON grid. Control: the
    TRITON group of the same tree is unchanged (10 s), so the equality is not a property of one shared axis.
    """
    a = _ready(golden_extractor, clear=False, extra=f"swmm_report_step_s: {step_s}\n")
    window_s = _regrid_swmm_stores(a, step_s)
    a.process.consolidate_timeseries()
    periods = int(window_s // step_s)
    for name in (_NODE, _LINK):
        axis = _group(a, name)["elapsed_s"].values
        assert axis[0] == step_s and np.all(np.diff(axis) == step_s), (name, axis)
        assert len(axis) >= periods, (name, len(axis), periods)
        written = _group(a, name)["depth" if name == _NODE else next(iter(_group(a, name).data_vars))]
        assert bool(written.isel(elapsed_s=slice(0, periods)).notnull().any())
    triton_axis = _group(a, _TRI)["elapsed_s"].values
    assert np.all(np.diff(triton_axis) == 10), triton_axis


def test_reconsolidating_a_tree_that_holds_the_own_axis_rewrites_nothing(golden_extractor):
    """Property: a second ``consolidate_timeseries`` on a tree already holding the own axis is a no-op, byte for byte.

    Why: the second pass takes the "verified" path of ``_plan_timeseries_write`` with the SWMM axis passed in; a pass
    that rebuilt the slice on the TRITON axis would find a mismatch and rewrite or refuse. Control: the node group
    holds events before and after, so the comparison is over real data.
    """
    a = _ready(golden_extractor, clear=False, extra="swmm_report_step_s: 5\n")
    _regrid_swmm_stores(a, 5)
    a.process.consolidate_timeseries()
    group_dir = Path(a.analysis_paths.analysis_datatree_zarr) / _NODE
    before = _store_files(group_dir)
    assert before and len(_group(a, _NODE)["event_iloc"]) == len(a.df_sims.index)
    a.process.consolidate_timeseries()
    assert _store_files(group_dir) == before


def test_clearing_the_scenario_stores_after_an_own_axis_consolidation_keeps_the_tree_and_survives_a_rerun(
    golden_extractor,
):
    """Property: with clearing on, the per-scenario SWMM stores are removed only after a read-back on the own axis
    matched, the tree keeps the own axis, and a rerun on the cleared scenarios neither refuses nor rewrites.

    Control: the per-scenario store exists before the first pass (the clearing is the thing under test).
    """
    a = _ready(golden_extractor, clear=True, extra="swmm_report_step_s: 5\n")
    _regrid_swmm_stores(a, 5)
    stores = [Path(_scen(a, int(i)).scen_paths.output_tritonswmm_node_time_series).exists() for i in a.df_sims.index]
    assert all(stores)
    a.process.consolidate_timeseries()
    assert not any(
        Path(_scen(a, int(i)).scen_paths.output_tritonswmm_node_time_series).exists() for i in a.df_sims.index
    )
    axis = _group(a, _NODE)["elapsed_s"].values
    assert axis[0] == 5 and np.all(np.diff(axis) == 5)
    group_dir = Path(a.analysis_paths.analysis_datatree_zarr) / _NODE
    before = _store_files(group_dir)
    a.process.consolidate_timeseries()
    assert _store_files(group_dir) == before
