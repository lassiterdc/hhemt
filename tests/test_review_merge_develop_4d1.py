"""Reviewer test for the merge of `develop`: a producer/consumer name seam no marker showed.

Property: a store that `eda.compute_sensitivity._artifact_vars` builds is drawn by
`eda._sensitivity_figures` with its magnitude data. The producer's grid variables (the maximum absolute
depth difference, the wetted-footprint error and the 95th-percentile relative difference) carry no unit
suffix and read `max_abs_depth_diff`, `rmse_wetted_depth` and `relative_depth_diff_p95`. A figure module that
still looks up the earlier, unit-suffixed names gets None from `_grid` for each, so the figure
draws without error and without the magnitude dots.

Input class: any labeled comparison record whose `magnitude` dict carries the producer's
current keys, drawn through the rank-sensitivity figure with no table.

Must-fire control: the same store drawn through the figure module with the producer's
variable name looked up directly (`_grid(ds, "max_abs_depth_diff", ...)`) returns a finite
grid, so the data is in the store and the absence is the reader's.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from hhemt.eda import _sensitivity_figures as sf
from hhemt.eda import compute_sensitivity as cs

_MAGNITUDE = {
    "max_abs_depth_diff": 0.5,
    "rmse_wetted_depth": 0.1,
    "relative_depth_diff_p95": 1.5,
    "n_newly_wet": 2,
    "n_extent_disagree": 3,
}


def _store(tmp_path):
    records = [("m1", [{"event_iloc": 0, "identical": False, "magnitude": dict(_MAGNITUDE)}])]
    ds = xr.Dataset(cs._artifact_vars(records))
    ds.to_zarr(tmp_path / "eda" / "eda_rank_sensitivity.zarr", mode="w", consolidated=False)
    return ds


def test_the_producer_wrote_the_magnitude_grid_under_the_name_the_control_looks_up(tmp_path) -> None:
    """Control: the data is in the store under the producer's name."""
    ds = _store(tmp_path)

    grid = sf._grid(ds, "max_abs_depth_diff", ["m1"], [0])

    assert grid is not None and np.isfinite(grid).all()


def test_the_rank_sensitivity_figure_draws_the_magnitude_dots_from_a_store_the_producer_wrote(tmp_path) -> None:
    """The figure carries one magnitude dot for the one member and event the store holds."""
    _store(tmp_path)

    fig = sf.build_rank_sensitivity_figure(tmp_path)

    dots = [t for t in fig.data if t.customdata is not None and t.x is not None and len(t.x) > 0]
    assert len(dots) == 1 and list(dots[0].x) == [0.5]
