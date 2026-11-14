"""The TRITON flood velocity |q| / h, derived in `summarize_triton_simulation_results` on constructed datasets.

The solver's QX and QY are unit discharge (depth times velocity, m2/s), so the toolkit derives a flood
velocity at summary time as the maximum over output prints of sqrt(qx^2 + qy^2) / h, with q and h of the SAME
print. Every test here builds a small timeseries by hand, so the expected value of each
cell is written down in the test and no solver runs; the raw-file oracle that needs a solver is
`tests/test_synth_flood_velocity_from_raw.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from hhemt.process_simulation import summarize_triton_simulation_results

FV_NAMES = (
    "max_flood_velocity",
    "time_of_max_flood_velocity",
    "x_flood_velocity_at_time_of_max_flood_velocity",
    "y_flood_velocity_at_time_of_max_flood_velocity",
)
THRESHOLD = 0.001


def _timeseries(h, qx, qy) -> xr.Dataset:
    """A (timestep, y, x) TRITON timeseries from three (nt, ny, nx) arrays; prints are 10 minutes apart."""
    h, qx, qy = (np.asarray(a, dtype="float64") for a in (h, qx, qy))
    nt, ny, nx = h.shape
    dims = ("timestep", "y", "x")
    return xr.Dataset(
        {
            "max_flood_depth": (dims, np.maximum.accumulate(np.nan_to_num(h), axis=0)),
            "flood_depth": (dims, h),
            "unit_discharge_x": (dims, qx),
            "unit_discharge_y": (dims, qy),
        },
        coords={
            "timestep": np.arange(1, nt + 1) * 10.0,
            "y": 10.0 * (ny - np.arange(ny)) - 5.0,
            "x": 10.0 * np.arange(nx) + 5.0,
        },
    )


def _summary(ds: xr.Dataset, **kwargs) -> xr.Dataset:
    return summarize_triton_simulation_results(ds, 0, 10.0, chunksize_mb=0.001, **kwargs).isel(event_iloc=0)


def _grid(value_by_cell: dict, nt: int, shape=(2, 2), fill=0.0) -> np.ndarray:
    """(nt, ny, nx) array of `fill` with per-cell time series set from {(iy, ix): [v_t0, v_t1, ...]}."""
    a = np.full((nt, *shape), fill, dtype="float64")
    for (iy, ix), series in value_by_cell.items():
        a[:, iy, ix] = series
    return a


def test_flood_velocity_is_discharge_over_the_same_print_depth_and_not_over_the_maximum_depth():
    """Property: on a wet cell the stored maximum is max over prints of |q| / h with q and h of one print.

    Class: NEW CAPABILITY. Why required: dividing by `max_flood_depth` (the running maximum) or by the depth
    of another print gives a different number whenever depth changes between prints, and the two readings
    agree on a steady cell, so only a cell whose depth moves can tell them apart.
    Kills it: a divisor of `max_flood_depth` (here 2.0 at both prints) or of the last print's depth.
    Second correct implementation: one that computes the velocity per print eagerly with numpy and reduces
    with `np.nanmax`; it passes because it writes the same values under the same names.
    """
    nt = 2
    #   print 1: q = (3, 4) so |q| = 5 over h = 0.5 -> 10 ; print 2: q = (1, 0) over h = 2.0 -> 0.5
    h = _grid({(0, 0): [0.5, 2.0]}, nt, fill=1.0)
    qx = _grid({(0, 0): [3.0, 1.0]}, nt)
    qy = _grid({(0, 0): [4.0, 0.0]}, nt)
    out = _summary(_timeseries(h, qx, qy))
    assert float(out["max_flood_velocity"][0, 0]) == pytest.approx(10.0)
    assert float(out["time_of_max_flood_velocity"][0, 0]) == 10.0
    assert float(out["x_flood_velocity_at_time_of_max_flood_velocity"][0, 0]) == pytest.approx(6.0)
    assert float(out["y_flood_velocity_at_time_of_max_flood_velocity"][0, 0]) == pytest.approx(8.0)


def test_time_of_max_flood_velocity_differs_from_time_of_max_unit_discharge_on_a_cell_where_they_differ():
    """Property: the flood-velocity time follows |q| / h, and can fall at a different print from the discharge time.

    Class: NEW CAPABILITY. Why required: a flood velocity that quietly reused the argmax of |q| would pass every
    test on a cell where q and h rise together. Cell (0, 0): |q| peaks at print 1 (5.0 over h = 4.0, velocity
    1.25) and the velocity peaks at print 2 (|q| = 2.0 over h = 0.1, velocity 20).
    Kills it: reusing `time_of_max_unit_discharge` (10.0) for the flood velocity (20.0 expected).
    Second correct implementation: any reduction that takes the first maximum of |q| / h; passes.
    """
    h = _grid({(0, 0): [4.0, 0.1]}, 2, fill=1.0)
    qx = _grid({(0, 0): [3.0, 2.0]}, 2)
    qy = _grid({(0, 0): [4.0, 0.0]}, 2)
    out = _summary(_timeseries(h, qx, qy))
    assert float(out["time_of_max_unit_discharge"][0, 0]) == 10.0
    assert float(out["time_of_max_flood_velocity"][0, 0]) == 20.0
    assert float(out["max_flood_velocity"][0, 0]) == pytest.approx(20.0)
    assert float(out["max_unit_discharge"][0, 0]) == pytest.approx(5.0)  # control: the discharge reduction is unchanged


def test_depth_at_or_below_the_threshold_gives_nan_and_just_above_gives_a_value():
    """Property: h <= threshold is undefined (NaN); h just above it is defined.

    Class: NEW CAPABILITY. Why required: the solver zeroes q below its hextra, so q / h is 0 / 0 there; the
    comparison is strict. Cell (0, 0) has h exactly at the threshold, cell (0, 1) just above it, with q = 0.002.
    Kills it: a `>=` comparison (cell (0, 0) would read 2.0), or no threshold (0 / 0 raises or reads NaN for the
    wrong reason on (0, 1) only if the divide is unguarded).
    Second correct implementation: a threshold applied to q instead of h that agrees on this fixture would still
    fail here only if it treats equal depth differently; one using `np.where` on the same strict comparison passes.
    """
    h = _grid({(0, 0): [THRESHOLD, THRESHOLD], (0, 1): [THRESHOLD * 1.0001, THRESHOLD * 1.0001]}, 2, fill=1.0)
    qx = _grid({(0, 0): [0.002, 0.002], (0, 1): [0.002, 0.002]}, 2)
    qy = _grid({}, 2)
    out = _summary(_timeseries(h, qx, qy), flood_velocity_min_depth_m=THRESHOLD)
    for name in FV_NAMES:
        assert np.isnan(float(out[name][0, 0])), name
    assert float(out["max_flood_velocity"][0, 1]) == pytest.approx(0.002 / (THRESHOLD * 1.0001))


def test_an_always_dry_cell_reads_nan_for_every_flood_velocity_variable_and_zero_for_the_discharge():
    """Property: an always-dry cell is NaN for the four flood-velocity variables and 0 for max_unit_discharge.

    Class: REGRESSION-shaped NEW CAPABILITY. Why required: the streaming reduction starts at minus
    infinity and updates on a strict greater-than, so a dry cell FILLED WITH 0 would report a maximum of 0 and the
    FIRST print as its time, a fabricated time; NaN keeps it unreported. Also covers a NaN depth.
    Kills it: filling the dry cells with 0 (time would read 10.0, maximum 0.0).
    Second correct implementation: masking after the reduction instead of before; passes if it also reports NaN.
    """
    h = _grid({(0, 0): [0.0, 0.0, 0.0], (1, 1): [np.nan, np.nan, np.nan]}, 3, fill=1.0)
    qx = _grid({(0, 0): [0.0, 0.0, 0.0]}, 3)
    qy = _grid({}, 3)
    out = _summary(_timeseries(h, qx, qy))
    for cell in ((0, 0), (1, 1)):
        for name in FV_NAMES:
            assert np.isnan(float(out[name][cell])), (cell, name)
    assert float(out["max_unit_discharge"][0, 0]) == 0.0


def test_a_dry_print_does_not_hide_a_later_wet_print_and_does_not_supply_a_value():
    """Property: only wet prints contribute; a cell dry at print 1 and wet at print 2 reads print 2's value.

    Class: NEW CAPABILITY. Why required: the mask must apply per print, not per cell. Kills it: masking a whole
    cell when its first print is dry (NaN), or taking the dry print's q / h (print 1 has q = 5 and h = 0).
    Second correct implementation: eager per-print numpy with `np.nanmax`; passes.
    """
    h = _grid({(0, 0): [0.0, 0.5]}, 2, fill=1.0)
    qx = _grid({(0, 0): [5.0, 1.0]}, 2)
    qy = _grid({}, 2)
    out = _summary(_timeseries(h, qx, qy))
    assert float(out["max_flood_velocity"][0, 0]) == pytest.approx(2.0)
    assert float(out["time_of_max_flood_velocity"][0, 0]) == 20.0


def test_the_threshold_is_the_parameter_and_is_stamped_into_every_comment():
    """Property: raising the threshold masks a cell the lower threshold reported, and each `comment` names it.

    Class: NEW CAPABILITY. Why required: the threshold is configuration, so a hard-coded 0.001 must fail.
    Kills it: ignoring `flood_velocity_min_depth_m`; a comment that does not state the threshold used.
    Second correct implementation: reading the threshold from a module constant set at call time; fails the
    parameter arm only if it cannot be overridden, as intended.
    """
    h = _grid({(0, 0): [0.05, 0.05]}, 2, fill=1.0)
    qx = _grid({(0, 0): [0.1, 0.1]}, 2)
    qy = _grid({}, 2)
    ds = _timeseries(h, qx, qy)
    low = _summary(ds, flood_velocity_min_depth_m=0.001)
    high = _summary(ds, flood_velocity_min_depth_m=0.1)
    assert float(low["max_flood_velocity"][0, 0]) == pytest.approx(2.0)
    assert np.isnan(float(high["max_flood_velocity"][0, 0]))
    for name in FV_NAMES:
        assert "0.1" in high[name].attrs["comment"], name
        assert "0.001" in low[name].attrs["comment"], name


def test_the_four_variables_get_their_cf_attributes_from_the_map():
    """Property: the magnitude carries `flood_water_speed` and `m s-1`; the time `min`; the components `m s-1`.

    Class: NEW CAPABILITY. Kills it: a velocity name on the components, a missing unit, the standard name
    `sea_water_speed`. Second correct implementation: stamping attributes inside the summary function.
    """
    from hhemt.cf_conventions import apply_cf_attributes

    out = apply_cf_attributes(
        summarize_triton_simulation_results(
            _timeseries(_grid({}, 2, fill=1.0), _grid({}, 2, fill=1.0), _grid({}, 2)), 0, 10.0, chunksize_mb=0.001
        ),
        mode="tritonswmm_triton",
    )
    assert out["max_flood_velocity"].attrs["standard_name"] == "flood_water_speed"
    assert out["max_flood_velocity"].attrs["units"] == "m s-1"
    assert out["max_flood_velocity"].attrs["cell_methods"] == "time: maximum"
    assert out["time_of_max_flood_velocity"].attrs["units"] == "min"
    for name in FV_NAMES[2:]:
        assert out[name].attrs["units"] == "m s-1"
        assert "standard_name" not in out[name].attrs
    assert "comment" in out["max_flood_velocity"].attrs  # the comment survives the CF stamp
