"""The simulation window must follow each event's own forcing extent.

A rectangular weather NetCDF cannot store ragged events, so a short event is
NaN-padded onto a shared timestep axis. The window is derived from that axis,
so an untrimmed selection runs every event for the LONGEST event's duration
with its coastal boundary frozen at the last tabulated value. These tests pin
the trim that prevents it. They FAIL against pre-fix code on the padded input.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

AXIS_STEPS = 100
STEP = "120s"


def _padded_event(n_finite: int) -> xr.Dataset:
    """A rectangular single-event dataset with n_finite leading finite steps."""
    ts = pd.date_range("2025-01-01", periods=AXIS_STEPS, freq=STEP)
    wl = np.full(AXIS_STEPS, np.nan)
    rain = np.full(AXIS_STEPS, np.nan)
    wl[:n_finite] = 1.0
    rain[:n_finite] = 2.0
    return xr.Dataset(
        {"waterlevel_m": ("timestep", wl), "mm_per_hr": ("timestep", rain)},
        coords={"timestep": ts},
    )


def _finite_extent_mask(ds: xr.Dataset) -> np.ndarray:
    """The union-of-finite mask scenario.py applies at selection time."""
    return ds.notnull().to_array().any("variable").values


@pytest.mark.parametrize("n_finite", [30, 1, AXIS_STEPS])
def test_trimmed_window_end_follows_the_forcing_extent(n_finite: int) -> None:
    ds = _padded_event(n_finite)
    keep = _finite_extent_mask(ds)
    trimmed = ds.isel(timestep=keep)
    derived_end = trimmed["timestep"].to_series().max()
    expected_end = ds["timestep"].to_series().iloc[n_finite - 1]
    assert derived_end == expected_end
    assert int(keep.sum()) == n_finite


def test_unpadded_event_is_an_identity_trim() -> None:
    """The trim must be byte-identical on an input with no padding."""
    ds = _padded_event(AXIS_STEPS)
    keep = _finite_extent_mask(ds)
    assert bool(keep.all())
    xr.testing.assert_identical(ds.isel(timestep=keep), ds)


def test_all_nan_event_trims_to_zero_steps() -> None:
    """The input class that reaches the ProcessingError guard."""
    ds = _padded_event(0)
    assert int(_finite_extent_mask(ds).sum()) == 0
