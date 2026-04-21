"""Synthesize weather NetCDF with triangular rainfall and sinusoidal storm tide."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _triangular_hyetograph(params) -> np.ndarray:
    """Triangular pulse: 0 at start, peak at rainfall_peak_min, 0 at sim_duration_min.

    Returned in mm/hr at 1-minute resolution.
    """
    dur = params.sim_duration_min
    peak = params.rainfall_peak_mm_per_hr
    peak_t = params.rainfall_peak_min
    t = np.arange(dur + 1, dtype=np.float32)
    left = (t / max(1, peak_t)) * peak
    right = ((dur - t) / max(1, dur - peak_t)) * peak
    arr = np.minimum(left, right).clip(min=0.0)
    return arr


def _sinusoidal_stormtide(params) -> np.ndarray:
    dur = params.sim_duration_min
    t = np.arange(dur + 1, dtype=np.float32)
    omega = 2 * np.pi / (params.stormtide_period_h * 60.0)
    return params.stormtide_mean_m + params.stormtide_amplitude_m * np.sin(omega * t)


_N_SYNTH_EVENTS = 4
"""Number of synthetic events packed into weather.nc. Must be >= the largest
``n_events`` any catalog test case uses; sensitivity variants currently use
1, multi_sim uses 2, so 4 is a safe upper bound. All events carry the same
rainfall and storm-tide series — the test tier exercises workflow plumbing,
not weather variability."""


def build_weather(params, dest: Path) -> Path:
    rain = _triangular_hyetograph(params)
    tide = _sinusoidal_stormtide(params)
    times = pd.date_range("2000-01-01", periods=len(rain), freq="1min")

    # Each variable gets shape (n_events, n_time) so toolkit's
    # ds.sel(event_index=K) indexing picks one event's timeseries cleanly.
    rain_events = np.broadcast_to(rain, (_N_SYNTH_EVENTS, len(rain))).copy()
    tide_events = np.broadcast_to(tide, (_N_SYNTH_EVENTS, len(tide))).copy()

    ds = xr.Dataset(
        data_vars={
            "RG_synth": (
                ("event_index", "time"),
                rain_events,
                {"units": "mm/hr", "long_name": "rainfall intensity"},
            ),
            "water_level": (
                ("event_index", "time"),
                tide_events,
                {"units": "m", "long_name": "storm tide water level"},
            ),
        },
        coords={
            "time": times,
            "event_index": (
                ("event_index",),
                np.arange(_N_SYNTH_EVENTS, dtype=np.int32),
            ),
        },
    )
    ds.attrs["title"] = "synthetic weather for TRITON-SWMM test suite"
    ds.to_netcdf(dest, engine="h5netcdf")
    return dest
