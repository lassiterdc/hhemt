"""Synthesize weather NetCDF with triangular rainfall and sinusoidal storm tide."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _triangular_hyetograph(params) -> np.ndarray:
    """Iter-9 peak_flood_depth (2026-04-28): now returns a CONSTANT
    `rainfall_peak_mm_per_hr` value at every minute, not a triangular pulse.
    Function name retained for backward compat with any callers; the value
    in `params.rainfall_peak_mm_per_hr` is now treated as the constant
    intensity (100 mm/hr by default) rather than the peak of a triangle.
    `params.rainfall_peak_min` is ignored.

    Returned in mm/hr at 1-minute resolution.
    """
    dur = params.sim_duration_min
    rate = params.rainfall_peak_mm_per_hr
    return np.full(dur + 1, rate, dtype=np.float32)


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

    # Iter-8 of `per_sim_peak_flood_depth` (2026-04-28): user-requested switch
    # from a triangular BC to a constant BC. Events 0/1/2 still show distinct
    # flooding mechanisms — hydrology-only vs storm-tide-only vs both — but
    # the BC is now flat at `bc_active_level` for the full sim duration so
    # the system has time to equilibrate to its steady-state water surfaces.
    #
    # BC water level convention (iter-15, 2026-04-29): peak BC head 3.0 → 2.0 m
    # to pair with the flattened upstream DEM (post-swale rims now 1.0 m
    # everywhere upstream). BC = 2 m → equilibrium surface flood depth above
    # every interior junction = BC − rim = 1.0 m, which sits in the depth
    # colorbar's [0.50, 1.00] bin.
    #   event 0 (hydro only)   : water_level = 0.0 (no BC forcing); rain on
    #   event 1 (BC only)      : water_level = 2.0 m constant; rain = 0
    #   event 2 (both)         : water_level = 2.0 m constant; rain on
    bc_active_level_m = 2.0
    if _N_SYNTH_EVENTS >= 1:
        tide_events[0, :] = np.zeros_like(tide, dtype=tide.dtype)
    if _N_SYNTH_EVENTS >= 2:
        rain_events[1, :] = np.zeros_like(rain, dtype=rain.dtype)
        tide_events[1, :] = np.full_like(tide, bc_active_level_m, dtype=tide.dtype)
    if _N_SYNTH_EVENTS >= 3:
        tide_events[2, :] = np.full_like(tide, bc_active_level_m, dtype=tide.dtype)

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
