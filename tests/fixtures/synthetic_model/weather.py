"""Synthesize weather NetCDF with triangular rainfall and sinusoidal storm tide."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _triangle(t: np.ndarray, t_peak: float, t_end: float, peak_val: float) -> np.ndarray:
    """Triangular pulse on the minute axis `t`: 0 at t=0, linear up to `peak_val`
    at `t_peak`, linear down to 0 at `t_end`, then 0. Used for both the rain burst
    and the storm surge so they share a peak time and recede together."""
    out = np.zeros_like(t, dtype=np.float32)
    rise = (t > 0) & (t <= t_peak)
    out[rise] = peak_val * (t[rise] / max(float(t_peak), 1.0))
    fall = (t > t_peak) & (t <= t_end)
    out[fall] = peak_val * (1.0 - (t[fall] - t_peak) / max(float(t_end - t_peak), 1.0))
    return out


def _triangular_hyetograph(params) -> np.ndarray:
    """Rainfall in mm/hr at 1-min resolution.

    If `rainfall_duration_min` is None: CONSTANT `rainfall_peak_mm_per_hr` for the
    whole sim (legacy behavior). Otherwise a TRIANGULAR burst — 0 ->
    `rainfall_peak_mm_per_hr` at `rainfall_peak_min` -> 0 at `rainfall_duration_min`,
    then dry (the recession/drainage tail).
    """
    dur = params.sim_duration_min
    peak = params.rainfall_peak_mm_per_hr
    if params.rainfall_duration_min is None:
        return np.full(dur + 1, peak, dtype=np.float32)
    t = np.arange(dur + 1, dtype=np.float32)
    return _triangle(t, params.rainfall_peak_min, params.rainfall_duration_min, peak)


def _storm_surge(params) -> np.ndarray:
    """Triangular storm surge (m) added to the base tide. Peaks `stormsurge_peak_m`
    at `rainfall_peak_min` (co-peaking with the rain) and recedes to 0 by
    `rainfall_duration_min`. Zeros when no surge / no rain window is configured."""
    dur = params.sim_duration_min
    if params.stormsurge_peak_m <= 0 or params.rainfall_duration_min is None:
        return np.zeros(dur + 1, dtype=np.float32)
    t = np.arange(dur + 1, dtype=np.float32)
    return _triangle(t, params.rainfall_peak_min, params.rainfall_duration_min, params.stormsurge_peak_m)


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

    if params.compound_event:
        # 2026-06-15 compound coastal-pluvial event (the experiment runs only
        # event 0): rain_events[0] already carries the triangular rain burst;
        # event 0's water level = base tide sinusoid + triangular storm surge
        # (co-peaking with the rain at rainfall_peak_min), receding to base-tide
        # for the drainage tail. Other events keep the base broadcast (unused).
        tide_events[0, :] = tide + _storm_surge(params)
    else:
        # Legacy event structure (other catalog cases):
        #   event 0 (hydro only) : water_level = 0.0 (no BC); rain on
        #   event 1 (BC only)    : water_level = 2.0 m constant; rain = 0
        #   event 2 (both)       : water_level = 2.0 m constant; rain on
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
