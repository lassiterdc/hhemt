"""Synthesize weather NetCDF with triangular rainfall and sinusoidal boundary water level."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _triangle(t: np.ndarray, t_peak: float, t_end: float, peak_val: float) -> np.ndarray:
    """Triangular pulse on the minute axis `t`: 0 at t=0, linear up to `peak_val`
    at `t_peak`, linear down to 0 at `t_end`, then 0. Used for both the rain burst
    and the boundary-level pulse so they share a peak time and recede together."""
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


def _bc_level_pulse(params) -> np.ndarray:
    """Triangular pulse (m) added to the base boundary level. Peaks `bc_level_peak_m`
    at `rainfall_peak_min` (co-peaking with the rain) and recedes to 0 by
    `rainfall_duration_min`. Zeros when no pulse / no rain window is configured."""
    dur = params.sim_duration_min
    if params.bc_level_peak_m <= 0 or params.rainfall_duration_min is None:
        return np.zeros(dur + 1, dtype=np.float32)
    t = np.arange(dur + 1, dtype=np.float32)
    return _triangle(t, params.rainfall_peak_min, params.rainfall_duration_min, params.bc_level_peak_m)


def _sinusoidal_bc_level(params) -> np.ndarray:
    dur = params.sim_duration_min
    t = np.arange(dur + 1, dtype=np.float32)
    omega = 2 * np.pi / (params.bc_level_period_h * 60.0)
    return params.bc_level_mean_m + params.bc_level_amplitude_m * np.sin(omega * t)


_N_SYNTH_EVENTS = 4
"""Number of synthetic events packed into weather.nc. Must be >= the largest
``n_events`` any catalog test case uses; sensitivity variants currently use
1, multi_sim uses 2, so 4 is a safe upper bound. All events carry the same
rainfall and boundary water-level series — the test tier exercises workflow plumbing,
not weather variability."""


def build_weather(params, dest: Path) -> Path:
    rain = _triangular_hyetograph(params)
    base_level = _sinusoidal_bc_level(params)
    times = pd.date_range("2000-01-01", periods=len(rain), freq="1min")

    # Each variable gets shape (n_events, n_time) so toolkit's
    # ds.sel(event_index=K) indexing picks one event's timeseries cleanly.
    rain_events = np.broadcast_to(rain, (_N_SYNTH_EVENTS, len(rain))).copy()
    bc_level_events = np.broadcast_to(base_level, (_N_SYNTH_EVENTS, len(base_level))).copy()

    # RAGGED MODE. Every event above is the same length and fully finite, so the
    # union-of-finite trim is the identity on this fixture and trimmed and
    # untrimmed code are byte-identical -- the CI tier is structurally blind to
    # the padded-event class rather than merely silent on it. When
    # params.ragged_event_fractions is set, NaN-pad each event's tail to the named
    # fraction, reproducing a rectangular store holding ragged events. The
    # generator source hash covers this file (cache.py::_generator_source_hash),
    # so changing this block auto-invalidates every cached fixture.
    fracs = getattr(params, "ragged_event_fractions", None)
    if fracs:
        n_time = rain_events.shape[1]
        for idx, frac in enumerate(fracs):
            if idx >= _N_SYNTH_EVENTS:
                break
            n_finite = max(int(round(n_time * float(frac))), 0)
            rain_events[idx, n_finite:] = np.nan
            bc_level_events[idx, n_finite:] = np.nan

    if params.compound_event:
        # 2026-06-15 compound coastal-pluvial event (the experiment runs only
        # event 0): rain_events[0] already carries the triangular rain burst;
        # event 0's water level = base level sinusoid + triangular pulse
        # (co-peaking with the rain at rainfall_peak_min), receding to the base level
        # for the drainage tail. Other events keep the base broadcast (unused).
        bc_level_events[0, :] = base_level + _bc_level_pulse(params)
    else:
        # Legacy event structure (other catalog cases):
        #   event 0 (hydro only) : water_level = 0.0 (no BC); rain on
        #   event 1 (BC only)    : water_level = 2.0 m constant; rain = 0
        #   event 2 (both)       : water_level = 2.0 m constant; rain on
        bc_active_level_m = 2.0
        if _N_SYNTH_EVENTS >= 1:
            bc_level_events[0, :] = np.zeros_like(base_level, dtype=base_level.dtype)
        if _N_SYNTH_EVENTS >= 2:
            rain_events[1, :] = np.zeros_like(rain, dtype=rain.dtype)
            bc_level_events[1, :] = np.full_like(base_level, bc_active_level_m, dtype=base_level.dtype)
        if _N_SYNTH_EVENTS >= 3:
            bc_level_events[2, :] = np.full_like(base_level, bc_active_level_m, dtype=base_level.dtype)

    ds = xr.Dataset(
        data_vars={
            "RG_synth": (
                ("event_index", "time"),
                rain_events,
                {"units": "mm/hr", "long_name": "rainfall intensity"},
            ),
            "water_level": (
                ("event_index", "time"),
                bc_level_events,
                {"units": "m", "long_name": "boundary water level"},
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
