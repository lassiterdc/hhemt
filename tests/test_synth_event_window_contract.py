"""The toolkit clips to a user-declared window and refuses incomplete forcing.

[Q85]: hhemt does not pad, interpolate, extend, or otherwise modify input weather, and
it does not infer a window from which values happen to be present. It runs what it is
given, over a window the user declares, and stops if the forcing is incomplete there.

THE INPUT THAT MAKES THESE FAIL PRE-FIX is a NaN-padded event with no window CSV, which
before the scrub prepared silently on a window inferred from the finite extent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.exceptions import ConfigurationError
from hhemt.scenario import _assert_forcing_complete, _weather_step_seconds, resolve_event_window

TDIM = "timestep"
N_AXIS = 100
N_FINITE = 30
FORCING = ["140", "waterlevel_m"]


def _event(n_finite: int, step: str = "120s") -> xr.Dataset:
    """One event on a rectangular axis, NaN-padded past `n_finite`."""
    ts = pd.date_range("2025-08-31", periods=N_AXIS, freq=step)
    gage = np.full(N_AXIS, np.nan)
    wl = np.full(N_AXIS, np.nan)
    gage[:n_finite] = 1.0
    wl[:n_finite] = 2.0
    return xr.Dataset(
        {
            "140": ((TDIM,), gage),
            "waterlevel_m": ((TDIM,), wl),
            # dimensionless: broadcasts under to_array() and blinded the retired detector
            "first_obs_tstep_w_rainfall": ((), 7.0),
        },
        coords={TDIM: ts},
    )


class _Cfg:
    weather_time_series_timestep_dimension_name = TDIM
    weather_event_indices = ["year"]
    weather_event_start_column = "window_start"
    weather_event_end_column = "window_end"
    weather_event_windows_csv = None
    weather_timeseries = None


def test_padded_event_is_incomplete_and_the_remedy_travels_with_the_failure():
    ds = _event(N_FINITE)
    with pytest.raises(ConfigurationError) as ei:
        _assert_forcing_complete(ds, TDIM, {"year": 0}, "w.nc", FORCING)
    msg = str(ei.value)
    assert "140" in msg and "waterlevel_m" in msg
    assert "weather_event_windows_csv" in msg
    assert "window_start" in msg and "window_end" in msg


def test_complete_event_over_a_declared_window_passes():
    """Arm (b): complete over a window SHORTER than the axis -- a proper-subset clip."""
    ds = _event(N_FINITE).isel({TDIM: slice(0, N_FINITE)})
    assert int(ds.sizes[TDIM]) == N_FINITE
    _assert_forcing_complete(ds, TDIM, {"year": 0}, "w.nc", FORCING)


def test_dimensionless_variable_does_not_mask_an_incomplete_one():
    """The Round-8 regression must not return through the fail-fast."""
    ds = _event(N_FINITE)
    retired = int(ds.notnull().to_array().any("variable").values.sum())
    assert retired == N_AXIS, "the retired expression reads the padded axis as fully finite"
    with pytest.raises(ConfigurationError):
        _assert_forcing_complete(ds, TDIM, {"year": 0}, "w.nc", FORCING)


def test_no_csv_returns_the_axis_endpoints_rather_than_raising(tmp_path):
    nc = tmp_path / "w.nc"
    _event(N_FINITE).expand_dims({"year": [0]}).to_netcdf(nc, engine="h5netcdf")
    cfg = _Cfg()
    cfg.weather_timeseries = nc
    start, end = resolve_event_window(cfg, {"year": 0})
    with xr.open_dataset(nc, engine="h5netcdf") as ds:
        axis = ds[TDIM].values
    assert start == pd.Timestamp(axis[0]) and end == pd.Timestamp(axis[-1])


def test_missing_window_column_raises_rather_than_falling_back(tmp_path):
    """A stated intent that cannot be honoured is a stop, not a fallback."""
    csv = tmp_path / "windows.csv"
    pd.DataFrame({"year": [0], "window_start": ["2025-08-31"]}).to_csv(csv, index=False)
    cfg = _Cfg()
    cfg.weather_event_windows_csv = csv
    with pytest.raises(ConfigurationError, match="window_end"):
        resolve_event_window(cfg, {"year": 0})


def test_unnamed_window_columns_raise_before_any_column_is_read(tmp_path):
    """The sentinel is rejected on its own terms, not by failing to exist."""
    from hhemt.config.analysis import WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED as SENT

    csv = tmp_path / "windows.csv"
    pd.DataFrame({"year": [0], "win_start": ["2025-08-31"], "win_end": ["2025-09-01"]}).to_csv(csv, index=False)
    cfg = _Cfg()
    cfg.weather_event_windows_csv = csv
    cfg.weather_event_start_column = SENT
    cfg.weather_event_end_column = SENT
    with pytest.raises(ConfigurationError, match="does not guess"):
        resolve_event_window(cfg, {"year": 0})


def test_a_csv_column_literally_named_unspecified_does_not_satisfy_the_sentinel(tmp_path):
    """Ordering test: sentinel BEFORE existence.

    Were existence checked first, this CSV would validate and the toolkit would read a
    column called 'unspecified' as the window -- silently, and wrongly. That is the
    accident the sentinel exists to prevent, so the ordering is the contract.
    """
    from hhemt.config.analysis import WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED as SENT

    csv = tmp_path / "windows.csv"
    pd.DataFrame({"year": [0], SENT: ["2025-08-31"]}).to_csv(csv, index=False)
    cfg = _Cfg()
    cfg.weather_event_windows_csv = csv
    cfg.weather_event_start_column = SENT
    cfg.weather_event_end_column = SENT
    with pytest.raises(ConfigurationError, match="does not guess"):
        resolve_event_window(cfg, {"year": 0})


def test_non_uniform_axis_raises_rather_than_taking_a_mode():
    ds = _event(N_AXIS)
    ragged = ds.isel({TDIM: [0, 1, 2, 50, 51]})
    with pytest.raises(ConfigurationError, match="NOT uniform"):
        _weather_step_seconds(ragged, TDIM)
    assert _weather_step_seconds(ds, TDIM) == 120.0


# --- preflight must COUNT over the declared window, not over the full axis -------


def _padded_master(tmp_path):
    """A 2-event master whose forcing is complete inside a window and NaN outside.

    Mirrors the real `norfolk_observed_event` shape: a rectangular file where each
    event occupies part of a shared axis, plus a DIMENSIONLESS variable
    (`first_obs_tstep_w_rainfall`) whose `to_array()` broadcast blinded the original
    detector. Nothing here may assume that variable carries the time dim.
    """
    n_axis, n_win = 40, 12
    ts = pd.date_range("2025-08-31", periods=n_axis, freq="120s")
    tide = np.full((2, n_axis), np.nan)
    tide[:, :n_win] = 1.0
    ds = xr.Dataset(
        {
            "waterlevel_m": (("year", TDIM), tide),
            "first_obs_tstep_w_rainfall": ((), 0),
        },
        coords={"year": [0, 1], TDIM: ts},
    )
    nc = tmp_path / "weather.nc"
    ds.to_netcdf(nc, engine="h5netcdf")

    sims = tmp_path / "sims.csv"
    pd.DataFrame({"year": [0, 1]}).to_csv(sims, index=False)

    windows = tmp_path / "windows.csv"
    pd.DataFrame(
        {
            "year": [0, 1],
            "win_start": [str(ts[0])] * 2,
            "win_end": [str(ts[n_win - 1])] * 2,
        }
    ).to_csv(windows, index=False)
    return nc, sims, windows, n_axis, n_win


class _PreflightCfgAnalysis:
    weather_time_series_timestep_dimension_name = TDIM
    weather_event_indices = ["year"]
    weather_time_series_storm_tide_datavar = "waterlevel_m"
    toggle_storm_tide_boundary = True
    weather_event_start_column = "win_start"
    weather_event_end_column = "win_end"


class _PreflightCfgSystem:
    toggle_use_swmm_for_hydrology = False
    subcatchment_raingage_mapping = None
    subcatchment_raingage_mapping_gage_id_colname = None


def test_preflight_counts_missing_over_the_declared_window_not_the_full_axis(tmp_path):
    """The regression test for the defect the local suite could not see.

    Every event's forcing is COMPLETE inside its declared window and NaN outside it, so
    preflight must report ZERO incomplete events. Before the fix it reported both,
    because it counted over `ds.sel(**sel)` -- the full axis -- while its own message
    claimed the count was taken "over the declared window". The existing contract tests
    exercised `resolve_event_window` directly and never ran preflight against a real
    window CSV over NaN-padded data, which is precisely the gap this closes.
    """
    from hhemt.validation import ValidationResult, _validate_selected_event_forcing_extent

    nc, sims, windows, _, _ = _padded_master(tmp_path)
    cfg_a, cfg_s = _PreflightCfgAnalysis(), _PreflightCfgSystem()
    cfg_a.weather_timeseries = nc
    cfg_a.weather_events_to_simulate = sims
    cfg_a.weather_event_windows_csv = windows

    result = ValidationResult()
    _validate_selected_event_forcing_extent(cfg_a, cfg_s, result)
    assert result.errors == [], (
        "preflight flagged a complete-inside-its-window event; the count is being taken "
        f"over the wrong span: {[e.message for e in result.errors]}"
    )


def test_preflight_still_catches_an_event_incomplete_INSIDE_its_window(tmp_path):
    """The other side of the same edit: narrowing the span must not blind the check."""
    from hhemt.validation import ValidationResult, _validate_selected_event_forcing_extent

    nc, sims, windows, _, n_win = _padded_master(tmp_path)
    ds = xr.open_dataset(nc, engine="h5netcdf").load()
    ds["waterlevel_m"][1, n_win - 3] = np.nan  # a hole INSIDE event 1's window
    nc2 = tmp_path / "weather_holed.nc"
    ds.to_netcdf(nc2, engine="h5netcdf")

    cfg_a, cfg_s = _PreflightCfgAnalysis(), _PreflightCfgSystem()
    cfg_a.weather_timeseries = nc2
    cfg_a.weather_events_to_simulate = sims
    cfg_a.weather_event_windows_csv = windows

    result = ValidationResult()
    _validate_selected_event_forcing_extent(cfg_a, cfg_s, result)
    assert len(result.errors) == 1
    msg = result.errors[0].message
    assert "year=1" in msg and "1 missing" in msg
    assert "year=0" not in msg, "event 0 is complete inside its window and must not be flagged"
