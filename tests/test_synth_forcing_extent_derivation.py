"""The forced extent must come from the weather COORDINATE, never from a config field.

`check_forcing_tail_influence` is the regression detector for the window trim, and its
whole value is being independent of the config under suspicion. An earlier form took the
step COUNT from the weather axis and the step INTERVAL from `TRITON_reporting_timestep_s`
-- two different clocks -- which understated the extent by the ratio between them.

THE CASE THAT HID IT IS THE CASE MOST FIXTURES HAPPEN TO BUILD. On the Norfolk campaigns
the weather step is 120 s and `TRITON_reporting_timestep_s` is 120, so the two clocks
coincide and the wrong formula returns the right answer. The defect is observable only
where the intervals DIFFER, which is why `interval != reporting_s` is parametrized here
and why the coincident case is included and labelled: a suite built only from coincident
fixtures passes happily over the bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hhemt.analysis_validation import (
    _FORCING_TAIL_TOLERANCE_MIN,
    _forced_extent_minutes,
)

TIME_DIM = "time"


def _weather_nc(tmp_path, n_steps: int, step_s: int, name: str = "sim_weather.nc"):
    """A per-scenario sim_weather.nc with a datetime coordinate at `step_s` spacing."""
    ts = pd.date_range("2000-01-01", periods=n_steps, freq=f"{step_s}s")
    ds = xr.Dataset(
        {
            "RG_synth": ((TIME_DIM,), np.ones(n_steps)),
            "water_level": ((TIME_DIM,), np.ones(n_steps)),
        },
        coords={TIME_DIM: ts},
    )
    path = tmp_path / name
    ds.to_netcdf(path, engine="h5netcdf")
    return path


# (n_steps, weather step_s, TRITON_reporting_timestep_s, true span in minutes, label)
_CASES = [
    (181, 60, 10.0, 180.0, "synth: 60s weather vs 10s reporting -- the case that broke it"),
    (3261, 120, 120.0, 6520.0, "norfolk: clocks COINCIDE -- the case that HIDES the defect"),
    (61, 300, 60.0, 300.0, "coarse weather, finer reporting"),
    (721, 30, 120.0, 360.0, "fine weather, coarser reporting"),
]


@pytest.mark.parametrize("n_steps,step_s,reporting_s,true_span_min,label", _CASES)
def test_extent_is_the_coordinate_span_not_a_config_multiple(
    tmp_path, n_steps, step_s, reporting_s, true_span_min, label
):
    wx = _weather_nc(tmp_path, n_steps, step_s)
    got = _forced_extent_minutes(wx, TIME_DIM)
    assert got == pytest.approx(true_span_min), label

    # The retired formula, stated explicitly so the regression is named rather than implied.
    retired = (n_steps - 1) * (reporting_s / 60.0)
    if step_s != reporting_s:
        assert got != pytest.approx(retired), (
            f"{label}: the coordinate span and the retired config-multiple must differ "
            f"here, or this case cannot detect the defect"
        )
    else:
        # Documented, not asserted away: where the clocks coincide the wrong formula is
        # indistinguishable from the right one. This branch is why the parametrization
        # cannot consist only of Norfolk-shaped fixtures.
        assert got == pytest.approx(retired), label


def test_the_synth_case_is_off_by_the_clock_ratio(tmp_path):
    """Pin the measured 6x, so a partial revert cannot look like a rounding difference."""
    wx = _weather_nc(tmp_path, 181, 60)
    got = _forced_extent_minutes(wx, TIME_DIM)
    retired = (181 - 1) * (10.0 / 60.0)
    assert got == pytest.approx(180.0)
    assert retired == pytest.approx(30.0)
    assert got / retired == pytest.approx(60.0 / 10.0)


@pytest.mark.parametrize(
    "n_steps,step_s,dim,expected_reason",
    [
        (1, 60, TIME_DIM, "fewer than two steps"),
        (181, 60, "not_a_dim", "coordinate absent"),
    ],
)
def test_underivable_extent_returns_none(tmp_path, n_steps, step_s, dim, expected_reason):
    """None, never a guess: a guessed extent manufactures the false positive this detects."""
    wx = _weather_nc(tmp_path, n_steps, step_s)
    assert _forced_extent_minutes(wx, dim) is None, expected_reason


def test_non_datetime_coordinate_returns_none(tmp_path):
    ds = xr.Dataset(
        {"water_level": ((TIME_DIM,), np.ones(5))},
        coords={TIME_DIM: np.arange(5, dtype=float)},
    )
    path = tmp_path / "sim_weather.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    assert _forced_extent_minutes(path, TIME_DIM) is None


def test_missing_file_returns_none(tmp_path):
    assert _forced_extent_minutes(tmp_path / "absent.nc", TIME_DIM) is None


def test_boundary_tolerance_is_bounded_on_both_sides():
    """The terminal snapshot lands ON the forced end, so the boundary is the normal case.

    Both bounds are asserted because either one alone permits a wrong constant: too small
    and float accumulation error fires the check on every healthy analysis; too large and
    it can mask a real overrun.
    """
    span_min = 180.0
    one_ulp = np.nextafter(span_min, np.inf) - span_min
    assert _FORCING_TAIL_TOLERANCE_MIN > one_ulp * 1e3, "must clear float64 accumulation error"
    assert _FORCING_TAIL_TOLERANCE_MIN < 1.0 / 60.0, "must stay well under one second"
