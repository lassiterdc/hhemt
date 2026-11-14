"""SIMULATION-BEARING: the stored flood velocity equals |q| / h recomputed from the solver's raw BIN rasters.

This module consumes `synthetic_multisim_completed`, whose closure compiles and runs TRITON, so the conftest
marks it `compile_tier` and the project's venue rule puts it on Rivanna scratch and never on the laptop. It is
collected everywhere and runs on Rivanna only.

The oracle is independent of the toolkit's summary code: it reads the raw `H*`, `QX*` and `QY*` files of each
print with `np.fromfile` (a 2-value float64 header, then rows x cols float64 values; the format
`load_triton_output_w_xarray` documents), casts to float32 as the timeseries store does, computes the flood
velocity per print where h exceeds the threshold, and reduces by a first-occurrence argmax over prints.

WHAT A GREEN RUN PROVES, AND WHAT IT DOES NOT. Green proves four things per arm and event: the files the oracle
read are the files the solver wrote (an independent count of the raw `H*`, `QX*`, `QY*` files equals the number
of prints the toolkit enumerated and the number of timesteps in the stored timeseries); the run discriminates
(a control computed from the raw rasters shows that dividing by the running maximum depth, or by the last
print's depth, would have given a different answer on at least one wet cell); all FOUR stored variables match
(maximum, time, and both signed components); and a NON-default threshold reaches the stored result through the
function (one event per arm, recomputed from the stored timeseries at 0.05 m). It does NOT prove the strict
threshold rule: whether any cell-print lies in (0, threshold] depends on the data the solver produced, so this
module RECORDS that count (`cell_prints_in_0_threshold`, via `record_property`) and does not fail on zero; when
it is zero, the strict `h > threshold` rule is covered only by the laptop tests in
`tests/test_flood_velocity_derivation.py` and by the golden. The config-to-caller plumbing of the default
threshold is covered statically by the reviewer's call-site test, not here.

Both arms must run: an arm that is not enabled FAILS (it does not skip), and each arm must check at least one
event.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import tests.utils_for_testing as tst_ut
from hhemt.process_simulation import return_fpath_wlevels, summarize_triton_simulation_results

pytestmark = pytest.mark.usefixtures("synthetic_multisim_completed")

_ARMS = (
    ("triton", "out_triton", "output_triton_only_summary", "output_triton_only_timeseries"),
    ("tritonswmm", "out_tritonswmm", "output_tritonswmm_triton_summary", "output_tritonswmm_triton_timeseries"),
)
_NON_DEFAULT_THRESHOLD_M = 0.05
_RTOL = 1e-6


def _read_raw(path) -> np.ndarray:
    """One raw BIN raster: two float64 header values (rows, cols), then the values, as float32."""
    data = np.fromfile(path, dtype=np.float64)
    rows, cols = int(data[0]), int(data[1])
    return data[2:].reshape(rows, cols).astype(np.float32)


def _load_prints(frame):
    """Per-print (time, depth, qx, qy) float32 arrays from the raw rasters named by `frame`."""
    times, hs, qxs, qys = [], [], [], []
    for t, row in frame.iterrows():
        times.append(float(t))
        hs.append(_read_raw(row["flood_depth"]))
        qxs.append(_read_raw(row["unit_discharge_x"]))
        qys.append(_read_raw(row["unit_discharge_y"]))
    return np.array(times), np.stack(hs), np.stack(qxs), np.stack(qys)


def _reduce(times, divisor, qx, qy, wet):
    """Max over prints of |q| / divisor where `wet`, with the first-occurrence time and q / divisor there."""
    mag = (qx**2 + qy**2) ** 0.5
    d = np.where(wet, divisor, np.float32(np.nan))
    v = mag / d
    best = np.nanmax(np.where(np.isnan(v), -np.inf, v), axis=0)
    k = np.argmax(np.where(np.isnan(v), -np.inf, v), axis=0)
    never = ~np.isfinite(best)
    iy, ix = np.ogrid[: v.shape[1], : v.shape[2]]
    out_max = np.where(never, np.nan, best)
    out_time = np.where(never, np.nan, times[k])
    out_x = np.where(never, np.nan, qx[k, iy, ix] / d[k, iy, ix])
    out_y = np.where(never, np.nan, qy[k, iy, ix] / d[k, iy, ix])
    return out_max, out_time, out_x, out_y


def _compare_four(got, want, label):
    names = (
        "max_flood_velocity",
        "time_of_max_flood_velocity",
        "x_flood_velocity_at_time_of_max_flood_velocity",
        "y_flood_velocity_at_time_of_max_flood_velocity",
    )
    for name, ref in zip(names, want, strict=True):
        np.testing.assert_allclose(
            got[name].values, ref, rtol=_RTOL, atol=0, equal_nan=True, err_msg=f"{label}: {name}"
        )


@pytest.mark.parametrize("arm", _ARMS, ids=[a[0] for a in _ARMS])
def test_stored_flood_velocity_equals_discharge_over_depth_from_the_raw_rasters(
    synthetic_multisim_completed, arm, record_property
):
    """Property: per arm and event, all four stored flood-velocity variables equal the raw-raster oracle.

    Class: NEW CAPABILITY, end to end. Why required: the constructed-dataset tests prove the arithmetic on
    arrays the test wrote, and only this one proves the stored values come from the files the solver wrote,
    with the solver's own q and h at each print.
    Kills it: dividing by `max_flood_depth` or by the last print's depth (the control below shows the run can
    tell them apart), a sign or axis error in a component, a wrong time, a float64 drift beyond 1e-6 relative,
    the discharge magnitude stored under the velocity name, a dropped or duplicated print (the independent
    file count), a skipped arm (this test fails rather than skips).
    Second correct implementation: any reduction that recomputes the same four quantities from the stored
    timeseries instead of the raw files; it passes because it agrees to float32 rounding.
    """
    arm_name, out_dir_attr, summary_attr, tseries_attr = arm
    analysis = synthetic_multisim_completed
    enabled = tst_ut.get_enabled_model_types(analysis)
    assert arm_name in enabled, f"the {arm_name} arm must be enabled for this test to mean anything: {enabled}"
    interval_s = analysis.cfg_analysis.TRITON_reporting_timestep_s
    threshold = analysis.cfg_analysis.flood_velocity_min_depth_m
    n_events = 0
    n_at_threshold = 0
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        bin_dir = getattr(proc.scen_paths, out_dir_attr) / "bin"
        frame = return_fpath_wlevels(bin_dir, interval_s)
        assert not frame.empty, f"no raw output under {bin_dir}: the completed tree must keep its raw rasters"

        # Independent count: the raw files on disk, per variable, against what was enumerated and what was stored.
        counts = {p: len(list(bin_dir.glob(f"{p}*"))) for p in ("H", "QX", "QY")}
        tseries = xr.open_zarr(getattr(proc.scen_paths, tseries_attr), consolidated=False)
        assert len(set(counts.values())) == 1, f"H, QX and QY file counts differ in {bin_dir}: {counts}"
        n_stored = tseries.sizes["timestep"]
        assert counts["H"] == len(frame) == n_stored >= 2, (counts, len(frame), n_stored)

        times, h, qx, qy = _load_prints(frame)
        wet = h > np.float32(threshold)
        n_at_threshold += int(((h > 0) & (h <= np.float32(threshold))).sum())

        # Control: the run must be able to tell the right divisor from two plausible wrong ones.
        right = _reduce(times, h, qx, qy, wet)
        wrong_max = _reduce(times, np.broadcast_to(h.max(axis=0), h.shape), qx, qy, wet)
        wrong_last = _reduce(times, np.broadcast_to(h[-1], h.shape), qx, qy, wet & (h[-1] > np.float32(threshold)))
        for label, wrong in (("running-maximum depth", wrong_max), ("last-print depth", wrong_last)):
            differs = ~np.isclose(right[0], wrong[0], rtol=_RTOL, equal_nan=True)
            assert differs[np.isfinite(right[0])].any(), (
                f"event {event_iloc}: dividing by the {label} gives the same answer as the same-print depth on "
                "every wet cell, so this run cannot discriminate the derivation"
            )

        summary = xr.open_zarr(getattr(proc.scen_paths, summary_attr), consolidated=False).isel(event_iloc=0)
        _compare_four(summary, right, f"{arm_name} event {event_iloc}")
        np.testing.assert_allclose(
            summary["max_unit_discharge"].values,
            np.nanmax(((qx**2 + qy**2) ** 0.5), axis=0),
            rtol=_RTOL,
            atol=0,
            equal_nan=True,
        )
        wet_cells = np.isfinite(right[0]) & (right[0] > 0)
        assert wet_cells.any(), f"event {event_iloc}: no wet cell, so this run cannot exercise the derivation"

        if n_events == 0:
            # One event per arm with a NON-default threshold, recomputed from the stored timeseries: the keyword
            # must change the stored result, which a caller that dropped it (default 0.001) would not.
            high = summarize_triton_simulation_results(
                tseries.load(),
                event_iloc,
                float(abs(tseries["x"].to_series().diff().mode().iloc[0])),
                chunksize_mb=analysis.cfg_analysis.process_append_batch_memory_budget_mb,
                flood_velocity_min_depth_m=_NON_DEFAULT_THRESHOLD_M,
            ).isel(event_iloc=0)
            wet_high = h > np.float32(_NON_DEFAULT_THRESHOLD_M)
            _compare_four(high, _reduce(times, h, qx, qy, wet_high), f"{arm_name} threshold {_NON_DEFAULT_THRESHOLD_M}")
        n_events += 1

    assert n_events >= 1, f"{arm_name}: no event was checked"
    record_property("cell_prints_in_0_threshold", n_at_threshold)
    record_property("events_checked", n_events)
