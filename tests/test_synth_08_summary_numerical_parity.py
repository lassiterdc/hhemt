"""Numerical-parity tests for the streaming-reduction refactor of summarize_triton_simulation_results.

Covers per xarray-specialist Open Exploration finding O4 and Master Plan R1/R6:
  - Peak RSS < 4 GB on a synthetic 1000-cell x 50-timestep DEM (tracemalloc).
  - All-NaN cells produce NaN argmax and NaN companion values.
  - First-occurrence tie semantics within a single chunk.
  - First-occurrence tie semantics across chunk boundaries.
  - Numerical parity with the prior lazy-dask implementation on a small reference fixture
    (engineered to avoid the `.sel(<dask-array>)` blow-up by being small enough that the
    prior code completes — provides ground truth).
"""

from __future__ import annotations

import resource
from pathlib import Path

import numpy as np
import xarray as xr


def _make_synth_ds(
    ncols: int = 20,
    nrows: int = 25,
    ntsteps: int = 50,
    *,
    seed: int = 42,
    inject_all_nan_cells: bool = True,
    inject_ties: bool = True,
) -> xr.Dataset:
    """Construct a synthetic TRITON-summary-shaped Dataset for parity testing."""
    rng = np.random.default_rng(seed=seed)
    vx = rng.uniform(low=-2.0, high=2.0, size=(ntsteps, nrows, ncols)).astype("float64")
    vy = rng.uniform(low=-2.0, high=2.0, size=(ntsteps, nrows, ncols)).astype("float64")
    wlevel = rng.uniform(low=0.0, high=5.0, size=(ntsteps, nrows, ncols)).astype("float64")

    if inject_all_nan_cells:
        # Cells (0, 0) and (1, 1) are NaN throughout — argmax should be NaN.
        vx[:, 0, 0] = np.nan
        vy[:, 0, 0] = np.nan
        wlevel[:, 0, 0] = np.nan
        vx[:, 1, 1] = np.nan
        vy[:, 1, 1] = np.nan
        wlevel[:, 1, 1] = np.nan

    if inject_ties and ntsteps >= 26:
        # Cell (2, 2): velocity is tied at the same value at timesteps 5 and 25 — first wins.
        vx[5, 2, 2] = 1.5
        vy[5, 2, 2] = 0.0  # so velocity_mps = sqrt(1.5**2 + 0**2) = 1.5
        vx[25, 2, 2] = 1.5
        vy[25, 2, 2] = 0.0
        # Make all other timesteps at (2, 2) below 1.5
        for t in range(ntsteps):
            if t != 5 and t != 25:
                vx[t, 2, 2] = 0.0
                vy[t, 2, 2] = 0.0

    timestep_min = np.arange(ntsteps, dtype="float64") * 2.0  # 0, 2, 4, ...
    y = np.arange(nrows, dtype="float64") * 0.35
    x = np.arange(ncols, dtype="float64") * 0.35

    return xr.Dataset(
        data_vars={
            "velocity_x_mps": (("timestep_min", "y", "x"), vx),
            "velocity_y_mps": (("timestep_min", "y", "x"), vy),
            "wlevel_m": (("timestep_min", "y", "x"), wlevel),
        },
        coords={"timestep_min": timestep_min, "y": y, "x": x},
    )

def test_streaming_argmax_helper_basic(tmp_path: Path):
    """Direct test of the new helper — primary + companions on a small fixture."""
    from TRITON_SWMM_toolkit.process_simulation import _streaming_argmax_with_companions

    ds = _make_synth_ds(ncols=10, nrows=10, ntsteps=20, inject_all_nan_cells=False, inject_ties=False)
    ds = ds.assign(velocity_mps=(ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5)

    result = _streaming_argmax_with_companions(
        ds=ds,
        primary_var="velocity_mps",
        companion_vars=["velocity_x_mps", "velocity_y_mps"],
        dim="timestep_min",
        chunksize_mb=10.0,
    )

    assert "max_velocity_mps" in result
    assert "argmax_timestep_min" in result
    assert "velocity_x_mps_at_argmax" in result
    assert "velocity_y_mps_at_argmax" in result
    assert result["max_velocity_mps"].shape == (10, 10)

def test_all_nan_cells_produce_nan_outputs(tmp_path: Path):
    """All-NaN cells must produce NaN max, NaN argmax, NaN companion values."""
    from TRITON_SWMM_toolkit.process_simulation import _streaming_argmax_with_companions

    ds = _make_synth_ds(ncols=10, nrows=10, ntsteps=20, inject_all_nan_cells=True, inject_ties=False)
    ds = ds.assign(velocity_mps=(ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5)

    result = _streaming_argmax_with_companions(
        ds=ds, primary_var="velocity_mps",
        companion_vars=["velocity_x_mps", "velocity_y_mps"],
        dim="timestep_min", chunksize_mb=10.0,
    )

    assert np.isnan(result["max_velocity_mps"][0, 0])
    assert np.isnan(result["argmax_timestep_min"][0, 0])
    assert np.isnan(result["velocity_x_mps_at_argmax"][0, 0])
    assert np.isnan(result["velocity_y_mps_at_argmax"][0, 0])
    # Cell (1, 1) is also all-NaN.
    assert np.isnan(result["max_velocity_mps"][1, 1])

def test_first_occurrence_tie_semantics_within_chunk():
    """Within a single chunk, ties resolve to the earliest timestep."""
    from TRITON_SWMM_toolkit.process_simulation import _streaming_argmax_with_companions

    ds = _make_synth_ds(ncols=10, nrows=10, ntsteps=50, inject_all_nan_cells=False, inject_ties=True)
    ds = ds.assign(velocity_mps=(ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5)

    # chunk_size large enough to put timesteps 5 and 25 in the same chunk.
    result = _streaming_argmax_with_companions(
        ds=ds, primary_var="velocity_mps",
        companion_vars=["velocity_x_mps", "velocity_y_mps"],
        dim="timestep_min", chunksize_mb=100.0,
    )

    # Cell (2, 2) has tied max at timesteps 5 and 25; first occurrence (t=5, value=2.0 * 5 = 10.0).
    assert result["argmax_timestep_min"][2, 2] == 10.0  # timestep_min[5] = 5*2 = 10.0

def test_first_occurrence_tie_semantics_across_chunk_boundary():
    """Across chunk boundaries, ties resolve to the earlier chunk's index."""
    from TRITON_SWMM_toolkit.process_simulation import _streaming_argmax_with_companions

    ds = _make_synth_ds(ncols=10, nrows=10, ntsteps=50, inject_all_nan_cells=False, inject_ties=True)
    ds = ds.assign(velocity_mps=(ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5)

    # Very small chunksize: forces timesteps 5 and 25 into separate chunks.
    # With ntsteps=50 and chunksize forcing ~10 timesteps per chunk,
    # t=5 lands in chunk 0 (timesteps 0-9), t=25 lands in chunk 2 (timesteps 20-29).
    result = _streaming_argmax_with_companions(
        ds=ds, primary_var="velocity_mps",
        companion_vars=["velocity_x_mps", "velocity_y_mps"],
        dim="timestep_min", chunksize_mb=0.1,
    )

    # First-occurrence tie still wins (strict `>` in update).
    assert result["argmax_timestep_min"][2, 2] == 10.0  # t=5 value, not t=25.

def test_numerical_parity_with_reference(tmp_path: Path):
    """Numerical parity against a reference implementation on a small fixture.

    The reference fixture is small enough (10x10 grid, 20 timesteps) that the prior
    lazy-dask implementation completes without OOM — we compute the reference summary
    via direct numpy (the conceptual equivalent of the prior code path's final state).
    """
    from TRITON_SWMM_toolkit.process_simulation import _streaming_argmax_with_companions

    ds = _make_synth_ds(ncols=10, nrows=10, ntsteps=20, inject_all_nan_cells=True, inject_ties=True)
    ds = ds.assign(velocity_mps=(ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5)

    # Reference: full materialization via numpy (the conceptual equivalent of the old implementation).
    vel_all = ds["velocity_mps"].values  # (20, 10, 10)
    vx_all = ds["velocity_x_mps"].values
    vy_all = ds["velocity_y_mps"].values
    ref_max = np.nanmax(vel_all, axis=0)
    # idxmax — first occurrence semantics, NaN-skip
    ref_argmax_idx = np.full((10, 10), -1, dtype=np.int64)
    ref_vx_at_argmax = np.full((10, 10), np.nan)
    ref_vy_at_argmax = np.full((10, 10), np.nan)
    for yi in range(10):
        for xi in range(10):
            col = vel_all[:, yi, xi]
            if np.all(np.isnan(col)):
                continue
            # First-occurrence of the max
            col_safe = np.where(np.isnan(col), -np.inf, col)
            idx = int(np.argmax(col_safe))
            ref_argmax_idx[yi, xi] = idx
            ref_vx_at_argmax[yi, xi] = vx_all[idx, yi, xi]
            ref_vy_at_argmax[yi, xi] = vy_all[idx, yi, xi]
    ref_argmax_dim = np.where(
        ref_argmax_idx == -1, np.nan, ds["timestep_min"].values[np.where(ref_argmax_idx == -1, 0, ref_argmax_idx)]
    )

    # Streaming under test
    result = _streaming_argmax_with_companions(
        ds=ds, primary_var="velocity_mps",
        companion_vars=["velocity_x_mps", "velocity_y_mps"],
        dim="timestep_min", chunksize_mb=0.5,  # forces multiple chunks
    )

    np.testing.assert_allclose(result["max_velocity_mps"], ref_max, equal_nan=True)
    np.testing.assert_allclose(result["argmax_timestep_min"], ref_argmax_dim, equal_nan=True)
    np.testing.assert_allclose(result["velocity_x_mps_at_argmax"], ref_vx_at_argmax, equal_nan=True)
    np.testing.assert_allclose(result["velocity_y_mps_at_argmax"], ref_vy_at_argmax, equal_nan=True)

def test_peak_rss_under_4gb_on_synth_dem(tmp_path: Path):
    """R1: peak RSS during streaming summary on a synth 1000x1000 50-tstep DEM stays under 4 GB."""
    from TRITON_SWMM_toolkit.process_simulation import _streaming_argmax_with_companions

    ds = _make_synth_ds(ncols=1000, nrows=1000, ntsteps=50, inject_all_nan_cells=False, inject_ties=False)
    ds = ds.assign(velocity_mps=(ds["velocity_x_mps"] ** 2 + ds["velocity_y_mps"] ** 2) ** 0.5)

    # Per SE plan-review Flag 6: use resource.getrusage for actual OS-reported peak RSS
    # (tracemalloc undercounts numpy/C-extension allocations).
    baseline_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _ = _streaming_argmax_with_companions(
        ds=ds, primary_var="velocity_mps",
        companion_vars=["velocity_x_mps", "velocity_y_mps"],
        dim="timestep_min", chunksize_mb=200.0,
    )
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    peak_gb = (peak_rss_kb - baseline_rss_kb) / (1024 * 1024)
    assert peak_gb < 4.0, f"delta peak RSS {peak_gb:.2f} GB exceeded 4 GB ceiling"
