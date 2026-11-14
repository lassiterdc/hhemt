"""Phase 3: D10's threshold-crossing test.

WHY THIS FILE EXISTS: the /design-figure loop runs against the synthetic fixture at
64 x 120 = 7,680 cells (config/synthetic_experiment.py), BELOW the 25,000-cell
datashader threshold (config/report.py). At that scale a renderer WITH the gate and a
renderer WITHOUT it execute the identical code path and emit byte-identical HTML. There
is no observation the loop could make that distinguishes them -- so the loop is not the
test, and this file is.

The gap only becomes observable above 25,000 cells, which the loop never reaches. At
Norfolk 0.35 m (29.5 M cells) an ungated panel measures 561.7 MB against a 15 MB budget.
The toolkit already lived this: report.py records lowering the threshold 1,000,000 ->
25,000 on 2026-05-17 because 29,542 cells produced 6.76 MB > a 5 MB budget. The DEM set
at 0.35 m is 1,182x that cell count.
"""

from __future__ import annotations

import numpy as np

from hhemt.config.report import report_config


def _threshold() -> int:
    return report_config().per_sim.interactive.datashader_threshold_cells


def test_synthetic_fixture_is_below_the_threshold():
    """Pins the premise. If the fixture ever grows past the threshold this test fails
    LOUDLY -- at which point the loop CAN see the gate and this file's rationale
    changes. That is a real signal, not a nuisance."""
    assert 64 * 120 < _threshold()


def test_gate_fires_above_threshold_and_returns_a_non_degenerate_aggregate():
    """D10 Option 1 + amendment 1: assert the branch FIRES *and* that its output is
    non-degenerate. A fires-only assertion passes on a branch that fires and returns an
    all-NaN aggregate -- which is exactly the failure a 512x512 raster of a wet/dry mask
    can produce."""
    from hhemt.eda._dem_resolution_plots import _dem_diff_heatmap_trace

    n = int(np.ceil(np.sqrt(_threshold() * 2)))  # comfortably above the threshold
    grid = np.random.default_rng(0).normal(size=(n, n))
    trace, used_datashader = _dem_diff_heatmap_trace(grid, threshold_cells=_threshold())

    assert used_datashader is True, f"{n * n} cells must trip the {_threshold()}-cell gate"
    z = np.asarray(trace.z)
    assert z.shape != grid.shape, "a datashaded aggregate must not be the source raster"
    assert np.isfinite(z).sum() > 0, "aggregate is all-NaN -- fired but degenerate"


def test_gate_does_not_fire_below_threshold():
    from hhemt.eda._dem_resolution_plots import _dem_diff_heatmap_trace

    grid = np.random.default_rng(0).normal(size=(40, 40))
    trace, used_datashader = _dem_diff_heatmap_trace(grid, threshold_cells=_threshold())
    assert used_datashader is False
    assert np.asarray(trace.z).shape == grid.shape, "below threshold the raster is exact"


def test_tau_restricted_pct_nans_below_tau():
    """DEVIATION 1: the tau-restricted %-diff is NaN wherever the FINE reference is below
    _TAU_M, so a shallow-cell blow-up (0.005 m ref vs 0.01 m error -> +200%) cannot set the
    shared pct range and wash the real signal to white."""
    import numpy as np

    from hhemt.eda._dem_resolution_plots import _TAU_M, _tau_restricted_pct

    base = np.array([[0.005, 0.20], [_TAU_M, 0.50]])  # cell (0,0) is below tau
    test = np.array([[0.015, 0.22], [0.033, 0.55]])  # coarse - fine everywhere
    pct = _tau_restricted_pct(test - base, base, tau_m=_TAU_M)

    assert np.isnan(pct[0, 0]), "below-tau reference cell must be NaN (undefined denominator)"
    assert np.isfinite(pct[0, 1]) and np.isfinite(pct[1, 0]) and np.isfinite(pct[1, 1])
    # +10% at (0,1): (0.22-0.20)/0.20*100
    assert abs(pct[0, 1] - 10.0) < 1e-9


def test_gate_raster_fires_for_a_non_negative_reference_field():
    """The gate is field-type-agnostic, and the REFERENCE depth panel goes through it.

    WHY THIS TEST EXISTS: the pre-existing gate tests above exercise `_dem_diff_heatmap_trace`,
    which only ever served the SIGNED diff panels. The reference depth panel called `_heatmap` on
    the raw grid with no threshold check at all, so every assertion in this file passed while one
    of three raster panels stayed ungated -- the exact panel this module's docstring measures at
    561.7 MB against a 15 MB budget at Norfolk 0.35 m. A gate that one caller can bypass is not a
    gate; this test pins the seam every raster caller must share.
    """
    from hhemt.eda._dem_resolution_plots import _gate_raster

    n = int(np.ceil(np.sqrt(_threshold() * 2)))
    grid = np.abs(np.random.default_rng(0).normal(size=(n, n)))  # non-negative, like a depth field
    z, xs, ys, used = _gate_raster(grid, threshold_cells=_threshold(), reduction="max")

    assert used is True, f"{n * n} cells must trip the {_threshold()}-cell gate"
    assert np.asarray(z).shape != grid.shape, "a datashaded aggregate must not be the source raster"
    assert np.isfinite(np.asarray(z)).sum() > 0, "aggregate is all-NaN -- fired but degenerate"
    assert xs is not None and ys is not None, "the aggregate must carry its own coords"


def test_gate_raster_passes_through_below_threshold_and_keeps_caller_coords():
    """Below threshold the grid is returned untouched and coords are left to the caller.

    Pins the invariant that makes the reference panel's adoption of this seam a no-op on the synth
    fixture (7,680 cells): same array, no coord substitution, so the shipped render is unchanged.
    """
    from hhemt.eda._dem_resolution_plots import _gate_raster

    grid = np.abs(np.random.default_rng(0).normal(size=(40, 40)))
    z, xs, ys, used = _gate_raster(grid, threshold_cells=_threshold(), reduction="max")

    assert used is False
    assert np.asarray(z) is grid, "below threshold the source grid is passed through by identity"
    assert xs is None and ys is None, "below threshold the caller's own coords must be used"


def test_reference_and_diff_panels_share_one_gate_seam():
    """`_dem_diff_heatmap_trace` must DELEGATE to `_gate_raster`, not re-implement the threshold.

    Verify-by-deletion analogue for the shared seam: if a future edit re-inlines the gate into the
    diff path, the two callers can drift and the reference panel can silently fall out of the
    budget again. Asserting both paths agree on the SAME threshold boundary is what detects that.
    """
    from hhemt.eda._dem_resolution_plots import _dem_diff_heatmap_trace, _gate_raster

    n = int(np.ceil(np.sqrt(_threshold() * 2)))
    grid = np.random.default_rng(0).normal(size=(n, n))

    _, _, _, gate_used = _gate_raster(grid, threshold_cells=_threshold(), reduction="mean")
    _, diff_used = _dem_diff_heatmap_trace(grid, threshold_cells=_threshold())
    assert gate_used is diff_used is True

    small = np.random.default_rng(0).normal(size=(40, 40))
    _, _, _, gate_used_small = _gate_raster(small, threshold_cells=_threshold(), reduction="mean")
    _, diff_used_small = _dem_diff_heatmap_trace(small, threshold_cells=_threshold())
    assert gate_used_small is diff_used_small is False
