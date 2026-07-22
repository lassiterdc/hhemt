"""Cross-resolution regrid + metric kernel for the dem-resolution EDA family.

The FIRST cross-grid comparator in ``eda/``. Every prior member
(``compute_sensitivity._aligned_depth_grids``, ``cross_sim_identity.compare_variable_exact``)
is ``xr.align(join="exact")``-locked to one grid and CANNOT express a resolution pair.
Those members are UNTOUCHED -- exact-alignment is correct for a same-grid comparison
and this module is not a replacement for it.

DIRECTION: resample the COARSER run onto the FINEST rung's grid (D2a). Rationale is
NOT that fine-grid is cheaper -- it is that the study measures TOTAL accuracy loss
(what a user eats when they pick a resolution), and block-averaging the fine run down
to the coarse grid destroys sub-grid detail IN THE REFERENCE, biasing measured error
DOWNWARD precisely at the coarse rungs where it should be largest. TRITON is a
first-order finite-volume scheme, so its solution IS piecewise-constant over its cells
by construction; restating a coarse cell's value on the fine cells it contains
evaluates the same function at more points and invents nothing.

RESAMPLER: ``Resampling.average`` via ``rioxarray.reproject_match``. On an UPSAMPLE
this is byte-identical to ``nearest`` -- MEASURED (EPSG:32618, divisor ratio 2:
average/max/min/nearest returned identical grids; with a nodata source cell the output
carried exactly the source values {0.0, 10.0, 30.0} with no partial average). The
control PASSED on a genuine half-wet block (2 m -> 4 m downsample: average -> 0.2,
max -> 0.4), independently reproducing gis's measured 50% peak suppression, so the
resampling argument demonstrably takes effect and the upsample degeneracy is real
rather than an ignored-argument artifact. Consequences, both deliberate:
  - ``Resampling.max`` as a second aggregate is STRUCK -- it would emit a
    byte-identical duplicate panel. Its peak-suppression gap cannot open on upsample.
  - The valid-fraction companion mask is STRUCK -- no partial averages exist on
    upsample (a fine cell falls entirely inside ONE coarse cell).
Both strikes are downsample-only mechanisms. If a fine->coarse panel is ever added,
BOTH must return.

``bilinear`` is PROHIBITED in this direction: it invents smooth sub-cell gradients the
coarse model never computed, so the map's spatial frequency exceeds the data's
information content.

DISCLOSED CAVEAT (D12): ``max_wlevel_m`` is depth above local bed, not WSE. Restating
depth piecewise-constantly on the fine grid implies a non-flat water surface and floods
fine cells whose bed sits above the coarse block's mean + d. This is a property of the
coarse run's own coarseness -- the thing under study -- so it is disclosed in the figure
caption, not corrected here.
"""

from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401  -- registers the .rio accessor
import xarray as xr
from rasterio.enums import Resampling

from hhemt.eda.compute_sensitivity import compute_magnitude
from hhemt.exceptions import ProcessingError

#: Minimum fraction of the smaller grid's extent that must overlap the other's.
#: Guards the disjoint-pair false-PASS (see _assert_comparable's docstring).
_MIN_EXTENT_OVERLAP = 0.99


def _assert_comparable(da_coarse: xr.DataArray, da_fine: xr.DataArray) -> None:
    """Guards 1-4: the positive contract REPLACING ``xr.align(join="exact")``.

    ``join="exact"`` raised on ANY grid difference, which made a wrong pair (bad CRS,
    disjoint extent, swapped direction) LOUD. Removing it to tolerate the INTENDED
    resolution difference removes that protection wholesale, and the resulting failure
    mode is WORSE, not merely different:

        today (join="exact") -> false FAILURE: passed=False, max_abs_diff=nan
        naive removal        -> false PASS:    passed=True,  max_abs=0.0

    VERIFIED: a disjoint pair 50 km apart regrids to 0 finite cells, and
    compute_magnitude's empty-domain branch (compute_sensitivity.py:115-124) returns
    max_abs=0.0 / rmse=0.0 / p95=0.0 -- i.e. PERFECT AGREEMENT, no raise, no NaN
    sentinel to notice.

    Exception TYPE cannot carry the distinction: ``xr.AlignmentError`` SUBCLASSES
    ``ValueError``, so an intended resolution difference and a genuine mismatch are
    indistinguishable in an except channel. The guards must be POSITIVE assertions.
    Guard 5 (post-regrid domain non-empty) lives in ``regrid_to_fine`` -- it can only
    be checked after the warp.
    """
    from shapely.geometry import box

    # Guard 1 -- CRS equality. reproject_match REPROJECTS across CRSs by design (a UTM
    # source onto a WGS84 template returns a populated array, no raise), so a CRS bug
    # would silently produce garbage rather than an error.
    if da_coarse.rio.crs != da_fine.rio.crs:
        raise ProcessingError(
            operation="dem_resolution_regrid",
            filepath=None,
            reason=(
                f"CRS mismatch: coarse={da_coarse.rio.crs} vs fine={da_fine.rio.crs}. "
                f"reproject_match would silently reproject rather than raise."
            ),
        )

    # Guards 2-3 -- extent overlap. A disjoint or barely-overlapping pair is the
    # false-PASS path above.
    bc, bf = box(*da_coarse.rio.bounds()), box(*da_fine.rio.bounds())
    if not bc.intersects(bf):
        raise ProcessingError(
            operation="dem_resolution_regrid",
            filepath=None,
            reason=(
                f"disjoint extents: coarse={da_coarse.rio.bounds()} vs "
                f"fine={da_fine.rio.bounds()}. Not a comparable pair."
            ),
        )
    overlap = bc.intersection(bf).area / min(bc.area, bf.area)
    if overlap < _MIN_EXTENT_OVERLAP:
        raise ProcessingError(
            operation="dem_resolution_regrid",
            filepath=None,
            reason=(
                f"extent overlap {overlap:.4f} < {_MIN_EXTENT_OVERLAP}: the two runs do "
                f"not cover the same domain, so a divergence metric over their "
                f"intersection would not be interpretable."
            ),
        )

    # Guard 4 -- direction. Both regrid directions succeed; nothing checks which member
    # is actually coarser. A swapped pair silently AGGREGATES instead of restating.
    rc = abs(float(da_coarse.rio.resolution()[0]))
    rf = abs(float(da_fine.rio.resolution()[0]))
    if not rc > rf:
        raise ProcessingError(
            operation="dem_resolution_regrid",
            filepath=None,
            reason=(
                f"regrid direction violated: da_coarse res {rc} m is not coarser than "
                f"da_fine res {rf} m. Arguments are swapped or the pair is degenerate."
            ),
        )


def regrid_to_fine(
    da_coarse: xr.DataArray,
    da_fine: xr.DataArray,
    *,
    horizontal_epsg: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resample ``da_coarse`` onto ``da_fine``'s grid; return (base, test, diag).

    ``base`` is the FINE field (the reference; compute_magnitude's baseline), ``test``
    is the coarse field restated on the fine grid -- both 2-D ``(ny, nx)``,
    shape-matched, NaN at nodata, ready for ``compute_magnitude``.

    ``horizontal_epsg`` comes from ``cfg_system.crs.horizontal_epsg``, NEVER a literal
    (the toolkit's Pydantic-model-control convention). It is REQUIRED because the
    summary grids arrive with ``rio.crs is None``: the CRS does not survive the zarr
    round-trip without ``decode_coords="all"``, which appears nowhere in ``src/``, and
    ``cf_conventions.apply_grid_mapping`` (which would attach crs_wkt at the source)
    has zero callers. Undeclared, reproject raises; this is the friendliest of the
    guards precisely because it is NOT silent.
    """
    # write_nodata BEFORE any reproject: undeclared nodata makes a single NaN annihilate
    # an entire target cell. Stamp the CRS only when ABSENT -- the summary grids arrive
    # rio.crs is None, so horizontal_epsg FILLS a missing CRS rather than OVERRIDING one.
    # An unconditional write_crs would clobber a caller-provided CRS and make Guard 1
    # (CRS-equality, in _assert_comparable) structurally unreachable, since both inputs
    # would always carry horizontal_epsg by the time the guard runs.
    if da_coarse.rio.crs is None:
        da_coarse = da_coarse.rio.write_crs(horizontal_epsg)
    if da_fine.rio.crs is None:
        da_fine = da_fine.rio.write_crs(horizontal_epsg)
    da_coarse = da_coarse.rio.write_nodata(np.nan)
    da_fine = da_fine.rio.write_nodata(np.nan)

    _assert_comparable(da_coarse, da_fine)

    test_da = da_coarse.rio.reproject_match(da_fine, resampling=Resampling.average)
    test = np.asarray(test_da, dtype="float64")
    base = np.asarray(da_fine, dtype="float64")

    # reproject_match can pad by a row/col for some ratios. Assert rather than trust:
    # compute_magnitude's ValueError is a backstop, not a diagnostic.
    if base.shape != test.shape:
        raise ProcessingError(
            operation="dem_resolution_regrid",
            filepath=None,
            reason=(
                f"regridded coarse grid {test.shape} does not match the fine grid "
                f"{base.shape} after reproject_match onto its own template"
            ),
        )

    # Guard 5 -- THE highest-value check. Without it an all-nodata regrid flows into
    # compute_magnitude and returns max_abs=0.0 / rmse=0.0 / p95=0.0: PERFECT AGREEMENT.
    # It is also the antidote to the CRS-domain trap (coordinates outside the declared
    # CRS's valid domain make reproject_match return all-nodata SILENTLY, even when the
    # source contains no NaN and the bounds match exactly).
    n_domain = int((np.isfinite(base) & np.isfinite(test)).sum())
    if n_domain == 0:
        raise ProcessingError(
            operation="dem_resolution_regrid",
            filepath=None,
            reason=(
                "regridded comparison domain is EMPTY (0 cells finite in both). The pair "
                "is not comparable; a zero-divergence result here would be an artifact, "
                "not agreement."
            ),
        )

    diag = {
        "coarse_res_m": abs(float(da_coarse.rio.resolution()[0])),
        "fine_res_m": abs(float(da_fine.rio.resolution()[0])),
        "n_domain": n_domain,
        "resampling": "average",
        "direction": "coarse_to_fine",
    }
    return base, test, diag


def compare_resolution_pair(
    da_coarse: xr.DataArray,
    da_fine: xr.DataArray,
    *,
    horizontal_epsg: int,
    dry_threshold_m: float,
) -> dict:
    """Regrid ``da_coarse`` onto ``da_fine``, then hand the pair to compute_magnitude.

    ``compute_magnitude`` is reused AS-IS (pure numpy, zero I/O, shape-matched). This
    function is the regrid step in front of it -- the kernel transfers; only the
    headline SELECTION differs, and that lives in the figure layer.

    ``dry_threshold_m`` is passed per-call because the DEM set evaluates the extent
    metric at a TWO-POINT disclosed band (tau=0.03 and 0.10 m). The module-level
    ``_DRY_THRESHOLD_M`` (0.0025) MUST NOT carry over: 2.5 mm sits an order of magnitude
    below the DEM-vertical-error floor that tau's own justification is anchored to, so
    at 2.5 mm the extent metric counts terrain-representation noise -- and terrain
    representation is precisely what this sweep varies.
    """
    base, test, diag = regrid_to_fine(da_coarse, da_fine, horizontal_epsg=horizontal_epsg)
    metrics = compute_magnitude(base, test, dry_threshold_m=dry_threshold_m)
    return {**metrics, **diag}
