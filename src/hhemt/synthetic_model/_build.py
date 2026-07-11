"""Pure synthetic-model builder (lifted out of the synthetic_model test cache module).

This module holds the deterministic ``(params) -> files`` generation surface:
the parameter/artifact dataclasses, the rim==DEM integrity assertion, and the
cache-root-agnostic ``build_synthetic_case(params, dest_dir)`` entry point. The
cache-reuse orchestration (platformdirs slug-root, filelock, ``build.complete``
sentinel, worktree-slug, ``_software`` handling) stays test-side in
``tests/fixtures/synthetic_model/cache.py``; import direction is tests -> src.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from . import (
    geometry,
    landuse,
    swmm_template,
    triton_cfg,
    vectors,
    weather,
)


@dataclasses.dataclass(frozen=True)
class SyntheticModelParams:
    n_cols: int = 16  # iter-8 peak_flood_depth: 20→16 (narrower DEM so the
    # figure has less surrounding wall area and the Y-shaped
    # channel reads as the dominant feature).
    n_rows: int = 30
    cell_size_m: float = 10.0
    xllcorner: float = 0.0  # test-case local origin (no real-world geolocation)
    yllcorner: float = 0.0
    epsg: int = 3857  # Web Mercator — accepts (0,0) at equator/prime meridian
    slope_ns: float = 0.01  # 1% N->S
    valley_depth_m: float = 0.5
    impervious_mannings: float = 0.015
    pervious_mannings: float = 0.035
    sim_duration_min: int = 180  # iter-10 peak_flood_depth: 30→180 (3h)
    # constant 5m BC has time to equilibrate
    # upstream-junction water surfaces via SWMM
    # backwater. Shorter durations (10, 30 min) can
    # be swept post-baseline by overriding this
    # field; rainfall keeps its early-burst shape
    # so the long tail is post-storm/relaxation.
    triton_timestep_s: float = 1.0
    reporting_timestep_s: float = 10.0
    rainfall_peak_mm_per_hr: float = 100.0  # iter-9 peak_flood_depth: switched
    # to a CONSTANT 100 mm/hr for the entire sim
    # duration (was 3000 mm/hr triangular). The
    # "peak" name is retained for backward compat
    # with the catalog; weather.py now treats it
    # as a constant value (no triangular shape).
    rainfall_peak_min: int = 10  # minutes from start to PEAK rainfall (the triangle's
    # time-to-peak when rainfall_duration_min is set; ignored under constant rain)
    stormtide_mean_m: float = 2.0
    stormtide_amplitude_m: float = 0.8
    stormtide_period_h: float = 12.0
    # Compound coastal-pluvial event shaping (2026-06-15). When set, the synth weather
    # becomes a realistic storm: a triangular rain burst + a base tide sinusoid with a
    # triangular surge co-peaking with the rain, then recession/drainage.
    rainfall_duration_min: int | None = None  # None=constant rain; else triangular window, peak at rainfall_peak_min
    stormsurge_peak_m: float = 0.0  # triangular surge added to base tide (0=none), peaks at rainfall_peak_min
    compound_event: bool = False  # True: event 0 = rain + tide+surge; False: legacy (0 hydro/1 BC/2 both)
    manhole_diameter_m: float = 1.2  # iter-18 (2026-04-29): reverted from
    # iter-14's 2.0 m back to the toolkit
    # default 1.2 m per user request.
    manhole_loss_coefficient: float = 0.1
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class SyntheticCaseArtifacts:
    cache_dir: Path
    dem: Path
    landuse: Path
    landuse_lookup: Path
    watershed: Path
    boundary: Path
    swmm_hydraulics: Path
    swmm_hydrology: Path
    swmm_full: Path
    subcatchment_raingage_mapping: Path
    weather: Path
    tritonswmm_cfg: Path


def _assert_rim_matches_dem(params, dem_path: Path, swmm_full_path: Path) -> None:
    """Verify that each SWMM junction's rim elevation equals the DEM cell
    elevation at its coordinate. Enforces the user-locked invariant in the
    Phase 2 STOP-gate iteration-2 feedback: rim(node) == DEM(col, row)."""
    import rioxarray as rxr  # local import — avoids GDAL-import-order crash
    import swmmio

    from . import swmm_template as _t

    dem = rxr.open_rasterio(dem_path).squeeze()
    m = swmmio.Model(str(swmm_full_path))
    cs = params.cell_size_m
    x0 = params.xllcorner
    y0 = params.yllcorner
    node_tables = [m.inp.junctions, m.inp.outfalls]
    coords = m.inp.coordinates
    for tbl in node_tables:
        for name, row in tbl.iterrows():
            x = float(coords.at[name, "X"])
            y = float(coords.at[name, "Y"])
            col = int(round((x - x0) / cs - 0.5))
            grid_row = int(round((y - y0) / cs - 0.5))
            matrix_row = params.n_rows - 1 - grid_row
            dem_elev = float(dem.values[matrix_row, col])
            depth = float(row.get("MaxDepth", 0.0)) if "MaxDepth" in row.index else _t._JUNCTION_DEPTH_M
            rim = float(row["InvertElev"]) + depth
            if abs(rim - dem_elev) > 1e-3:
                raise AssertionError(
                    f"node {name!r} rim {rim:.3f} != DEM {dem_elev:.3f} at (col={col}, row_from_bottom={grid_row})"
                )


def build_synthetic_case(params: SyntheticModelParams, dest_dir: Path) -> SyntheticCaseArtifacts:
    """Build the full synthetic TRITON-SWMM case under ``dest_dir`` and return
    the artifact paths.

    Cache-root-agnostic: the caller owns ``dest_dir`` (no platformdirs, no
    filelock, no ``build.complete`` sentinel — those live in the test-tier
    ``get_or_build_synthetic_case`` wrapper). ``dest_dir`` is created if absent.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dem = geometry.build_dem(params, dest_dir / "dem.tif")
    landuse.build_landuse(params, dest_dir / "landuse.tif")
    landuse.build_lookup(params, dest_dir / "landuse_lookup.csv")
    vectors.build_watershed(params, dest_dir / "watershed.geojson")
    vectors.build_boundary(params, dest_dir / "boundary.geojson")
    _hydraulics, _hydrology, full = swmm_template.build_templates(params, dest_dir)
    swmm_template.build_subcatchment_raingage_mapping(params, dest_dir / "subcatchment_raingage_mapping.csv")
    weather.build_weather(params, dest_dir / "weather.nc")
    triton_cfg.build_cfg(params, dest_dir / "tritonswmm.cfg")
    _assert_rim_matches_dem(params, dem, full)
    return SyntheticCaseArtifacts(
        cache_dir=dest_dir,
        dem=dest_dir / "dem.tif",
        landuse=dest_dir / "landuse.tif",
        landuse_lookup=dest_dir / "landuse_lookup.csv",
        watershed=dest_dir / "watershed.geojson",
        boundary=dest_dir / "boundary.geojson",
        swmm_hydraulics=dest_dir / "swmm_hydraulics.inp",
        swmm_hydrology=dest_dir / "swmm_hydrology.inp",
        swmm_full=dest_dir / "swmm_full.inp",
        subcatchment_raingage_mapping=dest_dir / "subcatchment_raingage_mapping.csv",
        weather=dest_dir / "weather.nc",
        tritonswmm_cfg=dest_dir / "tritonswmm.cfg",
    )
