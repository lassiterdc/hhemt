"""Cache orchestrator for synthetic test models.

Cache layout:
    <user_cache_dir>/TRITON_SWMM_toolkit/synthetic_test_models/<key>/
        dem.tif
        landuse.tif
        landuse_lookup.csv
        watershed.geojson
        boundary.geojson
        swmm_hydraulics.inp
        swmm_hydrology.inp
        swmm_full.inp
        subcatchment_raingage_mapping.csv
        weather.nc
        tritonswmm.cfg
        build.lock (filelock)
        build.complete (sentinel written after all artifacts exist)
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path

import filelock
import platformdirs

from tests.fixtures.synthetic_model import (
    geometry,
    landuse,
    swmm_template,
    triton_cfg,
    vectors,
    weather,
)


@dataclasses.dataclass(frozen=True)
class SyntheticModelParams:
    n_cols: int = 16   # iter-8 peak_flood_depth: 20→16 (narrower DEM so the
                       # figure has less surrounding wall area and the Y-shaped
                       # channel reads as the dominant feature).
    n_rows: int = 30
    cell_size_m: float = 10.0
    xllcorner: float = 0.0            # test-case local origin (no real-world geolocation)
    yllcorner: float = 0.0
    epsg: int = 3857                  # Web Mercator — accepts (0,0) at equator/prime meridian
    slope_ns: float = 0.01            # 1% N->S
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
    rainfall_peak_min: int = 10  # legacy field; ignored under iter-9 constant rainfall
    stormtide_mean_m: float = 2.0
    stormtide_amplitude_m: float = 0.8
    stormtide_period_h: float = 12.0
    manhole_diameter_m: float = 1.2  # iter-18 (2026-04-29): reverted from
                                     # iter-14's 2.0 m back to the toolkit
                                     # default 1.2 m per user request.
    manhole_loss_coefficient: float = 0.1
    seed: int = 0


DEFAULT_PARAMS = SyntheticModelParams()


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


def _toolkit_version() -> str:
    try:
        return importlib.metadata.version("TRITON-SWMM-toolkit")
    except importlib.metadata.PackageNotFoundError:
        try:
            return importlib.metadata.version("TRITON_SWMM_toolkit")
        except importlib.metadata.PackageNotFoundError:
            return "0.0.0+unknown"


# Computed once at import time. The SHA-1 of swmm_template.py source bytes is
# baked into the cache key so topology changes in swmm_template.py (which do
# NOT change SyntheticModelParams) auto-invalidate every existing
# synthetic-model cache directory. A params-only cache key would serve stale
# artifacts after a topology edit; this constant closes that gap. Computing
# at import time raises an observable ImportError if the source is unreadable
# (far more diagnosable than an opaque cache-miss surfaced inside _cache_key).
_SWMM_TEMPLATE_SOURCE_HASH: str = hashlib.sha1(
    Path(swmm_template.__file__).read_bytes()
).hexdigest()[:16]


def _cache_key(params: SyntheticModelParams) -> str:
    payload = {
        "params": dataclasses.asdict(params),
        "toolkit_version": _toolkit_version(),
        "swmm_template_source_hash": _SWMM_TEMPLATE_SOURCE_HASH,
    }
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


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
                    f"node {name!r} rim {rim:.3f} != DEM {dem_elev:.3f} at "
                    f"(col={col}, row_from_bottom={grid_row})"
                )


def _cache_root() -> Path:
    return Path(platformdirs.user_cache_dir("TRITON_SWMM_toolkit")) / "synthetic_test_models"


def get_or_build_synthetic_case(
    params: SyntheticModelParams = DEFAULT_PARAMS,
) -> SyntheticCaseArtifacts:
    key = _cache_key(params)
    cache_dir = _cache_root() / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = filelock.FileLock(str(cache_dir / "build.lock"))
    sentinel = cache_dir / "build.complete"
    with lock:
        if not sentinel.exists():
            dem = geometry.build_dem(params, cache_dir / "dem.tif")
            landuse_tif = landuse.build_landuse(params, cache_dir / "landuse.tif")
            lookup = landuse.build_lookup(params, cache_dir / "landuse_lookup.csv")
            watershed = vectors.build_watershed(params, cache_dir / "watershed.geojson")
            boundary = vectors.build_boundary(params, cache_dir / "boundary.geojson")
            hydraulics, hydrology, full = swmm_template.build_templates(params, cache_dir)
            mapping = swmm_template.build_subcatchment_raingage_mapping(
                params, cache_dir / "subcatchment_raingage_mapping.csv"
            )
            weather_nc = weather.build_weather(params, cache_dir / "weather.nc")
            cfg = triton_cfg.build_cfg(params, cache_dir / "tritonswmm.cfg")
            _assert_rim_matches_dem(params, dem, full)
            _ = (dem, landuse_tif, lookup, watershed, boundary, hydraulics,
                 hydrology, full, mapping, weather_nc, cfg)
            sentinel.write_text("ok\n", encoding="utf-8")
    return SyntheticCaseArtifacts(
        cache_dir=cache_dir,
        dem=cache_dir / "dem.tif",
        landuse=cache_dir / "landuse.tif",
        landuse_lookup=cache_dir / "landuse_lookup.csv",
        watershed=cache_dir / "watershed.geojson",
        boundary=cache_dir / "boundary.geojson",
        swmm_hydraulics=cache_dir / "swmm_hydraulics.inp",
        swmm_hydrology=cache_dir / "swmm_hydrology.inp",
        swmm_full=cache_dir / "swmm_full.inp",
        subcatchment_raingage_mapping=cache_dir / "subcatchment_raingage_mapping.csv",
        weather=cache_dir / "weather.nc",
        tritonswmm_cfg=cache_dir / "tritonswmm.cfg",
    )
