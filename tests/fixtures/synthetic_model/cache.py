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
    n_cols: int = 20
    n_rows: int = 30
    cell_size_m: float = 10.0
    xllcorner: float = 400000.0       # EPSG:32618 UTM zone 18N
    yllcorner: float = 4075000.0
    epsg: int = 32618
    slope_ns: float = 0.01            # 1% N->S
    valley_depth_m: float = 0.5
    impervious_mannings: float = 0.015
    pervious_mannings: float = 0.035
    sim_duration_min: int = 10
    triton_timestep_s: float = 1.0
    reporting_timestep_s: float = 10.0
    rainfall_peak_mm_per_hr: float = 100.0
    rainfall_peak_min: int = 3
    stormtide_mean_m: float = 2.0
    stormtide_amplitude_m: float = 0.8
    stormtide_period_h: float = 12.0
    manhole_diameter_m: float = 1.2
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
