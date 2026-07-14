"""Cache orchestrator for synthetic test models (test-tier wrapper over the
lifted ``hhemt.synthetic_model`` builder).

The deterministic ``(params) -> files`` generation surface now lives in
``hhemt.synthetic_model`` (``SyntheticModelParams``, ``SyntheticCaseArtifacts``,
``build_synthetic_case``). This module keeps the TEST-ONLY cache-reuse
orchestration — the platformdirs cache root, the ``build.lock`` filelock, and the
``build.complete`` sentinel — and delegates the actual build. Import direction is
tests -> src.

Cache layout:
    <user_cache_dir>/hhemt/synthetic_test_models/<key>/
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

import platformdirs

from hhemt._filelock_compat import resolve_filelock
from hhemt.synthetic_model import (
    SyntheticCaseArtifacts,
    SyntheticModelParams,
    build_synthetic_case,
)

# Re-export so existing ``from tests.fixtures.synthetic_model.cache import
# SyntheticModelParams`` callsites keep resolving.
__all__ = [
    "DEFAULT_PARAMS",
    "SyntheticCaseArtifacts",
    "SyntheticModelParams",
    "get_or_build_synthetic_case",
]


DEFAULT_PARAMS = SyntheticModelParams()


def _toolkit_version() -> str:
    try:
        return importlib.metadata.version("hhemt")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+unknown"


# Computed once at import time. The SHA-1 of ALL synthetic-model generator source
# bytes (every ``*.py`` under ``src/hhemt/synthetic_model/``) is baked into the
# cache key so ANY topology / DEM-geometry / weather / forcing edit — none of which
# changes ``SyntheticModelParams`` — auto-invalidates every existing synthetic-model
# cache directory. Hashing ONLY ``swmm_template.py`` (the prior behavior) left
# geometry.py, weather.py, landuse.py, vectors.py, triton_cfg.py and _build.py
# cache-blind, so a DEM/node-placement or rainfall/surge edit silently served stale
# artifacts. Computing at import time raises an observable ImportError if any source
# is unreadable (far more diagnosable than an opaque cache-miss inside _cache_key).
def _generator_source_hash() -> str:
    import hhemt.synthetic_model as _sm

    pkg_dir = Path(_sm.__file__).parent
    h = hashlib.sha1()
    for src in sorted(pkg_dir.glob("*.py")):
        h.update(src.name.encode("utf-8"))
        h.update(src.read_bytes())
    return h.hexdigest()[:16]


_GENERATOR_SOURCE_HASH: str = _generator_source_hash()


def _cache_key(params: SyntheticModelParams) -> str:
    payload = {
        "params": dataclasses.asdict(params),
        "toolkit_version": _toolkit_version(),
        "generator_source_hash": _GENERATOR_SOURCE_HASH,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _cache_root() -> Path:
    return Path(platformdirs.user_cache_dir("hhemt")) / "synthetic_test_models"


def get_or_build_synthetic_case(
    params: SyntheticModelParams = DEFAULT_PARAMS,
) -> SyntheticCaseArtifacts:
    key = _cache_key(params)
    cache_dir = _cache_root() / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = resolve_filelock(str(cache_dir / "build.lock"))
    sentinel = cache_dir / "build.complete"
    with lock:
        if not sentinel.exists():
            build_synthetic_case(params, cache_dir)
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
