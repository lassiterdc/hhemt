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
    swmm_template,
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


# Computed once at import time. The SHA-1 of swmm_template.py source bytes is
# baked into the cache key so topology changes in swmm_template.py (which do
# NOT change SyntheticModelParams) auto-invalidate every existing
# synthetic-model cache directory. A params-only cache key would serve stale
# artifacts after a topology edit; this constant closes that gap. Computing
# at import time raises an observable ImportError if the source is unreadable
# (far more diagnosable than an opaque cache-miss surfaced inside _cache_key).
_SWMM_TEMPLATE_SOURCE_HASH: str = hashlib.sha1(Path(swmm_template.__file__).read_bytes()).hexdigest()[:16]


def _cache_key(params: SyntheticModelParams) -> str:
    payload = {
        "params": dataclasses.asdict(params),
        "toolkit_version": _toolkit_version(),
        "swmm_template_source_hash": _SWMM_TEMPLATE_SOURCE_HASH,
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
