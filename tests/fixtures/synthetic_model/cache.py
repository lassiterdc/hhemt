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

from hhemt._filelock_compat import resolve_filelock
from hhemt.cache import paths as cache_paths
from hhemt.cache.keys import artifact_stack_digest
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


def _cache_key(params: SyntheticModelParams, *, source_raster: Path | None = None) -> str:
    """The tier-0 key: what the artifact directory under the cache root is NAMED.

    TWO TERMS WERE ADDED to the original (params, version, generator-hash) triple, and both
    answer the same question one level down: what else can change the bytes in this
    directory without any of the original three moving.

      * ``source_raster_digest`` -- the bytes of the DEM raster a case SUPPLIES. It must be
        an argument rather than something read back from this function's own output: the
        synthetic artifacts, ``dem.tif`` among them, are produced by the very call this key
        is an input to (``get_or_build_synthetic_case`` computes the key and then returns
        ``dem=cache_dir / "dem.tif"``), so reading it here would be circular by ordering.
      * ``artifact_stack_digest`` -- the geospatial / serialisation / CSV stack that PRODUCES
        the artifacts below, at the granularities ``hhemt.cache.keys.ENVIRONMENT_COMPONENTS``
        states. Attached to tier 0 rather than tier 1 because these transform a raster or
        write a file before any solver runs; a change here reaches tier 1 only through the
        ``tier0_key`` composition term, which is one mechanism rather than two.

    ``source_raster_digest`` is INERT on this tree, and that is a measurement rather than a
    hope: no case supplies or fetches an external raster, the sole non-smoke caller passes
    ``None``, and ``DEM_fullres`` is written downstream from the synthetic artifacts. So
    landing the term changes no key and invalidates nothing today. What it buys is
    correctness for the first case that does supply one.

    NOT ESTABLISHED: the FETCHED-raster arm of the same term -- a case that downloads its
    raster contributes "the fetch URL plus the digest it resolved to" -- is not implemented,
    because no case in this tree fetches one. A case that starts fetching would pass
    ``source_raster=None`` and be silently unkeyed, which is why the shape's arm for it
    asserts the SUPPLIED side positively rather than waiting for a stale tree.
    """
    payload = {
        "params": dataclasses.asdict(params),
        "toolkit_version": _toolkit_version(),
        "generator_source_hash": _GENERATOR_SOURCE_HASH,
        "source_raster_digest": _source_raster_digest(source_raster),
        "artifact_stack_digest": artifact_stack_digest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def raster_is_tier0_output(path: Path) -> bool:
    """True iff ``path`` resolves inside the tier-0 root, so it is an OUTPUT of this cache.

    The guard exists because the domain of ``source_raster`` is declared by the CALLER and
    this predicate refuses a caller error rather than inferring the domain from a path test.
    Passing a path that lives inside the tier-0 root would make the key depend on an artifact
    the key itself produces -- a self-reference that can never converge -- so it raises
    instead of silently selecting the no-op.

    CANONICAL resolution is part of the guard: a symlink into the cache root is judged by its
    TARGET rather than by its spelling, so a path spelled outside the root cannot smuggle
    itself in.
    """
    try:
        target = Path(path).resolve()
        root = _cache_root().resolve()
    except OSError:
        return False
    return target == root or root in target.parents


def _source_raster_digest(source_raster: Path | None) -> str:
    """SHA-256 of a SUPPLIED source raster's bytes, or ``""`` when none was supplied.

    An unreadable path raises rather than degrading to ``""``: ``""`` is the value that means
    "this case supplies no raster", and a case that meant to supply one and handed over a
    broken path would key itself into the no-raster cache entry and serve stale artifacts.
    """
    if source_raster is None:
        return ""
    if raster_is_tier0_output(source_raster):
        raise ValueError(
            f"source_raster={source_raster!r} resolves inside the tier-0 cache root, so it is "
            "an OUTPUT of the cache being keyed; a key cannot depend on an artifact it "
            "produces. Supply the external raster the cache will consume, or pass None for a "
            "case whose synthetic artifacts are generated here."
        )
    return hashlib.sha256(Path(source_raster).read_bytes()).hexdigest()


def _cache_root() -> Path:
    return cache_paths.synthetic_models_root()


def get_or_build_synthetic_case(
    params: SyntheticModelParams = DEFAULT_PARAMS,
    *,
    source_raster: Path | None = None,
) -> SyntheticCaseArtifacts:
    """The tier-0 artifacts for ``params``, building them once under a ``build.lock``.

    ``source_raster`` is FORWARDED to :func:`_cache_key` and used for nothing else: a cache
    HIT re-derives no raster, because no synthetic artifact is external. It is a
    keyword-only input rather than something this function could discover, for the ordering
    reason the key function states.
    """
    key = _cache_key(params, source_raster=source_raster)
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
