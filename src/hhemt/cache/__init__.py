"""The on-disk cache the test tier and the CLI share: WHERE it lives, WHAT names a tier,
and WHEN a cached tree may be handed out.

Before this package existed the same three answers were re-derived in five files across
``tests/`` and ``tests/fixtures/``, so the toolkit could not report on its own cache
without importing the test tree, and a fixture that moved a root while a sweeper kept the
old copy deleted a live tree. The three submodules are the answers:

  * :mod:`hhemt.cache.paths`   -- the four roots and the per-checkout tier address;
  * :mod:`hhemt.cache.keys`    -- the identity a marker records and compares;
  * :mod:`hhemt.cache.markers` -- the completion-marker state machine and its payload check;
  * :mod:`hhemt.cache.clear`   -- what a clear removes, what it refuses, and its liveness arms.

Nothing here imports ``tests/``: the direction is ``tests/`` -> ``src/``, so the mechanism
is reachable from the installed package (``hhemt cache status``) without a test session.

Importing ``clear`` here does not widen the cost of importing this package. Every module
whose heavy dependencies ``clear`` reaches -- ``filelock`` via ``_filelock_compat`` and
``hhemt.utils`` for ``fast_rmtree`` -- is already imported at module scope by
``tests/fixtures/_run_dir_reaper.py``, which conftest loads in every session, and the CLI
imports this package function-locally precisely so its own import path stays cheap.
"""

from __future__ import annotations

from hhemt.cache.clear import (
    TIER_LABELS,
    TIER_ORDER,
    CacheRoots,
    ClearPlan,
    RootDecision,
    apply_clear,
    detect_liveness,
    plan_clear,
    resolve_roots,
)
from hhemt.cache.keys import builder_digest, builder_provenance, toolkit_sha
from hhemt.cache.markers import (
    FIXTURE_MARKER_REL,
    finish_build,
    marker_state,
    unmet_payload,
    warn_unsatisfied,
)
from hhemt.cache.paths import (
    cache_root,
    canonical_root,
    checkout_root,
    slug_runs_root,
    software_root,
    synthetic_models_root,
    synthetic_runs_root,
    tier_address,
    tier_address_for,
)

__all__ = [
    "FIXTURE_MARKER_REL",
    "TIER_LABELS",
    "TIER_ORDER",
    "CacheRoots",
    "ClearPlan",
    "RootDecision",
    "apply_clear",
    "builder_digest",
    "builder_provenance",
    "cache_root",
    "canonical_root",
    "checkout_root",
    "detect_liveness",
    "finish_build",
    "marker_state",
    "plan_clear",
    "resolve_roots",
    "slug_runs_root",
    "software_root",
    "synthetic_models_root",
    "synthetic_runs_root",
    "tier_address",
    "tier_address_for",
    "toolkit_sha",
    "unmet_payload",
    "warn_unsatisfied",
]
