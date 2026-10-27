"""
Test fixtures for TRITON-SWMM toolkit.

This package contains test infrastructure for creating isolated test cases
with synthetic weather data and platform-specific configurations.
"""

from pathlib import Path

from tests.fixtures._triton_source_cache import checkout_slug


def worktree_slug() -> str:
    """Return the cache-tier address of the current checkout.

    Derived from ``Path.cwd().resolve()`` by
    ``tests.fixtures._triton_source_cache.checkout_slug``, whose docstring states both
    arms: the worktree name under ``.claude/worktrees/{name}/``, otherwise a
    fixed-length ``path-`` digest of the checkout path. Canonical shared utility
    consumed by ``test_case_builder.py`` and ``test_case_catalog.py`` to root
    per-checkout test artifacts under a contention-free path, so concurrent pytest
    runs in sibling checkouts do not fight over a single shared
    ``synthetic_test_runs/`` cache.
    """
    # NO CONSTANT FALLBACK. A literal here collapsed every checkout not under
    # `.claude/worktrees/` onto ONE tier: measured, a toolkit clone, a sibling
    # estate clone and /tmp all returned "main", so three cluster
    # checkouts shared one compiled solver tier. `cwd` is already canonical
    # (os.getcwd resolves symlinks); the harness side of that requirement is the
    # canonicalization of HHEMT_TOOLKIT in rerun.sh.
    return checkout_slug(Path.cwd().resolve())


# noqa: E402 justified — test_case_builder and test_case_catalog both do
# `from tests.fixtures import worktree_slug`, so that name must be BOUND before these
# imports run. Hoisting them to the top raises ImportError on a circular import.
from tests.fixtures.test_case_builder import retrieve_TRITON_SWMM_test_case  # noqa: E402
from tests.fixtures.test_case_catalog import GetTS_TestCases  # noqa: E402

__all__ = [
    "retrieve_TRITON_SWMM_test_case",
    "GetTS_TestCases",
    "worktree_slug",
]
