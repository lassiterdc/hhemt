"""
Test fixtures for TRITON-SWMM toolkit.

This package contains test infrastructure for creating isolated test cases
with synthetic weather data and platform-specific configurations.
"""

from pathlib import Path


def worktree_slug() -> str:
    """Return the worktree slug derived from ``Path.cwd()``.

    Matches the current working directory against ``.claude/worktrees/{slug}/``
    and returns ``{slug}``. Falls back to ``"main"`` when not inside a worktree.
    Canonical shared utility consumed by ``test_case_builder.py`` and
    ``test_case_catalog.py`` to root per-worktree test artifacts under a
    contention-free path so concurrent pytest runs in sibling worktrees do not
    fight over a single shared ``synthetic_test_runs/`` cache.
    """
    cwd = Path.cwd().resolve()
    parts = cwd.parts
    if ".claude" in parts:
        i = parts.index(".claude")
        if i + 1 < len(parts) and parts[i + 1] == "worktrees" and i + 2 < len(parts):
            return parts[i + 2]
    return "main"


from tests.fixtures.test_case_builder import retrieve_TRITON_SWMM_test_case
from tests.fixtures.test_case_catalog import GetTS_TestCases

__all__ = [
    "retrieve_TRITON_SWMM_test_case",
    "GetTS_TestCases",
    "worktree_slug",
]
