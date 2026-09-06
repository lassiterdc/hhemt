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
    # NO CONSTANT FALLBACK. A literal here collapsed every checkout not under
    # `.claude/worktrees/` onto ONE tier: measured, a toolkit clone, a sibling
    # estate clone and /tmp all returned "main", so three cluster
    # checkouts shared one compiled solver tier.
    #
    # REVERSIBLE ESCAPE, and the escape character is itself escaped. `/` -> `_s` and
    # `_` -> `_u`, so no rule ever emits a BARE `_`: every `_` in the output begins a
    # two-character escape, the left-to-right decoder never mis-aligns, and
    # decode(encode(p)) == p for every p. A left inverse exists, therefore the map is
    # injective -- a property, not a sampled claim. Do NOT "simplify" this to mapping
    # `/` to one underscore and `_` to two: that puts the escape and the escaped
    # character in the same alphabet, and a run of underscores then has several
    # pre-images (measured: 686 collisions over {_,/,a} strings of length 1-6, minimal
    # counterexample `/_` and `_/` both yielding `___`).
    #
    # NON-INJECTIVITY IS UNAVOIDABLE SOMEWHERE -- no length-bounded encoding of an
    # unbounded input set can be injective, and a directory component caps at 255
    # bytes. This scheme puts the failure where it is LOUD: an over-long path raises
    # OSError at tier creation instead of silently sharing a tier with another
    # checkout. `cwd` is already canonical (os.getcwd resolves symlinks); the harness
    # side of that requirement is rerun.sh:66.
    return "path" + "".join("_s" if ch == "/" else "_u" if ch == "_" else ch for ch in str(cwd))


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
