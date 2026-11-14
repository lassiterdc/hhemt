"""Reviewer tests for the bounded tier address of a long-named worktree (work package 1a, chunk 1a.1).

The coder's tests in `tests/test_review_synth_cache_path.py` pin three points of the worktree-name
axis (21, 22 and 63 characters) and pin the digest to one exact function of the root. These tests
hold the two PROPERTIES those points are samples of, over classes the points do not enumerate: the
bound on the address for every name length, and the distinctness of the digest addresses of
worktrees that share a name or a long prefix. Both are solver-free and touch no real cache.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.fixtures._triton_source_cache import checkout_slug

_WORKTREES = Path("/srv/alpha/repo/.claude/worktrees")
_DIGEST_ADDRESS = re.compile(r"^path-[0-9a-f]{16}$")
_DIGEST_ADDRESS_LENGTH = len("path-") + 16


def test_a_worktree_address_never_exceeds_the_digest_length_and_is_the_name_only_when_that_fits():
    """Property: for a worktree folder name of any length from 1 to 100 characters, the address is at
    most 21 characters long; it is the name itself exactly when the name is 21 characters or fewer,
    and a `path-` plus 16 hex digits address otherwise.

    Class: REGRESSION -- every ASCII worktree folder name of length 1 through 100. The coder's tests
    sample 21, 22 and 63; this holds the whole range, including the band 22 through 44 whose names
    fit the path budget under a 25-character cache root and the band 45 through 100 whose names do
    not.
    Why required: the address is one segment of every synthetic run path and SWMM keeps a file name
    only up to 259 characters, so an address longer than the digest address for any name length
    re-opens the path-budget red for that length; the bound must hold for every name, not only the
    three sampled.
    What kills it: a threshold above the digest length (a threshold of 44 returns a 44-character
    address at length 44, over the 21 bound), a threshold below it (20 returns a digest at length 21,
    where the name is required), or leaving the worktree arm unbounded as it was at `f22181a3`, which
    returns the name at every length and so exceeds 21 from length 22.
    Second correct implementation: one that returns `name` when `len(name) < 22` and takes the digest
    arm otherwise, or one that hashes with a different function and renders it as 16 hex digits;
    both satisfy every assertion here.
    """
    for length in range(1, 101):
        name = "a" * length
        address = checkout_slug(_WORKTREES / name)
        assert len(address) <= _DIGEST_ADDRESS_LENGTH, (length, address)
        if length <= _DIGEST_ADDRESS_LENGTH:
            assert address == name, (length, address)
        else:
            assert _DIGEST_ADDRESS.match(address), (length, address)


@pytest.mark.parametrize(
    "roots",
    [
        # Same folder name under two different repositories: a name-keyed address would share one tier.
        (
            Path("/srv/alpha/repo/.claude/worktrees") / ("w" * 40),
            Path("/srv/beta/other/.claude/worktrees") / ("w" * 40),
        ),
        # Sibling worktrees that share a 40-character prefix, the shape the run and its chunk trees have.
        (
            Path("/srv/alpha/repo/.claude/worktrees") / ("p" * 40 + "-a"),
            Path("/srv/alpha/repo/.claude/worktrees") / ("p" * 40 + "-b"),
        ),
        # A name that is a prefix of the other, both over the threshold.
        (
            Path("/srv/alpha/repo/.claude/worktrees") / ("q" * 30),
            Path("/srv/alpha/repo/.claude/worktrees") / ("q" * 31),
        ),
    ],
)
def test_two_distinct_long_named_worktrees_get_distinct_addresses(roots):
    """Property: two worktree roots that are different paths, both with a folder name over 21
    characters, get different addresses, whether they share the folder name, a long prefix, or one
    name is a prefix of the other.

    Class: NEW CAPABILITY -- pairs of distinct worktree roots whose names exceed the digest length.
    The plausible wrong implementation is a digest of the folder NAME rather than of the root, or a
    truncation of the name, either of which is a natural way to keep an address derived from the
    worktree name; red under it, pasted in the round file.
    Why required: before this change a worktree address was the name, so two worktrees shared a tier
    only if they shared a name; the digest arm must not make long-named worktrees share one more
    easily, because sharing a tier shares a compiled solver tree and a run directory between
    checkouts that are at different commits. The existing distinctness test excludes worktrees by
    construction, so this class is held nowhere else.
    What kills it: hashing `name` instead of `str(root)` fails the first pair, truncating the name
    to 21 characters fails the second and third.
    Second correct implementation: a different 64-bit hash rendered as 16 hex digits over
    `str(root)`; every assertion here still holds, which the exact-value pin in the coder's test
    would not allow.
    """
    first, second = roots
    assert first != second
    assert len(first.name) > _DIGEST_ADDRESS_LENGTH and len(second.name) > _DIGEST_ADDRESS_LENGTH
    one, other = checkout_slug(first), checkout_slug(second)
    assert _DIGEST_ADDRESS.match(one) and _DIGEST_ADDRESS.match(other), (one, other)
    assert one != other, (first, second, one)
