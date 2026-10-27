"""Reviewer tests for the synthetic cache's tier address, `checkout_slug`.

The tier address is the one path segment that decides which checkouts share a compiled solver
tree and a run directory. The coder's file `tests/test_synth_cache_path_budget.py` bounds the
address's LENGTH and the reaper's liveness arms, and it holds the address's identity properties
nowhere: a function returning the same string for every checkout passes all of it. These tests
hold the identity properties, over the input classes the function's own docstring states.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from tests.fixtures import worktree_slug
from tests.fixtures._triton_source_cache import checkout_slug

_DIGEST_ADDRESS = re.compile(r"^path-[0-9a-f]{16}$")


@pytest.mark.parametrize(
    ("cwd", "expected"),
    [
        ("/home/u/dev/hhemt/.claude/worktrees/build-foo", "build-foo"),
        ("/home/u/dev/hhemt/.claude/worktrees/build-foo/tests/fixtures", "build-foo"),
        ("/home/u/dev/hhemt/.claude/worktrees/w", "w"),
    ],
)
def test_a_cwd_inside_a_worktree_is_addressed_by_the_worktree_name(cwd, expected):
    """Property: under `.claude/worktrees/{name}/`, at any depth, the address is `{name}`.

    Class: every cwd whose first `.claude` component is followed by `worktrees` and a name.
    Why required: the reaper's worktree arm matches a tier directory name against the basename
    of a registered git worktree, so a worktree checkout addressed by anything but its name
    loses that protection and its live tier becomes reapable once the TTL passes.
    What kills it: dropping the worktree arm from `checkout_slug`, so a worktree falls through
    to the digest address.
    Second correct implementation: one that finds the name with a regular expression over the
    path string instead of walking `Path.parts`; it returns the same names and still passes.
    """
    assert checkout_slug(Path(cwd)) == expected


@pytest.mark.parametrize(
    "cwd",
    [
        "/home/u/.claude/skills/build-foo",
        "/home/u/.claude/worktrees",
        "/home/u/dev/worktrees/build-foo",
        "/home/u/dev/.claude/build-foo",
    ],
)
def test_a_cwd_that_only_resembles_a_worktree_gets_the_digest_address(cwd):
    """Property: only `.claude/worktrees/{name}` selects the worktree arm; near misses do not.

    Class: cwds that carry `.claude`, `worktrees` or both, but not as an adjacent pair followed
    by a name.
    Why required: treating a near miss as a worktree names a tier after a directory that is not
    a worktree, so the reaper's worktree arm can never match it and two unrelated checkouts
    that share a directory name share a tier.
    What kills it: matching on `worktrees` anywhere in the path, or accepting the pair with no
    name after it.
    Second correct implementation: any other derivation that returns the digest address for
    these inputs, for example a regular expression anchored on the adjacent pair.
    """
    assert _DIGEST_ADDRESS.match(checkout_slug(Path(cwd)))


def test_distinct_checkouts_get_distinct_digest_addresses():
    """Property: the digest address depends on the whole checkout path and nothing else.

    Class: checkout paths outside a worktree that share a basename, share a parent, or differ
    only in a deep prefix. The harness builds many checkouts whose last component is the same
    name, so basename-keyed or parent-keyed addresses collide in practice.
    Why required: the retired address was injective, and a constant or partial key is the exact
    failure the `NO CONSTANT FALLBACK` comment in `tests/fixtures/__init__.py` records, in which
    several checkouts shared one compiled solver tier.
    What kills it: hashing only `cwd.name`, only `cwd.parent`, or a constant.
    Second correct implementation: a different 64-bit hash, rendered as 16 hex digits, over
    `str(cwd)`; every property asserted here still holds.
    """
    cwds = [
        "/scratch/u/build_a/tree",
        "/scratch/u/build_b/tree",
        "/scratch/u/build_a/tree2",
        "/scratch/u/build_a/tre",
        "/scratch/v/build_a/tree",
        "/home/u/dev/hhemt",
        "/home/u/dev/hhemt_estate",
        "/tmp/tree",
    ]
    addresses = [checkout_slug(Path(c)) for c in cwds]
    assert all(_DIGEST_ADDRESS.match(a) for a in addresses), addresses
    assert len(set(addresses)) == len(cwds), dict(zip(cwds, addresses, strict=True))
    assert addresses == [checkout_slug(Path(c)) for c in cwds], "the address is not deterministic"


def test_a_cwd_with_undecodable_bytes_still_gets_a_digest_address():
    """Property: a checkout whose path holds bytes that are not UTF-8 is addressable, and uniquely.

    Class: cwds produced by `os.fsdecode` of a byte path with a non-UTF-8 byte, which is how
    `Path.cwd()` reports such a directory on Linux.
    Why required: the address is minted at fixture setup for whatever checkout the suite runs
    from, so raising there fails every synthetic test for that checkout, where the retired
    address handled it.
    What kills it: encoding the path with the strict error handler instead of `surrogateescape`.
    Second correct implementation: encoding with `os.fsencode`, which round-trips the same
    bytes.
    """
    one = Path(os.fsdecode(b"/tmp/\xff/tree"))
    other = Path(os.fsdecode(b"/tmp/\xfe/tree"))
    assert _DIGEST_ADDRESS.match(checkout_slug(one))
    assert checkout_slug(one) != checkout_slug(other)


def test_worktree_slug_follows_the_current_directory(tmp_path, monkeypatch):
    """Property: `worktree_slug()` is `checkout_slug` of the directory the process is in.

    Class: the cwd shapes `checkout_slug` distinguishes, reached through the public function
    the builder and the catalog call, in a temporary directory so no real cache is touched.
    Why required: the builder writes under this address and the reaper re-derives it with
    `checkout_slug`; a `worktree_slug` that ignores or mis-reads the cwd makes the two disagree.
    What kills it: `worktree_slug` returning `checkout_slug` of a fixed path.
    Second correct implementation: one that reads the directory from `os.getcwd()` and resolves
    it; it returns the same values.
    """
    root = tmp_path.resolve()
    if ".claude" in root.parts:
        pytest.skip("the temporary directory sits under a `.claude` component; the arms cannot be separated")
    worktree = root / ".claude" / "worktrees" / "wt_a"
    plain_one = root / "one"
    plain_two = root / "two"
    for d in (worktree, plain_one, plain_two):
        d.mkdir(parents=True)
    seen = {}
    for label, d in (("worktree", worktree), ("one", plain_one), ("two", plain_two)):
        monkeypatch.chdir(d)
        seen[label] = worktree_slug()
    assert seen["worktree"] == "wt_a"
    assert _DIGEST_ADDRESS.match(seen["one"]) and _DIGEST_ADDRESS.match(seen["two"])
    assert seen["one"] != seen["two"]
