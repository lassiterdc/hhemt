#!/usr/bin/env python3
"""Print the cache key under which the compile gate stores and finds its pass record.

PROBLEM. `compile-tests.yml` runs the compile-bearing tier on every push to `develop`, and a
push that changed only prose or release metadata waited 32 to 50 minutes for a compile of
sources that had not moved. A path filter on the workflow is not an answer: GitHub leaves a
path-filtered workflow's checks Pending, which blocks a required context, and a filter keys
on a push's diff from its parent, so it cannot see whether the parent passed. This script
keys a PASS RECORD on the content the gate tests instead. The workflow looks the key up; a hit
means a full run on a fresh runner already passed over a byte-identical compile-relevant tree.

THE KEY. `compile-gate-pass-v1-{runner os}-{ISO year-week}-{sha256}`.

  * The sha256 is taken over the `git ls-tree -r --full-tree HEAD` listing, one line per
    tracked path (`{mode} {blob id} {path}`), after the INERT set is removed. A blob id is a
    function of file content, so the key moves when a compile-relevant file's bytes, mode or
    name move and does not depend on commit history, author or message.
  * The INERT set is every path under `docs/` or `.vale/`, plus the root-level files
    `HISTORY.md`, `CONTRIBUTING.md`, `CITATION.cff`, `.vale.ini`, `mkdocs.yml` and
    `.readthedocs.yaml`. A path the set does not name is IN the key, which is the fail-safe
    direction: an unrecognised file makes the gate run in full. `README.md` is deliberately
    not inert, because `pyproject.toml` declares `readme = "README.md"` and the editable
    install reads it. `pyproject.toml` and `uv.lock` stay whole, so a version bump compiles.
  * The ISO year-week (`date -u +%G-W%V`) makes a record expire with its calendar week:
    `environment.yaml` leaves most conda packages unconstrained, so a fresh solve on an
    unchanged tree can resolve different packages on a later day.

EXIT STATUS. 0 with the key as the only line on stdout. 2 on a git error or a listing with
nothing left to hash. Deliberately NOT an empty digest: the shell form of this key printed
`e3b0c44298fc1c14` (the sha256 of nothing) for four different commits when git refused its
pathspec, and one constant key for every commit would reuse a pass for code that never ran.

Not established: whether a version string alone can change a compile-tier result. No run has
varied it, so `pyproject.toml` stays in the key and a version bump runs the gate in full.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

KEY_PREFIX = "compile-gate-pass-v1"

INERT_PREFIXES: tuple[bytes, ...] = (b"docs/", b".vale/")
# Root-level names only. `--full-tree` paths are repository-relative, so `src/HISTORY.md`
# is a different path and is not inert.
INERT_NAMES: frozenset[bytes] = frozenset(
    {
        b"HISTORY.md",
        b"CONTRIBUTING.md",
        b"CITATION.cff",
        b".vale.ini",
        b"mkdocs.yml",
        b".readthedocs.yaml",
    }
)


class GateKeyError(Exception):
    """Raised when no trustworthy key can be computed; `main` maps it to exit status 2."""


def is_inert(path: bytes) -> bool:
    """True when a tracked path cannot change a compile-tier result and stays out of the key."""
    return path in INERT_NAMES or path.startswith(INERT_PREFIXES)


def tree_listing(repo: Path) -> list[tuple[bytes, bytes, bytes]]:
    """The `(mode, blob id, path)` entries of HEAD's full tree, in git's own order.

    `-z` is used rather than the default quoted form, because git quotes a path holding a
    space-adjacent special character or non-ASCII byte and a quoted spelling would hash
    differently from the path the inert test reads.
    """
    done = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--full-tree", "-z", "HEAD"],
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        detail = done.stderr.decode("utf-8", "replace").strip()
        raise GateKeyError(f"git ls-tree exited {done.returncode} in {repo}: {detail}")
    entries = []
    for record in done.stdout.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, _type, blob = meta.split(b" ")
        entries.append((mode, blob, path))
    if not entries:
        raise GateKeyError(f"git ls-tree printed an empty listing in {repo}")
    return entries


def tree_digest(entries: list[tuple[bytes, bytes, bytes]]) -> str:
    """The sha256 hex digest of the listing with the inert set removed."""
    kept = [(mode, blob, path) for mode, blob, path in entries if not is_inert(path)]
    if not kept:
        raise GateKeyError("every tracked path is inert, so there is no compile-relevant content to hash")
    digest = hashlib.sha256()
    for mode, blob, path in kept:
        digest.update(mode + b" " + blob + b" " + path + b"\n")
    return digest.hexdigest()


def iso_week_now() -> str:
    """The current UTC ISO year and week, spelled as `date -u +%G-W%V` spells it."""
    # `timezone.utc`, not `datetime.UTC`: this script runs before conda is provisioned, on the runner's system
    # python3, and `datetime.UTC` raises AttributeError below Python 3.11, which would fail the key step.
    return datetime.datetime.now(datetime.timezone.utc).strftime("%G-W%V")  # noqa: UP017


def build_key(repo: Path, runner_os: str, iso_week: str) -> str:
    """Assemble the cache key for the tree at HEAD of `repo`."""
    return f"{KEY_PREFIX}-{runner_os}-{iso_week}-{tree_digest(tree_listing(repo))}"


def main(argv: list[str] | None = None) -> int:
    """Print the key and return 0, or print the reason to stderr and return 2."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=".", help="repository whose HEAD tree is keyed (default: cwd)")
    parser.add_argument("--iso-week", default=None, help="override the UTC ISO year-week (tests only)")
    args = parser.parse_args(argv)
    runner_os = os.environ.get("RUNNER_OS") or platform.system()
    try:
        key = build_key(Path(args.repo), runner_os, args.iso_week or iso_week_now())
    except (GateKeyError, OSError) as error:
        print(f"compile_gate_key: {error}", file=sys.stderr)
        return 2
    print(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
