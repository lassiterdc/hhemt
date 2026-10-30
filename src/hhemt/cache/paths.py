"""WHERE the toolkit's on-disk caches live, and WHAT addresses one checkout's tier.

Four roots sit under one platform cache directory, and the parties that touch them are
not all in one place: the pytest fixtures write into them, the run-dir reaper deletes
from them, the Snakefile byte-identity goldens normalize the root away, and the
``hhemt cache status`` verb reports on them. They agree on the answer by importing THIS
module. A second copy of the derivation is not a style problem: a producer writing into
one root while a sweeper deletes another is a data-loss bug that presents as an
unexplained cache miss.

    T0  model artifacts   ``{cache}/synthetic_test_models/``                  shared by every checkout
    T1  analysis trees    ``{cache}/synthetic_test_runs/{address}/``          per checkout
    T2  compiled builds   ``{cache}/synthetic_test_runs/{address}/_software/`` per checkout
    T3  source checkout   ``{cache}/_triton_canonical/triton``                shared by every checkout

T0 and T3 are shared because one is content-keyed and the other is a read-mostly object
store; T1 and T2 are per-address because two checkouts building at the same time must
not share a run tree or a CMake build directory. Concrete failure that motivated the
split, measured before the address existed: three cluster checkouts fell onto ONE tier
and shared one compiled-solver directory, so a rebuild in one corrupted the other.

The address itself is derived from the CHECKOUT a path belongs to, never from the
process working directory. ``pytest tests/`` does not change directory, so a
working-directory derivation gives one checkout two addresses -- each cold for the
other -- and every status surface then reports a machine that looks cold while its
trees sit on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import platformdirs


def cache_root() -> Path:
    """The single platform cache directory every root below is nested under."""
    # The application name is spelled as a LITERAL here rather than lifted into a constant,
    # and the reason is a measurement: "the derivation lives in exactly one place" is checked
    # by grepping for this literal, so a constant would satisfy the code and silently empty
    # the instrument that verifies it.
    return Path(platformdirs.user_cache_dir("hhemt"))


def synthetic_models_root() -> Path:
    """T0 -- generated model artifacts (DEM, landuse, watershed, boundary, ``.inp``s, weather).

    Shared by every checkout on the machine because each directory under it is keyed by a
    hash of the generator source, the params and the installed version; two checkouts at
    the same key produce the same bytes, so sharing is safe and saves a full regeneration.
    """
    return cache_root() / "synthetic_test_models"


def synthetic_runs_root() -> Path:
    """T1's parent -- the un-addressed root holding one directory per tier address."""
    return cache_root() / "synthetic_test_runs"


def slug_runs_root(address: str) -> Path:
    """T1 for one tier address: that checkout's analysis trees and ``_status`` flags."""
    return synthetic_runs_root() / address


def software_root(address: str) -> Path:
    """T2 for one tier address: the compiled solver builds.

    Nested INSIDE T1's address directory rather than beside it, because the build dirs bake
    absolute paths in and are deliberately non-relocatable: keeping them under the same
    address means the directory a reaper must not touch and the directory it may are one
    nesting apart, which is why ``_NEVER_REAP`` names ``_software`` explicitly.
    """
    return slug_runs_root(address) / "_software"


def canonical_root() -> Path:
    """T3 -- the shared TRITON object store that working-tree clones borrow objects from.

    Deliberately a SIBLING of ``synthetic_test_runs/`` and never nested inside it, so the
    run-dir reaper cannot target it by walking its own root. A reaped canonical store costs
    every later worktree a full re-clone.
    """
    return cache_root() / "_triton_canonical" / "triton"


def checkout_root(start: Path | None = None) -> Path | None:
    """The nearest enclosing git checkout of ``start``, or ``None`` when there is none.

    ``start`` defaults to THIS file, so the no-argument call answers "which checkout does
    the installed toolkit live in". The nearest ``.git`` ancestor is the checkout, whether
    that ``.git`` is a directory (a clone) or a file (a linked worktree), which is what
    makes a ``.claude/worktrees/{name}`` tree answer with its own root rather than with the
    main tree's.

    ``None`` rather than a cwd fallback is deliberate: two callers need opposite fallbacks.
    A reporter wants a stable address even when run outside any checkout, while the reaper
    asks the same question about a recorded path that may live anywhere, and there a cwd
    fallback would answer with the reaper's own directory instead of the path's. The
    fallback therefore lives at the call site, not here.
    """
    origin = Path(__file__).resolve() if start is None else Path(start).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def tier_address(cwd: Path) -> str:
    """The cache-tier address of the checkout at ``cwd``. SINGLE SOURCE.

    ``cwd`` must already be a checkout ROOT and already canonical; this function does not
    resolve it. Callers holding an arbitrary path want :func:`tier_address_for`.

    TWO ARMS.
      * Under ``.claude/worktrees/{name}/`` the address is ``{name}``, so the reaper's
        worktree arm, which matches a tier name against a git worktree basename, keeps
        working.
      * Anywhere else the address is ``path-`` followed by the first 16 hex digits of the
        SHA-256 of the root's string form: 21 characters for every checkout.

    WHY A FIXED-LENGTH DIGEST. The address is one segment of every synthetic run path, and
    SWMM keeps a file name only up to MAXFNAME (259) characters. The previous address
    escaped the whole checkout path into this segment, so it grew with the checkout's own
    path: measured on Rivanna, a 77-character address under a 78-character cache root made
    a sensitivity member's ``swmm_flooding_debug.txt`` path 272 characters long. This
    address is 21 characters wherever the checkout lives.

    WHAT THE DIGEST GIVES UP, stated rather than hidden. The escaped form was injective; a
    64-bit digest is only collision-resistant, so two checkouts would share a tier if their
    paths collided in 64 bits. The address is also not decodable: read a tier's checkout
    from its ``.origin`` breadcrumb.

    The ``path`` prefix is kept on purpose: the reaper's fail-safe for an unreadable
    ``.origin`` keys on it. The hyphen keeps every digest address distinct from the
    previous escaped form, which always continued ``path_s``.
    """
    parts = cwd.parts
    if ".claude" in parts:
        i = parts.index(".claude")
        if i + 1 < len(parts) and parts[i + 1] == "worktrees" and i + 2 < len(parts):
            return parts[i + 2]
    digest = hashlib.sha256(str(cwd).encode("utf-8", "surrogateescape")).hexdigest()
    return "path-" + digest[:16]


def tier_address_for(path: Path) -> str:
    """The address of the checkout containing ``path``, or of ``path`` itself if none does.

    This is the entry point for any caller holding a path rather than a checkout root, and
    it is what makes a subdirectory invocation and a root invocation of one checkout agree:
    both resolve to the same root before the address is minted. A path inside no checkout
    keeps the pre-resolution behaviour, so an uninstalled or unpacked-tree caller still
    gets a stable answer instead of an exception.
    """
    root = checkout_root(path)
    return tier_address(root if root is not None else Path(path).resolve())
