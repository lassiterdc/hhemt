"""Bind a script or hook to the source tree of the checkout it lives in.

Stdlib only, and it NEVER imports ``hhemt``: this module decides which ``src``
is authoritative, so it cannot itself come from one.

WHY THIS EXISTS. The editable install writes exactly ONE path into
site-packages, so whichever checkout most recently ran ``pip install -e .``
captures every other checkout's plain ``import hhemt``. A script that roots its
inputs on ``Path(__file__)`` and its population on ``import hhemt`` composes two
source trees, and the grade it prints is about neither. Measured 2026-09-12
(``### D115``): ``scripts/check_autodoc_coverage.py`` run bare from a worktree
graded the MAIN clone's 66 symbols against the worktree's built site, reported
one symbol unrendered that the worktree does not define, and reported
``all 67`` under a ``PYTHONPATH`` pin.

THREE HALVES, BECAUSE EACH HAS AN INPUT ONLY IT HANDLES. ``prepend_src`` puts
the checkout's ``src`` at ``sys.path[0]``, ahead of every ``PYTHONPATH`` entry
and of the editable ``.pth``; that is enough for a cold interpreter and does
nothing for a warm one, because ``importlib.import_module`` returns the
``sys.modules`` entry before any finder runs. ``purge_foreign`` evicts the warm
entries whose ``__file__`` lies outside ``src`` and KEEPS entries that claim no
``__file__`` at all: a synthetic module a test injected cannot have come from
the wrong tree, and evicting it turns that fixture into ``ModuleNotFoundError``.
``assert_resolved_under`` reports the case the first two cannot repair: a
``src`` that exists but holds no package directory (a mid-rebase checkout, a
script moved one directory deeper), where the prepend is inert, the ``.pth``
supplies another checkout's package, and nothing else says so. It raises, so a
wrong-tree grade is an error instead of a number.

TWO PREPEND IMPLEMENTATIONS COEXIST BY DESIGN. The repo-root ``conftest.py``
carries its own ``sys.path`` prepend and cannot consume this module: it is
copied alone into a bare pytester rootdir by ``tests/test_worktree_guard.py``
and must stand without siblings. Every other consumer (the three ``scripts/``
importers and both ``hooks/``) binds through this file, by sibling import or by
``importlib.util.spec_from_file_location``, because neither ``scripts/`` nor
``hooks/`` is a package.

The mismatch text restates the two-root body of
``hhemt._worktree_guard.worktree_mismatch_message`` in stdlib rather than
importing it, for the reason in the first paragraph.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

__all__ = ["assert_resolved_under", "bind_local_src", "prepend_src", "purge_foreign"]


def prepend_src(src_root: Path) -> None:
    """Put ``src_root`` at ``sys.path[0]`` unless it is already there."""
    entry = str(src_root)
    if sys.path and sys.path[0] == entry:
        return
    sys.path.insert(0, entry)


def _top_level(name: str) -> str:
    return name.partition(".")[0]


def _origin_under(module: ModuleType, src_root: Path) -> bool | None:
    """True or False for a module with a ``__file__``; None when it declares none."""
    origin = getattr(module, "__file__", None)
    if not origin:
        return None
    try:
        Path(origin).resolve().relative_to(src_root.resolve())
    except ValueError:
        return False
    return True


def purge_foreign(packages: Iterable[str], src_root: Path) -> list[str]:
    """Drop warm ``sys.modules`` entries of ``packages`` that resolved outside ``src_root``.

    ``packages`` may name submodules; each is reduced to its top-level package.
    Entries with no ``__file__`` are kept. Returns the evicted names, sorted.
    """
    wanted = {_top_level(name) for name in packages}
    evicted = sorted(
        name
        for name, module in list(sys.modules.items())
        if _top_level(name) in wanted and _origin_under(module, src_root) is False
    )
    for name in evicted:
        del sys.modules[name]
    return evicted


def assert_resolved_under(module: ModuleType, src_root: Path) -> None:
    """Raise ``RuntimeError`` when ``module.__file__`` resolves outside ``src_root``.

    A module with no ``__file__`` passes: it claims no on-disk origin, so it
    cannot have come from the wrong tree.
    """
    if _origin_under(module, src_root) is not False:
        return
    resolved = Path(module.__file__).resolve()
    raise RuntimeError(
        f"local-src: {module.__name__}.__file__ = {resolved}\n"
        f"  expected prefix: {src_root.resolve()}\n"
        f"  This process is running OTHER code than the one you are editing; results and\n"
        f"  error messages from this run are not evidence about this checkout.\n"
        f"  Either {src_root} supplies no {module.__name__} package, so the import fell\n"
        f"  through to another checkout, or the import ran before bind_local_src did."
    )


def bind_local_src(src_root: Path, packages: Iterable[str]) -> None:
    """``prepend_src`` then ``purge_foreign``: the rebind every consumer runs before its first import."""
    prepend_src(src_root)
    purge_foreign(packages, src_root)
