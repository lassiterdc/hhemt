"""No tracked top-level name may collide with a standard-library module.

WHY THIS FILE EXISTS. `scripts/profile/` was a tracked package named for a stdlib
module. Four scripts legitimately insert their own directory on `sys.path`, and a
fifth route reached the same state at RUN time through a hook that `exec_module`s
one of them -- so `import profile` bound the package, `hasattr(profile, "run")` was
False, and `cProfile.py:23` raised `AttributeError` at module scope. Three tests
failed for four months with a cause no `sys.path` grep could find, because the
injecting site contains no `sys.path` text at all.

The rename removed that instance. This removes the CLASS: it is keyed on the NAME,
so it is indifferent to how, when, or at what transitive depth a directory reaches
`sys.path` -- which is the property the search for that injector showed a
path-keyed check cannot have.

WHAT THIS GUARD DOES NOT COVER, stated so nobody reads it as broader than it is.
It ranges over TRACKED paths under the roots named below. It cannot see a name
introduced from outside the tree -- a site-packages entry, a `.pth`, or a
third-party module that itself `exec_module`s a file that inserts a directory --
and it cannot see an untracked file. A module bound inside such a window stays in
`sys.modules` after the path entry is popped, so a `finally` restore does not undo
it. Those classes need a different instrument; this one closes the in-repo class.

Ground truth is `git ls-files -z`, not a filesystem walk and not a line split. A
walk would flag the gitignored mkdocs output directory `site/`, which cannot shadow
(stdlib `site` is bound during interpreter startup, before the repo root is ever on
the path). `-z` rather than `.splitlines()` because `git ls-files` C-QUOTES any path
containing non-ASCII or control characters, which a line split preserves verbatim
and silently turns into a wrong name; `-z` emits raw bytes and needs no unquoting.
Neither form may be `.split()`, which breaks on whitespace: this repo carries
`docs/running a simulation.ipynb`, and `.split()` on it yields a phantom `io.py`.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

#: Directories this tree is known to place on `sys.path`: the repo root (pytest's
#: rootdir insert and `python -m`'s CWD entry), `scripts/` (four module-scope
#: inserts), `tests/`, `src/` (the editable install and the root conftest), and
#: `src/hhemt/suite/` (the suite harness prepends it to PYTHONPATH).
SYS_PATH_ROOTS = ("", "scripts", "tests", "src", "src/hhemt/suite")

_REPO = pathlib.Path(__file__).resolve().parent.parent


def _tracked_paths() -> list[str]:
    """Every tracked path, NUL-separated so a space or a quoted byte cannot split one."""
    out = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [entry for entry in out.split("\0") if entry]


def _importable_names(tracked: list[str] | None = None) -> list[tuple[str, str]]:
    """Every (name, path) a bare `import {name}` could bind from a known root.

    `tracked` is injectable so the predicate can be exercised against a pathological
    ground truth -- which is the arm that would have caught the `.split()` defect
    this function was first written with.
    """
    if tracked is None:
        tracked = _tracked_paths()
    found: set[tuple[str, str]] = set()
    for root in SYS_PATH_ROOTS:
        root_parts = pathlib.PurePosixPath(root).parts if root else ()
        for entry in tracked:
            parts = pathlib.PurePosixPath(entry).parts
            if parts[: len(root_parts)] != root_parts:
                continue
            rest = parts[len(root_parts) :]
            if not rest:
                continue
            if len(rest) == 1:
                if rest[0].endswith(".py"):
                    found.add((rest[0][:-3], entry))
            else:
                found.add((rest[0], "/".join([*root_parts, rest[0]])))
    return sorted(found)


def test_no_tracked_top_level_name_shadows_a_stdlib_module():
    stdlib = set(sys.stdlib_module_names)
    collisions = [(name, path) for name, path in _importable_names() if name in stdlib]
    assert not collisions, (
        "tracked top-level name(s) collide with a stdlib module and will shadow it "
        "whenever the containing directory reaches sys.path: "
        + ", ".join(f"{name!r} at {path}" for name, path in collisions)
    )


def test_the_guard_examines_a_non_empty_population():
    """A disclosed denominator: `assert not collisions` passes identically on an
    empty population, so the guard must prove it looked at something."""
    names = _importable_names()
    assert len(names) > 50, f"guard examined only {len(names)} names -- ground truth is empty or wrong"


def test_the_ground_truth_never_splits_a_path():
    """The arm that would have caught this file's own first defect.

    `_tracked_paths` was first written as `...stdout.split()`, which splits on ALL
    whitespace while `git ls-files` emits one path per record. This repo carries
    `docs/running a simulation.ipynb`, so that form yielded 3836 tokens where there
    are 3834 paths and manufactured a phantom top-level `io.py`.

    Asserting that every returned entry EXISTS is path-agnostic and self-maintaining:
    a split fragment cannot exist on disk, so the defect fails here whatever the
    offending filename happens to be on the day. A clean-corpus arm cannot do this --
    it returns "no finding" identically for a correct predicate and a broken one,
    which is exactly why the defect survived the first two-arm differential.
    """
    missing = [entry for entry in _tracked_paths() if not (_REPO / entry).exists()]
    assert not missing, f"ground truth yielded {len(missing)} path(s) that do not exist, e.g. {missing[:3]}"


def test_a_space_bearing_path_does_not_manufacture_a_phantom_name():
    """The over-firing arm, against a PATHOLOGICAL ground truth rather than a clean one.

    A clean-corpus arm returns 'no finding' identically for a correct predicate and
    for one that splits paths on whitespace, so it cannot observe that defect. This
    repo really does carry `docs/running a simulation.ipynb`; the first draft of this
    guard used `.split()` and would have reported a collision at a path that does not
    exist.
    """
    names = _importable_names(tracked=["docs/my io.py", "src/hhemt/__init__.py"])
    assert ("io", "io.py") not in names, "a space in a path manufactured a phantom top-level name"
    assert names == [
        ("docs", "docs"),
        ("hhemt", "src/hhemt"),
        ("src", "src"),
    ], f"unexpected names from the pathological corpus: {names}"
