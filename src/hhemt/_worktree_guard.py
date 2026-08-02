"""Worktree-vs-main source resolution guard for NON-pytest entrypoints.

The hazard (protocol `worktree aware project testing.md`, Problem section): the editable
install writes exactly ONE path into site-packages, so whichever worktree most recently ran
`pip install -e .` determines which source tree every other worktree's `import hhemt`
resolves to. The shipped 4-layer guard lives in the repo-root `conftest.py`, which pytest
loads and `python script.py` / the `hhemt` console script / a Jupyter kernel never do.

Measured consequence: `conda run -n hhemt python {script}` resolved `hhemt` to the MAIN
checkout while the operator was working in a worktree, and aborted a bundle combine with a
confident, domain-plausible, WRONG "Bundles are not combine-compatible" diagnostic naming
eight real toggle divergences. The message read as a correct domain observation because the
two arms genuinely DO differ on those toggles — that IS the two-arm design — so nothing
about the failure said "stale code".

Predicate. Outside pytest there is no declared repo root, so the guard cannot ask "is this
the right tree" in the abstract. What it CAN ask is whether the caller is standing in a
source checkout that the import did not honour: if the current working directory lies inside
a git worktree whose root contains `src/hhemt/__init__.py`, then THAT tree is the one the
operator meant, and `hhemt.__file__` resolving elsewhere is the failure. When the cwd is not
inside such a checkout (a deployed run, a user's project directory, an HPC scratch dir) the
guard has no expectation to check and returns silently. This is why it is not a per-script
`assert`: there is exactly one predicate, in one place, and it is a no-op for every
legitimate non-developer invocation.

Two tiers consume this module and they share ONE message body:

* `conftest.py` (pytest) calls `worktree_mismatch_message` directly, because it owns
  pytest-specific behaviour the shared module must not: the `HHEMT_FORCE_WRONG_SRC` pytester
  seam, the `sys.__stderr__`/`sys.__stdout__` writes that bypass pytest's already-installed
  capture middleware, and the success-path marker the protocol's Smoke Validation
  Requirement greps for.
* Every non-pytest entrypoint reaches `assert_worktree_source` through the Typer root
  callback.

The two tiers differ only in their LABEL and their exit behaviour, which is why the label is
a parameter rather than a literal: the pytest tier's message is contract-pinned to
``worktree-test-guard`` by `tests/test_worktree_guard.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

#: Escape hatch, spelled to match the conftest guard's vocabulary.
_ALLOW_ENV = "HHEMT_ALLOW_INSTALLED"
_DISABLE_ENV = "HHEMT_DISABLE_WORKTREE_GUARD"


def _checkout_root_of_cwd() -> Path | None:
    """The git toplevel of the cwd, if it contains ``src/hhemt/__init__.py``; else None."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    root = Path(res.stdout.strip())
    return root if (root / "src" / "hhemt" / "__init__.py").exists() else None


def worktree_mismatch_message(
    *,
    expected_src: Path,
    force_wrong_src: str | None = None,
    label: str = "worktree-guard",
) -> str | None:
    """Return None when ``hhemt`` resolves under ``expected_src``; else the message.

    This is the single implementation of the mismatch text. ``label`` exists because the
    pytest tier's wording is pinned by `tests/test_worktree_guard.py`
    (``worktree-test-guard: hhemt.__file__ = ...``) while the CLI tier uses its own name;
    the BODY is identical, which is the property the two-tier design is after.

    ``force_wrong_src`` is the pytester seam: when set, it is substituted for the real
    ``hhemt.__file__`` so a test can drive the mismatch branch without a second checkout.
    Import failure is ALSO a mismatch — the shared conda env's editable install can point at
    a removed worktree — and returns its own message rather than raising, so both callers
    handle one return type.
    """
    expected_src = Path(expected_src).resolve()
    if force_wrong_src is not None:
        resolved = Path(force_wrong_src).resolve()
    else:
        try:
            import hhemt
        except ImportError as exc:
            return (
                f"{label}: import hhemt failed ({exc}).\n"
                f"  expected prefix: {expected_src}\n"
                f"  shared conda env's editable install may point at a removed or stale path.\n"
                f"  Set {_ALLOW_ENV}=1 to bypass for installed-package testing."
            )
        resolved = Path(hhemt.__file__).resolve()

    try:
        resolved.relative_to(expected_src)
    except ValueError:
        return (
            f"{label}: hhemt.__file__ = {resolved}\n"
            f"  expected prefix: {expected_src}\n"
            f"  shared conda env's editable install points elsewhere.\n"
            f"  This process is running OTHER code than the one you are editing; results and\n"
            f"  error messages from this run are not evidence about this checkout.\n"
            f"  Fix: run with PYTHONPATH={expected_src}, or re-run `pip install -e .` from\n"
            f"  {expected_src.parent}. Set {_ALLOW_ENV}=1 to proceed deliberately."
        )
    return None


def assert_worktree_source(*, strict: bool = True) -> Path | None:
    """Assert the imported ``hhemt`` came from the checkout the cwd is standing in.

    Returns the expected src dir when a check was performed and passed, None when no
    expectation applies. ``strict=True`` exits with the conftest guard's code 99 on a
    mismatch; ``strict=False`` warns. Honours ``HHEMT_DISABLE_WORKTREE_GUARD=1`` (skip
    entirely) and ``HHEMT_ALLOW_INSTALLED=1`` (downgrade to a warning).
    """
    if os.environ.get(_DISABLE_ENV) == "1":
        return None
    root = _checkout_root_of_cwd()
    if root is None:
        return None
    expected_src = (root / "src").resolve()

    msg = worktree_mismatch_message(expected_src=expected_src)
    if msg is None:
        return expected_src

    if os.environ.get(_ALLOW_ENV) == "1" or not strict:
        warnings.warn(msg, stacklevel=2)
        return expected_src
    sys.__stderr__.write(f"[worktree-guard] {msg}\n")
    sys.__stderr__.flush()
    sys.exit(99)
