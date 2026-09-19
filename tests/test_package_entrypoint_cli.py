"""VMS-A2a: `python -m hhemt` must reach the CLI without the optional GUI stack.

`__main__.py` imported `.gui` at module level while `launch_gui()` was only called under
`len(sys.argv) == 1`. `gui.py` imports `tkinterdnd2`, which is declared in no dependency
file, so every argv-bearing invocation through the package entrypoint died with
ModuleNotFoundError before reaching the Typer app.

The assertions anchor on properties true in BOTH the pre-fix and post-fix worlds — the exit
code and the presence of a command name — never on wording this change introduced. A test
asserting on new text would be green pre-fix for the wrong reason.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"


def _run_module(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run `python -m hhemt` in a subprocess with src/ on the path.

    A subprocess is required rather than an in-process import: the defect is an IMPORT-time
    failure in `__main__`, and `runpy`/import caching inside the test process would not
    reproduce the entrypoint's own module-initialisation order.
    """
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    # The guard would otherwise fire on the cwd/import mismatch inside a tmp cwd.
    env["HHEMT_DISABLE_WORKTREE_GUARD"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hhemt", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=120,
    )


def test_package_entrypoint_help_reaches_the_cli():
    """VIOLATING input pre-fix: an argv-bearing `python -m hhemt` invocation.

    Pre-fix this exited 1 with `ModuleNotFoundError: No module named 'tkinterdnd2'`.
    """
    res = _run_module("--help")
    combined = res.stdout + res.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert res.returncode == 0, combined
    # A command name from the app's own tree — present in both states IF the app is reached.
    assert "run" in res.stdout, res.stdout


def test_the_zero_argument_invocation_reaches_the_cli_help():
    """The zero-argument invocation must reach the CLI's help, not a GUI import.

    Pre-fix this exited 1 with ModuleNotFoundError: No module named 'tkinterdnd2'.
    no_args_is_help=True on the Typer app makes the empty-argv case exit 2.
    Both assertions hold in the pre-fix and post-fix worlds only by their SHAPE,
    never by wording this change introduced.
    """
    res = _run_module()
    combined = res.stdout + res.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert res.returncode == 2, combined
    assert "Usage" in combined, combined


@pytest.mark.parametrize("args", [("--help",), ("run", "--help")])
def test_subcommands_reach_the_app(args):
    """The funnel holds for nested commands too, not only the bare root."""
    res = _run_module(*args)
    combined = res.stdout + res.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert res.returncode == 0, combined
