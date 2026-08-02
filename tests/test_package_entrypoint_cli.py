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


def test_package_entrypoint_does_not_import_the_gui_stack_for_a_cli_call():
    """The GUI module must not be imported at all on an argv-bearing invocation.

    Asserting only on exit code would pass if `tkinterdnd2` happened to be installed, which
    would make this test environment-dependent rather than behavioural.
    """
    res = _run_module("--help", env_extra={"PYTHONWARNINGS": "ignore"})
    assert res.returncode == 0, res.stdout + res.stderr
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hhemt.__main__; "
            "print('GUI_IMPORTED' if 'hhemt.gui' in sys.modules else 'GUI_NOT_IMPORTED')",
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(_SRC)},
        cwd=str(_REPO_ROOT),
        timeout=120,
    )
    assert "GUI_NOT_IMPORTED" in probe.stdout, probe.stdout + probe.stderr


def test_the_gui_entrypoint_is_still_wired():
    """Differently-positioned satisfying input: the GUI path must be UNCHANGED.

    The fix moves an import; it must not remove the capability. This asserts the call site
    still exists and still resolves, without launching a window.
    """
    src = (_SRC / "hhemt" / "__main__.py").read_text(encoding="utf-8")
    assert "launch_gui()" in src, "the GUI call site was removed, not just its import"
    assert "from .gui import launch_gui" in src, "the GUI import was dropped entirely"
    # And it is INSIDE the argv guard, not at module level.
    head = src.split("if __name__", 1)[0]
    assert "from .gui import launch_gui" not in head, "the gui import is still module-level"


@pytest.mark.parametrize("args", [("--help",), ("run", "--help")])
def test_subcommands_reach_the_app(args):
    """The funnel holds for nested commands too, not only the bare root."""
    res = _run_module(*args)
    combined = res.stdout + res.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert res.returncode == 0, combined
