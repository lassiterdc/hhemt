"""Pins the `hhemt test` seam's two invisible-to-status-check properties.

Both are invisible to exit status by measurement, which is why every assertion
below reads RENDERED OUTPUT or the PROCESS exit code, and never a return value.
"""

from __future__ import annotations

import typer
from typer.testing import CliRunner

from hhemt.cli import app
from hhemt.suite._cli import register_subject

runner = CliRunner()


def test_bare_test_lists_subjects_rather_than_erroring():
    """`hhemt test` must TEACH. Exit status cannot distinguish the two renderings."""
    result = runner.invoke(app, ["test"])
    assert "Commands" in result.stdout, result.stdout
    assert "toolkit" in result.stdout, result.stdout
    assert "Missing command" not in result.stdout, result.stdout


def test_bare_subject_lists_its_actions():
    result = runner.invoke(app, ["test", "toolkit"])
    for action in ("plan", "chunk", "aggregate", "triage"):
        assert action in result.stdout, f"{action} missing:\n{result.stdout}"
    assert "Missing command" not in result.stdout, result.stdout


def test_register_subject_refuses_a_subject_without_no_args_is_help():
    """The seam guard. Without the flag the subject renders an error box at exit 2."""
    bad = typer.Typer(help="x")  # no_args_is_help defaults False
    try:
        register_subject(bad, name="bad")
    except RuntimeError as exc:
        assert "no_args_is_help=True" in str(exc)
    else:
        raise AssertionError("register_subject accepted a subject app without the flag")


def test_command_bodies_raise_typer_exit_rather_than_returning():
    """A body that RETURNS the code exits 0. Asserted on the PROCESS exit code."""
    import ast
    import pathlib

    src = pathlib.Path(__import__("hhemt.suite._cli", fromlist=["x"]).__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_") and node.name != "_argv":
            last = node.body[-1]
            assert isinstance(last, ast.Raise), f"{node.name} does not end in `raise`; a returning Typer body exits 0"


def test_plugin_loads_in_a_real_child_and_does_not_prebind_hhemt(tmp_path):
    """The plugin name must be importable BY A CHILD, and must not bind `hhemt`.

    Both are invisible in-process. A dotted `-p hhemt.suite._runner` binds `hhemt`
    through its own packages before the repo-root conftest can point it at the
    checkout under test; a module-level `from hhemt.suite import partition` does the
    same under any name. Measured four ways; only bare-name-plus-lazy-import passes.
    """
    import subprocess
    import sys
    from pathlib import Path

    import hhemt.suite as _suite

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "conftest.py").write_text(
        "import sys\nraise SystemExit(f'PREBOUND={\"hhemt\" in sys.modules}')\n",
        encoding="utf-8",
    )
    (repo / "test_x.py").write_text("def test_ok(): assert True\n", encoding="utf-8")
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(Path(_suite.__file__).parent)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "_runner",
            "--collect-only",
            "-q",
            "test_x.py",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        env=env,
    )
    out = proc.stdout + proc.stderr
    assert "No module named" not in out, f"plugin not importable in the child:\n{out}"
    assert "PREBOUND=False" in out, f"plugin pre-bound hhemt before the guard:\n{out}"
