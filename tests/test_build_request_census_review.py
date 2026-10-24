"""Check that the build-request census does not excuse a real run because another call was a dry one.

The problem: ``tests/test_build_request_census.py`` excuses a test only when every workflow
call it reaches is a no-op, a literal ``--dry-run`` or ``--help``, or ``dry_run=True``. A
command-line argv usually reaches its call through a name, a helper or a fixture, and each of
those can have more than one source: a name bound twice, a helper or fixture with two
``return`` paths. If the census decides that a call is a no-op because ANY source carries the
flag, rather than EVERY source, then a test that runs a dry call and then a real one through one
name, or takes an argv helper's or fixture's unflagged path, is excused for a call that starts a
real workflow. The census would then stay green on exactly the kind of test it exists to catch.

How this module checks it: it copies the repository into a temporary directory, adds one test
file holding planted nodes, runs the census there in a child pytest, and reads which planted
node ids the census lists as failing. Only collection happens. No planted node is executed,
and nothing compiles.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PLANT_REL = "tests/test_zz_review_flow_plant.py"

# The planted nodes. Each `invoke` below names `hhemt run` with real config paths, so a call
# without `--dry-run`/`--help` would start a workflow if it executed. None requests the build.
_PLANT_SOURCE = """\
import pytest
from typer.testing import CliRunner

from hhemt.cli import app

runner = CliRunner()


@pytest.fixture
def real_or_dry_argv(tmp_path, request):
    s, a = str(tmp_path / "s.yaml"), str(tmp_path / "a.yaml")
    if getattr(request, "param", False):
        return ["run", "--system-config", s, "--analysis-config", a, "--dry-run"]
    return ["run", "--system-config", s, "--analysis-config", a]


@pytest.fixture
def dry_argv(tmp_path):
    s, a = str(tmp_path / "s.yaml"), str(tmp_path / "a.yaml")
    return ["run", "--system-config", s, "--analysis-config", a, "--dry-run"]


def test_fixture_taken_on_its_real_path(real_or_dry_argv):
    runner.invoke(app, real_or_dry_argv)


def test_control_fixture_with_one_dry_return(dry_argv):
    runner.invoke(app, dry_argv)


def _argv(s, a, dry):
    if dry:
        return ["run", "--system-config", s, "--analysis-config", a, "--dry-run"]
    return ["run", "--system-config", s, "--analysis-config", a]


def test_dry_then_real_one_name(tmp_path):
    s, a = str(tmp_path / "s.yaml"), str(tmp_path / "a.yaml")
    argv = ["run", "--system-config", s, "--analysis-config", a, "--dry-run"]
    runner.invoke(app, argv)
    argv = ["run", "--system-config", s, "--analysis-config", a]
    runner.invoke(app, argv)


def test_help_then_real_one_name(tmp_path):
    s, a = str(tmp_path / "s.yaml"), str(tmp_path / "a.yaml")
    argv = ["run", "--help"]
    runner.invoke(app, argv)
    argv = ["run", "--system-config", s, "--analysis-config", a]
    runner.invoke(app, argv)


def test_helper_called_on_its_real_path(tmp_path):
    runner.invoke(app, _argv(str(tmp_path / "s.yaml"), str(tmp_path / "a.yaml"), False))


def test_control_dry_then_real_two_names(tmp_path):
    s, a = str(tmp_path / "s.yaml"), str(tmp_path / "a.yaml")
    dry_argv = ["run", "--system-config", s, "--analysis-config", a, "--dry-run"]
    runner.invoke(app, dry_argv)
    real_argv = ["run", "--system-config", s, "--analysis-config", a]
    runner.invoke(app, real_argv)


def test_control_literal_dry_at_the_call(tmp_path):
    s, a = str(tmp_path / "s.yaml"), str(tmp_path / "a.yaml")
    runner.invoke(app, ["run", "--system-config", s, "--analysis-config", a, "--dry-run"])
"""

# Planted nodes whose second call is a real run: the census must list each one.
_MUST_FAIL = (
    "test_dry_then_real_one_name",
    "test_help_then_real_one_name",
    "test_helper_called_on_its_real_path",
    "test_fixture_taken_on_its_real_path",
)
# Positive control: the same real call through a separate name, which the census already
# lists. If it is missing, the plant was not collected or not walked, and the four above
# being absent would mean nothing.
_CONTROL_FAIL = "test_control_dry_then_real_two_names"
# Satisfying arms: calls that really are dry runs, once with the flag literal in the call and
# once through a fixture whose only return carries it. The census must list NEITHER.
_MUST_PASS = ("test_control_literal_dry_at_the_call", "test_control_fixture_with_one_dry_return")


def _tree_files(root: Path) -> list[str]:
    """Tracked files of ``root``, plus untracked, non-ignored files under ``tests/``.

    The census module may itself still be untracked, so ``tests/`` takes untracked files
    too. Elsewhere only tracked files are copied, so stray large files do not enter the copy.
    """
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout.split("\0")
    extra = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--others", "--exclude-standard", "--", "tests"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    return sorted({p for p in [*tracked, *extra] if p})


def _copy_tree(dest: Path) -> None:
    """Copy the working-tree content of every file `_tree_files` names into ``dest``."""
    for rel in _tree_files(_REPO_ROOT):
        src = _REPO_ROOT / rel
        if not src.is_file():
            continue  # a tracked path deleted in the working tree
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def _listed(output: str, name: str) -> bool:
    """True when the census output names the planted node ``name`` (not merely a longer id)."""
    return re.search(re.escape(f"{_PLANT_REL}::{name}") + r"(?![\w\[])", output) is not None


def test_census_does_not_excuse_a_real_call_because_a_sibling_binding_is_dry(tmp_path):
    """Property: a CLI call is a no-op only if the argv that reaches THAT call carries the flag.

    Class: every test whose real `hhemt` invocation reads its argv from a name that is also
    bound, elsewhere in the same function, to an argv carrying `--dry-run` or `--help`, and
    every test that takes an argv helper's or argv fixture's return path carrying neither flag
    while another return path carries one. Why it is required: on that class the census reports the
    real call as a dry or help call, auto-excuses the node, and stays green on a test that
    starts a workflow without requesting the build. Those are exactly the tests it exists to
    fail, and nothing else in the suite reds on them. What kills it: the census deciding
    `dry`/`help` per call from the argv that reaches that call, and for a helper or fixture
    only from flags carried on every return path, instead of from the union of every binding
    of the name and every `return`. A second correct implementation under which it still
    passes: a conservative rule that a name bound more than once, or a helper or fixture with
    more than one return, never makes a call a no-op. Under that rule the four planted nodes and
    the two-name control are listed, and the two genuine dry-run controls are not.
    """
    copy = tmp_path / "tree"
    _copy_tree(copy)
    (copy / _PLANT_REL).write_text(_PLANT_SOURCE, encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST")}
    env["HHEMT_FORBID_COMPILE"] = "1"
    env["PYTHONPATH"] = str(copy / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_build_request_census.py", "-p", "no:cacheprovider", "-q"],
        cwd=copy,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    output = proc.stdout + proc.stderr
    tail = output[-4000:]

    # Matched on the printed counts, because the bare label also appears in a traceback's source.
    assert re.search(r"build-request census: collected=\d+", output), (
        f"the census did not run in the copied tree:\n{tail}"
    )
    assert _listed(output, _CONTROL_FAIL), (
        f"positive control missing: the census did not list {_CONTROL_FAIL}, so the plant was not "
        f"collected or not walked and this check says nothing:\n{tail}"
    )
    over_fired = [name for name in _MUST_PASS if _listed(output, name)]
    assert not over_fired, (
        f"the census listed genuine dry runs, whose every argv source carries --dry-run: {over_fired}\n{tail}"
    )
    excused = [name for name in _MUST_FAIL if not _listed(output, name)]
    assert not excused, (
        "the census excused a real `hhemt run` call because another binding of its argv name, or another "
        f"return path of its argv helper or fixture, carries --dry-run/--help: {excused}\n{tail}"
    )
