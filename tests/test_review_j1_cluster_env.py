"""Reviewer tests for work package 1i, chunk 1i.1: the Snakemake thread guard must take effect when the script runs.

Problem: the coder's scan test accepts a script when the guard placeholder stands within three lines before the
``-m snakemake`` launch, which reads the source text and not what bash does with it. A placeholder that sits in
the window but outside the launch's backslash-continued logical command would pass that scan and set nothing
in the Snakemake child. Both emitted login-node scripts carry the launch as a ``env ... \\`` continuation chain,
so what must hold is a property of the rendered script, and these tests render the real f-string and run the
launch's logical command under bash.

No solver, compile or Snakemake process runs: the Python interpreter the script would call is replaced by a
shim that prints the three variables.
"""

from __future__ import annotations

import ast
import builtins
import os
import subprocess
from pathlib import Path

import pytest

from hhemt import workflow as wf

_GUARD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
_PLACEHOLDER = "_SNAKEMAKE_THREAD_GUARD_SHELL_ASSIGNMENTS"
_WORKFLOW_SOURCE = Path(wf.__file__).read_text()


class _Any:
    """Stands in for any name or attribute chain the f-string reads, so the real template renders."""

    def __getattr__(self, name):
        return _Any()

    def __call__(self, *args, **kwargs):
        return _Any()

    def __str__(self):
        return "X"

    def __format__(self, spec):
        return "X"

    def __bool__(self):
        return True

    def __fspath__(self):
        return "X"


class _Names(dict):
    """Locals for ``eval``: module values first, builtins second, a stand-in otherwise."""

    def __missing__(self, key):
        if hasattr(wf, key):
            return getattr(wf, key)
        if hasattr(builtins, key):
            return getattr(builtins, key)
        return _Any()


def _rendered_snakemake_scripts(source: str) -> list[str]:
    """Render every f-string script in ``source`` that carries the guard placeholder and a snakemake launch."""
    tree = ast.parse(source)
    scripts: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        if _PLACEHOLDER not in ast.unparse(node):
            continue
        expression = ast.Expression(body=node)
        ast.fix_missing_locations(expression)
        scripts.append(str(eval(compile(expression, "<script>", "eval"), {"__builtins__": builtins}, _Names())))  # noqa: S307
    return scripts


def _launch_logical_command(script: str) -> str:
    """Return the backslash-continued logical command that holds the ``-m snakemake`` launch.

    Bash joins a line ending in a backslash with the next line, so the statement bash runs starts at the first
    line of the chain, and a guard outside the chain is not part of the launch.
    """
    lines = script.split("\n")
    idx = next(i for i, ln in enumerate(lines) if "-m snakemake" in ln and ln.rstrip().endswith("\\"))
    start = idx
    while start > 0 and lines[start - 1].rstrip().endswith("\\"):
        start -= 1
    end = idx
    while end < len(lines) - 1 and lines[end].rstrip().endswith("\\"):
        end += 1
    return "\n".join(lines[start : end + 1])


def _run_launch(command: str, tmp_path: Path, extra_env: dict[str, str]) -> list[str]:
    """Run ``command`` with a python shim in place of the interpreter and return the three variables it sees."""
    prefix = tmp_path / "conda"
    (prefix / "bin").mkdir(parents=True)
    shim = prefix / "bin" / "python"
    shim.write_text(
        '#!/bin/sh\necho "${OPENBLAS_NUM_THREADS-unset} ${OMP_NUM_THREADS-unset} ${MKL_NUM_THREADS-unset}"\n'
    )
    shim.chmod(0o755)
    base = {k: v for k, v in os.environ.items() if k not in _GUARD_VARS}
    env = {**base, "CONDA_PREFIX": str(prefix), "SLURM_BIN": "/usr/bin", **extra_env}
    done = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True, check=True)
    return done.stdout.split()


def test_there_are_two_rendered_scripts_with_a_guard_placeholder():
    """Property: the rendering helper reaches exactly the two scripts the shape names, so later tests are not vacuous.

    Class: the f-strings of ``workflow.py`` carrying the guard placeholder. REGRESSION: red before the change
    (zero placeholders). Why required: a helper that renders nothing would let every behavioural test below pass
    over an empty list. What kills it: removing a placeholder or adding a third guarded script without the shape
    saying so. Second correct implementation: the guard spelled through a helper function instead of the
    placeholder, with this count reading the helper's call sites.
    """
    scripts = _rendered_snakemake_scripts(_WORKFLOW_SOURCE)
    assert len(scripts) == 2
    assert all("-m snakemake" in s for s in scripts)


@pytest.mark.skipif(not Path("/bin/bash").exists() and not Path("/usr/bin/bash").exists(), reason="needs bash")
def test_every_guarded_script_launch_runs_with_the_defaults_and_keeps_exported_values(tmp_path: Path):
    """Property: the logical command that launches Snakemake gives the child 1 for unset variables and keeps exports.

    Class: both emitted login-node scripts, each under three environments of the same variables (all unset,
    one exported, one empty beside one exported). The input is the rendered real f-string, not its source text.
    REGRESSION: red before the change, which printed ``unset`` for all three. Why required: the guard is only a
    guard if it is in the launch's own backslash chain; a placeholder on a line bash does not join to the launch
    sets nothing, and the coder's source scan cannot see that. What kills it: the placeholder moved outside the
    continuation chain, or single-quoted so bash never expands it. Second correct implementation:
    ``export VAR=${VAR:-1}`` lines ahead of the launch, which this test accepts because it asserts on the
    child's environment and not on the placeholder.
    """
    scripts = _rendered_snakemake_scripts(_WORKFLOW_SOURCE)
    assert scripts
    for i, script in enumerate(scripts):
        command = _launch_logical_command(script)
        where = tmp_path / str(i)
        where.mkdir()
        assert _run_launch(command, where / "a", {}) == ["1", "1", "1"]
        assert _run_launch(command, where / "b", {"OPENBLAS_NUM_THREADS": "8"}) == ["8", "1", "1"]
        assert _run_launch(command, where / "c", {"OMP_NUM_THREADS": "", "MKL_NUM_THREADS": "3"}) == ["1", "1", "3"]


@pytest.mark.skipif(not Path("/bin/bash").exists() and not Path("/usr/bin/bash").exists(), reason="needs bash")
def test_a_guard_line_ahead_of_the_env_chain_is_caught_by_the_behavioural_check(tmp_path: Path):
    """Property: the behavioural check reds on a script whose guard line is in the window but outside the launch chain.

    Class: one planted defect positioned differently from the coder's mutations (which delete the placeholder
    line): the guard line moves ahead of the ``env PATH=...`` line and loses its backslash, so it stays within
    three lines of the launch while bash reads it as a plain shell assignment that is never exported. NEW
    CAPABILITY: the plausible wrong implementation is a guard that is present and ineffective; this test is red
    under it and the source-text scan in ``test_snakemake_launch_env_guard`` is green under it. Why required: it
    is the control that shows the behavioural check above can fail. What kills it: this test going green on the
    planted source. Second correct implementation: a planted defect that comments the guard line out.
    """
    guard_line = f"    {{{_PLACEHOLDER}}} \\\\\n"
    assert _WORKFLOW_SOURCE.count(guard_line) == 2
    first = _WORKFLOW_SOURCE.index(guard_line)
    head, tail = _WORKFLOW_SOURCE[:first], _WORKFLOW_SOURCE[first + len(guard_line) :]
    env_line_start = head.rindex("env PATH=")
    planted = head[:env_line_start] + f"{{{_PLACEHOLDER}}}\n" + head[env_line_start:] + tail
    scripts = _rendered_snakemake_scripts(planted)
    command = _launch_logical_command(scripts[0])
    assert "OPENBLAS_NUM_THREADS" not in command
    assert _run_launch(command, tmp_path, {}) == ["unset", "unset", "unset"]
