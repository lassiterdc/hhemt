"""Structural guard: the `justfile` recipes run through `scripts/hook_env.sh`, never a bare `uv run`.

A recipe that runs `uv run --python=...` creates a `.venv/` inside the tree it runs in, about 1.6 GiB per
tree and never removed. The wrapper keeps the environment outside every tree. This file reads the
`justfile` text only: nothing is built or run.
"""

from __future__ import annotations

from pathlib import Path

_JUSTFILE = Path(__file__).resolve().parents[1] / "justfile"
_WRAPPER_LINE = "scripts/hook_env.sh --python"
# The number of `uv run --python=` lines the recipes carried before the wrapper (qa 7, testall 2,
# test 1, pdb 1, coverage 3; docs-check reaches the wrapper through RUN=).
_OLD_UV_RUN_LINES = 14


def _code_lines() -> list[str]:
    return [
        line.strip()
        for line in _JUSTFILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_no_recipe_line_runs_a_bare_uv_run_except_the_conda_hosted_ones():
    """Property: no non-comment line contains `uv run` except the four `conda run ... uv run --active` lines, which
    use the conda environment and create no `.venv`.

    Class: NEW CAPABILITY -- every recipe line shape in the file. The count of conda lines is part of the
    property, so a matcher that stops matching reds rather than passing an empty set.
    Why required: a bare `uv run` creates a per-tree `.venv`.
    What kills it: one recipe reverted to `uv run --python=...`.
    Second correct implementation: a regular expression anchored on a recipe line's first word.
    """
    uv_lines = [line for line in _code_lines() if "uv run" in line]
    conda_lines = [line for line in uv_lines if "conda run" in line and "uv run --active" in line]
    assert len(conda_lines) == 4, conda_lines
    assert [line for line in uv_lines if line not in conda_lines] == []


def test_every_line_that_ran_uv_run_python_now_begins_with_the_wrapper():
    """Property: the recipe lines that began `uv run --python=` before this change begin
    `scripts/hook_env.sh --python`, and there are at least 14 wrapper lines.

    Class: NEW CAPABILITY. The floor of 14 is the old line count, so a matcher that stops matching reds.
    Why required: a recipe line left on `uv run` recreates the per-tree environment, and a line deleted outright
    would lower the count.
    What kills it: dropping or reverting a recipe line.
    Second correct implementation: counting the lines with a `grep -c` of the same prefix; same figure.
    """
    wrapped = [line for line in _code_lines() if line.startswith(_WRAPPER_LINE)]
    assert len(wrapped) >= _OLD_UV_RUN_LINES, len(wrapped)
    assert [line for line in _code_lines() if line.startswith("uv run --python")] == []
