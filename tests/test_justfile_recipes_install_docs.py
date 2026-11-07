"""Every justfile recipe that runs pytest through `uv run` installs the `docs` extra, and the `--active` ones drop
`VIRTUAL_ENV` (work package 5a, chunk 5a.4).

`tests/test_check_docs_content.py` imports `mkdocstrings_handlers`, which only the `docs` extra provides; CI installs
`.[test,docs]` but no recipe did, so a run on an interpreter without the handler failed four tests. The shell
exports `VIRTUAL_ENV` and `uv run --active` follows it, so the three `conda run ... uv run --active` recipes are run
with `env -u VIRTUAL_ENV` BEFORE `conda run` (`conda run` re-exports its own prefix to what it launches, so removing
the variable inside the command would not work). The checks read the justfile text only: nothing is built or run.
"""

from __future__ import annotations

import re
from pathlib import Path

_JUSTFILE = Path(__file__).resolve().parents[1] / "justfile"


def _is_pytest_line(line: str) -> bool:
    """A non-comment line that runs pytest through `uv run` (its command is `pytest` or `-m pytest`)."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "uv run" not in stripped:
        return False
    return bool(re.search(r"(?:\bpytest\b|-m pytest\b)", stripped.split("uv run", 1)[1]))


def _violations(text: str) -> list[str]:
    """Lines breaking either rule: a pytest `uv run` line without `--extra docs`, or an `--active` line that does not
    run `env -u VIRTUAL_ENV` before `conda run`."""
    found = []
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        if _is_pytest_line(line) and "--extra docs" not in line:
            found.append(f"no --extra docs: {line.strip()[:70]}")
        if "uv run --active" in line and "env -u VIRTUAL_ENV conda run" not in line:
            found.append(f"VIRTUAL_ENV kept: {line.strip()[:70]}")
    return found


def test_every_pytest_recipe_installs_docs_and_every_active_recipe_drops_virtual_env():
    """Property: the real justfile has no violation.

    Class: REGRESSION, red on the justfile at `47e1074a` (the control below). Why required: a recipe that runs the
    docs-content tests without the extra fails them on an interpreter that lacks the handler. What kills it: removing
    `--extra docs` from any pytest line, or `env -u VIRTUAL_ENV` from an `--active` line. A second correct
    implementation: one that installs the extra through a shared variable, if the predicate followed it."""
    text = _JUSTFILE.read_text(encoding="utf-8")
    assert sum(_is_pytest_line(x) for x in text.splitlines()) >= 9, "control: the predicate must find the pytest lines"
    assert _violations(text) == []


_OLD_SAMPLE = """\
    uv run --python=3.12 --extra test pytest -m "not slow"
    uv run --python=3.11 --extra test pytest
    uv run --python=3.12 --extra test pytest
    uv run --python=3.12 --extra test pytest {{ARGS}}
    X=1 conda run --no-capture-output -n hhemt uv run --active --extra test pytest -p _runner
    X=1 conda run --no-capture-output -n hhemt uv run --active --extra test pytest -p _runner -n 1
    X=1 conda run --no-capture-output -n hhemt uv run --active --extra test pytest -n 12
    uv run --python=3.12  --extra test pytest --pdb --maxfail=10
    uv run --python=3.12 --extra test coverage run -m pytest .
#   conda run -n hhemt uv run --active --extra test pytest {{ARGS}}
    uv run --python=3.12 --extra test ruff format .
"""


def test_the_predicate_reports_the_old_lines_and_skips_comments_and_other_commands():
    """Must-fire control: nine pytest lines and three `--active` lines as they stood are all reported, and the comment
    and the non-pytest line are not.

    Class: NEW CAPABILITY. What kills it: a predicate that ignores `-m pytest`, comments or `--active`."""
    found = _violations(_OLD_SAMPLE)
    assert sum(f.startswith("no --extra docs") for f in found) == 9
    assert sum(f.startswith("VIRTUAL_ENV kept") for f in found) == 3
    assert len(found) == 12


def _marker_expression(text: str, recipe: str) -> str:
    """The `-m "..."` marker expression on a recipe's pytest command line."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"{recipe} "))
    for ln in lines[start + 1 :]:
        if not ln.startswith((" ", "\t")):
            break
        found = re.search(r'\s-m "([^"]*)"', ln)
        if found:
            return found.group(1)
    raise AssertionError(f"recipe {recipe} has no -m marker expression")


def _selects(expression: str, markers: set[str]) -> bool:
    """Whether pytest's own marker-expression evaluator selects a test carrying exactly `markers`."""
    from _pytest.mark.expression import Expression

    return Expression.compile(expression).evaluate(lambda name, **_kw: name in markers)


def test_the_no_compile_recipe_excludes_the_snakemake_subprocess_tests_and_their_own_recipe_selects_them():
    """Property: `test-no-compile-no-subprocess`'s `-m` expression does not select a test marked
    `requires_snakemake_subprocess`, and `test-snakemake-subprocess` selects exactly those that are not `compile_tier`.

    Class: the tests that start Snakemake as a child process stay out of the parallel tier. Why required: run serially
    they cost several times the rest of the pool. What kills it: dropping `not requires_snakemake_subprocess` from
    the no-compile recipe, or misspelling the
    marker (the misspelt expression selects a marked test again). Control: an unmarked non-compile test is still
    selected by the no-compile recipe, and a `compile_tier` test is selected by neither."""
    text = _JUSTFILE.read_text(encoding="utf-8")
    no_compile = _marker_expression(text, "test-no-compile-no-subprocess")
    assert not _selects(no_compile, {"requires_snakemake_subprocess"}), no_compile
    assert not _selects(no_compile, {"requires_snakemake_subprocess", "slow"}), no_compile
    assert _selects(no_compile, set()) and _selects(no_compile, {"slow"}), "control: ordinary tests stay selected"
    assert not _selects(no_compile, {"compile_tier"})
    own = _marker_expression(text, "test-snakemake-subprocess")
    assert _selects(own, {"requires_snakemake_subprocess"})
    assert not _selects(own, {"requires_snakemake_subprocess", "compile_tier"}) and not _selects(own, set())
