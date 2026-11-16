"""Review of the release-step venue table: the claims the table makes about files it does not own.

The builder's tests grade the table's own cells and one cross-check, that each `python scripts/` or
`bash scripts/` command appears somewhere in the justfile or a workflow. Three claims of the table
point at a neighbour file and no test reads that neighbour:

* a stand-in cell names a CI job (`docs-build`, `compile-synth-render-gate`), and a job under another key or
  a removed job leaves the cell naming a result that no run produces;
* the `just tag` stand-in quotes the first line of the recipe `tag:` and says how many rows
  precede it, and a recipe edit leaves both stale;
* the cache-clear row names the `--apply` flag and the exit code 2, which the cli declares.

Each check is a function over text that returns a list of problems, and each test feeds a mutated
input to the function and expects the named problem, so a function that always returns an empty
list fails the controls. The tests read text only and run no command of the release walk.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _ROOT / "docs" / "explanation" / "branching-and-release-model.md"
_JUSTFILE = _ROOT / "justfile"
_CLI = _ROOT / "src" / "hhemt" / "cli.py"
_WORKFLOWS = _ROOT / ".github" / "workflows"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _table(page_text: str) -> list[list[str]]:
    """The body rows of the table under the heading 'Release step venues', as lists of cells."""
    match = re.search(r"^### Release step venues\n(.*?)(?=^#{1,3} )", page_text, re.S | re.M)
    if match is None:
        return []
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in match.group(1).splitlines()
        if line.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", line)
    ]
    return rows[1:]


def _row(rows: list[list[str]], command: str) -> list[str]:
    return next((r for r in rows if command in r[0]), [])


def _tag_recipe(justfile_text: str) -> list[str]:
    lines = justfile_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("tag:")), None)
    if start is None:
        return []
    body = []
    for ln in lines[start + 1 :]:
        if ln and not ln.startswith((" ", "\t")):
            break
        body.append(ln.strip())
    return [ln for ln in body if ln]


def stand_in_job_problems(page_text: str, workflows: dict[str, str]) -> list[str]:
    """Problems: a CI job named in a stand-in cell is no job of a workflow, or the docs job runs a different line."""
    rows = _table(page_text)
    names = []
    for command in ("bash scripts/docs_gates.sh", "HHEMT_COMPILE_VENUE=toolchain"):
        names += re.findall(r"`([a-z]+(?:-[a-z]+)+)`", _row(rows, command)[3] if _row(rows, command) else "")
    problems = []
    if len(names) != 2:
        problems.append(f"expected 2 CI job names in the stand-in cells, found {names}")
    for name in names:
        if not any(re.search(rf"^  {re.escape(name)}:\s*$", text, re.M) for text in workflows.values()):
            problems.append(f"the stand-in names the job {name}, and no workflow declares that job")
    docs_build = workflows.get("docs-build.yml", "")
    if not re.search(r"^\s*run: bash scripts/docs_gates\.sh\s*$", docs_build, re.M):
        problems.append("the docs-build workflow does not run `bash scripts/docs_gates.sh` bare")
    return problems


def just_tag_problems(page_text: str, justfile_text: str) -> list[str]:
    """Problems: the `just tag` stand-in quotes a guard line the recipe lacks, or counts the preceding rows wrongly."""
    rows = _table(page_text)
    row = _row(rows, "`just tag`")
    if not row:
        return ["no `just tag` row"]
    problems = []
    quoted = re.search(r"after `(git diff --quiet [^`]+)`", row[3])
    recipe = _tag_recipe(justfile_text)
    if quoted is None:
        problems.append("the `just tag` stand-in quotes no `git diff --quiet` guard")
    elif not recipe or recipe[0] != quoted.group(1):
        problems.append(f"the recipe's first line is {recipe[:1]}, the stand-in quotes {quoted.group(1)!r}")
    stated = re.search(r"commands of the (\d+) preceding rows", row[3])
    needles = ("check_release_metadata.py", "check_ruleset_sync.py", "git tag -a")
    ran = [r for r in rows[: rows.index(row)] if any(n in r[0] for n in needles)]
    if stated is None or int(stated.group(1)) != len(ran):
        problems.append(f"the stand-in states {stated and stated.group(1)} preceding rows, the table has {len(ran)}")
    return problems


def cache_clear_problems(page_text: str, cli_text: str) -> list[str]:
    """Problems: the cache-clear row names a flag or an exit code the `cache clear` command does not declare."""
    row = _row(_table(page_text), "hhemt cache clear")
    if not row:
        return ["no `hhemt cache clear` row"]
    flag = re.search(r"cache clear (--[a-z-]+)", row[0])
    problems = []
    block = cli_text.split('@cache_app.command(name="clear")', 1)[-1][:3500]
    if flag is None or f'"{flag.group(1)}"' not in block:
        problems.append(f"the row names the flag {flag and flag.group(1)}, which `cache clear` does not declare")
    code = re.search(r"exits (\d+)", row[3])
    if code is None or f"exits {code.group(1)}" not in block:
        problems.append(f"the row states the exit {code and code.group(1)}, which the `cache clear` docstring lacks")
    return problems


def test_stand_in_cells_name_ci_jobs_that_exist_and_the_docs_job_runs_the_script_bare():
    """Property: each CI job a stand-in cell names is a job key of a workflow, and `docs-build` runs the script bare.

    Class: REGRESSION on drift. The wrong state is a job under another key that leaves the cell naming a result no run
    produces; the builder's command check cannot see it, because a job name is no `python scripts/` span.
    Why required: the stand-in is the only evidence a release record cites for two steps. What kills it:
    a job under another key, or a RUN prefix added to the workflow line. A second correct implementation: parse the
    workflows with a YAML loader and compare `jobs` keys.
    """
    workflows = {p.name: _read(p) for p in sorted(_WORKFLOWS.glob("*.yml"))}
    page = _read(_PAGE)
    assert stand_in_job_problems(page, workflows) == []
    altered = page.replace("compile-synth-render-gate", "compile-gate")
    assert any("compile-gate" in p for p in stand_in_job_problems(altered, workflows))
    prefixed = dict(workflows)
    prefixed["docs-build.yml"] = workflows["docs-build.yml"].replace(
        "run: bash scripts/docs_gates.sh", "run: RUN=x bash scripts/docs_gates.sh"
    )
    assert any("bare" in p for p in stand_in_job_problems(page, prefixed))


def test_just_tag_stand_in_quotes_the_recipes_guard_line_and_counts_the_preceding_rows():
    """Property: the `just tag` stand-in's quoted guard is the first line of the recipe, and its row count is true.

    Class: REGRESSION on drift. The wrong state is a recipe whose guard covers another file set, or a table
    that gains a row, with the page still instructing the old line and count. Why required: a reader on a
    machine without `just` runs the quoted line by hand. What kills it: an edit to the recipe's first
    line, or a row added before `just tag`. A second correct implementation: run `just --dry-run tag`.
    """
    page = _read(_PAGE)
    just = _read(_JUSTFILE)
    assert just_tag_problems(page, just) == []
    edited = just.replace("CITATION.cff pyproject.toml HISTORY.md", "CITATION.cff pyproject.toml", 1)
    assert any("first line" in p for p in just_tag_problems(page, edited))
    counted = page.replace("commands of the 3 preceding rows", "commands of the 4 preceding rows")
    assert any("preceding rows" in p for p in just_tag_problems(counted, just))


def test_cache_clear_row_names_a_declared_flag_and_the_exit_the_command_documents():
    """Property: the cache-clear row's flag is declared by `cache clear` and its exit code is in the docstring.

    Class: REGRESSION on drift. The wrong state is a row naming a flag the command lost, or an exit code
    the command changed, which the page then teaches. Why required: the stand-in cell for this row is
    the exit code itself, and no other test reads it against the cli. What kills it: a flag under another name or a
    changed exit code. A second correct implementation: invoke `hhemt cache clear --help` through the
    typer test runner.
    """
    page = _read(_PAGE)
    cli = _read(_CLI)
    assert cache_clear_problems(page, cli) == []
    assert any(
        "flag" in p for p in cache_clear_problems(page.replace("cache clear --apply", "cache clear --force"), cli)
    )
    assert any("exit" in p for p in cache_clear_problems(page.replace("exits 2 by design", "exits 3 by design"), cli))
