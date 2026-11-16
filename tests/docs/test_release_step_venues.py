"""The release page's venue table says where each command of the release walk runs.

The release walk has seven steps, and not every machine can run every step: a machine
without the documentation dependencies or a compiler runs only some, and `just` is optional. The table "Release step
venues" on `docs/explanation/branching-and-release-model.md` names, per step, the venue, the tool the step
needs, and what stands for the step when it does not run there. `docs/contributing.md` points at the
table and states that the CI `docs-build` job stands for the docs gates.

The tests read text only (the page, the contributing page, the justfile and the workflow files), run no
solver and execute no release command. Each check is a function that returns a list of problems, and a
control test feeds the function a mutated input and expects the named problem, so a function that
returns an empty list for every input fails the controls.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _ROOT / "docs" / "explanation" / "branching-and-release-model.md"
_CONTRIBUTING = _ROOT / "docs" / "contributing.md"
_JUSTFILE = _ROOT / "justfile"
_WORKFLOWS = _ROOT / ".github" / "workflows"

_HEADING = "Release step venues"
_ANCHOR = "release-step-venues"
_DOCS_SENTENCE = (
    "The CI `docs-build` job runs the same script, and the result of the job stands for this step on a machine "
    "without the docs dependencies."
)

# One row per release-walk command: the command in the first cell, then the tokens the venue,
# tool and stand-in cells must carry. An empty stand-in token means the cell is not graded.
_EXPECTED = (
    ("bash scripts/docs_gates.sh", "docs-build", "docs", "`docs-build`"),
    ("hhemt cache clear --apply", "compiling venue", "hhemt", "exits 2"),
    ("HHEMT_COMPILE_VENUE=toolchain", "compiling venue", "compiler", "`compile-synth-render-gate`"),
    ("python scripts/check_release_metadata.py", "any machine", "python", ""),
    ("python scripts/check_ruleset_sync.py --static --live", "any machine", "`gh`", ""),
    ("git tag -a", "any machine", "`git`", ""),
    ("`just tag`", "any machine", "`just`", "git diff --quiet HEAD -- CITATION.cff pyproject.toml HISTORY.md"),
)

# The order in which the recipe `tag:` of the justfile runs the three steps that have table rows.
_TAG_RECIPE_STEPS = (
    ("python scripts/check_release_metadata.py", "python scripts/check_release_metadata.py"),
    ("python scripts/check_ruleset_sync.py --static --live", "just ruleset-check"),
    ("git tag -a", "git tag -a"),
)


def section(text: str, heading: str) -> str:
    """The text from the heading line containing `heading` to the next heading of the same or higher level.

    An absent heading returns the empty string, so a page without the section reports a problem and
    does not raise.
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("#") and heading in ln), None)
    if start is None:
        return ""
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    end = next((i for i in range(start + 1, len(lines)) if re.match(rf"^#{{1,{level}}} ", lines[i])), len(lines))
    return "\n".join(lines[start:end])


def table_rows(body: str) -> list[list[str]]:
    """The cell lists of the first table in `body`, header and separator rows dropped."""
    rows = []
    for line in body.splitlines():
        if line.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", line):
            rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows[1:]


def recipe_body(justfile_text: str, name: str) -> list[str]:
    """The indented lines of the justfile recipe `name`."""
    lines = justfile_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"{name}:")), None)
    if start is None:
        return []
    body = []
    for ln in lines[start + 1 :]:
        if ln and not ln.startswith((" ", "\t")):
            break
        body.append(ln.strip())
    return [ln for ln in body if ln]


def page_problems(page_text: str, justfile_text: str) -> list[str]:
    """Problems: a row is missing, names the wrong venue, tool or stand-in, or the tag steps run out of order."""
    body = section(page_text, _HEADING)
    if not body:
        return [f"the page has no section '{_HEADING}'"]
    rows = table_rows(body)
    problems = []
    index = {}
    for command, venue, tool, stand_in in _EXPECTED:
        position = next((i for i, row in enumerate(rows) if command in row[0]), None)
        if position is None:
            problems.append(f"no row for the command {command}")
            continue
        index[command] = position
        row = rows[position]
        if len(row) != 4:
            problems.append(f"the row for {command} has {len(row)} cells, not 4")
            continue
        if venue not in row[1].lower():
            problems.append(f"the venue of {command} lacks '{venue}'")
        if tool not in row[2].lower() and tool not in row[2]:
            problems.append(f"the tool of {command} lacks '{tool}'")
        if stand_in and stand_in not in row[3]:
            problems.append(f"the stand-in of {command} lacks '{stand_in}'")
        if venue != "any machine" and not row[3]:
            problems.append(f"the row for {command} runs off any machine and names no stand-in")
    recipe = recipe_body(justfile_text, "tag")
    positions = {
        step: next((i for i, ln in enumerate(recipe) if needle in ln), None) for step, needle in _TAG_RECIPE_STEPS
    }
    recipe_order = sorted((step for step, at in positions.items() if at is not None), key=positions.get)
    table_order = sorted((step for step, _ in _TAG_RECIPE_STEPS if step in index), key=index.get)
    if recipe_order != table_order:
        problems.append(f"the table orders the tag steps {table_order}, the recipe runs {recipe_order}")
    return problems


def command_problems(page_text: str, sources: str) -> list[str]:
    """Problems: a `python scripts/` or `bash scripts/` command of the table is run by no recipe or workflow."""
    problems = []
    for row in table_rows(section(page_text, _HEADING)):
        for span in re.findall(r"`((?:python|bash) scripts/[^`]+)`", row[0]):
            if span not in sources:
                problems.append(f"the command '{span}' appears in no recipe or workflow")
    return problems


def contributing_problems(contributing_text: str, page_text: str) -> list[str]:
    """Problems: the sentence or the pointer is absent, or the pointer's anchor names no heading of the page."""
    problems = []
    paragraph = next((p for p in contributing_text.split("\n\n") if "`just docs-check` runs the same line" in p), "")
    if _DOCS_SENTENCE not in paragraph:
        problems.append("the `just docs-check` paragraph lacks the docs-build sentence")
    link = f"explanation/branching-and-release-model.md#{_ANCHOR}"
    if link not in contributing_text:
        problems.append(f"contributing.md does not link {link}")
    if not any(
        ln.startswith("#") and ln.lstrip("# ").lower().replace(" ", "-") == _ANCHOR for ln in page_text.splitlines()
    ):
        problems.append(f"the release page has no heading whose anchor is {_ANCHOR}")
    return problems


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sources() -> str:
    return _read(_JUSTFILE) + "\n".join(_read(p) for p in sorted(_WORKFLOWS.glob("*.yml")))


def test_release_page_table_names_venue_tool_and_stand_in_per_step():
    """Property: each of the seven release-walk commands has a row with the venue, tool and stand-in the walk needs.

    Class: NEW CAPABILITY. The plausible wrong implementation is a page that lists the commands without a
    venue, or puts the docs gates on any machine, which is the state the page was in before the table; the
    function reports a problem for the page without the section and for a wrong venue, shown red by the
    controls below. Why required: a walk that never says where a step runs sends the reader to a machine
    for a step that machine lacks the means to run (the `docs_gates.sh` exit 1 and the `cache clear` exit 2 of the
    v0.2.0 walk). What kills it: a row dropped, a venue changed, or the `just tag` row out of step with the recipe.
    A second correct implementation: parse the table with a markdown library and compare the cells.
    """
    assert page_problems(_read(_PAGE), _read(_JUSTFILE)) == []


def test_page_check_refuses_a_wrong_venue_a_dropped_stand_in_and_a_reordered_tag_step():
    """Property: `page_problems` reports each defect the table can carry, so its empty list above is a claim.

    Class: NEW CAPABILITY, the control of the test above. Each mutation breaks one property the real page
    holds, and the expected problem is the one that property's check emits. Why required: a checker that
    returns an empty list for every page passes the test above. What kills it: a checker that skips the
    venue, stand-in or order check. A second correct implementation: compare against a stored table.
    """
    page = _read(_PAGE)
    just = _read(_JUSTFILE)
    wrong_venue = page.replace(
        "| `bash scripts/docs_gates.sh` | CI `docs-build` job |", "| `bash scripts/docs_gates.sh` | Any machine |"
    )
    assert any("venue of bash scripts/docs_gates.sh" in p for p in page_problems(wrong_venue, just))
    assert page_problems("# Page\n\nNo table here.\n", just) == [f"the page has no section '{_HEADING}'"]
    dropped = re.sub(r"(\| `hhemt cache clear --apply` \|[^\n]*\|)[^|\n]*\|\n", r"\1 |\n", page)
    assert any("cache clear" in p and "stand-in" in p for p in page_problems(dropped, just))
    reordered = just.replace(
        "python scripts/check_release_metadata.py\n    just ruleset-check",
        "just ruleset-check\n    python scripts/check_release_metadata.py",
    )
    assert any("recipe runs" in p for p in page_problems(page, reordered))


def test_table_commands_are_run_by_a_recipe_or_a_workflow_and_the_check_sees_a_renamed_script():
    """Property: every `python scripts/` or `bash scripts/` command of the table appears in the justfile or a workflow.

    Class: REGRESSION on drift. The plausible wrong state is a table row that names a script the justfile
    and the workflows no longer run, so the page documents a command nobody runs; the control renames one
    script in the page and expects the problem. Why required: the table is the only place the walk's
    commands are listed together. What kills it: a script whose path changed, or a row that cites a command from memory.
    A second correct implementation: run `shlex.split` over each span and check the path with `Path.exists`.
    """
    page = _read(_PAGE)
    assert command_problems(page, _sources()) == []
    altered = page.replace("scripts/check_ruleset_sync.py --static --live", "scripts/check_ruleset.py --static --live")
    assert command_problems(altered, _sources()) == [
        "the command 'python scripts/check_ruleset.py --static --live' appears in no recipe or workflow"
    ]


def test_contributing_carries_the_docs_build_sentence_and_a_pointer_to_the_table():
    """Property: contributing.md states the docs-build stand-in sentence and links the table's heading anchor.

    Class: NEW CAPABILITY. The plausible wrong implementation is a pointer to a heading that does not exist,
    which the strict mkdocs build reports only for a link it can resolve, or the sentence placed outside the
    `just docs-check` paragraph; the controls remove the sentence and the heading. Why required: a reader of
    the contributing guide runs the docs gates on a machine without the docs dependencies. What kills it:
    the sentence missing, the link missing, or the heading text altered. A second correct implementation: build
    the site and follow the link.
    """
    contributing = _read(_CONTRIBUTING)
    page = _read(_PAGE)
    assert contributing_problems(contributing, page) == []
    assert any(
        "lacks the docs-build sentence" in p
        for p in contributing_problems(contributing.replace(_DOCS_SENTENCE, ""), page)
    )
    assert any(
        "no heading whose anchor" in p
        for p in contributing_problems(contributing, page.replace(f"### {_HEADING}", "### Other heading"))
    )
