"""Reviewer tests of the how-to pass: two classes of page defect the style test cannot see.

Both tests quantify over every tracked page, read through `tracked_pages` and `mask_code` of the
style test, so the page set and the code masking are the style test's own.

Class 1, `test_no_page_repeats_a_prose_line_back_to_back`: a prose line that a page repeats on the line
after it. A page that gained a duplicated line passes every other gate (the style rules grade words,
the single-source test compares different pages, `mkdocs build --strict` grades links).
Class 2, `test_every_extra_a_page_names_is_declared`: an installation extra that a page names and
`pyproject.toml` does not declare. A page that tells the reader an extra exists sends the reader to
`pip install '.[name]'`, which pip accepts with a warning and installs without the extra's packages.

Property, why required, what kills it and a second correct implementation are stated per test.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tests.docs.test_docs_style import mask_code, tracked_pages

_REPO = Path(__file__).resolve().parents[2]

# `pip install ... '.[hpc]'` or `.[hpc,test]`: the extras sit in square brackets after a path.
_PIP_EXTRAS = re.compile(r"""pip\s+install\b[^\n]*?(?<=[.\w'"])\[([A-Za-z][\w,\s-]*)\]""")
# "the empty `viz-export` extra": an inline code span followed by the word `extra`.
_NAMED_EXTRA = re.compile(r"`\[?([A-Za-z][\w-]*)\]?`\s+extras?\b")


def repeated_prose_lines(text: str) -> list[tuple[int, str]]:
    """Line numbers and text of each non-blank prose line identical to the line before it.

    Fenced blocks are masked (a code block may repeat a line on purpose); a line inside a fence is
    blank in the masked view and never counts.
    """
    raw = text.splitlines()
    masked = mask_code(text, keep_inline_code=True)
    return [
        (number + 1, raw[number])
        for number in range(1, len(raw))
        if raw[number].strip() and raw[number] == raw[number - 1] and masked[number].strip()
    ]


def extras_named(text: str) -> set[str]:
    """Extras a page names, by `pip install` bracket syntax (any fence or prose) or by `name` extra."""
    named: set[str] = set()
    for match in _PIP_EXTRAS.finditer(text):
        named.update(part.strip() for part in match.group(1).split(",") if part.strip())
    paragraphs = re.sub(r"\s+", " ", "\n".join(mask_code(text, keep_inline_code=True)))
    named.update(m.group(1) for m in _NAMED_EXTRA.finditer(paragraphs))
    return named


def declared_extras() -> set[str]:
    """The extras `pyproject.toml` declares: the keys of `[project.optional-dependencies]`."""
    data = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return set(data["project"]["optional-dependencies"])


def test_no_page_repeats_a_prose_line_back_to_back():
    """Property: no tracked page holds the same prose line twice in a row.

    Class: every non-blank line outside a fenced block, on every tracked page. Required because a
    duplicated line (here `that completed is kept.` closing a sentence twice) is read by the reader
    as a stutter and by no gate as a defect. Killed by a page that keeps the duplicate. A second correct
    implementation compares lines after collapsing runs of whitespace; both report the same page.
    """
    pages = tracked_pages(_REPO)
    found = {page: repeated_prose_lines(text) for page, text in pages.items() if repeated_prose_lines(text)}
    assert found == {}, found


def test_every_extra_a_page_names_is_declared():
    """Property: each installation extra a page names is a key of `[project.optional-dependencies]`.

    Class: every extra named by `pip install` bracket syntax or by the phrase `name` extra, on every
    tracked page. Required because a named extra that no longer exists tells the reader to install
    nothing. Killed by a page that keeps naming a removed extra (here `viz-export`). A second correct
    implementation reads the extras from `uv.lock` metadata; it reaches the same set.
    """
    declared = declared_extras()
    undeclared = {
        page: sorted(extras_named(text) - declared)
        for page, text in tracked_pages(_REPO).items()
        if extras_named(text) - declared
    }
    assert undeclared == {}, f"declared extras: {sorted(declared)}; undeclared: {undeclared}"


def test_control_the_repeat_finder_fires_on_a_repeated_line_and_not_on_a_repeated_code_line():
    """Must-fire control: the finder reports a repeated prose line and ignores a repeated fenced line."""
    assert repeated_prose_lines("A line.\nA line.\n") == [(2, "A line.")]
    assert repeated_prose_lines("```bash\nls\nls\n```\n") == []
    assert repeated_prose_lines("A line.\n\nA line.\n") == []


def test_control_the_extra_finder_fires_on_each_form_and_ignores_a_declared_name():
    """Must-fire control: both forms yield the name; a name that pyproject declares is not undeclared."""
    assert extras_named("Run `pip install -e '.[viz-export]'`.") == {"viz-export"}
    assert extras_named("pip install -e '.[hpc,test]'") == {"hpc", "test"}
    assert extras_named("The empty `viz-export` extra is kept.") == {"viz-export"}
    assert extras_named("Install the `[docs]` extra.") == {"docs"}
    assert extras_named("The `kaleido` package is core.") == set()
    assert extras_named("Run `pip install pre-commit` first, then read [pyproject.toml](x.md).") == set()
    assert extras_named("It is declared in no `pyproject.toml` extra.") == set()
    assert {"hpc"} - declared_extras() == set()
