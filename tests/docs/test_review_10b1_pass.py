"""Reviewer test of chunk 10b.1: every pipe table of a published page opens with a header row and a separator row.

Without the separator row Markdown renders the table as plain lines of pipes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_DOCS = Path(__file__).resolve().parents[2] / "docs"
_SEPARATOR = re.compile(r"^\|[-: |]+\|\s*$")


def tables_without_separator(text: str) -> list[int]:
    """Line numbers of each table whose second line is not a separator row, fences masked."""
    lines, bad, fenced = text.splitlines(), [], False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            fenced = not fenced
        if fenced or not line.startswith("|") or (i and lines[i - 1].startswith("|")):
            continue
        following = lines[i + 1] if i + 1 < len(lines) else ""
        if not _SEPARATOR.match(following):
            bad.append(i + 1)
    return bad


def test_every_table_of_a_published_page_has_a_separator_row():
    """Class: every pipe table in `docs/**/*.md`. REGRESSION: red on the tree for the events table of the
    observed-events tutorial, whose separator row was dropped. Reachability: tables are the runs of lines opening
    with a pipe outside fences. Both-states anchor: the separator row exists in the pre-change and the corrected
    page. Wrong implementation: deleting the separator row while rewriting a table. A second correct page: any
    alignment form such as `|:---|---:|`. The change that greens it: restore the row `|---|---|---|---|`."""
    problems = {
        str(p.relative_to(_DOCS)): tables_without_separator(p.read_text(encoding="utf-8"))
        for p in sorted(_DOCS.rglob("*.md"))
    }
    assert {k: v for k, v in problems.items() if v} == {}


@pytest.mark.parametrize("text", ["| a | b |\n| 1 | 2 |\n"])
def test_the_separator_check_refuses_a_table_without_one(text):
    """Control: a table with no separator row is reported, and the same table with one is not."""
    assert tables_without_separator(text) == [1]
    assert tables_without_separator("| a | b |\n|:---|---:|\n| 1 | 2 |\n") == []
