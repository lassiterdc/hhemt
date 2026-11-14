"""The field table of `docs/how-to/reprex-roundtrip.md` names exactly the fields of the printed `ReprexResult`.

`hhemt reprex check` prints a `ReprexResult` dataclass; it hands a reader no Python variable
named `result`. The how-to page lists the printed fields in a table under "Report problems and
amendments". This test locates that table by its section heading, takes the backticked first
cell of each row, and compares the set with the dataclass fields. Names are compared as sets;
no sentence of prose is compared, no count is written down and no solver runs. Two controls feed
the comparison a mutated input and expect the refusal.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[2] / "docs" / "how-to" / "reprex-roundtrip.md"
_SECTION = "Report problems and amendments"


def table_fields(page_text: str) -> set[str]:
    """The backticked first cells of the table rows in the section whose list item names `_SECTION`."""
    lines = page_text.splitlines()
    start = next(i for i, ln in enumerate(lines) if _SECTION in ln)
    names: set[str] = set()
    in_table = False
    for ln in lines[start + 1 :]:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        row = ln.strip().startswith("|")
        if row:
            in_table = True
            m = re.fullmatch(r"`([a-z_]+)`", cells[0])
            if m:
                names.add(m.group(1))
        elif in_table:
            break
    return names


def code_fields() -> set[str]:
    """The field names of the `ReprexResult` dataclass."""
    from hhemt.bundle._reprex import ReprexResult

    return {f.name for f in dataclasses.fields(ReprexResult)}


def check(names: set[str], fields: set[str]) -> list[str]:
    """Problems: a field the table omits, or a name the table lists and the dataclass lacks."""
    return [f"{n} is a field and the table omits it" for n in sorted(fields - names)] + [
        f"{n} is in the table and is not a field" for n in sorted(names - fields)
    ]


def test_the_field_table_names_the_fields_of_the_printed_result():
    """Property: the page's field table and `dataclasses.fields(ReprexResult)` name the same set.

    Class: every field of the dataclass and every first cell of the table. Class NEW CAPABILITY;
    the plausible wrong state is a page that names a field the dataclass lacks (or the old
    `result.*` references to a variable the command does not expose); red under the controls.
    Why required: a reader looks up a printed field by name, and a name the code does not print
    sends them to a field that is not there.
    What kills it: a field added to or removed from `ReprexResult` without the table row.
    A second correct implementation: parse the table with a Markdown library.
    """
    assert check(table_fields(_PAGE.read_text(encoding="utf-8")), code_fields()) == []


def test_control_a_dropped_name_is_refused():
    """Drop one parsed name: the comparison reports the field the table now omits."""
    names = table_fields(_PAGE.read_text(encoding="utf-8"))
    dropped = sorted(names)[0]
    assert check(names - {dropped}, code_fields()) == [f"{dropped} is a field and the table omits it"]


def test_control_an_added_name_is_refused():
    """Add `not_a_field`: the comparison reports a name the dataclass lacks."""
    names = table_fields(_PAGE.read_text(encoding="utf-8"))
    assert check(names | {"not_a_field"}, code_fields()) == ["not_a_field is in the table and is not a field"]


def test_the_table_is_found_and_not_empty():
    """The parser reads a non-empty table, so the equality above is not the equality of two empty sets."""
    assert table_fields(_PAGE.read_text(encoding="utf-8"))
