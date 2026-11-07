"""The explanation and root pass leaves no baseline style exception on its pages.

The claim has a class: every one of the 16 pages of the explanation and root subset is graded by the style test
with no `baseline` allowlist entry, over every rule, with one exception that the pass states:

- the `permitted` entry for the licence-grant line of `docs/contributing.md`, whose `address` hit is the project
  maintainer, a role the sentence defines.

The style test (`test_docs_style.py`) already fails on an unlisted hit and on a stale entry. It does not fail on a
`baseline` entry that is present and live, because that is exactly the state the rule's first version recorded.
This file closes that gap for the subset: a live `baseline` entry on a subset page is the finding.

Class: REGRESSION against the allowlist as it stood before the pass, which held 217 entries for these pages. The
controls below build the failing input: the live entries plus one `baseline` entry for a subset page.

The measuring function takes the entries and returns the problems, so the controls run it on constructed input.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from tests.docs.test_docs_style import EARLIER_RULES

_REPO = Path(__file__).resolve().parents[2]
_ALLOWLIST = Path(__file__).resolve().parent / "style_allowlist.yaml"

SUBSET_PAGES = frozenset(
    {
        "docs/explanation/branching-and-release-model.md",
        "docs/explanation/capabilities.md",
        "docs/explanation/coupled-inflow-nodes.md",
        "docs/explanation/exploratory-analysis.md",
        "docs/explanation/limitations.md",
        "docs/explanation/software-availability.md",
        "docs/explanation/testing-the-toolkit.md",
        "docs/index.md",
        "docs/contributing.md",
        "docs/_snippets/abbreviations.md",
        "docs/_snippets/hpc-system-config-role.md",
        "docs/_snippets/install-commands.md",
        "docs/_snippets/platform-support.md",
        "docs/_snippets/prerequisites-base.md",
        "README.md",
        "docs/faq/index.md",
    }
)

# A baseline key the measure exempts; the allowlist holds no such entry, and the pass leaves no baseline entry.
_N16 = ("docs/explanation/exploratory-analysis.md", "cruft", "## Some figures are not yet drawn")

# The one permitted entry the pass writes: the licence grant names the project maintainer and defines the role.
_PERMITTED = (
    "docs/contributing.md",
    "address",
    "2. **Forward relicensing.** You grant the project maintainer, the owner of the repository, permission to",
)

_DEVELOPER = re.compile(r"\bdeveloper\b", re.IGNORECASE)


def _page_entries(entries: list[dict]) -> list[dict]:
    """The entries of the rules this pass graded that grade a page of the subset.

    An entry with a `surface` field grades another surface. An entry of a banned-term rule (a rule outside
    `EARLIER_RULES`) is a baseline for the page pass of its own subset and is not this pass's to remove.
    """
    return [e for e in entries if e["page"] in SUBSET_PAGES and not e.get("surface") and e["rule"] in EARLIER_RULES]


def subset_problems(entries: list[dict]) -> list[str]:
    """Every entry on a subset page that the pass does not state, one message per entry."""
    problems = []
    for e in _page_entries(entries):
        key = (e["page"], e["rule"], e["line"])
        if e["status"] == "baseline" and key != _N16:
            problems.append(f"baseline entry left on a subset page: {e['page']} rule {e['rule']}: {e['line']}")
        elif e["status"] == "permitted" and key != _PERMITTED:
            problems.append(f"permitted entry the pass did not state: {e['page']} rule {e['rule']}: {e['line']}")
    return problems


def developer_lines_without_entry(contributing: str, entries: list[dict]) -> list[str]:
    """Lines of the contributing page that name `developer` as a word and carry no `permitted` `address` entry."""
    permitted = {
        e["line"]
        for e in entries
        if e["page"] == "docs/contributing.md" and e["rule"] == "address" and e["status"] == "permitted"
    }
    return [line for line in contributing.splitlines() if _DEVELOPER.search(line) and line.rstrip() not in permitted]


def _live_entries() -> list[dict]:
    return yaml.safe_load(_ALLOWLIST.read_text(encoding="utf-8"))["entries"]


def test_the_subset_pages_carry_no_baseline_entry_but_the_one_row_n16_assigns():
    """Class: REGRESSION. Red on the pre-pass allowlist (217 entries on these pages).

    Green when only the exempted key and the grant remain.
    """
    assert subset_problems(_live_entries()) == []


def test_the_two_stated_entries_are_present_so_the_pass_is_not_an_empty_allowlist():
    """The zero above agrees with the hypothesis, so this control varies the property: the two stated entries exist."""
    keys = {(e["page"], e["rule"], e["line"]) for e in _page_entries(_live_entries())}
    assert keys == {_PERMITTED}


def test_the_contributing_page_names_no_developer_role_outside_a_permitted_entry():
    """Class: the Development Principles lines address a human contributor, not an assistant or 'the developer'."""
    text = (_REPO / "docs" / "contributing.md").read_text(encoding="utf-8")
    assert developer_lines_without_entry(text, _live_entries()) == []


def test_the_contributing_page_names_no_tracker_it_cannot_point_to():
    """The page told the reader to note candidates in 'a dedicated tracking document' that no page or file names."""
    text = (_REPO / "docs" / "contributing.md").read_text(encoding="utf-8")
    assert "tracking document" not in text.lower()


def test_control_a_baseline_entry_on_a_subset_page_is_reported():
    """Must-fire: the live entries plus one baseline entry for a subset page fail the measure."""
    extra = {"page": "docs/index.md", "rule": "it", "line": "x", "status": "baseline", "reason": "r"}
    assert subset_problems([*_live_entries(), extra]) == [
        "baseline entry left on a subset page: docs/index.md rule it: x"
    ]


def test_control_a_baseline_entry_on_another_page_is_not_reported():
    """The measure reads the subset and nothing else: the same entry on a page of another subset passes."""
    extra = {"page": "docs/how-to/installation.md", "rule": "it", "line": "x", "status": "baseline", "reason": "r"}
    assert subset_problems([*_live_entries(), extra]) == []


def test_control_a_developer_line_with_no_entry_is_reported():
    """Must-fire: a line that names the developer fails the measure; one with a permitted address entry passes."""
    line = "Raise the concern with the developer."
    assert developer_lines_without_entry(line, []) == [line]
    permitted = [{"page": "docs/contributing.md", "rule": "address", "status": "permitted", "line": line}]
    assert developer_lines_without_entry(line, permitted) == []
