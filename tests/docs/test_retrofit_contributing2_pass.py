"""The contributing page and the two pages it links keep the how-to template, the paths and the allowlist honest.

Three statements about `docs/contributing.md`, `docs/explanation/testing-the-toolkit.md` and
`docs/explanation/branching-and-release-model.md` have a referent outside the prose, so each is compared with
that referent:

- the contributing page names the how-to template and names as its exceptions the how-to pages that do not
  open with it, and the template and the exception set both come from the pages under `docs/how-to/`;
- every repository path the contributing page names in an inline code span is a path that git tracks, a
  directory that holds a tracked path, or a path that the page marks as a generated one;
- the three pages carry no `baseline` entry in the style allowlist.

No test writes a count: a set is compared with a set. Each check has a control that feeds the check a mutated
input and expects the refusal. No test here compiles or runs a solver.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.docs

_REPO = Path(__file__).resolve().parents[2]
_CONTRIBUTING = "docs/contributing.md"
_PAGES = (
    _CONTRIBUTING,
    "docs/explanation/testing-the-toolkit.md",
    "docs/explanation/branching-and-release-model.md",
)
_HOWTO = "docs/how-to/"
_ALLOWLIST = _REPO / "tests" / "docs" / "style_allowlist.yaml"
#: Top-level locations whose paths the contributing page names; a path elsewhere (`.venv/`, `.git/hooks/`) is local.
_TRACKED_ROOTS = ("docs/", "scripts/", "tests/", "src/", ".github/workflows/")
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_H2 = re.compile(r"^## (.+?)\s*$", re.M)


def _tracked() -> list[str]:
    out = subprocess.run(["git", "-C", str(_REPO), "ls-files", "-z"], capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\0") if p]


def _read(path: str) -> str:
    return (_REPO / path).read_text(encoding="utf-8")


def _prose(text: str) -> str:
    """`text` without fenced code blocks."""
    out, fence = [], None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if m:
            fence = None if fence == m.group(1)[0] else (fence or m.group(1)[0])
            continue
        if fence is None:
            out.append(line)
    return "\n".join(out)


def _leading_h2(text: str, count: int) -> tuple[str, ...]:
    return tuple(_H2.findall(_prose(text)))[:count]


def howto_template_and_exceptions(pages: dict[str, str]) -> tuple[tuple[str, ...], set[str]]:
    """(the template, the pages without it) over the how-to pages in `pages`.

    The template is the longest run of leading H2 sections that every page in the majority shares; a page
    that does not open with the template is an exception. The majority is the most common opening.
    """
    openings = {path: _leading_h2(text, 3) for path, text in pages.items() if path.startswith(_HOWTO)}
    counts: dict[tuple[str, ...], int] = {}
    for opening in openings.values():
        counts[opening] = counts.get(opening, 0) + 1
    template = max(counts, key=counts.__getitem__)
    return template, {path for path, opening in openings.items() if opening != template}


def check_template_statement(page_text: str, template: tuple[str, ...], exceptions: set[str]) -> list[str]:
    """Problems: the page omits a template section, states them out of order, or misses an exception."""
    prose = _prose(page_text)
    problems = []
    positions = [prose.find(section) for section in template]
    if -1 in positions:
        problems.append(f"the page does not name the template sections {list(template)}")
    elif positions != sorted(positions):
        problems.append(f"the page names the template sections out of order: {list(template)}")
    problems += [f"the page does not name the exception {path}" for path in sorted(exceptions) if path not in prose]
    return problems


def _howto_pages() -> dict[str, str]:
    return {p: _read(p) for p in _tracked() if p.startswith(_HOWTO) and p.endswith(".md")}


def test_the_contributing_page_states_the_how_to_template_and_its_exceptions():
    """Property: the contributing page names the how-to template in order and each page that lacks it.

    Class: the pages under `docs/how-to/`. The template is the opening that most pages share and the
    exceptions are the pages with another opening, both derived from the pages and never typed here.

    Why it is required: a contributor who adds a how-to page learns the template from the page, and a page
    that states a different template or omits the index exception sends the next page the wrong way.

    What kills it: a page under `docs/how-to/` that changes its opening without the contributing page
    following, and a contributing page that drops the template sentence.

    A second correct implementation under which it still passes: a list of the template sections in a table.
    """
    pages = _howto_pages()
    template, exceptions = howto_template_and_exceptions(pages)
    assert template, "no template found"
    assert exceptions, "no exception found; the index page is expected to be one"
    assert check_template_statement(_read(_CONTRIBUTING), template, exceptions) == []


def test_control_a_contributing_page_that_drops_the_exception_or_reorders_the_template_is_refused():
    pages = _howto_pages()
    template, exceptions = howto_template_and_exceptions(pages)
    text = _read(_CONTRIBUTING)
    assert check_template_statement(text, template, exceptions) == []
    for path in exceptions:
        assert check_template_statement(text.replace(path, "docs/how-to/elsewhere.md"), template, exceptions)
    reordered = text.replace(template[0], "ZZZ", 1) + f"\n{template[0]}\n"
    assert check_template_statement(reordered, template, exceptions)
    assert check_template_statement(text.replace(template[1], "ZZZ"), template, exceptions)


def check_named_paths(page_text: str, tracked: list[str]) -> list[str]:
    """Problems: an inline code span names a repository path that git does not track.

    A path with a `{` or `*` is a pattern and is skipped. A path ending in `/` is a directory and passes when
    a tracked path starts with it.
    """
    problems = []
    for span in _CODE_SPAN.findall(_prose(page_text)):
        path = span.split()[0] if span.split() else ""
        if not path.startswith(_TRACKED_ROOTS) or any(ch in path for ch in "{}*<>"):
            continue
        path = path.split("#", 1)[0]
        known = any(p.startswith(path) for p in tracked) if path.endswith("/") else path in tracked
        if not known:
            problems.append(f"the page names {path}, which git does not track")
    return problems


def test_every_repository_path_the_contributing_page_names_is_tracked():
    """Property: each `docs/`, `scripts/`, `tests/`, `src/` or `.github/workflows/` path in a code span exists.

    Class: the inline code spans of the contributing page, fenced blocks excluded.

    Why it is required: a path to a deleted file or directory sends a contributor to a place that is gone,
    and the page does not otherwise change when the file moves.

    What kills it: a deleted or moved file or directory that the page still names.

    A second correct implementation under which it still passes: a link in place of a code span.
    """
    assert check_named_paths(_read(_CONTRIBUTING), _tracked()) == []


def test_control_a_path_that_git_does_not_track_is_refused():
    tracked = _tracked()
    assert check_named_paths("Edit `docs/contributing.md` and `docs/how-to/`.", tracked) == []
    assert check_named_paths("Add an entry under `docs/faq-removed/`.", tracked)
    assert check_named_paths("Open `docs/how-to/removed-page.md`.", tracked)
    without_howto = [p for p in tracked if not p.startswith(_HOWTO)]
    assert check_named_paths("Add a page under `docs/how-to/`.", without_howto)


def baseline_entries_of(pages: tuple[str, ...], entries: list[dict]) -> list[str]:
    """The `baseline` entries of `pages` that carry no `surface`, each as `page rule digest`."""
    return [
        f"{e['page']} {e['rule']} {e.get('line_sha1', '')}"
        for e in entries
        if e["status"] == "baseline" and e.get("surface") is None and e["page"] in pages
    ]


def test_the_three_pages_carry_no_baseline_entry():
    """Property: no `baseline` entry of the style allowlist names the contributing, testing or branching page.

    Class: the entries of `tests/docs/style_allowlist.yaml` with no `surface` field and one of the three pages.

    Why it is required: a `baseline` entry is a violation the page still holds, and the pass over these pages
    repairs every line the rules flag; the rules do not exempt a line by position.

    What kills it: a new flagged line on one of the pages that a `baseline` entry lists in place of a rewrite.

    A second correct implementation under which it still passes: a `permitted` entry that carries the
    clearer phrasing tried first in its reason.
    """
    entries = yaml.safe_load(_ALLOWLIST.read_text(encoding="utf-8"))["entries"]
    assert baseline_entries_of(_PAGES, entries) == []


def test_control_a_baseline_entry_of_one_of_the_pages_is_found():
    planted = {"page": _PAGES[1], "rule": "parenthesis", "status": "baseline", "line_sha1": "0" * 40}
    other_page = {**planted, "page": "docs/index.md"}
    on_surface = {**planted, "surface": "api-docstring"}
    permitted = {**planted, "status": "permitted"}
    assert baseline_entries_of(_PAGES, [planted])
    assert baseline_entries_of(_PAGES, [other_page, on_surface, permitted]) == []
