"""The fixture-cache section of `docs/explanation/testing-the-toolkit.md` states what the cache marker records.

The marker records builder provenance. `hhemt.cache.keys.builder_provenance` keys include a digest of the
toolkit's Python sources (`toolkit_src_digest`) and carry no commit identity (`toolkit_sha`). The page section
"The fixture cache" says so by naming `src/hhemt/`, and its paragraph opening "The marker records" holds no
`commit` token. The test builds the provenance dict with no solver (no compile, no run) and reads the page by
anchor. Two controls feed the comparison a mutated input and expect the refusal.
"""

from __future__ import annotations

import re
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[2] / "docs" / "explanation" / "testing-the-toolkit.md"


def provenance_problems(provenance: dict) -> list[str]:
    """Problems: the provenance lacks the source digest or carries a commit identity."""
    problems = []
    if "toolkit_src_digest" not in provenance:
        problems.append("provenance has no toolkit_src_digest")
    if "toolkit_sha" in provenance:
        problems.append("provenance carries toolkit_sha")
    return problems


def section(text: str, heading: str) -> str:
    """The text from the heading line containing `heading` to the next heading of the same or higher level."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("#") and heading in ln)
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    end = next((i for i in range(start + 1, len(lines)) if re.match(rf"^#{{1,{level}}} ", lines[i])), len(lines))
    return "\n".join(lines[start:end])


def page_problems(page_text: str) -> list[str]:
    """Problems: the section omits `src/hhemt/` or its marker paragraph names a commit."""
    body = section(page_text, "The fixture cache")
    problems = []
    if "src/hhemt/" not in body:
        problems.append("the section does not name src/hhemt/")
    para = next((p for p in body.split("\n\n") if p.startswith("The marker records")), None)
    if para is None:
        problems.append("no paragraph opens with 'The marker records'")
    elif re.search(r"\bcommit", para):
        problems.append("the marker paragraph names a commit")
    return problems


def test_builder_provenance_has_the_source_digest_and_no_commit(tmp_path):
    """Property: `builder_provenance` keys include `toolkit_src_digest` and exclude `toolkit_sha`.

    Class: the provenance of an empty builder tuple. Class NEW CAPABILITY; the plausible wrong state is a
    provenance that records the commit, which the page used to claim; red under the control below.
    Why required: the page states what invalidates the cache, and the key set decides it.
    What kills it: a `toolkit_sha` key added or the digest removed. A second correct implementation:
    inspect `keys.py` with `ast` for the key names.
    """
    from hhemt.cache.keys import builder_provenance

    assert provenance_problems(builder_provenance((), repo_root=tmp_path)) == []


def test_the_page_section_names_the_sources_and_no_commit():
    """Property: the section names `src/hhemt/` and the marker paragraph holds no `commit` token."""
    assert page_problems(_PAGE.read_text(encoding="utf-8")) == []


def test_control_the_old_page_wording_is_refused():
    """A marker paragraph carrying "plus the commit the toolkit was at" is refused."""
    old = (
        "## The fixture cache\n\nSee `src/hhemt/`.\n\n"
        "The marker records the provenance: a digest plus the commit the toolkit was at.\n"
    )
    assert page_problems(old) == ["the marker paragraph names a commit"]


def test_control_a_provenance_with_a_commit_key_is_refused():
    """A provenance dict carrying `toolkit_sha` is refused."""
    assert provenance_problems({"toolkit_src_digest": "x", "toolkit_sha": "y"}) == ["provenance carries toolkit_sha"]
