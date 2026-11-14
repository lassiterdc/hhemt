"""Reviewer test for chunk 10g.1: no published description puts a determiner before the word `own`.

Property: the rendered configuration schema and the rendered API docstrings contain no `the own`, `a own` or
`an own`, in any letter case.

Class: every line of the two surfaces the style test reads (`config-schema` and `api-docstring`), which render the
`Field` descriptions, the option glossaries and the docstrings of the public API.

Why it is required: a rewrite that removes a possessive (`its own`, `the running toolkit's own identity`) and keeps
`own` leaves an ungrammatical phrase (`the own identity of the running toolkit`) that no style rule matches and that a
reader sees as a typing error. The phrase is never grammatical English, so the matcher has no false positive.

What kills it: any description or docstring edit that drops a possessive in front of `own` and keeps the determiner.

A second correct implementation under which it still passes: a matcher over the pages `docs/reference/config-schema.md`
and `docs/reference/api-*.md` built by `mkdocs build` in place of the two surface renderers.
"""

from __future__ import annotations

import re

import pytest

from tests.docs.test_published_surfaces_style import surface_pages

pytestmark = pytest.mark.docs

_DETERMINER_OWN = re.compile(r"\b(?:the|a|an) own\b", re.IGNORECASE)
_SURFACES = ("config-schema", "api-docstring")


def determiner_own_hits(pages: dict[str, str]) -> list[str]:
    """Lines of `pages` (page -> text) that carry a determiner directly before `own`, each as `page:line: text`."""
    hits = []
    for page, text in sorted(pages.items()):
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _DETERMINER_OWN.search(line):
                hits.append(f"{page}:{lineno}: {line.strip()[:140]}")
    return hits


def test_no_published_surface_puts_a_determiner_before_own():
    pages: dict[str, str] = {}
    for surface in _SURFACES:
        pages.update({f"{surface}:{page}": text for page, text in surface_pages(surface).items()})
    assert determiner_own_hits(pages) == []


def test_control_a_determiner_before_own_is_found_and_a_possessive_is_not():
    """Must-fire control: the matcher flags each spelling of the defect and passes the correct spellings."""
    for bad in (
        "compares the sha to the own identity of the running toolkit",
        "against the reference of the OWN HARDWARE FAMILY of the member",
        "a own value",
        "An Own value",
    ):
        assert determiner_own_hits({"p.md": bad}), bad
    for good in (
        "compares the sha to the identity of the running toolkit",
        "against the reference of ITS OWN HARDWARE FAMILY",
        "the owner of the repository",
        "on its own",
    ):
        assert determiner_own_hits({"p.md": good}) == [], good
