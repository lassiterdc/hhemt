"""Reviewer checks for the TRITON and SWMM citations on both pages and in CITATION.cff.

The two works are cited three times (README, home page, CITATION.cff) and a reader
copies from whichever surface they open, so the three must name the same work. These
checks quantify over the surfaces and over the authors CITATION.cff lists; they read
text only and reach no solver. Implements approved-shape row 1a (D-1 design).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SURFACES = ("README.md", "docs/index.md", "CITATION.cff")
_PAGES = ("README.md", "docs/index.md")
EPA_REPORT = "EPA/600/R-22/030"


def _read(path: str) -> str:
    return (_REPO / path).read_text(encoding="utf-8")


def _cff_references() -> list[dict]:
    return yaml.safe_load(_read("CITATION.cff"))["references"]


@pytest.mark.parametrize("path", list(_SURFACES))
def test_swmm_manual_carries_the_epa_report_number(path: str) -> None:
    """Property: the SWMM 5.2 manual reference names EPA report number EPA/600/R-22/030.

    Class: the three citation surfaces. Class kind: REGRESSION, red on the pre-change tree
    (the coder wrote the reference without a number). Approved-shape row: D-1 design,
    "The EPA report number is read from the EPA record before writing. If the record
    carries none, the reference is written without one." The EPA record carries it.
    Killed by dropping the number from any one surface; a second correct form is the
    number in a CFF `number:` field or in the APA sentence, in either position.
    """
    assert EPA_REPORT in _read(path)


@pytest.mark.parametrize("path", list(_PAGES))
def test_every_cff_author_family_name_is_on_the_page(path: str) -> None:
    """Property: each author CITATION.cff lists for a reference also appears on each page.

    Class: the two pages crossed with every CFF reference author. Class kind: NEW
    CAPABILITY; a plausible wrong implementation is an author list shortened on one
    surface. Killed by deleting any author from the page. A second correct form is the
    names in BibTeX escaped form, since only the ASCII-safe surname stem is compared.
    """
    page = _read(path)
    missing = []
    for ref in _cff_references():
        for author in ref["authors"]:
            stem = author["family-names"].split("-")[0]
            if stem not in page:
                missing.append(author["family-names"])
    assert missing == []


@pytest.mark.parametrize("path", list(_PAGES))
def test_triton_volume_article_and_year_agree_with_cff(path: str) -> None:
    """Property: the TRITON volume, article number and year on a page equal CITATION.cff's.

    Class: the two pages. Class kind: NEW CAPABILITY; a plausible wrong implementation
    is a page citing a different year or volume than the file. Killed by editing any of
    the three values on one surface. A second correct form is a different citation style
    that still prints the three values.
    """
    ref = next(r for r in _cff_references() if r["type"] == "article")
    page = _read(path)
    for value in (str(ref["volume"]), str(ref["start"]), str(ref["year"])):
        assert value in page


def test_cff_reference_types_are_article_then_manual() -> None:
    """Property: CITATION.cff references are exactly one article and one manual, DOI on the article.

    Class: the references list. Class kind: NEW CAPABILITY. Killed by a third entry or a
    swapped type. A second correct form is the two entries in the other order.
    """
    refs = _cff_references()
    assert sorted(r["type"] for r in refs) == ["article", "manual"]
    assert [r["doi"] for r in refs if r["type"] == "article"] == ["10.1016/j.envsoft.2021.105034"]


def test_review_controls_fire() -> None:
    """Must-fire control: the report-number matcher is a substring test that fails on text lacking it."""
    assert EPA_REPORT in "Report EPA/600/R-22/030."
    assert EPA_REPORT not in "U.S. Environmental Protection Agency."
