"""The two works a hhemt user runs, TRITON and SWMM, are cited on both pages and in CITATION.cff.

hhemt drives the TRITON solver and the SWMM engine, so a reader who publishes a result
produced through hhemt owes those two works a citation as well. This module reads the
README, the home page and CITATION.cff as text and checks three things: the TRITON
DOI appears in all three, each page carries a BibTeX entry for both works, and
CITATION.cff keeps exactly one column-0 `version:` and one `date-released:` line,
because `scripts/check_release_metadata.py` reads those anchored at column 0 and
refuses a second hit (a reference entry's indented `version:` must stay indented).
Each check is a function over text, so a must-fire control plants the defect in a
string. No test compiles or runs a solver.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
TRITON_DOI = "10.1016/j.envsoft.2021.105034"
_PAGES = ("README.md", "docs/index.md")
_CFF_VERSION = re.compile(r"(?m)^version:")
_CFF_DATE = re.compile(r"(?m)^date-released:")
_BIBTEX_KEY = re.compile(r"(?m)^@(article|manual)\{(morales2021triton|rossman2022swmm),")


def _read(path: str) -> str:
    return (_REPO / path).read_text(encoding="utf-8")


def bibtex_keys(text: str) -> set[str]:
    """The cite keys of the two works' BibTeX entries present in ``text``."""
    return {m.group(2) for m in _BIBTEX_KEY.finditer(text)}


def column0_hits(text: str) -> tuple[int, int]:
    """Count of column-0 `version:` lines and of column-0 `date-released:` lines."""
    return len(_CFF_VERSION.findall(text)), len(_CFF_DATE.findall(text))


@pytest.mark.parametrize("path", ["README.md", "docs/index.md", "CITATION.cff"])
def test_triton_doi_present(path: str) -> None:
    """Property: the TRITON DOI is in the file. Class: the three published citation surfaces.

    Killed by removing the work from any one surface. A second correct form: the DOI
    written as a URL, which still contains the bare DOI string.
    """
    assert TRITON_DOI in _read(path)


@pytest.mark.parametrize("path", list(_PAGES))
def test_both_bibtex_entries_on_page(path: str) -> None:
    """Property: the page carries a BibTeX entry for TRITON and for SWMM.

    Class: the two pages. Killed by dropping either entry; a second correct form is
    the entries in either order or with extra fields.
    """
    assert bibtex_keys(_read(path)) == {"morales2021triton", "rossman2022swmm"}


def test_cff_has_one_column0_version_and_date() -> None:
    """Property: adding `references:` leaves one column-0 `version:` and `date-released:`.

    Killed by a reference entry written with a column-0 `version:`. The release
    metadata check refuses a second hit.
    """
    assert column0_hits(_read("CITATION.cff")) == (1, 1)


def test_controls_fire() -> None:
    """Must-fire controls: each check reports a planted defect, so a zero is not a broken matcher."""
    assert bibtex_keys("@article{morales2021triton,\n") == {"morales2021triton"}
    assert bibtex_keys("no entries") == set()
    assert column0_hits("version: 1\nversion: 2\ndate-released: x\n") == (2, 1)
    assert column0_hits("references:\n  - version: 1\n") == (0, 0)
