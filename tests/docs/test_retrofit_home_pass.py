"""Tests for the Home, explanation and navigation pass: six properties of the pages and the navigation.

Each property reads page text through a function that takes the text, so a must-fire control plants the defect in a
string and shows that the function reports it. No test compiles or runs a solver.

1. The pages the pass deletes or moves are absent, and the moved page is present.
2. The navigation lists the Quickstart after Home and before Tutorials.
3. The Home page carries the bold platform sentence through the snippet that is its one source, the license
   statement, and the citation anchor that the README links; it names no release by number.
4. The only pages that include the platform snippet or write the word `Linux` outside the snippet are the Home page
   and the installation guide, among the pages this pass owns.
5. The capabilities page has `Supported experiments` with three subsections and none of the removed headings.
6. The limitations page holds none of the removed sections and no solver commit hash.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_INCLUDE = re.compile(r'^--8<--\s+"([^"]+)"\s*$', re.M)
_REMOVED_HEADINGS = re.compile(
    r"^#{1,6}\s+(Example run|Analysis kinds|More GPUs can run slower|Scientific validity|Release maturity)\s*$",
    re.I | re.M,
)
_SOLVER_COMMIT = re.compile(r"\b(?:bd18215|bb13ea28)\b")
_RELEASE_NUMBER = re.compile(r"(?<![\w.])0\.1\.0(?![\w.])")


def _read(path: str) -> str:
    return (_REPO / path).read_text(encoding="utf-8")


def _tracked(path: str) -> bool:
    out = subprocess.run(
        ["git", "-C", str(_REPO), "ls-files", "--", path], capture_output=True, text=True, check=True
    ).stdout
    return bool(out.strip())


def expanded(text: str) -> str:
    """Page text with each snippet include replaced by the snippet file, as the build renders it."""
    return _INCLUDE.sub(lambda m: _read(f"docs/_snippets/{m.group(1)}"), text)


def nav_order(mkdocs_text: str) -> list[str]:
    """The top-level navigation labels in file order."""
    nav = mkdocs_text.split("\nnav:\n", 1)[1]
    return re.findall(r"^  - ([^:\n]+):", nav, re.M)


def removed_headings(text: str) -> list[str]:
    return _REMOVED_HEADINGS.findall(text)


def h3_titles_under(text: str, h2: str) -> list[str]:
    block = re.search(rf"^## {re.escape(h2)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    return re.findall(r"^### (.+)$", block.group(1), re.M) if block else []


def test_the_deleted_pages_are_gone_and_the_moved_page_is_present():
    assert not (_REPO / "docs/explanation/software-availability.md").exists()
    assert not (_REPO / "docs/tutorials/quickstart.md").exists()
    assert (_REPO / "docs/quickstart.md").is_file()
    assert _tracked("docs/index.md") and _tracked("docs/explanation/capabilities.md")


def test_the_navigation_lists_the_quickstart_after_home_and_before_tutorials():
    order = nav_order(_read("mkdocs.yml"))
    assert order.index("Home") + 1 == order.index("Quickstart") < order.index("Tutorials"), order
    assert "tutorials/quickstart.md" not in _read("mkdocs.yml")
    # Control: the order function reports a Quickstart that sits under Tutorials as a different position.
    moved = "x\nnav:\n  - Home:\n      - index.md\n  - Tutorials:\n      - Quickstart: q.md\n"
    assert "Quickstart" not in nav_order(moved)


def test_home_carries_the_bold_platform_sentence_the_license_and_the_citation_anchor():
    home = expanded(_read("docs/index.md"))
    assert "**Simulations run on Linux only." in home
    assert "Analysis and processing run on any operating system through the conda environment" in " ".join(home.split())
    assert "## License" in home and "AGPL-3.0" not in home.split("## License")[0]
    assert "GNU Affero General Public License" in home.split("## License")[1]
    assert '<a id="how-to-cite"></a>' in home
    assert not _RELEASE_NUMBER.search(home)
    # Control: the pattern reports a release number planted in a table row.
    assert _RELEASE_NUMBER.search("| `10.5281/zenodo.1` | the previous release, version 0.1.0 |")


def test_the_platform_snippet_is_included_by_the_home_page_and_the_installation_guide():
    owned = {
        page: _INCLUDE.findall(_read(page))
        for page in ("docs/index.md", "docs/how-to/installation.md", "docs/quickstart.md")
    }
    assert "platform-support.md" in owned["docs/index.md"]
    assert "platform-support.md" in owned["docs/how-to/installation.md"]
    assert "platform-support.md" not in owned["docs/quickstart.md"]
    for page in ("docs/quickstart.md", "docs/explanation/capabilities.md", "docs/explanation/limitations.md"):
        assert not re.search(r"\bLinux\b", expanded(_read(page))), page
    # Control: expanding a page that includes the snippet puts the word in the page text.
    assert re.search(r"\bLinux\b", expanded('--8<-- "platform-support.md"\n'))


def test_capabilities_has_supported_experiments_with_three_subsections():
    text = _read("docs/explanation/capabilities.md")
    assert h3_titles_under(text, "Supported experiments") == [
        "Ensemble flood studies",
        "Sensitivity analysis",
        "Inter-model comparison",
    ]
    assert removed_headings(text) == []
    assert "instead of running them in sequence" not in text
    # Control: the heading pattern reports the removed headings.
    assert removed_headings("## Analysis kinds\n\n### Release maturity\n") == ["Analysis kinds", "Release maturity"]


@pytest.mark.parametrize("phrase", ["Resuming a quality-routed", "More GPUs can run slower", "Only coupling-candidate"])
def test_limitations_holds_none_of_the_removed_sections_and_no_solver_commit(phrase):
    text = _read("docs/explanation/limitations.md")
    assert phrase.lower() not in text.lower()
    assert removed_headings(text) == [] and not _SOLVER_COMMIT.search(text)
    assert "Only junctions are eligible for coupling." in text
    # Control: the commit pattern reports a hash planted in a sentence.
    assert _SOLVER_COMMIT.search("A snapshot written before bd18215 is refused.")
