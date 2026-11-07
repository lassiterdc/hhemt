"""The how-to pages keep the template, the repository links, the index and the anchors that other pages use.

Five statements about the pages under `docs/how-to/` have a referent outside the prose, so each is compared
with that referent:

- every how-to page except the index opens with the sections Goal, Prerequisites and Key references, in that
  order, with an exception list that is empty;
- no page sends the reader to copy a file out of the install folder, and each configuration template is
  offered as a link to a path that git tracks;
- every link from a page to the repository names a path that git tracks;
- the section index lists exactly the pages of the section;
- every link into a how-to page that carries a fragment names a heading that the page holds.

No test writes a count: a set is compared with a set. Each check has a control that feeds the check a
mutated text and expects the refusal. No test here compiles or runs a solver.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.docs

_REPO = Path(__file__).resolve().parents[2]
_HOWTO = "docs/how-to/"
_TEMPLATE = ("Goal", "Prerequisites", "Key references")
#: Pages that omit the Prerequisites section because the page has none. Empty: every page has prerequisites.
_NO_PREREQUISITES: frozenset[str] = frozenset()
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_INSTALL_FOLDER = re.compile(r"cp test_data|install (?:folder|path|directory)|site-packages")
_TEMPLATE_FILE = re.compile(r"template_(?:system|analysis)_config\.yaml")
_REPO_LINK = re.compile(r"https://github\.com/lassiterdc/hhemt/(?:blob|tree)/main/([^)\s#]+)")
_LINK = re.compile(r"\]\(([^)\s#]*\.md)#([^)\s]+)\)")


def _tracked() -> list[str]:
    out = subprocess.run(["git", "-C", str(_REPO), "ls-files", "-z"], capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\0") if p]


def _pages(prefix: str = "docs/") -> dict[str, str]:
    return {
        p: (_REPO / p).read_text(encoding="utf-8")
        for p in _tracked()
        if p.startswith(prefix) and p.endswith(".md") and not p.startswith("docs/reference/bundles/")
    }


def _prose_lines(text: str) -> list[str]:
    """The lines of `text` outside fenced code blocks."""
    out, fence = [], None
    for line in text.splitlines():
        m = _FENCE.match(line)
        if m:
            fence = None if fence == m.group(1)[0] else (fence or m.group(1)[0])
            continue
        if fence is None:
            out.append(line)
    return out


def _h2(text: str) -> list[str]:
    return [ln[3:].strip() for ln in _prose_lines(text) if ln.startswith("## ")]


def _howto_pages() -> dict[str, str]:
    return {p: t for p, t in _pages(_HOWTO).items() if p != _HOWTO + "index.md"}


def check_template(page: str, text: str) -> list[str]:
    """Problems: the page's leading H2 sections differ from the template."""
    expected = [s for s in _TEMPLATE if not (s == "Prerequisites" and page in _NO_PREREQUISITES)]
    found = _h2(text)[: len(expected)]
    return [] if found == expected else [f"{page} opens with {found}; the template is {expected}"]


def test_every_how_to_page_opens_with_the_template():
    """Property: each page's first H2 sections are Goal, Prerequisites and Key references, in order.

    Class: every page under `docs/how-to/` except `index.md`. Every how-to page carries the template, and
    the exception list here is empty.

    Why it is required: a page that drops a section leaves a reader without the goal or the links to the
    reference pages, and a template that some pages follow is no template.

    What kills it: a page that renames, drops or reorders one of the three sections.

    A second correct implementation under which it still passes: any prose under the three headings.
    """
    problems = [p for page, text in _howto_pages().items() for p in check_template(page, text)]
    assert problems == []


def test_control_a_page_without_the_goal_section_is_refused():
    page = _HOWTO + "forcing-reruns.md"
    text = _howto_pages()[page]
    assert check_template(page, text) == []
    assert check_template(page, text.replace("## Goal\n", "## Aim\n", 1))
    assert check_template(page, text.replace("## Key references\n", "## See also\n", 1))


def check_install_folder(page: str, text: str) -> list[str]:
    """Problems: a line sends the reader to the install folder, or names a template file other than by link."""
    problems = []
    for n, line in enumerate(text.splitlines(), 1):
        if _INSTALL_FOLDER.search(line):
            problems.append(f"{page}:{n} sends the reader to the install folder: {line.strip()}")
        elif (
            _TEMPLATE_FILE.search(line)
            and "github.com/lassiterdc/hhemt/" not in line
            and "_config_template=" not in line
        ):
            problems.append(f"{page}:{n} names a template file that is not a repository link: {line.strip()}")
    return problems


def test_no_how_to_page_sends_the_reader_to_the_install_folder():
    """Property: no line of a how-to page copies from, or points into, the install folder.

    Class: every line of every page under `docs/how-to/`, code blocks included. A configuration template
    is offered as a link to the file on the repository; the keyword arguments of the Python
    call on the publishing page are the one other spelling.

    Why it is required: the install folder is not a place a reader looks for a file, and a template copied
    from the folder differs from the template on the repository whenever the two versions differ.

    What kills it: a `cp test_data/...` line, or the words install folder, install path, install directory
    or site-packages, on a how-to page.

    A second correct implementation under which it still passes: a dropdown that holds the template text.
    """
    problems = [p for page, text in _howto_pages().items() for p in check_install_folder(page, text)]
    assert problems == []


def test_control_a_copy_from_the_install_folder_is_refused():
    page = _HOWTO + "config-filling.md"
    text = _howto_pages()[page]
    assert check_install_folder(page, text) == []
    assert check_install_folder(
        page, text + "\ncp test_data/norfolk_coastal_flooding/template_system_config.yaml x.yaml\n"
    )
    assert check_install_folder(page, text + "\nCopy the file from the install folder.\n")
    assert check_install_folder(page, text + "\nEdit template_system_config.yaml in place.\n")


def check_repository_links(page: str, text: str, tracked: list[str]) -> list[str]:
    """Problems: a link to the repository names a path that git does not track."""
    problems = []
    for path in _REPO_LINK.findall(text):
        path = path.rstrip("/")
        if path not in tracked and not any(t.startswith(path + "/") for t in tracked):
            problems.append(f"{page} links to {path}, which git does not track")
    return problems


def test_repository_links_name_tracked_paths():
    """Property: every `github.com/lassiterdc/hhemt/blob|tree/main/...` link on a how-to page names a file or a
    directory that git tracks in this checkout.

    Class: the template links of the configuration page and the example-profile and example-config links of the
    profile and synthetic-experiment pages.

    Why it is required: a link to a moved or deleted template answers 404 on the repository.

    What kills it: moving or deleting a linked file without changing the link.

    A second correct implementation under which it still passes: a relative link to the file.
    """
    tracked = _tracked()
    pages = _howto_pages()
    assert any(_REPO_LINK.search(t) for t in pages.values()), "no page links to the repository; the check read nothing"
    problems = [p for page, text in pages.items() for p in check_repository_links(page, text, tracked)]
    assert problems == []


def test_control_a_link_to_a_renamed_template_is_refused():
    page = _HOWTO + "config-filling.md"
    text = _howto_pages()[page]
    tracked = _tracked()
    assert check_repository_links(page, text, tracked) == []
    assert check_repository_links(
        page, text.replace("template_system_config.yaml", "template_system_cfg.yaml"), tracked
    )


def check_index(text: str, pages: set[str]) -> list[str]:
    """Problems: the index lists a page the section lacks, or omits a page the section holds."""
    listed = {m for m in re.findall(r"\]\(([\w-]+\.md)\)", text)}
    return [f"the index omits {n}" for n in sorted(pages - listed)] + [
        f"the index lists {n}, which the section lacks" for n in sorted(listed - pages)
    ]


def _index_inputs() -> tuple[str, set[str]]:
    tracked = [t for t in _tracked() if t.startswith(_HOWTO) and t.endswith(".md")]
    index = (_REPO / _HOWTO / "index.md").read_text(encoding="utf-8")
    return index, {Path(t).name for t in tracked if Path(t).name != "index.md"}


def test_the_index_lists_exactly_the_pages_of_the_section():
    """Property: the set of page links in `docs/how-to/index.md` equals the set of other tracked pages.

    Class: every page of the section, the pages added later included.

    Why it is required: the index is the one list a reader scans for a guide, and an omitted page is a page
    that nobody finds.

    What kills it: a new page without a row, or a row that outlives its page.

    A second correct implementation under which it still passes: the rows in another order.
    """
    index, pages = _index_inputs()
    assert pages, "the section holds no page; the check read nothing"
    assert check_index(index, pages) == []


def test_control_an_omitted_or_unknown_page_is_refused():
    index, pages = _index_inputs()
    first = sorted(pages)[0]
    assert check_index("\n".join(ln for ln in index.splitlines() if f"({first})" not in ln), pages)
    assert check_index(index + "\n| [Ghost](ghost.md) | x |\n", pages)


def _slug(heading: str) -> str:
    """The anchor id the toc extension derives from a heading line's text."""
    from markdown.extensions.toc import slugify

    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    return slugify(text.replace("`", "").replace("*", "").strip(), "-")


def slugs(text: str) -> set[str]:
    """The anchor ids of the headings outside code, with the repeat suffix that the toc extension adds."""
    seen: dict[str, int] = {}
    out: set[str] = set()
    for line in _prose_lines(text):
        m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        base = _slug(m.group(1))
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.add(base if n == 0 else f"{base}_{n}")
    return out


def check_anchors(pages: dict[str, str]) -> list[str]:
    """Problems: a link into a how-to page carries a fragment that no heading of the page yields."""
    problems = []
    for page, text in pages.items():
        for target, fragment in _LINK.findall(text):
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page), target))
            if not resolved.startswith(_HOWTO) or resolved not in pages:
                continue
            if fragment not in slugs(pages[resolved]):
                problems.append(f"{page} links to {resolved}#{fragment}, which no heading yields")
    return problems


def test_fragments_into_how_to_pages_name_headings():
    """Property: each `page.md#fragment` link that targets a how-to page names a heading of that page.

    Class: every link in every tracked page under `docs/` whose target is a page of the section, so a heading
    that changes or disappears cannot leave a dead fragment on another page.

    Why it is required: the strict build reports a dead fragment as a warning only for some link forms, and a
    changed heading breaks every inbound link silently.

    What kills it: changing a heading that another page links to without changing the link.

    A second correct implementation under which it still passes: an explicit anchor id on the heading.
    """
    pages = _pages()
    assert any(f for text in pages.values() for _t, f in _LINK.findall(text)), (
        "no fragment link; the check read nothing"
    )
    assert check_anchors(pages) == []


def test_control_a_link_to_a_removed_heading_is_refused():
    pages = _pages()
    assert check_anchors(pages) == []
    pages = dict(pages)
    compile_page = _HOWTO + "compiling-the-solver.md"
    pages[compile_page] = pages[compile_page].replace("## On a cluster\n", "## On clusters\n", 1)
    assert check_anchors(pages)
