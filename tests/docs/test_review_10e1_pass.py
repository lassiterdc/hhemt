"""No published page tells the reader to copy a file out of the install folder.

The developer's finding on the configuration-filling page, in his words, is that "no instruction anywhere
should recommend copying a template out of the install folder". The coder's test of the how-to chunk holds the
property over `docs/how-to/` only. This test holds it over every tracked Markdown page of the site and
the root README, so a later chunk that edits a tutorial or an explanation page cannot reintroduce the
instruction.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.docs

_REPO = Path(__file__).resolve().parents[2]
_INSTRUCTION = re.compile(r"\bcp\s+(?:-\w+\s+)*\S*test_data/|(?:install|installation)\s+(?:folder|path|directory)")


def _published_pages() -> dict[str, str]:
    out = subprocess.run(["git", "-C", str(_REPO), "ls-files", "-z"], capture_output=True, text=True, check=True)
    paths = [
        p
        for p in out.stdout.split("\0")
        if p
        and p.endswith(".md")
        and (p.startswith("docs/") or p == "README.md")
        and not p.startswith("docs/reference/bundles/")
    ]
    return {p: (_REPO / p).read_text(encoding="utf-8") for p in paths}


def lines_that_send_the_reader_to_the_install_folder(page: str, text: str) -> list[str]:
    """Lines of `text`, code blocks included, that copy from `test_data/` or name the install folder."""
    return [f"{page}:{n}: {line.strip()}" for n, line in enumerate(text.splitlines(), 1) if _INSTRUCTION.search(line)]


def test_no_published_page_sends_the_reader_to_the_install_folder():
    """Property: no line of any published page is a `cp test_data/...` command or names the install folder.

    Class: every tracked `.md` file under `docs/` (generated bundles excluded) and `README.md`, every line,
    code blocks included. A page that offers a configuration template links the file on the repository.

    NEW CAPABILITY: the plausible wrong implementation is a tutorial page that keeps a
    `cp test_data/norfolk_coastal_flooding/template_system_config.yaml my_system.yaml` line, which a check
    scoped to `docs/how-to/` passes; this test is red under it (the control below).

    Reachability: a `cp` line in a fenced block of `docs/tutorials/*.md` is a line this test reads, and
    the two earlier forms of the instruction sat in `docs/how-to/config-filling.md` lines 15 and 16 at the
    commit before the chunk.

    Both-states anchor: the assertion turns on whether the line matches the instruction pattern, a property
    of the line in the current and the proposed text alike, and on no wording the chunk introduces.

    Why it is required: a template copied from the install folder differs from the template on the
    repository whenever the two versions differ, and the install folder is not a place a reader looks.

    What kills it: adding a `cp test_data/...` line, or the words install folder, installation path or
    install directory, to any published page.

    A second correct implementation under which it still passes: a dropdown that holds the template text, or
    a link to the template file on the repository.
    """
    problems = [
        p
        for page, text in _published_pages().items()
        for p in lines_that_send_the_reader_to_the_install_folder(page, text)
    ]
    assert problems == []


def test_control_the_pattern_reads_a_tutorial_line_and_refuses_the_copy_instruction():
    """Control: a mutated tutorial text is refused and the unmutated text is not."""
    clean = "Save each template as a file of your own, such as `my_system.yaml`.\n"
    planted = (
        clean + "```bash\ncp test_data/norfolk_coastal_flooding/template_system_config.yaml   my_system.yaml\n```\n"
    )
    assert lines_that_send_the_reader_to_the_install_folder("docs/tutorials/x.md", clean) == []
    assert len(lines_that_send_the_reader_to_the_install_folder("docs/tutorials/x.md", planted)) == 1
    named = "Copy the template from the install folder.\n"
    assert len(lines_that_send_the_reader_to_the_install_folder("docs/tutorials/x.md", named)) == 1
    assert _published_pages(), "the page set must not be empty"
