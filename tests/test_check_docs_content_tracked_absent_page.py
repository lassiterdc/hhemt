"""The docs gate's markdown population on a tree where a tracked page was deleted.

Deleting a docs page is a two-step change: the file leaves the working tree, then
the deletion is staged and committed. Until it is staged, `git ls-files` still
lists the path, because git's index records it. The docs gate's
population (`_scanned_markdown`) unites that tracked list with the markdown on
disk under the docs directory, and its three marker consumers -- the functions
that count generated, personal-voice and repo-internal pages -- read every
member's text. A member with no file behind it therefore raised
`FileNotFoundError` out of every consumer, and `mkdocs build` reaches all three
through `hooks/marker_governance_table.py`, so the build crashed on a tree whose
only defect was an unstaged deletion.

These tests build that tree in a throwaway git repository and check the
population and each consumer separately, so one failure cannot stop the next
from running. A control repository, identical except that the page is still on
disk, checks that the population and each consumer list a present marker page
under the docs directory. That control cannot see the tracked half on its own:
the on-disk half re-supplies any page under the docs directory, so a tracked half
that is dropped or filtered with its sense inverted still passes it. The page
outside the docs directory, deleted and as a control, is what constrains the
tracked half.

No test here runs a solver or touches the project's own docs tree.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "check_docs_content.py"


def _load():
    """The gate module, loaded by path because `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("check_docs_content", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_docs_content"] = mod
    spec.loader.exec_module(mod)
    return mod


cdc = _load()

#: The page the tests delete. It carries all three marker kinds, one per line,
#: so that every consumer lists it while it exists. A page carrying no marker
#: would be omitted by the consumers whether or not the fix were present, and the
#: consumer tests would then be unable to tell a population that skips the
#: deleted path from one that never offered it.
_PAGE_NAME = "retired.md"
_PAGE_TEXT = (
    f"<!-- {cdc.GENERATED_MARKER} -->\n"
    f"<!-- {cdc.PERSONAL_VOICE_MARKER} -->\n"
    f"<!-- {cdc.REPO_INTERNAL_MARKER} -->\n"
    "# Retired\n"
)

#: An untracked page under the docs directory, standing in for the generated
#: build output the population's on-disk half exists to reach. It is present in
#: both repositories and must stay in the population in both.
_UNTRACKED_NAME = "build_output.md"

#: The three marker consumers, each a separate test item.
_CONSUMERS = ("generated_files", "personal_voice_files", "repo_internal_files")


def _git(repo: Path, *args: str) -> None:
    """Run git in `repo` with an identity and no hooks or signing.

    The identity and the two disables are passed per command rather than read
    from the caller's configuration, because a commit in a throwaway repository
    otherwise fails on a machine with no global identity, or runs a global hook.
    """
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _repo(root: Path, *, delete_page: bool, page_dir: str = "docs") -> Path:
    """A git repository with the marker page committed, returning its docs dir.

    With `delete_page`, the page is then removed from the working tree with a
    plain filesystem delete and NOT staged, so the index still lists it -- the
    state a builder is in between deleting a page and committing the deletion.

    `page_dir` is the repository-relative directory the page is committed in.
    It defaults to the docs directory; a directory outside it puts the page
    where only the population's tracked half can reach it, because the on-disk
    half walks the docs directory alone.
    """
    root.mkdir()
    _git(root, "init", "-q")
    docs = root / "docs"
    docs.mkdir()
    page_parent = root / page_dir
    page_parent.mkdir(parents=True, exist_ok=True)
    (page_parent / _PAGE_NAME).write_text(_PAGE_TEXT, encoding="utf-8")
    _git(root, "add", page_dir)
    _git(root, "commit", "-q", "-m", "add page")
    (docs / _UNTRACKED_NAME).write_text("# Build output\n", encoding="utf-8")
    if delete_page:
        (page_parent / _PAGE_NAME).unlink()
    return docs


@pytest.fixture
def deleted_docs(tmp_path: Path) -> Path:
    """The docs dir of a repository whose tracked marker page was deleted, uncommitted."""
    return _repo(tmp_path / "deleted", delete_page=True)


@pytest.fixture
def present_docs(tmp_path: Path) -> Path:
    """The docs dir of the same repository with the marker page still on disk."""
    return _repo(tmp_path / "present", delete_page=False)


def _names(paths: list[Path]) -> set[str]:
    return {p.name for p in paths}


def test_the_population_omits_a_tracked_page_absent_from_disk(deleted_docs: Path) -> None:
    """Property: a tracked path with no file in the working tree is not a member.

    Class: every tracked markdown path whose file is absent -- deleted and not
    yet committed, or otherwise missing -- reached here by the commonest member,
    a plain filesystem delete of a committed page.

    Why required: every consumer of the population reads each member's text, so
    an absent member raises in all of them, and the docs build reaches three of
    them. A page absent from disk ships nothing, so omitting it narrows nothing
    that ships.

    What kills it: uniting the tracked list into the population with no
    existence check. That returns normally and lists the deleted page, which is
    why this is an assertion on membership rather than on a raise.

    A second correct implementation under which it still passes: subtracting the
    output of `git ls-files --deleted` from the tracked list instead of testing
    each path on disk. The untracked-page assertion holds under both, and kills
    an implementation that drops the on-disk half to get rid of the absent path.
    """
    population = _names(cdc._scanned_markdown(deleted_docs))
    assert _PAGE_NAME not in population, f"{_PAGE_NAME} is absent from disk but listed: {sorted(population)}"
    assert _UNTRACKED_NAME in population, sorted(population)


@pytest.mark.parametrize("consumer", _CONSUMERS)
def test_each_marker_consumer_omits_a_tracked_page_absent_from_disk(deleted_docs: Path, consumer: str) -> None:
    """Property: a marker consumer returns normally and does not list an absent page.

    Class: each of the three marker consumers, over a population containing a
    tracked path whose file is absent; the page carries every marker, so each
    consumer would list it if it were offered and could be read.

    Why required: `hooks/marker_governance_table.py` calls all three during the
    docs build, so a raise here is a failed `mkdocs build` on a tree whose only
    defect is an uncommitted deletion.

    What kills it: a population that still offers the absent path, which makes
    the consumer's `read_text` raise `FileNotFoundError`.

    A second correct implementation under which it still passes: a consumer that
    skips members it cannot read. The assertion is on the consumer's result,
    not on where the absent path was dropped.
    """
    listed = _names(getattr(cdc, consumer)(deleted_docs))
    assert _PAGE_NAME not in listed, f"{consumer} listed the absent {_PAGE_NAME}: {sorted(listed)}"


@pytest.mark.parametrize("finder", ("_scanned_markdown", *_CONSUMERS))
def test_control_the_page_is_listed_while_it_is_on_disk(present_docs: Path, finder: str) -> None:
    """Property: the same page, still on disk, is listed by the population and every consumer.

    Class: the satisfying position the deleted-page tests do not occupy -- a
    tracked page present in the working tree.

    Why required: the three consumer deleted-page tests assert only an
    absence, so a consumer that listed nothing would pass its own, and every
    consumer would pass its own over a population that listed nothing.

    What kills it: a consumer that returns an empty list (its own item reds),
    or a population that returns an empty list (all four red). What it does
    NOT kill: a tracked half dropped entirely, or an existence filter with its
    sense inverted -- this page sits under the docs directory, so the on-disk
    half lists it whatever the tracked half does. The outside-the-docs-dir
    control below is the one that reds under those two.

    A second correct implementation under which it still passes: the
    `git ls-files --deleted` subtraction, which leaves a present page in place.
    """
    listed = _names(getattr(cdc, finder)(present_docs))
    assert _PAGE_NAME in listed, f"{finder} did not list the present {_PAGE_NAME}: {sorted(listed)}"


#: A repository directory outside the docs directory. The population's tracked
#: half spans the whole repository while its on-disk half walks the docs
#: directory alone, so a page committed here is reachable ONLY through the
#: tracked half -- which is the half the existence check lives in. A page under
#: the docs directory is re-supplied by the on-disk walk whenever it exists, so
#: no outcome observed on it can show what the tracked half kept or dropped.
_OUTSIDE_DIR = "notes"

_FINDERS = ("_scanned_markdown", *_CONSUMERS)


@pytest.fixture
def outside_deleted_docs(tmp_path: Path) -> Path:
    """The docs dir of a repository whose tracked marker page OUTSIDE it was deleted, uncommitted."""
    return _repo(tmp_path / "outside_deleted", delete_page=True, page_dir=_OUTSIDE_DIR)


@pytest.fixture
def outside_present_docs(tmp_path: Path) -> Path:
    """The docs dir of the same repository with the marker page outside it still on disk."""
    return _repo(tmp_path / "outside_present", delete_page=False, page_dir=_OUTSIDE_DIR)


@pytest.mark.parametrize("finder", _FINDERS)
def test_a_tracked_page_outside_the_docs_dir_absent_from_disk_is_omitted(
    outside_deleted_docs: Path, finder: str
) -> None:
    """Property: an absent tracked page is omitted wherever in the repository it was tracked.

    Class: every tracked markdown path whose file is absent and which lies
    outside the docs directory -- the repository-internal notes, fixture READMEs
    and shipped metadata the population carries beyond `docs/`. Reached here by
    a plain filesystem delete of a committed page under `notes/`.

    Why required: the docs build calls the three marker consumers through
    `hooks/marker_governance_table.py`, and the population they read spans the
    whole repository, so a deleted page outside `docs/` crashes the build exactly
    as one inside it does. An existence check confined to the docs directory --
    or a population that drops tracked paths under `docs/` and leans on the
    on-disk walk for them -- passes every test positioned inside `docs/` and
    still lists this page.

    What kills it: limiting the tracked half's existence check to paths under
    the docs directory; or removing the check, which lists the page and makes
    each consumer's `read_text` raise `FileNotFoundError`.

    A second correct implementation under which it still passes: subtracting
    `git ls-files --deleted` from the tracked list, which is repository-wide by
    construction; for the consumer items, a consumer that skips members it
    cannot read.
    """
    listed = _names(getattr(cdc, finder)(outside_deleted_docs))
    assert _PAGE_NAME not in listed, f"{finder} listed the absent {_OUTSIDE_DIR}/{_PAGE_NAME}: {sorted(listed)}"


@pytest.mark.parametrize("finder", _FINDERS)
def test_control_a_tracked_page_outside_the_docs_dir_is_listed_while_on_disk(
    outside_present_docs: Path, finder: str
) -> None:
    """Property: a present tracked page outside the docs directory is kept by the tracked half.

    Class: the satisfying position no docs-positioned control can occupy -- a
    tracked page present in the working tree that the on-disk walk does not
    reach, so the population lists it only if the tracked half kept it.

    Why required: the docs-positioned control stays green under a population
    that drops every tracked page, and under an existence check with its sense
    inverted, because the on-disk walk re-supplies the page it checks. This one
    is the control that reaches the tracked half's keep branch.

    What kills it: an existence check with its sense inverted; emptying the
    tracked half; or resolving each tracked path against the process working
    directory instead of the repository root, which drops it whenever the gate
    runs from anywhere but the root (pytest here runs from the project root, not
    from this temporary repository).

    A second correct implementation under which it still passes: the
    `git ls-files --deleted` subtraction, which leaves a present page in place.
    """
    listed = _names(getattr(cdc, finder)(outside_present_docs))
    assert _PAGE_NAME in listed, f"{finder} did not list the present {_OUTSIDE_DIR}/{_PAGE_NAME}: {sorted(listed)}"
