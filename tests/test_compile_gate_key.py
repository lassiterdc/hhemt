"""Tests for `scripts/compile_gate_key.py`, the cache key of the compile gate's pass record.

PROBLEM. The compile gate reuses a pass when the compile-relevant content of a commit equals
that of an earlier full run that passed. The key must therefore move when a file the gate
tests moves and hold still when only prose or release metadata moves. A key that moves too
rarely certifies code that never compiled; a key that moves too often never reuses a pass.
Each test below runs the script against a temporary git repository, so the property is read
from the script's output and no part of the repository under test is needed.

Every test names its class, the change that kills it, and a second correct implementation.
The "unchanged" tests and the "changed" tests hold each other in place: a script that printed
a constant passes the first group and fails the second, and a script that hashed every path
passes the second and fails the first.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "compile_gate_key.py"
WEEK = "2026-W46"

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}

# One path per inert prefix and per inert name, plus a nested path under each prefix.
INERT_PATHS = [
    "docs/index.md",
    "docs/deep/nested/page.md",
    ".vale/styles/Rule.yml",
    "HISTORY.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    ".vale.ini",
    "mkdocs.yml",
    ".readthedocs.yaml",
]

# Every compile-relevant surface the shape names, plus README.md (the editable install reads
# it) and uv.lock (stays whole so a version bump compiles).
LIVE_PATHS = [
    "src/hhemt/module.py",
    "tests/test_something.py",
    "environment.yaml",
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/compile-tests.yml",
    "README.md",
    "conftest.py",
]

# Paths that look inert and are not: a prefix that is not `docs/` or `.vale/`, a root-level
# name that appears below the root, and a name that merely starts like an inert one.
LOOKALIKE_PATHS = [
    "src/docs/page.md",
    "docsx/page.md",
    "docs.md",
    "src/HISTORY.md",
    "tests/CITATION.cff",
    "mkdocs.yml.bak",
    ".vale.ini.bak",
    "vale/styles/Rule.yml",
]


def _git(repo: Path, *args: str) -> str:
    """Run git in `repo` with a fixed identity and no user or system config; return stdout."""
    done = subprocess.run(
        ["git", *args], cwd=repo, env={**os.environ, **_GIT_ENV}, check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def _write(repo: Path, rel: str, text: str) -> None:
    """Write `text` to `rel` under `repo`, creating parent directories."""
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def _repo(tmp_path: Path, name: str = "repo") -> Path:
    """A repository holding one file at every path of the three groups above, committed once."""
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "develop")
    for rel in [*INERT_PATHS, *LIVE_PATHS, *LOOKALIKE_PATHS]:
        _write(repo, rel, f"original {rel}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _commit(repo: Path, message: str = "change") -> None:
    """Stage everything and commit, allowing an empty change set."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)


def _run(repo: Path, week: str = WEEK, runner_os: str = "Linux") -> subprocess.CompletedProcess:
    """Run the script against `repo` the way the workflow does, with the week and OS pinned."""
    env = {**os.environ, "RUNNER_OS": runner_os}
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--repo", str(repo), "--iso-week", week],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _key(repo: Path, **kwargs: str) -> str:
    """The key the script prints for `repo`; fails the test when the script does not exit 0."""
    done = _run(repo, **kwargs)
    assert done.returncode == 0, f"exit {done.returncode}; stderr={done.stderr!r}"
    return done.stdout.strip()


def test_key_has_the_documented_shape_and_is_the_only_stdout_line(tmp_path):
    """Property: stdout is exactly one line `compile-gate-pass-v1-{os}-{ISO year-week}-{64 hex}`.

    Class: any repository the script exits 0 on. The workflow writes stdout into
    `$GITHUB_OUTPUT`, so a second line or a prefix change breaks the cache lookup silently.

    What kills it: a log line on stdout, a shortened digest, or the OS and week swapped.

    A second correct implementation: the digest computed over a different byte layout of the
    same listing still passes, because the test pins the shape and not the digest value.
    """
    done = _run(_repo(tmp_path))
    assert done.returncode == 0
    assert re.fullmatch(r"compile-gate-pass-v1-Linux-2026-W46-[0-9a-f]{64}\n", done.stdout), done.stdout


@pytest.mark.parametrize("rel", INERT_PATHS)
def test_editing_an_inert_path_leaves_the_key_unchanged(tmp_path, rel):
    """Property: changing the bytes of any inert path does not move the key.

    Class: every path under `docs/` or `.vale/` (including a nested one) and each of the six
    root-level inert names. The paired changed-key tests below are the must-fire control: a
    constant key would pass here and fail there.

    Why required: a prose or release-metadata push must reuse the pass record, or the gate
    keeps costing 30 to 50 minutes on commits that moved no compile-relevant byte.

    What kills it: hashing the working tree instead of the inert-filtered listing, or
    dropping one inert name from the set.

    A second correct implementation: filtering inert paths by a path-prefix test with a
    different spelling (`startswith(("docs/", ".vale/"))`) passes the same inputs.
    """
    repo = _repo(tmp_path)
    before = _key(repo)
    _write(repo, rel, f"edited {rel}\n")
    _commit(repo)
    assert _key(repo) == before


def test_adding_and_deleting_inert_files_leaves_the_key_unchanged(tmp_path):
    """Property: a new inert file and a deleted inert file both leave the key as it was.

    Class: a listing that gains or loses entries inside the inert set, as against the edit of
    an existing entry. The two differ because an edit changes a blob id on a line the filter
    drops, while an add or delete changes how many lines the listing has.

    What kills it: filtering after counting entries, or hashing the unfiltered listing's length.

    A second correct implementation: removing inert entries with `grep -v` on the shell form
    of the listing.
    """
    repo = _repo(tmp_path)
    before = _key(repo)
    _write(repo, "docs/new/added.md", "added\n")
    (repo / "HISTORY.md").unlink()
    _commit(repo)
    assert _key(repo) == before


@pytest.mark.parametrize("rel", [*LIVE_PATHS, *LOOKALIKE_PATHS])
def test_editing_a_compile_relevant_or_lookalike_path_changes_the_key(tmp_path, rel):
    """Property: changing the bytes of any path outside the inert set moves the key.

    Class: the surfaces the gate tests (`src/`, `tests/`, `environment.yaml`,
    `pyproject.toml`, the workflow itself), the two that look inert and are not (`README.md`,
    which the editable install reads, and `uv.lock`), and the lookalike paths, which share a
    spelling with an inert one without being it. The lookalikes are the differently
    positioned inputs: a prefix test written as `"docs" in path` passes the plain inert cases
    and fails here.

    Why required: the unlisted direction is the fail-safe one. A path the inert set does not
    name must make the gate run in full.

    What kills it: matching `docs` or `HISTORY.md` anywhere in the path, or ignoring a file
    class by extension.

    A second correct implementation: an explicit allowlist of every live path, hashing only
    those, passes the same inputs for the same reason.
    """
    repo = _repo(tmp_path)
    before = _key(repo)
    _write(repo, rel, f"edited {rel}\n")
    _commit(repo)
    assert _key(repo) != before


def test_a_new_top_level_file_changes_the_key(tmp_path):
    """Property: adding a root-level file no rule names moves the key.

    Class: an unrecognised path. This is the fail-safe direction of the inert set: a file the
    set does not name makes the gate run.

    What kills it: an inert test that treats any root-level file as inert.

    A second correct implementation: an inert test keyed on the file extension of docs pages.
    """
    repo = _repo(tmp_path)
    before = _key(repo)
    _write(repo, "newfile.txt", "new\n")
    _commit(repo)
    assert _key(repo) != before


def test_a_mode_change_and_a_rename_on_a_live_path_change_the_key(tmp_path):
    """Property: the listing's mode and path columns are in the hash, not only the blob ids.

    Class: a change that leaves every blob id as it was. An executable bit on a script and a
    rename both change what the compile tier runs without changing content.

    What kills it: hashing the sorted blob ids alone.

    A second correct implementation: hashing `git ls-files -s` output, which carries the same
    three columns for a clean index.
    """
    repo = _repo(tmp_path)
    original = _key(repo)
    (repo / "src/hhemt/module.py").chmod(0o755)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "mode")
    moded = _key(repo)
    _git(repo, "mv", "src/hhemt/module.py", "src/hhemt/renamed.py")
    _git(repo, "commit", "-q", "-m", "rename")
    renamed = _key(repo)
    assert len({original, moded, renamed}) == 3


def test_the_key_follows_content_and_not_history(tmp_path):
    """Property: two repositories with identical tracked content, built by different commit
    histories, print one key.

    Class: a pair of repositories reaching one tree by different commit sequences. The reuse
    claim is "an identical compile-relevant tree passed", so history must not enter the hash.

    What kills it: hashing the commit id or the commit message.

    A second correct implementation: hashing `git ls-tree` over `HEAD^{tree}`, which is
    independent of the commit object.
    """
    first = _repo(tmp_path, "first")
    second = _repo(tmp_path, "second")
    _write(second, "src/hhemt/module.py", "temporary\n")
    _commit(second, "detour")
    _write(second, "src/hhemt/module.py", "original src/hhemt/module.py\n")
    _commit(second, "back again")
    assert _key(first) == _key(second)


def test_the_week_and_the_runner_os_are_part_of_the_key(tmp_path):
    """Property: one tree prints different keys for a different ISO week and a different OS.

    Class: the two non-content parts of the key. A record must not outlive its calendar week
    (a fresh conda solve can differ on a later day) and must not cross runner operating
    systems.

    What kills it: dropping the week or reading the OS from a constant.

    A second correct implementation: appending the week after the digest instead of before it.
    """
    repo = _repo(tmp_path)
    base = _key(repo)
    assert _key(repo, week="2026-W47") != base
    assert _key(repo, runner_os="macOS") != base
    assert _key(repo, week="2026-W47", runner_os="macOS") not in {base, _key(repo, week="2026-W47")}


def test_the_default_week_is_the_utc_iso_week_that_date_prints(tmp_path):
    """Property: with no `--iso-week`, the key's week equals `date -u +%G-W%V`.

    Class: the production invocation, which passes no week. The year is the ISO week-year
    (`%G`) and not the calendar year (`%Y`), which differ for a few days around 1 January.

    What kills it: `strftime("%Y-W%W")` or a local-time clock.

    A second correct implementation: shelling out to `date -u +%G-W%V` instead of reading the
    clock in Python.
    """
    repo = _repo(tmp_path)
    expected = subprocess.run(["date", "-u", "+%G-W%V"], capture_output=True, text=True, check=True).stdout.strip()
    done = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--repo", str(repo)],
        env={**os.environ, "RUNNER_OS": "Linux"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert f"-Linux-{expected}-" in done.stdout


def test_a_path_with_spaces_and_non_ascii_characters_is_keyed_not_an_error(tmp_path):
    """Property: a tracked path git would quote in its default listing is read and keyed.

    Class: a path holding a space and a non-ASCII character. Git quotes such a path unless
    the listing is NUL-terminated, and a quoted spelling would neither match the inert test
    nor hash like the real name.

    What kills it: parsing the default newline-terminated listing.

    A second correct implementation: `git ls-files -s -z`, which also leaves paths unquoted.
    """
    repo = _repo(tmp_path)
    odd = "src/hhemt/odd name é.py"
    _write(repo, odd, "one\n")
    _commit(repo)
    before = _key(repo)
    _write(repo, odd, "two\n")
    _commit(repo)
    assert _key(repo) != before


def test_empty_repository_and_non_repository_exit_2_with_no_key(tmp_path):
    """Property: with no commit, or outside a repository, the script exits 2 and prints no key.

    Class: a git failure. The shell form of this key printed one constant digest for four
    commits when git refused its pathspec, so a failure that reads as an empty listing
    certifies code that never ran.

    What kills it: treating a non-zero `git ls-tree` as an empty listing and hashing nothing.

    A second correct implementation: checking `git rev-parse HEAD` first and exiting 2 there.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    _git(empty, "init", "-q", "-b", "develop")
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    for target in (empty, not_a_repo):
        done = _run(target)
        assert done.returncode == 2, (target.name, done.returncode, done.stderr)
        assert done.stdout == ""
        assert done.stderr.strip(), "exit 2 owes a reason on stderr"


def test_a_commit_with_an_empty_tree_and_a_commit_of_only_inert_files_exit_2(tmp_path):
    """Property: a listing with nothing to hash exits 2 whether git printed no entries or the
    inert filter removed all of them.

    Class: the two ways the hashed content is empty. Both would otherwise print the digest of
    nothing, one key for every such commit.

    What kills it: hashing whatever remains without checking it is non-empty.

    A second correct implementation: comparing the digest with the sha256 of the empty string.
    """
    no_entries = tmp_path / "no-entries"
    no_entries.mkdir()
    _git(no_entries, "init", "-q", "-b", "develop")
    _git(no_entries, "commit", "-q", "--allow-empty", "-m", "empty tree")
    only_inert = tmp_path / "only-inert"
    only_inert.mkdir()
    _git(only_inert, "init", "-q", "-b", "develop")
    _write(only_inert, "docs/index.md", "x\n")
    _write(only_inert, "HISTORY.md", "x\n")
    _commit(only_inert)
    for target in (no_entries, only_inert):
        done = _run(target)
        assert done.returncode == 2, (target.name, done.returncode, done.stdout, done.stderr)
        assert done.stdout == ""
