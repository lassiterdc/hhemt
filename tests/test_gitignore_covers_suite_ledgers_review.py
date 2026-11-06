"""Reviewer arms for chunk 1e.1 (`.gitignore` entry for the suite ledgers).

The coder's guard test checks that the four names the `justfile` writes today are ignored and that
one schema test is not. These arms grade the BOUNDARY of the pattern: what it ignores beyond
today's four names, and what it must leave alone. They run `git check-ignore` in a throwaway
repository seeded with this tree's own `.gitignore`, so no verdict depends on the checkout's state.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

GITIGNORE = Path(__file__).resolve().parent.parent / ".gitignore"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(GITIGNORE.read_text())
    return tmp_path


def _ignored(repo: Path, name: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", "--", name], cwd=repo).returncode == 0


@pytest.mark.parametrize(
    "name",
    ["suite_ledger_fast.jsonl", "suite_ledger_a_recipe_added_later.jsonl", "suite_ledger_x.y.jsonl"],
)
def test_root_level_suite_ledgers_are_ignored_including_names_no_recipe_uses_yet(repo: Path, name: str) -> None:
    """Class (i): every repo-root file named `suite_ledger_{anything}.jsonl`, including names a future
    recipe may choose. (iii): the justfile target set is open (shape R2), so the pattern must be a
    glob and not a list of today's four names. (iv) mutation: `/suite_ledger_fast.jsonl` alone reds the
    later-recipe names; `/suite_ledger_*.json` reds all three. Second correct implementation:
    `/suite_ledger_*` would also pass here, and is bounded by the next test."""
    assert _ignored(repo, name)


@pytest.mark.parametrize(
    "name",
    [
        "tests/suite_ledger_fast.jsonl",
        "docs/suite_ledger_fast.jsonl",
        "suite_ledger_fast.json",
        "suite_ledger_fast.jsonl.bak",
        "suite_ledger.jsonl",
        "my_suite_ledger_fast.jsonl",
        "tests/test_suite_ledger_schema.py",
    ],
)
def test_pattern_ignores_only_root_level_suite_ledger_jsonl(repo: Path, name: str) -> None:
    """Class (i): near-miss names - nested copies, other extensions, other stems. (iii): the recipes
    write only to `${PWD}`, so only the root needs ignoring; a nested or differently named file is
    real work that must stay visible to `git status`. (iv) mutation: an unanchored
    `suite_ledger_*.jsonl` reds the two nested names; `/suite_ledger*` reds `suite_ledger.jsonl`,
    `.json` and `.bak`. Second correct implementation: `/suite_ledger_*.jsonl` moved to
    any other line of the file, or written as `/suite_ledger_[a-z]*.jsonl`; both still pass."""
    assert not _ignored(repo, name)


def _tracked_files_matched_by_the_ledger_pattern(root: Path) -> list[str]:
    """Tracked files whose ignore verdict (`--no-index`, so tracking does not mask it) comes from a
    `suite_ledger` pattern. Scoped to that pattern because the repository already tracks one file
    that an older pattern ignores (`test_data/TRITON_SWMM_test_model/swmm/hydraulics.inp`)."""
    tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
    verdict = subprocess.run(
        ["git", "check-ignore", "--no-index", "-v", "--stdin"],
        cwd=root,
        input=tracked,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in verdict.splitlines() if "suite_ledger" in line.split("\t")[0]]


def test_the_ledger_pattern_instrument_fires_on_a_tracked_ledger(repo: Path) -> None:
    """Must-fire control for the next test: a force-added `suite_ledger_x.jsonl` IS reported."""
    (repo / "suite_ledger_x.jsonl").write_text("{}\n")
    subprocess.run(["git", "add", "-f", "suite_ledger_x.jsonl"], cwd=repo, check=True)
    assert _tracked_files_matched_by_the_ledger_pattern(repo)


def test_no_tracked_file_is_swallowed_by_the_ledger_pattern() -> None:
    """Class (i): every file the real repository tracks. (iii): an ignore pattern that matches a
    tracked file silently stops showing new files of that name. (iv) mutation: a tracked root file
    named `suite_ledger_*.jsonl` (the control above builds one). Second correct implementation:
    ignoring via `.git/info/exclude`-style per-clone config would also pass, as no `.gitignore`
    pattern exists to match."""
    root = GITIGNORE.parent
    assert _tracked_files_matched_by_the_ledger_pattern(root) == []
