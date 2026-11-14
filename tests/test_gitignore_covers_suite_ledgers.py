"""Git ignores every suite-ledger file the `just` test recipes write into the repository root.

Implements approved-shape chunk 1e.1 (item C3). Each recipe sets
`HHEMT_SUITE_LOGREPORT_OUT="${PWD}/NAME"`, so NAME lands in the repo root; the target set is read
from `justfile` rather than listed, because a recipe can be added (shape R2: OPEN domain).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGET_RE = re.compile(r'HHEMT_SUITE_LOGREPORT_OUT="\$\{PWD\}/([^"/]+)"')


def _ledger_targets() -> list[str]:
    return sorted(set(_TARGET_RE.findall((REPO_ROOT / "justfile").read_text())))


def _check_ignore(name: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "check-ignore", "-q", name], cwd=REPO_ROOT, capture_output=True, text=True)


def test_justfile_declares_at_least_one_ledger_target() -> None:
    """Must-fire control: a matcher that finds none would let the test below pass vacuously."""
    assert _ledger_targets(), "no HHEMT_SUITE_LOGREPORT_OUT target found in justfile; the matcher is broken"


@pytest.mark.parametrize("name", _ledger_targets())
def test_gitignore_covers_every_justfile_suite_ledger(name: str) -> None:
    result = _check_ignore(name)
    assert result.returncode == 0, f"{name} is written to the repo root by a just recipe but git does not ignore it"


def test_gitignore_does_not_swallow_the_ledger_schema_test() -> None:
    """Control the other way: the pattern is anchored to the ledgers, not to the word 'ledger'."""
    assert _check_ignore("tests/test_suite_ledger_schema.py").returncode == 1
