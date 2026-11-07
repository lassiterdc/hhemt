"""Every test under `tests/docs/` carries the `docs` marker.

`tests/docs/conftest.py` adds the marker by a collection hook. This test checks the
result from outside: a collection run that deselects `docs` must find nothing under
this directory. A hook that stops marking, or marks only some files, makes the
deselected run collect tests and fails this test.

The check runs in a subprocess because the marker filter applies to the session
that is already running; only a fresh session shows what a user selecting
`-m "not docs"` would get.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DOCS_TEST_DIR = Path(__file__).resolve().parent


def collect_unmarked_docs_tests(docs_test_dir: Path, repo: Path) -> subprocess.CompletedProcess:
    """Collect `docs_test_dir` with `-m "not docs"` in a fresh pytest session.

    A caller reads the return code and the output. Return code 5 and a summary of
    `no tests collected` or `deselected` mean the directory holds no unmarked test.
    The repo-root `conftest.py` loads from `repo`, so `cwd` is the repository root.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "not docs", str(docs_test_dir)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def collected_test_ids(stdout: str) -> list[str]:
    """Test ids in `pytest --collect-only -q` output: the lines that name `path::test`."""
    return [line for line in stdout.splitlines() if "::" in line]


def test_not_docs_collects_nothing_under_tests_docs():
    result = collect_unmarked_docs_tests(_DOCS_TEST_DIR, _REPO)
    assert collected_test_ids(result.stdout) == [], (
        "a test under tests/docs/ lacks the `docs` marker; output:\n" + result.stdout + result.stderr
    )
    # An error in the child session (an import failure, a bad option) also collects
    # nothing, so a run that did not reach collection must not pass as an empty one.
    assert result.returncode in (0, 5), result.stdout + result.stderr
