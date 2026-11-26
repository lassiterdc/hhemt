"""Reviewer test: every page the documentation pass edited passes `scripts/check_vale.py`."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The pages commit 6fa45b80 edited.
EDITED_PAGES = (
    "docs/explanation/branching-and-release-model.md",
    "docs/explanation/limitations.md",
    "docs/explanation/testing-the-toolkit.md",
    "docs/how-to/diagnosing-a-failed-run.md",
    "docs/how-to/per-node-manhole-values.md",
    "docs/how-to/running-an-experiment-definition.md",
    "docs/reference/cli.md",
    "docs/reference/output-data-model.md",
)


@pytest.fixture(scope="module")
def vale_alerts() -> dict[str, list[dict]]:
    vale = shutil.which("vale")
    if vale is None:
        pytest.skip("`vale` is not on PATH")
    run = subprocess.run(
        [vale, "--config=.vale.ini", "--output=JSON", *EDITED_PAGES],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode in (0, 1), run.stderr or run.stdout
    return json.loads(run.stdout or "{}")


@pytest.mark.parametrize("page", EDITED_PAGES)
def test_edited_page_exists(page: str) -> None:
    assert (REPO_ROOT / page).is_file()


@pytest.mark.parametrize("page", EDITED_PAGES)
def test_edited_page_has_no_vale_alert(page: str, vale_alerts: dict[str, list[dict]]) -> None:
    found = vale_alerts.get(page, [])
    assert not found, "\n".join(f"{page}:{a['Line']}: {a['Check']}: {a['Message']}" for a in found)


def test_check_vale_script_exits_zero() -> None:
    if shutil.which("vale") is None:
        pytest.skip("`vale` is not on PATH")
    run = subprocess.run(
        [sys.executable, "scripts/check_vale.py"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stdout + run.stderr
