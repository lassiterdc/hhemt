"""Reviewer pin for the 1i.1 repair: the Vale vocabulary accepts PMIx and the how-to page is spelling-clean.

Invariant: every word the published how-to page uses that is outside the en_US dictionary is listed once, in
byte order, in the one accept file the Spelling rule reads. The two arms below vary the property (word present
versus absent in a copy of the vocabulary), not the token.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ACCEPT = ROOT / ".vale" / "styles" / "HhemtQuality" / "accept.txt"
PAGE = "docs/how-to/diagnosing-a-failed-run.md"


def _entries() -> list[str]:
    return ACCEPT.read_text(encoding="utf-8").splitlines()


def test_pmix_is_accepted_exactly_once() -> None:
    assert _entries().count("PMIx") == 1


def test_accept_file_stays_in_byte_order_without_duplicates() -> None:
    entries = _entries()
    assert entries == sorted(entries)
    assert len(entries) == len(set(entries))


def _spelling_alerts(styles: Path, tmp_path: Path) -> list[dict]:
    ini = tmp_path / "vale.ini"
    base = (ROOT / ".vale.ini").read_text(encoding="utf-8").splitlines()
    ini.write_text(
        "\n".join(f"StylesPath = {styles}" if line.startswith("StylesPath") else line for line in base),
        encoding="utf-8",
    )
    run = subprocess.run(
        [shutil.which("vale"), f"--config={ini}", "--output=JSON", PAGE],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert run.returncode in (0, 1), run.stderr
    found = json.loads(run.stdout or "{}").get(PAGE, [])
    return [a for a in found if a["Check"] == "HhemtQuality.Spelling" and a["Match"] == "PMIx"]


@pytest.mark.skipif(shutil.which("vale") is None, reason="vale is not on PATH")
def test_page_has_no_pmix_spelling_alert_and_the_alert_fires_without_the_entry(tmp_path: Path) -> None:
    assert _spelling_alerts(ROOT / ".vale" / "styles", tmp_path) == []
    ctl = tmp_path / "styles"
    shutil.copytree(ROOT / ".vale" / "styles", ctl)
    accept = ctl / "HhemtQuality" / "accept.txt"
    accept.write_text("\n".join(e for e in _entries() if e != "PMIx") + "\n", encoding="utf-8")
    assert len(_spelling_alerts(ctl, tmp_path)) >= 1
