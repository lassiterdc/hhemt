"""`write()` must never persist an information-free instance over real state.

The clobber path, measured on the live stochastic arm: `_refresh_log`'s absent-log
branch yields an ALL-DEFAULTS instance; `write()` then re-reads under the lock, that
read ALSO fails on absence, the `except` arm sets `disk = {}`, `overlay` is therefore
empty, and `merged == mine` -- so all-nulls is written over whatever was on disk.

ABSENCE is not CORRUPTION. A parse failure has nothing recoverable and the quarantine
+ proceed behaviour is right for it. An absent file is a race whose alternative is
state destruction, so it must retry and then REFUSE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.exceptions import ProcessingError
from hhemt.log import TRITONSWMM_analysis_log

REAL = {
    "logfile": None,  # filled per-test
    "cpu_backend_available": True,
    "datatree_consolidation_complete": True,
}


def _seed(tmp_path: Path) -> Path:
    p = tmp_path / "log.json"
    payload = dict(REAL, logfile=str(p))
    p.write_text(json.dumps(payload, indent=2))
    return p


def _vanish_on_open(monkeypatch, target: Path, *, times: int) -> dict:
    """Make `target.open()` raise FileNotFoundError the first `times` calls."""
    state = {"n": 0}
    real_open = Path.open

    def fake_open(self, *a, **kw):
        if Path(self) == target and state["n"] < times:
            state["n"] += 1
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_open(self, *a, **kw)

    monkeypatch.setattr(Path, "open", fake_open)
    return state


def test_absent_read_does_not_clobber_real_state(tmp_path: Path, monkeypatch) -> None:
    """THE REGRESSION. Pre-fix this writes all-nulls over a populated log."""
    p = _seed(tmp_path)
    on_disk_before = json.loads(p.read_text())

    # An information-free instance, exactly what _refresh_log's absent branch yields.
    blank = TRITONSWMM_analysis_log(logfile=p)
    _vanish_on_open(monkeypatch, p, times=99)  # absent for every attempt

    # Anchor on a property that exists in BOTH states -- the on-disk CONTENT --
    # not on the new exception type, which cannot exist pre-fix and would make
    # this arm green-by-construction in the wrong direction.
    try:
        blank.write()
    except ProcessingError:
        pass  # post-fix: refusing is the correct terminal

    monkeypatch.undo()
    assert json.loads(p.read_text()) == on_disk_before, (
        "write() destroyed on-disk state when its own re-read failed on absence"
    )


def test_transient_absence_recovers_on_retry(tmp_path: Path, monkeypatch) -> None:
    """One failed open, then present: the read succeeds and the merge protects."""
    p = _seed(tmp_path)
    blank = TRITONSWMM_analysis_log(logfile=p)
    _vanish_on_open(monkeypatch, p, times=1)

    blank.write()  # must NOT raise

    monkeypatch.undo()
    after = json.loads(p.read_text())
    assert after["cpu_backend_available"] is True
    assert after["datatree_consolidation_complete"] is True


def test_parse_failure_still_quarantines_and_proceeds(tmp_path: Path) -> None:
    """UNCHANGED behaviour: a corrupt-but-present log is a diagnostic, not a race."""
    p = tmp_path / "log.json"
    p.write_text("{ this is not json")
    log = TRITONSWMM_analysis_log(logfile=p)

    log.write()  # must NOT raise

    assert (tmp_path / "log.json.unreadable").is_file(), "corrupt bytes must be preserved"
    assert json.loads(p.read_text())["logfile"] == str(p)
