"""`write()` is a read-modify-write; it must never persist over state it did not read.

The measured clobber, on the live stochastic ensemble: `_refresh_log` handed
`__init__` an information-free instance, two setters fired, and `write()`'s own
re-read of the master log failed. The old code answered that failure with
`disk = {}` -- so `overlay` was empty, `merged == mine`, and an all-defaults
document was persisted over a populated one. It PARSES, so every later reader
propagated the nulls faithfully. The loss sustains itself through the healthy path.

"I could not read it" is not "there is nothing there". Absence is settled by an
EXCLUSIVE create, which asks the filesystem; it is never inferred from a failed read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt.exceptions import ProcessingError
from hhemt.log import TRITONSWMM_analysis_log


def _seed(tmp_path: Path) -> Path:
    p = tmp_path / "log.json"
    p.write_text(
        json.dumps(
            {
                "logfile": str(p),
                "cpu_backend_available": True,
                "datatree_consolidation_complete": True,
            },
            indent=2,
        )
    )
    return p


def _reads_fail(monkeypatch, target: Path) -> None:
    """Make every READ of `target` raise FileNotFoundError, leaving writes alone.

    Both the pre-fix implementation (`Path.exists` + `Path.open`) and the post-fix
    one (`Path.read_bytes`) are covered, so the arm discriminates on BEHAVIOUR
    rather than on which syscall the implementation happens to use.
    """
    real_open = Path.open
    real_read_bytes = Path.read_bytes

    def fake_open(self, *a, **kw):
        if Path(self) == target:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_open(self, *a, **kw)

    def fake_read_bytes(self):
        if Path(self) == target:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "open", fake_open)
    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)


def test_unreadable_absence_does_not_clobber_on_disk_state(tmp_path, monkeypatch):
    """THE REGRESSION. Pre-fix this persists all-nulls over a populated log."""
    p = _seed(tmp_path)
    before = json.loads(p.read_text())

    blank = TRITONSWMM_analysis_log(logfile=p)  # what _refresh_log's degraded branch yields
    _reads_fail(monkeypatch, p)
    try:
        blank.write()
    except ProcessingError:
        pass  # post-fix terminal: refusing is correct
    monkeypatch.undo()

    assert json.loads(p.read_text()) == before, (
        "write() persisted this instance's unchanged fields over on-disk state it never read"
    )


def test_unparseable_log_refuses_and_preserves_the_bytes_that_failed(tmp_path):
    """A corrupt-but-present log is unknown state, not empty state."""
    p = tmp_path / "log.json"
    p.write_bytes(b"{ this is not json")
    log = TRITONSWMM_analysis_log(logfile=p)

    with pytest.raises(ProcessingError):
        log.write()

    assert p.read_bytes() == b"{ this is not json", "the corrupt log was overwritten"
    assert (tmp_path / "log.json.unreadable").read_bytes() == b"{ this is not json", (
        "the quarantine must hold the bytes that actually failed to parse"
    )


def test_genuine_first_create_still_works(tmp_path):
    """REGRESSION GUARD (green in both states): a truly absent log is created."""
    p = tmp_path / "log.json"
    log = TRITONSWMM_analysis_log(logfile=p)
    log.write()
    assert json.loads(p.read_text())["logfile"] == str(p)


def test_established_read_still_merges_concurrent_fields(tmp_path):
    """REGRESSION GUARD (green in both states): the lost-update overlay is untouched."""
    p = _seed(tmp_path)
    blank = TRITONSWMM_analysis_log(logfile=p)
    blank.gpu_backend_available.set(True)
    after = json.loads(p.read_text())
    assert after["datatree_consolidation_complete"] is True
    assert after["cpu_backend_available"] is True
    assert after["gpu_backend_available"] is True


def test_unchanged_backend_flags_author_no_write(synth_multi_sim_analysis_cached):
    """The constructor must not rewrite the SHARED master log to restate what it says.

    `LogField.set()` is a full read-modify-write of the shared analysis log, so an
    unconditional set at construction made every `skip_log_update=False` consumer two
    writers of one file. On an ensemble that is thousands of writers churning two
    immutable booleans -- the population the measured clobber came from.
    """
    from hhemt.analysis import TRITONSWMM_analysis

    a = synth_multi_sim_analysis_cached
    a._update_log()
    logfile = a.log.logfile
    before = (logfile.stat().st_mtime_ns, logfile.read_bytes())

    TRITONSWMM_analysis(
        a.analysis_config_yaml,
        a._system,
        skip_log_update=False,
        is_main_orchestrator=False,
    )

    after = (logfile.stat().st_mtime_ns, logfile.read_bytes())
    assert before == after, "construction rewrote the master log with values it already held"
