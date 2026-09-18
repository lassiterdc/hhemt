"""Fixture-free differentials for the log-write hardening (arm A, [Q404]).

T1 (M1) and T2 (M3/M3b) live in test_log_write_refuses_over_unread_state.py, rewritten
from the two oracles they reverse. Here: T3 the empty-file state M1 introduces, T4/T5 the
M8 detector and its always-mine stamp, T6 the M2 atomic exclusive create, T7 the M4
deferral. Every test names the mutation that reddens it. No fixture, no solver, no compile.
"""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path

import pytest

import hhemt.log as hlog
from hhemt.log import TRITONSWMM_analysis_log
from hhemt.utils import write_json_exclusive


def _seed(tmp_path: Path, **extra) -> Path:
    p = tmp_path / "log.json"
    doc = {"logfile": str(p), "cpu_backend_available": True, "datatree_consolidation_complete": True}
    doc.update(extra)
    p.write_text(json.dumps(doc, indent=2))
    return p


def _foreign_stamp(hostname: str = "node-B", restart_count: int = 0) -> dict:
    return {"slurm_jobid": "1", "hostname": hostname, "restart_count": restart_count, "pid": 1, "written_at": "x"}


def test_empty_document_is_absent_not_corrupt(tmp_path):
    """T3 (M1/M3 boundary). A 0-byte log is 'absent, mine alone': written from this
    instance, NOT quarantined, no recovery record. Mutation: route `len(_raw) == 0` into
    the unparseable arm (json.loads(b"") raises) and a `.unreadable` sibling appears."""
    p = tmp_path / "log.json"
    p.write_bytes(b"")
    log = TRITONSWMM_analysis_log(logfile=p)
    log.gpu_backend_available.set(True)
    assert not (tmp_path / "log.json.unreadable").exists()
    after = json.loads(p.read_text())
    assert after["gpu_backend_available"] is True
    assert after["recovered_from_unparseable"] is None

    # Read side (SPEC 4): an empty document loads as defaults with NO record.
    e = tmp_path / "empty.json"
    e.write_bytes(b"")
    assert TRITONSWMM_analysis_log.from_json(e).recovered_from_unparseable is None


def test_foreign_write_between_load_and_first_write_is_detected(tmp_path, caplog):
    """T4 (M8). A stamp on disk that differs from the LOADED stamp in hostname WARNs at the
    very first write; the same document with no foreign write does not; and a baseline
    with NO stamp is not exempt. Mutation: delete the `_disk_key != _base_key` comparison
    and the first WARN disappears; restore `isinstance(_base_stamp, dict)` as a guard and
    the None-baseline WARN disappears."""
    p = _seed(tmp_path, last_writer=_foreign_stamp("node-A"))
    log = TRITONSWMM_analysis_log.from_json(p)  # baseline stamp: node-A
    doc = json.loads(p.read_text())
    doc["last_writer"] = _foreign_stamp("node-B")
    p.write_text(json.dumps(doc, indent=2))  # a second instance wrote in between

    with caplog.at_level(logging.WARNING, logger="hhemt.log"):
        log.gpu_backend_available.set(True)
    hits = [r for r in caplog.records if "written by ANOTHER process" in r.getMessage()]
    assert hits, "the M8 detector did not fire on a foreign stamp"
    assert "node-B" in hits[0].getMessage() and "node-A" in hits[0].getMessage()
    assert json.loads(p.read_text())["last_writer"]["hostname"] == socket.gethostname()

    # Negative arm: no foreign write -> no detection.
    caplog.clear()
    (tmp_path / "quiet").mkdir()
    q = _seed(tmp_path / "quiet", last_writer=_foreign_stamp("node-A"))
    quiet = TRITONSWMM_analysis_log.from_json(q)
    with caplog.at_level(logging.WARNING, logger="hhemt.log"):
        quiet.gpu_backend_available.set(True)
    assert not [r for r in caplog.records if "written by ANOTHER process" in r.getMessage()]

    # None-baseline arm (no exemption): this process synced with NO stamp (a stampless
    # document), a peer then wrote one -> detected at the first write.
    caplog.clear()
    (tmp_path / "legacy").mkdir()
    r = _seed(tmp_path / "legacy")  # no last_writer key: a pre-M8 document
    legacy = TRITONSWMM_analysis_log.from_json(r)
    doc = json.loads(r.read_text())
    doc["last_writer"] = _foreign_stamp("node-C")
    r.write_text(json.dumps(doc, indent=2))
    with caplog.at_level(logging.WARNING, logger="hhemt.log"):
        legacy.gpu_backend_available.set(True)
    assert [x for x in caplog.records if "written by ANOTHER process" in x.getMessage() and "node-C" in x.getMessage()]


def test_last_writer_is_always_mine_even_when_the_stamp_is_unchanged(tmp_path, monkeypatch):
    """T5 (M8 always-mine). With a CONSTANT stamp (two writes inside one second) the
    changed-keys test reads last_writer as unchanged; the overlay must still not persist a
    foreign stamp as ours. Mutation: drop `and k != "last_writer"` from the overlay and
    the second write persists node-B."""
    mine = {"slurm_jobid": None, "hostname": "me", "restart_count": 0, "pid": 7, "written_at": "t"}
    monkeypatch.setattr(hlog, "_writer_stamp", lambda: dict(mine))
    p = _seed(tmp_path, last_writer=_foreign_stamp("node-A"))
    log = TRITONSWMM_analysis_log.from_json(p)
    log.gpu_backend_available.set(True)  # write 1: stamp -> mine
    doc = json.loads(p.read_text())
    doc["last_writer"] = _foreign_stamp("node-B")
    p.write_text(json.dumps(doc, indent=2))  # foreign write between write 1 and write 2
    log.cpu_backend_available.set(True)  # write 2: mine == baseline -> 'unchanged'
    assert json.loads(p.read_text())["last_writer"] == mine


def test_exclusive_create_never_leaves_a_partial_document(tmp_path, monkeypatch):
    """T6 (M2). A death inside json.dump leaves NO document under the final name; an
    existing name is refused with FileExistsError and its bytes are untouched. Mutation:
    restore the open(O_EXCL)-on-the-final-name body and the first assertion fails (a
    partial file exists)."""
    target = tmp_path / "log.json"

    def boom(*a, **kw):
        f = a[1]
        f.write("{ partial")
        raise RuntimeError("died mid-dump")

    monkeypatch.setattr(hlog.json, "dump", boom)  # utils.write_json_exclusive uses the json module
    monkeypatch.setattr("hhemt.utils.json.dump", boom)
    with pytest.raises(RuntimeError):
        write_json_exclusive({"a": 1}, target)
    assert not target.exists(), "a partial document was published under the final name"
    assert not list(tmp_path.glob("*.tmp")), "the temp name was not unlinked"
    monkeypatch.undo()

    target.write_text("{}")
    with pytest.raises(FileExistsError):
        write_json_exclusive({"a": 1}, target)
    assert target.read_text() == "{}"


def test_deferred_writes_collapses_sets_into_one_write(tmp_path, monkeypatch):
    """T7 (M4). Two sets inside deferred_writes() land as ONE write_json; outside, two.
    Mutation: remove the `_defer_depth > 0` early return from write() and the count is 3
    (two immediate writes plus the flush)."""
    calls = []
    real = hlog.write_json

    def counting(data, file):
        calls.append(str(file))
        return real(data, file)

    monkeypatch.setattr(hlog, "write_json", counting)
    p = tmp_path / "log.json"
    log = TRITONSWMM_analysis_log(logfile=p)
    with log.deferred_writes():
        log.cpu_backend_available.set(True)
        log.gpu_backend_available.set(True)
    assert len(calls) == 1
    after = json.loads(p.read_text())
    assert after["cpu_backend_available"] is True and after["gpu_backend_available"] is True

    calls.clear()
    log.datatree_consolidation_complete.set(True)
    log.datatree_consolidation_complete.set(False)
    assert len(calls) == 2
