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
    """Make every pathlib READ of `target` raise FileNotFoundError, leaving writes alone.

    This is the STALE-NEGATIVE-LOOKUP model: pathlib's `Path.open` / `Path.read_bytes`
    answer ENOENT for a name that exists. The M1 write path resolves the name ONCE with
    `os.open(O_RDWR|O_CREAT)`, which this helper deliberately does NOT patch -- a
    create-capable open reaches the filesystem and finds the inode, which is exactly the
    property M1 relies on. So under M1 a write through this helper SUCCEEDS and merges,
    and the pre-M1 implementations (`Path.exists`+`Path.open`, then `Path.read_bytes` +
    exclusive create) REFUSED. That difference is the mutation differential for M1.
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


def test_stale_negative_lookup_merges_instead_of_refusing(tmp_path, monkeypatch):
    """M1 DIFFERENTIAL. A stale negative pathlib lookup on a present log no longer refuses.

    Pre-M1: read_bytes() ENOENT -> exclusive create -> EEXIST -> ProcessingError (the
    campaign's attributed :359 failure). Post-M1: os.open(O_RDWR|O_CREAT) reads the real
    inode, the merge preserves every on-disk value, and this instance's own field lands.
    The invariant the retired test protected -- no NON-NULL on-disk value is clobbered by
    an information-free instance -- is asserted key by key. Mutation: restore
    `self.logfile.read_bytes()` in write() and this test fails at the write.
    """
    p = _seed(tmp_path)
    before = json.loads(p.read_text())

    log = TRITONSWMM_analysis_log.from_json(p)
    _reads_fail(monkeypatch, p)
    log.gpu_backend_available.set(True)  # pre-M1 this raises ProcessingError
    monkeypatch.undo()

    after = json.loads(p.read_text())
    for key, value in before.items():
        assert after[key] == value, f"on-disk value of {key!r} was clobbered by a merge"
    assert after["gpu_backend_available"] is True
    assert isinstance(after["last_writer"], dict) and after["last_writer"]["hostname"]


def test_unparseable_log_is_quarantined_by_move_and_reconstructed(tmp_path):
    """M3 DIFFERENTIAL (decision reversal under [Q402]: "never leave a corrupting problem
    unfixed, ever"). A corrupt-but-present log is MOVED to `.unreadable`, the name is
    rebuilt from this instance's state, and the recovery is recorded on the document.
    Pre-M3 this raised ProcessingError and left the corrupt bytes under the live name.
    Mutation: restore the `raise ProcessingError("log write (read-modify-write)" ...)` in
    the unparseable arm and the `.set()` below raises.
    """
    p = tmp_path / "log.json"
    p.write_bytes(b"{ this is not json")
    log = TRITONSWMM_analysis_log(logfile=p)

    log.gpu_backend_available.set(True)  # pre-M3 raises

    assert (tmp_path / "log.json.unreadable").read_bytes() == b"{ this is not json", (
        "the quarantine must hold the bytes that actually failed to parse"
    )
    after = json.loads(p.read_text())
    assert after["gpu_backend_available"] is True
    assert after["recovered_from_unparseable"][0]["quarantine"] == str(tmp_path / "log.json.unreadable")
    assert after["recovered_from_unparseable"][0]["bytes"] == len(b"{ this is not json")

    # The read side (M3b): from_json on the SAME corrupt bytes emits the same record in
    # memory and moves nothing.
    p2 = tmp_path / "log2.json"
    p2.write_bytes(b"{ this is not json")
    loaded = TRITONSWMM_analysis_log.from_json(p2)
    assert loaded.recovered_from_unparseable[0]["quarantine"] is None
    assert p2.read_bytes() == b"{ this is not json", "from_json must never move the document"
    assert not (tmp_path / "log2.json.unreadable").exists()


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
