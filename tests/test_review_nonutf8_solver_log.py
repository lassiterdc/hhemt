"""Reviewer tests for work package `toolkit-nonutf8-3`: a coupled solver model log that carries
bytes which are not UTF-8 must not change what any toolkit reader concludes from it.

A solver can write non-UTF-8 bytes into its model log (a stack overflow in the SWMM flooding
consistency routine was measured writing six heap-pointer bytes after a debug-file path). Every
marker the toolkit's readers look for is ASCII, so the strict text read is the defect: it raises
`UnicodeDecodeError`, which each reader either lets escape or swallows into an unrelated fallback.

Two property families live here, both quantified over a CLASS of reader and a CLASS of byte
position, neither of which the coder's three File-7 tests cover:

* Files 4-6 (the resume-evidence readers) have no coder test at all. Their property is
  INVARIANCE: the verdict or stamp read from a log with bad bytes equals the one read from the
  same log without them, which is anchored on a value present before and after the change (the
  clean-log verdict), never on wording the change introduces.
* Files 1-3 (the run-completion readers) are covered by the coder at ONE byte position (a bad
  line BEFORE the marker). The positions added here are bytes AFTER the marker, bytes glued to the
  marker on the same line, and a multi-byte sequence truncated at end of file.

Runs solver-free: no compile, no simulation, no Rivanna job.
"""

from __future__ import annotations

import json
import types
from types import SimpleNamespace

import pytest

import hhemt.analysis_validation as av
from hhemt.analysis_validation import check_coupled_resume_validity
from hhemt.run_simulation import TRITONSWMM_run

# The resumed-sim status frame is imported from the classified sibling rather than rebuilt here: rebuilding it
# would name the resume-attempt counter in this file and add a consumer the semantics-parity census must classify.
from tests.test_coupled_resume_validity import _resumed_df

# ---------------------------------------------------------------------------
# Byte-position classes. Each entry is (id, transform) where the transform wraps a CLEAN log body
# (bytes) into a log carrying invalid UTF-8 at a different position relative to the markers.
# ---------------------------------------------------------------------------
_BAD = b"\xf0\xec\x9e\xe0\xf5U"  # six bytes, none a valid UTF-8 sequence start in this context
_TRUNCATED_SEQUENCE = b"\xe2\x82"  # first two bytes of a three-byte sequence, cut off by EOF

_POSITIONS = {
    "bad_line_before_body": lambda body: b"  path: /x/swmm/debug.txt" + _BAD + b"\n" + body,
    "bad_bytes_glued_to_first_line": lambda body: _BAD + body,
    "bad_line_after_body": lambda body: body + b"  trailing: " + _BAD + b"\n",
    "truncated_multibyte_at_eof": lambda body: body + _TRUNCATED_SEQUENCE,
}
_POSITION_IDS = list(_POSITIONS)

_CKPT = b"[..] Reading checkpoint files\n[OK] Checkpoint files read\n"
_SNAPSHOT = b"[..] SWMM state restored from snapshot to t=3600 s (12 steps); resuming live segment\n"
_REPLAY = b"[..] SWMM exchange history replayed to t=3600 s (12 steps); resuming live segment\n"
_ENDS = b"Simulation ends\n"

_MECHANISM_BODIES = [_CKPT + _SNAPSHOT + _ENDS, _CKPT + _REPLAY + _ENDS]

_SHA_POST_ALL = "9db367ddc79f86c7f708686d1dd805dc992fb0a4"  # a real post-fix TRITON commit (arm B reachable)


def _strict_decodable(blob: bytes) -> bool:
    try:
        blob.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


# ---------------------------------------------------------------------------
# File 6 -- parse_resume_timestep
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("position", _POSITION_IDS)
@pytest.mark.parametrize("mechanism_body", _MECHANISM_BODIES, ids=["snapshot", "replay"])
def test_parse_resume_timestep_is_invariant_under_non_utf8_bytes(tmp_path, position, mechanism_body):
    """Property: the resume boundary parsed from a log is the same with and without bad bytes.

    Class: every byte position in `_POSITIONS` x both resume markers (snapshot, replay). Both
    states anchor: the clean log's parse result (3600.0), a value true before and after the fix.
    Why required: the EDA b4b figure loses its resume vline silently if this read raises, and the
    docstring promises None-or-float, never an exception. What kills it: reverting
    `read_text(errors="replace")` to `read_text()` in `parse_resume_timestep`. Second correct
    implementation: `read_bytes().decode("utf-8", errors="replace")` or `errors="ignore"` passes
    too, so the test does not pin the keyword.
    """
    from hhemt.eda.raw_resume_identity import parse_resume_timestep

    clean = tmp_path / "clean.log"
    clean.write_bytes(mechanism_body)
    bad = tmp_path / "bad.log"
    blob = _POSITIONS[position](mechanism_body)
    assert not _strict_decodable(blob), "fixture must really be non-UTF-8, or the test proves nothing"
    bad.write_bytes(blob)

    expected = parse_resume_timestep(clean)
    assert expected == 3600.0
    assert parse_resume_timestep(bad) == expected


def test_parse_resume_timestep_log_without_marker_stays_none_with_bad_bytes(tmp_path):
    """Negative arm of the File 6 property: a bad-byte log with NO resume marker is still None,
    so tolerance did not turn into a spurious boundary.

    Class: marker-free logs carrying invalid UTF-8. Why required: the function's contract is
    None for a log that carries neither marker. What kills it: a change that treats any decode
    problem as a resume boundary. Second correct implementation: any decoder that maps the bad
    bytes to a non-digit still returns None.
    """
    from hhemt.eda.raw_resume_identity import parse_resume_timestep

    log = tmp_path / "nomarker.log"
    log.write_bytes(_POSITIONS["bad_line_before_body"](_CKPT + _ENDS))
    assert parse_resume_timestep(log) is None


# ---------------------------------------------------------------------------
# Files 4 and 5 -- the resume-evidence stamp and the resume-mechanism check
# ---------------------------------------------------------------------------
def _analysis_stub(*, df, simlog_dir):
    return SimpleNamespace(
        _system=SimpleNamespace(cfg_system=SimpleNamespace(toggle_tritonswmm_model=True)),
        cfg_analysis=SimpleNamespace(
            toggle_sensitivity_analysis=False,
            is_experiment_member=False,
            resume_interruption_schedule=None,
            TRITON_reporting_timestep_s=None,
        ),
        analysis_paths=SimpleNamespace(
            analysis_datatree_zarr=None,
            sensitivity_datatree_zarr=None,
            simlog_directory=simlog_dir,
        ),
        df_status=df,
    )


def _write_log_bytes(analysis, blob):
    """Place the log where the PRODUCER's own resolver puts it, never at a hand-built path."""
    from hhemt.run_simulation import model_logfile_for

    p = model_logfile_for(analysis, 0, "tritonswmm")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(blob)
    return p


@pytest.mark.parametrize("position", _POSITION_IDS)
@pytest.mark.parametrize("mechanism_body", _MECHANISM_BODIES, ids=["snapshot", "replay"])
def test_consolidation_resume_stamp_is_invariant_under_non_utf8_bytes(tmp_path, position, mechanism_body):
    """Property: `_stamp_coupled_resume_evidence` records the same evidence for a log with bad
    bytes as for the same log without them.

    Class: every byte position x both resume markers. Both states anchor: the clean-log record
    (mechanism, resume_t, resumed, completed). Why required: the decode error is swallowed by the
    function's best-effort `except Exception: continue`, so the sub's durable stamp silently
    vanishes and every later consumer sees an unstamped sub. What kills it: reverting File 4's
    `read_text(errors="replace")`. Second correct implementation: decoding bytes with
    `errors="ignore"` yields the same record.
    """
    import xarray as xr

    from hhemt.processing_analysis import _stamp_coupled_resume_evidence

    def _record(blob, sub_dir):
        scen = sub_dir / "sim_0"
        a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=sub_dir / "logs" / "sims")
        _write_log_bytes(a, blob)
        tree = xr.DataTree.from_dict({"/": xr.Dataset()})
        _stamp_coupled_resume_evidence(tree, a, df_status=a.df_status)
        raw = tree.attrs.get("coupled_resume_replay_evidence")
        return None if raw is None else json.loads(raw)[str(scen)]

    blob = _POSITIONS[position](mechanism_body)
    assert not _strict_decodable(blob)
    clean_dir = tmp_path / "clean"
    bad_dir = tmp_path / "bad"
    clean_rec = _record(mechanism_body, clean_dir)
    assert clean_rec is not None and clean_rec["resume_t"] == 3600.0
    bad_rec = _record(blob, bad_dir)
    assert bad_rec is not None, "the stamp vanished for a log with non-UTF-8 bytes"
    assert bad_rec == clean_rec


@pytest.mark.parametrize("position", _POSITION_IDS)
@pytest.mark.parametrize(
    "body, replayed",
    [(_CKPT + _REPLAY + _ENDS, True), (_CKPT + _ENDS, False)],
    ids=["marker_present_passes", "marker_absent_warns"],
)
def test_resume_validity_check_is_invariant_under_non_utf8_bytes(monkeypatch, tmp_path, position, body, replayed):
    """Property: `check_coupled_resume_validity` returns the same verdict, summary and details
    for a log with bad bytes as for the same log without them -- in BOTH verdict states.

    Class: every byte position x {marker present -> PASS, marker absent -> WARN}. Both states
    anchor: the clean-log verdict. Why required: a decode error falls to the durable stamp, and
    with no stamp the row is counted INDETERMINATE, so a sim that really lacks the resume marker
    would be reported as unexaminable instead of WARN, and a good one would lose its positive
    examination. What kills it: reverting File 5's `read_text(errors="replace")`. Second correct
    implementation: `errors="ignore"` yields the same verdicts.
    """
    monkeypatch.setattr(av, "_read_triton_provenance", lambda a: _SHA_POST_ALL)

    def _verdict(blob, sub_dir):
        scen = sub_dir / "sim_0"
        a = _analysis_stub(df=_resumed_df(str(scen)), simlog_dir=sub_dir / "logs" / "sims")
        _write_log_bytes(a, blob)
        res = check_coupled_resume_validity(a)
        return res.passed, res.summary, [d["detail"] for d in res.details]

    blob = _POSITIONS[position](body)
    assert not _strict_decodable(blob)
    clean = _verdict(body, tmp_path / "clean")
    assert clean[0] is replayed, "control: the clean log must produce the verdict the arm is named for"
    assert "1 resumed coupled sim(s) examined" in clean[1]
    assert _verdict(blob, tmp_path / "bad") == clean


# ---------------------------------------------------------------------------
# Files 1-3 -- run_simulation readers, at byte positions the coder's tests do not use
# ---------------------------------------------------------------------------
class _FakeField:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _fake_run(tmp_path, blob, *, model_type, recorded, real_rpt_gate):
    """The coder's `_bytes_gate` shape, rebuilt here so this file stands alone."""
    log_file = tmp_path / f"model_{model_type}.log"
    log_file.write_bytes(blob)
    perf = tmp_path / "performance.txt"
    perf.write_text("x")

    class _ModelLog:
        simulation_completed = _FakeField(recorded)

    fake = types.SimpleNamespace(
        _scenario=types.SimpleNamespace(get_log=lambda mt: _ModelLog()),
        _analysis_level_model_logfile=lambda mt: log_file,
        performance_file=lambda model_type: perf,
    )
    if real_rpt_gate:
        fake._coupled_swmm_report_finalized = lambda mt: TRITONSWMM_run._coupled_swmm_report_finalized(fake, mt)
    else:
        fake._coupled_swmm_report_finalized = lambda mt: True
    return fake


_RUN_POSITIONS = [p for p in _POSITION_IDS if p != "bad_line_before_body"]  # that one is the coder's own


@pytest.mark.parametrize("position", _RUN_POSITIONS)
@pytest.mark.parametrize("has_marker", [True, False], ids=["finished", "unfinished"])
def test_raw_marker_fallback_verdict_is_invariant_under_non_utf8_position(tmp_path, position, has_marker):
    """Property: `model_run_completed`'s raw-marker fallback answers by the ASCII marker alone,
    wherever the bad bytes sit and in both marker states (no false green, no false red).

    Class: byte positions the coder's test does not use x {marker present, marker absent}. Both
    states anchor: the verdict of the same log without bad bytes. Why required: a decode error
    raises out of the completion gate and fails a finished simulation. What kills it: reverting
    File 2. Second correct implementation: decoding with `errors="ignore"` passes.
    """
    body = (b"Time: 100 dt: 0.5\n" + _ENDS) if has_marker else b"Time: 100 dt: 0.5\n"
    blob = _POSITIONS[position](body)
    assert not _strict_decodable(blob)
    fake = _fake_run(tmp_path, blob, model_type="tritonswmm", recorded=None, real_rpt_gate=False)
    assert TRITONSWMM_run.model_run_completed(fake, "tritonswmm") is has_marker


@pytest.mark.parametrize("position", _RUN_POSITIONS)
@pytest.mark.parametrize("has_marker", [True, False], ids=["finished", "unfinished"])
def test_triton_arm_of_rpt_gate_verdict_is_invariant_under_non_utf8_position(tmp_path, position, has_marker):
    """Property: the triton arm of the report gate (a REAL call, since the gate stubs it) answers
    by the ASCII marker alone at every byte position, in both marker states.

    Class: byte positions the coder's test does not use x {marker present, marker absent}. Both
    states anchor: the verdict of the same log without bad bytes. Why required: this read sits in
    the success path of a pure-TRITON run; a decode error fails a finished sim. What kills it:
    reverting File 1. Second correct implementation: decoding with `errors="ignore"` passes.
    """
    body = (b"Time: 100 dt: 0.5\n" + _ENDS) if has_marker else b"Time: 100 dt: 0.5\n"
    blob = _POSITIONS[position](body)
    assert not _strict_decodable(blob)
    fake = _fake_run(tmp_path, blob, model_type="triton", recorded=True, real_rpt_gate=True)
    assert TRITONSWMM_run.model_run_completed(fake, "triton") is has_marker


@pytest.mark.parametrize("position", _RUN_POSITIONS)
@pytest.mark.parametrize(
    "tail, expected",
    [
        (b"slurmstepd: error: *** STEP 1.0 CANCELLED DUE TO TIME LIMIT ***\n", "timeout"),
        (b"Time: 100 dt: 0.5\n", None),
    ],
    ids=["timeout_marker", "no_marker"],
)
def test_failure_classifier_is_invariant_under_non_utf8_position(tmp_path, position, tail, expected):
    """Property: `_classify_model_log_failure` classifies by its ASCII markers at every byte
    position, and a marker-free log is not forced into the timeout class.

    Class: byte positions the coder's test does not use x {timeout marker, none}. Both states
    anchor: the classification of the same log without bad bytes. Why required: a decode error
    here masks the real failure class behind a UnicodeDecodeError. What kills it: reverting File 3.
    Second correct implementation: decoding with `errors="ignore"` passes.
    """
    blob = _POSITIONS[position](tail)
    assert not _strict_decodable(blob)
    fake = _fake_run(tmp_path, blob, model_type="tritonswmm", recorded=None, real_rpt_gate=False)
    result = TRITONSWMM_run._classify_model_log_failure(fake, "tritonswmm")
    if expected is None:
        assert result != "timeout"
    else:
        assert result == expected
