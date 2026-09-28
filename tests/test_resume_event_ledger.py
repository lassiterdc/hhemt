"""The durable resume-event ledger (WP-2D) and its two-exec durability property.

WHY THIS MODULE EXISTS. The per-sim model runtime log is opened ``"w"`` on every runner
exec, so exec N+1 destroys exec N's resume evidence -- measured at 60 of 90 resume events
(67%) on the solver-replay-precision-accounting experiment. The package's acceptance
condition is stated as a property rather than as a code shape: **a resume-event record
written by exec N is readable after exec N+1 has run**. ``test_two_exec_durability`` IS
that condition, and it is written so that the model log's own truncation is asserted in
the same test -- so the test states the defect it is protecting against, not merely that
one file was opened in append mode.

NO SOLVER RUNS HERE. Every test drives ``hhemt.resume_events`` directly with a tmp_path
and text; nothing reaches ``system.compile_*``, ``TRITONSWMM_run``, ``analysis.run()`` or
a ``test_synth_*`` fixture, so this module is laptop-safe by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hhemt import resume_events as re_mod

# A snapshot-path line in the emitted shape: marker literal, then the time as the first
# whitespace token (the rule `eda/raw_resume_identity.parse_resume_timestep` relies on),
# then the remaining contract fields.
SNAPSHOT_LINE = "[OK] SWMM state restored from snapshot to t=3000 s; checkpoint_id=0042"

# A replay-path line carrying the classified fallback reason.
REPLAY_LINE = "[..] SWMM exchange history replayed to t=6000 s (11435 steps); checkpoint_id=0084 reason=absent"


def _log(tmp_path: Path, name: str = "model_tritonswmm_evt0.log") -> Path:
    p = tmp_path / "logs" / "sims" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------
# The acceptance condition
# --------------------------------------------------------------------------------------


def test_two_exec_durability(tmp_path):
    """A record written by exec N is readable after exec N+1 has run.

    The fixture reproduces the exact mechanism that destroys the evidence: each "exec"
    opens the model log with ``"w"`` -- byte-identical to what
    ``run_simulation_runner.py`` does -- so exec N+1 genuinely erases exec N's marker.
    Asserting that erasure in the same test is what makes this a statement about the
    DEFECT and not merely about a file opened in append mode: if the model log did not
    truncate, the ledger would be redundant and this test would pass for the wrong reason.
    """
    model_log = _log(tmp_path)
    ledger = re_mod.ledger_path_for(model_log)

    # ---- exec N: a snapshot-path resume ----
    with open(model_log, "w") as fh:  # exactly the runner's open mode
        fh.write(f"some solver preamble\n{SNAPSHOT_LINE}\nSimulation ends\n")
    n_events = re_mod.harvest_resume_events(model_log.read_text())
    assert re_mod.append_resume_events(ledger, n_events, attempt=1, slurm_jobid="111") == "appended"

    # ---- exec N+1: a replay-path resume; the SAME log path is truncated ----
    with open(model_log, "w") as fh:
        fh.write(f"some solver preamble\n{REPLAY_LINE}\nSimulation ends\n")
    n1_events = re_mod.harvest_resume_events(model_log.read_text())
    assert re_mod.append_resume_events(ledger, n1_events, attempt=2, slurm_jobid="222") == "appended"

    # THE PREMISE: exec N's evidence is gone from the model log.
    surviving_log = model_log.read_text()
    assert re_mod.SNAPSHOT_MARKER not in surviving_log, (
        "fixture did not reproduce the truncation this ledger exists for"
    )
    assert re_mod.REPLAY_MARKER in surviving_log

    # THE ACCEPTANCE CONDITION: exec N's record survives exec N+1.
    rows = re_mod.read_resume_events(ledger)
    assert [r["path"] for r in rows] == ["snapshot", "replay"]
    assert [r["attempt"] for r in rows] == [1, 2]
    assert [r["slurm_jobid"] for r in rows] == ["111", "222"]
    assert rows[0]["resume_time_s"] == 3000.0
    assert rows[1]["resume_time_s"] == 6000.0


def test_ledger_is_not_under_the_model_log_and_is_never_recreated(tmp_path):
    """The ledger's directory is a sibling of the log, so truncating the log cannot reach it.

    A path-level statement of the same property the durability test asserts behaviourally.
    """
    model_log = _log(tmp_path)
    ledger = re_mod.ledger_path_for(model_log)
    assert ledger.parent == model_log.parent / "_resume_events"
    assert ledger.name == "model_tritonswmm_evt0.jsonl"
    assert model_log.parent in ledger.parents
    # Mirrors the in-tree `_walltime` sibling exactly: same parent, sibling dir, {stem}.jsonl.
    walltime_sibling = model_log.parent / "_walltime" / f"{model_log.stem}.jsonl"
    assert ledger.parent.parent == walltime_sibling.parent.parent
    assert ledger.name == walltime_sibling.name


# --------------------------------------------------------------------------------------
# The four contract fields
# --------------------------------------------------------------------------------------


def test_snapshot_line_yields_all_four_fields(tmp_path):
    (ev,) = re_mod.harvest_resume_events(SNAPSHOT_LINE)
    assert ev["path"] == "snapshot"
    assert ev["resume_time_s"] == 3000.0
    assert ev["checkpoint_id"] == "0042"
    # The reason field is meaningful only on the replay path.
    assert ev["replay_reason"] is None
    assert ev["replay_reason_status"] == "not-applicable"
    assert ev["raw"] == SNAPSHOT_LINE


@pytest.mark.parametrize("reason", sorted(re_mod.REPLAY_REASONS))
def test_every_classified_replay_reason_is_recognised(reason):
    line = f"[..] SWMM exchange history replayed to t=6000 s; checkpoint_id=7 reason={reason}"
    (ev,) = re_mod.harvest_resume_events(line)
    assert ev["path"] == "replay"
    assert ev["replay_reason"] == reason
    assert ev["replay_reason_status"] == "parsed"


def test_a_reason_outside_the_closed_set_is_unclassified_not_dropped():
    """An emitter that grows a fourth reason must produce a VISIBLE row, never a silent one."""
    line = "[..] SWMM exchange history replayed to t=1 s; checkpoint_id=7 reason=some-new-reason"
    (ev,) = re_mod.harvest_resume_events(line)
    assert ev["path"] == "replay"
    assert ev["replay_reason"] is None
    assert ev["replay_reason_status"] == "unclassified"
    assert ev["replay_reason_raw"] == line


def test_an_unlabelled_checkpoint_id_is_unparsed_not_silently_null():
    """The one field the emitted shape does not pin must fail LOUDLY and losslessly.

    The row still carries the verbatim line, so a later parser recovers the value without
    re-running the sim -- which is what makes writing this ledger before the emitter's
    exact spelling is observed a safe trade.
    """
    line = "[OK] SWMM state restored from snapshot to t=3000 s <<0042>>"
    (ev,) = re_mod.harvest_resume_events(line)
    assert ev["checkpoint_id"] is None
    assert ev["checkpoint_id_status"] == "unparsed"
    assert ev["raw"] == line  # lossless: the value is still in the row


@pytest.mark.parametrize(
    "label",
    ["checkpoint_id", "checkpoint id", "checkpoint", "ckpt", "cp", "CHECKPOINT_ID"],
)
def test_checkpoint_id_label_family_is_tolerated(label):
    line = f"[OK] SWMM state restored from snapshot to t=3000 s; {label}=0042"
    (ev,) = re_mod.harvest_resume_events(line)
    assert ev["checkpoint_id"] == "0042", f"label {label!r} not tolerated"


def test_a_fresh_exec_produces_no_events_and_writes_no_ledger(tmp_path):
    """Neither marker is not an error -- it is the correct answer for a non-resumed exec."""
    model_log = _log(tmp_path)
    ledger = re_mod.ledger_path_for(model_log)
    events = re_mod.harvest_resume_events("solver preamble\nSimulation ends\n")
    assert events == []
    assert re_mod.append_resume_events(ledger, events, attempt=0) == "no-events"
    assert not ledger.exists()
    assert re_mod.read_resume_events(ledger) == []


@pytest.mark.parametrize("line,expected", [(SNAPSHOT_LINE, 3000.0), (REPLAY_LINE, 6000.0)])
def test_resume_time_agrees_with_the_existing_numeric_parse(tmp_path, line, expected):
    """This module's time parse must agree with `eda/raw_resume_identity.parse_resume_timestep`.

    That existing consumer is what constrains the emitted line shape in the first place, so
    a divergence here would mean two parsers disagreeing about the same line. The check
    RUNS the existing parser against the same text rather than restating its token rule --
    restating it would make the two agree by construction and prove nothing.
    """
    from hhemt.eda.raw_resume_identity import parse_resume_timestep

    log = tmp_path / "model_tritonswmm_evt0.log"
    log.write_text(f"preamble\n{line}\nSimulation ends\n")

    (ev,) = re_mod.harvest_resume_events(log.read_text())
    assert ev["resume_time_s"] == expected
    assert parse_resume_timestep(log) == expected


# --------------------------------------------------------------------------------------
# Drift detectors for the deliberately-duplicated marker literals
# --------------------------------------------------------------------------------------


def test_marker_literals_match_analysis_validation_byte_for_byte():
    """`resume_events` binds its own copy of the two markers to stay a leaf module.

    The copy is accepted; silent drift is not. This is the detector that makes the
    duplication safe, and it is the difference between this copy and the untested
    `run_simulation.py` / `analysis_validation.py` completion-marker pair the tree already
    carries under a "keep the two in sync" comment.
    """
    from hhemt import analysis_validation as av

    assert re_mod.SNAPSHOT_MARKER == av._TRITON_SNAPSHOT_RESTORE_MARKER
    assert re_mod.REPLAY_MARKER == av._TRITON_REPLAY_MARKER


def test_path_vocabulary_matches_the_resume_mechanism_type():
    """`path` uses the same two affirmative tokens `ResumeMechanism` already defines."""
    from hhemt import analysis_validation as av

    for line, expected in ((SNAPSHOT_LINE, "snapshot"), (REPLAY_LINE, "replay")):
        (ev,) = re_mod.harvest_resume_events(line)
        assert ev["path"] == expected
        # The in-tree log-text classifier must agree on the same text.
        assert av.resume_mechanism_from_log(line) == expected


# --------------------------------------------------------------------------------------
# Failure posture: an observability ledger must never fail a finished simulation
# --------------------------------------------------------------------------------------


def test_append_never_raises_on_an_unwritable_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    ledger = blocker / "_resume_events" / "x.jsonl"
    events = re_mod.harvest_resume_events(SNAPSHOT_LINE)
    assert re_mod.append_resume_events(ledger, events, attempt=1) == "write-failed"


def test_read_tolerates_absent_and_corrupt_rows(tmp_path):
    ledger = tmp_path / "x.jsonl"
    assert re_mod.read_resume_events(ledger) == []
    ledger.write_text(
        json.dumps({"path": "snapshot", "attempt": 1}) + "\n" + "{not json\n" + "\n" + json.dumps(["not a dict"]) + "\n"
    )
    rows = re_mod.read_resume_events(ledger)
    assert len(rows) == 1 and rows[0]["path"] == "snapshot"
