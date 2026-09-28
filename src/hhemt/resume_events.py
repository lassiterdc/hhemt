"""Durable, append-only per-sim record of coupled-resume events (WP-2D).

WHY THIS MODULE EXISTS, stated as the measured defect rather than as a feature. The
per-sim model runtime log is opened ``"w"`` on EVERY runner exec
(``run_simulation_runner.py``), so it holds only the last exec. On the
``solver-replay-precision-accounting`` experiment that destroyed **60 of 90 resume
events -- 67% -- leaving no surviving path evidence**. The question the developer needs
answered after the fact -- *did this resume take the full-precision SNAPSHOT path or fall
back to REPLAY, and if it fell back, why* -- is therefore checkable only at the moment,
and after the fact is when it is read.

THE REQUIREMENT IS DELIBERATELY NARROW. Every resume event leaves one row that survives
the NEXT exec, carrying four fields: the checkpoint id, the resume time, the path taken
(``snapshot`` | ``replay``), and -- on ``replay`` -- the classified reason (``absent`` |
``retention-collision`` | ``pre-snapshot-checkpoint``). Four fields, one row per event,
append-only.

THE SHAPE FOLLOWS AN IN-TREE PRECEDENT rather than inventing one: ``_walltime/{stem}.jsonl``
beside the same model log is already an append-only per-sim JSONL ledger written with
``open(..., "a")``. This ledger is its sibling, in ``_resume_events/{stem}.jsonl``, under
the same analysis-level ``logs/sims/`` directory -- a directory nothing re-creates per exec.

LEAF MODULE, DELIBERATELY. This imports stdlib only and no ``hhemt.*`` sibling, so the
runner (which writes) and any future validator or renderer (which reads) can both import
it with no cycle. The same property is why ``summary_paths.py`` and ``slurm_liveness.py``
exist in this tree. It matters concretely here: ``analysis_validation.py`` is the natural
home of a future consumer, and a module-level import from this file into that one would
close a cycle if the constants had been sourced the other way.

WHAT IS *NOT* PINNED, AND HOW THIS MODULE SURVIVES THAT. The solver emits the four fields
in one line, in the existing ``...to t=`` shape, so the numeric parse at
``eda/raw_resume_identity.py`` keeps working unchanged. That shape pins TWO of the four
fields exactly -- the path (which marker literal appears) and the resume time (the first
whitespace token after the literal). The remaining two are pinned by VOCABULARY rather
than by position: the reason is a closed three-value set matched as a literal token, and
the checkpoint id is matched by a labelled ``key=value`` scan over the spellings below.
**Every row also carries the verbatim source line**, so a row written before the emitter's
exact spelling was known loses nothing: a later parser reads ``raw`` and recovers the
field. That is what makes writing this ledger before the emitter lands a safe trade rather
than a guess.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Literal

__all__ = [
    "LEDGER_DIRNAME",
    "REPLAY_MARKER",
    "REPLAY_REASONS",
    "SNAPSHOT_MARKER",
    "ResumePath",
    "append_resume_events",
    "harvest_resume_events",
    "ledger_path_for",
    "read_resume_events",
]

#: TRITON's affirmative marker for the full-precision SNAPSHOT restore path.
#:
#: SECOND DEFINITION, DELIBERATE AND TESTED. ``analysis_validation.py`` binds the identical
#: literal as ``_TRITON_SNAPSHOT_RESTORE_MARKER``. Importing it from there would make this
#: module a non-leaf and would close a cycle the moment a validator consumer imports this
#: one, and re-homing the constants into here would mean editing ``analysis_validation.py``
#: -- which is another work package's file. The copy is therefore accepted and its drift is
#: made machine-detectable instead of comment-managed: ``tests/test_resume_event_ledger.py``
#: asserts byte equality against both siblings. The tree already carries an unTESTED copy of
#: the sibling completion marker (``run_simulation.py`` vs ``analysis_validation.py``,
#: managed by a "keep the two in sync" comment); this one is the tested kind.
SNAPSHOT_MARKER = "SWMM state restored from snapshot to t="

#: TRITON's affirmative marker for the exchange-history REPLAY fallback path. Same
#: ``...to t=`` shape as the snapshot marker by construction, so one numeric parse serves
#: both. See ``SNAPSHOT_MARKER`` for why this literal is bound here rather than imported.
REPLAY_MARKER = "SWMM exchange history replayed to t="

#: The classified reasons a resume fell back from snapshot to replay. A CLOSED set: a token
#: outside it is recorded verbatim under ``replay_reason_raw`` and leaves ``replay_reason``
#: None, so an emitter that grows a fourth reason produces a visible unclassified row rather
#: than a silently-dropped one.
REPLAY_REASONS: frozenset[str] = frozenset({"absent", "retention-collision", "pre-snapshot-checkpoint"})

#: Sibling of ``_walltime`` under the analysis-level ``logs/sims/`` directory.
LEDGER_DIRNAME = "_resume_events"

ResumePath = Literal["snapshot", "replay"]

#: Accepted spellings for the checkpoint-id field.
#:
#: THE ONE FIELD THE EMITTED SHAPE DOES NOT PIN. The marker literal pins the path and the
#: first-token rule pins the time; the reason is pinned by its closed vocabulary. The
#: checkpoint id's LABEL is pinned by nothing, so this scan accepts the plausible family
#: rather than mandating one member of it. A line whose label falls outside the family
#: yields ``checkpoint_id=None`` with ``checkpoint_id_status="unparsed"`` -- an explicit,
#: greppable non-answer -- and the verbatim line is retained under ``raw`` so the value is
#: recoverable by a one-line change here once the emitter's spelling is observed.
_CHECKPOINT_ID_RE = re.compile(
    r"\b(?:checkpoint[_ ]?id|checkpoint|ckpt|cp)\s*(?:id)?\s*[=:]\s*(?P<v>[A-Za-z0-9_.\-]+)",
    re.IGNORECASE,
)


def ledger_path_for(model_logfile: Path) -> Path:
    """The append-only resume-event ledger for one sim's model log.

    PATH-ONLY and a pure function of its argument: it resolves a path and touches no
    filesystem, so a validator may call it without the directory-creating side effects a
    ``TRITONSWMM_scenario`` construction carries. Mirrors ``_walltime``'s convention exactly
    -- same parent, sibling directory, ``{stem}.jsonl`` -- because the two ledgers answer
    different questions about the same sim and a reader should not have to learn two
    layouts.
    """
    return model_logfile.parent / LEDGER_DIRNAME / f"{model_logfile.stem}.jsonl"


def _parse_leading_float(tail: str) -> float | None:
    """The resume time from the text immediately following a marker literal.

    The token rule is IDENTICAL to ``eda/raw_resume_identity.parse_resume_timestep``'s --
    first whitespace token, trailing ``.,;`` stripped -- deliberately, because that consumer
    is what constrains the emitted shape in the first place. Restating the rule rather than
    importing it keeps this module a leaf; the two are pinned together by a test.
    """
    parts = tail.strip().split()
    if not parts:
        return None
    try:
        return float(parts[0].rstrip(".,;"))
    except ValueError:
        return None


def harvest_resume_events(log_text: str) -> list[dict]:
    """Every resume event in ONE exec's model-log text, in file order.

    Returns a list of dicts carrying the four contract fields plus the verbatim source
    line. NEVER raises on malformed input: an unparseable field is represented as None
    beside an explicit ``*_status`` token, because a resume event that happened and parsed
    badly must still leave a row -- dropping it would reproduce, at field granularity,
    exactly the evidence loss this ledger exists to stop.

    Returns [] for text carrying neither marker, which is the correct answer for a fresh
    exec and is NOT an error.
    """
    events: list[dict] = []
    for line in log_text.splitlines():
        if SNAPSHOT_MARKER in line:
            path: ResumePath = "snapshot"
            tail = line.split(SNAPSHOT_MARKER, 1)[1]
        elif REPLAY_MARKER in line:
            path = "replay"
            tail = line.split(REPLAY_MARKER, 1)[1]
        else:
            continue

        resume_time_s = _parse_leading_float(tail)

        m = _CHECKPOINT_ID_RE.search(line)
        checkpoint_id = m.group("v") if m else None

        # The reason is meaningful ONLY on the replay path. Scanning for it on a snapshot
        # line would let a reason-shaped word elsewhere in the line manufacture a fallback
        # reason for a run that never fell back.
        replay_reason: str | None = None
        replay_reason_raw: str | None = None
        if path == "replay":
            # Longest-first so `pre-snapshot-checkpoint` is not shadowed by a substring of
            # itself in a future vocabulary extension.
            for reason in sorted(REPLAY_REASONS, key=len, reverse=True):
                if re.search(rf"(?<![A-Za-z0-9_\-]){re.escape(reason)}(?![A-Za-z0-9_\-])", line):
                    replay_reason = reason
                    break
            if replay_reason is None:
                replay_reason_raw = line.strip()

        events.append(
            {
                "path": path,
                "resume_time_s": resume_time_s,
                "resume_time_status": "parsed" if resume_time_s is not None else "unparsed",
                "checkpoint_id": checkpoint_id,
                "checkpoint_id_status": "parsed" if checkpoint_id is not None else "unparsed",
                "replay_reason": replay_reason,
                "replay_reason_status": (
                    "not-applicable" if path == "snapshot" else ("parsed" if replay_reason else "unclassified")
                ),
                "replay_reason_raw": replay_reason_raw,
                # LOSSLESS. Every later parser reads this rather than re-running the sim.
                "raw": line.strip(),
            }
        )
    return events


def append_resume_events(
    ledger: Path,
    events: list[dict],
    *,
    attempt: int,
    slurm_jobid: str | None = None,
) -> str:
    """Append one row per resume event to the durable per-sim ledger.

    Returns a short outcome token -- ``"appended"``, ``"no-events"`` or ``"write-failed"``
    -- so the caller can log WHICH arm ran and a test can assert it without reading the
    filesystem twice. The same observable-arm shape ``_record_queue_time`` uses, and for the
    same reason: an inline version of this has identical behaviour and no way to check it.

    NEVER raises. An observability ledger must not be able to fail a finished simulation,
    so the write path is wrapped and a failure degrades to an absent record -- which every
    reader here already represents as an empty list.

    ``attempt`` is the sim's persisted ``n_resumes`` as read BEFORE this exec, matching the
    ``_walltime`` ledger's own ``attempt`` key so the two ledgers join on it. It also carries
    the only available stale-row signature: a member-scoped force-rerun deletes the member
    directory including ``log_{model_type}.json``, so ``n_resumes`` restarts and a reader
    seeing the attempt sequence restart knows a wipe intervened. The force-rerun path is NOT
    wired to delete this ledger -- doing so would mean editing ``analysis.py``, another work
    package's file -- so that signature is how a reader tells the eras apart today.
    """
    if not events:
        return "no-events"
    stamped = [
        {
            **event,
            "attempt": int(attempt),
            "slurm_jobid": slurm_jobid,
            # UTC and offset-aware. A ledger is read on a different machine from the one
            # that wrote it, and a naive local timestamp is not orderable across the two.
            "recorded_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        for event in events
    ]
    try:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger, "a") as handle:
            for row in stamped:
                handle.write(json.dumps(row) + "\n")
    except OSError:
        return "write-failed"
    return "appended"


def read_resume_events(ledger: Path) -> list[dict]:
    """Every durably-recorded resume event for one sim, oldest first.

    THIS IS THE READER, and its contract is to answer across execs -- which is the whole
    point of the package. Reading the model log answers for the last exec only.

    Returns [] when the ledger is absent or unreadable: a sim that never resumed has no
    ledger, and that is the same answer as "no resume events", so an absent file is not an
    error. A malformed line is SKIPPED rather than fatal, for the same reason the writer
    never raises -- one corrupt row must not hide the rows around it.
    """
    try:
        text = ledger.read_text()
    except OSError:
        return []
    rows: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows
