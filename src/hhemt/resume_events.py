"""Durable, append-only per-sim record of coupled-resume events.

WHY THIS MODULE EXISTS, stated as the measured defect rather than as a feature. The
per-sim model runtime log is opened ``"w"`` on EVERY runner exec
(``run_simulation_runner.py``), so it holds only the last exec. On one measured
multi-resume experiment that destroyed **60 of 90 resume events -- 67% -- leaving no
surviving path evidence**. The question the developer needs
answered after the fact -- *did this resume take the full-precision SNAPSHOT path or fall
back to REPLAY, and if it fell back, why* -- is therefore checkable only at the moment,
and after the fact is when it is read.

THE REQUIREMENT IS DELIBERATELY NARROW. Every resume event leaves one row that survives
the NEXT exec, append-only, and a row is one of two kinds. A RESUME row records a resume
that completed and carries four fields: the checkpoint id, the resume time, the path taken
(``snapshot`` | ``replay``), and -- on ``replay`` -- the classified reason (``absent`` |
``retention-collision`` | ``pre-snapshot-checkpoint``). A REFUSAL row records a launch the
solver declared unusable and therefore resumed nothing: its path is ``step-back-required``,
``stop`` or ``clean-start``, it has no resume time, and it carries the solver's ``reason``
and ``target`` (see ``_refusal_row``). A refusal row carries every key a resume row does,
with the resume-only fields set to None or ``not-applicable``, plus those two; so a reader
written for resume rows finds every key it expects on a refusal row (though
``resume_time_s`` is None there), and only a reader that wants ``reason`` / ``target``
needs to know the second kind exists.

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

WHAT THE EMITTED SHAPE PINS. The solver
emits the four fields in one line, in the existing ``...to t=`` shape, so the numeric parse
at ``eda/raw_resume_identity.py`` keeps working unchanged. That shape pins TWO of the four
fields exactly -- the path (which marker literal appears) and the resume time (the first
whitespace token after the literal). The remaining two are pinned by VOCABULARY rather than
by position: the reason is a closed three-value set matched as a literal token, and the
checkpoint id is matched by a labelled ``key=value`` scan over the spellings below.

The emitter's spelling is OBSERVED, not assumed: ``swmm_triton.h:1229`` and ``:853`` in
the solver tree at ``f9c28da`` emit the label ``checkpoint=`` inside a trailing
``[resume-event ...]`` bracket, behind the ANSI-wrapped ``[..]`` prefix that
``constants.h:196``'s ``IN`` macro expands to. ``tests/test_resume_event_ledger.py`` pins
BOTH byte-reconstructed real lines, so the parser is held against what the solver writes
rather than against this module author's model of it. The spelling family below is
therefore a TOLERANCE, not a hedge against an unknown: its value is that a solver-side
relabelling inside the family does not silently zero the ledger. **Every row also carries
the verbatim source line**, so a field this module parses wrongly is still recoverable: a
later parser reads ``raw`` rather than re-running the sim.
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
    "STEPBACK_IMPOSSIBLE_PREFIX",
    "STEPBACK_REQUIRED_PREFIX",
    "STEPBACK_TAKEN_PREFIX",
    "STOP_PREFIX",
    "ResumePath",
    "annotate_stepback",
    "append_resume_events",
    "audit_resume_evidence",
    "harvest_from_logfile",
    "harvest_resume_events",
    "ledger_path_for",
    "read_resume_events",
    "stepback_confirmations",
    "stepback_confirmations_from_logfile",
]

#: TRITON's affirmative marker for the full-precision SNAPSHOT restore path.
#:
#: SECOND DEFINITION, DELIBERATE AND TESTED. ``analysis_validation.py`` binds the identical
#: literal as ``_TRITON_SNAPSHOT_RESTORE_MARKER``. Importing it from there would make this
#: module a non-leaf and would close a cycle the moment a validator consumer imports this
#: one. The reverse -- defining the constants only here and having ``analysis_validation.py``
#: import them -- is cycle-free and is the cleaner end state; it is not done here because it
#: changes ``analysis_validation.py``'s own definitions, which this module does not own.
#: Until it is, the copy is accepted and its drift is
#: made machine-detectable instead of comment-managed: ``tests/test_resume_event_ledger.py``
#: asserts byte equality against both siblings. The sibling TRITON completion literal is the
#: contrast: ``"Simulation ends"`` is written inline at its uses in ``run_simulation.py`` and
#: bound separately as ``_TRITON_COMPLETION_MARKER`` in ``analysis_validation.py``, with no
#: shared name between them; this copy is bound by name and byte-tested.
SNAPSHOT_MARKER = "SWMM state restored from snapshot to t="

#: TRITON's affirmative marker for the exchange-history REPLAY fallback path. Same
#: ``...to t=`` shape as the snapshot marker by construction, so one numeric parse serves
#: both. See ``SNAPSHOT_MARKER`` for why this literal is bound here rather than imported.
REPLAY_MARKER = "SWMM exchange history replayed to t="

#: The classified reasons a resume fell back from snapshot to replay. A CLOSED set: a token
#: outside it is recorded verbatim under ``replay_reason_raw`` and leaves ``replay_reason``
#: None, so an emitter that grows a fourth reason produces a visible unclassified row rather
#: than a silently-dropped one.
REPLAY_REASONS: frozenset[str] = frozenset({"absent", "retention-collision", "pre-snapshot-checkpoint", "write-failed"})

#: Sibling of ``_walltime`` under the analysis-level ``logs/sims/`` directory.
LEDGER_DIRNAME = "_resume_events"

#: The solver's lines for a launch at a checkpoint whose snapshot is unusable, and for the
#: launch that then resumed from an earlier one. Each is a CROSS-REPOSITORY CONTRACT with the
#: TRITON solver, which must emit these lines byte-for-byte, integers excepted:
#:
#:   ROW 4a  SWMM snapshot step-back required: checkpoint N unusable
#:             [resume-event checkpoint=N path=step-back reason=killed-write target=M]
#:   ROW 4b  SWMM snapshot stop: checkpoint N unusable
#:             [resume-event checkpoint=N path=stop reason=write-failed target=M]
#:   ROW 4c  SWMM snapshot step-back impossible: checkpoint N unusable, no published snapshot below it
#:             [resume-event checkpoint=N path=clean-start reason=R target=0]
#:   ROW 5   SWMM snapshot step-back taken: resumed from checkpoint M
#:             [resume-event checkpoint=M path=step-back-taken from=N]
#:
#: (each is ONE output line; it is wrapped here only for width, at the single space before
#: the bracket.)
#:
#: Rows 4a-4c each become ONE ledger row; Row 5 becomes none and is read only to corroborate
#: the toolkit picker's own record of a step-back. "The toolkit picker" throughout this module
#: is ``run_simulation.py``'s ``_pick_resume_cfg_past_unusable_snapshot``: before launch it
#: chooses the hotstart cfg, stepping back past a checkpoint whose SWMM snapshot is unusable,
#: and leaves its decision on ``stepback_record``. The harvest keys on the prefixes, not on the
#: bracket tokens, so that grepping the solver source for these four strings whenever the solver
#: pin moves is a complete check that this harvest still matches what the solver writes; a
#: rename that check misses would silently empty the harvest rather than fail.
STEPBACK_REQUIRED_PREFIX = "SWMM snapshot step-back required:"
STOP_PREFIX = "SWMM snapshot stop:"
STEPBACK_IMPOSSIBLE_PREFIX = "SWMM snapshot step-back impossible:"
STEPBACK_TAKEN_PREFIX = "SWMM snapshot step-back taken:"

#: Row-4 prefix -> the ledger ``path`` its row carries. ``step-back-required`` rather than the
#: solver's own ``step-back`` token, because the toolkit picker is supposed to have stepped
#: back already: the row records that it did not, which is an alarm (a picker miss).
_REFUSAL_PATHS: tuple[tuple[str, str], ...] = (
    (STEPBACK_REQUIRED_PREFIX, "step-back-required"),
    (STOP_PREFIX, "stop"),
    (STEPBACK_IMPOSSIBLE_PREFIX, "clean-start"),
)
_BRACKET_FIELD_RE = re.compile(r"\b(?P<k>reason|target|from)=(?P<v>[A-Za-z0-9_.\-]+)")

ResumePath = Literal["snapshot", "replay", "step-back-required", "stop", "clean-start"]

#: Accepted spellings for the checkpoint-id field.
#:
#: THE ONE FIELD THE EMITTED SHAPE DOES NOT PIN BY POSITION. The marker literal pins the
#: path and the first-token rule pins the time; the reason is pinned by its closed
#: vocabulary. The checkpoint id is found by its LABEL, which the solver emits as
#: ``checkpoint=`` (the observed spelling, pinned by ``tests/test_resume_event_ledger.py``);
#: this scan accepts a family around that spelling so a solver-side relabelling inside the
#: family does not silently zero the field. A line whose label falls outside the family
#: yields ``checkpoint_id=None`` with ``checkpoint_id_status="unparsed"`` -- an explicit,
#: greppable non-answer -- and the verbatim line is retained under ``raw`` so the value is
#: recoverable by a one-line change here.
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

    Returns [] for text carrying neither resume marker and no Row-4 line, which is the
    correct answer for a fresh exec and is NOT an error.

    Also returns one row per Row-4 line (``path`` ``step-back-required``, ``stop`` or
    ``clean-start``; see ``_REFUSAL_PATHS``): the solver declaring the launch's checkpoint
    unusable is that launch's event, so it is recorded as one rather than leaving the
    attempt with no row. Row-5 lines add no row (see ``stepback_confirmations``).
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
            refusal = _refusal_row(line)
            if refusal is not None:
                events.append(refusal)
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


def harvest_from_logfile(model_logfile: Path) -> list[dict]:
    """Every resume event in one sim's model log, read from the file and never raising.

    THE READ IS THE RAISING STEP AND IT BELONGS HERE RATHER THAN AT THE CALL SITE, which
    is the measured defect this function exists to close. The runner previously called
    ``harvest_resume_events(model_logfile.read_text())`` under ``except OSError``, and
    ``Path.read_text()`` raises ``UnicodeDecodeError`` on a byte the ambient encoding
    cannot decode -- a ``ValueError``, not an ``OSError``. It escaped to the runner's outer
    ``except Exception``, which writes ``_status/_failed/{rule_token}.json`` and returns 1,
    so an observability ledger could record a FINISHED simulation as FAILED and have it
    re-dispatched. Putting the read inside the module whose contract is NEVER-RAISES is
    what makes that unreachable by construction rather than by a handler someone must keep
    wide.

    ``errors="replace"`` rather than a bare ``try``/``return []``, and the difference is
    the whole point: returning [] on one undecodable byte would discard the WHOLE exec's
    resume evidence, which is the evidence loss this ledger exists to stop, reproduced at
    exec granularity by its own guard. Both marker lines are pure ASCII, so a replacement
    character elsewhere in the log leaves every contract field intact and the damage is
    confined to the bytes that were already unreadable.

    Deliberately NOT a UTF-8-then-CP-1252 ladder (the approach ``swmm_output_parser.py``
    takes for SWMM ``.rpt`` files, whose encoding varies by platform): that ladder exists
    to recover MEANING from a known second encoding, and it still raises when both fail. Here the meaning lives in
    ASCII marker lines and the requirement is that nothing raises at all.

    Returns [] when the file is absent or unreadable -- the same answer a non-resumed exec
    gives, which every reader here already represents as an empty list.
    """
    try:
        text = model_logfile.read_text(errors="replace")
    except Exception:
        return []
    return harvest_resume_events(text)


def _refusal_row(line: str) -> dict | None:
    """The ledger row for one Row-4 line, or None for any other line.

    The row carries the same keys a resume row does, so every reader keeps one shape: no
    resume time (the launch resumed nothing), ``replay_reason_status`` ``not-applicable``,
    plus the solver's ``reason`` and ``target`` from the trailing bracket.
    """
    path = next((p for prefix, p in _REFUSAL_PATHS if prefix in line), None)
    if path is None:
        return None
    fields = {m.group("k"): m.group("v") for m in _BRACKET_FIELD_RE.finditer(line)}
    m = _CHECKPOINT_ID_RE.search(line)
    checkpoint_id = m.group("v") if m else None
    return {
        "path": path,
        "resume_time_s": None,
        "resume_time_status": "not-applicable",
        "checkpoint_id": checkpoint_id,
        "checkpoint_id_status": "parsed" if checkpoint_id is not None else "unparsed",
        "reason": fields.get("reason"),
        "target": fields.get("target"),
        "replay_reason": None,
        "replay_reason_status": "not-applicable",
        "replay_reason_raw": None,
        "raw": line.strip(),
    }


def stepback_confirmations(log_text: str) -> list[str]:
    """The ``from=`` checkpoint of every Row-5 line in one exec's log text, in file order.

    Row 5 is how the solver reports, from the directory alone, which unusable checkpoint the
    resume it just completed had to step back past. It adds no ledger row; its only use is
    to corroborate the toolkit picker's own record. A Row-5 line whose ``from=`` is missing
    contributes ``None`` rather than being dropped, so a malformed line still reads as
    "the solver reported a step-back" and is compared, not silently ignored.
    """
    out: list = []
    for line in log_text.splitlines():
        if STEPBACK_TAKEN_PREFIX in line:
            fields = {m.group("k"): m.group("v") for m in _BRACKET_FIELD_RE.finditer(line)}
            out.append(fields.get("from"))
    return out


def stepback_confirmations_from_logfile(model_logfile: Path) -> list[str]:
    """``stepback_confirmations`` over one sim's model log, never raising.

    Same read posture as ``harvest_from_logfile`` and for the same reason: the caller is the
    runner finishing a simulation, and an undecodable byte must not fail it.
    """
    try:
        text = model_logfile.read_text(errors="replace")
    except Exception:
        return []
    return stepback_confirmations(text)


def annotate_stepback(events: list[dict], record: dict | None, confirmations: list) -> list[dict]:
    """Attach the toolkit picker's step-back record to the exec's ``snapshot`` row.

    ``record`` is ``{unusable_step, marker, reason, used_step}`` as the picker left it, or
    None when the launch did not step back (then ``events`` is returned unchanged). The
    snapshot row gains ``stepback_from``, ``stepback_marker`` and ``stepback_reason``.

    THE PICKER IS THE SOURCE and Row 5 is a check on it. "Marker" in this docstring and in
    ``stepback_marker`` means a snapshot-write status file beside a checkpoint's snapshot
    (``.part``, ``.killed`` or ``.failed``; ``absent`` when there is neither file nor
    published snapshot) -- NOT a log literal like ``SNAPSHOT_MARKER``. The picker made the
    decision from the status files it saw, and Row 5 is re-derived by the solver later from
    the directory as it stands then. So when this exec's log carries a Row 5 line and its ``from=`` differs from
    the picker's ``unusable_step``, the row gains ``stepback_corroboration="mismatch"`` and
    the solver's value as ``stepback_solver_from``; a matching Row 5 adds nothing. No Row 5
    is NOT a mismatch: the solver emits Row 5 only when a marker survives above the restored
    checkpoint, so a step-back past a checkpoint with no marker normally carries none.

    Returns a new list; the input rows are not mutated. A launch that stepped back and then
    produced no ``snapshot`` row (it aborted, or fell back to replay) keeps no annotation in
    the ledger -- the picker's printed line in the runner log is then the only record.
    """
    if not record:
        return events
    idx = next((i for i in range(len(events) - 1, -1, -1) if events[i].get("path") == "snapshot"), None)
    if idx is None:
        return events
    row = {
        **events[idx],
        "stepback_from": record.get("unusable_step"),
        "stepback_marker": record.get("marker"),
        "stepback_reason": record.get("reason"),
    }
    if confirmations:
        solver_from = confirmations[-1]
        if str(solver_from) != str(record.get("unusable_step")):
            row["stepback_corroboration"] = "mismatch"
            row["stepback_solver_from"] = solver_from
    return [*events[:idx], row, *events[idx + 1 :]]


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

    ``attempt`` is the sim's persisted ``n_resumes`` as the runner reads it AFTER
    ``prepare_simulation_command`` has run -- so on a resumed exec it ALREADY COUNTS THIS
    ONE (``run_simulation.py``'s hotstart branch increments it inside
    ``prepare_simulation_command``, and ``run_simulation_runner.py`` re-reads it after that
    call returns, as the ``_n_done`` it passes here). Stated exactly because
    ``audit_resume_evidence`` below rests on it: after k cumulative resume ATTEMPTS,
    ``attempt`` is k, and a ledger in which every attempt left a solver line holds k rows,
    so the two are directly comparable and a shortfall counts attempts that left no line --
    the audit's docstring enumerates what can cause that. Deliberately NOT "as read BEFORE
    this exec": that describes a DIFFERENT number and would make that comparison off by
    one. The key name matches the ``_walltime`` ledger's own
    ``attempt`` so the two ledgers join on it.

    It also carries the only available stale-row signature: a member-scoped force-rerun
    deletes the member directory including ``log_{model_type}.json``, so ``n_resumes``
    restarts while this analysis-level ledger survives, and the attempt sequence goes
    non-monotone. ``audit_resume_evidence`` counts those restarts rather than leaving the
    inference to a reader nobody instructs.

    Deliberately NOT wired into the force-rerun deletion: ``analysis.py``'s
    ``_clean_restart_wipe`` already removes each ``model_*.log`` together with its
    ``_walltime/{stem}.jsonl`` sibling, and adding ``_resume_events`` there is one line
    inside that existing loop, so the mechanism EXISTS and is three lines away. It is
    DECLINED ON SCOPE: the change belongs to ``analysis.py``, not to this module. Read this
    as a scope decision with an owner, never as "no mechanism exists"; until that line is
    added, the audit's ``era_restarts`` is what makes the un-deleted ledger readable.
    """
    if not events:
        return "no-events"
    # EVERY step that can raise is inside the try, and the handler is total. Both halves
    # were measured wrong: `int(attempt)` and `json.dumps(row)` sat where an `except
    # OSError` could not reach them, so this function's own NEVER-RAISES docstring was
    # false of the public signature it exports in `__all__` -- `attempt="x"` raised
    # ValueError straight through a caller that had been told it could not. Widening the
    # clause rather than validating the inputs is deliberate: an observability ledger has
    # no input it would rather reject than record, so every failure mode has the same
    # correct disposition (lose the row, keep the simulation) and a total handler states
    # that once instead of enumerating causes it will not keep up with.
    try:
        stamped = [
            {
                **event,
                "attempt": int(attempt),
                "slurm_jobid": slurm_jobid,
                # UTC and offset-aware. A ledger is read on a different machine from the
                # one that wrote it, and a naive local timestamp is not orderable across
                # the two.
                "recorded_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            for event in events
        ]
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with open(ledger, "a") as handle:
            for row in stamped:
                handle.write(json.dumps(row) + "\n")
    except Exception:
        return "write-failed"
    return "appended"


def read_resume_events(ledger: Path) -> list[dict]:
    """Every durably-recorded resume event for one sim, oldest first.

    THIS IS THE READER, and its contract is to answer across execs -- which is the whole
    point of this module. Reading the model log answers for the last exec only.

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


#: The cause enumeration every evidence-gap summary carries, in the order worth checking.
#:
#: The abort cause leads because it is the one an operator reading the message would never
#: otherwise check: an attempt the solver aborted emits neither marker, so it is counted by
#: ``n_resumes`` and absent from the ledger exactly as a reworded marker would be. The
#: quoted ``Coupled resume aborted`` is the start of the solver's own abort line, named so
#: the operator knows what to grep the sim log for; this module never matches on it.
_SHORTFALL_CAUSES = (
    "Causes, in the order worth checking: every such attempt aborted before the solver "
    "could emit a resume marker (the sim log then carries 'Coupled resume aborted'), the "
    "solver reworded a resume marker, the ledger write failed, the model-log read "
    "degraded, or the sink was not reached."
)


def audit_resume_evidence(rows: list[dict], *, n_resumes: int) -> dict:
    """Compare what the ledger durably holds against how many times the sim resumed.

    THIS IS WHAT MAKES THE LEDGER READ IN PRODUCTION, and the comparison is the only
    drift detector available without a solver checkout.

    THE TWO OPERANDS COUNT DIFFERENT THINGS, and the audit is only sound when that is
    said. ``n_resumes`` counts resume ATTEMPTS: ``run_simulation.py``'s hotstart branch
    increments it when it hands the solver a ``config_NNNN.cfg``, BEFORE the solver is
    launched, so an attempt the solver then aborts is still counted. It is cumulative per
    ``(model_type, event)`` and never reset, and a launch that steps back to an earlier
    checkpoint is ONE attempt like any other. ``len(rows)`` counts what the solver's own
    lines actually produced across every exec -- one row per resume that COMPLETED (a
    snapshot restore or a replay), plus one row per launch the solver explicitly declared
    unusable when the pin emits such a line. So the two are equal only when every attempt
    left a line, and a shortfall is attempts the solver produced no line for.

    Five conditions produce a shortfall and a reader must surface all five, the first
    being the one this audit was found to omit: every such attempt ABORTED before the
    solver could emit a resume marker, in which case the solver's log carries
    ``Coupled resume aborted`` rather than either marker; the solver reworded a marker
    literal (the drift this module's two copied literals cannot otherwise detect -- a
    reword makes both copies agree and both wrong, and the suite stays green); the ledger
    write failed; the model-log read degraded; or the sink was never reached -- "the sink"
    being the runner's harvest-and-append step that writes this ledger after the solver
    exits, which the runtime summary names in that one word. The audit
    does NOT claim which -- it claims that a resume was attempted and left no durable
    evidence, which is the condition, and the five causes are the enumeration a reader
    starts from.

    ONE-SIDED BY CONSTRUCTION, and the asymmetry is load-bearing rather than a
    simplification. Only a shortfall (``recorded < expected``) is reported as a defect,
    because the one mechanism that makes ``recorded`` EXCEED ``expected`` runs the other
    way: a member-scoped force-rerun deletes ``log_{model_type}.json`` and restarts
    ``n_resumes`` while this analysis-level ledger survives, leaving a ledger that
    legitimately holds more rows than the live counter knows about. Treating that as a
    defect would fire on every force-rerun. It is reported separately as
    ``era_restarts``, counted from the ``attempt`` sequence going non-monotone, which is
    the signature ``append_resume_events``' docstring describes.

    THE VERDICT DISCRIMINATES THE TWO SHORTFALLS AN OPERATOR ACTS ON DIFFERENTLY.
    ``evidence-absent`` is a sim that attempted at least one resume and holds NO ledger row
    at all: no resume of it has ever left a line -- what a sim leaves behind when every
    attempt aborts and it makes no progress past its first checkpoint.
    ``evidence-shortfall`` is a sim some of whose attempts left a line and some did not.
    A machine consumer keys on the token, so one token for both would make "one of ten
    resumes left no row" and "no resume has ever worked" indistinguishable to it.

    ALARMS ARE REPORTED BESIDE THE VERDICT, never folded into it. Six row conditions are
    alarms: a ``step-back-required`` row (the solver had to refuse a checkpoint the toolkit
    picker should already have stepped back from -- a picker miss); a ``stop`` row and a
    ``clean-start`` row whose reason is ``write-failed`` (a snapshot write failed -- the
    fail-loudly class); a ``stepback_reason`` of ``write-failed``; a ``replay`` row whose
    reason is ``write-failed`` (the opt-in replay past a failed write); and a
    ``stepback_corroboration`` of ``mismatch`` (the solver's own account of the step-back
    disagrees with the picker's). Each of those rows is still one launch's event and is
    counted in ``recorded``, so it never manufactures a shortfall -- but counting it and
    saying nothing would hide it inside a ``complete`` verdict, which is why the alarm list
    is returned and named in the summary. A picker step-back row is a ``snapshot`` row: it
    is a completed resume and raises nothing.

    THE TWO OPERANDS ARE ALSO KEYED DIFFERENTLY. The ledger is keyed by the event's
    position (the model-log stem ``model_{m}_evt{event_iloc}``) while ``n_resumes`` is keyed
    by the scenario. After a multi-sim rerun that adds or removes events, a position can
    hold another event's rows, so this audit's verdict is valid only while event positions
    are unchanged since the rows were written. That positional-key defect is not repaired
    here.

    Deliberately NOT a raise and NOT a verdict the caller must act on. There are two
    production callers: the runner (``run_simulation_runner.py``), finishing a simulation
    that has already succeeded, and the post-completion check
    ``analysis_validation.check_resume_event_evidence``. This module's standing contract is
    that an observability ledger cannot fail a simulation, so the return is a record and
    each caller decides what to do with it -- the log level for the runner, the check
    result for the validator.

    PURE over ``rows`` -- it touches no filesystem -- so a test drives it with literals and
    a caller composes it with ``read_resume_events``. That split is what lets the audit be
    tested without a ledger and used without a second read.
    """
    expected = max(0, int(n_resumes or 0))
    recorded = len(rows)

    # A restart is an attempt value that fails to exceed its predecessor. Rows whose
    # attempt is absent or non-integral are skipped rather than coerced: a malformed row
    # is already represented as survivable everywhere else in this module, and guessing a
    # value here would manufacture a restart the ledger does not record.
    attempts = [row.get("attempt") for row in rows]
    ordered = [a for a in attempts if isinstance(a, int) and not isinstance(a, bool)]
    era_restarts = sum(1 for prev, cur in zip(ordered, ordered[1:], strict=False) if cur <= prev)

    shortfall = max(0, expected - recorded)
    if expected == 0:
        verdict = "no-resumes"
        summary = f"no resumes recorded for this sim; ledger holds {recorded} row(s)."
    elif recorded == 0:
        verdict = "evidence-absent"
        summary = (
            f"{expected} resume attempt(s) recorded in the model log and NO durable ledger "
            "row at all: no resume of this sim has ever left a line, so no resume is known "
            f"to have completed. {_SHORTFALL_CAUSES}"
        )
    elif shortfall:
        verdict = "evidence-shortfall"
        summary = (
            f"{expected} resume attempt(s) recorded in the model log but only {recorded} durable "
            f"ledger row(s): {shortfall} resume attempt(s) left NO surviving evidence. "
            f"{_SHORTFALL_CAUSES}"
        )
    else:
        verdict = "complete"
        summary = f"{recorded} durable ledger row(s) for {expected} resume attempt(s) -- complete."
    if era_restarts:
        summary += (
            f" Also: the attempt sequence restarts {era_restarts} time(s), so this ledger "
            "spans more than one force-rerun era."
        )
    alarms = [alarm for row in rows for alarm in _row_alarms(row)]
    if alarms:
        kinds = sorted({a["kind"] for a in alarms})
        summary += f" ALARM: {len(alarms)} alarm row condition(s) in this ledger ({', '.join(kinds)})."
    return {
        "verdict": verdict,
        "expected": expected,
        "recorded": recorded,
        "shortfall": shortfall,
        "era_restarts": era_restarts,
        "alarms": alarms,
        "summary": summary,
    }


def _row_alarms(row: dict) -> list[dict]:
    """The alarm conditions one ledger row carries, as ``{kind, path, checkpoint_id, detail}``.

    A row can carry more than one (a snapshot row whose step-back was both write-failed and
    uncorroborated), so this returns a list rather than a token. ``kind`` is one of three:
    ``picker-miss``, ``write-failed`` or ``corroboration-mismatch``. Rows that are not dicts
    carry none, because ``read_resume_events`` only ever returns dicts and a hand-built list
    that does not is not this function's to judge.
    """
    if not isinstance(row, dict):
        return []
    path = row.get("path")
    out: list[dict] = []

    def _alarm(kind: str, detail: str) -> None:
        out.append({"kind": kind, "path": path, "checkpoint_id": row.get("checkpoint_id"), "detail": detail})

    if path == "step-back-required":
        _alarm("picker-miss", "the solver refused a checkpoint the picker should have stepped back from")
    if path == "stop":
        _alarm("write-failed", "the restart stopped at a checkpoint whose snapshot write failed")
    if path == "clean-start" and row.get("reason") == "write-failed":
        _alarm("write-failed", "the solver refused a checkpoint whose snapshot write failed, nothing intact below it")
    if row.get("stepback_reason") == "write-failed":
        _alarm("write-failed", "the picker stepped back past a checkpoint whose snapshot write failed")
    if path == "replay" and row.get("replay_reason") == "write-failed":
        _alarm(
            "write-failed",
            "the resume replayed past a checkpoint whose snapshot write failed "
            "(swmm_snapshot_failed_replay); its numbers are correct, but the write failure is "
            "undiagnosed and may recur",
        )
    if row.get("stepback_corroboration") == "mismatch":
        _alarm(
            "corroboration-mismatch",
            f"the solver reported stepping back from checkpoint {row.get('stepback_solver_from')} "
            f"while the picker recorded {row.get('stepback_from')}",
        )
    return out
