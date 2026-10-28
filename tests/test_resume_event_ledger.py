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

# TWO FIXTURE TIERS, AND THE DISTINCTION IS LOAD-BEARING. The pair below is NOT the
# emitted shape and never was -- it is this module author's model of it, written before
# WP-1C chunk (5) landed, and it differs from what the solver writes in five places. It is
# RETAINED rather than replaced because it is the TOLERANCE arm: it proves the parser is
# not coupled to the solver's exact punctuation, which is what keeps a relabelling inside
# the accepted spelling family from silently zeroing the ledger. The REAL emitted shapes,
# reconstructed byte-for-byte from the solver source, are `SNAPSHOT_LINE_EMITTED` and
# `REPLAY_LINE_EMITTED` further down, and those are the pair that holds the parser against
# what it will actually process.
#
# A tolerance-arm line: marker literal, then the time as the first whitespace token (the
# rule `eda/raw_byte_identity.parse_resume_timestep` relies on), then the contract fields
# in a plausible-but-not-emitted punctuation.
SNAPSHOT_LINE = "[OK] SWMM state restored from snapshot to t=3000 s; checkpoint_id=0042"

# The tolerance arm's replay-path sibling, carrying the classified fallback reason.
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
    """This module's time parse must agree with `eda/raw_byte_identity.parse_resume_timestep`.

    That existing consumer is what constrains the emitted line shape in the first place, so
    a divergence here would mean two parsers disagreeing about the same line. The check
    RUNS the existing parser against the same text rather than restating its token rule --
    restating it would make the two agree by construction and prove nothing.
    """
    from hhemt.eda.raw_byte_identity import parse_resume_timestep

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


def _marker_family():
    """The solver log literals whose emitter is readable at ``TRITON_PIN``, as (name, value).

    The family is eleven bound constants: the two resume markers, the four snapshot
    step-back prefixes the ledger harvest reads, the three TRITON progress/completion
    markers, and the two EPA SWMM end-of-run clauses. It is a HAND enumeration, not a
    derivation, so a literal the toolkit reads can be missing from it while every test stays
    green; a new constant is a one-line addition here.

    The two EPA SWMM constants are checked against the TRITON fork's vendored copy, by proxy
    for the standalone build: the toolkit reads them only from a ``model_type == "swmm"`` run,
    whose log comes from the standalone ``runswmm`` cloned from ``SWMM_git_URL`` at
    ``SWMM_tag_key``, a program the coupled model never runs. The recorded measurement
    (``_SWMM_DEFAULT_MEASUREMENT`` below) covers the default (``SWMM_git_URL``, the commit the
    tag names, ``SWMM_tag_key``). Not established: a standalone SWMM built at a non-default
    ``SWMM_tag_key``, at ``SWMM_tag_key=None`` (the moving branch tip), or from a tag moved
    upstream after the recorded measurement.
    """
    from hhemt import analysis_validation as av
    from hhemt import run_simulation as rs

    return [
        ("resume_events.SNAPSHOT_MARKER", re_mod.SNAPSHOT_MARKER),
        ("resume_events.REPLAY_MARKER", re_mod.REPLAY_MARKER),
        ("resume_events.STEPBACK_REQUIRED_PREFIX", re_mod.STEPBACK_REQUIRED_PREFIX),
        ("resume_events.STOP_PREFIX", re_mod.STOP_PREFIX),
        ("resume_events.STEPBACK_IMPOSSIBLE_PREFIX", re_mod.STEPBACK_IMPOSSIBLE_PREFIX),
        ("resume_events.STEPBACK_TAKEN_PREFIX", re_mod.STEPBACK_TAKEN_PREFIX),
        ("analysis_validation._TRITON_COMPLETION_MARKER", av._TRITON_COMPLETION_MARKER),
        ("analysis_validation._TRITON_CHECKPOINT_READ_MARKER", av._TRITON_CHECKPOINT_READ_MARKER),
        ("analysis_validation._TRITON_CHECKPOINT_ATTEMPT_MARKER", av._TRITON_CHECKPOINT_ATTEMPT_MARKER),
        ("run_simulation.SWMM_COMPLETION_MARKER", rs.SWMM_COMPLETION_MARKER),
        ("run_simulation.SWMM_ERROR_MARKER", rs.SWMM_ERROR_MARKER),
    ]


def _solver_tree_at_pin():
    """(canonical store, pin) -- or a pytest skip when the store is unavailable BY POLICY.

    The availability predicate is the source-cache module's own: ``HHEMT_DISABLE_TRITON_CANONICAL=1``
    or ``CI`` set means "do not provision", and it is tested BEFORE ``ensure_canonical`` is
    called, because that function has no such guard and would otherwise clone the solver
    fork on a CI runner. Any OTHER failure to provision raises, so the test fails rather than
    skips: a drift detector that silently skips where it cannot run reports nothing.
    """
    import os
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures._triton_source_cache import _DISABLE_ENV, TRITON_PIN, ensure_canonical

    if os.environ.get(_DISABLE_ENV) == "1":
        pytest.skip(f"{_DISABLE_ENV}=1: the solver canonical store is disabled, so the pin cannot be read here")
    if os.environ.get("CI"):
        pytest.skip("CI is set: the solver canonical store is not provisioned on CI; the pin-advance gate runs this")
    return ensure_canonical(pin=TRITON_PIN), TRITON_PIN


def _occurrences_at_pin(canonical, pin, literal):
    """Lines of solver source at ``pin`` carrying ``literal``, over the TRITON and vendored SWMM sources."""
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(canonical), "grep", "-F", "-c", "-e", literal, pin, "--", "src", "external/swmm/src"],
        capture_output=True,
        text=True,
    )
    return sum(int(line.rsplit(":", 1)[1]) for line in out.stdout.splitlines() if line.strip())


def test_every_toolkit_marker_constant_is_emitted_at_the_pin():
    """Property: every member of ``_marker_family`` exists in the solver source at ``TRITON_PIN``.

    Class: the eleven-member marker family in ``_marker_family``, each read against the solver's
    own bytes at the pinned commit through the toolkit's canonical object store, over ``src``
    and ``external/swmm/src``. The family is not every literal the toolkit reads, and its two
    EPA SWMM members are read here by proxy; ``_marker_family``'s docstring states both limits.

    Why required, checkable against the code: the drift detector above compares a toolkit copy
    with a toolkit copy, so a solver-side rename leaves every copy byte-equal, this suite green,
    and every consumer matching nothing -- a completion check that never completes, or a resume
    audit that sees no resume. Reading the pin is the only comparison a solver rename can fail.

    What kills it: a constant whose value no longer occurs in the solver source at the pin --
    a toolkit-side typo or a solver-side rename that the pin advance carried in.

    A second correct implementation under which it still passes: one that shells out to
    ``git show {pin}:{path}`` per known file instead of ``git grep`` over the two source trees;
    the assertion reads only whether each literal occurs.
    """
    canonical, pin = _solver_tree_at_pin()
    missing = [name for name, value in _marker_family() if _occurrences_at_pin(canonical, pin, value) == 0]
    assert missing == [], f"marker constants absent from the solver source at TRITON_PIN {pin[:12]}: {missing}"


# The one-time recorded measurement of the standalone SWMM program at the configured
# defaults, taken on 2026-10-18T01:23:47Z (UTC) on UVA Rivanna scratch, because the
# vendored copy the pin test above reads is a proxy for that program, not the program.
# How it was taken, so it can be repeated when the tripwire below reds:
#   git clone --depth 1 --branch {SWMM_tag_key} {SWMM_git_URL} {clone}
#   git -C {clone} rev-parse HEAD^{commit}          -> the commit recorded below
#   grep -nF "EPA SWMM completed" {clone}/src/run/main.c
#     91:        printf("\n\n... EPA SWMM completed in %.2f seconds.", runTime);
#   grep -nF "There are errors." {clone}/src/run/main.c
#     92:        if      ( swmm_getError(errMsg, msgLen) > 0 ) printf(" There are errors.\n");
# The grep is scoped to src/run/main.c, the print site the toolkit's log read depends on:
# src/solver/text.h also carries " There are errors." (as #define FMT03), so a match
# anywhere under src/ would pass even if main.c stopped printing the clause. USEPA's
# v5.2.4 is a lightweight tag, which is why the commit is read with rev-parse on the
# clone rather than from a peeled ls-remote line (that line does not exist for it).
_SWMM_DEFAULT_MEASUREMENT = (
    "https://github.com/USEPA/Stormwater-Management-Model.git",
    "7952ca837988b1c32f791812eccc9fd64547e093",
    "v5.2.4",
)


def test_the_swmm_defaults_are_the_ones_the_recorded_measurement_covers():
    """Property: the configured ``SWMM_git_URL`` and ``SWMM_tag_key`` defaults equal the recorded measurement's.

    Class: the two ``system_config`` field defaults that pick the standalone SWMM program
    whose log the toolkit reads for ``"EPA SWMM completed"`` and ``"There are errors."``.

    Why required, checkable against the code: the parity test above reads those two
    literals in the TRITON fork's vendored SWMM, not in the standalone program a
    ``model_type == "swmm"`` run executes. ``_SWMM_DEFAULT_MEASUREMENT`` records that the
    standalone program at the defaults prints both. If either default moves, that record
    no longer describes the program being built, and ``"There are errors."`` is a NEGATIVE
    conjunct of the completion read, so a reworded clause would let a failed run read
    complete. This test reds on the move and forces the measurement to be retaken. It also
    covers the container path, whose SWMM tag default is derived from the same field default.

    What kills it: changing either field default in ``config/system.py`` without retaking
    the measurement and updating ``_SWMM_DEFAULT_MEASUREMENT``.

    A second correct implementation under which it still passes: one that builds a
    ``system_config`` and reads the two attributes off the instance instead of reading
    ``model_fields``; the assertion reads only the two default values.

    Not established: that the recorded commit is still what ``SWMM_tag_key`` names upstream
    -- a tag moved after the measurement passes this test.
    """
    from hhemt.config.system import system_config

    url, _commit, tag = _SWMM_DEFAULT_MEASUREMENT
    assert system_config.model_fields["SWMM_git_URL"].default == url
    assert system_config.model_fields["SWMM_tag_key"].default == tag


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


# --------------------------------------------------------------------------------------
# The REAL emitted shapes, reconstructed byte-for-byte from the solver source
# --------------------------------------------------------------------------------------
#
# WHY THESE EXIST BESIDE THE TWO ABOVE. The two fixtures above were written BEFORE WP-1C
# chunk (5) landed and encode this module author's model of the emitted line. That model
# differs from what the solver writes in five places -- the ANSI-wrapped `[..]` prefix
# rather than a bare `[OK]`, the label `checkpoint=` rather than `checkpoint_id=`, a
# trailing ` [resume-event ...]` bracket rather than a `;`, a `path=` token, and a
# `(N steps ...); resuming live segment` clause. A suite holding only the guessed shape
# stays green against a parser that fails on every real line, which is the oracle gap
# these close.
#
# Reconstructed from the solver source at the toolkit's `TRITON_PIN`
# (`tests/fixtures/_triton_source_cache.py`), named rather than given as a sha so this
# comment follows the pin instead of going stale beside it:
#   src/swmm_triton.h  -- the snapshot arm (`SWMM state restored from snapshot to t=`)
#                         and the replay arm (`SWMM exchange history replayed to t=`)
#   src/constants.h    -- GRAY "\033[90m", RESET "\033[0m", IN = GRAY "[..] " RESET
# `test_every_toolkit_marker_constant_is_emitted_at_the_pin` below reads the markers at the
# pin itself; these hand-built lines are not that check.
# `up_to_time` is a `value_t` (double) rendered by the default ostream formatter, so an
# integral value prints without a decimal point; `rec_count` is a `long`.
_IN = "\x1b[90m[..] \x1b[0m"
SNAPSHOT_LINE_EMITTED = (
    _IN + "SWMM state restored from snapshot to t=3600 s (11435 steps skipped);"
    " resuming live segment [resume-event checkpoint=42 path=snapshot]"
)
REPLAY_LINE_EMITTED = (
    _IN + "SWMM exchange history replayed to t=3600 s (11435 steps);"
    " resuming live segment [resume-event checkpoint=42 path=replay reason=absent]"
)


@pytest.mark.parametrize(
    "line,expected_path,expected_reason",
    [
        (SNAPSHOT_LINE_EMITTED, "snapshot", None),
        (REPLAY_LINE_EMITTED, "replay", "absent"),
    ],
)
def test_the_real_solver_emitted_shapes_parse(line, expected_path, expected_reason):
    """Property: all four contract fields parse out of the line the SOLVER actually emits.

    Class: NEW CAPABILITY. The plausible wrong implementation is a `_CHECKPOINT_ID_RE`
    tightened to the spelling the guessed fixtures use -- dropping the bare `checkpoint`
    alternative and keeping only `checkpoint[_ ]?id|ckpt|cp` -- which is exactly the edit
    an author checking the regex against this module's own older fixtures would make.
    Under it the real line's `checkpoint=42` matches nothing (`checkpoint` contains no
    `cp` or `ckpt` substring), `checkpoint_id` is None, and this test reds while every
    guessed-shape test stays green.

    Why required, as a consequence a reader can check against the code: the harvest admits
    a line by SUBSTRING containment of a marker and then parses three fields out of
    whatever surrounds it. Every one of those three parses meets solver-authored text this
    suite had never seen -- the ANSI prefix precedes the marker, and the bracket, the
    `path=` token and the `steps skipped` clause all follow it. A green suite over guessed
    text establishes nothing about any of them.

    What kills it: narrowing the checkpoint label family, anchoring the marker match to
    the start of the line (the ANSI prefix would then defeat it), or scoping the reason
    scan to anything that excludes the trailing bracket.

    A second correct implementation under which it still passes: one that parses the
    trailing `[resume-event ...]` bracket as a `key=value` map and reads `checkpoint` and
    `reason` out of it, ignoring the rest of the line entirely. That differs from the
    current whole-line regex in the thing this test asserts on and returns the same four
    fields.
    """
    (ev,) = re_mod.harvest_resume_events(line)
    assert ev["path"] == expected_path
    assert ev["resume_time_s"] == 3600.0
    assert ev["resume_time_status"] == "parsed"
    assert ev["checkpoint_id"] == "42"
    assert ev["checkpoint_id_status"] == "parsed"
    assert ev["replay_reason"] == expected_reason
    assert ev["raw"] == line


def test_a_snapshot_line_carrying_a_reason_word_gets_no_reason():
    """Property: the reason field is a function of the PATH alone, never of other line text.

    Input class: a snapshot-marker line that ALSO contains a bare token from
    `REPLAY_REASONS`. The guard at `resume_events.py`'s reason block is the module's one
    deliberate, documented safety decision, and it is the only thing standing between that
    class and a fabricated fallback reason for a run that never fell back.

    Class: NEW CAPABILITY. The plausible wrong implementation is the guard removed --
    scanning for a reason on every line regardless of path. Under it this test reds with
    `replay_reason == 'retention-collision'`; every pre-existing snapshot fixture stays
    green, because each occupies a satisfying position the guarded and unguarded code
    agree on, which is why the guard shipped unheld.

    Why required, as a consequence a reader can check against the code: the reason scan is
    whole-line and word-bounded, and the three reason tokens are not reserved vocabulary --
    `absent` is an ordinary English word and the other two are hyphenated identifiers that
    can appear in an operator-chosen snapshot stem or a diagnostic. The markers also reach
    one shared file descriptor from every rank (the runner opens the model log once and
    passes it as both stdout and stderr), so two ranks' partial writes can land on one
    physical line. Whatever the route, a fabricated reason is worse than an absent one: it
    is a field a reader will act on that says a run fell back when it did not.

    What kills it: removing the path conditioning, or deriving `path` from a `path=` token
    in the text rather than from which marker literal matched.

    A second correct implementation under which it still passes: one that reads the reason
    only from a `reason=` key inside the trailing `[resume-event ...]` bracket, on either
    path. The stray token here is outside any such bracket, so that implementation also
    returns None, and it differs from the current one in exactly the thing asserted on.
    """
    line = (
        _IN + "SWMM state restored from snapshot to t=3600 s (11435 steps skipped);"
        " resuming live segment [resume-event checkpoint=42 path=snapshot]"
        " (prior attempt logged retention-collision)"
    )
    (ev,) = re_mod.harvest_resume_events(line)
    assert ev["path"] == "snapshot"
    assert ev["replay_reason"] is None, "a reason-shaped word on a snapshot line manufactured a fallback reason"
    assert ev["replay_reason_status"] == "not-applicable"


# --------------------------------------------------------------------------------------
# Totality: the ledger cannot fail a finished simulation
# --------------------------------------------------------------------------------------


def test_harvest_from_logfile_survives_an_undecodable_byte(tmp_path):
    """Property: an undecodable byte costs its own character and NOT the exec's evidence.

    Class: NEW CAPABILITY. The plausible wrong implementation is the runner's pre-repair
    form lifted verbatim -- `harvest_resume_events(model_logfile.read_text())`. Under it
    this test reds with `UnicodeDecodeError`, which is a `ValueError`: in the runner that
    escaped a handler written as `except OSError`, reached `main()`'s outer `except
    Exception`, wrote `_status/_failed/{rule_token}.json` and returned 1 -- recording a
    FINISHED simulation as FAILED. A second wrong implementation, `try: ... except:
    return []`, also reds here, and that is the point of asserting on the harvested event
    rather than on the absence of a raise.

    Why required, as a consequence a reader can check against the code: the model log is
    a merged stdout+stderr stream from a whole HPC command stack, and this codebase
    already treats solver-log decode failure as a real handled condition elsewhere
    (`swmm_output_parser.py` tries UTF-8 then CP-1252). Both marker lines are pure ASCII,
    so there is no reason for one bad byte anywhere else in the file to cost the resume
    evidence the ledger exists to preserve.

    What kills it: reading without `errors="replace"`, or handling the decode failure by
    returning an empty list.

    A second correct implementation under which it still passes: reading the file as bytes
    and decoding with `latin-1`, which is total over every byte sequence and leaves the
    ASCII marker line identical. It differs from the current implementation in the decode
    strategy, which is the thing this test asserts on.
    """
    log = tmp_path / "model_tritonswmm_evt0.log"
    log.write_bytes(b"preamble \xff\xfe garbage\n" + SNAPSHOT_LINE_EMITTED.encode() + b"\nSimulation ends\n")
    (ev,) = re_mod.harvest_from_logfile(log)
    assert ev["path"] == "snapshot"
    assert ev["resume_time_s"] == 3600.0
    assert ev["checkpoint_id"] == "42"


def test_harvest_from_logfile_returns_empty_for_an_absent_log(tmp_path):
    """An absent log is the same answer as 'no resume events', never a raise."""
    assert re_mod.harvest_from_logfile(tmp_path / "nope.log") == []


def test_append_never_raises_on_a_non_integral_attempt(tmp_path):
    """Property: NO argument to the public append can raise; a bad one costs the row only.

    Class: REGRESSION -- red on the pre-change tree with `ValueError: invalid literal for
    int() with base 10: 'x'`, because `int(attempt)` was evaluated OUTSIDE the try whose
    handler was `except OSError`.

    Why required, as a consequence a reader can check against the code: this function is
    exported in `__all__` and its own docstring promises it NEVER raises, and the module
    is deliberately a leaf so that a future validator or renderer can import it. A
    documented promise the code does not keep is worse than no promise, because the
    caller writes no handler.

    What kills it: moving any raising step back outside the try, or narrowing the handler
    to an exception tuple.

    A second correct implementation under which it still passes: one that validates
    `attempt` up front and returns `"write-failed"` on a non-integral value without
    entering the try at all. It differs from the current total-handler approach in exactly
    the mechanism asserted on and returns the same token.
    """
    ledger = tmp_path / "_resume_events" / "x.jsonl"
    events = re_mod.harvest_resume_events(SNAPSHOT_LINE_EMITTED)
    assert re_mod.append_resume_events(ledger, events, attempt="x") == "write-failed"


# --------------------------------------------------------------------------------------
# Chunk (3): the reader, and the drift signal it computes
# --------------------------------------------------------------------------------------


def _rows(*attempts):
    return [{"path": "replay", "attempt": a} for a in attempts]


def test_the_audit_reports_a_shortfall_when_a_resume_left_no_durable_row():
    """Property: a sim that resumed against a ledger holding fewer rows is a loud defect.

    Input class: `n_resumes >= 1` with strictly fewer durable ledger rows. That is the
    marker-drift signal, and it is the only one available without a solver checkout: a
    solver-side reword makes BOTH in-tree copies of the literal agree and both wrong, the
    harvest silently returns nothing, and no literal-vs-literal assertion anywhere can
    see it. This arithmetic can.

    Class: NEW CAPABILITY. The plausible wrong implementation is an audit that compares
    `recorded` against `len(rows)` or against zero -- any predicate not keyed on hhemt's
    own `n_resumes` -- under which a ledger of zero rows for a sim that resumed twice
    reports healthy. Under it this test reds.

    Why required, as a consequence a reader can check against the code: the solver emits
    exactly one marker per resumed exec, and `n_resumes` is incremented by hhemt in a
    repository the solver cannot reach, so the two counts are independent measurements of
    the same event and a divergence has no benign reading in this direction.

    What kills it: comparing on the wrong side of the inequality, dropping the `expected`
    operand, or returning a non-defect verdict when `shortfall > 0`.

    A second correct implementation under which it still passes: one that returns a
    boolean `ok` plus a cause list instead of a verdict token, computed from the same two
    counts. It differs in the return shape, which is not what this assertion turns on --
    the shortfall arithmetic is.
    """
    a = re_mod.audit_resume_evidence(_rows(1), n_resumes=3)
    assert a["verdict"] == "evidence-shortfall"
    assert (a["expected"], a["recorded"], a["shortfall"]) == (3, 1, 2)
    assert "NO surviving evidence" in a["summary"]


def test_the_audit_is_complete_when_every_resume_left_a_row():
    """The healthy case: equal counts are `complete`, never a shortfall."""
    a = re_mod.audit_resume_evidence(_rows(1, 2, 3), n_resumes=3)
    assert a["verdict"] == "complete"
    assert a["shortfall"] == 0
    assert a["era_restarts"] == 0


def test_an_era_restart_is_counted_and_is_not_a_shortfall():
    """Property: the force-rerun era signature is reported, and never as a defect.

    Input class: a ledger holding MORE rows than the live `n_resumes` knows about, with a
    non-monotone attempt sequence. A member-scoped force-rerun deletes
    `log_{model_type}.json` and restarts `n_resumes` while this analysis-level ledger
    survives, so this is the one direction in which the two counts legitimately diverge.

    Class: NEW CAPABILITY. The plausible wrong implementation is a symmetric audit --
    `recorded != expected` is a defect -- which is the obvious generalisation of the
    shortfall rule and which would fire on every force-rerun, training an operator to
    ignore the signal that matters.

    Why required, as a consequence a reader can check against the code: the shortfall rule
    is one-sided BY CONSTRUCTION and nothing in the arithmetic says so. Without this test
    a later author reads `max(0, expected - recorded)` as a convenience and symmetrises it.

    What kills it: making the audit symmetric, or counting a restart from a strictly
    increasing sequence.

    A second correct implementation under which it still passes: one that detects the era
    boundary from the rows' `recorded_at` timestamps going backwards rather than from the
    attempt sequence. It differs in the detection mechanism and returns the same counts on
    this input.
    """
    a = re_mod.audit_resume_evidence(_rows(1, 2, 3, 1, 2), n_resumes=2)
    assert a["verdict"] == "complete"
    assert a["shortfall"] == 0
    assert a["era_restarts"] == 1
    assert "more than one force-rerun era" in a["summary"]


def test_a_never_resumed_sim_is_no_resumes_not_a_shortfall():
    """A sim that never resumed is `no-resumes`, which is the audit's third verdict.

    Without this arm the shortfall rule has no stated behaviour at `expected == 0`, and
    the natural wrong reading -- that zero rows is always a defect -- would fire a WARNING
    on every fresh, never-resumed exec, which is the overwhelming majority of them.
    """
    a = re_mod.audit_resume_evidence([], n_resumes=0)
    assert a["verdict"] == "no-resumes"
    assert a["shortfall"] == 0


def test_the_runner_reads_the_ledger_in_production():
    """Property: the reader is WIRED, not merely defined -- chunk (3)'s own condition.

    Class: REGRESSION -- red on the pre-change tree, where `read_resume_events` existed
    and was called only from tests, making WP-2D write-only in production. Output pasted
    in the coder round: `AssertionError: the runner never reads the ledger -- WP-2D is
    write-only`.

    Why required, as a consequence a reader can check against the code: every other test
    in this module drives `resume_events` directly, so all of them stay green against a
    runner that never calls the reader. The drift detection the audit computes is worth
    nothing unless something in production runs it, and nothing else in this suite can
    tell the difference.

    What kills it: deleting either call from the runner's ledger block.

    A second correct implementation under which it still passes: a runner that routes both
    through a module-level helper in this same package -- the assertion is over the call
    graph reachable from the runner's source, which an aliased import still satisfies, not
    over a particular statement shape. A reader relocated to `analysis_validation.py`
    instead WOULD red this, and correctly so: that is a different §9.1 pair-row outcome
    and the test is the record of which one shipped.

    The consumer of the asserted value outside this test is the shape's WP-2D chunk (3)
    and its §9.1 pair-row consequence: the sink-only form is the form in which no pair row
    moves, and this assertion is what pins the package to it.
    """
    import ast

    import hhemt

    src = (Path(hhemt.__file__).parent / "run_simulation_runner.py").read_text()
    called = {
        node.func.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "read_resume_events" in called, "the runner never reads the ledger -- WP-2D is write-only"
    assert "audit_resume_evidence" in called, "the runner never audits the ledger against n_resumes"
