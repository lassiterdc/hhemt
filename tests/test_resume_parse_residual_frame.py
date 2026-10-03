"""The SAMPLE FRAME of the three-parser residual characterization, measured.

WHY THIS MODULE EXISTS. `tests/test_resume_parse_residual_disagreement.py` records the
residual disagreement among the three in-tree resume-time parses and states two
completeness claims in its own prose: that the 2026-10-03 repair "closed the EMPTY-TAIL
class across all three implementations", and (in
`test_the_disagreement_set_is_token_versus_regex_only`) that if a future row appears "in
which the two TOKEN implementations differ from each other, the class has regressed in a
different and worse direction, and this test names that".

BOTH CLAIMS ARE TRUE OVER THAT MODULE'S FIXTURE FAMILY AND NEITHER IS TRUE OF THE CLASS,
because the fixture family holds two dimensions constant that the implementations do not.
As of 2026-10-03 the sibling's own prose carries that scoping and cites this module for the
two held-constant dimensions, so the frame is stated where the claims are made and measured
here; the paragraphs below remain the measurement rather than a second assertion of it.
Measured at `5c2ef5a3`:

  * TAIL SCOPE. `eda/raw_resume_identity.parse_resume_timestep` takes its tail to END OF
    FILE (`text.rsplit(marker, 1)[1]`); `resume_events.harvest_resume_events` takes it to
    END OF LINE (`line.split(MARKER, 1)[1]`) before handing it to `_parse_leading_float`.
    Every in-tree fixture builds a SINGLE-LINE text, where file tail == line tail, so the
    two are indistinguishable there. On a multi-line log whose marker ends its own line the
    two token rules return DIFFERENT answers, and the file-scoped one returns a number read
    off a LATER LINE -- a wrong answer wearing a right shape, which is strictly worse than
    the None the line-scoped one returns.
  * THE NON-FINITE FAMILY. `float()` accepts `nan`, `-nan`, `NaN`, `inf`, `-inf` and
    `Infinity`; the regex char class `([-+0-9.eE]+)` can match none of them. The landed
    `_RESIDUAL` records `inf s` and calls it "the one row whose divergence could reach a
    durable artifact", and omits its five neighbours -- which a diverged `value_t` streamed
    with the default `operator<<` emits by exactly the same mechanism.

PRODUCTION REACH, measured, because it bounds how much any of this matters and because it
corrects a reachability sentence in the sibling module. `parse_resume_timestep` has ZERO
production call sites: `grep -rn 'parse_resume_timestep' --include=*.py src/ scripts/` at
`5c2ef5a3` returns one definition and three DOCSTRING mentions, and nothing else. The b4b
figure's resume boundary comes from `resume_boundaries_from_schedule`
(`raw_resume_identity.py:706`), which that function's own docstring describes as
"Model-agnostic and K-complete, unlike `parse_resume_timestep`"; the durable stamp's
`resume_t` comes from `processing_analysis._parse_replay_t` (`processing_analysis.py:1024`,
the REGEX rule). So the sibling module's claim that `inf` "is the one row whose divergence
could reach a durable artifact: the stamp written by `processing_analysis` would carry
`resume_t: null` while the figure's parser reads `inf`" was wrong in its second clause -- the
figure does not call that parser, so there is no second reader for the stamp to disagree
with. THAT SENTENCE WAS REPAIRED IN THE SIBLING ON 2026-10-03 and now carries this same
measurement; the quoted wording is retained there as the superseded claim, so a reader
arriving from here finds the two modules in agreement rather than in contradiction. The
correction is recorded in both places deliberately -- this module measured it, and the
sibling is where the wrong sentence was read.
CONSEQUENCE FOR READING THIS MODULE: every divergence recorded here is a property of
the FUNCTIONS and none of them is a live wrong answer today. They matter because
`parse_resume_timestep` is under contract from six test modules and would be the natural
thing to wire in when the figure needs a MEASURED rather than a REQUESTED boundary -- which
is exactly when a wrong answer read off a later line would start reaching an artifact.

THIS MODULE IS A CHARACTERIZATION OF THE FRAME, NOT A REPAIR. Nothing here asserts a
behaviour anyone has ruled on. It records what the two held-constant dimensions produce, so
a change to either token rule -- including a legitimate collapse of the two onto one
primitive -- goes red HERE rather than staying green on a pair that genuinely disagrees.

WHAT A GREEN HERE DOES AND DOES NOT SAY. It says the three parses disagree in exactly the
ways measured over `_CORPUS` below. It does NOT say `_CORPUS` is the input space: the
completeness test is explicitly scoped to that declared list, which is the honest form of a
claim no finite corpus can make unconditionally.

COMPILE-FREE AND ANALYSIS-FREE BY CONSTRUCTION: no `*_compiled` fixture, no
`TRITONSWMM_analysis`. Screened by fixture closure, never by module text.
"""

from __future__ import annotations

import pytest

import hhemt.analysis_validation as av
from hhemt.eda.raw_resume_identity import parse_resume_timestep
from hhemt.processing_analysis import _parse_replay_t
from hhemt.resume_events import _parse_leading_float, harvest_resume_events
from tests.test_resume_parse_residual_disagreement import _RESIDUAL

_REPLAY = av._TRITON_REPLAY_MARKER
_SNAPSHOT = av._TRITON_SNAPSHOT_RESTORE_MARKER


# -------------------------------------------------------------------------------------
# A. TAIL SCOPE -- the dimension every in-tree fixture holds constant.
# -------------------------------------------------------------------------------------


def _token_file(tmp_path, text: str):
    """The production file-scoped token rule, called the way the b4b figure calls it."""
    log = tmp_path / "model_tritonswmm_evt0.log"
    log.write_text(text)
    return parse_resume_timestep(log)


def _token_line(text: str):
    """The production LINE-scoped token rule, read off `harvest_resume_events`' own rows.

    Deliberately routed through the real caller rather than by hand-slicing a tail: the
    hand-sliced form is what the sibling module's `_three` helper does, and hand-slicing a
    WHOLE-TEXT tail is itself the substitution that makes the two rules look identical.
    """
    events = harvest_resume_events(text)
    return events[-1]["resume_time_s"] if events else None


#: (label, text, file-scoped token value, line-scoped token value, regex value)
#: Measured at 5c2ef5a3. A single-line row is included FIRST as the positive control: if
#: the two scopes ever stop agreeing there, this table is measuring the wrong thing.
_TAIL_SCOPE = [
    (
        "single-line well-formed -- the only shape any in-tree fixture builds",
        f"[..] {_REPLAY}3600 s (12 steps)\n",
        3600.0,
        3600.0,
        3600.0,
    ),
    (
        "marker ends its line; the NEXT line begins with a number",
        f"[..] {_REPLAY}\n4242.0 s (continuation of some other record)\n",
        4242.0,
        None,
        4242.0,
    ),
    (
        "marker ends its line; the next line begins non-numeric",
        f"[..] {_REPLAY}\n[..] something 42\n",
        None,
        None,
        None,
    ),
    (
        "marker ends the FILE -- the only empty-tail position the sibling module samples",
        f"[..] {_REPLAY}\n",
        None,
        None,
        None,
    ),
    (
        # NOT a token-vs-token row: both token rules answer 7.0 here, because
        # `harvest_resume_events` scans for BOTH marker literals and the snapshot line is
        # the LAST event, while the file-scoped rule reaches the same number through its
        # fixed marker-tuple order. Kept because it is the row on which the REGEX rule
        # alone is silent, and because an earlier hand-sliced probe of mine reported None
        # for the line scope by filtering to replay lines only -- the production caller is
        # the only instrument that gets this row right.
        "marker ends its line; the next line is the OTHER marker carrying a number",
        f"[..] {_REPLAY}\n[..] {_SNAPSHOT}7 s\n",
        7.0,
        7.0,
        None,
    ),
]


@pytest.mark.parametrize(
    "label,text,want_file,want_line,want_regex",
    [pytest.param(*row, id=row[0][:48]) for row in _TAIL_SCOPE],
)
def test_tail_scope_is_the_dimension_the_single_line_fixtures_cannot_express(
    tmp_path, label, text, want_file, want_line, want_regex
):
    got_file, got_line, got_regex = _token_file(tmp_path, text), _token_line(text), _parse_replay_t(text, _REPLAY)
    assert repr(got_file) == repr(want_file), f"file-scoped token rule moved: {label}"
    assert repr(got_line) == repr(want_line), f"line-scoped token rule moved: {label}"
    assert repr(got_regex) == repr(want_regex), f"regex rule moved: {label}"


def test_the_two_token_rules_are_extensionally_UNEQUAL_off_the_single_line_family():
    """The claim the sibling module's structural test promises to name, and cannot.

    `test_the_disagreement_set_is_token_versus_regex_only` iterates ONLY its own
    `_RESIDUAL` rows, every one of which is a single-line text, so the condition it
    undertakes to detect -- the two TOKEN implementations differing from each other --
    already holds and that test is green. This test is where it is visible.

    Positive control first: the two rules MUST agree on the single-line family, or this
    test is measuring a broken helper rather than a real divergence.
    """
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())

    def pair(text):
        log = d / "model_tritonswmm_evt0.log"
        log.write_text(text)
        return parse_resume_timestep(log), _token_line(text)

    agree = pair(f"[..] {_REPLAY}3600 s (12 steps)\n")
    assert agree[0] == agree[1] == 3600.0, f"positive control failed: {agree}"

    disagree = pair(f"[..] {_REPLAY}\n4242.0 s (continuation of some other record)\n")
    assert disagree != (None, None), "positive control failed: neither rule answered at all"
    assert disagree[0] != disagree[1], (
        "the two token rules now AGREE on a marker-ends-its-line text -- if that is a "
        "deliberate repair (one rule delegating to the other, or the file-scoped rule "
        "narrowed to its own line) delete this test and the _TAIL_SCOPE rows it guards"
    )
    # And name WHICH way, because the direction is the whole point: the file-scoped rule
    # answers with a number it read off a LATER LINE.
    assert disagree == (4242.0, None), disagree


# -------------------------------------------------------------------------------------
# B. THE NON-FINITE FAMILY -- disagreeing tails the landed _RESIDUAL omits.
# -------------------------------------------------------------------------------------

#: (tail, token-rule value, regex-rule value). Same mechanism as the landed `inf s` row:
#: `float()` accepts the literal, the regex char class cannot match an alphabetic token.
#: `-inf` is in the IDENTICAL reachability class as the `inf` row the sibling module flags
#: as the one that reaches a durable artifact, and `nan`/`-nan` are what a diverged
#: `value_t` streamed with the default `operator<<` emits most readily.
_NONFINITE = [
    ("nan s", float("nan"), None),
    ("-nan s", float("nan"), None),
    ("NaN s", float("nan"), None),
    ("-inf s", float("-inf"), None),
    ("Infinity s", float("inf"), None),
]


@pytest.mark.parametrize("tail,want_token,want_regex", _NONFINITE)
def test_the_non_finite_family_disagrees_and_is_absent_from_the_landed_residual(tmp_path, tail, want_token, want_regex):
    text = f"[..] {_REPLAY}{tail}\n"
    token = _token_file(tmp_path, text)
    regex = _parse_replay_t(text, _REPLAY)
    leaf = _parse_leading_float(text.split(_REPLAY, 1)[1])
    assert repr(token) == repr(want_token), f"token rule moved on {tail!r}"
    assert repr(regex) == repr(want_regex), f"regex rule moved on {tail!r}"
    assert repr(leaf) == repr(token), f"the two token rules diverged on {tail!r}"
    assert tail not in {row[0] for row in _RESIDUAL}, (
        f"{tail!r} is now recorded in the landed _RESIDUAL set -- the frame gap this module "
        "exists to name has been closed there; delete this row"
    )


# -------------------------------------------------------------------------------------
# C. COMPLETENESS, scoped to a DECLARED corpus rather than claimed unconditionally.
# -------------------------------------------------------------------------------------

#: The tail corpus this module's completeness claim ranges over. It is NOT the input space
#: and no test here says it is. It is the union of: the shapes the solver emits (read from
#: the carrier's own family), the landed residual rows, the non-finite family, and the
#: empty-tail class.
_CORPUS = [
    "3600 s (12 steps)",
    "6000 s (11435 steps skipped)",
    "6e+06 s (1 steps)",
    "1.5e3 s",
    "1.5E3 s",
    "-1 s",
    "+2.5 s",
    "0 s",
    ".5 s",
    "1e-3 s",
    "3600",
    "3600.",
    "3600,",
    "3600;",
    "  3600 s",
    "not_a_number s",
    "",
    " ",
    "\t",
    "  \t ",
    *[row[0] for row in _RESIDUAL],
    *[row[0] for row in _NONFINITE],
]


def test_the_disagreement_set_over_the_declared_corpus_is_exactly_the_two_recorded_tables(tmp_path):
    """Every disagreeing tail in `_CORPUS` is recorded in `_RESIDUAL` or in `_NONFINITE`.

    This is the guard that makes a NEW disagreement visible somewhere. It reds in both
    directions: a newly-disagreeing corpus member that neither table records, and a
    recorded member that has stopped disagreeing. The second direction is why the
    assertion is an EQUALITY rather than a subset -- a subset test would stay green while
    a row it claims to characterize quietly became inert.
    """
    recorded = {row[0] for row in _RESIDUAL} | {row[0] for row in _NONFINITE}
    observed = set()
    for tail in _CORPUS:
        text = f"[..] {_REPLAY}{tail}\n"
        token = repr(_token_file(tmp_path, text))
        regex = repr(_parse_replay_t(text, _REPLAY))
        leaf = repr(_parse_leading_float(text.split(_REPLAY, 1)[1]))
        if not (token == regex == leaf):
            observed.add(tail)
    assert observed == recorded, (
        f"unrecorded disagreements: {sorted(observed - recorded)!r}; "
        f"recorded-but-no-longer-disagreeing: {sorted(recorded - observed)!r}"
    )


def test_the_corpus_actually_exercises_both_tables():
    """Instrument check: a corpus that dropped either table would make the test above vacuous."""
    assert {row[0] for row in _RESIDUAL} <= set(_CORPUS)
    assert {row[0] for row in _NONFINITE} <= set(_CORPUS)
    assert len(_CORPUS) == len(set(_CORPUS)), "duplicate corpus members hide a coverage hole"
