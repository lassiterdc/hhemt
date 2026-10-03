"""REVIEWER artifacts for `WP-2B` (resume-mechanism handling) — software-engineering-specialist:36.

AUTHORED BY THE REVIEWER HALF OF THE BUILDER PAIR, NOT BY THE IMPLEMENTER. Nothing here
edits `analysis_validation.py`, `processing_analysis.py`, `model_defects.py`, or
`eda/raw_resume_identity.py`; this file only ASSERTS over them. SOP section 8's
no-self-review rule is intact.

FIVE TESTS, AND THEY ARE NOT THE SAME KIND. Read the per-test docstring before reading its
assertion, because a green here means three different things depending on the test:

  * `test_parse_resume_timestep_is_total_over_an_empty_marker_tail` is RED ON THE TREE AS
    LANDED. It is the review finding, expressed as the failing test the review protocol
    asks for. It fails with `IndexError`, not `AssertionError`, and that is the point: the
    function's docstring promises `None` and the expression raises.
  * `test_marker_literal_module_sets_are_equal` and
    `test_the_three_numeric_parses_agree_over_the_solver_emitted_shape_family` are CLASS
    DRIFT GUARDS over properties MEASURED TO HOLD at `358e0e51`. A green is not evidence of
    a finding; it is a tripwire for the next consumer or the next parser.
  * `test_check_known_resume_defects_cannot_distinguish_absent_from_indeterminate` and
    `test_resume_mechanism_from_stamp_is_total_over_its_declared_literal` are
    CHARACTERIZATIONS. The first records a behaviour I am NOT ruling on — see its docstring.

    AMENDED 2026-10-03, and the amendment is the condition that docstring itself named.
    The developer ruled FOR disclosure ("agreed on both fronts"), scoping his earlier "we
    dont need a third category" to verdict CATEGORIES and leaving the summary WORDING open.
    So the first test's terminal assertion is INVERTED: it now asserts that the two registry
    states render DISTINGUISHABLY, which is the property the ruling delivered. Its NAME is
    left alone deliberately — the name states the finding as FOUND, and renaming it would
    break this campaign's node-id citations while the body and this note already say which
    way it points. Every other assertion in that test, its positive control included, is the
    reviewer's text unchanged.

COMPILE-FREE AND ANALYSIS-FREE BY CONSTRUCTION. No fixture here reaches the `*_compiled`
family and nothing constructs a `TRITONSWMM_analysis`, so `analysis.py:532-533`'s
modified-tracked-file guard cannot fire at setup. Screened by fixture closure
(`pytest --collect-only --fixtures-per-test`), never by module text.
"""

from __future__ import annotations

import pathlib
import re
import typing

import pandas as pd
import pytest

import hhemt.analysis_validation as av
from hhemt.eda.raw_resume_identity import parse_resume_timestep
from hhemt.model_defects import REGISTRY, resolve
from hhemt.processing_analysis import _parse_replay_t
from hhemt.resume_events import _parse_leading_float

_SRC_ROOT = pathlib.Path(av.__file__).parent

#: A sha in NO registry set — any future branch tip before someone registers it. The
#: campaign has already reached this state once (`01e95a76`, before the WP-1C repair).
_SHA_UNREGISTERED = "deadbeefcafef00ddeadbeefcafef00ddeadbeef"
#: The sha the synthetic tier actually builds, per `tests/fixtures/_triton_source_cache.py`.
_SHA_CURRENT_PIN = "e53c2fa01a64583fb57bc58082245fd687882b8f"
#: A build that genuinely carries all three defects — the POSITIVE CONTROL. Without it a
#: green on the two rows above is indistinguishable from an instrument that reports nothing.
_SHA_PRE_FIX = "15eb18a5d25afe5da295cb4b559a62669dbe5bc3"


class _StubCfgSystem:
    toggle_tritonswmm_model = True


class _StubSystem:
    cfg_system = _StubCfgSystem()


class _StubAnalysis:
    """The narrowest object `check_known_resume_defects` reads, and nothing more.

    It touches `analysis._system.cfg_system.toggle_tritonswmm_model` and, when `df_status`
    is passed, nothing else; `_read_triton_provenance` is monkeypatched module-globally the
    way every existing test in `test_coupled_resume_validity.py` does it.
    """

    _system = _StubSystem()


def _one_resumed_coupled_row() -> pd.DataFrame:
    return pd.DataFrame({"model_type": ["tritonswmm"], "n_resumes": [1], "event_iloc": [0]})


# ---------------------------------------------------------------------------------------
# 1. THE FINDING, as a failing test. RED on the tree as landed.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tail,label",
    [
        ("", "marker is the final text, no newline"),
        ("\n", "marker followed by a newline only"),
        ("   \n", "marker followed by spaces only"),
        ("\t", "marker followed by a tab only"),
    ],
)
@pytest.mark.parametrize("marker_name", ["_TRITON_SNAPSHOT_RESTORE_MARKER", "_TRITON_REPLAY_MARKER"])
def test_parse_resume_timestep_is_total_over_an_empty_marker_tail(tmp_path, marker_name, tail, label):
    """`parse_resume_timestep` must return None on a marker with no numeric tail, never raise.

    THE CONTRACT IS THE FUNCTION'S OWN, quoted from its docstring: "Returns the float
    (TRITON sim-time units) or None when the log is unreadable or carries neither marker
    (-> no vline; never a false verdict)." An empty tail is excluded by neither clause, so a
    third outcome is a contract violation rather than an undefined input.

    THE ASSERTION IS ANCHORED ON A PROPERTY TRUE IN BOTH STATES. It does not look for a new
    message or a new branch — it asks whether the call RETURNS. Pre-fix it raises
    `IndexError`; post-fix it returns `None`. A wording change cannot make it pass.

    WHY `None` AND NOT SOME OTHER VALUE: both sibling implementations of the same parse
    already return `None` here (`processing_analysis._parse_replay_t` via its regex
    returning no match, and `resume_events._parse_leading_float` via an explicit
    `if not parts: return None`), so `None` is the in-tree answer rather than this
    reviewer's preference. The repair is that guard, moved into this function.

    REACHABILITY OF THE INPUT CLASS IS NOT ESTABLISHED and this test does not claim it: a
    scan of the 38 `model_tritonswmm_*.log` files under `~/.cache/hhemt/synthetic_test_runs`
    found ZERO carrying any resume marker, so the local corpus bounds the rate in neither
    direction. What IS established is the call site: the conjunct-(C) b4b classifier in the
    deployment estate calls this function guarded only on its path argument's truthiness.
    """
    marker = getattr(av, marker_name)
    log = tmp_path / "model_tritonswmm_evt0.log"
    log.write_text(f"[..] Checkpoint files read\n[..] {marker}{tail}")

    assert parse_resume_timestep(log) is None, label


# ---------------------------------------------------------------------------------------
# 2. CLASS DRIFT GUARD over a property measured to hold.
# ---------------------------------------------------------------------------------------


def test_marker_literal_module_sets_are_equal():
    """No `src/` module may read the REPLAY marker without also reading the SNAPSHOT marker.

    THIS IS THE INSTRUMENT 4R PRESCRIBED, CORRECTED. 4R's enumeration is
    `grep -rn "_TRITON_REPLAY_MARKER\\|SWMM exchange history replayed" --include=*.py`,
    which covers only the replay half — and a silent v1 consumer is by definition a module
    that reads the replay marker and NOT the snapshot one, so the prescribed grep cannot see
    the class it was prescribed to find. The discriminating instrument is the SET DIFFERENCE.

    SET EQUALITY, NOT CONTAINMENT, DELIBERATELY. A containment assertion ("every module in
    the snapshot set is in the replay set") passes on a future replay-only module, which is
    exactly the member the class exists to exclude.

    Measured at `358e0e51`: both sets are
    {analysis_validation, eda/raw_resume_identity, processing_analysis, resume_events}.
    A green is a tripwire, not a finding.
    """
    replay_pat = re.compile(r"_TRITON_REPLAY_MARKER|REPLAY_MARKER|SWMM exchange history replayed")
    snap_pat = re.compile(r"_TRITON_SNAPSHOT_RESTORE_MARKER|SNAPSHOT_MARKER|SWMM state restored from snapshot")

    replay_mods, snap_mods = set(), set()
    for p in sorted(_SRC_ROOT.rglob("*.py")):
        text = p.read_text()
        rel = p.relative_to(_SRC_ROOT).as_posix()
        if replay_pat.search(text):
            replay_mods.add(rel)
        if snap_pat.search(text):
            snap_mods.add(rel)

    # Positive control on the instrument itself: if neither pattern matched anything, the
    # equality below would hold vacuously and certify nothing.
    assert replay_mods, "instrument is blind: no module matched the replay pattern"
    assert snap_mods, "instrument is blind: no module matched the snapshot pattern"

    assert replay_mods == snap_mods, (
        "a src module reads one marker without the other -- replay-only modules are silent "
        f"v1 readers. replay-only={sorted(replay_mods - snap_mods)} "
        f"snapshot-only={sorted(snap_mods - replay_mods)}"
    )


# ---------------------------------------------------------------------------------------
# 3. CHARACTERIZATION — and the one test in this file whose DIRECTION is not mine to set.
# ---------------------------------------------------------------------------------------


def test_check_known_resume_defects_cannot_distinguish_absent_from_indeterminate(monkeypatch):
    """`check_known_resume_defects` reports identically at a registered and an unregistered sha.

    THIS IS THE FIRST TEST THAT CALLS THIS FUNCTION AT ALL. Before it, the enumeration
    `grep -rn "check_known_resume_defects" --include=*.py tests/ src/` returned four hits:
    three prose mentions and one production call site. The two tests that exist for 4R
    sub-item (a) both NAME this function in their docstrings and both exercise
    `model_defects.resolve` instead -- so the premise that was pinned is the one 4R
    advertised, and the DECIDING premise (what the CHECK reports when `resolve` cannot
    answer) was unpinned.

    WHAT IS CHARACTERIZED, measured at `358e0e51`: three `absent`/`known_absent_set`
    verdicts and three `indeterminate`/`ancestry_unresolvable` verdicts render as the same
    `passed=True` and the same summary sentence once the pin substring is masked.
    `DefectVerdict.rule` exists -- per `model_defects`' docstring -- "so a reader can tell a
    cached-ancestry verdict from a live-ancestry one from a genuine override", and this
    consumer discards it on the passing branch.

    THE DIRECTION OF THIS TEST IS NOT THE REVIEWER'S TO SET, which is why it asserts the
    CURRENT behaviour rather than the behaviour the reviewer would prefer. The developer
    ruled on this surface on 2026-09-26, quoted in `analysis_validation.py`: "reporting no
    known defects is fine, because that phrase already implies what it needs to apply, we
    dont need a third category." The reviewer's reading is that the ruling governs a third
    VERDICT CATEGORY while what is measured here is a DISCLOSED DENOMINATOR in the summary
    string -- a distinction the same file applies one function below, where
    `check_coupled_resume_validity`'s summary names its examined and indeterminate counts.
    That reading may be wrong. If the developer rules for disclosure, INVERT this test; if
    the developer rules that "no third category" settles wording too, this test is the
    record of a deliberate choice rather than of an unnoticed one. Either way the next
    unregistered branch tip becomes visible here instead of silent.
    """
    df = _one_resumed_coupled_row()

    def _report(sha):
        monkeypatch.setattr(av, "_read_triton_provenance", lambda _a: sha)
        r = av.check_known_resume_defects(_StubAnalysis(), df_status=df)
        return r.passed, r.summary.replace(sha[:12], "{PIN}"), tuple(map(str, r.details))

    # The resolver DOES discriminate; only the report does not.
    assert {resolve(d, _SHA_CURRENT_PIN).status for d in REGISTRY} == {"absent"}
    assert {resolve(d, _SHA_UNREGISTERED).status for d in REGISTRY} == {"indeterminate"}
    assert {resolve(d, _SHA_CURRENT_PIN).rule for d in REGISTRY} == {"known_absent_set"}
    assert {resolve(d, _SHA_UNREGISTERED).rule for d in REGISTRY} == {"ancestry_unresolvable"}

    # POSITIVE CONTROL first, so a green below cannot come from an instrument that never
    # reports anything.
    monkeypatch.setattr(av, "_read_triton_provenance", lambda _a: _SHA_PRE_FIX)
    pre = av.check_known_resume_defects(_StubAnalysis(), df_status=df)
    assert pre.passed is False
    assert "known resume defect(s)" in pre.summary

    registered, unregistered = _report(_SHA_CURRENT_PIN), _report(_SHA_UNREGISTERED)
    assert registered[0] is True and unregistered[0] is True
    # INVERTED 2026-10-03 on the developer's ruling, per this docstring's own named
    # condition ("If the developer rules for disclosure, INVERT this test"). Nothing else
    # in this test changed: the four resolver assertions and the positive control above are
    # the reviewer's text byte-for-byte, and this line now asserts the REPAIRED property
    # rather than the characterized one. The verdict pair is still asserted unchanged one
    # line up, because the developer declined a third verdict CATEGORY and this test is
    # still what would catch one appearing.
    assert registered != unregistered, (
        "the registered and unregistered reports are indistinguishable once the pin "
        "substring is masked -- the 2026-10-03 summary-basis disclosure has regressed and "
        "the next unregistered branch tip is silent again"
    )
    assert "no known resume defect" in unregistered[1]


# ---------------------------------------------------------------------------------------
# 4. CHARACTERIZATION — sub-item (c)'s totality, enumerated rather than sampled.
# ---------------------------------------------------------------------------------------


def test_resume_mechanism_from_stamp_is_total_over_its_declared_literal():
    """Every stamp shape the read-back can receive returns a member of `ResumeMechanism`.

    ENUMERATED, NOT SAMPLED. 4R sub-item (c)'s named hazard is that "a replacement-shape
    stamp makes `.get()` return `None`", and the only way to retire it is to exhaust the
    input space rather than to test the shapes someone thought of.

    THE ROW WORTH READING is `v2 mech='indeterminate'`: the Literal declares FOUR members
    and the membership test admits THREE, so the type's own fourth value falls through to
    the v1 boolean fallback. For every PRODUCIBLE record that is still correct, because
    `processing_analysis` derives both `resume_mechanism` and `replayed` from one
    `resume_mechanism_from_log` call -- which never returns 'indeterminate' -- so the two
    cannot disagree. The shape that WOULD be wrong (mech='indeterminate' with
    replayed=True -> 'replay') is reachable only from a hand-edited or foreign-written
    stamp, and is pinned here so that a producer change making it reachable goes red.
    """
    members = set(typing.get_args(av.ResumeMechanism))
    assert members == {"snapshot", "replay", "none", "indeterminate"}

    cases = {
        "v1 replayed=True": ({"replayed": True}, "replay"),
        "v1 replayed=False": ({"replayed": False}, "indeterminate"),
        "v1 key absent": ({}, "indeterminate"),
        "v1 replayed=None": ({"replayed": None}, "indeterminate"),
        "v2 snapshot": ({"stamp_schema": 2, "resume_mechanism": "snapshot", "replayed": False}, "snapshot"),
        "v2 replay": ({"stamp_schema": 2, "resume_mechanism": "replay", "replayed": True}, "replay"),
        "v2 none": ({"stamp_schema": 2, "resume_mechanism": "none", "replayed": False}, "none"),
        "v2 4th-member": ({"stamp_schema": 2, "resume_mechanism": "indeterminate"}, "indeterminate"),
        "v2 4th-member + replayed": (
            {"stamp_schema": 2, "resume_mechanism": "indeterminate", "replayed": True},
            "replay",
        ),
        "v2 unknown + replayed": ({"resume_mechanism": "bogus", "replayed": True}, "replay"),
        "v2 unknown + not replayed": ({"resume_mechanism": "bogus", "replayed": False}, "indeterminate"),
        "v2 wrong case": ({"resume_mechanism": "REPLAY"}, "indeterminate"),
        "v2 mech None": ({"resume_mechanism": None, "replayed": True}, "replay"),
        "malformed None": (None, "indeterminate"),
        "malformed list": ([1, 2], "indeterminate"),
        "malformed str": ("replay", "indeterminate"),
        "malformed int": (0, "indeterminate"),
    }
    got = {label: av.resume_mechanism_from_stamp(rec) for label, (rec, _) in cases.items()}

    # Totality: no path escapes the declared domain, and none raises.
    assert set(got.values()) <= members, {k: v for k, v in got.items() if v not in members}
    # And every row lands where measured.
    assert got == {label: want for label, (_, want) in cases.items()}

    # The producer's half of the argument, so the 'unproducible' claim above is asserted
    # rather than recalled: the log reader never emits the fourth member.
    for text in ("", "nothing\n", f"{av._TRITON_SNAPSHOT_RESTORE_MARKER}1 s\n", f"{av._TRITON_REPLAY_MARKER}1 s\n"):
        assert av.resume_mechanism_from_log(text) != "indeterminate"


# ---------------------------------------------------------------------------------------
# 5. CLASS DRIFT GUARD — the three parses, over the shape the SOLVER emits.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tail,expected",
    [
        # The two shapes read from `src/swmm_triton.h` @ e53c2fa0 (:793 and :1136): the number,
        # then " s (", then the step count, then the resume-event fields.
        ("3600 s (12 steps); resuming live segment [resume-event checkpoint=7 path=replay reason=absent]", 3600.0),
        ("6000 s (11435 steps skipped); resuming live segment [resume-event checkpoint=84 path=snapshot]", 6000.0),
        # `up_to_time` is a value_t streamed with default operator<<, so a large value renders
        # in scientific notation. All three parsers must agree there too.
        ("6e+06 s (1 steps); resuming", 6_000_000.0),
        ("1.5e3 s", 1500.0),
        ("-1 s", -1.0),
        # Unparseable: all three must agree on None rather than on three different answers.
        ("not_a_number s", None),
    ],
)
def test_the_three_numeric_parses_agree_over_the_solver_emitted_shape_family(tmp_path, tail, expected):
    """All THREE in-tree implementations of "the" resume-time parse must agree.

    4R says "the existing numeric parse ... works on either unchanged", singular, and
    `analysis_validation.py` names two of them in one breath as if interchangeable. There
    are three: `eda/raw_resume_identity.parse_resume_timestep` (token rule),
    `processing_analysis._parse_replay_t` (regex rule), and
    `resume_events._parse_leading_float` (token rule plus an empty-tail guard).

    THIS WIDENS AN EXISTING TWO-POINT GUARD INTO A CLASS GUARD.
    `tests/test_resume_event_ledger.py::test_resume_time_agrees_with_the_existing_numeric_parse`
    is a two-element `parametrize` over two well-formed lines, pins only two of the three
    implementations, and covers neither member of the measured disagreement set.

    THE FAMILY IS BOUNDED BY WHAT THE SOLVER EMITS, read at the live pin rather than
    described: both call sites stream `up_to_time` and then the literal `" s ("`, so a
    whitespace-delimited numeric token always follows `t=`. The measured disagreement at
    `t=3600s` (no space) is therefore UNPRODUCIBLE at this pin and is deliberately NOT
    asserted here -- asserting it would encode a repair the solver's shape makes
    unnecessary. If the emitted shape ever drops that space, this family is where the
    divergence belongs.
    """
    marker = av._TRITON_REPLAY_MARKER
    text = f"[..] {marker}{tail}\n"
    log = tmp_path / "model_tritonswmm_evt0.log"
    log.write_text(text)

    token_rule = parse_resume_timestep(log)
    regex_rule = _parse_replay_t(text, marker)
    leaf_rule = _parse_leading_float(text.split(marker, 1)[1])

    assert token_rule == expected
    assert regex_rule == expected
    assert leaf_rule == expected
