"""Reviewer tests for chunk 1d.2: the provenance summary after the merge with develop.

The coder's file, `tests/test_provenance_completeness_absence_categories.py`, pins the closed
table, the note on each arm, and that `plots` and `report` stamps at another build or dirty
change no verdict. This file aims at what that file leaves positioned elsewhere:

- the note appears ONCE in a summary arm (the merge spliced a second copy into the missing arm);
- the denominator is the counted-stage count in the arm text, derived from the tuple;
- every counted stage, not one of them, moves the verdict when stamped at another build;
- an uncounted stamp that is UNDETERMINED or a sentinel changes no figure the arms print;
- `_absence_clause` keeps a stage that has no row, and states stages in the table's order.

No solver, no compile and no `synth_*` fixture: every test builds its stamps in memory.
"""

from __future__ import annotations

import pytest

from hhemt import analysis_validation as av
from hhemt.analysis_validation import _PROVENANCE_NOT_COUNTED_NOTE, _PROVENANCE_STAGES, _VALIDATED_STAGES

_UNCOUNTED = tuple(s for s in _PROVENANCE_STAGES if s not in _VALIDATED_STAGES)
_BASE = {"hhemt_sha": "a" * 40, "hhemt_version": "v0.1.0+A", "hhemt_dirty": "false"}
_OTHER = {"hhemt_sha": "b" * 40, "hhemt_version": "v0.1.0+B", "hhemt_dirty": "false"}
_DIRTY = {"hhemt_sha": "a" * 40, "hhemt_version": "v0.1.0+A", "hhemt_dirty": "true"}


def _stamped(**overrides):
    stamps = {s: dict(_BASE) for s in _PROVENANCE_STAGES}
    stamps.update(overrides)
    return stamps


def _run(monkeypatch, stamps):
    monkeypatch.setattr(av, "_collect_stage_stamps", lambda _a: stamps)
    return av.check_provenance_completeness(object())


# One stamp set per summary arm. The `missing` arm has two counted stages absent so the
# categories grouping is exercised, and every set leaves at least one counted stage stamped.
_ARMS = {
    "all_agree": lambda: _stamped(),
    "missing": lambda: _stamped(setup=None, sim=None),
    "disagree": lambda: _stamped(consolidate=dict(_OTHER)),
    "dirty": lambda: _stamped(consolidate=dict(_DIRTY)),
}


@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_the_not_counted_note_appears_exactly_once_in_every_arm(monkeypatch, arm):
    """Class: each of the four summary arms, built from a stamp set that selects it.

    Why required: a summary that states the note twice reads as two findings, and one that
    states it zero times tells the reader nothing about the two uncounted stages.
    Kills: a second copy of `_PROVENANCE_NOT_COUNTED_NOTE` appended to an arm (the merge left
    one in the missing arm), or the note dropped from an arm.
    Second correct implementation: the note appended once, after the arm is chosen.
    """
    res = _run(monkeypatch, _ARMS[arm]())
    assert res.summary.count(_PROVENANCE_NOT_COUNTED_NOTE.strip()) == 1


@pytest.mark.parametrize("absent", sorted(_VALIDATED_STAGES))
def test_the_missing_arm_states_its_ratio_once_over_the_counted_stages(monkeypatch, absent):
    """Class: the missing arm with each single counted stage absent (4 inputs).

    Property: the ratio phrase occurs once, its denominator is `len(_VALIDATED_STAGES)`, and
    the older sentence that called an absence benign is not appended after it.
    Why required: the merge left two ratio sentences in this arm, the second built on the
    older wording; a reader sees the same count twice with two different explanations.
    Kills: a leftover older sentence after the ratio; a denominator of the six-stage tuple.
    Second correct implementation: the ratio built through a helper shared by the arms.
    """
    n = len(_VALIDATED_STAGES)
    res = _run(monkeypatch, _stamped(**{absent: None}))
    assert res.passed is True
    assert res.summary.count("stages stamped") == 1
    assert res.summary.startswith(f"{n - 1}/{n} stages stamped; not captured: ")
    assert "legitimately has no stamp" not in res.summary


def test_the_agreeing_and_disagreeing_arms_state_the_counted_denominator(monkeypatch):
    """Class: the two arms that state a ratio over all counted stages (all stamped, disagree).

    Why required: the shape's example reads "{n}/4 stages stamped"; a denominator taken from
    the six-stage tuple prints "4/6" while the stamps counted are four.
    Kills: `len(_PROVENANCE_STAGES)` in place of `len(_VALIDATED_STAGES)` in either arm.
    Second correct implementation: the count held in a local named for the counted set.
    """
    n = len(_VALIDATED_STAGES)
    agree = _run(monkeypatch, _stamped())
    assert agree.summary.startswith(f"all {n} stages stamped at one hhemt build")
    disagree = _run(monkeypatch, _stamped(processing=dict(_OTHER)))
    assert disagree.summary.startswith(f"{n}/{n} stages stamped, and they disagree")


@pytest.mark.parametrize("stage", sorted(_VALIDATED_STAGES))
def test_every_counted_stage_moves_the_verdict_when_stamped_otherwise(monkeypatch, stage):
    """Class: each of the 4 counted stages, stamped at another build and stamped dirty.

    Control varying the property: `test_uncounted_stamps_...` shows the 2 uncounted stages
    move nothing. Kills: a count that stops reading one counted stage (a wrong filter that
    keeps three of four, which no single-stage test sees).
    Second correct implementation: the comparison done on a dict keyed by stage in any order.
    """
    other = _run(monkeypatch, _stamped(**{stage: dict(_OTHER)}))
    assert other.passed is False
    assert "disagree" in other.summary
    dirty = _run(monkeypatch, _stamped(**{stage: dict(_DIRTY)}))
    assert dirty.passed is False
    assert "uncommitted tracked changes" in dirty.summary
    assert f"({stage})" in dirty.summary


# Uncounted-stage stamps the coder's file does not position: an UNDETERMINED dirty flag, a
# sentinel sha, a legacy partial capture. Each feeds a figure the all-agree and disagree arms
# print (undetermined, excluded as unresolved, read from a legacy scalar carrier).
_UNCOUNTED_ODD = {
    "dirty_unknown": {"hhemt_sha": "c" * 40, "hhemt_version": "v0.1.0+C", "hhemt_dirty": "unknown"},
    "dirty_key_absent": {"hhemt_sha": "c" * 40, "hhemt_version": "v0.1.0+C"},
    "sentinel_sha": {"hhemt_sha": "unknown", "hhemt_version": "", "hhemt_dirty": "false"},
    "legacy_scalar": {"hhemt_sha": "c" * 40, "provenance_source": "legacy_scalar"},
}


@pytest.mark.parametrize("arm", sorted(_ARMS))
@pytest.mark.parametrize("shape", sorted(_UNCOUNTED_ODD))
@pytest.mark.parametrize("stage", _UNCOUNTED)
def test_uncounted_stamps_with_undetermined_or_sentinel_values_change_no_printed_figure(monkeypatch, arm, shape, stage):
    """Class: 2 uncounted stages x 4 stamp shapes x 4 arms = 32 inputs.

    Property: the verdict, the whole summary and the details rows equal those of the same arm
    without the stamp. The coder's file varies build and dirty-true; this varies the figures
    those leave untouched (`undetermined`, `n_excluded`, `n_legacy`).
    Kills: computing any of those figures over the unfiltered stamps while the ratio uses the
    filtered ones; a details list that gains a row for an uncounted stage.
    Second correct implementation: filtering `_collect_stage_stamps` before it is passed in.
    """
    baseline = _run(monkeypatch, _ARMS[arm]())
    res = _run(monkeypatch, {**_ARMS[arm](), stage: dict(_UNCOUNTED_ODD[shape])})
    assert res.passed is baseline.passed
    assert res.summary == baseline.summary
    assert res.details == baseline.details
    assert [d["stage"] for d in res.details] == sorted(_VALIDATED_STAGES)


@pytest.mark.parametrize("arm", ["missing", "disagree", "dirty"])
@pytest.mark.parametrize("absent", [("sim",), ("setup", "sim"), ("processing", "consolidate")])
def test_each_listing_arm_states_the_whole_absent_set_through_the_one_clause(monkeypatch, arm, absent):
    """Class: the 3 arms that list missing stages x 3 absent sets (9 inputs).

    Property: the arm's summary contains `_absence_clause(sorted(absent))` verbatim, so no arm
    hand-formats or truncates the list. The coder's file checks one substring of one set.
    Why required: the three arms must not drift apart on what an absence means.
    Kills: an arm that builds its own list, or passes only part of the missing set.
    Second correct implementation: one helper that all three arms call with the same list.
    """
    stamps = _stamped(**dict.fromkeys(absent))
    changed = "consolidate" if "consolidate" not in absent else "setup"
    if arm == "disagree":
        stamps[changed] = dict(_OTHER)
    if arm == "dirty":
        stamps[changed] = dict(_DIRTY)
    res = _run(monkeypatch, stamps)
    assert res.passed is (arm == "missing")
    assert av._absence_clause(sorted(absent)) in res.summary


def test_absence_clause_keeps_a_stage_with_no_row_and_names_stages_in_table_order():
    """Class: a missing list mixing rowed and unrowed stages, given in an order not the table's.

    Property: unrowed stages are listed once, in the order given, under their own phrase and
    never under a counted category; rowed stages follow the table's order, as the shape's
    example reads "(setup, sim)" and "(processing, consolidate)" for a list given sorted.
    Why required: a summary that dropped an unrowed stage would understate the absence.
    Kills: iterating the given list (alphabetical from the callers) instead of the table; an
    unrowed stage mapped to a counted category or dropped.
    Second correct implementation: ordering rowed stages by their index in `_VALIDATED_STAGES`.
    """
    out = av._absence_clause(["report", "sim", "plots", "processing", "consolidate", "setup"])
    assert "no site in this toolkit (setup, sim)" in out
    assert "not decidable from the tree (processing, consolidate)" in out
    assert out.endswith("no absence category recorded (report, plots)")
    assert out.count("report") == 1 and out.count("plots") == 1


def test_the_note_names_every_uncounted_stage_and_points_to_the_rendered_by_row():
    """Class: the stages in `_PROVENANCE_STAGES` and not in `_VALIDATED_STAGES` (2 today).

    Why required: the shape says the note names the uncounted stages and where their build is
    read; the set is derived here so a stage added to the uncounted side forces the note.
    Kills: a note edited to name one uncounted stage; a note that drops the Metadata pointer.
    Second correct implementation: the note composed from `_UNCOUNTED`-style names.
    """
    assert _UNCOUNTED, "the control: there is at least one uncounted stage to name"
    for stage in _UNCOUNTED:
        assert stage in _PROVENANCE_NOT_COUNTED_NOTE
    assert "Rendered by" in _PROVENANCE_NOT_COUNTED_NOTE
