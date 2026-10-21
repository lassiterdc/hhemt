"""The resume-evidence verdict discriminates, alarms are reported, and the denominator is disclosed.

Two layers are under test. ``hhemt.resume_events.audit_resume_evidence`` turns one sim's
ledger rows and its ``n_resumes`` into a verdict token; before this repair "no resume of
this sim ever left evidence" and "one of ten left none" were the same token. And
``hhemt.analysis_validation.check_resume_event_evidence`` runs that audit over every resumed
sim in an analysis so the report surfaces it; its summary must say how many sims it
examined, because a pass over zero examined sims is not a pass.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from hhemt import resume_events as re_mod
from hhemt.analysis_validation import check_resume_event_evidence


def _rows(*attempts):
    return [{"path": "snapshot", "attempt": a} for a in attempts]


def test_no_evidence_and_partial_evidence_get_different_tokens():
    """Property: zero rows for a resumed sim and some-but-not-all rows are different verdicts.

    Class: every ``n_resumes >= 1`` input whose row count is below it, split by whether the
    row count is zero. A third input -- rows equal to ``n_resumes`` -- is the satisfying
    position, and it stays ``complete``.

    Why required, checkable against the code: a machine consumer keys on the token, and the
    two cases call for different operator actions -- "this sim has never resumed
    successfully" against "one resume left no trace".

    What kills it: collapsing the ``recorded == 0`` branch back into ``evidence-shortfall``,
    or testing it after the shortfall branch so it is never reached.

    A second correct implementation under which it still passes: a verdict computed from a
    lookup over ``(recorded == 0, recorded < expected)``; the assertion is on the tokens, not
    on the branch order that produces them.
    """
    absent = re_mod.audit_resume_evidence([], n_resumes=3)
    partial = re_mod.audit_resume_evidence(_rows(1), n_resumes=3)
    full = re_mod.audit_resume_evidence(_rows(1, 2, 3), n_resumes=3)
    assert absent["verdict"] == "evidence-absent"
    assert partial["verdict"] == "evidence-shortfall"
    assert absent["verdict"] != partial["verdict"]
    assert full["verdict"] == "complete"
    assert absent["shortfall"] == 3 and partial["shortfall"] == 2


@pytest.mark.parametrize(
    ("row", "kind"),
    [
        ({"path": "step-back-required", "checkpoint_id": "5"}, "picker-miss"),
        ({"path": "stop", "reason": "write-failed", "checkpoint_id": "5"}, "write-failed"),
        ({"path": "clean-start", "reason": "write-failed", "checkpoint_id": "1"}, "write-failed"),
        ({"path": "snapshot", "stepback_from": 5, "stepback_reason": "write-failed"}, "write-failed"),
        (
            {"path": "snapshot", "stepback_from": 5, "stepback_corroboration": "mismatch"},
            "corroboration-mismatch",
        ),
    ],
)
def test_each_alarm_row_condition_is_reported_and_never_a_shortfall(row, kind):
    """Property: each of the five alarm row conditions is reported, and none creates a shortfall.

    Class: the five row conditions ``resume_events._row_alarms`` treats as alarms.

    Why required, checkable against the code: each alarm row is still one launch's event and
    is counted in ``recorded``, so the evidence verdict alone reads ``complete``. Without the
    alarm list the condition would be counted and never mentioned.

    What kills it: dropping any one condition from ``_row_alarms``, or leaving the alarm out
    of the summary.

    A second correct implementation under which it still passes: one that returns a count
    per kind instead of a list of dicts; the assertion reads kinds and the summary word.
    """
    audit = re_mod.audit_resume_evidence([row], n_resumes=1)
    assert audit["verdict"] == "complete"
    assert audit["shortfall"] == 0
    assert [a["kind"] for a in audit["alarms"]] == [kind]
    assert "ALARM" in audit["summary"]


@pytest.mark.parametrize(
    "row",
    [
        {"path": "snapshot"},
        {"path": "replay", "replay_reason": "absent"},
        {"path": "clean-start", "reason": "killed-write", "checkpoint_id": "1"},
        {"path": "snapshot", "stepback_from": 5, "stepback_reason": "killed-write"},
        {"path": "snapshot", "stepback_from": 5, "stepback_reason": "absent"},
    ],
)
def test_a_healthy_or_step_back_row_raises_no_alarm(row):
    """Property: rows that are not alarm conditions raise none -- including a picker step-back row.

    Class: the differently-positioned satisfying inputs for the alarm test above: a plain
    completed resume, a replay, a clean-start refusal caused by a killed write, and two
    picker step-backs (a step-back is a completed snapshot resume).

    Why required, checkable against the code: an alarm that fires on a normal step-back
    would fire on every resume the picker rescues, and an operator would learn to ignore it.

    What kills it: an alarm keyed on ``path == "clean-start"`` alone, or on the presence of
    ``stepback_from``.

    A second correct implementation under which it still passes: an allowlist of healthy row
    shapes rather than a denylist of alarm shapes, if it admits these five.
    """
    audit = re_mod.audit_resume_evidence([row], n_resumes=1)
    assert audit["alarms"] == []
    assert "ALARM" not in audit["summary"]


def _analysis(tmp_path):
    return SimpleNamespace(
        analysis_paths=SimpleNamespace(simlog_directory=tmp_path / "logs" / "sims"),
        cfg_analysis=SimpleNamespace(is_experiment_member=False, toggle_sensitivity_analysis=False),
    )


def _status(n_resumes):
    return pd.DataFrame(
        {
            "scenario_directory": ["sims/evt0"],
            "model_type": ["tritonswmm"],
            "n_resumes": [n_resumes],
            "event_iloc": [0],
        }
    )


def _write_ledger(tmp_path, rows):
    ledger = re_mod.ledger_path_for(tmp_path / "logs" / "sims" / "model_tritonswmm_evt0.log")
    re_mod.append_resume_events(ledger, rows, attempt=len(rows))


def test_the_check_discloses_an_empty_denominator_as_not_applicable(tmp_path):
    """Property: with no resumed sim the check says it examined zero and does not claim a pass.

    Class: every analysis whose status table records no ``n_resumes >= 1`` row.

    Why required, checkable against the code: ``passed`` alone cannot tell "examined 28,
    found nothing" from "examined 0", so the summary carries the count and the result is
    marked not applicable.

    What kills it: a summary without the examined count, or ``applicable`` left True.

    A second correct implementation under which it still passes: a summary worded "no sim
    attempted a resume (0 resumed sim(s) examined)"; the assertion reads the count phrase.
    """
    res = check_resume_event_evidence(_analysis(tmp_path), df_status=_status(0))
    assert res.applicable is False
    assert res.passed is True
    assert "0 resumed sim(s) examined" in res.summary


def test_the_check_fails_a_resumed_sim_with_no_ledger_and_names_the_denominator(tmp_path):
    """Property: a sim that attempted resumes and left no ledger fails the check, with its count named.

    Class: a status row with ``n_resumes >= 1`` and no ledger file -- the shape every
    aborted-resume sim leaves.

    Why required, checkable against the code: an absent ledger is what an all-aborted sim
    produces, and before this check nothing in the report surfaced it.

    What kills it: treating an absent ledger as INDETERMINATE, or not registering the
    ``evidence-absent`` verdict as a failure.

    A second correct implementation under which it still passes: one that stats the ledger
    first and records absence directly instead of reading an empty list.
    """
    res = check_resume_event_evidence(_analysis(tmp_path), df_status=_status(4))
    assert res.passed is False
    assert "1 sim(s) with NO evidence" in res.summary
    assert "1 resumed sim(s) examined" in res.summary
    assert len(res.details) == 1


def test_the_check_passes_a_complete_ledger_and_still_names_the_denominator(tmp_path):
    """Property: a complete, alarm-free ledger passes, and the pass says what it examined.

    Class: the satisfying position for the test above -- the same status row with a ledger
    holding one row per attempt.

    What kills it: a check that fails every resumed sim, or a pass summary with no count.

    A second correct implementation under which it still passes: one that also lists the
    examined scenarios in ``details`` on a pass, which this assertion does not read.
    """
    _write_ledger(tmp_path, [{"path": "snapshot"}, {"path": "snapshot"}])
    res = check_resume_event_evidence(_analysis(tmp_path), df_status=_status(2))
    assert res.passed is True
    assert res.applicable is True
    assert "1 resumed sim(s) examined" in res.summary
