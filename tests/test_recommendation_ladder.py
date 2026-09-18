"""Fixture-free unit test of the workflow-status recommendation ladder (ITEM 1b(ii)).

No fixture, no compile, no shared cache: the ladder is a pure function of five booleans.
Each case pins the ORDERING (upstream incompleteness beats downstream completion), which is
the property `test_workflow_status_recommendations` arm 1 used to assert wrongly.
"""

import pytest

from hhemt.analysis import _recommendation_ladder

_ALL_TRUE = dict(setup_complete=True, all_prepared=True, all_run=True, proc_complete=True, summaries_exist=True)


def test_all_complete_is_fresh():
    assert _recommendation_ladder(**_ALL_TRUE)[:2] == ("complete", "fresh")


@pytest.mark.parametrize(
    "falsy, phase, mode",
    [
        ("setup_complete", "setup", "fresh"),
        ("all_prepared", "preparation", "resume"),
        ("all_run", "simulation", "resume"),
        ("proc_complete", "processing", "resume"),
        ("summaries_exist", "consolidation", "resume"),
    ],
)
def test_first_false_arm_wins(falsy, phase, mode):
    kw = dict(_ALL_TRUE)
    kw[falsy] = False
    assert _recommendation_ladder(**kw)[:2] == (phase, mode)


def test_stale_consolidation_marker_over_unprepared_scenario_says_resume():
    """The R11 divergence: summaries_exist True while all_prepared False -> RESUME, never fresh."""
    kw = dict(_ALL_TRUE)
    kw["all_prepared"] = False
    assert _recommendation_ladder(**kw)[:2] == ("preparation", "resume")
