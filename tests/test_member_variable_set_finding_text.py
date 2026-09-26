"""The short-member finding's CAUSAL sentence must be licensed by the mode it is emitted at.

`describe_member_variable_sets` is invoked for all eight `_MODE_CONFIG` modes, and the
mechanical case for that generality is sound: the member join NaN-fills identically at
every mode, and because the guard RECORDS rather than raises, a false positive costs one
coordinate string. What was not general is the emitted TEXT. The short-member branch ended
with "Most likely these members were produced by different solver builds emitting
different column sets" -- a claim licensed by the TRITON coupled-timer column split, and
therefore licensed at exactly two of the eight modes. At `swmm_only_node`,
`tritonswmm_swmm_link` or `tritonswmm_triton` a differing variable set comes from
differing model configuration or a differing parse, and has nothing to do with solver
builds. The guard was stamping a confident causal attribution it had no basis for onto a
published artifact, in the voice of a diagnostic.

These tests assert on WHICH CLAUSE IS EMITTED, which is a property of the returned string
in both the pre-fix and post-fix worlds: pre-fix the solver-build clause appears at every
mode, post-fix only at the two performance ones. Measured against pre-fix
`processing_analysis.py`, `test_a_non_performance_mode_does_not_blame_the_solver_build`
fails and the other three pass.
"""

from __future__ import annotations

import pytest

from hhemt.processing_analysis import (
    _PERFORMANCE_MODES,
    TRITONSWMM_analysis_post_processing,
    describe_member_variable_sets,
)

_SOLVER_BUILD_CLAUSE = "different solver builds emitting different column sets"

#: A short member and a complete one, which is the population that reaches the short
#: branch at all. The variable names are deliberately mode-agnostic: what is under test is
#: the CAUSE clause, not the detection, and the detection is already covered elsewhere.
_LABELS = ["0", "1"]
_SETS: list[object] = [{"a"}, {"a", "b"}]

_NON_PERFORMANCE_MODES = sorted(set(TRITONSWMM_analysis_post_processing._MODE_CONFIG) - _PERFORMANCE_MODES)


def test_the_performance_mode_names_are_real_mode_config_keys():
    """THE DRIFT GUARD for the two-name mirror list.

    `_PERFORMANCE_MODES` is a hand-written mirror of two `_MODE_CONFIG` keys. A mirror list
    is the failure this codebase has hit repeatedly, and the mitigation that makes it
    acceptable here is that it selects a clause of prose rather than gating detection --
    plus this test, which fails loudly if either name stops naming a live mode instead of
    the mirror silently degrading to "no mode is a performance mode".
    """
    live = set(TRITONSWMM_analysis_post_processing._MODE_CONFIG)
    assert _PERFORMANCE_MODES <= live, (
        f"_PERFORMANCE_MODES names {sorted(_PERFORMANCE_MODES - live)}, which is not a live "
        "_MODE_CONFIG key. The mirror has drifted; a name that no longer exists silently "
        "routes its mode to the mode-neutral cause sentence."
    )
    assert _NON_PERFORMANCE_MODES, "the test below needs at least one non-performance mode to be meaningful"


@pytest.mark.parametrize("mode", sorted(_PERFORMANCE_MODES))
def test_a_performance_mode_still_blames_the_solver_build(mode):
    """The attribution is correct at the two modes that license it and must be retained."""
    findings = describe_member_variable_sets(_LABELS, _SETS, mode=mode)
    short_finding = findings[0]
    assert "LACKS" in short_finding, "member 0 is the short one; the fixture has drifted"
    assert _SOLVER_BUILD_CLAUSE in short_finding, (
        f"the coupled-timer column split is exactly what {mode!r} is exposed to, so removing "
        "the attribution here would lose a true and useful diagnostic"
    )


@pytest.mark.parametrize("mode", _NON_PERFORMANCE_MODES)
def test_a_non_performance_mode_does_not_blame_the_solver_build(mode):
    """THE DEFECT. At these six modes the attribution cannot be true and must not be emitted."""
    findings = describe_member_variable_sets(_LABELS, _SETS, mode=mode)
    short_finding = findings[0]
    assert "LACKS" in short_finding, "member 0 is the short one; the fixture has drifted"
    assert _SOLVER_BUILD_CLAUSE not in short_finding, (
        f"{mode!r} does not carry the TRITON coupled-timer column set, so a differing "
        "variable set here cannot be caused by a differing solver build. Emitting that "
        "sentence stamps a wrong explanation onto a published artifact."
    )
    # And it does not simply go silent: a reader still learns that the cause is open.
    assert "not diagnosed here" in short_finding


@pytest.mark.parametrize("mode", sorted(TRITONSWMM_analysis_post_processing._MODE_CONFIG))
def test_the_detection_itself_stays_general_across_every_mode(mode):
    """The repair narrows the CLAUSE, never the detection.

    Pinned because narrowing the detection to the performance modes was the rejected
    option, and a later reader could mistake this file for having taken it.
    """
    findings = describe_member_variable_sets(_LABELS, _SETS, mode=mode)
    assert len(findings) == 2
    assert findings[0].startswith("HETEROGENEOUS"), f"{mode!r} lost its short-member detection"
    assert findings[1].startswith("HETEROGENEOUS"), f"{mode!r} lost its complete-member detection"
    assert mode in findings[0] and mode in findings[1], "the finding must still name its own mode"
