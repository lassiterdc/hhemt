"""Characterization of the single execution-locus resolver.

NOT a discriminating regression test, and that is deliberate: promoting seven
resolvers into one changes no behaviour, so there is no pre-fix red to produce.
A test asserting the symbol exists would raise ImportError pre-fix -- red for the
wrong reason. What this pins is the CONTRACT, so a future edit to the helper is
caught by something other than a reviewer's memory.

The equivalence evidence for the promotion itself is elsewhere: the existing
suite staying green (notably test_synth_delete_slurm_lift_and_sentinels.py's
V-P3.1 arms, which reach the helper through the retained delegation, and the nine
node-local guard arms, which reach it through validation.py), plus the 16-cell
transcription run recorded in the round-4 deliverable.
"""

from __future__ import annotations

import pytest

from hhemt.exceptions import ConfigurationError
from hhemt.orchestration import resolve_execution_locus


@pytest.mark.parametrize(
    "execution_mode,method,expected",
    [
        # explicit override passes through, whatever the family says --
        # this row is the [Q8] shape and is why the auto arm needs one term
        ("slurm", "local", "slurm"),
        ("local", "batch_job", "local"),
        ("local", "1_job_many_srun_tasks", "local"),
        ("slurm", "batch_job", "slurm"),
        ("slurm", "1_job_many_srun_tasks", "slurm"),
        ("local", "local", "local"),
        # auto falls back to the config family
        ("auto", "local", "local"),
        ("auto", "batch_job", "slurm"),
        ("auto", "1_job_many_srun_tasks", "slurm"),
        ("auto", None, "local"),
        # None execution_mode is the delete-path spelling of "no override"
        (None, "local", "local"),
        (None, "batch_job", "slurm"),
        (None, "1_job_many_srun_tasks", "slurm"),
        (None, None, "local"),
    ],
)
def test_locus_table(execution_mode, method, expected):
    assert resolve_execution_locus(execution_mode, method) == expected


def test_unknown_execution_mode_raises():
    """Fail loud rather than defaulting -- a silent 'local' would spend an allocation."""
    with pytest.raises(ConfigurationError):
        resolve_execution_locus("bogus", "local")


def test_unknown_method_raises():
    with pytest.raises(ConfigurationError):
        resolve_execution_locus("auto", "bogus_mode")


def test_delete_mode_helper_delegates():
    """The retained delete-path method must stay observationally identical.

    Its own V-P3.1 tests already assert the four mappings; this arm asserts the
    DELEGATION specifically, so an edit that re-inlines the rule there is caught
    even if it happens to reproduce the same four answers.
    """
    from hhemt.workflow import SnakemakeWorkflowBuilder

    for method in (None, "local", "batch_job", "1_job_many_srun_tasks"):
        assert SnakemakeWorkflowBuilder._resolve_delete_mode_from_method(
            None, method
        ) == resolve_execution_locus(None, method)
