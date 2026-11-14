"""Reviewer tests for ``hhemt.log_utils.log_workflow_context`` (chunk 1b.1, decision D-B1 form 3).

Problem: the context block printed the SLURM account under a job-id label. The coder's
tests pin single environment points (one variable set at a time). The claim here
quantifies over the whole environment space of the three variables the function may
read for the master and job-name lines, because a wrong implementation can hide in
the combinations a single-variable test never visits.
"""

import itertools
import logging

import pytest

from hhemt.log_utils import log_workflow_context

_ACCOUNT = "zz" + "acct" + "_review"
_ARRAY = "7701"
_NAME = "run-review-name"

_ALL_VARS = ("SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_ARRAY_JOB_ID", "SLURM_JOB_NAME")

# Each tuple is (account set, array id set, job name set): 8 points, SLURM_JOB_ID always set.
_POINTS = list(itertools.product((False, True), repeat=3))


def _messages(caplog, monkeypatch, *, account, array, name) -> list[str]:
    """Run the function under one environment point and return the logged messages."""
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "4321")
    if account:
        monkeypatch.setenv("SLURM_JOB_ACCOUNT", _ACCOUNT)
    if array:
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", _ARRAY)
    if name:
        monkeypatch.setenv("SLURM_JOB_NAME", _NAME)
    logger = logging.getLogger("test_log_workflow_context_review")
    with caplog.at_level(logging.INFO, logger=logger.name):
        caplog.clear()
        log_workflow_context(logger)
    return [r.getMessage() for r in caplog.records if r.name == logger.name]


@pytest.mark.parametrize(("account", "array", "name"), _POINTS)
def test_master_and_job_name_lines_across_environment_space(caplog, monkeypatch, account, array, name):
    """Property: at every point of the (account, array id, job name) environment space the
    master line carries only the array id or the explicit absence text, the job-name line
    immediately follows it and carries only the job name or N/A, and the account value is
    on no line.

    Class: NEW CAPABILITY, the plausible wrong implementation is a master line that falls
    back to SLURM_JOB_NAME (handoff B's form 2 mixed into form 3) or to the account only
    when another variable is also set; red under it at the point (account unset, array
    unset, name set), output pasted in the reviewer's file.
    Why required: the label "Master SLURM job ID" must state what the value is in every
    run method; a fallback that fires only when a second variable is set is invisible to a
    test that sets one variable at a time.
    What kills it: any fallback on the master line other than the absence text, a job-name
    line that is absent, not adjacent to the master line, or that reads another variable.
    Second correct implementation: building both lines from one helper that returns the
    pair, or logging the two lines in one call, still passes (adjacency is by message order).
    """
    lines = _messages(caplog, monkeypatch, account=account, array=array, name=name)
    assert not [line for line in lines if _ACCOUNT in line]
    master_idx = [i for i, line in enumerate(lines) if line.startswith("Master SLURM job ID:")]
    assert len(master_idx) == 1
    expected_master = f"Master SLURM job ID: {_ARRAY}" if array else "Master SLURM job ID: N/A (not an array job)"
    assert lines[master_idx[0]] == expected_master
    expected_name = f"SLURM job name: {_NAME}" if name else "SLURM job name: N/A"
    assert lines[master_idx[0] + 1] == expected_name
