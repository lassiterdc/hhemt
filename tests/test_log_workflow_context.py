"""Tests for ``hhemt.log_utils.log_workflow_context``.

Problem: the context block at the head of each runner log printed the SLURM
account under the label "Master SLURM job ID", so a reader correlating logs
chased the wrong identifier. These tests pin the repaired lines (chunk 1b.1,
decision D-B1 form 3). Laptop-safe: ``monkeypatch`` and ``caplog`` only, no
solver, no compile, no synth fixture.
"""

import logging

import pytest

from hhemt.log_utils import log_workflow_context

# Built from fragments so the value reads as a made-up account and cannot
# collide with any real field value.
_ACCOUNT = "zz" + "acct" + "_probe"

_SLURM_VARS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_ACCOUNT",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_JOB_NAME",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Return a monkeypatch with every SLURM variable the function reads unset."""
    for var in _SLURM_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _lines(caplog) -> list[str]:
    """Return the messages ``log_workflow_context`` logs under the current environment."""
    logger = logging.getLogger("test_log_workflow_context")
    with caplog.at_level(logging.INFO, logger=logger.name):
        caplog.clear()
        log_workflow_context(logger)
    return [r.getMessage() for r in caplog.records if r.name == logger.name]


def test_account_is_logged_on_no_line(clean_env, caplog):
    """Property: no logged line contains the SLURM account value.

    Class: REGRESSION, red on the pre-change tree where the master line printed the account.
    Why required: the account is not a job id; printing it under a job-id label sends a
    reader to correlate logs against the wrong identifier.
    What kills it: any fallback that reads SLURM_JOB_ACCOUNT into any logged line.
    Second correct implementation: printing a fixed placeholder for the master id still passes.
    """
    clean_env.setenv("SLURM_JOB_ID", "1234")
    clean_env.setenv("SLURM_JOB_ACCOUNT", _ACCOUNT)
    lines = _lines(caplog)
    assert any("WORKFLOW CONTEXT" in line for line in lines)
    offenders = [line for line in lines if _ACCOUNT in line]
    assert offenders == []


def test_context_block_logged_when_in_slurm(clean_env, caplog):
    """Property: the WORKFLOW CONTEXT header is logged whenever SLURM_JOB_ID is set.

    Class: control, green before and after; shows the capture reads this function's output.
    Why required: the account-absence arm would pass vacuously if nothing were logged.
    What kills it: an early return before the header.
    Second correct implementation: a differently worded rule line around the header still passes.
    """
    clean_env.setenv("SLURM_JOB_ID", "1234")
    lines = _lines(caplog)
    assert any("WORKFLOW CONTEXT" in line for line in lines)


def test_master_line_reads_array_job_id(clean_env, caplog):
    """Property: inside an array job the master line carries SLURM_ARRAY_JOB_ID.

    Class: control, green before and after.
    Why required: pins the one case where a master id exists, so the not-an-array-job arm
    cannot be satisfied by always printing the absence text.
    What kills it: dropping the array-id read.
    Second correct implementation: reading the id through a helper still passes.
    """
    clean_env.setenv("SLURM_JOB_ID", "1234")
    clean_env.setenv("SLURM_ARRAY_JOB_ID", "77")
    lines = _lines(caplog)
    assert "Master SLURM job ID: 77" in lines


def test_master_line_says_not_an_array_job(clean_env, caplog):
    """Property: outside an array job the master line says so instead of printing a value.

    Class: REGRESSION, red on the pre-change tree, which printed bare N/A or the account.
    Why required: the label states what the value is; with no master job the honest value
    is an explicit absence.
    What kills it: reintroducing any fallback value on this line.
    Second correct implementation: none by design; this test pins form 3 of D-B1 exactly.
    """
    clean_env.setenv("SLURM_JOB_ID", "1234")
    lines = _lines(caplog)
    assert "Master SLURM job ID: N/A (not an array job)" in lines


def test_job_name_line_set_and_unset(clean_env, caplog):
    """Property: a SLURM job name line names SLURM_JOB_NAME when set and N/A when unset.

    Class: NEW CAPABILITY, red on the pre-change tree, which had no such line; the
    plausible wrong implementation prints the name under a run-id label, which the exact
    text here rejects.
    Why required: pins the line form 3 of D-B1 adds, which the account-absence arm cannot see.
    What kills it: omitting the line or labelling it other than "SLURM job name".
    Second correct implementation: reading the name through os.getenv still passes.
    """
    clean_env.setenv("SLURM_JOB_ID", "1234")
    clean_env.setenv("SLURM_JOB_NAME", "run-x")
    assert "SLURM job name: run-x" in _lines(caplog)
    clean_env.delenv("SLURM_JOB_NAME")
    assert "SLURM job name: N/A" in _lines(caplog)


def test_local_run_logs_single_message(clean_env, caplog):
    """Property: outside SLURM exactly one message is logged.

    Class: control, green before and after.
    Why required: guards the local path against picking up the new SLURM lines.
    What kills it: logging the context block when SLURM_JOB_ID is unset.
    Second correct implementation: none by design; the local message text is pinned.
    """
    lines = _lines(caplog)
    assert lines == ["Execution environment: local (not SLURM)"]
