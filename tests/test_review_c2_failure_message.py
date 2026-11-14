"""Reviewer tests for chunk 1c.2: the message a blocking local Snakemake failure hands the caller.

Problem: chunk 1c.2 makes ``run_snakemake_local`` return a message with the return code, the log file and
each ``Error in rule`` block. The blocking submit path then passes that result through
``_augment_result_with_partial_failures``, which rewrites ``message`` whenever the log or the ``_failed/``
markers name a rule that never finished. These tests drive the two calls in the order ``submit_workflow``
makes them, so they grade the message the caller actually receives and not the formatter in isolation.
"""

from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from hhemt import workflow as wf


def _block(rule: str, jobid: int) -> str:
    """One fabricated ``Error in rule`` block ending in the blank line Snakemake writes after it."""
    return (
        f"Error in rule {rule}:\n"
        "    message: None\n"
        f"    jobid: {jobid}\n"
        f"    output: _status/{rule}_{jobid}.flag\n"
        "    shell:\n"
        "        python run.py\n"
        "        (one of the commands exited with non-zero exit code)\n"
        "\n"
    )


_TAIL = "Exiting because a job execution failed. Look below for error messages\n"


def _blocking_local_result(tmp_path: Path, monkeypatch, log_text: str) -> dict:
    """Run ``run_snakemake_local`` to a nonzero exit over ``log_text`` and then the blocking path's sweep."""
    analysis_dir = tmp_path / "analysis"
    (analysis_dir / "logs").mkdir(parents=True)

    def fake_run(cmd, cwd, stdout, stderr, text, check):
        stdout.write(log_text)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(wf.subprocess, "run", fake_run)
    stub = SimpleNamespace(
        analysis_paths=SimpleNamespace(analysis_dir=analysis_dir, analysis_log_directory=analysis_dir / "logs"),
        cfg_analysis=SimpleNamespace(local_cpu_cores_for_workflow=1),
        generate_snakemake_config=lambda mode: {},
        write_snakemake_config=lambda config, mode: analysis_dir / "profile",
        _get_snakemake_base_cmd=lambda: ["snakemake"],
        _pre_snakemake_invocation_guards=lambda *a, **k: None,
    )
    stub._sweep_failed_rules = types.MethodType(wf.SnakemakeWorkflowBuilder._sweep_failed_rules, stub)
    result = wf.SnakemakeWorkflowBuilder.run_snakemake_local(stub, analysis_dir / "Snakefile", verbose=False)
    return wf.SnakemakeWorkflowBuilder._augment_result_with_partial_failures(stub, result)


_NEVER_FINISHED = _block("c_run_tritonswmm", 7)
_RETRIED_THEN_NEVER_FINISHED = (
    _block("a_prepare", 3) + "Finished jobid: 3 (Rule: a_prepare)\n\n" + _block("c_run_tritonswmm", 7)
)


@pytest.mark.parametrize(
    "log_body",
    [_NEVER_FINISHED, _RETRIED_THEN_NEVER_FINISHED],
    ids=["one-never-finished-block", "retried-to-success-block-then-never-finished-block"],
)
def test_blocking_local_failure_message_keeps_the_log_path_and_the_failing_block(tmp_path, monkeypatch, log_body):
    """Property: after the blocking path's sweep, ``message`` still names the log file and the failing block.

    Class: every blocking local run whose Snakemake log holds an ``Error in rule`` block for a job that never
    finished, which is the sweep's trigger for rewriting ``message``. NEW CAPABILITY: the plausible wrong
    implementation is the one that builds the message inside ``run_snakemake_local`` only, because the sweep then
    replaces it with a one-line list of rule tokens; the test is red under it. Reachability: ``submit_workflow``
    calls ``run_snakemake_local`` and then ``_augment_result_with_partial_failures`` on the returned dict, and a
    never-finished block is what the sweep reads from the log. Both-states anchor: the assertions read the log path
    out of the returned dict and the block text out of the log the test wrote, so they do not depend on the wording
    the change introduces. Why required: a local user reads ``result.message``, and the sweep's rewrite is the last
    writer. What kills it: leaving the sweep's ``result["message"] = ...`` assignment as a replacement. Second
    correct implementation: the sweep appends its sentence to the incoming message, or the caller composes the
    block text into the final message after the sweep; both pass.
    """
    result = _blocking_local_result(tmp_path, monkeypatch, log_body + _TAIL)

    assert result["success"] is False
    message = result["message"]
    assert str(result["snakemake_logfile"]) in message
    assert "    jobid: 7" in message
    assert "c_run_tritonswmm" in message


def test_failure_message_reaches_the_caller_when_the_sweep_names_no_rule(tmp_path, monkeypatch):
    """Property: a nonzero exit whose every error block was retried to success still returns the block text.

    Class: nonzero exits where the sweep finds no never-finished job, so it leaves ``message`` untouched. This is
    the held arm of the test above: green before and after, it proves the harness reaches the message and that the
    first test's red comes from the sweep's rewrite and not from the formatter. Held mutation: restoring the old
    ``error_msg`` f-string in ``run_snakemake_local`` reds it. Second correct implementation: composing the block
    text from the log in the caller instead of the builder.
    """
    log_body = _block("a_prepare", 3) + "Finished jobid: 3 (Rule: a_prepare)\n\n"
    result = _blocking_local_result(tmp_path, monkeypatch, log_body + _TAIL)

    assert result["partial_failures"] == []
    assert "    jobid: 3" in result["message"]
    assert str(result["snakemake_logfile"]) in result["message"]
