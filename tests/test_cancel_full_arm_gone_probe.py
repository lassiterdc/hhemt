"""The full cancel reports the running state as unknown when the driver's tmux session is gone.

`analysis.cancel()` probes the driver's tmux session. When the probe answers "gone", the full
cancel (no `rule_classes`) has nothing to signal and returns without asking the scheduler about
the workers that session submitted; those outlive the driver. Reporting `jobs_were_running:
False` there states an absence that no probe established, so the field is `None` ("not
determined"), the same value the method already uses when a probe could not answer. The tests
stand in for tmux and the scheduler, so no session, `scancel` or solver is touched.
"""

from __future__ import annotations

import subprocess
import types

from hhemt.analysis import TRITONSWMM_analysis


def _analysis(adir, session="sess"):
    """The minimal analysis one `cancel()` call needs: a recorded session name and an empty log."""
    (adir / "logs").mkdir(parents=True)
    a = TRITONSWMM_analysis.__new__(TRITONSWMM_analysis)
    a.analysis_paths = types.SimpleNamespace(analysis_dir=adir, analysis_log_directory=adir / "logs")
    a.log = types.SimpleNamespace(
        tmux_session_name=types.SimpleNamespace(get=lambda: session),
        snakemake_pid=types.SimpleNamespace(get=lambda: None),
        workflow_canceled=types.SimpleNamespace(set=lambda _v: None),
        workflow_cancellation_time=types.SimpleNamespace(set=lambda _v: None),
    )
    a.cfg_analysis = types.SimpleNamespace(analysis_id="gone_probe")
    a._workflow_builder = types.SimpleNamespace(
        _get_snakemake_pid_from_tmux=lambda s: None,
        _tmux_slurm_run_uuids=lambda: (),
    )
    return a


def _probe(monkeypatch, *, has_session_rc=0, raise_on_probe=False):
    """Replace `subprocess.run`: the `tmux has-session` probe answers as asked; any other call succeeds."""

    def _run(argv, *args, **kw):
        if list(argv[:2]) == ["tmux", "has-session"]:
            if raise_on_probe:
                raise FileNotFoundError(2, "No such file or directory", "tmux")
            return types.SimpleNamespace(returncode=has_session_rc, stdout="", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)


def test_a_gone_session_on_the_full_cancel_reports_the_running_state_as_unknown(tmp_path, monkeypatch):
    """Property: with a GONE probe, the full cancel returns `jobs_were_running` None, still succeeds,
    cancels nothing, reports no error, and its message does not call the workflow completed.

    Class: the full cancel (`rule_classes=None`) with `tmux has-session` returning non-zero.
    Class REGRESSION: red on the tree of `6097a72e`, whose return holds `jobs_were_running: False`
    (control: the same file run in an extract of that tree, output pasted in the chunk return).
    Why required: the command prints this record to the reader, and `False` tells the reader that
    nothing was running although the scheduler was never asked.
    What kills it: the field returned as `False` or omitted, `success` changed, or the message
    restored to "already completed". Second correct implementation: probe the scheduler and
    return the observed answer; this test then needs a scheduler stand-in and fails by design.
    """
    _probe(monkeypatch, has_session_rc=1)
    res = _analysis(tmp_path / "a").cancel(verbose=False)
    assert res["jobs_were_running"] is None, res
    assert res["success"] is True, res
    assert res["session_canceled"] is False and res["workers_canceled"] is False, res
    assert not res["errors"], res
    assert "already completed" not in res["message"], res


def test_the_neighbouring_arms_keep_their_values(tmp_path, monkeypatch):
    """Property: no session recorded stays `False`, and an unanswerable probe stays `None` with
    `success` False.

    Class: two other `cancel()` returns that carry `jobs_were_running`. Class NEW CAPABILITY;
    plausible wrong implementation: the fix applied to the `if not session_name:` branch (documented
    False) or to every branch; each is red here.
    Why required: the fix is for the one branch that answered a question it did not ask.
    What kills it: `None` returned for an unrecorded session, or `False` for the unanswerable probe.
    Second correct implementation: factor the three returns into one builder with the field as an argument.
    """
    _probe(monkeypatch)
    none_recorded = _analysis(tmp_path / "none", session=None).cancel(verbose=False)
    assert none_recorded["jobs_were_running"] is False and none_recorded["success"] is True, none_recorded

    _probe(monkeypatch, raise_on_probe=True)
    unknown = _analysis(tmp_path / "unknown").cancel(verbose=False)
    assert unknown["jobs_were_running"] is None and unknown["success"] is False, unknown


def test_the_full_cancel_transcript_on_a_gone_session_does_not_claim_completion(tmp_path, monkeypatch, capsys):
    """Property: the console transcript of the full cancel on a gone session does not say the workflow completed.

    Class: the full cancel with `verbose=True` (the default `hhemt cancel` leaves on) and
    `tmux has-session` returning non-zero. Class REGRESSION: red on the tree before the fix,
    whose console line read "(workflow already completed)" ahead of a record that says
    nothing was checked (output pasted in the chunk return).
    Why required: the command prints the console line and then the record, so the two channels
    must make one statement; the selective arm's transcript already omits the claim.
    What kills it: the console line restored to the completion claim. A second correct
    implementation: any wording that states the session is gone; the assertion reads the
    absence of the claim, not a preferred phrasing.
    """
    _probe(monkeypatch, has_session_rc=1)
    res = _analysis(tmp_path / "a").cancel(verbose=True)
    out = capsys.readouterr().out
    assert res["jobs_were_running"] is None
    assert "already completed" not in out, out


def test_the_gone_message_says_nothing_was_cancelled_and_the_jobs_were_not_checked(tmp_path, monkeypatch):
    """Property: the gone-session record's message states that the session no longer exists, that
    nothing was cancelled and that the jobs were not checked.

    Class: the full cancel with a gone session. Class NEW CAPABILITY; the plausible wrong
    implementation is a message that claims every job was cancelled, which passes the
    "already completed" absence check; red here.
    Why required: the command prints the message to the reader as it stands.
    What kills it: a message that states the opposite fact. A second correct implementation:
    any wording with the same three statements; the phrases below are matched after wrap
    collapsing, and a rewording must change them together.
    """
    _probe(monkeypatch, has_session_rc=1)
    res = _analysis(tmp_path / "a").cancel(verbose=False)
    msg = " ".join(res["message"].split())
    assert "no longer exists" in msg and "Nothing was cancelled" in msg and "not checked" in msg, msg
