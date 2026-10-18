"""``TRITONSWMM_analysis.cancel()`` counts this user's live worker jobs, hidden partitions included.

After the SIGINT, ``cancel()`` checks whether any SLURM job of this analysis is still
alive. It asks the accounting database (``sacct -u {user}``) rather than the queue
(``squeue -u {user}``), because a per-user ``squeue`` does not list jobs in SLURM Hidden
partitions -- on UVA Rivanna that includes the GPU partitions -- so a GPU job that
survived the cancel would be reported as gone. The worker jobs are found by NAME: the
Snakemake SLURM executor names each one with the driver's run uuid, which the tmux log
records and the workflow builder reads back. ``sacct`` also lists jobs that already ended,
so only rows whose state is not terminal are counted. When the lookup fails, cannot run,
or has no run uuid to ask about, ``cancel()`` says it could not verify the jobs instead of
saying they are all canceled.

How this is checked without SLURM. ``subprocess.run`` is replaced by a wrapper that, for
the ``squeue`` and ``sacct`` calls, runs the ORIGINAL ``subprocess.run`` with the caller's
own arguments, but with ``PATH`` pointing at stub executables. The stub ``squeue`` lists
nothing, as a per-user ``squeue`` does for hidden-partition jobs. The stub ``sacct`` serves
a fixed job table whose names follow the executor's run-uuid format, honouring the ``-o``
field list, the ``-P``/``--parsable2`` pipe delimiter, a ``--name`` filter and a
``-s``/``--state`` filter, and exits with a chosen status. tmux and ps get canned results.

No ``TRITONSWMM_analysis`` is constructed: the method is called unbound on a stand-in
object carrying only the attributes ``cancel()`` reads.
"""

from __future__ import annotations

import json
import os
import pwd
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hhemt.analysis import TRITONSWMM_analysis

_REAL_RUN = subprocess.run

_FAKE_SACCT = r"""
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_SACCT_RECORD"], "w") as fh:
    json.dump(args, fh)
code = int(os.environ.get("FAKE_SACCT_EXIT", "0"))
if code:
    sys.stderr.write("sacct: error: Problem talking to the database\n")
    sys.exit(code)
rows = json.loads(os.environ["FAKE_SACCT_ROWS"])
fields, states, names, sep = ["JobID", "JobName", "State"], None, None, " "
i = 0
while i < len(args):
    a = args[i]
    if a in ("-o", "--format"):
        fields = [f.split("%")[0] for f in args[i + 1].split(",")]
        i += 1
    elif a.startswith("--format="):
        fields = [f.split("%")[0] for f in a.split("=", 1)[1].split(",")]
    elif a in ("-s", "--state"):
        states = set(args[i + 1].upper().split(","))
        i += 1
    elif a.startswith("--state="):
        states = set(a.split("=", 1)[1].upper().split(","))
    elif a == "--name":
        names = set(args[i + 1].split(","))
        i += 1
    elif a.startswith("--name="):
        names = set(a.split("=", 1)[1].split(","))
    elif a in ("-P", "--parsable2"):
        sep = "|"
    i += 1
for row in rows:
    if states is not None and row["State"].split()[0] not in states:
        continue
    if names is not None and row["JobName"] not in names:
        continue
    print(sep.join(row[f] for f in fields))
"""


class _Field:
    """A log field stand-in: ``get()`` returns a fixed value and ``set()`` records nothing."""

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, _value):
        return None


_RUN_UUID = "cb4cc5fb-b338-409e-8a54-4aba886c3284"
_OTHER_UUID = "0f0e0d0c-0b0a-4908-8706-050403020100"


def _stand_in(run_uuids=(_RUN_UUID,)):
    log = SimpleNamespace(
        tmux_session_name=_Field("hhemt_session"),
        snakemake_pid=_Field(424242),
        workflow_canceled=_Field(None),
        workflow_cancellation_time=_Field(None),
    )
    builder = SimpleNamespace(
        _get_snakemake_pid_from_tmux=lambda _name: 424242,
        _tmux_slurm_run_uuids=lambda: tuple(run_uuids),
    )
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(analysis_id="an_x"),
        log=log,
        _workflow_builder=builder,
    )


def _install(monkeypatch, tmp_path, *, rows, lookup_exit=0):
    """Put stub ``squeue``/``sacct`` on PATH, fake tmux/ps, and pin the user name."""
    user = pwd.getpwuid(os.geteuid()).pw_name
    record = tmp_path / "sacct_argv.json"
    sacct = tmp_path / "sacct"
    sacct.write_text(f"#!{sys.executable}\n{_FAKE_SACCT}")
    squeue = tmp_path / "squeue"
    squeue.write_text(f"#!/bin/sh\nexit {lookup_exit}\n")
    for stub in (sacct, squeue):
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_SACCT_RECORD", str(record))
    monkeypatch.setenv("FAKE_SACCT_ROWS", json.dumps(rows))
    monkeypatch.setenv("FAKE_SACCT_EXIT", str(lookup_exit))
    for var in ("LOGNAME", "USER", "LNAME", "USERNAME"):
        monkeypatch.setenv(var, user)

    def fake_run(args, *a, **kw):
        head = args if isinstance(args, str) else args[0]
        if str(head).startswith(("squeue", "sacct")):
            return _REAL_RUN(args, *a, **kw)
        return subprocess.CompletedProcess(args, returncode=1 if head == "ps" else 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "kill", lambda _pid, _sig: None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    return user, record


def test_surviving_hidden_partition_job_is_counted(monkeypatch, tmp_path):
    """Property: a live job of this analysis that only the accounting database lists is counted.

    Class: every live (non-terminal) worker named with one of this run's uuids that a
    per-user ``squeue`` does not list -- the hidden-partition case. This test is one RUNNING
    worker beside one already CANCELLED worker of the same run and one RUNNING job of
    another run.

    Why it is required: a per-user ``squeue`` misses jobs in hidden partitions, so
    ``cancel()`` would report ``workers_canceled=True`` while a GPU job is still running.

    What kills it: asking ``squeue -u {user}`` instead of ``sacct -u {user}``; counting
    every ``sacct`` row, which also counts the already CANCELLED worker (2 instead of 1);
    or not restricting to this run's uuids, which counts the other run's job.

    A second correct implementation under which it still passes: ``sacct -u {user}
    --name {uuids} -s PENDING,RUNNING,REQUEUED,SUSPENDED -o JobName`` counting the rows it
    returns.
    """
    user, record = _install(
        monkeypatch,
        tmp_path,
        rows=[
            {"JobID": "101", "JobName": _RUN_UUID, "State": "RUNNING"},
            {"JobID": "102", "JobName": _RUN_UUID, "State": "CANCELLED by 0"},
            {"JobID": "103", "JobName": _OTHER_UUID, "State": "RUNNING"},
        ],
    )

    result = TRITONSWMM_analysis.cancel(_stand_in(), verbose=False, wait_timeout=10)

    assert result["workers_canceled"] is False
    assert any(e.startswith("1 worker jobs still running") for e in result["errors"])
    args = json.loads(record.read_text())
    assert args[args.index("-u") + 1] == user


def test_failed_lookup_is_reported_as_unverified(monkeypatch, tmp_path, capsys):
    """Property: when the job lookup fails, ``cancel()`` does not report the jobs as canceled.

    Class: any lookup that exits non-zero (database unreachable, refused query). This test
    makes both stubs exit 1.

    Why it is required: a failed lookup returns no rows, which the counter reads as zero
    workers, so ``cancel()`` printed "All worker jobs canceled" without having checked.

    What kills it: printing the success line whenever the count is zero, whatever the
    lookup's exit status was.

    A second correct implementation under which it still passes: raising
    ``subprocess.CalledProcessError`` via ``check=True`` and handling it in the existing
    "could not verify" arm.
    """
    _install(monkeypatch, tmp_path, rows=[], lookup_exit=1)

    TRITONSWMM_analysis.cancel(_stand_in(), verbose=True, wait_timeout=10)

    out = capsys.readouterr().out
    assert "Could not verify worker job status" in out
    assert "All worker jobs canceled" not in out


def test_differently_positioned_only_ended_jobs_reads_all_canceled(monkeypatch, tmp_path, capsys):
    """Satisfying input in a different position: every job of the analysis has ended.

    Green before and after: with no live job anywhere, both the queue and the accounting
    database lead to the success line.
    """
    _install(
        monkeypatch,
        tmp_path,
        rows=[{"JobID": "102", "JobName": _RUN_UUID, "State": "CANCELLED by 0"}],
    )

    result = TRITONSWMM_analysis.cancel(_stand_in(), verbose=True, wait_timeout=10)

    assert result["workers_canceled"] is True
    assert "All worker jobs canceled" in capsys.readouterr().out


def _assert_unverified_result(result):
    assert result["workers_canceled"] is False
    assert "Could not verify worker job status" in result["errors"]
    assert "Could not verify worker job status" in result["message"]


def test_failed_lookup_returns_workers_not_canceled(monkeypatch, tmp_path):
    """Property: when the lookup exits non-zero, the RETURNED result does not claim the workers are gone.

    Class: any lookup that runs and exits non-zero. This test makes the stub exit 1.

    Why it is required: a caller reading the result -- not the printed log -- otherwise
    sees ``workers_canceled: True`` for a check that never happened, which is how a
    script would decide the cancel was clean.

    What kills it: deriving ``workers_canceled`` from the job count alone, which is zero
    whenever the lookup returned no rows.

    A second correct implementation under which it still passes: returning early from the
    Step-4 block with ``workers_canceled`` set False and the reason appended, before the
    count is read.
    """
    _install(monkeypatch, tmp_path, rows=[], lookup_exit=1)

    result = TRITONSWMM_analysis.cancel(_stand_in(), verbose=False, wait_timeout=10)

    _assert_unverified_result(result)


@pytest.mark.parametrize(
    "exc",
    [subprocess.TimeoutExpired(cmd="sacct", timeout=5), FileNotFoundError("sacct")],
    ids=["timeout", "missing-binary"],
)
def test_lookup_that_cannot_run_returns_workers_not_canceled(monkeypatch, tmp_path, exc):
    """Property: when the lookup cannot run at all, the RETURNED result does not claim the workers are gone.

    Class: the lookup raising before it returns -- a timeout, or no ``sacct`` binary on
    the PATH. The wrapper raises the exception for the ``sacct`` call only.

    Why it is required: the same as the non-zero arm. This arm is reached through the
    ``except`` clause rather than through the exit-status branch, so it is a separate
    route to the same false claim.

    What kills it: handling the exception by printing only, leaving ``workers_canceled``
    derived from a count that was never taken.

    A second correct implementation under which it still passes: mapping both the
    exception and the non-zero exit onto one ``verified`` flag read once at the return.
    """
    _install(monkeypatch, tmp_path, rows=[])
    stubbed = subprocess.run

    def raising_run(args, *a, **kw):
        head = args if isinstance(args, str) else args[0]
        if str(head).startswith("sacct"):
            raise exc
        return stubbed(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", raising_run)

    result = TRITONSWMM_analysis.cancel(_stand_in(), verbose=False, wait_timeout=10)

    _assert_unverified_result(result)


def test_no_recorded_run_uuid_returns_workers_not_canceled(monkeypatch, tmp_path):
    """Property: with no run uuid to ask about, the RETURNED result does not claim the workers are gone.

    Class: any analysis whose tmux logs record no ``SLURM run ID`` -- a log that was
    removed, or a driver that died before the executor started. The stand-in builder
    returns no uuids.

    Why it is required: the workers can only be found by their run-uuid name, so with no
    uuid nothing was checked, and a zero count would read as a clean cancel.

    What kills it: querying with an empty name list, or skipping the query and leaving
    ``workers_canceled`` derived from the zero count.

    A second correct implementation under which it still passes: raising inside the
    Step-4 ``try`` when the uuid set is empty and handling it in the "could not verify"
    arm.
    """
    _install(monkeypatch, tmp_path, rows=[{"JobID": "101", "JobName": _RUN_UUID, "State": "RUNNING"}])

    result = TRITONSWMM_analysis.cancel(_stand_in(run_uuids=()), verbose=False, wait_timeout=10)

    _assert_unverified_result(result)
