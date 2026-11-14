"""``cancel()`` must see the worker jobs the Snakemake SLURM executor actually submits.

Under ``batch_job`` the per-rule worker jobs are submitted by
``snakemake-executor-plugin-slurm``, which names every job with the run's uuid
(``--job-name {run_uuid}``) so that its own status loop can find them with
``sacct --name``. The analysis id does not appear in that name. The executor logs the
uuid as ``SLURM run ID: {uuid}`` into the driver's tmux log, and the workflow builder
already reads it back (``_tmux_slurm_run_uuids``) to query the same jobs by name.

So a worker check that looks for the analysis id inside each job name matches no worker
the executor submitted, counts zero, and reports every worker canceled while one is still
running. This module gives ``cancel()`` a still-running worker named the way the executor
names it, and requires ``cancel()`` to count it.

How this is checked without SLURM. ``subprocess.run`` is replaced: ``sacct`` and
``squeue`` get a fixed job table, filtered by ``--name`` and ``-s``/``--state`` when those
are passed and printed in the ``-o`` field order; ``squeue`` lists nothing, as a per-user
``squeue`` does for a hidden partition; ``tmux`` and ``ps`` get canned results. The run
uuid is offered both ways a correct implementation might read it: from the workflow
builder's ``_tmux_slurm_run_uuids()`` and from a tmux log in the analysis log directory.
No ``TRITONSWMM_analysis`` is constructed; the method is called on a stand-in.
"""

from __future__ import annotations

import os
import subprocess
import time
from types import SimpleNamespace

from hhemt.analysis import TRITONSWMM_analysis

_RUN_UUID = "cb4cc5fb-b338-409e-8a54-4aba886c3284"
_OTHER_UUID = "0f0e0d0c-0b0a-4908-8706-050403020100"

# One live worker of this analysis, one ended worker of this analysis, and one live job
# of another run. Names and comments follow the executor's own format.
_ROWS = [
    {
        "JobID": "201",
        "JobIDRaw": "201",
        "JobName": _RUN_UUID,
        "State": "RUNNING",
        "Comment": "rule_run_triton_wildcards_evt-1",
    },
    {
        "JobID": "202",
        "JobIDRaw": "202",
        "JobName": _RUN_UUID,
        "State": "CANCELLED by 0",
        "Comment": "rule_run_triton_wildcards_evt-2",
    },
    {
        "JobID": "203",
        "JobIDRaw": "203",
        "JobName": _OTHER_UUID,
        "State": "RUNNING",
        "Comment": "rule_run_triton_wildcards_evt-1",
    },
]


class _Field:
    """A log field stand-in: ``get()`` returns a fixed value and ``set()`` records nothing."""

    def __init__(self, value):
        self._value = value

    def get(self):
        """Return the fixed value."""
        return self._value

    def set(self, _value):
        """Accept a write and keep nothing, so ``cancel()``'s log updates touch no file."""
        return None


def _job_table(args: list[str]) -> str:
    """Print ``_ROWS`` the way ``sacct`` would for this argument list."""
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
        elif a in ("-P", "--parsable2", "-p", "--parsable"):
            sep = "|"
        i += 1
    lines = []
    for row in _ROWS:
        if states is not None and row["State"].split()[0] not in states:
            continue
        if names is not None and row["JobName"] not in names:
            continue
        lines.append(sep.join(row.get(f, "") for f in fields))
    return "\n".join(lines) + ("\n" if lines else "")


def _stand_in(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "tmux_session_20261017_090000.log").write_text(f"SLURM run ID: {_RUN_UUID}\n", encoding="utf-8")
    log = SimpleNamespace(
        tmux_session_name=_Field("hhemt_session"),
        snakemake_pid=_Field(424242),
        workflow_canceled=_Field(None),
        workflow_cancellation_time=_Field(None),
    )
    builder = SimpleNamespace(
        _get_snakemake_pid_from_tmux=lambda _name: 424242,
        _tmux_slurm_run_uuids=lambda: (_RUN_UUID,),
        _tmux_slurm_run_uuid=lambda: _RUN_UUID,
        analysis_paths=SimpleNamespace(analysis_log_directory=log_dir),
    )
    return SimpleNamespace(
        cfg_analysis=SimpleNamespace(analysis_id="an_x"),
        log=log,
        _workflow_builder=builder,
        analysis_paths=SimpleNamespace(analysis_log_directory=log_dir, analysis_dir=tmp_path),
    )


def _install(monkeypatch):
    def fake_run(args, *a, **kw):
        """Answer ``sacct``/``squeue`` from the job table and give tmux/ps canned results."""
        argv = args.split() if isinstance(args, str) else list(args)
        head = os.path.basename(str(argv[0]))
        if head == "sacct":
            return subprocess.CompletedProcess(args, returncode=0, stdout=_job_table(argv[1:]), stderr="")
        if head == "squeue":
            return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args, returncode=1 if head == "ps" else 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "kill", lambda _pid, _sig: None)
    monkeypatch.setattr(time, "sleep", lambda _s: None)


def test_live_worker_named_by_the_executor_run_uuid_is_counted(monkeypatch, tmp_path):
    """Property: a still-running worker named with this run's uuid is counted as not canceled.

    Class: live (non-terminal) worker jobs submitted by the Snakemake SLURM executor, whose
    job name is the run uuid the driver's tmux log records and carries no analysis id. This
    test is one RUNNING worker of this run, beside one CANCELLED worker of this run and one
    RUNNING job of another run.

    Why it is required: ``batch_job`` is the mode ``cancel()`` serves, and every worker in
    that mode is named this way, so a check that cannot see this name reports
    ``workers_canceled=True`` while a worker is still running -- and the published how-to
    tells the user that ``cancel()`` cancels the workers.

    What kills it: matching the analysis id inside the job name (no executor worker
    matches, so the count is 0); or counting every row for the run uuid without the
    terminal-state filter (2 instead of 1); or counting the other run's job.

    A second correct implementation under which it still passes: ``sacct --name {uuids}
    -X -n -P -o State`` over the uuids from ``_tmux_slurm_run_uuids()``, counting the rows
    whose state is not terminal -- the query ``workflow.py`` already uses for the same jobs.
    """
    _install(monkeypatch)

    result = TRITONSWMM_analysis.cancel(_stand_in(tmp_path), verbose=False, wait_timeout=10)

    assert result["workers_canceled"] is False
    assert any(e.startswith("1 worker jobs still running") for e in result["errors"]), result["errors"]
