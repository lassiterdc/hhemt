"""Orchestration-liveness sentinels for the reprocess concurrency gate.

A driver writes ``{analysis_dir}/_status/_orchestrator/{driver_id}.json`` at
start of ``run()`` / ``submit_workflow()`` (and the sensitivity-master
equivalent). The reprocess path consults these sentinels — NOT Snakemake's
working-dir lock — to decide whether a live orchestration *driver* exists for
the same analysis (see the decision doc "reprocess uses --nolock + orchestrator
sentinel as concurrency authority"). reprocess ALSO writes its own sentinel so
two concurrent reprocess drivers are mutually exclusive.

Lifecycle: blocking-local drivers remove the sentinel via try/finally on Python
return; detached drivers (batch_job tmux / single-job sbatch) leave a durable
sentinel reclaimed by the gate's liveness probes. Mirrors the
``_status/_submitted/`` sim-sentinel pattern in run_simulation_runner.py.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import datetime
from pathlib import Path

_ORCH_SUBDIR = ("_status", "_orchestrator")


def orchestrator_dir(analysis_dir: Path) -> Path:
    """Return ``{analysis_dir}/_status/_orchestrator`` (not created)."""
    return Path(analysis_dir).joinpath(*_ORCH_SUBDIR)


def new_driver_id() -> str:
    """Unique driver id: ``{pid}-{hostname}-{uuid4hex8}`` (collision-free across hosts)."""
    return f"{os.getpid()}-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"


def write_orchestrator_sentinel(
    analysis_dir: Path,
    *,
    driver_id: str,
    workflow_submission_mode: str,
    pid: int | None = None,
    slurm_jobid: str | None = None,
    tmux_session_name: str | None = None,
) -> Path:
    """Atomically write the orchestrator sentinel; return its path.

    temp + ``os.replace`` so a concurrent reader never sees a partial file.
    ``pid`` defaults to ``os.getpid()`` (the blocking-local driver's own pid).
    """
    d = orchestrator_dir(analysis_dir)
    d.mkdir(parents=True, exist_ok=True)
    sentinel = d / f"{driver_id}.json"
    tmp = sentinel.with_suffix(".json.tmp")
    payload = {
        "driver_id": driver_id,
        "pid": pid if pid is not None else os.getpid(),
        # Origin host as a FIRST-CLASS field. It is also embedded in driver_id
        # ({pid}-{hostname}-{uuid8}), but hostnames themselves contain dashes
        # (e.g. "udc-aw33-4c0"), so back-parsing that string is fragile and its
        # failure mode is SILENTLY-WRONG (a wrong host -> a wrong liveness
        # verdict). The writer always runs on the origin host by definition, so
        # this value cannot be wrong. Additive + OPTIONAL: a legacy sentinel
        # written before this field existed reads back as None, which the
        # liveness gate treats as UNKNOWN-host -- never as same-host. That is
        # why no schema-version field and no migration are needed: the absence
        # of the key IS the version signal, and it carries strictly more
        # information than a version int would.
        "hostname": socket.gethostname(),
        # SELF-SUFFICIENT AT WRITE TIME, and host-agnostic — the two properties the
        # `hostname` field above deliberately does NOT have. `enrich_orchestrator_sentinel`
        # runs only AFTER submit_workflow() returns (analysis.py, sensitivity_analysis.py)
        # and sits OUTSIDE the try/finally, so a driver killed mid-run never enriches and
        # leaves a permanently unresolvable record. A driver launched inside an allocation
        # already knows its job id here, and SLURM is cluster-wide, so this is the one
        # identifier a LATER resume on a DIFFERENT node can resolve — `tmux has-session`
        # and `ps -p` cannot, and for a compute-node origin the gate's foreign-host branch
        # short-circuits above them so `tmux_session_name` is never even read.
        #
        # NOT STAMPED ON A "local" SENTINEL, and that exclusion is load-bearing rather
        # than fastidious. FOUR of the five write sites (workflow.py:7477, :7703, :8069,
        # :11392) are reprocess/static-plots SELF-sentinels that hardcode
        # workflow_submission_mode="local" and pid=os.getpid(), and :7476 documents their
        # reclaim path as the gate's `ps -p {pid}` arm. `_orchestrator_liveness_gate`
        # does NOT dispatch on mode before identity, and it is not one branch that reads
        # the job id: identity decides first, and TWO arms read it — the
        # `1_job_many_srun_tasks` arm, and the arm that fires on any record carrying a
        # job id whose origin host is foreign or unrecorded. Both are host-agnostic and
        # are tested ahead of every host-local probe, so a job id written on a record
        # away from its own host IS examined. The exclusion is still right, for a ground
        # that does not depend on dispatch order: a `local` sentinel's own reclaim arm is
        # the `ps -p` probe over its `pid`, so on its own host the recorded job id is
        # never consulted, while away from its own host a stamped job id would promote
        # the record to the machine-wide arms and classify a DRIVER by its enclosing
        # ALLOCATION's liveness — a different subject, and the wrong one. Only :8226
        # (_acquire_submit_driver_claim) passes a real mode, so this condition confines
        # the new field to the submit path.
        #
        # Additive + OPTIONAL, exactly like `hostname`: unset outside an allocation (a
        # login-node batch_job driver) -> None -> the gate's existing UNKNOWN bucket,
        # unchanged. An explicit `slurm_jobid` argument still wins, and
        # `enrich_orchestrator_sentinel` only overwrites when its own argument is not
        # None, so a single-job driver's post-submit jobid still replaces this one. No
        # schema version and no migration: absence of the key IS the version signal, and
        # `_status/_orchestrator/` is ephemeral run-state, not analysis-tree layout, so no
        # LAYOUT_VERSION bump is owed.
        "slurm_jobid": (
            slurm_jobid
            if slurm_jobid is not None
            else (os.environ.get("SLURM_JOB_ID") if workflow_submission_mode != "local" else None)
        ),
        "tmux_session_name": tmux_session_name,
        "workflow_submission_mode": workflow_submission_mode,
        "submitted_at": datetime.now().isoformat(),
    }
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, sentinel)
    return sentinel


def remove_orchestrator_sentinel(analysis_dir: Path, driver_id: str) -> None:
    """Remove the sentinel (idempotent). Called from the driver's try/finally."""
    # EXEMPT-DU: status-flag
    (orchestrator_dir(analysis_dir) / f"{driver_id}.json").unlink(missing_ok=True)


def enrich_orchestrator_sentinel(
    analysis_dir: Path,
    driver_id: str,
    *,
    slurm_jobid: str | None = None,
    tmux_session_name: str | None = None,
) -> None:
    """Merge detached-driver identity fields into an existing sentinel in place.

    Reads the existing {driver_id}.json, updates only the supplied fields, and
    atomically rewrites (temp + os.replace). Preserves the original
    ``submitted_at`` and ``pid`` so those keep their driver-start meaning. No-op
    if the sentinel does not exist (blocking-local driver already removed it).
    """
    sentinel = orchestrator_dir(analysis_dir) / f"{driver_id}.json"
    try:
        payload = json.loads(sentinel.read_text())
    except (json.JSONDecodeError, OSError):
        return
    if slurm_jobid is not None:
        payload["slurm_jobid"] = slurm_jobid
    if tmux_session_name is not None:
        payload["tmux_session_name"] = tmux_session_name
    tmp = sentinel.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, sentinel)


def read_orchestrator_sentinels(analysis_dir: Path) -> list[dict]:
    """Return the parsed payloads of all ``_orchestrator/*.json`` sentinels.

    Corrupt/partial files are skipped (a concurrent writer's temp is named
    ``.json.tmp`` and excluded by the ``*.json`` glob).
    """
    d = orchestrator_dir(analysis_dir)
    out: list[dict] = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            payload = json.loads(p.read_text())
            payload["_path"] = str(p)
            out.append(payload)
        except (json.JSONDecodeError, OSError):
            continue
    return out
