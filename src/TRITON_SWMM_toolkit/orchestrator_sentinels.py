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
        "slurm_jobid": slurm_jobid,
        "tmux_session_name": tmux_session_name,
        "workflow_submission_mode": workflow_submission_mode,
        "submitted_at": datetime.now().isoformat(),
    }
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, sentinel)
    return sentinel


def remove_orchestrator_sentinel(analysis_dir: Path, driver_id: str) -> None:
    """Remove the sentinel (idempotent). Called from the driver's try/finally."""
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
