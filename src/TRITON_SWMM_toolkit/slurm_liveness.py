"""Leaf SLURM-liveness primitives — stdlib-only, NO TRITON_SWMM_toolkit imports.

Extracted from workflow.py so the lightweight wait_for_sentinel_runner.py
subprocess can probe job liveness without importing the Snakemake-builder
surface. workflow.py re-exports these names for backward compatibility.

Per the wait-rule in-loop-liveness plan (R10). The per-job query forms are
load-bearing: on UVA Rivanna's `shen` cluster the user's GPU partitions are
SLURM Hidden partitions, so per-USER squeue is blind to live jobs while
per-JOB `squeue -j` / `sacct -j` report correctly. Never add a per-user form
here. See library/knowledge/slurm/hidden_partition_makes_per_user_squeue_blind.md.
"""

from __future__ import annotations

import subprocess
import sys

_LIVE_SQUEUE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
}

# Terminal sacct State codes that mean the job is gone. Anything NOT in this
# set AND present in sacct is treated as alive (still in the scheduler).
_SACCT_DEAD_STATES: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
    }
)


def _slurm_job_is_live(job_id: str, *, timeout_s: float = 10.0) -> bool:
    """True if ``job_id`` is PENDING/RUNNING/etc. in squeue.

    squeue is authoritative for live jobs (sacct gaps cannot hide a live job).
    Absent from squeue -> treated as not-live. On ``subprocess.TimeoutExpired``
    returns False (safe direction — caller treats unknown as not-live).
    """
    try:
        r = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[liveness] WARNING: squeue {job_id} timed out after {timeout_s}s — treating as not-live",
            file=sys.stderr,
            flush=True,
        )
        return False
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip().split()[0] in _LIVE_SQUEUE_STATES
    return False


def _sacct_states_batched(job_ids: list[str], *, timeout_s: float = 30.0) -> dict[str, tuple[str, str, str]]:
    """Batched sacct probe: ONE call for N job-ids.

    Returns ``{job_id: (state, exit_code, reason)}`` for every job-id sacct
    returned a row for. Job-ids ABSENT from the map are UNKNOWN. ``CANCELLED by
    <uid>`` is normalized to ``CANCELLED``. Empty map on timeout / not-found /
    non-zero return.
    """
    if not job_ids:
        return {}
    try:
        r = subprocess.run(
            ["sacct", "-j", ",".join(job_ids), "-n", "-P", "-X", "-o", "JobIDRaw,State,ExitCode,Reason"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[liveness] WARNING: sacct batched probe ({len(job_ids)} jobs) timed out after {timeout_s}s",
            file=sys.stderr,
            flush=True,
        )
        return {}
    except FileNotFoundError:
        print(
            f"[liveness] WARNING: sacct not found on PATH — batched probe ({len(job_ids)} jobs) skipped",
            file=sys.stderr,
            flush=True,
        )
        return {}
    if r.returncode != 0:
        return {}
    out: dict[str, tuple[str, str, str]] = {}
    for line in r.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid, state, exit_code, reason = parts[0], parts[1], parts[2], parts[3]
        out[jid] = (state.split()[0], exit_code, reason)
    return out


def job_is_dead_confirmed(job_id: str) -> bool:
    """Two-tier per-job death confirmation for the wait-rule in-loop probe.

    Returns True ONLY when the job is CONFIRMED gone. Conservative by design —
    any ambiguity returns False (keep waiting), so a transient scheduler blip
    never false-declares death.

    Tier 1 (squeue, authoritative for LIVE): if ``squeue -j`` shows the job in
    any live state, it is alive -> return False immediately.

    Tier 2 (sacct, confirms DEATH): squeue absent does not by itself prove
    death (MinJobAge aging, just-submitted gap). Confirm with one ``sacct -j``:
    a terminal state in ``_SACCT_DEAD_STATES`` -> dead (True). If sacct returns
    a NON-terminal state (still in the scheduler) -> alive (False). If sacct
    returns NO row (UNKNOWN — empty map from timeout/not-found/purge) -> NOT
    confirmed dead -> return False (defer to the wait cap / next reconcile).
    """
    if _slurm_job_is_live(job_id):
        return False
    states = _sacct_states_batched([job_id])
    row = states.get(job_id)
    if row is None:
        # UNKNOWN — squeue absent AND sacct has no row. Do not declare death;
        # the bounded wait cap and the next driver reconcile resolve it.
        return False
    return row[0] in _SACCT_DEAD_STATES
