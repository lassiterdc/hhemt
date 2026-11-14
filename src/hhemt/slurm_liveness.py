"""Leaf SLURM-liveness primitives — stdlib-only, NO hhemt imports.

Extracted from workflow.py so the lightweight wait_for_sentinel_runner.py
subprocess can probe job liveness without importing the Snakemake-builder
surface. workflow.py re-exports these names for backward compatibility.

Per the wait-rule in-loop-liveness plan (R10). The per-job query forms are
load-bearing: on UVA Rivanna's `shen` cluster the user's GPU partitions are
SLURM Hidden partitions, so per-USER squeue is blind to live jobs while
per-JOB `squeue -j` / `sacct -j` report correctly. Never add a per-user form
here. See library/knowledge/slurm/hidden_partition_makes_per_user_squeue_blind.md.

ONE IDIOM FOR "THE CHECK DID NOT ANSWER", applied DIFFERENTLY by each probe
below — which is the part to read before copying either. A scheduler query has
three outcomes and reporting two of them is a lie about the third; but what
"could not answer" looks like depends on the probe, so this section states each
probe's answer on each unanswerable class instead of one rule over both.

    _slurm_job_is_live       binary absent ............ None
                             launch OSError ............ None
                                 (PermissionError and every other OSError
                                  subclass, not only FileNotFoundError)
                             TimeoutExpired ............ False
                             non-zero exit ............. False

    _sacct_states_batched    binary absent ............ row omitted (empty map)
                             TimeoutExpired ............ row omitted (empty map)
                             non-zero exit ............. row omitted (empty map)
                             any OTHER OSError ......... RAISES

Three consequences, each SCOPED, because a universal over "what an unanswerable
launch may raise" or over "what a caller does with the third state" cannot be
made true by adding another exception class or another call site. The clauses
below therefore name what they do NOT cover.

On `_slurm_job_is_live` a `False` means "asked, and it is not live" and a `None`
means "could not ask" — the distinction its callers need. That sentence is NOT
true of `_sacct_states_batched`: it absorbs three named conditions and lets any
other `OSError` out, which is REACHABLE (an `sacct` that exists but is not
executable raises `PermissionError`). `_corroborate_not_live`'s own docstring
promises it never raises, and that promise is narrower than its name: of its
THREE callers only `_orchestrator_liveness_gate` wraps the call in
`try`/`except OSError`, while `_recover_inflight_via_comment` and
`_classify_live_sentinels` do not — so a `PermissionError` from the accounting
query escapes both, out of the delete preflight's refusal-set builder and out of
the sentinel classifier. Each of those sites needs its own guard; nothing above
them supplies one.

A TIMEOUT is an ANSWER on the probe and not-known on the batched probe, and the
asymmetry is deliberate: on the probe it is a slow cluster, which the callers'
second query exists to check; on the batched probe there is no second query to
appeal to. Deliberately NOT unified: `TimeoutExpired` is a `SubprocessError`, not
an `OSError`, so sweeping it into the probe's `except OSError` would move a
slow-but-present squeue off the corroborated path.

The third state earns its place only where a caller does something with it that
a `bool` cannot express. A caller-supplied TIMESTAMP TIER is the bound at three
of the probe's five call sites — `_classify_live_sentinels` and the gate's two
job-id arms — where the hold is bounded by `_max_plausible_job_lifetime_min` over
the held record's mtime. This module holds no clock over another process's
artifacts, so the age of the held record is the caller's to measure. The other
TWO sites are exempt and are named rather than assumed: the delete preflight's
comment-recovery path resolves not-known to INCLUSION in the refusal set,
because its candidates come out of the accounting database and have no
per-candidate file whose age could bound a hold, so its bound is the operator's
`override_in_flight` valve; and `job_is_dead_confirmed` answers "not confirmed
dead" and defers to its caller's wait cap.

The corroboration is that idiom APPLIED, not a second contract.
`job_is_dead_confirmed` asks squeue, and on a not-live answer asks
`_sacct_states_batched` before it will claim death at all; its absent-row answer
IS the NOT-KNOWN state again, and it resolves to "keep waiting" rather than to a
death claim.
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


def _slurm_job_is_live(job_id: str, *, timeout_s: float = 10.0) -> bool | None:
    """True if ``job_id`` is live in squeue, False if squeue answered "no", None if it could not be asked.

    squeue is authoritative for live jobs (sacct gaps cannot hide a live job).
    Absent from squeue's answer -> not-live (False). ``None`` is the NOT-KNOWN
    state of this module's docstring: the binary was never launched — no
    ``squeue`` on PATH, or another ``OSError`` from the launch — so the question
    was not asked, and it is deliberately NOT False. A caller that reclaimed a
    sentinel or admitted a delete on this answer would be acting on silence.

    ``TimeoutExpired`` keeps returning False, and that asymmetry is the whole
    design: a squeue that RAN and did not answer is a slow cluster, which the
    callers' second-query corroboration is built to catch, while a squeue that
    could not be launched is a host that cannot ask — and no corroboration of a
    squeue answer would notice it.

    Deliberately NOT ``except subprocess.SubprocessError``: without
    ``check=True`` its only reachable member is ``TimeoutExpired``, which the
    handler above keeps on the not-live path. Widening the arm to the parent
    class would move a slow-but-present squeue to NOT-KNOWN and reverse the
    ruled behaviour for it.
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
    except OSError as exc:
        # NOT-KNOWN, not not-live: the check could not run, and reporting that as
        # death is the false-proceed the corroboration chain exists to remove.
        # Every consumer already routes this third state to a hold bounded by its
        # own timestamp tier.
        print(
            f"[liveness] WARNING: squeue not runnable ({exc}) — liveness of {job_id} is NOT KNOWN",
            file=sys.stderr,
            flush=True,
        )
        return None
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
    any live state, it is alive -> return False immediately. A NOT-KNOWN answer
    (``None`` from the probe) does not short-circuit: it falls through to Tier 2,
    which is exactly the "squeue could not answer, so silence proves nothing"
    case that tier exists for.

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
