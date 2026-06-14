"""Wait-on-sentinel runner: polls v2 state markers for the original SLURM job
to write its completion-or-failure marker, then exits 0 (success) or 1
(failure).

Per sentinel-system-v2 Phase 2.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Probe SLURM at most this often (seconds). The backoff loop sleeps 5->60 s,
# so a time-based gate (not a cycle count) gives a stable ~5 min cadence
# independent of where the backoff has ramped to. ~15 ms/call measured on a
# live shen login node, so per-wait independent probing is negligible load.
_PROBE_INTERVAL_S = 300


def _read_submitted_jobid(status_dir: Path, rule_token: str) -> str | None:
    """Recover the original SLURM job-id from the submitted-sentinel, or None
    (local mode / no sentinel / no jobid) — in which case the probe is skipped."""
    sentinel = status_dir / "_submitted" / f"{rule_token}.json"
    if not sentinel.exists():
        # mechanism (b): a PENDING-recovered token has no _submitted/ sentinel yet
        # (its worker never started). Fall back to _queued/{token}.json so the
        # in-loop liveness probe (Gotcha 43) re-enables instead of degrading to a
        # pure marker-poll that stalls to the flat cap on an in-queue cancellation.
        # The _queued/ payload carries the same top-level slurm_jobid key — null on
        # executor-owns-sbatch clusters (probe stays disabled there, byte-identical
        # to today), the allocation jobid on toolkit-owns-sbatch clusters.
        sentinel = status_dir / "_queued" / f"{rule_token}.json"
    try:
        return str(json.loads(sentinel.read_text()).get("slurm_jobid") or "") or None
    except (json.JSONDecodeError, OSError):
        return None


def _write_failed_marker_and_reclaim(status_dir: Path, rule_token: str, job_id: str) -> None:
    """Atomically write _status/_failed/{rule_token}.json (mirroring
    run_simulation_runner's temp+os.replace pattern — NOT write_status_flag,
    which is reserved for _status/*.flag) and unlink the submitted-sentinel so
    the next reconcile classifies the death via _classify_via_state_markers.
    Does NOT write the c_run completion flag (D-Q1 worker-once invariant)."""
    failed_dir = status_dir / "_failed"
    failed_dir.mkdir(parents=True, exist_ok=True)
    marker = failed_dir / f"{rule_token}.json"
    payload = {
        "slurm_jobid": job_id,
        "rule_token": rule_token,
        "status": "failed",
        "reason": (
            "in-loop liveness probe confirmed SLURM job is dead (no "
            "completion/failure marker written by worker — OS-level death)"
        ),
        "finished_at": datetime.datetime.now().isoformat(),
    }
    tmp = marker.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, marker)
    (status_dir / "_submitted" / f"{rule_token}.json").unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule-token", required=True)
    parser.add_argument("--flag-output", required=True)
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--max-wait-minutes", type=int, required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    status_dir = analysis_dir / "_status"
    if not status_dir.is_dir():
        logger.error(
            "[wait_for_sentinel] %s: analysis _status/ directory does not "
            "exist at %s; refusing to poll. This likely indicates "
            "--analysis-dir is wrong or the analysis tree was deleted "
            "between Snakefile build and rule launch.",
            args.rule_token,
            status_dir,
        )
        return 1
    completed = status_dir / "_completed" / f"{args.rule_token}.json"
    failed = status_dir / "_failed" / f"{args.rule_token}.json"

    deadline = time.monotonic() + (args.max_wait_minutes * 60)
    backoff_s = 5
    backoff_max = 60
    # In-loop SLURM-liveness: probe at most every _PROBE_INTERVAL_S so an
    # OS-killed (marker-less) original job is detected WITHIN the wait rather
    # than only at the next driver reconcile. Skipped entirely when there is no
    # jobid (local mode) or squeue is not on PATH (R6 — byte-identical to the
    # prior marker-only poll). Imported lazily so the leaf module is only
    # loaded when a probe actually fires.
    job_id = _read_submitted_jobid(status_dir, args.rule_token)
    probe_enabled = bool(job_id) and shutil.which("squeue") is not None
    last_probe = time.monotonic()

    logger.info(
        "[wait_for_sentinel] polling for %s markers under %s (max %d min)",
        args.rule_token,
        status_dir,
        args.max_wait_minutes,
    )

    while time.monotonic() < deadline:
        if completed.exists():
            # The original SLURM worker is the sole writer of the completion
            # flag (per the v1 at-most-once contract); the wait-rule does NOT
            # touch the flag. Snakemake observes the existing flag at exit-0.
            logger.info(
                "[wait_for_sentinel] %s: completed marker observed; exit 0",
                args.rule_token,
            )
            return 0
        if failed.exists():
            logger.error(
                "[wait_for_sentinel] %s: failed marker observed; exit 1",
                args.rule_token,
            )
            return 1
        if probe_enabled and (time.monotonic() - last_probe) >= _PROBE_INTERVAL_S:
            last_probe = time.monotonic()
            from TRITON_SWMM_toolkit.slurm_liveness import job_is_dead_confirmed

            if job_is_dead_confirmed(job_id):
                logger.error(
                    "[wait_for_sentinel] %s: in-loop probe CONFIRMED job %s is dead "
                    "(no worker marker — OS-level death); writing _failed and exiting 1",
                    args.rule_token,
                    job_id,
                )
                _write_failed_marker_and_reclaim(status_dir, args.rule_token, job_id)
                return 1
        time.sleep(backoff_s)
        backoff_s = min(backoff_s * 2, backoff_max)

    logger.error(
        "[wait_for_sentinel] %s: walltime cap (%d min) exceeded with neither completed nor failed marker; exit 1",
        args.rule_token,
        args.max_wait_minutes,
    )
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    sys.exit(main())
