"""Per-scenario PROCESSED-only deletion runner for the SLURM-offloaded
reprocess delete (R8). Invoked by ``delete_processed_*`` rules in
``Snakefile.reprocess_delete``. Deletes ONLY ``sims/{event_id}/processed/``,
PRESERVING the sibling raw ``out_*`` binaries (the rebuild source,
scenario.py:76-80). Distinct from ``delete_scenario_runner`` (whole
``sims/{event_id}/``). Writes ``_status/_deleting_reprocess/processed_evt-{event_id}.flag``
+ the ``_status/_submitted/`` submission sentinel (try/finally cleanup)."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

from TRITON_SWMM_toolkit.status_flags import write_status_flag
from TRITON_SWMM_toolkit.utils import fast_rmtree

logger = logging.getLogger(__name__)

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args(argv)

def _write_submission_sentinel(sentinel_path: Path, *, rule_token: str, slurm_job_id: str, event_id: str) -> None:
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sentinel_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "slurm_jobid": slurm_job_id,
        "run_uuid": os.environ.get("SLURM_JOB_NAME"),
        "rule_token": rule_token,
        "event_id": event_id,
        "submitted_at": datetime.datetime.now().isoformat(),
    }))
    os.replace(tmp, sentinel_path)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    analysis_dir = args.analysis_dir.resolve()
    _sentinel: Path | None = None
    _slurm_jobid = os.environ.get("SLURM_JOB_ID")
    if _slurm_jobid:
        _rule_token = f"delete_processed_{args.event_id}"
        _sentinel = analysis_dir / "_status" / "_submitted" / f"{_rule_token}.json"
        _write_submission_sentinel(_sentinel, rule_token=_rule_token, slurm_job_id=_slurm_jobid, event_id=args.event_id)
    try:
        processed_dir = analysis_dir / "sims" / args.event_id / "processed"
        if processed_dir.exists():
            print(f"[delete-processed] removing {processed_dir}", flush=True)
            fast_rmtree(processed_dir, analysis_dir=analysis_dir)  # PATTERN A
        else:
            logger.warning("sims/%s/processed absent; recording sentinel anyway (idempotent).", args.event_id)
        flag_path = analysis_dir / "_status" / "_deleting_reprocess" / f"processed_evt-{args.event_id}.flag"
        write_status_flag(flag_path, rule_name=f"delete_processed_{args.event_id}", event_id=args.event_id)
        return 0
    finally:
        if _sentinel is not None:
            # EXEMPT-DU: status-flag
            _sentinel.unlink(missing_ok=True)

if __name__ == "__main__":
    sys.exit(main())
