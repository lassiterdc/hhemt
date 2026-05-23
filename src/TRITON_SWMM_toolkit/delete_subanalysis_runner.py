"""Per-sub-analysis deletion runner invoked by ``delete_subanalysis_*`` rules.

Deletes the sub-analysis's ``subanalyses/sa_{sa_id}/`` subtree (which contains
its own nested ``sims/`` tree) and writes the
``_status/_deleting/subanalysis_sa-{sa_id}.flag`` sentinel.

Per cleanup-rerun-delete-redesign Phase 2 + D-DeleteBoundary Option 1, and
distributed-delete-and-du-recording Phase 3 (submission-sentinel write-at-
entry + try/finally cleanup so the pre-flight reconciliation guard can
detect stuck-mid-delete workers).
"""

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
    parser.add_argument("--sa-id", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _write_submission_sentinel(
    sentinel_path: Path, *, rule_token: str, slurm_job_id: str, sa_id: str
) -> None:
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sentinel_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "slurm_jobid": slurm_job_id,
                "run_uuid": os.environ.get("SLURM_JOB_NAME"),
                "rule_token": rule_token,
                "sa_id": sa_id,
                "submitted_at": datetime.datetime.now().isoformat(),
            }
        )
    )
    os.replace(tmp, sentinel_path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    analysis_dir = args.analysis_dir.resolve()

    _sentinel: Path | None = None
    _slurm_jobid = os.environ.get("SLURM_JOB_ID")
    if _slurm_jobid:
        _rule_token = f"delete_subanalysis_sa-{args.sa_id}"
        _sentinel = (
            analysis_dir / "_status" / "_submitted" / f"{_rule_token}.json"
        )
        _write_submission_sentinel(
            _sentinel,
            rule_token=_rule_token,
            slurm_job_id=_slurm_jobid,
            sa_id=args.sa_id,
        )

    try:
        sa_dir = analysis_dir / "subanalyses" / f"sa_{args.sa_id}"
        if not sa_dir.exists():
            logger.warning(
                "sa_dir %s does not exist; recording sentinel anyway (idempotent delete).",
                sa_dir,
            )
        else:
            print(f"[delete] removing {sa_dir}", flush=True)
            fast_rmtree(sa_dir)

        flag_path = (
            analysis_dir / "_status" / "_deleting" / f"subanalysis_sa-{args.sa_id}.flag"
        )
        write_status_flag(
            flag_path,
            rule_name=f"delete_subanalysis_{args.sa_id}",
            sa_id=args.sa_id,
        )
        return 0
    finally:
        if _sentinel is not None:
            _sentinel.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
