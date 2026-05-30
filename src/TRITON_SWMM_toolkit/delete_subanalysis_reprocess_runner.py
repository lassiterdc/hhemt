"""Per-sub-analysis PROCESSED+zarr scoped deletion runner for the SLURM-offloaded
reprocess delete (R8, D-scope Option C). Invoked by ``delete_subanalysis_reprocess_*``
rules in ``Snakefile.reprocess_delete``. Deletes the sub's ``sims/*/processed/``
dirs (only when ``--delete-processed``) + the sub's ``analysis_datatree.zarr``,
PRESERVING the sibling raw ``out_*`` binaries and never touching report/plots/_status.
Mirrors ``delete_subanalysis_runner``'s per-sub granularity but reprocess-scoped."""

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
    parser.add_argument("--analysis-dir", type=Path, required=True, help="The SUB-analysis dir.")
    parser.add_argument("--delete-processed", action="store_true",
                        help="Also delete sims/*/processed/ (set when start_with=='process').")
    return parser.parse_args(argv)

def _write_submission_sentinel(sentinel_path: Path, *, rule_token: str, slurm_job_id: str, sa_id: str) -> None:
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sentinel_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "slurm_jobid": slurm_job_id,
        "run_uuid": os.environ.get("SLURM_JOB_NAME"),
        "rule_token": rule_token,
        "sa_id": sa_id,
        "submitted_at": datetime.datetime.now().isoformat(),
    }))
    os.replace(tmp, sentinel_path)

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sub_dir = args.analysis_dir.resolve()
    _sentinel: Path | None = None
    _slurm_jobid = os.environ.get("SLURM_JOB_ID")
    if _slurm_jobid:
        _rule_token = f"delete_subanalysis_reprocess_{args.sa_id}"
        _sentinel = sub_dir / "_status" / "_submitted" / f"{_rule_token}.json"
        _write_submission_sentinel(_sentinel, rule_token=_rule_token, slurm_job_id=_slurm_jobid, sa_id=args.sa_id)
    try:
        if args.delete_processed:
            for processed_dir in sorted(sub_dir.glob("sims/*/processed")):
                if processed_dir.is_dir():
                    print(f"[delete-subanalysis-reprocess] removing {processed_dir}", flush=True)
                    fast_rmtree(processed_dir, analysis_dir=sub_dir)  # PATTERN A
        # "analysis_datatree.zarr" is the canonical consolidated-zarr default
        # (analysis.py:160 sets analysis_paths.analysis_datatree_zarr to
        # analysis_dir / "analysis_datatree.zarr"). Hardcoded here because the
        # runner only receives --analysis-dir and reconstructing AnalysisPaths
        # in-subprocess is heavier than this scoped deleter warrants — KEEP IN
        # SYNC if the consolidated-zarr filename ever becomes config-driven.
        sub_zarr = sub_dir / "analysis_datatree.zarr"
        if sub_zarr.exists():
            print(f"[delete-subanalysis-reprocess] removing {sub_zarr}", flush=True)
            fast_rmtree(sub_zarr, analysis_dir=sub_dir)  # PATTERN A
        flag_path = sub_dir / "_status" / "_deleting_reprocess" / "subanalysis_reprocess.flag"
        write_status_flag(flag_path, rule_name=f"delete_subanalysis_reprocess_{args.sa_id}", sa_id=args.sa_id)
        return 0
    finally:
        if _sentinel is not None:
            _sentinel.unlink(missing_ok=True)

if __name__ == "__main__":
    sys.exit(main())
