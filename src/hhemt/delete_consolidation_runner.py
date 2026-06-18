"""Analysis-level deletion runner — final consolidation step.

Deletes the analysis-level artifacts (consolidated zarrs, plots,
``_generated/``, ``_status/`` minus ``_status/_deleting/``,
``analysis_report.{html,zip}``, ``scenario_status.csv``,
``workflow_summary.md``, ``.snakemake/``) and emits the
``analysis_consolidation`` sentinel.

The orchestrator's post-Snakemake check (``Analysis.delete()`` /
``TRITONSWMM_sensitivity_analysis.delete()``) reads the sentinel set and
removes ``analysis_dir/`` itself if all sentinels are present.

Per cleanup-rerun-delete-redesign Phase 2 + D-DeleteBoundary Option 1, and
distributed-delete-and-du-recording Phase 3 (submission-sentinel write-at-
entry + try/finally cleanup so the pre-flight reconciliation guard can
detect stuck-mid-delete workers). Note: this runner deletes ``_status/``
contents (which includes the submission sentinel it just wrote); the
finally block uses ``missing_ok=True`` so the unlink is safe whether the
inner cleanup removed the sentinel first or not.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

from hhemt.status_flags import write_status_flag
from hhemt.utils import fast_rmtree

logger = logging.getLogger(__name__)

_ANALYSIS_LEVEL_ARTIFACTS = [
    "analysis_datatree.zarr",
    "sensitivity_datatree.zarr",
    "system_datatree.zarr",
    "analysis_report.html",
    "analysis_report.zip",
    "plots",
    "report",
    "scenario_status.csv",
    "workflow_summary.md",
    "_generated",
    ".snakemake",
    ".snakemake_reprocess",
    "Snakefile",
    "Snakefile.reprocess",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _write_submission_sentinel(
    sentinel_path: Path, *, rule_token: str, slurm_job_id: str
) -> None:
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sentinel_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "slurm_jobid": slurm_job_id,
                "run_uuid": os.environ.get("SLURM_JOB_NAME"),
                "rule_token": rule_token,
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
        _rule_token = "delete_analysis_consolidation"
        _sentinel = (
            analysis_dir / "_status" / "_submitted" / f"{_rule_token}.json"
        )
        _write_submission_sentinel(
            _sentinel,
            rule_token=_rule_token,
            slurm_job_id=_slurm_jobid,
        )

    try:
        for name in _ANALYSIS_LEVEL_ARTIFACTS:
            path = analysis_dir / name
            if path.exists():
                print(f"[delete-consolidation] removing {path}", flush=True)
                if path.is_dir():
                    # EXEMPT-DU: delete-workflow-leaf
                    fast_rmtree(path)
                else:
                    # EXEMPT-DU: delete-workflow-leaf
                    path.unlink()

        # Delete _status/ contents EXCEPT _status/_deleting/ (which holds the
        # sentinels the orchestrator's post-check reads). Note: this removes
        # _status/_submitted/ — including this runner's own submission
        # sentinel — which is fine because the finally block's unlink uses
        # missing_ok=True.
        status_dir = analysis_dir / "_status"
        if status_dir.exists():
            for child in status_dir.iterdir():
                if child.name == "_deleting":
                    continue
                if child.is_dir():
                    # EXEMPT-DU: status-dir-cleanup
                    fast_rmtree(child)
                else:
                    # EXEMPT-DU: status-dir-cleanup
                    child.unlink()

        # Emit the consolidation sentinel.
        flag_path = analysis_dir / "_status" / "_deleting" / "analysis_consolidation.flag"
        write_status_flag(
            flag_path,
            rule_name="delete_analysis_consolidation",
        )
        return 0
    finally:
        if _sentinel is not None:
            # EXEMPT-DU: status-flag
            _sentinel.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
