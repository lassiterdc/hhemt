"""Analysis-level deletion runner — final consolidation step.

Deletes the analysis-level artifacts (consolidated zarrs, plots,
``_generated/``, ``_status/`` minus ``_status/_deleting/``,
``analysis_report.{html,zip}``, ``scenario_status.csv``,
``workflow_summary.md``, ``.snakemake/``) and emits the
``analysis_consolidation`` sentinel.

The orchestrator's post-Snakemake check (``Analysis.delete()`` /
``TRITONSWMM_sensitivity_analysis.delete()``) reads the sentinel set and
removes ``analysis_dir/`` itself if all sentinels are present.

Per cleanup-rerun-delete-redesign Phase 2 + D-DeleteBoundary Option 1.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from TRITON_SWMM_toolkit.status_flags import write_status_flag
from TRITON_SWMM_toolkit.utils import fast_rmtree

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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    analysis_dir = args.analysis_dir.resolve()

    for name in _ANALYSIS_LEVEL_ARTIFACTS:
        path = analysis_dir / name
        if path.exists():
            print(f"[delete-consolidation] removing {path}", flush=True)
            if path.is_dir():
                fast_rmtree(path)
            else:
                path.unlink()

    # Delete _status/ contents EXCEPT _status/_deleting/ (which holds the
    # sentinels the orchestrator's post-check reads).
    status_dir = analysis_dir / "_status"
    if status_dir.exists():
        for child in status_dir.iterdir():
            if child.name == "_deleting":
                continue
            if child.is_dir():
                fast_rmtree(child)
            else:
                child.unlink()

    # Emit the consolidation sentinel.
    flag_path = analysis_dir / "_status" / "_deleting" / "analysis_consolidation.flag"
    write_status_flag(
        flag_path,
        rule_name="delete_analysis_consolidation",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
