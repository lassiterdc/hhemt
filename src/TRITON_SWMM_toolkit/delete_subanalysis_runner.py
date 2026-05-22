"""Per-sub-analysis deletion runner invoked by ``delete_subanalysis_*`` rules.

Deletes the sub-analysis's ``subanalyses/sa_{sa_id}/`` subtree (which contains
its own nested ``sims/`` tree) and writes the
``_status/_deleting/subanalysis_sa-{sa_id}.flag`` sentinel.

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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sa-id", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    analysis_dir = args.analysis_dir.resolve()
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


if __name__ == "__main__":
    sys.exit(main())
