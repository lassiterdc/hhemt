"""Per-scenario deletion runner invoked by ``delete_scenario_*`` Snakemake rules.

Deletes the scenario's ``sims/{event_id}/`` subtree and writes the
``_status/_deleting/scenario_evt-{event_id}.flag`` sentinel via the
toolkit-managed :func:`write_status_flag` helper.

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
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    analysis_dir = args.analysis_dir.resolve()
    scenario_dir = analysis_dir / "sims" / args.event_id

    if scenario_dir.exists():
        print(f"[delete] removing {scenario_dir}", flush=True)
        fast_rmtree(scenario_dir)
    else:
        logger.warning(
            "sims/%s does not exist; recording sentinel anyway (idempotent delete).",
            args.event_id,
        )

    flag_path = (
        analysis_dir / "_status" / "_deleting" / f"scenario_evt-{args.event_id}.flag"
    )
    write_status_flag(
        flag_path,
        rule_name=f"delete_scenario_{args.event_id}",
        event_id=args.event_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
