"""Wait-on-sentinel runner: polls v2 state markers for the original SLURM job
to write its completion-or-failure marker, then exits 0 (success) or 1
(failure).

Per sentinel-system-v2 Phase 2.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


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
        time.sleep(backoff_s)
        backoff_s = min(backoff_s * 2, backoff_max)

    logger.error(
        "[wait_for_sentinel] %s: walltime cap (%d min) exceeded with neither "
        "completed nor failed marker; exit 1",
        args.rule_token,
        args.max_wait_minutes,
    )
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    sys.exit(main())
