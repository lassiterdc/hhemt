# %%
"""
Standalone script for consolidating TRITON-SWMM simulation outputs.

This script handles Phase 3 of the consolidated SLURM workflow:
1. Verify all simulations completed successfully
2. Consolidate TRITON and SWMM simulation summaries

This script is designed to run as a single task in a heterogeneous SLURM job,
after all array simulation tasks have completed.

Usage:
    python -m TRITON_SWMM_toolkit.consolidate_workflow \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--consolidate-outputs] \
        [--overwrite-if-exist] \
        [--compression-level 5]

Exit codes:
    0: Success
    1: Failure (exception occurred, validation failed, or simulations failed)
    2: Invalid arguments
"""

import sys
import argparse
from pathlib import Path
import traceback
import logging

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for workflow consolidation."""
    parser = argparse.ArgumentParser(
        description="Consolidate TRITON-SWMM simulation outputs after ensemble run"
    )
    parser.add_argument(
        "--system-config",
        type=Path,
        required=True,
        help="Path to system configuration YAML file",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        required=True,
        help="Path to analysis configuration YAML file",
    )
    parser.add_argument(
        "--consolidate-outputs",
        action="store_true",
        default=False,
        help="Consolidate TRITON and SWMM simulation summaries",
    )
    parser.add_argument(
        "--overwrite-if-exist",
        action="store_true",
        default=False,
        help="Overwrite existing consolidated outputs",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Compression level for output files (0-9)",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=False,
        help="(Optional) path to the analysis directory",
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code != 0:
            logger.error("Failed to parse command-line arguments")
            return 2
        return 0

    # Validate paths
    if not args.analysis_config.exists():
        logger.error(f"Analysis config not found: {args.analysis_config}")
        return 2
    if not args.system_config.exists():
        logger.error(f"System config not found: {args.system_config}")
        return 2
    if args.analysis_dir and not args.analysis_dir.exists():
        logger.error(f"Analysis directory not found: {args.analysis_dir}")
        return 2

    try:
        # Import here to avoid import errors if dependencies are missing
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system
        from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            analysis_dir=args.analysis_dir,
            skip_log_update=False,
        )

        # Phase 3a: Verify all simulations completed successfully
        logger.info("Verifying simulation completion status...")
        analysis._update_log()

        # Check if all simulations ran
        if not analysis.log.all_sims_run.get():
            logger.error("Not all simulations completed successfully")
            logger.error(f"Scenarios not run: {analysis.scenarios_not_run}")
            return 1

        logger.info("All simulations completed successfully")

        # Check if all timeseries were processed
        if not analysis.log.all_TRITON_timeseries_processed.get():
            logger.warning("Not all TRITON timeseries were processed")
            logger.warning(
                f"Scenarios with unprocessed TRITON timeseries: {analysis.TRITON_time_series_not_processed()}"
            )

        if not analysis.log.all_SWMM_timeseries_processed.get():
            logger.warning("Not all SWMM timeseries were processed")
            logger.warning(
                f"Scenarios with unprocessed SWMM timeseries: {analysis.SWMM_time_series_not_processed()}"
            )

        # Phase 3b: Consolidate outputs
        if args.consolidate_outputs:
            logger.info("Consolidating TRITON and SWMM simulation summaries...")
            try:
                analysis.consolidate_TRITON_and_SWMM_simulation_summaries(
                    overwrite_if_exist=args.overwrite_if_exist,
                    verbose=True,
                    compression_level=args.compression_level,
                )
                logger.info("Consolidation completed successfully")
            except Exception as e:
                logger.error(f"Failed to consolidate outputs: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info("Skipping consolidation (--consolidate-outputs not specified)")

        logger.info("Consolidation workflow completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during consolidation workflow: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
