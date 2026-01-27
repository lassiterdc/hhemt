# %%
"""
Standalone script for processing TRITON-SWMM scenario timeseries outputs in a subprocess.

This script is designed to be executed as a subprocess (with or without srun)
to avoid potential conflicts when processing multiple scenarios' outputs concurrently.

Usage:
    python -m TRITON_SWMM_toolkit.process_timeseries_runner \
        --event-iloc 0 \
        --analysis-config /path/to/analysis.yaml \
        --system-config /path/to/system.yaml \
        --which both \
        --clear-raw-outputs \
        --overwrite-if-exist \
        --compression-level 5 \


Exit codes:
    0: Success
    1: Failure (exception occurred)
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


def main():
    """Main entry point for timeseries processing subprocess."""
    parser = argparse.ArgumentParser(
        description="Process TRITON-SWMM scenario timeseries outputs in a subprocess"
    )
    parser.add_argument(
        "--event-iloc",
        type=int,
        required=True,
        help="Integer index of the weather event to process",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        required=True,
        help="Path to analysis configuration YAML file",
    )
    parser.add_argument(
        "--system-config",
        type=Path,
        required=True,
        help="Path to system configuration YAML file",
    )
    parser.add_argument(
        "--which",
        type=str,
        default="both",
        choices=["TRITON", "SWMM", "both"],
        help="Which outputs to process: TRITON, SWMM, or both",
    )
    parser.add_argument(
        "--clear-raw-outputs",
        action="store_true",
        default=False,
        help="Clear raw outputs after processing",
    )
    parser.add_argument(
        "--overwrite-if-exist",
        action="store_true",
        default=False,
        help="Overwrite processed outputs if they already exist",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Compression level for output files (0-9)",
    )
    parser.add_argument(
        "--clear-full-timeseries",
        action="store_true",
        default=False,
        help="Clear full timeseries files after creating summaries (to save disk space)",
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

    try:
        # Import here to avoid import errors if dependencies are missing
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system
        from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
        from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
        from TRITON_SWMM_toolkit.process_simulation import (
            TRITONSWMM_sim_post_processing,
        )

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=True,
        )

        logger.info(f"Processing timeseries for scenario {args.event_iloc}")
        scenario = TRITONSWMM_scenario(args.event_iloc, analysis)

        # Verify that the simulation has been run
        if not scenario.sim_run_completed:
            logger.error(
                f"Simulation not completed for scenario {args.event_iloc}. "
                f"Cannot process outputs. Log: {scenario.latest_simlog}"
            )
            return 1

        # Get the processing object and process the outputs
        run = scenario.run
        proc = TRITONSWMM_sim_post_processing(run)

        # Call the write_timeseries_outputs method
        proc.write_timeseries_outputs(
            which=args.which,  # type: ignore
            clear_raw_outputs=args.clear_raw_outputs,
            overwrite_if_exist=args.overwrite_if_exist,
            verbose=True,
            compression_level=args.compression_level,
        )
        proc._export_TRITONSWMM_performance_tseries(
            comp_level=args.compression_level, verbose=True
        )

        # Verify that processing was successful
        scenario.log.refresh()
        if not proc.TRITONSWMM_performance_timeseries_written:
            logger.error(
                f"TRITONSWMM performance time series not processed for scenario {args.event_iloc}"
            )
            return 1
        if args.which == "TRITON" or args.which == "both":
            if not proc.TRITON_outputs_processed:
                logger.error(
                    f"TRITON outputs not processed for scenario {args.event_iloc}"
                )
                return 1
        if args.which == "SWMM" or args.which == "both":
            if not proc.SWMM_outputs_processed:
                logger.error(
                    f"SWMM outputs not processed for scenario {args.event_iloc}"
                )
                return 1

        logger.info(f"Scenario {args.event_iloc} timeseries processed successfully")

        # create summaries from full timeseries
        logger.info(f"Creating summaries for scenario {args.event_iloc}")
        proc.write_summary_outputs(
            which=args.which,  # type: ignore
            overwrite_if_exist=args.overwrite_if_exist,
            verbose=True,
            compression_level=args.compression_level,
        )

        # Verify summary creation
        scenario.log.refresh()
        if args.which == "TRITON" or args.which == "both":
            if not proc.TRITON_summary_processed:
                logger.error(
                    f"TRITON summary not created for scenario {args.event_iloc}"
                )
                return 1
        if args.which == "SWMM" or args.which == "both":
            if not proc.SWMM_summary_processed:
                logger.error(
                    f"SWMM summaries not created for scenario {args.event_iloc}"
                )
                return 1

        logger.info(f"Scenario {args.event_iloc} summaries created successfully")

        # Optionally clear full timeseries files to save disk space
        if args.clear_full_timeseries:
            logger.info(f"Clearing full timeseries for scenario {args.event_iloc}")
            proc._clear_full_timeseries_outputs(
                which=args.which,  # type: ignore
                verbose=True,
            )
            scenario.log.refresh()
            logger.info(f"Full timeseries cleared for scenario {args.event_iloc}")

        return 0

    except Exception as e:
        logger.error(f"Exception occurred during timeseries processing: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
