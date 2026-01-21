# %%
"""
Standalone script for running a single TRITON-SWMM simulation with optional
scenario preparation and timeseries post-processing.

This script is designed to be executed as a SLURM array task, where each task
runs a single simulation identified by event_iloc (which maps to SLURM_ARRAY_TASK_ID).

Usage:
    python -m TRITON_SWMM_toolkit.run_single_simulation \
        --event-iloc 0 \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--prepare-scenario] \
        [--process-timeseries] \
        [--which both] \
        [--clear-raw-outputs] \
        [--overwrite-if-exist] \
        [--compression-level 5] \
        [--pickup-where-leftoff] \
        [--overwrite-scenario] \
        [--rerun-swmm-hydro]

Exit codes:
    0: Success
    1: Failure (exception occurred)
    2: Invalid arguments
"""

import sys
import argparse
import subprocess
from pathlib import Path
import traceback
import logging
from typing import Literal

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for single simulation execution."""
    parser = argparse.ArgumentParser(
        description="Run a single TRITON-SWMM simulation with optional scenario prep and post-processing"
    )
    parser.add_argument(
        "--event-iloc",
        type=int,
        required=True,
        help="Integer index of the weather event to simulate (maps to SLURM_ARRAY_TASK_ID)",
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
        "--prepare-scenario",
        action="store_true",
        default=False,
        help="Prepare the scenario before running simulation",
    )
    parser.add_argument(
        "--process-timeseries",
        action="store_true",
        default=False,
        help="Process timeseries outputs after simulation completes",
    )
    parser.add_argument(
        "--which",
        type=str,
        default="both",
        choices=["TRITON", "SWMM", "both"],
        help="Which outputs to process: TRITON, SWMM, or both (only used if --process-timeseries)",
    )
    parser.add_argument(
        "--clear-raw-outputs",
        action="store_true",
        default=False,
        help="Clear raw outputs after processing (only used if --process-timeseries)",
    )
    parser.add_argument(
        "--overwrite-if-exist",
        action="store_true",
        default=False,
        help="Overwrite processed outputs if they already exist (only used if --process-timeseries)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Compression level for output files (0-9, only used if --process-timeseries)",
    )
    parser.add_argument(
        "--pickup-where-leftoff",
        action="store_true",
        default=False,
        help="Resume simulation from last checkpoint if available",
    )
    parser.add_argument(
        "--overwrite-scenario",
        action="store_true",
        default=False,
        help="Overwrite scenario if it already exists (only used if --prepare-scenario)",
    )
    parser.add_argument(
        "--rerun-swmm-hydro",
        action="store_true",
        default=False,
        help="Rerun SWMM hydrology model even if outputs exist (only used if --prepare-scenario)",
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

        event_iloc = args.event_iloc
        logger.info(f"Processing simulation for event_iloc={event_iloc}")

        scenario = TRITONSWMM_scenario(event_iloc, analysis)

        # Step 1: Prepare scenario if requested
        if args.prepare_scenario:
            logger.info(f"[{event_iloc}] Preparing scenario...")
            launcher = scenario._create_subprocess_prepare_scenario_launcher(
                overwrite_scenario=args.overwrite_scenario,
                rerun_swmm_hydro_if_outputs_exist=args.rerun_swmm_hydro,
                verbose=True,
            )
            launcher()
            scenario.log.refresh()
            if not scenario.log.scenario_creation_complete.get():
                logger.error(f"[{event_iloc}] Scenario preparation failed")
                return 1
            logger.info(f"[{event_iloc}] Scenario preparation complete")
        else:
            # Verify scenario is already prepared
            scenario.log.refresh()
            if not scenario.log.scenario_creation_complete.get():
                logger.error(
                    f"[{event_iloc}] Scenario not prepared and --prepare-scenario not specified"
                )
                return 1

        # Step 2: Run simulation
        logger.info(f"[{event_iloc}] Running simulation...")
        launcher, finalize_sim = scenario.run._create_subprocess_sim_run_launcher(
            pickup_where_leftoff=args.pickup_where_leftoff,
            verbose=True,
        )
        # Launch the simulation (non-blocking)
        proc, start_time, sim_logfile, lf = launcher()
        # Wait for simulation to complete and update simlog
        finalize_sim(proc, start_time, sim_logfile, lf)

        # Check if simulation completed successfully
        scenario.log.refresh()
        if not scenario.sim_run_completed:
            logger.error(f"[{event_iloc}] Simulation did not complete successfully")
            logger.error(f"[{event_iloc}] Latest sim log: {scenario.latest_simlog}")
            return 1
        logger.info(f"[{event_iloc}] Simulation completed successfully")

        # Step 3: Process timeseries if requested
        if args.process_timeseries:
            logger.info(f"[{event_iloc}] Processing timeseries outputs...")
            run = scenario.run
            proc = TRITONSWMM_sim_post_processing(run)
            launcher = proc._create_subprocess_timeseries_processing_launcher(
                which=args.which,  # type: ignore
                clear_raw_outputs=args.clear_raw_outputs,
                overwrite_if_exist=args.overwrite_if_exist,
                verbose=True,
                compression_level=args.compression_level,
            )
            # Execute the launcher (which handles simlog updates internally)
            launcher()

            # Verify processing was successful
            scenario.log.refresh()
            proc = TRITONSWMM_sim_post_processing(run)

            if args.which == "TRITON" or args.which == "both":
                if not proc.TRITON_outputs_processed:
                    logger.error(f"[{event_iloc}] TRITON outputs not processed")
                    return 1
            if args.which == "SWMM" or args.which == "both":
                if not proc.SWMM_outputs_processed:
                    logger.error(f"[{event_iloc}] SWMM outputs not processed")
                    return 1

            logger.info(f"[{event_iloc}] Timeseries processing complete")

        logger.info(f"[{event_iloc}] All tasks completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during simulation execution: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
