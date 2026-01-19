# %%
"""
Standalone script for running a single TRITON-SWMM simulation in a subprocess.

This script is designed to be executed as a subprocess (with or without srun)
to run a single simulation identified by event_iloc.

Usage:
    python -m TRITON_SWMM_toolkit.run_simulation_runner \
        --event-iloc 0 \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--pickup-where-leftoff] \
        [--compiled-model-dir /path/to/compiled] \
        [--analysis-dir /path/to/analysis]

Exit codes:
    0: Success
    1: Failure (exception occurred)
    2: Invalid arguments
"""

import sys
import argparse
import subprocess
import os
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
    """Main entry point for simulation execution subprocess."""
    parser = argparse.ArgumentParser(
        description="Run a single TRITON-SWMM simulation in a subprocess"
    )
    parser.add_argument(
        "--event-iloc",
        type=int,
        required=True,
        help="Integer index of the weather event to simulate",
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
        "--pickup-where-leftoff",
        action="store_true",
        default=False,
        help="Resume simulation from last checkpoint if available",
    )
    parser.add_argument(
        "--compiled-model-dir",
        type=Path,
        required=False,
        help="(Optional) path to compiled TRITON-SWMM directory",
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
    if args.compiled_model_dir and not args.compiled_model_dir.exists():
        logger.error(f"Compiled model directory not found: {args.compiled_model_dir}")
        return 2
    if args.analysis_dir and not args.analysis_dir.exists():
        logger.error(f"Analysis directory not found: {args.analysis_dir}")
        return 2

    try:
        # Import here to avoid import errors if dependencies are missing
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system
        from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
        from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            analysis_dir=args.analysis_dir,
            compiled_software_directory=args.compiled_model_dir,
            skip_log_update=True,
        )

        event_iloc = args.event_iloc
        logger.info(f"Running simulation for event_iloc={event_iloc}")

        scenario = TRITONSWMM_scenario(event_iloc, analysis)

        # Verify scenario is prepared
        scenario.log.refresh()
        if not scenario.log.scenario_creation_complete.get():
            logger.error(
                f"[{event_iloc}] Scenario not prepared. Cannot run simulation."
            )
            return 1

        # Verify compilation
        if not analysis.compilation_successful:
            logger.error(f"[{event_iloc}] TRITON-SWMM has not been compiled")
            return 1

        # Get the run object and prepare the simulation command
        run = scenario.run
        logger.info(f"[{event_iloc}] Preparing simulation command...")

        simprep_result = run.prepare_simulation_command(
            pickup_where_leftoff=args.pickup_where_leftoff,
            using_srun=False,
            verbose=True,
        )

        if simprep_result is None:
            logger.info(
                f"[{event_iloc}] Simulation already completed, skipping execution"
            )
            return 0

        cmd, env, tritonswmm_logfile, sim_start_reporting_tstep = simprep_result

        # Execute the simulation command
        logger.info(f"[{event_iloc}] Executing simulation command...")
        logger.info(f"[{event_iloc}] Log file: {tritonswmm_logfile}")

        with open(tritonswmm_logfile, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, **env},
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
            proc.wait()

        logger.info(f"[{event_iloc}] Simulation process completed")

        # Check if simulation completed successfully
        scenario.log.refresh()
        if not scenario.sim_run_completed:
            logger.error(f"[{event_iloc}] Simulation did not complete successfully")
            logger.error(f"[{event_iloc}] Latest sim log: {scenario.latest_simlog}")
            return 1

        logger.info(f"[{event_iloc}] Simulation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during simulation execution: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
