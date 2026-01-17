# %%
"""
Standalone script for preparing TRITON-SWMM scenarios in a subprocess.

This script is designed to be executed as a subprocess (with or without srun)
to avoid PySwmm's MultiSimulationError when preparing multiple scenarios concurrently.

Usage:
    python -m TRITON_SWMM_toolkit.prepare_scenario_runner \
        --event-iloc 0 \
        --analysis-config /path/to/analysis.yaml \
        --system-config /path/to/system.yaml \
        [--overwrite-scenario] \
        [--rerun-swmm-hydro]

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
    """Main entry point for scenario preparation subprocess."""
    parser = argparse.ArgumentParser(
        description="Prepare a TRITON-SWMM scenario in a subprocess"
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
        "--overwrite-scenario",
        action="store_true",
        default=False,
        help="Overwrite scenario if it already exists",
    )
    parser.add_argument(
        "--rerun-swmm-hydro",
        action="store_true",
        default=False,
        help="Rerun SWMM hydrology model even if outputs exist",
    )
    parser.add_argument(
        "--compiled-model-dir",
        type=Path,
        required=False,
        help="(Optional) path to compiled TRITON-SWMM directory (mainly used for sensitivity analysis)",
    )

    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=False,
        help="(Optional) path to the analysis (mainly used for sensitivity analysis)",
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
    if args.compiled_model_dir:
        compiled_model_dir = args.compiled_model_dir
        assert args.compiled_model_dir.exists()
        logger.info(
            f"Assigning compiled model directory to analysis: {str(compiled_model_dir)}"
        )
    else:
        compiled_model_dir = None
    if args.analysis_dir:
        analysis_dir = args.analysis_dir
        assert args.analysis_dir.exists()
        logger.info(f"Assigning analysis to directory {str(analysis_dir)}")
    else:
        analysis_dir = None

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
            analysis_dir=analysis_dir,
            compiled_software_directory=compiled_model_dir,
        )

        logger.info(f"Preparing scenario {args.event_iloc}")
        scenario = analysis.scenarios[args.event_iloc]

        # Call the prepare_scenario method
        scenario.prepare_scenario(
            overwrite_scenario=args.overwrite_scenario,
            rerun_swmm_hydro_if_outputs_exist=args.rerun_swmm_hydro,
        )

        # Check if scenario was created successfully
        scenario.log.refresh()
        if scenario.log.scenario_creation_complete.get():
            logger.info(f"Scenario {args.event_iloc} prepared successfully")
            return 0
        else:
            logger.error(
                f"Scenario {args.event_iloc} preparation failed (log indicates incomplete)"
            )
            return 1

    except Exception as e:
        logger.error(f"Exception occurred during scenario preparation: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
