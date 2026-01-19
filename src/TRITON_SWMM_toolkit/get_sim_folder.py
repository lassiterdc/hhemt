# %%
"""
Utility script to retrieve the simulation folder path for a given event_iloc.

This is used by SLURM job array scripts to determine where to save logs.

Usage:
    python -m TRITON_SWMM_toolkit.get_sim_folder \
        --event-iloc 0 \
        --analysis-config /path/to/analysis.yaml \
        --system-config /path/to/system.yaml

Output:
    Prints the absolute path to the simulation folder to stdout
"""

import sys
import argparse
from pathlib import Path
import logging

# Configure logging to stderr (so stdout is clean for the path)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for retrieving simulation folder."""
    parser = argparse.ArgumentParser(
        description="Get the simulation folder path for a given event_iloc"
    )
    parser.add_argument(
        "--event-iloc",
        type=int,
        required=True,
        help="Integer index of the weather event",
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

        system = TRITONSWMM_system(args.system_config)
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=True,
        )

        scenario = TRITONSWMM_scenario(args.event_iloc, analysis)
        sim_folder = scenario.scen_paths.sim_folder

        # Print to stdout (clean output for shell script)
        print(str(sim_folder))
        return 0

    except Exception as e:
        logger.error(f"Exception occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# %%
