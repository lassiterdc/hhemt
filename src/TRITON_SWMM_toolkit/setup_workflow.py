# %%
"""
Standalone script for setting up the TRITON-SWMM workflow.

This script handles Phase 1 of the consolidated SLURM workflow:
1. Process system-level inputs (DEM, Mannings files)
2. Compile TRITON-SWMM

This script is designed to run as a single task in a heterogeneous SLURM job,
before the array of simulation tasks begins.

Usage:
    python -m TRITON_SWMM_toolkit.setup_workflow \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--process-system-inputs] \
        [--overwrite-system-inputs] \
        [--compile-triton-swmm] \
        [--recompile-if-already-done]

Exit codes:
    0: Success
    1: Failure (exception occurred or validation failed)
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
    """Main entry point for workflow setup."""
    parser = argparse.ArgumentParser(
        description="Setup TRITON-SWMM workflow: process system inputs and compile"
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
        "--process-system-inputs",
        action="store_true",
        default=True,
        help="Process system-level inputs (DEM, Mannings files)",
    )
    parser.add_argument(
        "--overwrite-system-inputs",
        action="store_true",
        default=False,
        help="Overwrite existing system input files (only used if --process-system-inputs)",
    )
    parser.add_argument(
        "--compile-triton-swmm",
        action="store_true",
        default=True,
        help="Compile TRITON-SWMM",
    )
    parser.add_argument(
        "--recompile-if-already-done",
        action="store_true",
        default=False,
        help="Recompile even if already compiled successfully (only used if --compile-triton-swmm)",
    )
    parser.add_argument(
        "--compiled-model-dir",
        type=Path,
        required=False,
        default=None,
        help="(Optional) path to compiled TRITON-SWMM directory",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=False,
        default=None,
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

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            analysis_dir=args.analysis_dir,
            compiled_TRITONSWMM_directory=args.compiled_model_dir,
            skip_log_update=False,
        )

        # Phase 1a: Process system-level inputs
        if args.process_system_inputs:
            logger.info("Processing system-level inputs...")
            try:
                system.process_system_level_inputs(
                    overwrite_if_exists=args.overwrite_system_inputs,
                    verbose=True,
                )
                logger.info("System-level inputs processed successfully")
            except Exception as e:
                logger.error(f"Failed to process system-level inputs: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info(
                "Skipping system-level input processing (--process-system-inputs not specified)"
            )

        # Phase 1b: Compile TRITON-SWMM
        if args.compile_triton_swmm:
            logger.info("Compiling TRITON-SWMM...")
            try:
                analysis.compile_TRITON_SWMM(
                    recompile_if_already_done_successfully=args.recompile_if_already_done,
                    verbose=True,
                )
                # Verify compilation was successful
                if not analysis.compilation_successful:
                    logger.error("TRITON-SWMM compilation failed")
                    logger.error(
                        f"Compilation log:\n{analysis.retrieve_compilation_log()}"
                    )
                    return 1
                logger.info("TRITON-SWMM compiled successfully")
            except Exception as e:
                logger.error(f"Failed to compile TRITON-SWMM: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info(
                "Skipping TRITON-SWMM compilation (--compile-triton-swmm not specified)"
            )
            # Verify compilation is already done
            if not analysis.compilation_successful:
                logger.error(
                    "TRITON-SWMM has not been compiled and --compile-triton-swmm not specified"
                )
                return 1
            logger.info("TRITON-SWMM already compiled successfully")

        logger.info("Setup workflow completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during setup workflow: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
