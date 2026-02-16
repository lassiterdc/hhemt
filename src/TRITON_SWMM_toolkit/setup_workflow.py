# %%
"""
Standalone script for setting up the TRITON-SWMM workflow.

This script handles Phase 1 of the consolidated SLURM workflow:
1. Process system-level inputs (DEM, Mannings files)
2. Compile enabled model types (TRITON-SWMM, TRITON-only, SWMM)

This script is designed to run as a single task in a heterogeneous SLURM job,
before the array of simulation tasks begins.

Usage:
    python -m TRITON_SWMM_toolkit.setup_workflow \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--process-system-inputs] \
        [--overwrite-system-inputs] \
        [--compile-triton-swmm] \
        [--compile-triton-only] \
        [--compile-swmm] \
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

from TRITON_SWMM_toolkit.log_utils import log_workflow_context


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
        default=False,
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
        default=False,
        help="Compile TRITON-SWMM (coupled model)",
    )
    parser.add_argument(
        "--compile-triton-only",
        action="store_true",
        default=False,
        help="Compile TRITON-only (no SWMM coupling)",
    )
    parser.add_argument(
        "--compile-swmm",
        action="store_true",
        default=False,
        help="Compile standalone SWMM",
    )
    parser.add_argument(
        "--recompile-if-already-done",
        action="store_true",
        default=False,
        help="Recompile even if already compiled successfully (applies to all compilation flags)",
    )
    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code != 0:
            logger.error("Failed to parse command-line arguments")
            return 2
        return 2

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

        # Log workflow context for traceability
        log_workflow_context(logger)

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=False,
        )
        any_compile = (
            args.compile_triton_swmm or args.compile_triton_only or args.compile_swmm
        )
        if (not any_compile) and not (args.process_system_inputs):
            logger.info(
                "No compilation or processing flags were passed. Doing nothing."
            )
            return 0

        # Phase 1a: Process system-level inputs
        if args.process_system_inputs:
            logger.info("Processing system-level inputs...")
            try:
                system.process_system_level_inputs(
                    overwrite_outputs_if_already_created=args.overwrite_system_inputs,
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

        # Phase 1b: Compile TRITON-SWMM (coupled model)
        if args.compile_triton_swmm:
            logger.info("Compiling TRITON-SWMM (coupled model)...")
            try:
                system.compile_TRITON_SWMM(
                    recompile_if_already_done_successfully=args.recompile_if_already_done,
                    verbose=True,
                )

                # Verify compilation was successful
                if len(system.available_backends) == 0:
                    logger.error("TRITON-SWMM: No backends compiled successfully")
                    logger.error(f"CPU log:\n{system.retrieve_compilation_log('cpu')}")
                    if system.cfg_system.gpu_compilation_backend:
                        logger.error(
                            f"GPU log:\n{system.retrieve_compilation_log('gpu')}"
                        )
                    return 1
                logger.info(
                    f"TRITON-SWMM available backends: {', '.join(system.available_backends)}"
                )
            except Exception as e:
                logger.error(f"Failed to compile TRITON-SWMM: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info(
                "Skipping TRITON-SWMM compilation (--compile-triton-swmm not specified)"
            )
            # Verify compilation if model is enabled
            if (
                system.cfg_system.toggle_tritonswmm_model
                and not system.compilation_successful
            ):
                logger.error(
                    "TRITON-SWMM is enabled but not compiled and --compile-triton-swmm not specified"
                )
                return 1

        # Phase 1c: Compile TRITON-only (no SWMM coupling)
        if args.compile_triton_only:
            logger.info("Compiling TRITON-only (no SWMM coupling)...")
            try:
                backends = []
                if system.cfg_system.gpu_compilation_backend:
                    backends = ["cpu", "gpu"]
                else:
                    backends = ["cpu"]

                system.compile_TRITON_only(
                    backends=backends,
                    recompile_if_already_done_successfully=args.recompile_if_already_done,
                    verbose=True,
                )

                # Verify compilation was successful
                if not system.compilation_triton_only_cpu_successful:
                    logger.error("TRITON-only CPU compilation failed")
                    return 1
                if (
                    system.cfg_system.gpu_compilation_backend
                    and not system.compilation_triton_only_gpu_successful
                ):
                    logger.error("TRITON-only GPU compilation failed")
                    return 1
                logger.info("TRITON-only compiled successfully")
            except Exception as e:
                logger.error(f"Failed to compile TRITON-only: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info(
                "Skipping TRITON-only compilation (--compile-triton-only not specified)"
            )
            # Verify compilation if model is enabled
            if (
                system.cfg_system.toggle_triton_model
                and not system.compilation_triton_only_successful
            ):
                logger.error(
                    "TRITON-only is enabled but not compiled and --compile-triton-only not specified"
                )
                return 1

        # Phase 1d: Compile standalone SWMM
        if args.compile_swmm:
            logger.info("Compiling standalone SWMM...")
            try:
                system.compile_SWMM(
                    recompile_if_already_done_successfully=args.recompile_if_already_done,
                    verbose=True,
                )

                # Verify compilation was successful
                if not system.compilation_swmm_successful:
                    logger.error("SWMM compilation failed")
                    return 1
                logger.info("SWMM compiled successfully")
            except Exception as e:
                logger.error(f"Failed to compile SWMM: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info("Skipping SWMM compilation (--compile-swmm not specified)")
            # Verify compilation if model is enabled
            if (
                system.cfg_system.toggle_swmm_model
                and not system.compilation_swmm_successful
            ):
                logger.error(
                    "SWMM is enabled but not compiled and --compile-swmm not specified"
                )
                return 1

        logger.info("Setup workflow completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during setup workflow: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
