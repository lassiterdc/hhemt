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
from TRITON_SWMM_toolkit.status_flags import emit_runner_flag as _emit_runner_flag


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
        "--hpc-system-config",
        type=Path,
        required=False,
        default=None,
        help="Optional path to the per-HPC-system configuration YAML file",
    )
    parser.add_argument(
        "--target-partition",
        type=str,
        required=False,
        default=None,
        help=(
            "Phase-4 (4c): partition whose PartitionSpec GPU hardware/backend is "
            "resolved + injected into TRITONSWMM_system for compilation (the ensemble "
            "/ sim partition — the binary's run target). Optional; absent => CPU/no-GPU."
        ),
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
    parser.add_argument(
        "--flag-output",
        type=Path,
        default=None,
        help="Path to the _status/*.flag marker to write on success (toolkit-managed; optional for legacy CLI use)",
    )
    parser.add_argument(
        "--rule-name",
        type=str,
        default=None,
        help="Snakemake rule name for the flag sidecar payload",
    )
    parser.add_argument(
        "--target-id",
        type=str,
        default=None,
        help="UniqueSystemTarget id for the flag sidecar payload (sensitivity per-target setup)",
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
    if args.hpc_system_config is not None and not args.hpc_system_config.exists():
        logger.error(f"HPC system config not found: {args.hpc_system_config}")
        return 2

    try:
        # Import here to avoid import errors if dependencies are missing
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system
        from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
        from TRITON_SWMM_toolkit.config.loaders import load_hpc_system_config
        from TRITON_SWMM_toolkit.config.hpc_system import resolve_gpu_target, resolve_additional_modules
        from TRITON_SWMM_toolkit.exceptions import ConfigurationError

        # Log workflow context for traceability
        log_workflow_context(logger)

        logger.info(f"Loading system configuration from {args.system_config}")
        # Phase-4 (4c): resolve GPU hardware/backend + module list from the
        # per-HPC-system config + the target (ensemble/sim) partition, and inject
        # them into TRITONSWMM_system (they were retired off system_config).
        cfg_hpc = load_hpc_system_config(args.hpc_system_config) if args.hpc_system_config else None
        gpu_hardware, gpu_compilation_backend = resolve_gpu_target(cfg_hpc, args.target_partition)
        additional_modules = resolve_additional_modules(cfg_hpc)
        system = TRITONSWMM_system(
            args.system_config,
            gpu_hardware=gpu_hardware,
            gpu_compilation_backend=gpu_compilation_backend,
            additional_modules=additional_modules,
        )

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            hpc_system_config_yaml=args.hpc_system_config,
            skip_log_update=False,
            is_main_orchestrator=False,
        )

        # Fail-loud guard (Phase-4 4c): a GPU run whose target partition resolves no
        # gpu_compilation_backend would silently skip the GPU build (gpu_suffix="",
        # build_dir_gpu=None). Convert that into a named preflight error rather than a
        # silent CPU-only compile that later fails the GPU sims.
        if getattr(analysis.cfg_analysis, "n_gpus", 0) and not gpu_compilation_backend:
            raise ConfigurationError(
                field="hpc_ensemble_partition",
                message=(
                    f"GPU run requested (n_gpus={getattr(analysis.cfg_analysis, 'n_gpus', 0)}) "
                    f"but target partition '{args.target_partition}' declares no "
                    f"gpu_compilation_backend in the hpc_system_config. Declare "
                    f"gpu_hardware + gpu_compilation_backend on that partition's PartitionSpec."
                ),
                config_path=str(args.hpc_system_config) if args.hpc_system_config else None,
            )
        any_compile = (
            args.compile_triton_swmm or args.compile_triton_only or args.compile_swmm
        )
        if (not any_compile) and not (args.process_system_inputs):
            logger.info(
                "No compilation or processing flags were passed. Doing nothing."
            )
            _emit_runner_flag(args)
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
                    if system.gpu_compilation_backend:
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
                if system.gpu_compilation_backend:
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
                    system.gpu_compilation_backend
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
        _emit_runner_flag(args)
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during setup workflow: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
