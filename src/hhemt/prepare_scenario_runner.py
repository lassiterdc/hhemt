# %%
"""
Standalone script for preparing TRITON-SWMM scenarios in a subprocess.

This script is designed to be executed as a subprocess (with or without srun)
to avoid PySwmm's MultiSimulationError when preparing multiple scenarios concurrently.

Usage:
    python -m hhemt.prepare_scenario_runner \
        --event-iloc 0 \
        --analysis-config /path/to/analysis.yaml \
        --system-config /path/to/system.yaml \
        [--overwrite-scenario-if-already-set-up] \
        [--rerun-swmm-hydro]

Exit codes:
    0: Success
    1: Failure (exception occurred)
    2: Invalid arguments
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from hhemt.log_utils import log_workflow_context
from hhemt.status_flags import emit_runner_flag as _emit_runner_flag

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
        "--hpc-system-config",
        type=Path,
        required=False,
        default=None,
        help="Optional path to the per-HPC-system configuration YAML file",
    )
    parser.add_argument(
        "--overwrite-scenario-if-already-set-up",
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
        "--event-id",
        type=str,
        default=None,
        help="Event id slug for the flag sidecar payload",
    )
    parser.add_argument(
        "--sa-id",
        type=str,
        default=None,
        help="Sub-analysis id for the flag sidecar payload (sensitivity)",
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
    if args.hpc_system_config is not None and not args.hpc_system_config.exists():
        logger.error(f"HPC system config not found: {args.hpc_system_config}")
        return 2

    try:
        # Import here to avoid import errors if dependencies are missing
        from hhemt.analysis import TRITONSWMM_analysis
        from hhemt.config.hpc_system import resolve_gpu_target
        from hhemt.config.loaders import load_analysis_config, load_hpc_system_config
        from hhemt.scenario import TRITONSWMM_scenario
        from hhemt.system import TRITONSWMM_system

        # Log workflow context for traceability
        log_workflow_context(logger)

        logger.info(f"Loading system configuration from {args.system_config}")
        # Parity with setup_workflow.py / sensitivity_analysis.py: gpu_hardware and
        # gpu_compilation_backend were retired off system_config onto the per-HPC-system
        # PartitionSpec. The prepare GPU gate (scenario.compilation_gpu_successful) resolves
        # the GPU build dir from these fields, so they MUST be injected here or a GPU prepare
        # finds build_dir_gpu=None and raises CompilationError("Log: missing"). Unlike
        # setup/simulation (which read --target-partition from the Snakefile), prepare is
        # always per-single-sub-analysis, so the ensemble partition is unambiguous and read
        # directly from the loaded analysis config — no Snakefile thread needed. The helper
        # returns (None, None) for null selectors, so CPU/local prepares are byte-identical.
        cfg_hpc = (
            load_hpc_system_config(args.hpc_system_config) if args.hpc_system_config else None
        )
        _ensemble_partition = load_analysis_config(args.analysis_config).hpc_ensemble_partition
        gpu_hardware, gpu_compilation_backend = resolve_gpu_target(cfg_hpc, _ensemble_partition)
        system = TRITONSWMM_system(
            args.system_config,
            gpu_hardware=gpu_hardware,
            gpu_compilation_backend=gpu_compilation_backend,
        )

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=True,
            is_main_orchestrator=False,
            hpc_system_config_yaml=args.hpc_system_config,
        )

        logger.info(f"Preparing scenario {args.event_iloc}")

        scenario = TRITONSWMM_scenario(args.event_iloc, analysis)

        # Call the prepare_scenario method
        scenario.prepare_scenario(
            overwrite_scenario_if_already_set_up=args.overwrite_scenario_if_already_set_up,
            rerun_swmm_hydro_if_outputs_exist=args.rerun_swmm_hydro,
        )

        # Verify preparation succeeded via scenario prep log
        scenario.log.refresh()
        if scenario.log.scenario_creation_complete.get():
            logger.info(f"Scenario {args.event_iloc} prepared successfully")
            _emit_runner_flag(args)
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
