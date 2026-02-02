# %%
"""
Standalone script for running a single simulation in a subprocess.

This script is designed to be executed as a subprocess (with or without srun)
to run a single simulation identified by event_iloc.

Supports three model types:
- triton: TRITON-only (2D hydrodynamic, no SWMM coupling)
- tritonswmm: Coupled TRITON-SWMM model (default)
- swmm: SWMM-only (standalone EPA SWMM)

Usage:
    python -m TRITON_SWMM_toolkit.run_simulation_runner \
        --event-iloc 0 \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--model-type tritonswmm] \
        [--pickup-where-leftoff]


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
    """Main entry point for simulation execution subprocess."""
    parser = argparse.ArgumentParser(
        description="Run a single simulation in a subprocess"
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
        "--model-type",
        type=str,
        choices=["triton", "tritonswmm", "swmm"],
        default="tritonswmm",
        help="Model type to run (default: tritonswmm)",
    )
    parser.add_argument(
        "--pickup-where-leftoff",
        action="store_true",
        default=False,
        help="Resume simulation from last checkpoint if available",
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
        from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=True,
        )

        event_iloc = args.event_iloc
        model_type = args.model_type
        logger.info(f"Running {model_type} simulation for event_iloc={event_iloc}")

        scenario = TRITONSWMM_scenario(event_iloc, analysis)

        # Verify scenario is prepared
        scenario.log.refresh()
        if not scenario.log.scenario_creation_complete.get():
            logger.error(
                f"[{event_iloc}] Scenario not prepared. Cannot run simulation."
            )
            return 1

        # Verify model-specific compilation
        if model_type == "triton":
            if not hasattr(system, "compilation_triton_only_successful"):
                logger.error(
                    f"[{event_iloc}] TRITON-only compilation check not implemented"
                )
                return 1
            if not system.compilation_triton_only_successful:
                logger.error(f"[{event_iloc}] TRITON-only has not been compiled")
                return 1
        elif model_type == "tritonswmm":
            if not system.compilation_successful:
                logger.error(f"[{event_iloc}] TRITON-SWMM has not been compiled")
                return 1
        elif model_type == "swmm":
            if not hasattr(system, "compilation_swmm_successful"):
                logger.error(f"[{event_iloc}] SWMM compilation check not implemented")
                return 1
            if not system.compilation_swmm_successful:
                logger.error(f"[{event_iloc}] SWMM has not been compiled")
                return 1

        # Get the run object and prepare the simulation command
        run = scenario.run
        logger.info(f"[{event_iloc}] Preparing {model_type} simulation...")

        # Use prepare_simulation_command to get the actual executable command
        # (NOT the recursive runner command)
        simprep_result = run.prepare_simulation_command(
            pickup_where_leftoff=args.pickup_where_leftoff,
            verbose=True,
            model_type=model_type,
        )

        # Check if simulation already completed
        if simprep_result is None:
            logger.info(
                f"[{event_iloc}] {model_type} simulation already completed, skipping execution"
            )
            logger.info(f"{model_type} simulation completed successfully")
            return 0

        # Unpack simulation command and metadata
        cmd, env, model_logfile, sim_start_reporting_tstep = simprep_result

        # Record simulation metadata in log
        run_mode = analysis.cfg_analysis.run_mode
        n_mpi_procs = analysis.cfg_analysis.n_mpi_procs or 1
        n_omp_threads = analysis.cfg_analysis.n_omp_threads or 1
        n_gpus = analysis.cfg_analysis.n_gpus if run_mode == "gpu" else 0

        import TRITON_SWMM_toolkit.utils as ut

        sim_datetime = ut.current_datetime_string()

        scenario.log.add_sim_entry(
            sim_datetime=sim_datetime,
            sim_start_reporting_tstep=sim_start_reporting_tstep,
            tritonswmm_logfile=model_logfile,  # Note: field name kept for backward compatibility
            time_elapsed_s=0,
            status="not started",
            run_mode=(
                run_mode if model_type != "swmm" else "serial"
            ),  # SWMM is always serial
            cmd=" ".join(cmd),
            n_mpi_procs=n_mpi_procs,
            n_omp_threads=n_omp_threads,
            n_gpus=n_gpus if model_type != "swmm" else 0,  # SWMM has no GPU support
            env=env,
        )

        # Launch the executable (not the runner!)
        logger.info(f"[{event_iloc}] Running {model_type} simulation...")
        logger.info(f"[{event_iloc}] Command: {' '.join(cmd)}")
        logger.info(f"[{event_iloc}] Log file: {model_logfile}")

        import time
        import subprocess
        import os

        start_time = time.time()
        with open(model_logfile, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, **env},
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
            _rc = proc.wait()  # Return code checked via status below

        # Update simulation log with results
        end_time = time.time()
        elapsed = end_time - start_time

        # Check simulation status
        if model_type == "swmm":
            # SWMM: Exit code check (future: parse .rpt file for success message)
            status = (
                "simulation completed"
                if _rc == 0
                else "simulation started but did not finish"
            )
        else:
            # TRITON and TRITON-SWMM: Use log file checking
            status, _last_cfg = run._check_simulation_run_status()

        # Update log entry
        log_dic = scenario.latest_simlog
        log_dic["time_elapsed_s"] = elapsed
        log_dic["status"] = status
        scenario.log.add_sim_entry(**log_dic)

        # Check if simulation completed successfully
        scenario.log.refresh()
        if not scenario.sim_run_completed:
            logger.error(f"[{event_iloc}] Simulation did not complete successfully")
            logger.error(f"[{event_iloc}] Latest sim log: {scenario.latest_simlog}")
            return 1

        logger.info(f"Simulation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during simulation execution: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
