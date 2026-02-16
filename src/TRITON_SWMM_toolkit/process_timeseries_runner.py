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
        --overwrite-outputs-if-already-created \
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

from TRITON_SWMM_toolkit.log_utils import log_workflow_context
import gc

# Memory profiling imports (always-on, minimal overhead)
import tracemalloc
import psutil
import os

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory_profile(description: str):
    """Log current memory usage with description."""
    mem_mb = get_memory_mb()
    logger.info(f"[MEMORY] {description}: {mem_mb:.1f} MB")


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
        "--model-type",
        type=str,
        required=True,
        choices=["triton", "tritonswmm", "swmm"],
        help="Model type to process (triton=TRITON-only, tritonswmm=coupled, swmm=SWMM-only)",
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
        "--overwrite-outputs-if-already-created",
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

    # Start always-on memory profiling (minimal overhead <1%)
    tracemalloc.start()
    gc.collect()
    initial_memory = get_memory_mb()
    logger.info("[MEMORY PROFILING] Enabled (overhead <1%)")
    logger.info(f"[MEMORY] Initial: {initial_memory:.1f} MB")
    snapshot_before = tracemalloc.take_snapshot()

    try:
        # Import here to avoid import errors if dependencies are missing
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system
        from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
        from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
        from TRITON_SWMM_toolkit.process_simulation import (
            TRITONSWMM_sim_post_processing,
        )

        # Log workflow context for traceability
        log_workflow_context(logger)

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
        scenario.log.refresh()

        log_memory_profile("After scenario initialization")

        model_types_enabled = scenario.run.model_types_enabled

        # Verify the specified model type is enabled
        if args.model_type not in model_types_enabled:
            logger.error(
                f"Model type '{args.model_type}' requested but not enabled "
                f"for scenario {args.event_iloc}. Enabled models: {model_types_enabled}"
            )
            return 1

        # Verify the specified model type has completed
        if not scenario.model_run_completed(args.model_type):
            logger.error(
                f"Model type '{args.model_type}' simulation not completed "
                f"for scenario {args.event_iloc}. Check model log files in {scenario.scen_paths.logs_dir}"
            )
            return 1

        # Get model-specific log
        model_log = scenario.get_log(args.model_type)

        # Get the processing object and process the outputs
        run = scenario.run
        proc = TRITONSWMM_sim_post_processing(run, model_log=model_log)

        # Memory checkpoint before timeseries processing
        log_memory_profile("Before write_timeseries_outputs")
        gc.collect()

        # Call the write_timeseries_outputs method
        proc.write_timeseries_outputs(
            which=args.which,  # type: ignore
            model_type=args.model_type,  # type: ignore
            clear_raw_outputs=args.clear_raw_outputs,
            overwrite_outputs_if_already_created=args.overwrite_outputs_if_already_created,
            verbose=True,
            compression_level=args.compression_level,
        )

        # Memory checkpoint after timeseries processing
        log_memory_profile("After write_timeseries_outputs")
        gc.collect()

        # Write model log to disk (processing methods update in-memory log)
        model_log.write()

        # Verify that processing was successful using log fields
        # (Now safe because each model has its own log - no race conditions!)

        # Performance time series verification (TRITON models only)
        if args.which == "TRITON" or args.which == "both":
            if args.model_type in ("triton", "tritonswmm"):
                if (
                    not model_log.performance_timeseries_written
                    or not model_log.performance_timeseries_written.get()
                ):
                    logger.error(
                        f"Performance timeseries not created for scenario {args.event_iloc}"
                    )
                    return 1
        # TRITON outputs verification (TRITON models only)
        if args.which == "TRITON" or args.which == "both":
            if args.model_type in ("triton", "tritonswmm"):
                if (
                    not model_log.TRITON_timeseries_written
                    or not model_log.TRITON_timeseries_written.get()
                ):
                    logger.error(
                        f"TRITON timeseries not created for scenario {args.event_iloc}"
                    )
                    return 1

        # SWMM outputs verification (SWMM models only)
        if args.which == "SWMM" or args.which == "both":
            if args.model_type in ("swmm", "tritonswmm"):
                node_ok = (
                    model_log.SWMM_node_timeseries_written
                    and model_log.SWMM_node_timeseries_written.get()
                )
                link_ok = (
                    model_log.SWMM_link_timeseries_written
                    and model_log.SWMM_link_timeseries_written.get()
                )
                if not (node_ok and link_ok):
                    logger.error(
                        f"SWMM timeseries not created for scenario {args.event_iloc}"
                    )
                    return 1

        logger.info(f"Scenario {args.event_iloc} timeseries processed successfully")

        # Memory checkpoint before summary generation
        log_memory_profile("Before write_summary_outputs")

        # create summaries from full timeseries
        logger.info(f"Creating summaries for scenario {args.event_iloc}")
        proc.write_summary_outputs(
            which=args.which,  # type: ignore
            model_type=args.model_type,  # type: ignore
            overwrite_outputs_if_already_created=args.overwrite_outputs_if_already_created,
            verbose=True,
            compression_level=args.compression_level,
        )

        # Verify summary creation using log fields
        # (Now safe because each model has its own log - no race conditions!)
        if args.which == "TRITON" or args.which == "both":
            if args.model_type in ("triton", "tritonswmm"):
                if (
                    not model_log.TRITON_summary_written
                    or not model_log.TRITON_summary_written.get()
                ):
                    logger.error(
                        f"TRITON summary not created for scenario {args.event_iloc}"
                    )
                    return 1
        if args.which == "SWMM" or args.which == "both":
            if args.model_type in ("swmm", "tritonswmm"):
                node_ok = (
                    model_log.SWMM_node_summary_written
                    and model_log.SWMM_node_summary_written.get()
                )
                link_ok = (
                    model_log.SWMM_link_summary_written
                    and model_log.SWMM_link_summary_written.get()
                )
                if not (node_ok and link_ok):
                    logger.error(
                        f"SWMM summaries not created for scenario {args.event_iloc}"
                    )
                    return 1

        # Memory checkpoint after summary generation
        log_memory_profile("After write_summary_outputs")

        logger.info(f"Scenario {args.event_iloc} summaries created successfully")

        # Final memory profiling summary
        gc.collect()
        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

        logger.info("[MEMORY] Top 10 memory allocations:")
        for stat in top_stats[:10]:
            logger.info(f"  {stat}")

        peak_memory = get_memory_mb()
        logger.info(f"[MEMORY] Peak: {peak_memory:.1f} MB (delta: +{peak_memory - initial_memory:.1f} MB)")
        logger.info("[MEMORY PROFILING] Complete - data available in logfile for analysis")

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

        # Log memory state at failure (helpful for OOM debugging)
        failure_memory = get_memory_mb()
        logger.error(f"[MEMORY] At failure: {failure_memory:.1f} MB")

        return 1
    finally:
        tracemalloc.stop()


if __name__ == "__main__":
    sys.exit(main())
