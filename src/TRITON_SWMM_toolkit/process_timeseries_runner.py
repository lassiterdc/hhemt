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
        --overwrite-if-exist \
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

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


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
        "--overwrite-if-exist",
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

        logger.info(f"Processing timeseries for scenario {args.event_iloc}")
        scenario = TRITONSWMM_scenario(args.event_iloc, analysis)
        scenario.log.refresh()

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
                f"for scenario {args.event_iloc}. Log: {scenario.latest_simlog}"
            )
            return 1

        # Get the processing object and process the outputs
        run = scenario.run
        proc = TRITONSWMM_sim_post_processing(run)

        # Call the write_timeseries_outputs method
        proc.write_timeseries_outputs(
            which=args.which,  # type: ignore
            model_type=args.model_type,  # type: ignore
            clear_raw_outputs=args.clear_raw_outputs,
            overwrite_if_exist=args.overwrite_if_exist,
            verbose=True,
            compression_level=args.compression_level,
        )
        # Write log to disk (processing methods update in-memory log via add_sim_processing_entry)
        scenario.log.write()

        # Verify that processing was successful using file-based checks
        # (More reliable than log fields in multi-model scenarios where concurrent processes
        #  may overwrite each other's log updates)

        # Performance time series verification
        if args.which == "TRITON" or args.which == "both":
            perf_ok = False
            perf_path = None
            if args.model_type == "tritonswmm":
                perf_path = proc.scen_paths.output_tritonswmm_performance_timeseries
                perf_ok = perf_path.exists() if perf_path else False
            elif args.model_type == "triton":
                perf_path = proc.scen_paths.output_triton_only_performance_timeseries
                perf_ok = perf_path.exists() if perf_path else False
            if not perf_ok:
                logger.error(
                    f"Performance timeseries not created for scenario {args.event_iloc}. "
                    f"Expected file: {perf_path}"
                )
                return 1
        # TRITON outputs verification
        if args.which == "TRITON" or args.which == "both":
            triton_ok = False
            triton_path = None
            if args.model_type == "triton":
                triton_path = proc.scen_paths.output_triton_only_timeseries
                triton_ok = triton_path.exists() if triton_path else False
            elif args.model_type == "tritonswmm":
                triton_path = proc.scen_paths.output_tritonswmm_triton_timeseries
                triton_ok = triton_path.exists() if triton_path else False
            if not triton_ok:
                logger.error(
                    f"TRITON timeseries not created for scenario {args.event_iloc}. "
                    f"Expected file: {triton_path}"
                )
                return 1
        # SWMM outputs verification
        if args.which == "SWMM" or args.which == "both":
            swmm_ok = False
            node_path = None
            link_path = None
            if args.model_type == "swmm":
                node_path = proc.scen_paths.output_swmm_only_node_time_series
                link_path = proc.scen_paths.output_swmm_only_link_time_series
                swmm_ok = (
                    (node_path.exists() if node_path else False) and
                    (link_path.exists() if link_path else False)
                )
            elif args.model_type == "tritonswmm":
                node_path = proc.scen_paths.output_tritonswmm_node_time_series
                link_path = proc.scen_paths.output_tritonswmm_link_time_series
                swmm_ok = (
                    (node_path.exists() if node_path else False) and
                    (link_path.exists() if link_path else False)
                )
            if not swmm_ok:
                logger.error(
                    f"SWMM timeseries not created for scenario {args.event_iloc}. "
                    f"Expected files: {node_path}, {link_path}"
                )
                return 1

        logger.info(f"Scenario {args.event_iloc} timeseries processed successfully")

        # create summaries from full timeseries
        logger.info(f"Creating summaries for scenario {args.event_iloc}")
        proc.write_summary_outputs(
            which=args.which,  # type: ignore
            model_type=args.model_type,  # type: ignore
            overwrite_if_exist=args.overwrite_if_exist,
            verbose=True,
            compression_level=args.compression_level,
        )

        # Verify summary creation using file-based checks
        # (More reliable than log fields which can have stale reference issues)
        if args.which == "TRITON" or args.which == "both":
            triton_summary_ok = False
            summary_path = None
            if args.model_type == "triton":
                summary_path = proc.scen_paths.output_triton_only_summary
                triton_summary_ok = proc._already_written(summary_path)
            elif args.model_type == "tritonswmm":
                summary_path = proc.scen_paths.output_tritonswmm_triton_summary
                triton_summary_ok = proc._already_written(summary_path)
            if not triton_summary_ok:
                logger.error(
                    f"TRITON summary not created for scenario {args.event_iloc}. "
                    f"Expected file: {summary_path}"
                )
                return 1
        if args.which == "SWMM" or args.which == "both":
            swmm_summary_ok = False
            node_path = None
            link_path = None
            if args.model_type == "swmm":
                node_path = proc.scen_paths.output_swmm_only_node_summary
                link_path = proc.scen_paths.output_swmm_only_link_summary
                swmm_summary_ok = (
                    proc._already_written(node_path) and
                    proc._already_written(link_path)
                )
            elif args.model_type == "tritonswmm":
                node_path = proc.scen_paths.output_tritonswmm_node_summary
                link_path = proc.scen_paths.output_tritonswmm_link_summary
                swmm_summary_ok = (
                    proc._already_written(node_path) and
                    proc._already_written(link_path)
                )
            if not swmm_summary_ok:
                logger.error(
                    f"SWMM summaries not created for scenario {args.event_iloc}. "
                    f"Expected files: {node_path}, {link_path}"
                )
                return 1

        logger.info(f"Scenario {args.event_iloc} summaries created successfully")

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
        return 1


if __name__ == "__main__":
    sys.exit(main())
