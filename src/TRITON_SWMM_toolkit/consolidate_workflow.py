# %%
"""
Standalone script for consolidating TRITON-SWMM simulation outputs.

This script handles Phase 3 of the consolidated SLURM workflow:
1. Verify all simulations completed successfully
2. Consolidate TRITON and SWMM simulation summaries

This script is designed to run as a single task in a heterogeneous SLURM job,
after all array simulation tasks have completed.

Usage:
    python -m TRITON_SWMM_toolkit.consolidate_workflow \
        --system-config /path/to/system.yaml \
        --analysis-config /path/to/analysis.yaml \
        [--consolidate-outputs] \
        [--overwrite-outputs-if-already-created] \
        [--compression-level 5]

Exit codes:
    0: Success
    1: Failure (exception occurred, validation failed, or simulations failed)
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


def validate_resource_usage(analysis, logger=None):
    """
    Validate that actual resource usage matches expected configuration.

    Logs warnings if mismatches are detected between expected and actual
    compute resources (MPI tasks, OMP threads, GPUs, backend).

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis object containing scenario status
    logger : logging.Logger, optional
        Logger for writing warnings. If None, uses print statements.

    Returns
    -------
    bool
        True if all resources match expected values, False if any mismatches found
    """
    import pandas as pd

    if logger:
        logger.info("Validating actual vs expected resource usage...")
    else:
        print("Validating actual vs expected resource usage...")

    df_status = analysis.df_status

    # Skip validation if no log.out files were found (all actual values are None)
    if df_status["actual_nTasks"].isna().all():
        msg = (
            "No log.out files found - skipping resource validation. "
            "This is normal for simulations that haven't run yet or older TRITON versions."
        )
        if logger:
            logger.warning(msg)
        else:
            print(f"WARNING: {msg}")
        return True  # Return True since this is not a validation failure

    # Check for mismatches
    # For sensitivity analysis, each row has its own config values (run_mode, n_mpi_procs, etc.)
    # For regular analysis, use the analysis config
    mismatches = []

    for idx, row in df_status.iterrows():
        if not row["run_completed"]:
            continue  # Skip scenarios that didn't complete

        scenario_dir = row["scenario_directory"]
        issues = []

        # Get expected values from row if available (sensitivity analysis),
        # otherwise from analysis config (regular analysis)
        if "run_mode" in df_status.columns:
            # Sensitivity analysis: each row has its own config
            run_mode = row["run_mode"]
            expected_nTasks = row.get("n_mpi_procs", 1) or 1
            expected_omp_threads = row.get("n_omp_threads", 1) or 1
            expected_gpus = row.get("n_gpus", 0) or 0
        else:
            # Regular analysis: use analysis config
            cfg = analysis.cfg_analysis
            run_mode = cfg.run_mode
            expected_nTasks = cfg.n_mpi_procs or 1
            expected_omp_threads = cfg.n_omp_threads or 1
            expected_gpus = cfg.n_gpus or 0

        # Determine expected GPU backend
        if run_mode == "gpu":
            expected_gpu_backend = (
                analysis._system.cfg_system.gpu_compilation_backend or "unknown"
            )
        else:
            expected_gpu_backend = "none"

        # Check nTasks
        if pd.notna(row["actual_nTasks"]) and row["actual_nTasks"] != expected_nTasks:
            issues.append(
                f"  - MPI tasks: expected {expected_nTasks}, actual {row['actual_nTasks']}"
            )

        # Check OMP threads
        if (
            pd.notna(row["actual_omp_threads"])
            and row["actual_omp_threads"] != expected_omp_threads
        ):
            issues.append(
                f"  - OMP threads: expected {expected_omp_threads}, actual {row['actual_omp_threads']}"
            )

        # Check GPUs (for GPU mode)
        if run_mode == "gpu":
            if (
                pd.notna(row["actual_total_gpus"])
                and row["actual_total_gpus"] < expected_gpus
            ):
                issues.append(
                    f"  - Total GPUs: expected >={expected_gpus}, actual {row['actual_total_gpus']}"
                )

        # Check GPU backend
        if pd.notna(row["actual_gpu_backend"]):
            if run_mode == "gpu" and row["actual_gpu_backend"] == "none":
                issues.append(
                    f"  - GPU backend: expected {expected_gpu_backend}, actual {row['actual_gpu_backend']} (GPU not used!)"
                )
            elif run_mode != "gpu" and row["actual_gpu_backend"] != "none":
                issues.append(
                    f"  - GPU backend: expected 'none', actual {row['actual_gpu_backend']} (unexpected GPU usage)"
                )

        if issues:
            mismatch_msg = (
                f"\n⚠ Resource mismatch in scenario: {scenario_dir}\n"
                + "\n".join(issues)
            )
            mismatches.append(mismatch_msg)
            if logger:
                logger.warning(mismatch_msg)
            else:
                print(f"WARNING: {mismatch_msg}")

    if mismatches:
        summary = (
            f"\n{'='*70}\n"
            f"⚠ RESOURCE VALIDATION SUMMARY: {len(mismatches)} scenario(s) with mismatches\n"
            f"{'='*70}\n"
            "Possible causes:\n"
            "  1. SLURM/HPC scheduler allocated different resources than requested\n"
            "  2. Machine files overrode configuration (use TRITON_IGNORE_MACHINE_FILES)\n"
            "  3. Compilation used different backend than runtime configuration\n"
            "  4. Environment variables affected runtime behavior\n"
            f"{'='*70}"
        )
        if logger:
            logger.warning(summary)
        else:
            print(f"WARNING: {summary}")
        return False  # Validation failed
    else:
        msg = "✓ All scenarios used expected compute resources"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return True  # Validation passed


def main() -> int:
    """Main entry point for workflow consolidation."""
    parser = argparse.ArgumentParser(
        description="Consolidate TRITON-SWMM simulation outputs after ensemble run"
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
        "--which",
        type=str,
        default="both",
        choices=["TRITON", "SWMM", "both"],
        help="Which outputs to process: TRITON, SWMM, or both (only used if --process-timeseries)",
    )
    parser.add_argument(
        "--overwrite-outputs-if-already-created",
        action="store_true",
        default=False,
        help="Overwrite existing consolidated outputs",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Compression level for output files (0-9)",
    )
    parser.add_argument(
        "--consolidate-sensitivity-analysis-outputs",
        action="store_true",
        default=False,
        help="If True, consolidate subanalysis-level outputs into master analysis outputs (for sensitivity analysis)",
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

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=False,
        )

        # Phase 3a: Verify all simulations completed successfully
        logger.info("Verifying simulation completion status...")
        analysis._update_log()

        # Check if all simulations ran
        if not analysis.log.all_sims_run.get():
            logger.error("Not all simulations completed successfully")
            logger.error(f"Scenarios not run: {analysis.scenarios_not_run}")
            return 1

        logger.info("All simulations completed successfully")

        # Validate resource usage (skipped for subanalysis)
        if not analysis.cfg_analysis.is_subanalysis:
            validate_resource_usage(analysis, logger)

        # Check if all timeseries were processed
        if args.which in ["both", "TRITON"]:
            if not analysis.log.all_TRITON_timeseries_processed.get():
                logger.warning("Not all TRITON timeseries were processed")
                logger.warning(
                    f"Scenarios with unprocessed TRITON timeseries: {analysis.TRITON_time_series_not_processed}"
                )
        if args.which in ["both", "SWMM"]:
            if not analysis.log.all_SWMM_timeseries_processed.get():
                logger.warning("Not all SWMM timeseries were processed")
                logger.warning(
                    f"Scenarios with unprocessed SWMM timeseries: {analysis.SWMM_time_series_not_processed}"
                )

        # Phase 3b: Consolidate outputs
        if args.consolidate_sensitivity_analysis_outputs:
            logger.info(
                "Consolidating subanalysis-level outputs into master analysis outputs..."
            )
            try:
                analysis.sensitivity.consolidate_subanalysis_outputs(
                    which=args.which,
                    overwrite_outputs_if_already_created=args.overwrite_outputs_if_already_created,
                    verbose=True,
                    compression_level=args.compression_level,
                )
                logger.info("Sensitivity analysis consolidation completed successfully")
            except Exception as e:
                logger.error(f"Failed to consolidate sensitivity analysis outputs: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info("Consolidating TRITON and SWMM simulation summaries...")
            try:
                analysis.consolidate_TRITON_and_SWMM_simulation_summaries(
                    overwrite_outputs_if_already_created=args.overwrite_outputs_if_already_created,
                    verbose=True,
                    compression_level=args.compression_level,
                )
                logger.info("Consolidation completed successfully")
            except Exception as e:
                logger.error(f"Failed to consolidate outputs: {e}")
                logger.error(traceback.format_exc())
                return 1

        logger.info("Consolidation workflow completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during consolidation workflow: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
