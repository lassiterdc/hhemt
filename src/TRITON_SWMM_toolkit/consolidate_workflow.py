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
        [--compression-level 5]

Exit codes:
    0: Success
    1: Failure (exception occurred, validation failed, or simulations failed)
    2: Invalid arguments
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

from TRITON_SWMM_toolkit.log_utils import log_workflow_context
from TRITON_SWMM_toolkit.status_flags import emit_runner_flag as _emit_runner_flag

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
    tuple[bool, list[dict]]
        ``(passed, issues)`` where ``passed`` is True if all resources match
        expected values, False if any mismatches found. ``issues`` is the
        per-scenario flat list of mismatch records (each row carries
        ``scenario_dir``, ``resource``, ``expected``, ``actual``) so callers
        that want to render or aggregate the failures don't need to re-parse
        the log/print stream.
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
        return True, []  # Return True since this is not a validation failure

    # Check for mismatches
    # For sensitivity analysis, each row has its own config values (run_mode, n_mpi_procs, etc.)
    # For regular analysis, use the analysis config
    mismatches = []
    issues_flat: list[dict] = []  # structured per-mismatch records for downstream renderers

    for _idx, row in df_status.iterrows():
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
            expected_gpu_backend = analysis._system.cfg_system.gpu_compilation_backend or "unknown"
        else:
            expected_gpu_backend = "none"

        # Check nTasks
        if pd.notna(row["actual_nTasks"]) and row["actual_nTasks"] != expected_nTasks:
            issues.append(f"  - MPI tasks: expected {expected_nTasks}, actual {row['actual_nTasks']}")
            issues_flat.append(
                {
                    "scenario_dir": str(scenario_dir),
                    "scenario": Path(str(scenario_dir)).name,
                    "resource": "MPI tasks",
                    "expected": int(expected_nTasks),
                    "actual": int(row["actual_nTasks"]),
                    "detail": f"MPI tasks: expected {expected_nTasks}, actual {row['actual_nTasks']}",
                }
            )

        # Check OMP threads
        if pd.notna(row["actual_omp_threads"]) and row["actual_omp_threads"] != expected_omp_threads:
            issues.append(f"  - OMP threads: expected {expected_omp_threads}, actual {row['actual_omp_threads']}")
            issues_flat.append(
                {
                    "scenario_dir": str(scenario_dir),
                    "scenario": Path(str(scenario_dir)).name,
                    "resource": "OMP threads",
                    "expected": int(expected_omp_threads),
                    "actual": int(row["actual_omp_threads"]),
                    "detail": f"OMP threads: expected {expected_omp_threads}, actual {row['actual_omp_threads']}",
                }
            )

        # Check GPUs (for GPU mode)
        if run_mode == "gpu":
            if pd.notna(row["actual_total_gpus"]) and row["actual_total_gpus"] < expected_gpus:
                issues.append(f"  - Total GPUs: expected >={expected_gpus}, actual {row['actual_total_gpus']}")
                issues_flat.append(
                    {
                        "scenario_dir": str(scenario_dir),
                        "scenario": Path(str(scenario_dir)).name,
                        "resource": "Total GPUs",
                        "expected": f">={int(expected_gpus)}",
                        "actual": int(row["actual_total_gpus"]),
                        "detail": f"Total GPUs: expected >={expected_gpus}, actual {row['actual_total_gpus']}",
                    }
                )

        # Check GPU backend
        if pd.notna(row["actual_gpu_backend"]):
            if run_mode == "gpu" and row["actual_gpu_backend"] == "none":
                issues.append(
                    f"  - GPU backend: expected {expected_gpu_backend}, actual {row['actual_gpu_backend']} (GPU not used!)"  # noqa: E501
                )
                issues_flat.append(
                    {
                        "scenario_dir": str(scenario_dir),
                        "scenario": Path(str(scenario_dir)).name,
                        "resource": "GPU backend",
                        "expected": str(expected_gpu_backend),
                        "actual": str(row["actual_gpu_backend"]),
                        "detail": f"GPU backend: expected {expected_gpu_backend}, actual {row['actual_gpu_backend']} (GPU not used!)",  # noqa: E501
                    }
                )
            elif run_mode != "gpu" and row["actual_gpu_backend"] != "none":
                issues.append(
                    f"  - GPU backend: expected 'none', actual {row['actual_gpu_backend']} (unexpected GPU usage)"
                )
                issues_flat.append(
                    {
                        "scenario_dir": str(scenario_dir),
                        "scenario": Path(str(scenario_dir)).name,
                        "resource": "GPU backend",
                        "expected": "none",
                        "actual": str(row["actual_gpu_backend"]),
                        "detail": f"GPU backend: expected 'none', actual {row['actual_gpu_backend']} (unexpected GPU usage)",  # noqa: E501
                    }
                )

        if issues:
            mismatch_msg = f"\n⚠ Resource mismatch in scenario: {scenario_dir}\n" + "\n".join(issues)
            mismatches.append(mismatch_msg)
            if logger:
                logger.warning(mismatch_msg)
            else:
                print(f"WARNING: {mismatch_msg}")

    if mismatches:
        summary = (
            f"\n{'=' * 70}\n"
            f"⚠ RESOURCE VALIDATION SUMMARY: {len(mismatches)} scenario(s) with mismatches\n"
            f"{'=' * 70}\n"
            "Possible causes:\n"
            "  1. SLURM/HPC scheduler allocated different resources than requested\n"
            "  2. Machine files overrode configuration (use TRITON_IGNORE_MACHINE_FILES)\n"
            "  3. Compilation used different backend than runtime configuration\n"
            "  4. Environment variables affected runtime behavior\n"
            f"{'=' * 70}"
        )
        if logger:
            logger.warning(summary)
        else:
            print(f"WARNING: {summary}")
        return False, issues_flat  # Validation failed
    else:
        msg = "✓ All scenarios used expected compute resources"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return True, []  # Validation passed


def main() -> int:
    """Main entry point for workflow consolidation."""
    parser = argparse.ArgumentParser(description="Consolidate TRITON-SWMM simulation outputs after ensemble run")
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
        "--which",
        type=str,
        default="both",
        choices=["TRITON", "SWMM", "both"],
        help="Which outputs to process: TRITON, SWMM, or both (only used if --process-timeseries)",
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
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        default=False,
        help=(
            "If True, demote the 'all simulations completed' hard fail to a warning and consolidate only the "
            "completed scenarios. Set by the reprocess Snakefile generators when running against a partially-"
            "complete analysis dir. Canonical (non-reprocess) workflow invocations leave this False so missing "
            "sims still fail fast."
        ),
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
        "--sa-id",
        type=str,
        default=None,
        help="Sub-analysis id for the flag sidecar payload (sensitivity per-sa consolidate)",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Event id for the flag sidecar payload AND the per-scenario consolidate dispatch arm (multisim per-scenario consolidate; writes {scenario_dir}/_status/_du.json via du_sentinels.compute_and_write_scope_sentinel).",  # noqa: E501
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
        from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
        from TRITON_SWMM_toolkit.system import TRITONSWMM_system

        # Log workflow context for traceability
        log_workflow_context(logger)

        logger.info(f"Loading system configuration from {args.system_config}")
        system = TRITONSWMM_system(args.system_config)

        logger.info(f"Loading analysis configuration from {args.analysis_config}")
        analysis = TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=False,
            is_main_orchestrator=False,
            hpc_system_config_yaml=args.hpc_system_config,
        )

        # Per-scenario dispatch — writes {scenario_dir}/_status/_du.json via the DU sentinel
        # helper. Skips the analysis-level all_sims_run / all_timeseries_processed checks
        # below because those are analysis-scope invariants that would spuriously fail
        # for a per-scenario invocation while sibling scenarios are still in flight.
        if args.event_id is not None:
            from TRITON_SWMM_toolkit.du_sentinels import compute_and_write_scope_sentinel

            scenario_dir = analysis.analysis_paths.analysis_dir / "sims" / args.event_id
            if not scenario_dir.exists():
                logger.error(f"Per-scenario consolidate target does not exist: {scenario_dir}")
                return 1
            try:
                compute_and_write_scope_sentinel(
                    scenario_dir,
                    scope="scenario",
                    include_breakdown=True,
                )
                logger.info(f"Per-scenario DU sentinel written at {scenario_dir}/_status/_du.json")
            except Exception as e:
                logger.error(f"Failed to write per-scenario DU sentinel: {e}")
                logger.error(traceback.format_exc())
                return 1
            _emit_runner_flag(args)
            return 0

        # Phase 3a: Verify all simulations completed successfully
        logger.info("Verifying simulation completion status...")
        analysis._update_log()

        # Check if all simulations ran. Under --allow-incomplete (set by the reprocess Snakefile
        # generators), demote to a warning and continue — the reprocess Snakefile's `input:` directive
        # has already scope-limited the rule to completed sim flags, so the runner consolidates only
        # those completed scenarios and the un-run ones are surfaced via the operator-facing log
        # below (and the rendered report's Errors-and-Warnings sidebar).
        if not analysis.log.all_sims_run.get():
            if args.allow_incomplete:
                logger.warning(
                    "Not all simulations completed successfully; --allow-incomplete is set, proceeding with completed scenarios only"  # noqa: E501
                )
                logger.warning(f"Scenarios not run: {analysis.scenarios_not_run}")
            else:
                logger.error("Not all simulations completed successfully")
                logger.error(f"Scenarios not run: {analysis.scenarios_not_run}")
                return 1
        else:
            logger.info("All simulations completed successfully")

        # Validate resource usage (skipped for subanalysis)
        if not analysis.cfg_analysis.is_subanalysis:
            validate_resource_usage(analysis, logger)

        # Check if all timeseries were processed. The all_*_timeseries_processed log fields are computed
        # over the full sensitivity definition just like all_sims_run, so under --allow-incomplete the
        # warnings would otherwise fire spuriously for the out-of-scope (un-run) scenarios. Demote to
        # an info-level note in that case; the canonical workflow path (flag absent) keeps the warnings.
        if args.which in ["both", "TRITON"]:
            if not analysis.log.all_TRITON_timeseries_processed.get():
                if args.allow_incomplete:
                    logger.info(
                        "Skipping all-TRITON-timeseries-processed warning under --allow-incomplete (expected for reprocess against partial completion)"  # noqa: E501
                    )
                else:
                    logger.warning("Not all TRITON timeseries were processed")
                    logger.warning(
                        f"Scenarios with unprocessed TRITON timeseries: {analysis.TRITON_time_series_not_processed}"
                    )
        if args.which in ["both", "SWMM"]:
            if not analysis.log.all_SWMM_timeseries_processed.get():
                if args.allow_incomplete:
                    logger.info(
                        "Skipping all-SWMM-timeseries-processed warning under --allow-incomplete (expected for reprocess against partial completion)"  # noqa: E501
                    )
                else:
                    logger.warning("Not all SWMM timeseries were processed")
                    logger.warning(
                        f"Scenarios with unprocessed SWMM timeseries: {analysis.SWMM_time_series_not_processed}"
                    )

        # Phase 3b: Consolidate outputs
        if args.consolidate_sensitivity_analysis_outputs:
            logger.info("Consolidating subanalysis-level outputs into master sensitivity DataTree...")
            try:
                analysis.sensitivity.consolidate_sensitivity_datatree(
                    verbose=True,
                    compression_level=args.compression_level,
                )
                logger.info("Sensitivity analysis consolidation completed successfully")
            except Exception as e:
                logger.error(f"Failed to consolidate sensitivity analysis outputs: {e}")
                logger.error(traceback.format_exc())
                return 1
        else:
            logger.info("Assembling per-scenario summaries into master DataTree...")
            try:
                analysis.process.consolidate_to_datatree(
                    verbose=True,
                    compression_level=args.compression_level,
                )
                logger.info("DataTree consolidation completed successfully")
                # D6 — when this is a per-sub-analysis consolidate (--sa-id is passed
                # by the sensitivity-master `consolidate_{prefix}{sa_id}` rule, which
                # relies on fall-through to this branch), write a correctly-labeled
                # scope="sub_analysis" DU sentinel at the sub-analysis root. The
                # runner's `analysis` is built from the sub's config, so
                # `analysis.analysis_paths.analysis_dir` IS the sub-analysis dir.
                # Without this, the sub root carries a mislabeled scope="analysis"
                # sentinel written by consolidate_to_datatree
                # (processing_analysis.py:184). No separate `rule consolidate_subanalysis`
                # is needed — folding the write into the existing per-sub rule's
                # invocation avoids the NEW_RULE first-run rerun cost.
                if args.sa_id is not None:
                    from TRITON_SWMM_toolkit.du_sentinels import (
                        compute_and_write_scope_sentinel,
                    )

                    sub_analysis_dir = analysis.analysis_paths.analysis_dir
                    compute_and_write_scope_sentinel(
                        sub_analysis_dir,
                        scope="sub_analysis",
                        include_breakdown=True,
                    )
                    logger.info(
                        f"Sub-analysis DU sentinel written at "
                        f"{sub_analysis_dir}/_status/_du.json"
                    )
            except Exception as e:
                logger.error(f"Failed to consolidate to DataTree: {e}")
                logger.error(traceback.format_exc())
                return 1

        logger.info("Consolidation workflow completed successfully")
        # Option D (Class-Y resolution, renderer_io_provenance_audit): persist the
        # whole-tree ValidationReport as a single read-model artifact
        # ({analysis_dir}/validation_report.json) so errors_and_warnings.render() (and
        # the bundle re-render) reads ONE file instead of re-inspecting the tree at
        # render time. Runs at every analysis-level consolidation (sensitivity-master,
        # regular, and per-sa); the per-scenario --event-id path returned early above,
        # so no scenario-level report is written. Non-fatal: a persist failure must
        # never block an otherwise-successful consolidation.
        try:
            from TRITON_SWMM_toolkit.analysis_validation import persist_validation_report

            persist_validation_report(analysis)
            logger.info("Persisted validation_report.json read-model artifact")
        except Exception as e:
            logger.warning(f"validation_report.json persist failed (non-fatal): {e}")
        _emit_runner_flag(args)
        return 0

    except Exception as e:
        logger.error(f"Exception occurred during consolidation workflow: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
