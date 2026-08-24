"""
Module for exporting scenario status DataFrame to CSV after all simulations.

This module provides functionality to export the df_status() DataFrame to a CSV file
after all simulations complete, regardless of their success or failure. This enables
debugging by identifying which scenarios failed and why.

The exported CSV includes performance breakdown columns (perf_Total, perf_Compute,
perf_SWMM, perf_MPI, etc.) drawn from the processed performance summary dataset.
These columns are populated only for rows where output processing completed; they
are NaN for SWMM model type rows (no TRITON performance dataset) and for any scenario
where processing did not finish.

Additionally writes a workflow_summary.md file with get_workflow_status() output
and optionally includes HPC partition information for debugging resource allocation issues.
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hhemt.analysis as anlysis
from hhemt.log_utils import log_workflow_context
from hhemt.system import TRITONSWMM_system

# Configure logging to stderr (will be redirected to logfile by Snakefile)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def parse_partition_limits(scontrol_output: str) -> dict:
    """
    Parse scontrol show partition output into structured data.

    Parameters
    ----------
    scontrol_output : str
        Output from `scontrol show partition -o`

    Returns
    -------
    dict
        Dictionary mapping partition name to dict of key limits
    """
    partitions = {}
    for line in scontrol_output.strip().split("\n"):
        if not line.strip():
            continue

        # Parse key=value pairs
        parts = {}
        for segment in line.split():
            if "=" in segment:
                key, value = segment.split("=", 1)
                parts[key] = value

        if "PartitionName" not in parts:
            continue

        partition_name = parts["PartitionName"]
        partitions[partition_name] = {
            "MaxNodes": parts.get("MaxNodes", "N/A"),
            "MaxCPUsPerNode": parts.get("MaxCPUsPerNode", "N/A"),
            "MaxTime": parts.get("MaxTime", "N/A"),
            "DefMemPerCPU": parts.get("DefMemPerCPU", "N/A"),
            "State": parts.get("State", "N/A"),
        }

    return partitions


def gather_hpc_partition_info() -> str:
    """
    Gather HPC partition information for debugging resource allocation issues.

    Runs SLURM commands to collect partition details and node configurations.
    This helps diagnose why jobs might fail due to resource constraints.

    Returns
    -------
    str
        Formatted markdown section with partition information, or empty string if not on HPC
    """
    # Check if we're on an HPC cluster with SLURM
    if (
        not os.environ.get("SLURM_CLUSTER_NAME")
        and subprocess.run(["which", "scontrol"], capture_output=True).returncode != 0
    ):
        return ""

    md_lines = ["## HPC Partition Information", ""]
    md_lines.append(f"**Collected**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    md_lines.append("> Use this information to understand resource limits that may affect job allocation.")
    md_lines.append("")

    # Partition overview
    md_lines.append("### Partition Overview")
    md_lines.append("```")
    try:
        result = subprocess.run(
            "sinfo -O partitionname,nodes,cpus,memory,time,gres -a",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            md_lines.append(result.stdout.strip())
        else:
            md_lines.append(f"Command failed (exit code: {result.returncode})")
    except Exception as e:
        md_lines.append(f"Error: {str(e)}")
    md_lines.append("```")
    md_lines.append("")

    # Partition limits (parsed into table)
    md_lines.append("### Partition Resource Limits")
    md_lines.append("")
    md_lines.append("| Partition | Max Nodes | Max CPUs/Node | Max Time | Mem/CPU (MB) | State |")
    md_lines.append("|-----------|-----------|---------------|----------|--------------|-------|")

    try:
        result = subprocess.run(
            "scontrol show partition -o",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            partitions = parse_partition_limits(result.stdout)

            # Show commonly used partitions first
            priority_partitions = ["standard", "parallel", "gpu", "gpu-a6000"]
            shown_partitions = set()

            for partition in priority_partitions:
                if partition in partitions:
                    p = partitions[partition]
                    md_lines.append(
                        f"| {partition} | {p['MaxNodes']} | {p['MaxCPUsPerNode']} | "
                        f"{p['MaxTime']} | {p['DefMemPerCPU']} | {p['State']} |"
                    )
                    shown_partitions.add(partition)

            # Show remaining partitions
            for partition, p in sorted(partitions.items()):
                if partition not in shown_partitions:
                    md_lines.append(
                        f"| {partition} | {p['MaxNodes']} | {p['MaxCPUsPerNode']} | "
                        f"{p['MaxTime']} | {p['DefMemPerCPU']} | {p['State']} |"
                    )
        else:
            md_lines.append("| (command failed) | | | | | |")
    except Exception as e:
        md_lines.append(f"| Error: {str(e)} | | | | | |")

    md_lines.append("")
    md_lines.append(
        "**Note**: `MaxNodes=1` on `standard` partition means multi-node jobs must use `parallel` partition."
    )
    md_lines.append("")

    # GPU partitions
    md_lines.append("### GPU Partitions")
    md_lines.append("```")
    try:
        result = subprocess.run(
            "sinfo -o '%P %G' | grep -i gpu",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            md_lines.append(result.stdout.strip())
        else:
            md_lines.append("No GPU partitions found")
    except Exception as e:
        md_lines.append(f"Error: {str(e)}")
    md_lines.append("```")
    md_lines.append("")
    md_lines.append(
        "**Note**: GPU allocation may be subject to QOS limits (e.g., max GPUs per user). "
        "Check `sacctmgr show assoc where account=<account> format=qos,GrpTRES,MaxTRES -p` for limits."
    )
    md_lines.append("")

    return "\n".join(md_lines)


def write_workflow_summary_md(analysis) -> Path:
    """
    Write workflow status summary to markdown file.

    Generates a workflow_summary.md file with:
    - Workflow status from get_workflow_status()
    - Completion statistics
    - Phase details
    - HPC partition information (if on cluster)

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis object containing workflow status

    Returns
    -------
    Path
        Path to the saved markdown file
    """
    summary_path = analysis.analysis_paths.analysis_dir / "workflow_summary.md"

    # Get workflow status
    status = analysis.get_workflow_status()

    # Determine if workflow is complete (all phases done)
    workflow_complete = all(
        [
            status.setup.complete,
            status.preparation.complete,
            status.simulation.complete,
            status.processing.complete,
            status.consolidation.complete,
        ]
    )

    # Build markdown content
    md_lines = [
        "# Workflow Summary",
        "",
        f"**⏰ Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**📋 Analysis ID**: `{analysis.cfg_analysis.analysis_id}`",
        f"**📁 Analysis Directory**: `{analysis.analysis_paths.analysis_dir}`",
        "",
        "> **Note**: This summary reflects the state at generation time above. "
        "Check timestamp to ensure it matches your current debugging session.",
        "",
        "---",
        "",
        "## Overall Status",
        "",
        f"**Workflow Complete**: {workflow_complete}",
        f"**Current Phase**: {status.current_phase}",
        f"**Recommended Execution Mode**: `{status.recommended_mode}`",
        "",
        "## Progress Summary",
        "",
        f"- **Total Simulations**: {status.total_simulations}",
        f"- **Completed**: {status.simulations_completed}",
        f"- **Failed**: {status.simulations_failed}",
        f"- **Pending**: {status.simulations_pending}",
        "",
        "---",
        "",
        "## Phase Details",
        "",
    ]

    # Add phase status for each phase
    phases = [
        ("Setup", status.setup),
        ("Preparation", status.preparation),
        ("Simulation", status.simulation),
        ("Processing", status.processing),
        ("Consolidation", status.consolidation),
    ]

    for phase_name, phase in phases:
        md_lines.extend(
            [
                f"### {phase_name}",
                "",
                f"- **Complete**: {phase.complete}",
                f"- **Progress**: {phase.progress:.1%}",
            ]
        )

        if phase.failed_items:
            md_lines.append(f"- **Failed Items**: {len(phase.failed_items)}")
            md_lines.append("  ```")
            for item in phase.failed_items[:10]:  # Show first 10
                md_lines.append(f"  {item}")
            if len(phase.failed_items) > 10:
                md_lines.append(f"  ... and {len(phase.failed_items) - 10} more")
            md_lines.append("  ```")

        if phase.details:
            md_lines.append("- **Details**:")
            for key, value in phase.details.items():
                md_lines.append(f"  - {key}: {value}")

        md_lines.append("")

    # Add HPC partition info if on cluster
    partition_info = gather_hpc_partition_info()
    if partition_info:
        md_lines.append("---")
        md_lines.append("")
        md_lines.append(partition_info)

    # Write to file
    summary_path.write_text("\n".join(md_lines))

    print(f"Workflow summary exported to: {summary_path}", flush=True)
    return summary_path


def export_scenario_status_to_csv(analysis, output_path: Path | None = None) -> Path:
    """
    Export the scenario status DataFrame to a CSV file.

    Detects whether the analysis is a regular or sensitivity analysis and exports
    the appropriate df_status DataFrame to a CSV file. This includes:
    - Configuration parameters (from df_setup/df_sims)
    - Scenario preparation status
    - Simulation completion status
    - Scenario directory paths for debugging

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis object (regular or sensitivity) containing scenario status
    output_path : Path, optional
        Path where to save the CSV file. If None, defaults to analysis_dir/scenario_status.csv

    Returns
    -------
    Path
        Path to the saved CSV file
    """
    # Determine output path
    if output_path is None:
        output_path_final = analysis.analysis_paths.analysis_dir / "scenario_status.csv"
    else:
        output_path_final = Path(output_path)

    # Ensure parent directory exists
    output_path_final.parent.mkdir(parents=True, exist_ok=True)

    df_status = analysis.df_status

    # Write to CSV
    df_status.to_csv(output_path_final, index=False)

    print(f"Scenario status exported to: {output_path_final}", flush=True)
    return output_path_final


def main():
    """Command-line interface for exporting scenario status."""
    parser = argparse.ArgumentParser(description="Export scenario status to CSV after simulations complete")
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
        "--output-path",
        type=Path,
        default=None,
        help="Path to save CSV file (defaults to analysis_dir/scenario_status.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    # Log workflow context for traceability
    log_workflow_context(logger)

    logger.info("Exporting scenario status for analysis")
    logger.info(f"System config: {args.system_config}")
    logger.info(f"Analysis config: {args.analysis_config}")
    logger.info(
        f"Output path: {args.output_path if args.output_path else 'default (analysis_dir/scenario_status.csv)'}"
    )

    try:
        # Load system configuration
        logger.info("Loading system configuration...")
        if args.verbose:
            print(f"Loading system config from: {args.system_config}", flush=True)

        system = TRITONSWMM_system(
            system_config_yaml=args.system_config,
        )

        # Load analysis configuration
        logger.info("Loading analysis configuration...")
        if args.verbose:
            print(f"Loading analysis config from: {args.analysis_config}", flush=True)
        analysis = anlysis.TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            hpc_system_config_yaml=args.hpc_system_config,
            skip_log_update=False,
            is_main_orchestrator=False,
        )

        # Export status
        logger.info("Exporting scenario status to CSV...")
        if args.verbose:
            print("Exporting scenario status...", flush=True)

        csv_path = export_scenario_status_to_csv(analysis, args.output_path)
        logger.info(f"Scenario status exported to: {csv_path}")

        # F3: re-persist validation_report.json now that scenario_status.csv exists, so the
        # `scenario_status.csv created` check (which runs at consolidation, BEFORE this rule
        # writes the CSV) no longer bakes a false "missing" failure into the read-model the
        # errors_and_warnings figure reads. The E&W plot rule declares scenario_status.csv as
        # an input, so it runs after this rule -> it reads the corrected (passing) report.
        # Non-fatal: a persist failure must never fail the CSV export.
        try:
            from hhemt.analysis_validation import persist_validation_report

            persist_validation_report(analysis)
            logger.info("Re-persisted validation_report.json (post-CSV) so the CSV-created check passes")
        except Exception:
            # Traceback, not a one-line repr: this rule DECLARES validation_report.json as an
            # output, so Snakemake already fails the job. What the operator loses to a bare
            # `{e}` is the diagnosis -- a master-blind check raising KeyError renders as the
            # single character `1`. Non-fatal is retained so scenario_status.csv and
            # workflow_summary.md still land; the output declaration is the hard stop.
            logger.exception(
                "post-export validation_report.json re-persist failed (rule will fail on the declared output)"
            )

        # Write workflow summary markdown
        # Persist the jobid -> rule index into _status/, which is copytree'd into every
        # render bundle. Harvested from the SLURM executor's own per-job log tree, which is
        # the ONLY retroactive record of jobs whose flag sidecar a later submission
        # overwrote (measured: 511 such allocations on the delivered experiment). Merged
        # into any existing index rather than replacing it, because the plugin prunes its
        # logs by age and a later harvest can legitimately see FEWER jobs than an earlier
        # one -- a straight overwrite would discard history exactly as the sidecar does.
        # Non-fatal, matching the two validation_report re-persists: a reporting nicety must
        # never fail the status export.
        try:
            import json as _json_idx

            from hhemt.status_flags import harvest_slurm_job_index

            _idx_path = analysis.analysis_paths.analysis_dir / "_status" / "_job_index.json"
            _merged: dict[str, str] = {}
            if _idx_path.exists():
                try:
                    _prior = _json_idx.loads(_idx_path.read_text())
                    if isinstance(_prior, dict):
                        _merged.update({str(k): str(v) for k, v in _prior.items()})
                except (OSError, ValueError):
                    pass
            _merged.update(harvest_slurm_job_index(analysis.analysis_paths.analysis_dir))
            if _merged:
                _idx_path.parent.mkdir(parents=True, exist_ok=True)
                _idx_path.write_text(_json_idx.dumps(_merged, indent=2, sort_keys=True))
                logger.info(f"Persisted SLURM job index for {len(_merged)} job(s)")
        except Exception as e:
            logger.warning(f"SLURM job-index harvest failed (non-fatal): {e}")

        # Make job efficiency data a PRODUCT rather than an operator errand. The capture
        # existed already but had no caller anywhere in src/, scripts/ or the Snakefile
        # generators -- it ran only when a human typed `python -m hhemt.slurm_job_recovery`,
        # which is why the store is empty on trees nobody remembered to back-fill. This rule
        # is the right home: it already runs on the cluster where `sacct` resolves, it already
        # re-persists two other read-models here, and it runs after consolidation, so the jobs
        # it captures are finished and their accounting rows are final.
        #
        # `run_method` is passed from here because this is the one place it is authoritatively
        # known; the capture module is stdlib-only and must not read a toolkit config.
        #
        # Non-fatal by the same rule as its neighbours. `backfill` degrades internally to
        # "recovered nothing" on a missing sacct, a timeout, or a non-zero exit, so the local
        # and off-cluster cases cost one no-op call rather than an error path.
        try:
            from hhemt.slurm_job_recovery import backfill

            _rep = backfill(
                analysis.analysis_paths.analysis_dir,
                run_method=str(analysis.cfg_analysis.multi_sim_run_method or ""),
            )
            logger.info(
                "SLURM job store: ids=%s rows=%s job_rows=%s batch_rows=%s missing=%s",
                _rep["ids_in_csv"],
                _rep["rows_recovered"],
                _rep["ids_with_job_row"],
                _rep["ids_with_batch_row"],
                _rep["ids_missing"],
            )
        except Exception as e:
            logger.warning(f"SLURM job-store capture failed (non-fatal): {e}")

        logger.info("Writing workflow summary markdown...")
        if args.verbose:
            print("Writing workflow summary...", flush=True)

        write_workflow_summary_md(analysis)

        # Re-persist validation_report.json now that scenario_status.csv EXISTS.
        # consolidate_workflow.py also persists it, but that call runs at
        # consolidation and this rule takes `_status/e_consolidate_complete.flag`
        # as input — so the consolidation-time report is written STRICTLY BEFORE
        # the CSV and its `scenario_status.csv created` check reports "missing"
        # on every fresh run by construction, while the file demonstrably exists
        # (Rivanna run 17102207: report 17:52, CSV 17:53). That is a false
        # negative in a machine-readable artifact whose whole purpose is to be
        # trusted without re-inspecting the tree.
        #
        # This re-persist is ordering-correct rather than merely later: the plot
        # rules declare scenario_status.csv as input, so they run after THIS rule,
        # which keeps the refreshed report ahead of every renderer that consumes
        # it (Gotcha 53 / Option D: errors_and_warnings reads the persisted
        # read-model, never re-inspecting the tree at render time).
        #
        # Non-fatal, matching the consolidate-side call: a persist failure must
        # never block an otherwise-successful status export.
        try:
            from hhemt.analysis_validation import persist_validation_report

            persist_validation_report(analysis)
            logger.info("Re-persisted validation_report.json (post-CSV, ordering-correct)")
        except Exception:
            # See the sibling handler above: the enclosing rule declares this artifact, so the
            # traceback is the only thing the swallow was still costing.
            logger.exception("validation_report.json re-persist failed (rule will fail on the declared output)")

        logger.info("Status export completed successfully")
        if args.verbose:
            print("Status export completed successfully", flush=True)

    except Exception as e:
        print(f"Error exporting scenario status: {str(e)}", flush=True)
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
