"""
Module for exporting scenario status DataFrame to CSV after all simulations.

This module provides functionality to export the df_status() DataFrame to a CSV file
after all simulations complete, regardless of their success or failure. This enables
debugging by identifying which scenarios failed and why.

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
from typing import Optional
from TRITON_SWMM_toolkit.system import TRITONSWMM_system
from TRITON_SWMM_toolkit.log_utils import log_workflow_context
import TRITON_SWMM_toolkit.analysis as anlysis

# Configure logging to stderr (will be redirected to logfile by Snakefile)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def gather_hpc_partition_info() -> str:
    """
    Gather HPC partition information for debugging resource allocation issues.

    Runs SLURM commands to collect partition details, QOS limits, and node configurations.
    This helps diagnose why jobs might fail due to resource constraints.

    Returns
    -------
    str
        Formatted markdown section with partition information, or empty string if not on HPC
    """
    # Check if we're on an HPC cluster with SLURM
    if not os.environ.get("SLURM_CLUSTER_NAME") and subprocess.run(
        ["which", "scontrol"], capture_output=True
    ).returncode != 0:
        return ""

    md_lines = ["## HPC Partition Information", ""]
    md_lines.append(f"**Collected**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")

    commands = [
        ("Partition Overview", "sinfo -O partitionname,nodes,cpus,memory,time,gres -a"),
        ("Partition Details", "scontrol show partition -o"),
        ("QOS Limits", "sacctmgr show qos format=name,maxwall,maxsubmit,maxnodes,maxcpus,maxgres -p"),
        ("User QOS", f"sacctmgr show user $USER format=User,Account,DefaultQOS,QOS -p"),
        ("GPU Partitions", "sinfo -o '%P %G' | grep -i gpu"),
    ]

    for section_name, command in commands:
        md_lines.append(f"### {section_name}")
        md_lines.append("```")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                md_lines.append(result.stdout.strip())
            else:
                md_lines.append(f"Command failed or returned no output (exit code: {result.returncode})")
        except Exception as e:
            md_lines.append(f"Error running command: {str(e)}")
        md_lines.append("```")
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

    # Build markdown content
    md_lines = [
        "# Workflow Summary",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Analysis ID**: `{analysis.cfg_analysis.analysis_id}`",
        f"**Analysis Directory**: `{analysis.analysis_paths.analysis_dir}`",
        "",
        "---",
        "",
        "## Overall Status",
        "",
        f"**Workflow Complete**: {status.workflow_complete}",
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
        md_lines.extend([
            f"### {phase_name}",
            "",
            f"- **Complete**: {phase.complete}",
            f"- **Progress**: {phase.progress:.1%}",
        ])

        if phase.failed_items:
            md_lines.append(f"- **Failed Items**: {len(phase.failed_items)}")
            md_lines.append("  ```")
            for item in phase.failed_items[:10]:  # Show first 10
                md_lines.append(f"  {item}")
            if len(phase.failed_items) > 10:
                md_lines.append(f"  ... and {len(phase.failed_items) - 10} more")
            md_lines.append("  ```")

        if phase.message:
            md_lines.append(f"- **Message**: {phase.message}")

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


def export_scenario_status_to_csv(analysis, output_path: Optional[Path] = None) -> Path:
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
    parser = argparse.ArgumentParser(
        description="Export scenario status to CSV after simulations complete"
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

    logger.info(f"Exporting scenario status for analysis")
    logger.info(f"System config: {args.system_config}")
    logger.info(f"Analysis config: {args.analysis_config}")
    logger.info(f"Output path: {args.output_path if args.output_path else 'default (analysis_dir/scenario_status.csv)'}")

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
            skip_log_update=True,
        )

        # Export status
        logger.info("Exporting scenario status to CSV...")
        if args.verbose:
            print("Exporting scenario status...", flush=True)

        csv_path = export_scenario_status_to_csv(analysis, args.output_path)
        logger.info(f"Scenario status exported to: {csv_path}")

        # Write workflow summary markdown
        logger.info("Writing workflow summary markdown...")
        if args.verbose:
            print("Writing workflow summary...", flush=True)

        write_workflow_summary_md(analysis)

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
