"""
Module for exporting scenario status DataFrame to CSV after all simulations.

This module provides functionality to export the df_status() DataFrame to a CSV file
after all simulations complete, regardless of their success or failure. This enables
debugging by identifying which scenarios failed and why.
"""

import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
from TRITON_SWMM_toolkit.config import load_analysis_config
from TRITON_SWMM_toolkit.system import TRITONSWMM_system
import TRITON_SWMM_toolkit.analysis as anlysis
import yaml


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

    # Get the appropriate status DataFrame based on analysis type
    if analysis.cfg_analysis.toggle_sensitivity_analysis:
        # Sensitivity analysis
        df_status = analysis.sensitivity.df_status
    else:
        # Regular analysis
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

    try:
        # Load system configuration
        if args.verbose:
            print(f"Loading system config from: {args.system_config}", flush=True)

        system = TRITONSWMM_system(
            system_config_yaml=args.system_config,
        )

        # Load analysis configuration
        if args.verbose:
            print(f"Loading analysis config from: {args.analysis_config}", flush=True)
        analysis = anlysis.TRITONSWMM_analysis(
            analysis_config_yaml=args.analysis_config,
            system=system,
            skip_log_update=True,
        )

        # Export status
        if args.verbose:
            print("Exporting scenario status...", flush=True)

        export_scenario_status_to_csv(analysis, args.output_path)

        if args.verbose:
            print("Status export completed successfully", flush=True)

    except Exception as e:
        print(f"Error exporting scenario status: {str(e)}", flush=True)
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
