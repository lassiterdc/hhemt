"""
Snakemake Workflow Generation Module

This module handles the generation and execution of Snakemake workflows for
TRITON-SWMM simulations. It provides a clean interface for creating workflow
files and submitting them to either local or SLURM execution environments.

Key Components:
- SnakemakeWorkflowBuilder: Main class for workflow generation and submission
"""

import subprocess
import yaml
import psutil
from pathlib import Path
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class SnakemakeWorkflowBuilder:
    """
    Builder class for generating and executing Snakemake workflows.

    This class encapsulates all Snakemake-related functionality including:
    - Snakefile content generation
    - Dynamic configuration generation
    - Local execution
    - SLURM/HPC execution

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The parent analysis object containing configuration and paths
    """

    def __init__(self, analysis: "TRITONSWMM_analysis"):
        self.analysis = analysis
        self.cfg_analysis = analysis.cfg_analysis
        self.system = analysis._system
        self.analysis_paths = analysis.analysis_paths
        self.python_executable = analysis._python_executable

    def generate_snakefile_content(
        self,
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: str = "TRITON",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = False,
    ) -> str:
        """
        Generate Snakefile content for the three-phase workflow using wildcards.

        Parameters
        ----------
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings) in Phase 1
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM in Phase 1
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, each simulation will prepare its scenario before running
        overwrite_scenario : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after each simulation
        which : str
            Which outputs to process: "TRITON", "SWMM", or "both"
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        overwrite_if_exist : bool
            If True, overwrite existing processed outputs
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint

        Returns
        -------
        str
            Complete Snakefile content as a string
        """
        n_sims = len(self.analysis.df_sims)
        hpc_time_min = self.cfg_analysis.hpc_time_min_per_sim or 30

        mpi_ranks = self.cfg_analysis.n_mpi_procs or 1
        omp_threads = self.cfg_analysis.n_omp_threads or 1
        cpus_per_sim = mpi_ranks * omp_threads

        # Conservative estimate: 2GB per CPU (can be made configurable later)
        mem_mb_per_sim = self.cfg_analysis.mem_gb_per_cpu * cpus_per_sim * 1000
        n_nodes = self.cfg_analysis.n_nodes or 1

        # Get absolute path to conda environment file
        triton_toolkit_root = Path(__file__).parent.parent.parent
        conda_env_path = triton_toolkit_root / "workflow" / "envs" / "triton_swmm.yaml"
        skip_setup = not (process_system_level_inputs or compile_TRITON_SWMM)

        # Make log dirs
        analysis_dir = self.analysis_paths.analysis_dir
        (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
        (analysis_dir / "logs" / "sims").mkdir(parents=True, exist_ok=True)

        if skip_setup:
            setup_shell = f'''"""
        touch {{output}}
        """
        '''
        else:
            setup_shell = f'''"""
        {self.python_executable} -m TRITON_SWMM_toolkit.setup_workflow \\
            --system-config {self.system.system_config_yaml} \\
            --analysis-config {self.analysis.analysis_config_yaml} \\
            {"--process-system-inputs " if process_system_level_inputs else ""}\\
            {"--overwrite-system-inputs " if overwrite_system_inputs else ""}\\
            {"--compile-triton-swmm " if compile_TRITON_SWMM else ""}\\
            {"--recompile-if-already-done " if recompile_if_already_done_successfully else ""}\\
            > {{log}} 2>&1
        touch {{output}}
        """'''

        snakefile_content = f'''# Auto-generated by TRITONSWMM_analysis

import os
import glob
import subprocess

# Read simulation IDs from config
SIM_IDS = {list(range(n_sims))}

rule all:
    input: "_status/output_consolidation_complete.flag"

onsuccess:
    shell("""
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            --system-config {self.system.system_config_yaml} \\
            --analysis-config {self.analysis.analysis_config_yaml}
    """)

onerror:
    shell("""
        {self.python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            --system-config {self.system.system_config_yaml} \\
            --analysis-config {self.analysis.analysis_config_yaml}
    """)

rule setup:
    output: "_status/setup_complete.flag"
    log: "logs/setup.log"
    conda: "{conda_env_path}"
    resources:
        slurm_partition="{self.cfg_analysis.hpc_setup_and_analysis_processing_partition}",
        runtime=5,
        mem_mb={self.cfg_analysis.mem_gb_per_cpu * 1000},
        tasks=1,
        cpus_per_task=1,
        nodes=1
    shell:
        {setup_shell}

rule simulation:
    input: "_status/setup_complete.flag"
    output: "_status/sims/sim_{{event_iloc}}_complete.flag"
    log: "logs/sim_{{event_iloc}}.log"
    conda: "{conda_env_path}"
    threads: {cpus_per_sim}
    resources:
        slurm_partition="{self.cfg_analysis.hpc_ensemble_partition}",
        runtime={int(hpc_time_min)},
        tasks={self.cfg_analysis.n_mpi_procs or 1},
        cpus_per_task={self.cfg_analysis.n_omp_threads or 1},
        mem_mb={mem_mb_per_sim},
        nodes={n_nodes}
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.run_single_simulation \\
            --event-iloc {{wildcards.event_iloc}} \\
            --system-config {self.system.system_config_yaml} \\
            --analysis-config {self.analysis.analysis_config_yaml} \\
            {"--prepare-scenario " if prepare_scenarios else ""}\\
            {"--overwrite-scenario " if overwrite_scenario else ""}\\
            {"--rerun-swmm-hydro " if rerun_swmm_hydro_if_outputs_exist else ""}\\
            {"--process-timeseries " if process_timeseries else ""}\\
            --which {which} \\
            {"--clear-raw-outputs " if clear_raw_outputs else ""}\\
            {"--overwrite-if-exist " if overwrite_if_exist else ""}\\
            --compression-level {compression_level} \\
            {"--pickup-where-leftoff " if pickup_where_leftoff else ""}\\
            > {{log}} 2>&1
        touch {{output}}
        """

rule consolidate:
    input: expand("_status/sims/sim_{{event_iloc}}_complete.flag", event_iloc=SIM_IDS)
    output: "_status/output_consolidation_complete.flag"
    log: "logs/consolidate.log"
    conda: "{conda_env_path}"
    resources:
        slurm_partition="{self.cfg_analysis.hpc_setup_and_analysis_processing_partition}",
        runtime=30,
        mem_mb={self.cfg_analysis.mem_gb_per_cpu * 1000},
        tasks=1,
        cpus_per_task=1,
        nodes=1
    shell:
        """
        {self.python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            --system-config {self.system.system_config_yaml} \\
            --analysis-config {self.analysis.analysis_config_yaml} \\
            --compression-level {compression_level} \\
            {"--overwrite-if-exist " if overwrite_if_exist else ""}\\
            --which {which} \\
            > {{log}} 2>&1
        touch {{output}}
        """
'''
        return snakefile_content

    def generate_snakemake_config(self, mode: Literal["local", "slurm"]) -> dict:
        """
        Generate dynamic snakemake config based on analysis_config and system_config.

        Supports dual-mode execution:
        - Modern mode (default): Uses 'executor: slurm' with job steps
        - Legacy mode: Uses 'cluster' with direct sbatch submission

        Parameters
        ----------
        mode : Literal["local", "slurm"]
            Execution mode (local or slurm)

        Returns
        -------
        dict
            Snakemake configuration dictionary
        """
        # Base config shared by both modes
        config = {
            "use-conda": False,
            "conda-frontend": "mamba",
            "printshellcmds": True,
            "rerun-incomplete": True,
        }

        if mode == "local":
            # Local mode: use cores based on system capabilities
            physical_cores = psutil.cpu_count(logical=False)
            cores = max(1, (physical_cores or 2) - 1)  # Leave one core free
            config.update(
                {
                    "cores": cores,
                    "keep-going": False,
                }
            )
        else:  # slurm
            # SLURM mode: support both modern executor and legacy cluster modes
            slurm_partition = self.cfg_analysis.hpc_ensemble_partition or "standard"
            # Modern executor mode: uses 'executor: slurm' with job steps
            config.update(
                {
                    "executor": "slurm",
                    "jobs": self.cfg_analysis.hpc_max_simultaneous_sims or 100,
                    "latency-wait": 60,
                    "max-jobs-per-second": 5,
                    "max-status-checks-per-second": 10,
                    "default-resources": [
                        f"nodes=1",
                        f"mem_mb=2000",
                        f"runtime=30",
                        f"slurm_partition={slurm_partition}",
                    ],
                    "slurm": {
                        "sbatch": {
                            "partition": "{resources.slurm_partition}",
                            "time": "{resources.runtime}:00",
                            "mem": "{resources.mem_mb}",
                            "nodes": "{resources.nodes}",
                            "ntasks": "{resources.tasks}",
                            "cpus-per-task": "{resources.cpus_per_task}",
                        }
                    },
                }
            )

            # Add account if specified
            if self.cfg_analysis.hpc_account:
                config["slurm"]["sbatch"]["account"] = self.cfg_analysis.hpc_account

        return config

    def write_snakemake_config(
        self, config: dict, mode: Literal["local", "slurm"]
    ) -> Path:
        """
        Write snakemake config to analysis directory.

        Parameters
        ----------
        config : dict
            Snakemake configuration dictionary
        mode : Literal["local", "slurm"]
            Execution mode (local or slurm)

        Returns
        -------
        Path
            Path to the written config directory
        """
        config_dir = self.analysis_paths.analysis_dir / ".snakemake_profile" / mode
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"

        with open(config_path, "w") as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                sort_keys=False,
                width=float("inf"),  # Prevent YAML from breaking long lines
            )

        return config_dir

    def run_snakemake_local(
        self,
        snakefile_path: Path,
        verbose: bool = True,
    ) -> dict:
        """
        Run Snakemake workflow on local machine.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        verbose : bool
            If True, print progress messages

        Returns
        -------
        dict
            Status dictionary
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Running workflow locally with Snakemake",
                    flush=True,
                )

            # Generate and write dynamic config
            config = self.generate_snakemake_config(mode="local")
            config_dir = self.write_snakemake_config(config, mode="local")

            if verbose:
                print(
                    f"[Snakemake] Using dynamic config from: {config_dir}", flush=True
                )

            result = subprocess.run(
                [
                    "snakemake",
                    "--profile",
                    str(config_dir),
                    "--snakefile",
                    str(snakefile_path),
                ],
                cwd=str(self.analysis_paths.analysis_dir),
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                error_msg = f"Snakemake workflow failed:\n{result.stderr}"
                if verbose:
                    print(f"[Snakemake] ERROR: {error_msg}", flush=True)
                return {
                    "success": False,
                    "mode": "local",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": error_msg,
                }

            if verbose:
                print("[Snakemake] Workflow completed successfully", flush=True)

            return {
                "success": True,
                "mode": "local",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": "Workflow completed successfully",
            }

        except Exception as e:
            error_msg = f"Failed to run Snakemake: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "local",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": error_msg,
            }

    def run_snakemake_slurm(
        self,
        snakefile_path: Path,
        verbose: bool = True,
        wait_for_completion: bool = False,
    ) -> dict:
        """
        Run Snakemake workflow on SLURM HPC system.

        Parameters
        ----------
        snakefile_path : Path
            Path to the Snakefile
        verbose : bool
            If True, print progress messages
        wait_for_completion : bool
            If True, block and wait for workflow completion. If False (default),
            return immediately after submission (non-blocking).

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool - Did submission succeed?
            - mode: str - "slurm"
            - snakefile_path: Path - Path to Snakefile
            - job_id: str | None - Always None (job ID not extracted)
            - message: str - Status message
            - process: Popen - Process object
            - wait_for_completion: bool - Whether we waited
            - completed: bool - True only if wait_for_completion=True and job finished
            - completion_status: str | None - "success"/"failed" (only if waited)
            - snakemake_logfile: Path - Path to snakemake output log
        """
        try:
            if verbose:
                print(
                    "[Snakemake] Running workflow on SLURM with Snakemake",
                    flush=True,
                )

            # Generate and write dynamic config
            config = self.generate_snakemake_config(mode="slurm")
            config_dir = self.write_snakemake_config(config, mode="slurm")

            if verbose:
                print(f"[Snakemake] Using config from: {config_dir}", flush=True)

            # Create log directory and file for Snakemake output
            logs_dir = self.analysis_paths.analysis_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            snakemake_logfile = logs_dir / "snakemake_master.log"

            if verbose:
                print(
                    f"[Snakemake] Snakemake output will be logged to: {snakemake_logfile}",
                    flush=True,
                )

            # Submit workflow as detached background process, capturing output
            # Don't pass --executor; let Snakemake read the config (either 'executor' or 'cluster' mode)
            cmd_args = [
                "snakemake",
                "--profile",
                str(config_dir),
                "--snakefile",
                str(snakefile_path),
                "--executor",
                "slurm",
                "--default-resources",
                "--slurm-efficiency-report",
            ]
            if verbose:
                cmd_args.append("--verbose")

            # Open log file for writing Snakemake output
            with open(snakemake_logfile, "w") as log_f:
                proc = subprocess.Popen(
                    cmd_args,
                    cwd=str(self.analysis_paths.analysis_dir),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    start_new_session=True,  # Detach from parent process
                )

            # Return immediately (non-blocking)
            if not wait_for_completion:
                if verbose:
                    print(
                        f"[Snakemake] Workflow submitted to background (PID: {proc.pid})",
                        flush=True,
                    )
                    print(
                        f"[Snakemake] Monitor progress with: tail -f {snakemake_logfile}",
                        flush=True,
                    )
                return {
                    "success": True,
                    "mode": "slurm",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": "Workflow submitted to background",
                    "process": proc,
                    "wait_for_completion": False,
                    "completed": False,
                    "completion_status": None,
                    "snakemake_logfile": snakemake_logfile,
                }

            # Wait for completion (blocking)
            else:
                if verbose:
                    print("[Snakemake] Waiting for workflow completion...", flush=True)
                proc.wait()
                success = proc.returncode == 0
                completion_status = "success" if success else "failed"

                if verbose:
                    print(
                        f"[Snakemake] Workflow completed with status: {completion_status}",
                        flush=True,
                    )
                    print(
                        f"[Snakemake] Full output available in: {snakemake_logfile}",
                        flush=True,
                    )

                return {
                    "success": success,
                    "mode": "slurm",
                    "snakefile_path": snakefile_path,
                    "job_id": None,
                    "message": f"Workflow completed with status: {completion_status}",
                    "process": proc,
                    "wait_for_completion": True,
                    "completed": True,
                    "completion_status": completion_status,
                    "snakemake_logfile": snakemake_logfile,
                }

        except Exception as e:
            error_msg = f"Failed to submit workflow: {str(e)}"
            if verbose:
                print(f"[Snakemake] EXCEPTION: {error_msg}", flush=True)
            return {
                "success": False,
                "mode": "slurm",
                "snakefile_path": snakefile_path,
                "job_id": None,
                "message": error_msg,
                "process": None,
                "wait_for_completion": wait_for_completion,
                "completed": False,
                "completion_status": None,
                "snakemake_logfile": None,
            }

    def submit_workflow(
        self,
        mode: Literal["local", "slurm", "auto"] = "auto",
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = False,
        wait_for_completion: bool = False,
        verbose: bool = True,
    ) -> dict:
        """
        Submit workflow using Snakemake.

        Automatically detects execution context (local vs. HPC) and submits accordingly.

        Parameters
        ----------
        mode : Literal["local", "slurm", "auto"]
            Execution mode. If "auto", detects based on SLURM environment variables.
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings) in Phase 1
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM in Phase 1
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, each simulation will prepare its scenario before running
        overwrite_scenario : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after each simulation
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process (only used if process_timeseries=True)
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        overwrite_if_exist : bool
            If True, overwrite existing processed outputs
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        wait_for_completion : bool
            If True, wait for workflow completion (relevant for slurm jobs only)
        verbose : bool
            If True, print progress messages

        Returns
        -------
        dict
            Status dictionary with keys defined by run_snakemake_local or run_snakemake_slurm
        """
        # Detect execution mode
        if mode == "auto":
            mode = "slurm" if self.analysis.in_slurm else "local"

        if verbose:
            print(f"[Snakemake] Submitting workflow in {mode} mode", flush=True)

        # Generate Snakefile content
        snakefile_content = self.generate_snakefile_content(
            process_system_level_inputs=process_system_level_inputs,
            overwrite_system_inputs=overwrite_system_inputs,
            compile_TRITON_SWMM=compile_TRITON_SWMM,
            recompile_if_already_done_successfully=recompile_if_already_done_successfully,
            prepare_scenarios=prepare_scenarios,
            overwrite_scenario=overwrite_scenario,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            process_timeseries=process_timeseries,
            which=which,
            clear_raw_outputs=clear_raw_outputs,
            overwrite_if_exist=overwrite_if_exist,
            compression_level=compression_level,
            pickup_where_leftoff=pickup_where_leftoff,
        )

        # Write Snakefile to disk
        snakefile_path = self.analysis_paths.analysis_dir / "Snakefile"
        snakefile_path.write_text(snakefile_content)

        if verbose:
            print(f"[Snakemake] Snakefile generated: {snakefile_path}", flush=True)

        # Submit workflow based on mode
        if mode == "local":
            result = self.run_snakemake_local(
                snakefile_path=snakefile_path,
                verbose=verbose,
            )
        else:  # slurm
            result = self.run_snakemake_slurm(
                snakefile_path=snakefile_path,
                wait_for_completion=wait_for_completion,
                verbose=verbose,
            )

        self.analysis._refresh_log()
        return result
