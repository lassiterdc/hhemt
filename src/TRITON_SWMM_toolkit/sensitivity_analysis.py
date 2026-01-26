# %%
import subprocess
import shutil
import TRITON_SWMM_toolkit.utils as ut
from pathlib import Path
from TRITON_SWMM_toolkit.config import analysis_config
import pandas as pd
from typing import Literal
import time

# from TRITON_SWMM_toolkit.paths import SensitivityAnalysisPaths
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.run_simulation import TRITONSWMM_run
from TRITON_SWMM_toolkit.process_simulation import TRITONSWMM_sim_post_processing

from TRITON_SWMM_toolkit.processing_analysis import TRITONSWMM_analysis_post_processing
from TRITON_SWMM_toolkit.constants import Mode
from TRITON_SWMM_toolkit.plot_utils import print_json_file_tree
from TRITON_SWMM_toolkit.log import TRITONSWMM_analysis_log
from TRITON_SWMM_toolkit.plot_analysis import TRITONSWMM_analysis_plotting
import yaml
from pprint import pprint
import json
import TRITON_SWMM_toolkit.analysis as anlysis
import xarray as xr

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class TRITONSWMM_sensitivity_analysis:
    """
    Docstring for TRITONSWMM_sensitivity_analysis
    - Creates subanalyses for each sensitivity analysis table row
    - Consolidates results at the 'master_analysis' level
    """

    def __init__(
        self,
        analysis: "TRITONSWMM_analysis",
    ) -> None:
        self.master_analysis = analysis
        self._system = analysis._system
        self.analysis_paths = analysis.analysis_paths
        self.sub_analyses_prefix = "sa_"
        self.subanalysis_dir = (
            self.master_analysis.analysis_paths.analysis_dir / "subanalyses"
        )
        self.independent_vars = self._attributes_varied_for_analysis()
        self.df_setup = self._retieve_df_setup().loc[:, self.independent_vars]  # type: ignore
        self.sub_analyses = self._create_sub_analyses()

        # validate
        if "run_mode" in self.df_setup.columns:
            run_modes = self.df_setup.loc[:, "run_mode"].unique()
            if ("gpu" in run_modes) and len(run_modes) > 1:
                raise ValueError(
                    "A single sensitivity analysis is not currently configured to handle multi-CPU and multi-GPU "
                    "configurations. The solution currently is to run two different sensitivity analyses."
                )

    def prepare_scenarios_in_each_subanalysis(
        self,
        overwrite_scenarios: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        concurrent: bool = True,
        verbose: bool = False,
    ):
        if self.master_analysis.cfg_analysis.multi_sim_run_method in [
            "local",
            "1_job_many_srun_tasks",
        ]:
            prepare_scenario_launchers = []
            for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
                prepare_scenario_launchers += sub_analysis.retrieve_prepare_scenario_launchers(
                    overwrite_scenario=overwrite_scenarios,
                    rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                    verbose=verbose,
                )
            if concurrent:
                self.master_analysis.run_python_functions_concurrently(
                    prepare_scenario_launchers, verbose=verbose
                )
            else:
                for launcher in prepare_scenario_launchers:
                    launcher()

            if self.all_scenarios_created != True:
                scens_not_created = "\n\t".join(self.scenarios_not_created)
                raise RuntimeError(
                    f"Preparation failed for the following scenarios:\n{scens_not_created}"
                )
            self._update_master_analysis_log()
        elif self.master_analysis.cfg_analysis.multi_sim_run_method in ["batch_job"]:
            raise ValueError(
                "prepare scenarios is not currently executable as batch_job."
            )

    def submit_workflow(
        self,
        mode: Literal["local", "slurm", "auto"] = "auto",
        # setup stuff
        process_system_level_inputs: bool = True,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        # ensemble run stuff
        prepare_scenarios: bool = True,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = True,
        wait_for_completion: bool = False,  # relevant for slurm jobs only
        verbose: bool = True,
    ) -> dict:
        """
        Submit sensitivity analysis workflow using Snakemake.

        This orchestrates multiple sub-analysis workflows and a final master
        consolidation step that combines all sub-analysis outputs.

        Parameters
        ----------
        mode : Literal["local", "slurm", "auto"]
            Execution mode. If "auto", detects based on SLURM environment variables.
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings)
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, prepare scenarios before running
        overwrite_scenario : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after simulations
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        overwrite_if_exist : bool
            If True, overwrite existing processed outputs
        compression_level : int
            Compression level for output files (0-9)
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        verbose : bool
            If True, print progress messages

        Returns
        -------
        dict
            Status dictionary with keys:
            - success: bool
            - mode: str
            - snakefile_path: Path
            - message: str
        """
        # Detect execution mode
        if mode == "auto":
            mode = "slurm" if self.master_analysis.in_slurm else "local"

        if verbose:
            print(
                f"[Snakemake] Submitting sensitivity analysis workflow in {mode} mode",
                flush=True,
            )

        # Generate master Snakefile with flattened hierarchy
        # (no nested Snakemake calls - all rules in one file)
        master_snakefile_content = self._generate_master_snakefile_content(
            which=which,
            overwrite_if_exist=overwrite_if_exist,
            compression_level=compression_level,
            process_system_level_inputs=process_system_level_inputs,
            overwrite_system_inputs=overwrite_system_inputs,
            compile_TRITON_SWMM=compile_TRITON_SWMM,
            recompile_if_already_done_successfully=recompile_if_already_done_successfully,
            prepare_scenarios=prepare_scenarios,
            overwrite_scenario=overwrite_scenario,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            process_timeseries=process_timeseries,
            clear_raw_outputs=clear_raw_outputs,
            pickup_where_leftoff=pickup_where_leftoff,
        )

        master_snakefile_path = (
            self.master_analysis.analysis_paths.analysis_dir / "Snakefile"
        )
        master_snakefile_path.write_text(master_snakefile_content)

        if verbose:
            print(
                f"[Snakemake] Generated master Snakefile: {master_snakefile_path}",
                flush=True,
            )

        # Create required directories BEFORE Snakemake DAG construction
        # (onstart: in Snakefile runs AFTER DAG parsing, too late for file validation)
        analysis_dir = self.master_analysis.analysis_paths.analysis_dir
        (analysis_dir / "_status").mkdir(parents=True, exist_ok=True)
        (analysis_dir / "logs" / "sims").mkdir(parents=True, exist_ok=True)

        if verbose:
            print(
                f"[Snakemake] Created required directories (_status, logs/sims)",
                flush=True,
            )

        # Submit workflow based on mode
        if mode == "local":
            result = self.master_analysis._run_snakemake_local(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
            )
        else:  # slurm
            result = self.master_analysis._run_snakemake_slurm(
                snakefile_path=master_snakefile_path,
                verbose=verbose,
                wait_for_completion=wait_for_completion,
            )

        # Print snakemake log file location if available
        if (
            verbose
            and result.get("snakemake_logfile") is not None
            and not wait_for_completion
        ):
            print(
                f"[Snakemake] Sensitivity analysis workflow submitted in background.",
                flush=True,
            )
            print(
                f"[Snakemake] Monitor progress with: tail -f {result.get('snakemake_logfile')}",
                flush=True,
            )

        self._update_master_analysis_log()
        return result

    def _generate_master_snakefile_content(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        process_timeseries: bool = True,
        clear_raw_outputs: bool = True,
        pickup_where_leftoff: bool = True,
    ) -> str:
        """
        Generate flattened master Snakefile with individual simulation rules.

        This method generates a single Snakefile with all simulation rules
        flattened directly into it (no nested Snakemake calls). Each simulation
        gets its own rule with exact resource requirements from its sub-analysis config.

        This avoids resource contention issues where sub-analyses with different
        CPU/GPU requirements would fail due to incorrect resource allocation.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process
        overwrite_if_exist : bool
            If True, overwrite existing consolidated outputs
        compression_level : int
            Compression level for output files (0-9)
        verbose : bool
            If True, print progress messages
        process_system_level_inputs : bool
            If True, process system-level inputs in master setup rule
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM in master setup rule
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, prepare scenarios before running
        overwrite_scenario : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        process_timeseries : bool
            If True, process timeseries outputs after simulations
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint

        Returns
        -------
        str
            Master Snakefile content
        """
        python_executable = self.master_analysis._python_executable

        # Get absolute path to conda environment file
        from pathlib import Path

        triton_toolkit_root = Path(__file__).parent.parent.parent
        conda_env_path = triton_toolkit_root / "workflow" / "envs" / "triton_swmm.yaml"

        # Start building the Snakefile
        snakefile_content = f'''# Auto-generated flattened master Snakefile for sensitivity analysis
# Each sub-analysis simulation gets its own rule with exact resource requirements

import os

onstart:
    shell("mkdir -p _status logs/sims logs")

onsuccess:
    shell("""
        {self.master_analysis._python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            --system-config {self._system.system_config_yaml} \\
            --analysis-config {self.master_analysis.analysis_config_yaml}
    """)

onerror:
    shell("""
        {self.master_analysis._python_executable} -m TRITON_SWMM_toolkit.export_scenario_status \\
            --system-config {self._system.system_config_yaml} \\
            --analysis-config {self.master_analysis.analysis_config_yaml}
    """)


'''

        # Build the rule all with all dependencies
        consolidation_flags = []
        for sa_id in self.sub_analyses.keys():
            consolidation_flags.append(
                f"_status/consolidate_{self.sub_analyses_prefix}{sa_id}_complete.flag"
            )

        snakefile_content += f'''rule all:
    input: 
        {', '.join([f'"{flag}"' for flag in consolidation_flags])},
        "_status/master_consolidation_complete.flag"

rule setup:
    output: "_status/setup_complete.flag"
    log: "logs/setup.log"
    conda: "{conda_env_path}"
    resources:
        slurm_partition="{self.master_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition}",
        runtime=5,
        mem_mb={self.master_analysis.cfg_analysis.mem_gb_per_cpu * 1000},
        tasks=1,
        cpus_per_task=1,
        nodes=1
    shell:
        """
        mkdir -p logs _status
        {python_executable} -m TRITON_SWMM_toolkit.setup_workflow \\
            --system-config {self.master_analysis._system.system_config_yaml} \\
            --analysis-config {self.master_analysis.analysis_config_yaml} \\
            {"--process-system-inputs " if process_system_level_inputs else ""}\\
            {"--overwrite-system-inputs " if overwrite_system_inputs else ""}\\
            {"--compile-triton-swmm " if compile_TRITON_SWMM else ""}\\
            {"--recompile-if-already-done " if recompile_if_already_done_successfully else ""}\\
            > {{log}} 2>&1
        touch {{output}}
        """

'''

        # Generate simulation rules for each sub-analysis
        subanalysis_flags = []
        for sa_id, sub_analysis in self.sub_analyses.items():
            # Extract resource requirements from sub-analysis config
            n_mpi = sub_analysis.cfg_analysis.n_mpi_procs or 1
            n_omp = sub_analysis.cfg_analysis.n_omp_threads or 1
            n_gpus = sub_analysis.cfg_analysis.n_gpus or 0
            n_nodes = sub_analysis.cfg_analysis.n_nodes or 1
            hpc_time = sub_analysis.cfg_analysis.hpc_time_min_per_sim or 30
            mem_per_cpu = sub_analysis.cfg_analysis.mem_gb_per_cpu or 2

            # For each simulation in this sub-analysis
            sub_analysis_sim_flags = []
            for event_iloc in sub_analysis.df_sims.index:
                rule_name = f"simulation_sa{sa_id}_evt{event_iloc}"
                outflag = f"_status/{rule_name}_complete.flag"
                sub_analysis_sim_flags.append(outflag)
                mem_mb = int(mem_per_cpu * n_mpi * n_omp * 1000)

                # Build resources block, handling optional gpus_per_task
                resources_block = f"""        slurm_partition="{sub_analysis.cfg_analysis.hpc_ensemble_partition}",
        runtime={int(hpc_time * 1.1)},
        mem_mb={mem_mb},
        nodes={n_nodes},
        tasks={n_mpi},
        cpus_per_task={n_omp}"""
                if n_gpus > 0:
                    resources_block += f",\n        gpus_per_task={n_gpus}"

                snakefile_content += f'''rule {rule_name}:
    input: "_status/setup_complete.flag"
    output: "{outflag}"
    log: "logs/sims/{rule_name}.log"
    conda: "{conda_env_path}"
    resources:
{resources_block}
    shell:
        """
        mkdir -p logs _status
        {python_executable} -m TRITON_SWMM_toolkit.run_single_simulation \\
            --event-iloc {event_iloc} \\
            --system-config {self.master_analysis._system.system_config_yaml} \\
            --analysis-config {sub_analysis.analysis_config_yaml} \\
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

'''
            subanalysis_flag = (
                f"_status/consolidate_{self.sub_analyses_prefix}{sa_id}_complete.flag"
            )
            subanalysis_flags.append(subanalysis_flag)
            # consolidate outputs after all sims have been run
            snakefile_content += f'''rule consolidate_{self.sub_analyses_prefix}{sa_id}:
    input: {', '.join([f'"{flag}"' for flag in sub_analysis_sim_flags])}
    output: "{subanalysis_flag}"
    log: "logs/sims/consolidate_{self.sub_analyses_prefix}{sa_id}.log"
    conda: "{conda_env_path}"
    resources:
        slurm_partition="{sub_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition}",
        runtime=30,
        mem_mb={sub_analysis.cfg_analysis.mem_gb_per_cpu * 1000},
        tasks=1,
        cpus_per_task=1,
        nodes=1
    shell:
        """
        mkdir -p logs _status
        {python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            --system-config {self.master_analysis._system.system_config_yaml} \\
            --analysis-config {sub_analysis.analysis_config_yaml} \\
            --which {which} \\
            {"--overwrite-if-exist " if overwrite_if_exist else ""}\\
            --compression-level {compression_level} \\
            > {{log}} 2>&1
        touch {{output}}
        """

'''

        # Generate master consolidation rule
        snakefile_content += f'''rule master_consolidation:
    input: {', '.join([f'"{flag}"' for flag in subanalysis_flags])}
    output: "_status/master_consolidation_complete.flag"
    log: "logs/master_consolidation.log"
    conda: "{conda_env_path}"
    resources:
        slurm_partition="{self.master_analysis.cfg_analysis.hpc_setup_and_analysis_processing_partition}",
        runtime=5,
        mem_mb={self.master_analysis.cfg_analysis.mem_gb_per_cpu * 1000},
        tasks=1,
        cpus_per_task=1,
        nodes=1
    shell:
        """
        mkdir -p logs _status
        {python_executable} -m TRITON_SWMM_toolkit.consolidate_workflow \\
            --system-config {self.master_analysis._system.system_config_yaml} \\
            --analysis-config {self.master_analysis.analysis_config_yaml} \\
            --consolidate-sensitivity-analysis-outputs \\
            --which {which} \\
            {"--overwrite-if-exist " if overwrite_if_exist else ""}\\
            --compression-level {compression_level} \\
            > {{log}} 2>&1
        touch {{output}}
        """
'''
        return snakefile_content

    def run_all_sims(
        self,
        pickup_where_leftoff,
        concurrent: bool = False,
        process_outputs_after_sim_completion: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        verbose=False,
    ):
        if concurrent:
            raise RuntimeError(
                "Running sensitivity analyses concurrently requires"
                "more intelligent handling of compute resource availability"
                "tracking. Update run_simulations_concurrently function"
                "in analysis.py to enable this."
            )
            launch_functions = []
            for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
                launch_functions += sub_analysis._create_launchable_sims(
                    pickup_where_leftoff=pickup_where_leftoff,
                    verbose=verbose,
                )
            self.master_analysis.run_simulations_concurrently(
                launch_functions, verbose=verbose
            )
        else:
            for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
                sub_analysis.run_sims_in_sequence(
                    pickup_where_leftoff=pickup_where_leftoff,
                    process_outputs_after_sim_completion=process_outputs_after_sim_completion,
                    which=which,
                    clear_raw_outputs=clear_raw_outputs,
                    overwrite_if_exist=overwrite_if_exist,
                    compression_level=compression_level,
                    verbose=verbose,
                )
        self._update_master_analysis_log()
        return

    def process_simulation_timeseries_concurrently(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        scenario_timeseries_processing_launchers = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            launchers = sub_analysis.retreive_scenario_timeseries_processing_launchers(
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
            scenario_timeseries_processing_launchers += launchers
        self.master_analysis.run_python_functions_concurrently(
            scenario_timeseries_processing_launchers
        )
        return

    def _combine_TRITON_outputs_per_subanalysis(self):
        assert self.TRITON_subanalyses_outputs_consolidated

        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            ds = sub_analysis.TRITON_summary
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)
            lst_ds.append(ds)

        ds_triton_outputs = xr.combine_by_coords(
            lst_ds, combine_attrs="drop", join="outer"
        )
        return ds_triton_outputs

    def _combine_SWMM_node_outputs_per_subanalysis(self):
        assert self.SWMM_subanalyses_outputs_consolidated

        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            ds = sub_analysis.SWMM_node_summary
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)

            lst_ds.append(ds)

        ds_node_outputs = xr.combine_by_coords(
            lst_ds, combine_attrs="drop", join="outer"
        )
        return ds_node_outputs

    def _combine_SWMM_link_outputs_outputs_per_subanalysis(self):
        assert self.SWMM_subanalyses_outputs_consolidated

        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            ds = sub_analysis.SWMM_link_summary
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)

            lst_ds.append(ds)

        ds_link_outputs = xr.combine_by_coords(
            lst_ds, combine_attrs="drop", join="outer"
        )
        return ds_link_outputs

    def _combine_TRITONSWMM_performance_per_subanalysis(self):
        """
        Combine TRITONSWMM performance summaries from all sub-analyses.

        Returns
        -------
        xr.Dataset
            Combined performance dataset with sensitivity analysis dimensions
        """
        lst_ds = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            config = self.df_setup.iloc[sub_analysis_iloc,]
            # Access the performance summary from the sub-analysis
            ds = sub_analysis.process.TRITONSWMM_performance_summary

            # Add sensitivity analysis dimensions
            for new_dim, dim_value in config.items():
                ds = ds.assign_coords(coords={new_dim: dim_value})
                ds = ds.expand_dims(new_dim)

            lst_ds.append(ds)

        ds_performance = xr.combine_by_coords(
            lst_ds, combine_attrs="drop", join="outer"
        )
        return ds_performance

    def _consolidate_outputs_in_each_subanalysis(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            sub_analysis.consolidate_analysis_outptus(
                which=which,
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        self._update_master_analysis_log()
        return

    @property
    def TRITON_subanalyses_outputs_consolidated(self):
        success = True
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            success = success and sub_analysis.TRITON_analysis_summary_created
        return success

    @property
    def SWMM_subanalyses_outputs_consolidated(self):
        node_success = True
        link_success = True
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            node_success = (
                node_success and sub_analysis.SWMM_node_analysis_summary_created
            )
            link_success = (
                link_success and sub_analysis.SWMM_link_analysis_summary_created
            )
        return node_success and link_success

    def consolidate_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        self.create_subanalysis_summaries(
            which=which,
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
        self.consolidate_subanalysis_outputs(
            which=which,
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def consolidate_subanalysis_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        if which in ["TRITON", "both"]:
            ds_combined_outputs = self._combine_TRITON_outputs_per_subanalysis()
            self.master_analysis.process._consolidate_outputs(
                ds_combined_outputs,
                mode="TRITON",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        if which in ["SWMM", "both"]:
            ds_combined_outputs = self._combine_SWMM_node_outputs_per_subanalysis()
            self.master_analysis.process._consolidate_outputs(
                ds_combined_outputs,
                mode="SWMM_node",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
            ds_combined_outputs = (
                self._combine_SWMM_link_outputs_outputs_per_subanalysis()
            )
            self.master_analysis.process._consolidate_outputs(
                ds_combined_outputs,
                mode="SWMM_link",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )

        # consolidate performance summaries (independent of 'which' parameter)
        start_time = time.time()
        ds_performance = self._combine_TRITONSWMM_performance_per_subanalysis()
        proc_log = (
            self.master_analysis.log.TRITONSWMM_performance_analysis_summary_created
        )
        fname_out = (
            self.master_analysis.analysis_paths.output_tritonswmm_performance_summary
        )
        self.master_analysis.process._write_output(
            ds=ds_performance,
            fname_out=fname_out,
            compression_level=compression_level,
            chunks="auto",
            verbose=verbose,
        )

        proc_log.set(True)
        elapsed_s = time.time() - start_time
        self.master_analysis.log.add_sim_processing_entry(
            fname_out, ut.get_file_size_MiB(fname_out), elapsed_s, True
        )

        return

    def create_subanalysis_summaries(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        if which in ["TRITON", "both"]:
            self._consolidate_outputs_in_each_subanalysis(
                which="TRITON",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        if which in ["SWMM", "both"]:
            self._consolidate_outputs_in_each_subanalysis(
                which="SWMM",
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        return

    @property
    def SWMM_node_summary(self):
        return self.master_analysis.SWMM_node_summary

    @property
    def SWMM_link_summary(self):
        return self.master_analysis.SWMM_link_summary

    @property
    def TRITON_summary(self):
        return self.master_analysis.TRITON_summary

    @property
    def TRITONSWMM_runtimes(self):
        return self.master_analysis.TRITONSWMM_runtimes

    def _attributes_varied_for_analysis(self):
        df_setup = self._retieve_df_setup()
        keys_targeted_for_sensitivity = []
        for key, val in self.master_analysis.cfg_analysis.model_dump().items():
            # print(key)
            if key in df_setup.columns:
                if len(df_setup[key].unique()) > 1:
                    keys_targeted_for_sensitivity.append(key)
        return keys_targeted_for_sensitivity

    def _retieve_df_setup(self) -> pd.DataFrame:
        snstivity_definition = self.master_analysis.cfg_analysis.sensitivity_analysis
        f_extension = snstivity_definition.name.lower().split(".")[-1]  # type: ignore
        if f_extension == "csv":
            df_setup = pd.read_csv(snstivity_definition)  # type: ignore
        elif f_extension == "xlsx":
            df_setup = pd.read_excel(snstivity_definition)
        else:
            raise ValueError(
                "File extension not recognized for file defining sensitivity analysis."
            )
        return df_setup

    def _create_sub_analyses(self):
        # create sub analyses
        dic_sensitivity_analyses = dict()
        for idx, row in self.df_setup.iterrows():
            cfg_snstvty_analysis = self.master_analysis.cfg_analysis.model_copy()

            for key, val in row.items():
                setattr(cfg_snstvty_analysis, key, val)  # type: ignore
            sa_id = f"{self.sub_analyses_prefix}{idx}"
            cfg_snstvty_analysis.analysis_id = sa_id  # type: ignore
            sub_analysis_directory = self.subanalysis_dir / str(
                cfg_snstvty_analysis.analysis_id
            )
            sub_analysis_directory.mkdir(parents=True, exist_ok=True)
            cfg_snstvty_analysis.toggle_sensitivity_analysis = False

            cfg_anlysys_yaml = sub_analysis_directory / f"{sa_id}.yaml"

            cfg_snstvty_analysis.analysis_dir = sub_analysis_directory

            cfg_anlysys_yaml.write_text(
                yaml.safe_dump(
                    cfg_snstvty_analysis.model_dump(mode="json"),
                    sort_keys=False,  # .dict() for pydantic v1
                )
            )
            anlsys = anlysis.TRITONSWMM_analysis(
                analysis_config_yaml=cfg_anlysys_yaml,
                system=self._system,
            )
            dic_sensitivity_analyses[idx] = anlsys
        return dic_sensitivity_analyses

    def compile_TRITON_SWMM_for_sensitivity_analysis(
        self,
        verbose: bool = False,
        recompile_if_already_done_successfully: bool = False,
    ):
        self._system.compile_TRITON_SWMM(
            recompile_if_already_done_successfully=recompile_if_already_done_successfully,
            verbose=verbose,
        )
        self._update_master_analysis_log()
        return

    @property
    def scenarios_not_created(self):
        scenarios_not_created = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            for event_iloc in sub_analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
                if scen.log.scenario_creation_complete.get() != True:
                    scenarios_not_created.append(str(scen.log.logfile.parent))
        return scenarios_not_created

    @property
    def scenarios_not_run(self):
        scens_not_run = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            for event_iloc in sub_analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
                if scen.sim_run_completed != True:
                    scens_not_run.append(str(scen.log.logfile.parent))
        return scens_not_run

    @property
    def df_status(self):
        scenarios_setup = []
        scen_runs_completed = []
        scenario_dirs = []
        sub_analysis_ilocs = []
        event_ilocs = []
        df_setup_rows = []
        for sub_analysis_iloc, sub_analysis in self.sub_analyses.items():
            for event_iloc in sub_analysis.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, sub_analysis)
                sub_analysis_ilocs.append(sub_analysis_iloc)
                event_ilocs.append(event_iloc)
                scenarios_setup.append(
                    scen.log.scenario_creation_complete.get() == True
                )
                scen_runs_completed.append(scen.sim_run_completed)
                scenario_dirs.append(str(scen.log.logfile.parent))
                subanalysis_definition_row = self.df_setup.iloc[sub_analysis_iloc, :]
                df_setup_rows.append(subanalysis_definition_row)
        df_status = self.df_setup.iloc[sub_analysis_ilocs, :].reset_index(drop=True)
        # df_status = pd.concat(df_setup_rows)
        df_status["sub_analysis_ilocs"] = sub_analysis_ilocs
        df_status["event_ilocs"] = event_ilocs
        df_status["scenarios_setup"] = scenarios_setup
        df_status["scen_runs_completed"] = scen_runs_completed
        df_status["scenario_directory"] = scenario_dirs
        return df_status

    @property
    def all_scenarios_created(self):
        all_scenarios_created = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_scenarios_created = (
                all_scenarios_created and sub_analysis.log.all_scenarios_created.get()
            )
        return all_scenarios_created == True

    @property
    def all_sims_run(self):
        all_sims_run = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_sims_run = all_sims_run and sub_analysis.log.all_sims_run.get()
        return all_sims_run == True

    @property
    def all_TRITON_timeseries_processed(self):
        all_TRITON_timeseries_processed = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_TRITON_timeseries_processed = (
                all_TRITON_timeseries_processed
                and sub_analysis.log.all_TRITON_timeseries_processed.get()
            )
        return all_TRITON_timeseries_processed == True

    @property
    def all_SWMM_timeseries_processed(self):
        all_SWMM_timeseries_processed = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_SWMM_timeseries_processed = (
                all_SWMM_timeseries_processed
                and sub_analysis.log.all_SWMM_timeseries_processed.get()
            )
        return all_SWMM_timeseries_processed == True

    @property
    def all_TRITONSWMM_performance_timeseries_processed(self):
        all_TRITONSWMM_performance_timeseries_processed = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_TRITONSWMM_performance_timeseries_processed = (
                all_TRITONSWMM_performance_timeseries_processed
                and sub_analysis.log.all_TRITONSWMM_performance_timeseries_processed.get()
            )
        return all_TRITONSWMM_performance_timeseries_processed == True

    @property
    def TRITON_time_series_not_processed(self):
        lst_scens = []
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            lst_scens += sub_analysis.TRITON_time_series_not_processed()
        return lst_scens

    @property
    def SWMM_time_series_not_processed(self):
        lst_scens = []
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            lst_scens += sub_analysis.SWMM_time_series_not_processed()
        return lst_scens

    @property
    def all_raw_TRITON_outputs_cleared(self):
        all_raw_TRITON_outputs_cleared = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_raw_TRITON_outputs_cleared = (
                all_raw_TRITON_outputs_cleared
                and sub_analysis.log.all_raw_TRITON_outputs_cleared.get()
            )
        return all_raw_TRITON_outputs_cleared == True

    @property
    def all_raw_SWMM_outputs_cleared(self):
        all_raw_SWMM_outputs_cleared = True
        for key, sub_analysis in self.sub_analyses.items():
            sub_analysis._update_log()
            all_raw_SWMM_outputs_cleared = (
                all_raw_SWMM_outputs_cleared
                and sub_analysis.log.all_raw_SWMM_outputs_cleared.get()
            )
        return all_raw_SWMM_outputs_cleared == True

    def _update_master_analysis_log(self):
        self.master_analysis._update_log()
        return
