# %%

from pathlib import Path
from TRITON_SWMM_toolkit.config import load_analysis_config
import pandas as pd
from typing import Literal, Callable, List, Optional, TYPE_CHECKING
from TRITON_SWMM_toolkit.paths import AnalysisPaths
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.process_simulation import TRITONSWMM_sim_post_processing
from TRITON_SWMM_toolkit.processing_analysis import TRITONSWMM_analysis_post_processing
from TRITON_SWMM_toolkit.plot_utils import print_json_file_tree
from TRITON_SWMM_toolkit.log import TRITONSWMM_analysis_log
from TRITON_SWMM_toolkit.plot_analysis import TRITONSWMM_analysis_plotting
from TRITON_SWMM_toolkit.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
from TRITON_SWMM_toolkit.resource_management import ResourceManager
from TRITON_SWMM_toolkit.execution import (
    SerialExecutor,
    LocalConcurrentExecutor,
    SlurmExecutor,
)
from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from .system import TRITONSWMM_system


class TRITONSWMM_analysis:
    def __init__(
        self,
        analysis_config_yaml: Path,
        system: "TRITONSWMM_system",
        skip_log_update: bool = False,
    ) -> None:
        """
        Initialize a TRITON-SWMM analysis orchestrator.

        This class manages the complete lifecycle of a TRITON-SWMM analysis including
        scenario preparation, simulation execution, output processing, and result
        consolidation. It supports multiple execution strategies (serial, local
        concurrent, SLURM) and workflow management via Snakemake.

        Parameters
        ----------
        analysis_config_yaml : Path
            Path to the analysis configuration YAML file
        system : TRITONSWMM_system
            The TRITON-SWMM system object containing system configuration
        skip_log_update : bool, optional
            If True, skip initial log update (default: False)
        """
        self._system = system
        self.analysis_config_yaml = analysis_config_yaml
        cfg_analysis = load_analysis_config(analysis_config_yaml)
        self.cfg_analysis = cfg_analysis
        if cfg_analysis.analysis_dir:
            analysis_dir = cfg_analysis.analysis_dir
        else:
            analysis_dir = (
                self._system.cfg_system.system_directory / self.cfg_analysis.analysis_id
            )

        self.analysis_paths = AnalysisPaths(
            f_log=analysis_dir / "log.json",
            analysis_dir=analysis_dir,
            simulation_directory=analysis_dir / "sims",
            output_triton_summary=analysis_dir
            / f"TRITON.{self.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_links_summary=analysis_dir
            / f"SWMM_links.{self.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_node_summary=analysis_dir
            / f"SWMM_nodes.{self.cfg_analysis.TRITON_processed_output_type}",
            output_tritonswmm_performance_summary=analysis_dir
            / f"TRITONSWMM_performance.{self.cfg_analysis.TRITON_processed_output_type}",
        )

        # if self.cfg_analysis.toggle_run_ensemble_with_bash_script is True:
        #     self.analysis_paths.bash_script_path = analysis_dir / "run_ensemble.sh"
        self.df_sims = pd.read_csv(self.cfg_analysis.weather_events_to_simulate).loc[
            :, self.cfg_analysis.weather_event_indices
        ]
        self._sim_run_objects = {}
        self._sim_run_processing_objects = {}
        self._simulation_run_statuses = {}
        # self.run_modes = Mode
        # self._system.compilation_successful = False
        self.in_slurm = "SLURM_JOB_ID" in os.environ.copy()
        self._resource_manager = ResourceManager(self.cfg_analysis, self.in_slurm)
        self._execution_strategy = self._select_execution_strategy()
        if self.cfg_analysis.python_path is not None:
            python_executable = str(self.cfg_analysis.python_path)
        else:
            python_executable = "python"
        self._python_executable = python_executable
        self._workflow_builder = SnakemakeWorkflowBuilder(self)
        self.process = TRITONSWMM_analysis_post_processing(self)
        self.plot = TRITONSWMM_analysis_plotting(self)
        if self.cfg_analysis.toggle_sensitivity_analysis is True:
            self.sensitivity = TRITONSWMM_sensitivity_analysis(self)
        if not skip_log_update:
            # self._add_all_scenarios()
            self._refresh_log()
            if self._system.sys_paths.compilation_logfile.exists():
                self._system.compilation_successful
            self._update_log()

    def _refresh_log(self):
        if self.analysis_paths.f_log.exists():
            self.log = TRITONSWMM_analysis_log.from_json(self.analysis_paths.f_log)
        else:
            self.log = TRITONSWMM_analysis_log(logfile=self.analysis_paths.f_log)

    def _select_execution_strategy(self):
        """
        Select the appropriate execution strategy based on configuration.

        Returns
        -------
        ExecutionStrategy
            The appropriate executor (SerialExecutor, LocalConcurrentExecutor, or SlurmExecutor)
        """
        method = self.cfg_analysis.multi_sim_run_method
        if method == "1_job_many_srun_tasks":
            return SlurmExecutor(self)
        elif method == "local":
            return LocalConcurrentExecutor(self)
        else:
            # Default to serial execution for safety
            return SerialExecutor(self)

    def consolidate_TRITON_and_SWMM_simulation_summaries(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
        which: Literal["TRITON", "SWMM", "both"] = "both",
    ):
        """
        Consolidate simulation outputs from all scenarios into analysis-level summaries.

        This method aggregates individual scenario outputs into consolidated files for
        TRITON outputs, SWMM outputs, and performance metrics.

        Parameters
        ----------
        overwrite_if_exist : bool, optional
            If True, overwrite existing consolidated files (default: False)
        verbose : bool, optional
            If True, print progress messages (default: False)
        compression_level : int, optional
            Compression level for output files, 0-9 (default: 5)
        which : Literal["TRITON", "SWMM", "both"], optional
            Which outputs to consolidate (default: "both")
        """
        if which in ["TRITON", "both"]:
            self.consolidate_TRITON_simulation_summaries(
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        if which in ["SWMM", "both"]:
            self.consolidate_SWMM_simulation_summaries(
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )

        # ALWAYS consolidate performance summaries (independent of 'which' parameter)
        self.process.consolidate_TRITONSWMM_performance_summaries(
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
        return

    def consolidate_TRITON_simulation_summaries(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Consolidate TRITON simulation outputs from all scenarios.

        Aggregates TRITON output files from individual scenarios into a single
        analysis-level summary file.

        Parameters
        ----------
        overwrite_if_exist : bool, optional
            If True, overwrite existing consolidated file (default: False)
        verbose : bool, optional
            If True, print progress messages (default: False)
        compression_level : int, optional
            Compression level for output file, 0-9 (default: 5)
        """
        self.process.consolidate_TRITON_outputs_for_analysis(
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )

    def consolidate_SWMM_simulation_summaries(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Consolidate SWMM simulation outputs from all scenarios.

        Aggregates SWMM node and link output files from individual scenarios into
        analysis-level summary files.

        Parameters
        ----------
        overwrite_if_exist : bool, optional
            If True, overwrite existing consolidated files (default: False)
        verbose : bool, optional
            If True, print progress messages (default: False)
        compression_level : int, optional
            Compression level for output files, 0-9 (default: 5)
        """
        self.process.consolidate_SWMM_outputs_for_analysis(
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )

    def print_cfg(self, which: Literal["system", "analysis", "both"] = "both"):
        """
        Print configuration settings in tabular format.

        Parameters
        ----------
        which : Literal["system", "analysis", "both"], optional
            Which configuration to print (default: "both")
        """
        if which == ["system", "both"]:
            print("=== System Configuration ===", flush=True)
            self._system.cfg_system.display_tabulate_cfg()
        if which == "both":
            print("\n", flush=True)
        if which in ["analysis", "both"]:
            print("=== analysis Configuration ===", flush=True)
            self.cfg_analysis.display_tabulate_cfg()

    def print_all_yaml_defined_input_files(self):
        print_json_file_tree(self.dict_of_exp_and_sys_config())

    def dict_of_exp_and_sys_config(self):
        dic_exp = self._system.cfg_system.model_dump()
        dic_sys = self.cfg_analysis.model_dump()
        return dic_exp | dic_sys

    def dict_of_all_sim_files(self, event_iloc):
        dic_syspaths = self._system.sys_paths.as_dict()
        dic_analysis_paths = self.analysis_paths.as_dict()
        scen = TRITONSWMM_scenario(event_iloc, self)
        dic_sim_paths = scen.scen_paths.as_dict()
        dic_all_paths = dic_syspaths | dic_analysis_paths | dic_sim_paths
        return dic_all_paths

    def print_all_sim_files(self, event_iloc):
        dic_all_paths = self.dict_of_all_sim_files(event_iloc)
        print_json_file_tree(dic_all_paths)

    def _retrieve_weather_indexer_using_integer_index(
        self,
        event_iloc,
    ):
        row = self.df_sims.loc[event_iloc, self.cfg_analysis.weather_event_indices]
        weather_event_indexers = row.to_dict()
        return weather_event_indexers

    @property
    def scenarios_not_created(self):
        """
        Get list of scenarios that have not been created successfully.

        Returns
        -------
        list of str
            Paths to scenario directories where creation is incomplete
        """
        scens_not_created = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            scen.log.refresh()
            if scen.log.scenario_creation_complete.get() is not True:
                scens_not_created.append(str(scen.log.logfile.parent))
        return scens_not_created

    @property
    def scenarios_not_run(self):
        """
        Get list of scenarios that have not been run successfully.

        Returns
        -------
        list of str
            Paths to scenario directories where simulation is incomplete
        """
        scens_not_run = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            if scen.sim_run_completed is not True:
                scens_not_run.append(str(scen.log.logfile.parent))
        return scens_not_run

    def TRITON_time_series_not_processed(self):
        scens_not_processed = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            if scen.log.TRITON_timeseries_written.get() is not True:
                scens_not_processed.append(str(scen.log.logfile.parent))
        return scens_not_processed

    def SWMM_time_series_not_processed(self):
        scens_not_processed = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            node_tseries_written = bool(scen.log.SWMM_node_timeseries_written.get())
            link_tseries_written = bool(scen.log.SWMM_link_timeseries_written.get())
            if not (node_tseries_written and link_tseries_written):
                scens_not_processed.append(str(scen.log.logfile.parent))
        return scens_not_processed

    def _update_log(self):
        self._refresh_log()
        # dict_all_logs = {}
        all_scens_created = True
        all_sims_run = True
        all_TRITON_outputs_processed = True
        all_SWMM_outputs_processed = True
        all_TRITONSWMM_performance_outputs_processed = True
        all_raw_TRITON_outputs_cleared = True
        all_raw_SWMM_outputs_cleared = True
        if self.cfg_analysis.toggle_sensitivity_analysis is True:
            sens = self.sensitivity
            all_scens_created = sens.all_scenarios_created
            all_sims_run = sens.all_sims_run
            all_TRITON_outputs_processed = sens.all_TRITON_timeseries_processed
            all_SWMM_outputs_processed = sens.all_SWMM_timeseries_processed
            all_TRITONSWMM_performance_outputs_processed = (
                sens.all_TRITONSWMM_performance_timeseries_processed
            )
            all_raw_TRITON_outputs_cleared = sens.all_raw_TRITON_outputs_cleared
            all_raw_SWMM_outputs_cleared = sens.all_raw_SWMM_outputs_cleared
        else:
            for event_iloc in self.df_sims.index:
                scen = TRITONSWMM_scenario(event_iloc, self)
                scen.log.refresh()
                # dict_all_logs[event_iloc] = scen.log.model_dump()
                # sim run status
                all_sims_run = all_sims_run and scen.sim_run_completed
                # sim creation status
                scen_created = bool(scen.log.scenario_creation_complete.get())
                all_scens_created = all_scens_created and scen_created
                # sim output processing status
                proc = self._retrieve_sim_run_processing_object(event_iloc)
                all_TRITON_outputs_processed = (
                    all_TRITON_outputs_processed and proc.TRITON_outputs_processed
                )
                all_SWMM_outputs_processed = (
                    all_SWMM_outputs_processed and proc.SWMM_outputs_processed
                )
                # TRITONSWMM performance processing status
                all_TRITONSWMM_performance_outputs_processed = (
                    all_TRITONSWMM_performance_outputs_processed
                    and proc.TRITONSWMM_performance_timeseries_written
                )
                # output clear status
                all_raw_TRITON_outputs_cleared = (
                    all_raw_TRITON_outputs_cleared and proc.raw_TRITON_outputs_cleared
                )
                all_raw_SWMM_outputs_cleared = (
                    all_raw_SWMM_outputs_cleared and proc.raw_SWMM_outputs_cleared
                )
        self.log.all_scenarios_created.set(all_scens_created)
        self.log.all_sims_run.set(all_sims_run)
        self.log.all_TRITON_timeseries_processed.set(all_TRITON_outputs_processed)
        self.log.all_SWMM_timeseries_processed.set(all_SWMM_outputs_processed)
        self.log.all_TRITONSWMM_performance_timeseries_processed.set(
            all_TRITONSWMM_performance_outputs_processed
        )
        self.log.all_raw_TRITON_outputs_cleared.set(all_raw_TRITON_outputs_cleared)
        self.log.all_raw_SWMM_outputs_cleared.set(all_raw_SWMM_outputs_cleared)
        return

    def retrieve_prepare_scenario_launchers(
        self,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
    ):
        """
        Create subprocess-based launchers for scenario preparation.

        Each launcher runs scenario preparation in an isolated subprocess to avoid
        PySwmm's MultiSimulationError when preparing multiple scenarios concurrently.

        Parameters
        ----------
        overwrite_scenario : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        verbose : bool
            If True, print progress messages

        Returns
        -------
        list
            List of launcher functions that execute scenario preparation in subprocesses
        """
        prepare_scenario_launchers = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)

            # Create a subprocess-based launcher
            launcher = scen._create_subprocess_prepare_scenario_launcher(
                overwrite_scenario=overwrite_scenario,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
                verbose=verbose,
            )
            prepare_scenario_launchers.append(launcher)

        return prepare_scenario_launchers

    def retrieve_scenario_timeseries_processing_launchers(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Create subprocess-based launchers for scenario timeseries processing.

        Each launcher runs timeseries processing in an isolated subprocess to avoid
        potential conflicts when processing multiple scenarios' outputs concurrently.

        Parameters
        ----------
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process: TRITON, SWMM, or both
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        overwrite_if_exist : bool
            If True, overwrite existing processed outputs
        verbose : bool
            If True, print progress messages
        compression_level : int
            Compression level for output files (0-9)

        Returns
        -------
        list
            List of launcher functions that execute timeseries processing in subprocesses
        """
        scenario_timeseries_processing_launchers = []
        for event_iloc in self.df_sims.index:
            proc = self._retrieve_sim_run_processing_object(event_iloc=event_iloc)

            # Create a subprocess-based launcher
            launcher = proc._create_subprocess_timeseries_processing_launcher(
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
            scenario_timeseries_processing_launchers.append(launcher)

        return scenario_timeseries_processing_launchers

    def calculate_effective_max_parallel(
        self,
        min_memory_per_function_MiB: int | None = 1024,
        max_concurrent: int | None = None,
        verbose: bool = False,
    ) -> int:
        """
        Calculate the effective maximum parallelism based on CPU, GPU, memory, and SLURM constraints.

        This method delegates to ResourceManager for resource allocation calculations.

        Parameters
        ----------
        min_memory_per_function_MiB : int | None
            Minimum memory required per function (MiB).
            If provided, concurrency is reduced to avoid oversubscription.
        max_concurrent : int | None
            CPU-based upper bound on parallelism (e.g., based on cores/threads per task).
            If None, defaults to physical CPU count - 1 (or SLURM allocation if in SLURM).
        verbose : bool
            Print progress messages.

        Returns
        -------
        int
            The effective maximum number of parallel tasks.
        """
        return self._resource_manager.calculate_effective_max_parallel(
            min_memory_per_function_MiB=min_memory_per_function_MiB,
            max_concurrent=max_concurrent,
            verbose=verbose,
        )

    def run_python_functions_concurrently(
        self,
        function_launchers: List[Callable[[], None]],
        min_memory_per_function_MiB: int | None = 1024,
        max_parallel: int | None = None,
        verbose: bool = True,
    ) -> List[int]:
        """
        Run Python functions concurrently, limiting parallelism by CPU and memory.

        Parameters
        ----------
        function_launchers : List[Callable[[], None]]
            Functions to execute concurrently.
        max_parallel : int | None
            Upper bound on parallelism (defaults to CPU count).
        min_memory_per_function_MiB : int | None
            Minimum memory required per function (MiB).
            If provided, concurrency is reduced to avoid oversubscription.
        verbose : bool
            Print progress messages.

        Returns
        -------
        List[int]
            Indices of functions that completed successfully.
        """

        effective_max_parallel = self.calculate_effective_max_parallel(
            min_memory_per_function_MiB=min_memory_per_function_MiB,
            max_concurrent=max_parallel,
            verbose=verbose,
        )

        if verbose:
            print(
                f"Running {len(function_launchers)} functions "
                f"(max parallel = {effective_max_parallel})",
                flush=True,
            )

        results: List[int] = []
        batch_start = time.time()  # Reference point for all tasks

        def wrapper(idx: int, launcher: Callable[[], None]):
            task_start = time.time()
            launcher()
            task_end = time.time()

            duration = task_end - task_start
            completed_at = task_end - batch_start

            if verbose:
                print(
                    f"Function {idx}: duration={duration:.2f}s, "
                    f"completed_at={completed_at:.2f}s",
                    flush=True,
                )
            return idx

        # ----------------------------
        # Execute
        # ----------------------------
        with ThreadPoolExecutor(max_workers=effective_max_parallel) as executor:
            futures = {
                executor.submit(wrapper, idx, launcher): idx
                for idx, launcher in enumerate(function_launchers)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    if verbose:
                        print(f"Function {idx} failed with error: {e}", flush=True)

        self._update_log()
        return results

    def run_prepare_scenarios_serially(
        self,
        overwrite_scenarios: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
    ):
        """
        Prepare all scenarios sequentially.

        Executes scenario preparation for all scenarios in serial order, updating
        logs after each scenario completes.

        Parameters
        ----------
        overwrite_scenarios : bool, optional
            If True, overwrite existing scenarios (default: False)
        rerun_swmm_hydro_if_outputs_exist : bool, optional
            If True, rerun SWMM hydrology model even if outputs exist (default: False)
        verbose : bool, optional
            If True, print progress messages (default: False)
        """
        prepare_scenario_launchers = self.retrieve_prepare_scenario_launchers(
            overwrite_scenario=overwrite_scenarios,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            verbose=verbose,
        )
        for launcher in prepare_scenario_launchers:
            launcher()
            self._update_log()  # update logs
        return

    def print_logfile_for_scenario(self, event_iloc):
        scen = TRITONSWMM_scenario(event_iloc, self)
        scen.log.print()

    def run_sim(
        self,
        event_iloc: int,
        pickup_where_leftoff,
        process_outputs_after_sim_completion: bool,
        which: Literal["TRITON", "SWMM", "both"],
        clear_raw_outputs: bool,
        overwrite_if_exist: bool,
        compression_level: int,
        verbose=False,
    ):
        """
        Run a single simulation for the specified scenario.

        Executes the TRITON-SWMM simulation for a specific weather event scenario,
        optionally processing outputs after completion.

        Parameters
        ----------
        event_iloc : int
            Integer index of the scenario in df_sims
        pickup_where_leftoff : bool
            If True, resume simulation from last checkpoint
        process_outputs_after_sim_completion : bool
            If True, process timeseries outputs after simulation completes
        which : Literal["TRITON", "SWMM", "both"]
            Which outputs to process (only used if process_outputs_after_sim_completion=True)
        clear_raw_outputs : bool
            If True, clear raw outputs after processing
        overwrite_if_exist : bool
            If True, overwrite existing processed outputs
        compression_level : int
            Compression level for output files, 0-9
        verbose : bool, optional
            If True, print progress messages (default: False)

        Raises
        ------
        ValueError
            If scenario creation is incomplete or TRITONSWMM is not compiled
        """
        scen = TRITONSWMM_scenario(event_iloc, self)

        if not scen.log.scenario_creation_complete.get():
            print("Log file:", flush=True)
            print(scen.log.print())
            raise ValueError("scenario_creation_complete must be 'success'")
        if not self._system.compilation_successful:
            print("Log file:", flush=True)
            print(scen.log.print())
            raise ValueError("TRITONSWMM has not been compiled")
        run = self._retrieve_sim_runs(event_iloc)
        if verbose:
            print("run instance instantiated", flush=True)

        # Use the subprocess launcher pattern, mirroring process_sim_timeseries
        launcher, finalize_sim = run._create_subprocess_sim_run_launcher(
            pickup_where_leftoff=pickup_where_leftoff,
            verbose=verbose,
        )
        # Launch the simulation (non-blocking)
        proc, start_time, sim_logfile, lf = launcher()
        # Wait for simulation to complete and update simlog
        finalize_sim(proc, start_time, sim_logfile, lf)

        self.sim_run_status(event_iloc)
        # self._update_log()  # updates analysis log
        if process_outputs_after_sim_completion and run._scenario.sim_run_completed:
            self.process_sim_timeseries(
                event_iloc,
                which,
                clear_raw_outputs,
                overwrite_if_exist,
                verbose,
                compression_level,
            )
        return

    def process_sim_timeseries(
        self,
        event_iloc,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        """
        Process and write timeseries outputs for a single simulation.

        Converts raw TRITON and/or SWMM outputs into processed timeseries files,
        optionally clearing raw outputs after processing.

        Parameters
        ----------
        event_iloc : int
            Integer index of the scenario in df_sims
        which : Literal["TRITON", "SWMM", "both"], optional
            Which outputs to process (default: "both")
        clear_raw_outputs : bool, optional
            If True, clear raw outputs after processing (default: True)
        overwrite_if_exist : bool, optional
            If True, overwrite existing processed outputs (default: False)
        verbose : bool, optional
            If True, print progress messages (default: False)
        compression_level : int, optional
            Compression level for output files, 0-9 (default: 5)
        """
        proc = self._retrieve_sim_run_processing_object(event_iloc=event_iloc)
        proc.write_timeseries_outputs(
            which=which,
            clear_raw_outputs=clear_raw_outputs,
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )

    def process_all_sim_timeseries_serially(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        for event_iloc in self.df_sims.index:
            self.process_sim_timeseries(
                event_iloc=event_iloc,
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        self._update_log()
        return

    def consolidate_analysis_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        if which == "TRITON" or which == "both":
            self.process.consolidate_TRITON_outputs_for_analysis(
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        if which == "SWMM" or which == "both":
            self.process.consolidate_SWMM_outputs_for_analysis(
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        return

    def sim_run_status(self, event_iloc):
        run = self._retrieve_sim_runs(event_iloc)
        status = run._scenario.latest_simlog
        self._simulation_run_statuses[event_iloc] = status
        return status

    def _retrieve_sim_runs(self, event_iloc):
        scen = TRITONSWMM_scenario(event_iloc, self)
        run = scen.run
        self._sim_run_objects[event_iloc] = run
        return run

    def _retrieve_sim_run_processing_object(self, event_iloc):
        run = self._retrieve_sim_runs(event_iloc)
        proc = TRITONSWMM_sim_post_processing(run)
        self._sim_run_processing_objects[event_iloc] = proc
        return proc

    def _create_launchable_sims(
        self,
        pickup_where_leftoff: bool = False,
        verbose: bool = False,
    ):
        """
        Create launcher functions for all simulations.

        Uses the consolidated _create_subprocess_sim_run_launcher pattern
        which handles the complete simulation lifecycle including simlog updates.

        The execution method (local, batch_job, or 1_job_many_srun_tasks) is
        determined by self.cfg_analysis.multi_sim_run_method.

        Parameters
        ----------
        pickup_where_leftoff : bool
            If True, resume simulations from last checkpoint
        verbose : bool
            If True, print progress messages

        Returns
        -------
        list
            List of launcher functions
        """
        launch_and_finalize_functions_tuples = []
        for event_iloc in self.df_sims.index:
            run = self._retrieve_sim_runs(event_iloc)
            launch_and_finalize_functions_tuple = (
                run._create_subprocess_sim_run_launcher(
                    pickup_where_leftoff=pickup_where_leftoff,
                    verbose=verbose,
                )
            )
            if launch_and_finalize_functions_tuple is None:
                continue
            launch_and_finalize_functions_tuples.append(
                launch_and_finalize_functions_tuple
            )
        return launch_and_finalize_functions_tuples

    def run_simulations_concurrently(
        self,
        launch_functions: list[tuple],
        max_concurrent: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Run simulations concurrently using the configured execution strategy.

        Automatically selects the appropriate executor based on cfg_analysis.multi_sim_run_method:
        - "1_job_many_srun_tasks": Uses SlurmExecutor for HPC execution
        - "local": Uses LocalConcurrentExecutor for parallel local execution
        - Other: Uses SerialExecutor for sequential execution

        Parameters
        ----------
        launch_functions : list[tuple]
            List of tuples (launcher, finalize_sim) from _create_subprocess_sim_run_launcher()
        max_concurrent : Optional[int]
            Maximum number of concurrent simulations
        verbose : bool
            If True, print progress messages

        Returns
        -------
        list
            List of simulation statuses
        """
        return self._execution_strategy.execute_simulations(
            launch_functions, max_concurrent, verbose
        )

    def run_sims_in_sequence(
        self,
        pickup_where_leftoff,
        process_outputs_after_sim_completion: bool = False,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        verbose=False,
    ):
        """
        Arguments passed to run:
            - mode: Mode | Literal["single_core"]
            - pickup_where_leftoff
        Arguments passed to processing process_sim_timeseriess (and only needed if process_outputs_after_sim_completion=True):
            - which: Literal["TRITON", "SWMM", "both"]
            - clear_raw_outputs: bool
            - overwrite_if_exist: bool
            - compression_level: int
        """
        if verbose:
            print("Running all sims in series...", flush=True)
        for event_iloc in self.df_sims.index:
            if verbose:
                print(
                    f"Running sim {event_iloc} and pickup_where_leftoff = {pickup_where_leftoff}",
                    flush=True,
                )
            self.run_sim(
                event_iloc=event_iloc,
                pickup_where_leftoff=pickup_where_leftoff,
                verbose=verbose,
                process_outputs_after_sim_completion=process_outputs_after_sim_completion,
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                compression_level=compression_level,
            )
        self._update_log()

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
        wait_for_completion: bool = False,  # relevant for slurm jobs only
        verbose: bool = True,
    ) -> dict:
        """
        Submit workflow using Snakemake (replaces submit_SLURM_job_array).

        Automatically detects execution context (local vs. HPC) and submits accordingly.

        Delegates to SnakemakeWorkflowBuilder.

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
            Status dictionary with keys:
            - success: bool - Whether workflow succeeded
            - mode: str - "local" or "slurm"
            - snakefile_path: Path - Path to generated Snakefile
            - job_id: str | None - Job ID (only for slurm mode)
            - message: str - Status message
        """
        return self._workflow_builder.submit_workflow(
            mode=mode,
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
            wait_for_completion=wait_for_completion,
            verbose=verbose,
        )

    @property
    def TRITONSWMM_runtimes(self):
        return (
            self.TRITON_summary["compute_time_min"]
            .to_dataframe()
            .dropna()["compute_time_min"]
        )

    @property
    def TRITON_analysis_summary_created(self):
        return bool(self.log.TRITON_analysis_summary_created.get())

    @property
    def SWMM_node_analysis_summary_created(self):
        return bool(self.log.SWMM_node_analysis_summary_created.get())

    @property
    def SWMM_link_analysis_summary_created(self):
        return bool(self.log.SWMM_link_analysis_summary_created.get())

    @property
    def SWMM_node_summary(self):
        return self.process.SWMM_node_summary

    @property
    def SWMM_link_summary(self):
        return self.process.SWMM_link_summary

    @property
    def df_status(self):
        """
        Get status DataFrame for all scenarios in the analysis.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - All columns from df_sims (weather event indices)
            - scenarios_setup: bool - Whether scenario preparation succeeded
            - scen_runs_completed: bool - Whether simulation completed
            - scenario_directory: str - Path to scenario directory
        """
        scenarios_setup = []
        scen_runs_completed = []
        scenario_dirs = []

        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            scenarios_setup.append(scen.log.scenario_creation_complete.get() is True)
            scen_runs_completed.append(scen.sim_run_completed)
            scenario_dirs.append(str(scen.log.logfile.parent))

        df_status = self.df_sims.copy()
        df_status["scenarios_setup"] = scenarios_setup
        df_status["scen_runs_completed"] = scen_runs_completed
        df_status["scenario_directory"] = scenario_dirs

        return df_status

    @property
    def TRITON_summary(self):
        """
        Get consolidated TRITON output summary for the analysis.

        Returns
        -------
        xarray.Dataset
            Consolidated TRITON outputs from all scenarios
        """
        return self.process.TRITON_summary
