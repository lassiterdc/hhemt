# %%
import subprocess
import shutil
from TRITON_SWMM_toolkit.utils import (
    create_from_template,
    read_text_file_as_string,
)
from pathlib import Path
from TRITON_SWMM_toolkit.config import load_analysis_config
import pandas as pd
from typing import Literal
from TRITON_SWMM_toolkit.paths import AnalysisPaths
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.run_simulation import TRITONSWMM_run
from TRITON_SWMM_toolkit.process_simulation import TRITONSWMM_sim_post_processing

from TRITON_SWMM_toolkit.processing_analysis import TRITONSWMM_analysis_post_processing
from TRITON_SWMM_toolkit.constants import Mode
from TRITON_SWMM_toolkit.plot_utils import print_json_file_tree
from TRITON_SWMM_toolkit.log import TRITONSWMM_analysis_log
from TRITON_SWMM_toolkit.plot_analysis import TRITONSWMM_analysis_plotting
from TRITON_SWMM_toolkit.sensitivity_analysis import TRITONSWMM_sensitivity_analysis
import yaml
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Optional
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import threading
import psutil

if TYPE_CHECKING:
    from .system import TRITONSWMM_system


class TRITONSWMM_analysis:
    def __init__(
        self,
        analysis_config_yaml: Path,
        system: "TRITONSWMM_system",
        analysis_dir: Optional[Path] = None,
        compiled_TRITONSWMM_directory: Optional[Path] = None,
        skip_log_update: bool = False,
    ) -> None:
        self._system = system
        self.analysis_config_yaml = analysis_config_yaml
        cfg_analysis = load_analysis_config(analysis_config_yaml)
        self.cfg_analysis = cfg_analysis
        # define additional paths not defined in cfg
        if compiled_TRITONSWMM_directory is None:
            compiled_TRITONSWMM_directory = (
                self._system.cfg_system.system_directory
                / self.cfg_analysis.analysis_id
                / "compiled_software"
            )
        if analysis_dir is None:
            analysis_dir = (
                self._system.cfg_system.system_directory / self.cfg_analysis.analysis_id
            )

        self.analysis_paths = AnalysisPaths(
            f_log=analysis_dir / "log.json",
            analysis_dir=analysis_dir,
            compiled_TRITONSWMM_directory=compiled_TRITONSWMM_directory,
            TRITON_build_dir=compiled_TRITONSWMM_directory / "build",
            compilation_script=compiled_TRITONSWMM_directory / "compile.sh",
            simulation_directory=analysis_dir / "sims",
            compilation_logfile=compiled_TRITONSWMM_directory / f"compilation.log",
            output_triton_summary=analysis_dir
            / f"TRITON.{self.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_links_summary=analysis_dir
            / f"SWMM_links.{self.cfg_analysis.TRITON_processed_output_type}",
            output_swmm_node_summary=analysis_dir
            / f"SWMM_nodes.{self.cfg_analysis.TRITON_processed_output_type}",
        )
        # if self.cfg_analysis.toggle_run_ensemble_with_bash_script == True:
        #     self.analysis_paths.bash_script_path = analysis_dir / "run_ensemble.sh"
        self.df_sims = pd.read_csv(self.cfg_analysis.weather_events_to_simulate).loc[
            :, self.cfg_analysis.weather_event_indices
        ]
        self._sim_run_objects = {}
        self._sim_run_processing_objects = {}
        self._simulation_run_statuses = {}
        # self.run_modes = Mode
        # self.compilation_successful = False
        self.in_slurm = "SLURM_JOB_ID" in os.environ.copy()
        self.process = TRITONSWMM_analysis_post_processing(self)
        self.plot = TRITONSWMM_analysis_plotting(self)
        if not skip_log_update:
            # self._add_all_scenarios()
            self._refresh_log()
            if self.analysis_paths.compilation_logfile.exists():
                self._validate_compilation()
            self._update_log()
        if self.cfg_analysis.toggle_sensitivity_analysis == True:
            self.sensitivity = TRITONSWMM_sensitivity_analysis(self)

    def _refresh_log(self):
        if self.analysis_paths.f_log.exists():
            self.log = TRITONSWMM_analysis_log.from_json(self.analysis_paths.f_log)
        else:
            self.log = TRITONSWMM_analysis_log(logfile=self.analysis_paths.f_log)

    @property
    def compilation_successful(self):
        return self._validate_compilation()

    def consolidate_TRITON_and_SWMM_simulation_summaries(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
        which: Literal["TRITON", "SWMM", "both"] = "both",
    ):
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
        return

    def consolidate_TRITON_simulation_summaries(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
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
        self.process.consolidate_SWMM_outputs_for_analysis(
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )

    def print_cfg(self, which: Literal["system", "analysis", "both"] = "both"):
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
        scens_not_created = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            scen.log.refresh()
            if scen.log.scenario_creation_complete.get() != True:
                scens_not_created.append(str(scen.log.logfile.parent))
        return scens_not_created

    @property
    def scenarios_not_run(self):
        scens_not_run = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            if scen.sim_run_completed != True:
                scens_not_run.append(str(scen.log.logfile.parent))
        return scens_not_run

    def TRITON_time_series_not_processed(self):
        scens_not_processed = []
        for event_iloc in self.df_sims.index:
            scen = TRITONSWMM_scenario(event_iloc, self)
            if scen.log.TRITON_timeseries_written.get() != True:
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
        # dict_all_logs = {}
        all_scens_created = True
        all_sims_run = True
        all_TRITON_outputs_processed = True
        all_SWMM_outputs_processed = True
        all_raw_TRITON_outputs_cleared = True
        all_raw_SWMM_outputs_cleared = True
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
        self.log.all_raw_TRITON_outputs_cleared.set(all_raw_TRITON_outputs_cleared)
        self.log.all_raw_SWMM_outputs_cleared.set(all_raw_SWMM_outputs_cleared)
        return

    def retrieve_prepare_scenario_launchers(
        self,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
        compiled_TRITONSWMM_directory: Optional[Path] = None,
        analysis_dir: Optional[Path] = None,
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
                compiled_TRITONSWMM_directory=compiled_TRITONSWMM_directory,
                analysis_dir=analysis_dir,
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
        analysis_dir: Optional[Path] = None,
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
        analysis_dir : Optional[Path]
            Optional path to analysis directory (mainly used for sensitivity analysis)

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
                analysis_dir=analysis_dir,
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

        This method respects SLURM job allocation constraints when running in a SLURM environment,
        ensuring that concurrent tasks don't exceed the job's resource allocation.

        Parameters
        ----------
        min_memory_per_function_MiB : int | None
            Minimum memory required per function (MiB).
            If provided, concurrency is reduced to avoid oversubscription.
        max_concurrent : int | None
            CPU-based upper bound on parallelism (e.g., based on cores/threads per task).
            If None, defaults to physical CPU count - 1 (or SLURM allocation if in SLURM).
        use_slurm_constraints : bool | None
            If True, use SLURM environment variables to constrain parallelism.
            If None (default), automatically uses SLURM constraints if self.in_slurm is True.
            If False, uses pure hardware-based constraints.
        verbose : bool
            Print progress messages.

        Returns
        -------
        int
            The effective maximum number of parallel tasks.
        """
        # ----------------------------
        # Determine if we should use SLURM constraints
        # ----------------------------
        use_slurm_constraints = self.in_slurm == True

        # ----------------------------
        # CPU-based limit (with SLURM awareness)
        # ----------------------------
        if max_concurrent is None:
            if use_slurm_constraints:
                # Get SLURM-aware CPU limit
                constraints = self._get_slurm_resource_constraints(verbose=verbose)
                max_concurrent = int(constraints["max_concurrent"])
            else:
                # Pure hardware-based calculation
                physical_cores = psutil.cpu_count(logical=False)
                if isinstance(physical_cores, int) and physical_cores > 1:
                    physical_cores -= 1  # more conservative process count
                max_concurrent_cpu = physical_cores or 1
                # ----------------------------
                # Memory-based limit
                # ----------------------------
                mem_limit = max_concurrent_cpu
                if min_memory_per_function_MiB is not None:
                    available_mem_MiB = psutil.virtual_memory().available // (1024**2)
                    mem_limit = max(1, available_mem_MiB // min_memory_per_function_MiB)

                    if verbose:
                        print(
                            f"Memory-based limit: {mem_limit} "
                            f"(available {available_mem_MiB} MiB, "
                            f"{min_memory_per_function_MiB} MiB per task)",
                            flush=True,
                        )
                # ----------------------------
                # Final concurrency (apply all constraints)
                # ----------------------------
                limits = [int(max_concurrent_cpu), int(mem_limit)]

                max_concurrent = min(limits)

                if verbose and use_slurm_constraints and self.in_slurm:
                    print(
                        f"[SLURM] Using SLURM-aware concurrency limit: {max_concurrent}",
                        flush=True,
                    )

        return max_concurrent

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
                except:
                    pass

        self._update_log()
        return results

    def run_prepare_scenarios_serially(
        self,
        overwrite_scenarios: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
        compiled_TRITONSWMM_directory: Optional[Path] = None,
        analysis_dir: Optional[Path] = None,
    ):
        prepare_scenario_launchers = self.retrieve_prepare_scenario_launchers(
            overwrite_scenario=overwrite_scenarios,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            verbose=verbose,
            compiled_TRITONSWMM_directory=compiled_TRITONSWMM_directory,
            analysis_dir=analysis_dir,
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
        analysis_dir: Optional[Path] = None,
        compiled_TRITONSWMM_directory: Optional[Path] = None,
    ):
        scen = TRITONSWMM_scenario(event_iloc, self)

        if not scen.log.scenario_creation_complete.get():
            print("Log file:", flush=True)
            print(scen.log.print())
            raise ValueError("scenario_creation_complete must be 'success'")
        if not self.compilation_successful:
            print("Log file:", flush=True)
            print(scen.log.print())
            raise ValueError("TRITONSWMM has not been compiled")
        run = self._retreive_sim_runs(event_iloc)
        if verbose:
            print("run instance instantiated", flush=True)

        # Use the subprocess launcher pattern, mirroring process_sim_timeseries
        launcher, finalize_sim = run._create_subprocess_sim_run_launcher(
            pickup_where_leftoff=pickup_where_leftoff,
            verbose=verbose,
            analysis_dir=analysis_dir,
            compiled_TRITONSWMM_directory=compiled_TRITONSWMM_directory,
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

    def retreive_scenario_timeseries_processing_launchers(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
        analysis_dir: Optional[Path] = None,
    ):
        scenario_timeseries_processing_launchers = []
        for event_iloc in self.df_sims.index:
            proc = self._retrieve_sim_run_processing_object(event_iloc=event_iloc)
            launcher = proc._create_subprocess_timeseries_processing_launcher(
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
                analysis_dir=analysis_dir,
            )
            scenario_timeseries_processing_launchers.append(launcher)
        return scenario_timeseries_processing_launchers

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
        Creates time series of TRITON-SWMM outputs
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

    def consolidate_analysis_outptus(
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
        run = self._retreive_sim_runs(event_iloc)
        status = run._scenario.latest_simlog
        self._simulation_run_statuses[event_iloc] = status
        return status

    def _retreive_sim_runs(self, event_iloc):
        scen = TRITONSWMM_scenario(event_iloc, self)
        run = scen.run
        self._sim_run_objects[event_iloc] = run
        return run

    def _retrieve_sim_run_processing_object(self, event_iloc):
        run = self._retreive_sim_runs(event_iloc)
        proc = TRITONSWMM_sim_post_processing(run)
        self._sim_run_processing_objects[event_iloc] = proc
        return proc

    def _parse_slurm_tasks_per_node(self, tasks_per_node_str: str) -> list[int]:
        """
        Parse SLURM_TASKS_PER_NODE format.

        Examples:
            "4,4,4,4" -> [4, 4, 4, 4]
            "4(x4)" -> [4, 4, 4, 4]
            "8(x2),4" -> [8, 8, 4]

        Parameters
        ----------
        tasks_per_node_str : str
            SLURM_TASKS_PER_NODE environment variable value

        Returns
        -------
        list[int]
            Tasks per node for each allocated node
        """
        tasks = []
        for part in tasks_per_node_str.split(","):
            part = part.strip()
            if "(x" in part:
                # Format: "4(x4)" means 4 tasks repeated 4 times
                count_str, repeat_str = part.split("(x")
                count = int(count_str)
                repeat = int(repeat_str.rstrip(")"))
                tasks.extend([count] * repeat)
            else:
                # Simple format: "4" means 4 tasks
                tasks.append(int(part))
        return tasks

    def _get_slurm_resource_constraints(
        self, verbose: bool = False, min_mem_per_sim_MiB: int = 1024
    ) -> dict:
        """
        Extract and validate SLURM resource constraints from environment variables.

        This method reads all relevant SLURM environment variables and calculates
        the effective maximum concurrency based on:
        - CPU allocation (SLURM_CPUS_PER_TASK, SLURM_CPUS_ON_NODE)
        - GPU allocation (SLURM_GPUS, SLURM_GPUS_ON_NODE, SLURM_GPUS_PER_TASK)
        - Memory constraints (SLURM_MEM_PER_NODE, SLURM_MEM_PER_CPU)
        - Multi-node distribution (SLURM_TASKS_PER_NODE, SLURM_JOB_NUM_NODES)

        Returns
        -------
        dict
            Dictionary containing:
            - max_concurrent: int - Maximum concurrent simulations
            - total_cpus: int - Total CPUs allocated to job
            - total_gpus: int - Total GPUs allocated (if GPU mode)
            - cpus_per_task: int - CPUs per task from SLURM
            - gpus_per_task: int - GPUs per task from config
            - num_nodes: int - Number of nodes allocated
            - cpus_per_node: int - CPUs per node
            - memory_per_node_MiB: int - Memory per node in MiB
            - run_mode: str - CPU or GPU mode
        """
        # ----------------------------
        # Read basic SLURM allocation
        # ----------------------------
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        cpus_on_node = int(os.environ.get("SLURM_CPUS_ON_NODE", 0))

        # If SLURM_CPUS_ON_NODE not set, use psutil
        if cpus_on_node == 0:
            cpus_on_node = psutil.cpu_count(logical=False) or 1

        # ----------------------------
        # Calculate total CPUs available
        # ----------------------------
        # Total CPUs = CPUs per node × number of nodes
        total_cpus = cpus_on_node * num_nodes

        # But respect SLURM_CPUS_PER_TASK if it's more restrictive
        # (i.e., if job was allocated fewer CPUs than node capacity)
        slurm_total_cpus = int(os.environ.get("SLURM_NTASKS", 1)) * cpus_per_task
        if slurm_total_cpus > 0 and slurm_total_cpus < total_cpus:
            total_cpus = slurm_total_cpus

        # ----------------------------
        # Memory constraints
        # ----------------------------
        mem_per_node_MiB = 0
        mem_per_cpu_MiB = 0

        # Try to get memory per node
        mem_per_node_str = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_per_node_str:
            # Format: "123456" (in MB) or "123456M" or "123G"
            mem_per_node_str = mem_per_node_str.rstrip("M")
            try:
                mem_per_node_MiB = int(mem_per_node_str)
            except ValueError:
                if mem_per_node_str.endswith("G"):
                    mem_per_node_MiB = int(mem_per_node_str[:-1]) * 1024

        # Try to get memory per CPU
        mem_per_cpu_str = os.environ.get("SLURM_MEM_PER_CPU")
        if mem_per_cpu_str:
            mem_per_cpu_str = mem_per_cpu_str.rstrip("M")
            try:
                mem_per_cpu_MiB = int(mem_per_cpu_str)
            except ValueError:
                if mem_per_cpu_str.endswith("G"):
                    mem_per_cpu_MiB = int(mem_per_cpu_str[:-1]) * 1024

        # Calculate effective memory per node
        if mem_per_node_MiB == 0 and mem_per_cpu_MiB > 0:
            mem_per_node_MiB = mem_per_cpu_MiB * cpus_on_node

        # ----------------------------
        # GPU constraints (if GPU mode)
        # ----------------------------
        total_gpus = 0
        gpus_per_task = self.cfg_analysis.n_gpus or 0

        if self.cfg_analysis.run_mode == "gpu":
            # Try SLURM_GPUS first (total GPUs)
            total_gpus = int(os.environ.get("SLURM_GPUS", 0))

            # If not set, try SLURM_GPUS_ON_NODE
            if total_gpus == 0:
                gpus_on_node = int(os.environ.get("SLURM_GPUS_ON_NODE", 0))
                total_gpus = gpus_on_node * num_nodes

            # Validate GPU allocation
            if total_gpus == 0:
                raise RuntimeError(
                    "GPU run mode requested, but no GPUs detected via SLURM. "
                    "Check SLURM_GPUS or SLURM_GPUS_ON_NODE environment variables."
                )

            if gpus_per_task > total_gpus:
                raise RuntimeError(
                    f"Each simulation requires {gpus_per_task} GPU(s), "
                    f"but only {total_gpus} GPU(s) allocated to the job."
                )

        # ----------------------------
        # Calculate max concurrency based on mode
        # ----------------------------
        if self.cfg_analysis.run_mode == "gpu":
            # GPU-based limit
            max_concurrent = max(1, total_gpus // gpus_per_task)
        else:
            # CPU-based limit
            mpi_ranks = self.cfg_analysis.n_mpi_procs or 1
            omp_threads = self.cfg_analysis.n_omp_threads or 1
            cpus_per_sim = mpi_ranks * omp_threads

            max_concurrent = max(1, total_cpus // cpus_per_sim) - 1
        # ----------------------------
        # Apply memory constraints
        # ----------------------------
        if mem_per_node_MiB > 0:
            # Estimate memory per simulation
            # This is conservative: assume each sim uses proportional memory
            available_mem_MiB = mem_per_node_MiB * num_nodes
            # For now, we don't have a per-sim memory requirement from config
            # But we can add a safety factor to prevent oversubscription
            # Assume each concurrent task uses ~1GB by default (can be overridden)
            mem_based_limit = max(1, available_mem_MiB // min_mem_per_sim_MiB)
            max_concurrent = min(max_concurrent, mem_based_limit)

        # ----------------------------
        # Respect multi-node task distribution
        # ----------------------------
        tasks_per_node_env = os.environ.get("SLURM_TASKS_PER_NODE")
        if tasks_per_node_env:
            tasks_per_node = self._parse_slurm_tasks_per_node(tasks_per_node_env)
            # Use minimum to be conservative
            if tasks_per_node:
                max_tasks_per_node = min(tasks_per_node)
                max_concurrent = min(max_concurrent, max_tasks_per_node * num_nodes)

        # ----------------------------
        # Verbose logging
        # ----------------------------
        if verbose:
            print(f"[SLURM] Resource Constraints:", flush=True)
            print(f"  Nodes: {num_nodes}", flush=True)
            print(f"  CPUs per node: {cpus_on_node}", flush=True)
            print(f"  Total CPUs Allocated (SLURM): {slurm_total_cpus}", flush=True)
            print(f"  CPUs per task (SLURM): {cpus_per_task}")
            if self.cfg_analysis.run_mode == "gpu":
                print(f"  Total GPUs: {total_gpus}", flush=True)
                print(f"  GPUs per task: {gpus_per_task}", flush=True)
            if mem_per_node_MiB > 0:
                print(f"  Memory per node: {mem_per_node_MiB} MiB", flush=True)
            print(f"  Max concurrent srun tasks: {max_concurrent}", flush=True)

        return {
            "max_concurrent": max_concurrent,
            "total_cpus": total_cpus,
            "total_gpus": total_gpus,
            "cpus_per_task": cpus_per_task,
            "gpus_per_task": gpus_per_task,
            "num_nodes": num_nodes,
            "cpus_per_node": cpus_on_node,
            "memory_per_node_MiB": mem_per_node_MiB,
        }

    def _create_launchable_sims(
        self,
        pickup_where_leftoff: bool = False,
        verbose: bool = False,
        compiled_TRITONSWMM_directory: Optional[Path] = None,
        analysis_dir: Optional[Path] = None,
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
        analysis_dir : Optional[Path]
            Optional path to analysis directory (mainly used for sensitivity analysis)

        Returns
        -------
        list
            List of launcher functions
        """
        launch_and_finalize_functions_tuples = []
        for event_iloc in self.df_sims.index:
            run = self._retreive_sim_runs(event_iloc)
            launch_and_finalize_functions_tuple = (
                run._create_subprocess_sim_run_launcher(
                    pickup_where_leftoff=pickup_where_leftoff,
                    verbose=verbose,
                    compiled_TRITONSWMM_directory=compiled_TRITONSWMM_directory,
                    analysis_dir=analysis_dir,
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
        Docstring for run_simulations_concurrently

        :param self: automatically chooses whether to use SLURM or ThreadPoolExecutor for concurrent runs
        """
        multi_sim_run_method = self.cfg_analysis.multi_sim_run_method
        using_srun = multi_sim_run_method == "1_job_many_srun_tasks"
        if using_srun:
            if verbose:
                print(f"running multiple sims using approach {multi_sim_run_method}")
            self.run_simulations_concurrently_on_SLURM_HPC_using_many_srun_tasks(
                launch_functions=launch_functions,
                verbose=verbose,
                max_concurrent=max_concurrent,
            )
        else:
            self.run_simulations_concurrently_on_local_machine(
                launch_functions=launch_functions,
                verbose=verbose,
                max_concurrent=max_concurrent,
            )

    def run_simulations_concurrently_on_SLURM_HPC_using_many_srun_tasks(
        self,
        launch_functions: list[tuple],
        max_concurrent: Optional[int] = None,
        verbose: bool = True,
    ) -> list[str]:
        """
        Launch simulations concurrently on an HPC system using SLURM.
        Uses a pool-based approach with process polling to limit concurrent srun tasks
        and avoid resource contention.

        This method honors all relevant SLURM environment variables to ensure
        concurrent simulations respect the job's resource allocation:

        **CPU-based constraints:**
        - SLURM_JOB_NUM_NODES: Number of nodes allocated
        - SLURM_CPUS_ON_NODE: CPUs per node
        - SLURM_CPUS_PER_TASK: CPUs per task (from job allocation)
        - SLURM_NTASKS: Total tasks allocated
        - SLURM_TASKS_PER_NODE: Task distribution across nodes

        **GPU-based constraints (if run_mode == "gpu"):**
        - SLURM_GPUS: Total GPUs allocated
        - SLURM_GPUS_ON_NODE: GPUs per node
        - SLURM_GPUS_PER_TASK: GPUs per task (from config)

        **Memory constraints:**
        - SLURM_MEM_PER_NODE: Memory per node
        - SLURM_MEM_PER_CPU: Memory per CPU

        Parameters
        ----------
        launch_functions : list of tuples
            Each tuple is (launcher, finalize_sim) from _create_subprocess_sim_run_launcher().
            launcher() starts the process (non-blocking), finalize_sim() waits and updates logs.
        max_concurrent : int | None
            Maximum number of concurrent tasks. If None, automatically
            calculated from SLURM environment variables and job configuration.
        verbose : bool
            If True, prints detailed resource constraint information.

        Returns
        -------
        list[str]
            List of simulation statuses, in completion order.

        Raises
        ------
        RuntimeError
            If GPU mode is requested but no GPUs are detected, or if
            simulation resource requirements exceed job allocation.
        """
        # ----------------------------
        # Get SLURM resource constraints
        # ----------------------------
        constraints = self._get_slurm_resource_constraints(verbose=verbose)
        num_nodes = constraints["num_nodes"]
        total_cpus = constraints["total_cpus"]
        total_gpus = constraints["total_gpus"]
        if max_concurrent is None:
            max_concurrent = int(constraints["max_concurrent"])

        # ----------------------------
        # Validate simulation requirements
        # ----------------------------
        n_nodes_per_sim = self.cfg_analysis.n_nodes or 1
        mpi_ranks = self.cfg_analysis.n_mpi_procs or 1
        omp_threads = self.cfg_analysis.n_omp_threads or 1
        cpus_per_sim = mpi_ranks * omp_threads
        gpus_per_sim = self.cfg_analysis.n_gpus or 0

        if n_nodes_per_sim > num_nodes:  # type: ignore
            raise RuntimeError(
                f"Each simulation requires {n_nodes_per_sim} node(s), "
                f"but job only has {num_nodes}."  # type: ignore
            )

        if cpus_per_sim > total_cpus:
            raise RuntimeError(
                f"Each simulation requires {cpus_per_sim} CPUs, "
                f"but job only has {total_cpus}."
            )

        if self.cfg_analysis.run_mode == "gpu":
            if total_gpus == 0:
                raise RuntimeError(
                    "GPU run mode requested, but no GPUs detected via SLURM. "
                    "Check SLURM_GPUS or SLURM_GPUS_ON_NODE environment variables."
                )
            if gpus_per_sim > total_gpus:
                raise RuntimeError(
                    f"Each simulation requires {gpus_per_sim} GPU(s), "
                    f"but only {total_gpus} GPU(s) allocated to the job."
                )

        if verbose:
            print(
                f"[SLURM] Running {len(launch_functions)} simulations "
                f"(max {max_concurrent} concurrent tasks)",
                flush=True,
            )

        # ----------------------------
        # Process polling-based concurrent execution
        # ----------------------------
        results: list[str] = []
        running_processes = {}  # {proc: (finalize_sim, start_time, sim_logfile, lf)}
        pending_launchers = list(
            launch_functions
        )  # Queue of (launcher, finalize_sim) tuples

        # Launch initial batch up to max_concurrent
        while len(running_processes) < max_concurrent and pending_launchers:
            launcher, finalize_sim = pending_launchers.pop(0)
            proc, start_time, sim_logfile, lf = launcher()
            running_processes[proc] = (finalize_sim, start_time, sim_logfile, lf)
            if verbose:
                print(
                    f"[SLURM] Launched simulation ({len(running_processes)} running, "
                    f"{len(pending_launchers)} pending)",
                    flush=True,
                )

        # Poll and manage running processes
        while running_processes or pending_launchers:
            # Check which processes have completed
            completed_procs = []
            for proc in list(running_processes.keys()):
                if proc.poll() is not None:  # Process finished
                    completed_procs.append(proc)

            # Finalize completed processes
            for proc in completed_procs:
                finalize_sim, start_time, sim_logfile, lf = running_processes.pop(proc)
                finalize_sim(proc, start_time, sim_logfile, lf)
                results.append("completed")

                if verbose:
                    print(
                        f"[SLURM] Simulation completed ({len(running_processes)} running, "
                        f"{len(pending_launchers)} pending)",
                        flush=True,
                    )

                # Launch next pending simulation
                if pending_launchers:
                    launcher, finalize_sim = pending_launchers.pop(0)
                    proc, start_time, sim_logfile, lf = launcher()
                    running_processes[proc] = (
                        finalize_sim,
                        start_time,
                        sim_logfile,
                        lf,
                    )
                    if verbose:
                        print(
                            f"[SLURM] Launched simulation ({len(running_processes)} running, "
                            f"{len(pending_launchers)} pending)",
                            flush=True,
                        )

            # Small sleep to avoid busy-waiting
            if running_processes:
                time.sleep(0.1)

        self._update_log()
        return results

    def run_simulations_concurrently_on_local_machine(
        self,
        launch_functions: List[tuple],
        max_concurrent: Optional[int] = None,
        min_memory_per_sim_MiB: int | None = 1024,
        verbose: bool = True,
    ):
        """
        Run simulations concurrently on a desktop/local machine.

        The new launcher pattern is non-blocking - each launcher starts a process
        and returns immediately, while finalize_sim waits for completion.
        This method uses ThreadPoolExecutor to run multiple simulations concurrently.

        Parameters
        ----------
        launch_functions : List[tuple]
            List of tuples (launcher, finalize_sim) from _create_subprocess_sim_run_launcher().
            launcher() starts the process (non-blocking), finalize_sim() waits and updates logs.
        max_concurrent : Optional[int]
            Maximum number of concurrent simulations
        min_memory_per_sim_MiB : int | None
            Minimum memory required per simulation
        verbose : bool
            If True, print progress messages
        """
        use_gpu = self.cfg_analysis.run_mode == "gpu"
        # ----------------------------
        # Determine parallelism
        # ----------------------------
        if max_concurrent is None:
            # ----------------------------
            # Determine GPU parallelism (TODO)
            # ----------------------------
            if use_gpu:
                raise ValueError(
                    "Currently desktop-based simulations are not designed to use GPUs. Feature must be built out."
                )

            # ----------------------------
            # Calculate effective max parallel with all constraints
            # ----------------------------
            max_concurrent = self.calculate_effective_max_parallel(
                min_memory_per_function_MiB=min_memory_per_sim_MiB,
                max_concurrent=max_concurrent,
                verbose=verbose,
            )

        if verbose:
            print(
                f"Running up to {max_concurrent} simulations concurrently", flush=True
            )

        # ----------------------------
        # Execute launchers with ThreadPoolExecutor
        # ----------------------------
        results = []

        def execute_sim(launcher, finalize_sim):
            """Execute a single simulation: launch and finalize."""
            proc, start_time, sim_logfile, lf = launcher()
            finalize_sim(proc, start_time, sim_logfile, lf)
            return "completed"

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(execute_sim, launcher, finalize_sim)
                for launcher, finalize_sim in launch_functions
            ]

            for idx, future in enumerate(as_completed(futures)):
                if verbose:
                    print(
                        f"Simulation {idx + 1}/{len(launch_functions)} completed",
                        flush=True,
                    )
                try:
                    results.append(future.result())
                except Exception as e:
                    if verbose:
                        print(f"Simulation failed with error: {e}", flush=True)
                    results.append("failed")

        self._update_log()
        return results

    def run_sims_in_sequence(
        self,
        pickup_where_leftoff,
        process_outputs_after_sim_completion: bool = False,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        verbose=False,
        analysis_dir: Optional[Path] = None,
        compiled_TRITONSWMM_directory: Optional[Path] = None,
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
                analysis_dir=analysis_dir,
                compiled_TRITONSWMM_directory=compiled_TRITONSWMM_directory,
            )
        self._update_log()

    def generate_setup_workflow_script(
        self,
        job_script_path: Optional[Path] = None,
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        verbose: bool = True,
    ) -> Path:
        """
        Generate a SLURM script for Phase 1: Setup (process system inputs and compile).

        Parameters
        ----------
        job_script_path : Optional[Path]
            Path where the job script will be saved. If None, defaults to
            analysis_dir/run_setup_workflow.sh
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings)
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        verbose : bool
            If True, print progress messages

        Returns
        -------
        Path
            Path to the generated job script
        """
        if job_script_path is None:
            job_script_path = self.analysis_paths.analysis_dir / "run_setup_workflow.sh"
        else:
            job_script_path = Path(job_script_path)

        job_script_path.parent.mkdir(parents=True, exist_ok=True)

        from TRITON_SWMM_toolkit.utils import current_datetime_string

        logdir = self.analysis_paths.analysis_dir / "slurm_logs"
        logdir.mkdir(exist_ok=True, parents=True)
        dtime = current_datetime_string(filepath_friendly=True)
        logdir_job = logdir / dtime
        logdir_job.mkdir(parents=True, exist_ok=True)

        hpc_partition = self.cfg_analysis.hpc_partition
        hpc_allocation = self.cfg_analysis.hpc_allocation

        sbatch_lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=TRITON_SWMM_setup",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --time=00:30:00",
            f"#SBATCH --partition={hpc_partition}",
            f"#SBATCH --account={hpc_allocation}",
            f"#SBATCH --output={logdir_job}/setup_%j.out",
            f"#SBATCH --error={logdir_job}/setup_%j.out",
            "",
        ]

        modules = (
            self._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
        )
        if modules:
            sbatch_lines.append(f"module load {modules}")
            sbatch_lines.append("conda activate triton_swmm_toolkit")
            sbatch_lines.append("")

        setup_cmd_parts = [
            "python -m TRITON_SWMM_toolkit.setup_workflow \\",
            f"    --system-config {self._system.system_config_yaml} \\",
            f"    --analysis-config {self.analysis_config_yaml} \\",
        ]

        if process_system_level_inputs:
            setup_cmd_parts.append("    --process-system-inputs \\")
            if overwrite_system_inputs:
                setup_cmd_parts.append("    --overwrite-system-inputs \\")

        if compile_TRITON_SWMM:
            setup_cmd_parts.append("    --compile-triton-swmm \\")
            if recompile_if_already_done_successfully:
                setup_cmd_parts.append("    --recompile-if-already-done \\")

        setup_cmd_parts[-1] = setup_cmd_parts[-1].rstrip(" \\")
        sbatch_lines.extend(setup_cmd_parts)

        script_content = "\n".join(sbatch_lines)
        job_script_path.write_text(script_content)
        job_script_path.chmod(0o755)

        if verbose:
            print(f"Setup workflow script generated: {job_script_path}", flush=True)

        return job_script_path

    def generate_ensemble_simulations_script(
        self,
        job_script_path: Optional[Path] = None,
        prepare_scenarios: bool = True,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = False,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = True,
    ) -> Path:
        """
        Generate a SLURM script for Phase 2: Ensemble simulations (job array).

        Parameters
        ----------
        job_script_path : Optional[Path]
            Path where the job script will be saved. If None, defaults to
            analysis_dir/run_ensemble_simulations.sh
        prepare_scenarios : bool
            If True, each array task will prepare its scenario before running
        process_timeseries : bool
            If True, each array task will process timeseries outputs after simulation
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
        overwrite_scenario : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        verbose : bool
            If True, print progress messages

        Returns
        -------
        Path
            Path to the generated job script
        """
        if job_script_path is None:
            job_script_path = (
                self.analysis_paths.analysis_dir / "run_ensemble_simulations.sh"
            )
        else:
            job_script_path = Path(job_script_path)

        job_script_path.parent.mkdir(parents=True, exist_ok=True)

        analysis_id = self.cfg_analysis.analysis_id
        run_mode = self.cfg_analysis.run_mode
        n_mpi_procs = self.cfg_analysis.n_mpi_procs or 1
        n_omp_threads = self.cfg_analysis.n_omp_threads or 1
        n_gpus = self.cfg_analysis.n_gpus or 0
        n_nodes = self.cfg_analysis.n_nodes or 1
        hpc_time_min = self.cfg_analysis.hpc_time_min_per_sim
        hpc_partition = self.cfg_analysis.hpc_partition
        hpc_allocation = self.cfg_analysis.hpc_allocation

        if hpc_time_min is None:
            raise ValueError("sim run time not specified in analysis config file")

        n_sims = len(self.df_sims)
        array_range = f"0-{n_sims - 1}"

        from TRITON_SWMM_toolkit.utils import current_datetime_string

        logdir = self.analysis_paths.analysis_dir / "slurm_logs"
        logdir.mkdir(exist_ok=True, parents=True)
        dtime = current_datetime_string(filepath_friendly=True)
        logdir_job = logdir / dtime
        logdir_job.mkdir(parents=True, exist_ok=True)

        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=TRITON_SWMM_{analysis_id}",
            f"#SBATCH --array={array_range}",
            f"#SBATCH --nodes={n_nodes}",
            f"#SBATCH --ntasks={n_mpi_procs}",
            f"#SBATCH --cpus-per-task={n_omp_threads}",
            f"#SBATCH --time={minutes_to_hhmmss(hpc_time_min)}",
            f"#SBATCH --partition={hpc_partition}",
            f"#SBATCH --account={hpc_allocation}",
        ]

        if run_mode == "gpu":
            sbatch_lines.append(f"#SBATCH --gres=gpu:{n_gpus}")

        sbatch_lines.extend(
            [
                f"#SBATCH --output={logdir_job}/sim_%A_%a.out",
                f"#SBATCH --error={logdir_job}/sim_%A_%a.out",
                "",
            ]
        )

        modules = (
            self._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
        )
        if modules:
            sbatch_lines.append(f"module load {modules}")
            sbatch_lines.append("")

        cmd_parts = [
            "python -m TRITON_SWMM_toolkit.run_single_simulation \\",
            "    --event-iloc ${SLURM_ARRAY_TASK_ID} \\",
            f"    --system-config {self._system.system_config_yaml} \\",
            f"    --analysis-config {self.analysis_config_yaml} \\",
        ]

        if prepare_scenarios:
            cmd_parts.append("    --prepare-scenario \\")
            if overwrite_scenario:
                cmd_parts.append("    --overwrite-scenario \\")
            if rerun_swmm_hydro_if_outputs_exist:
                cmd_parts.append("    --rerun-swmm-hydro \\")

        if process_timeseries:
            cmd_parts.append("    --process-timeseries \\")
            cmd_parts.append(f"    --which {which} \\")
            if clear_raw_outputs:
                cmd_parts.append("    --clear-raw-outputs \\")
            if overwrite_if_exist:
                cmd_parts.append("    --overwrite-if-exist \\")
            cmd_parts.append(f"    --compression-level {compression_level} \\")

        if pickup_where_leftoff:
            cmd_parts.append("    --pickup-where-leftoff \\")

        cmd_parts[-1] = cmd_parts[-1].rstrip(" \\")
        sbatch_lines.extend(cmd_parts)

        script_content = "\n".join(sbatch_lines)
        job_script_path.write_text(script_content)
        job_script_path.chmod(0o755)

        if verbose:
            print(
                f"Ensemble simulations script generated: {job_script_path}", flush=True
            )

        return job_script_path

    def generate_consolidation_workflow_script(
        self,
        job_script_path: Optional[Path] = None,
        consolidate_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        verbose: bool = True,
    ) -> Path:
        """
        Generate a SLURM script for Phase 3: Consolidation.

        Parameters
        ----------
        job_script_path : Optional[Path]
            Path where the job script will be saved. If None, defaults to
            analysis_dir/run_consolidation_workflow.sh
        consolidate_outputs : bool
            If True, consolidate TRITON and SWMM simulation summaries
        overwrite_if_exist : bool
            If True, overwrite existing consolidated outputs
        compression_level : int
            Compression level for output files (0-9)
        verbose : bool
            If True, print progress messages

        Returns
        -------
        Path
            Path to the generated job script
        """
        if job_script_path is None:
            job_script_path = (
                self.analysis_paths.analysis_dir / "run_consolidation_workflow.sh"
            )
        else:
            job_script_path = Path(job_script_path)

        job_script_path.parent.mkdir(parents=True, exist_ok=True)

        from TRITON_SWMM_toolkit.utils import current_datetime_string

        logdir = self.analysis_paths.analysis_dir / "slurm_logs"
        logdir.mkdir(exist_ok=True, parents=True)
        dtime = current_datetime_string(filepath_friendly=True)
        logdir_job = logdir / dtime
        logdir_job.mkdir(parents=True, exist_ok=True)

        hpc_partition = self.cfg_analysis.hpc_partition
        hpc_allocation = self.cfg_analysis.hpc_allocation

        sbatch_lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=TRITON_SWMM_consolidate",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=1",
            "#SBATCH --time=00:30:00",
            f"#SBATCH --partition={hpc_partition}",
            f"#SBATCH --account={hpc_allocation}",
            f"#SBATCH --output={logdir_job}/consolidate_%j.out",
            f"#SBATCH --error={logdir_job}/consolidate_%j.out",
            "",
        ]

        modules = (
            self._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
        )
        if modules:
            sbatch_lines.append(f"module load {modules}")
            sbatch_lines.append("conda activate triton_swmm_toolkit")
            sbatch_lines.append("")

        consolidate_cmd_parts = [
            "python -m TRITON_SWMM_toolkit.consolidate_workflow \\",
            f"    --system-config {self._system.system_config_yaml} \\",
            f"    --analysis-config {self.analysis_config_yaml} \\",
        ]

        if consolidate_outputs:
            consolidate_cmd_parts.append("    --consolidate-outputs \\")
            consolidate_cmd_parts.append(
                f"    --compression-level {compression_level} \\"
            )

        consolidate_cmd_parts[-1] = consolidate_cmd_parts[-1].rstrip(" \\")
        sbatch_lines.extend(consolidate_cmd_parts)

        script_content = "\n".join(sbatch_lines)
        job_script_path.write_text(script_content)
        job_script_path.chmod(0o755)

        if verbose:
            print(
                f"Consolidation workflow script generated: {job_script_path}",
                flush=True,
            )

        return job_script_path

    def generate_SLURM_job_array_script(
        self,
        job_script_path: Optional[Path] = None,
        prepare_scenarios: bool = True,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = True,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = True,
    ) -> Path:
        """
        Generate a SLURM job array script for running all simulations.

        Each array task runs a single simulation identified by SLURM_ARRAY_TASK_ID,
        which maps directly to event_iloc. Logs are saved to each simulation's folder.

        Parameters
        ----------
        job_script_path : Optional[Path]
            Path where the job script will be saved. If None, defaults to
            analysis_dir/run_job_array.sh
        prepare_scenarios : bool
            If True, each task will prepare its scenario before running
        process_timeseries : bool
            If True, each task will process timeseries outputs after simulation
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
        overwrite_scenario : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        verbose : bool
            If True, print progress messages

        Returns
        -------
        Path
            Path to the generated job script

        Examples
        --------
        >>> analysis = TRITONSWMM_analysis(...)
        >>> script_path = analysis.generate_SLURM_job_array_script(
        ...     prepare_scenarios=True,
        ...     process_timeseries=True,
        ... )
        >>> print(f"Job script generated at: {script_path}")
        >>> # User can then submit with: sbatch {script_path}
        """
        # Default job script path
        if job_script_path is None:
            job_script_path = self.analysis_paths.analysis_dir / "run_job_array.sh"
        else:
            job_script_path = Path(job_script_path)

        # Ensure parent directory exists
        job_script_path.parent.mkdir(parents=True, exist_ok=True)

        # Get configuration parameters
        analysis_id = self.cfg_analysis.analysis_id
        run_mode = self.cfg_analysis.run_mode
        n_mpi_procs = self.cfg_analysis.n_mpi_procs or 1
        n_omp_threads = self.cfg_analysis.n_omp_threads or 1
        n_gpus = self.cfg_analysis.n_gpus or 0
        n_nodes = self.cfg_analysis.n_nodes or 1

        # Get HPC parameters (if available in config)
        hpc_time_min = self.cfg_analysis.hpc_time_min_per_sim
        hpc_partition = self.cfg_analysis.hpc_partition
        hpc_allocation = self.cfg_analysis.hpc_allocation

        if hpc_time_min is None:
            raise ValueError("sim run time not specified in analysis config file")

        # Calculate array range
        n_sims = len(self.df_sims)
        array_range = f"0-{n_sims - 1}"

        # Build SBATCH directives
        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name=TRITON_SWMM_{analysis_id}",
            f"#SBATCH --array={array_range}",
            f"#SBATCH --nodes={n_nodes}",
            f"#SBATCH --ntasks={n_mpi_procs}",
            f"#SBATCH --cpus-per-task={n_omp_threads}",
            f"#SBATCH --time={minutes_to_hhmmss(hpc_time_min)}",
            f"#SBATCH --partition={hpc_partition}",
            f"#SBATCH --account={hpc_allocation}",
        ]

        # Add GPU directive if in GPU mode
        if run_mode == "gpu":
            sbatch_lines.append(f"#SBATCH --gres=gpu:{n_gpus}")

        from TRITON_SWMM_toolkit.utils import current_datetime_string

        logdir = self.analysis_paths.analysis_dir / "slurm_logs"
        # archive outputs from previous runs
        archive_dir = logdir / "_archive"
        archive_dir.mkdir(exist_ok=True, parents=True)
        for folder in logdir.iterdir():
            if folder.is_dir() and folder.name != "_archive":
                shutil.move(str(folder), archive_dir / folder.name)

        dtime = current_datetime_string(filepath_friendly=True)
        logdir_job = logdir / dtime
        logdir_job.mkdir(parents=True, exist_ok=True)

        # Add output/error log directives
        sbatch_lines.extend(
            [
                f"#SBATCH --output={logdir_job}/sim_%A_%a.out",
                f"#SBATCH --error={logdir_job}/sim_%A_%a.out",
                "",
            ]
        )

        # Add module loading if needed
        modules = (
            self._system.cfg_system.additional_modules_needed_to_run_TRITON_SWMM_on_hpc
        )
        if modules:
            sbatch_lines.append(f"module load {modules}")
            sbatch_lines.append("")

        # Add environment setup
        sbatch_lines.extend(
            [
                "# Determine simulation folder for this task",
                "SIM_FOLDER=$(python -m TRITON_SWMM_toolkit.get_sim_folder \\",
                "    --event-iloc ${SLURM_ARRAY_TASK_ID} \\",
                f"    --analysis-config {self.analysis_config_yaml} \\",
                f"    --system-config {self._system.system_config_yaml})",
                "",
                # "# Redirect output to simulation-specific log",
                # 'LOG_FILE="${SIM_FOLDER}/slurm_job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"',
                "",
            ]
        )

        # Build the run_single_simulation command
        cmd_parts = [
            "python -m TRITON_SWMM_toolkit.run_single_simulation \\",
            "    --event-iloc ${SLURM_ARRAY_TASK_ID} \\",
            f"    --system-config {self._system.system_config_yaml} \\",
            f"    --analysis-config {self.analysis_config_yaml} \\",
        ]

        if prepare_scenarios:
            cmd_parts.append("    --prepare-scenario \\")
            if overwrite_scenario:
                cmd_parts.append("    --overwrite-scenario \\")
            if rerun_swmm_hydro_if_outputs_exist:
                cmd_parts.append("    --rerun-swmm-hydro \\")

        if process_timeseries:
            cmd_parts.append("    --process-timeseries \\")
            cmd_parts.append(f"    --which {which} \\")
            if clear_raw_outputs:
                cmd_parts.append("    --clear-raw-outputs \\")
            if overwrite_if_exist:
                cmd_parts.append("    --overwrite-if-exist \\")
            cmd_parts.append(f"    --compression-level {compression_level} \\")

        if pickup_where_leftoff:
            cmd_parts.append("    --pickup-where-leftoff \\")

        # Remove trailing backslash from last line and add log redirection
        cmd_parts[-1] = cmd_parts[-1].rstrip(" \\")
        # cmd_parts.append('    > "${LOG_FILE}" 2>&1')

        sbatch_lines.extend(cmd_parts)

        # Write the script
        script_content = "\n".join(sbatch_lines)
        job_script_path.write_text(script_content)
        job_script_path.chmod(0o755)

        if verbose:
            print(f"SLURM job array script generated: {job_script_path}", flush=True)
            print(f"Array range: {array_range}", flush=True)
            print(f"To submit: sbatch {job_script_path}", flush=True)

        return job_script_path

    def submit_SLURM_job_array(
        self,
        job_script_path: Optional[Path] = None,
        process_system_level_inputs: bool = False,
        overwrite_system_inputs: bool = False,
        compile_TRITON_SWMM: bool = True,
        recompile_if_already_done_successfully: bool = False,
        prepare_scenarios: bool = True,
        process_timeseries: bool = True,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        compression_level: int = 5,
        pickup_where_leftoff: bool = False,
        overwrite_scenario: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        consolidate_outputs: bool = True,
        verbose: bool = True,
    ) -> tuple[Path, str]:
        """
        Generate and submit a consolidated SLURM workflow script.

        This method creates a heterogeneous SLURM job with three phases:
        1. Setup: Process system inputs and compile TRITON-SWMM
        2. Ensemble: Run all simulations in parallel using job array
        3. Consolidation: Consolidate outputs after all simulations complete

        Each phase verifies the success of the previous phase before proceeding.

        Parameters
        ----------
        job_script_path : Optional[Path]
            Path where the job script will be saved. If None, defaults to
            analysis_dir/run_consolidated_workflow.sh
        process_system_level_inputs : bool
            If True, process system-level inputs (DEM, Mannings) in Phase 1
        overwrite_system_inputs : bool
            If True, overwrite existing system input files
        compile_TRITON_SWMM : bool
            If True, compile TRITON-SWMM in Phase 1
        recompile_if_already_done_successfully : bool
            If True, recompile even if already compiled successfully
        prepare_scenarios : bool
            If True, each array task will prepare its scenario before running
        process_timeseries : bool
            If True, each array task will process timeseries outputs after simulation
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
        overwrite_scenario : bool
            If True, overwrite existing scenarios
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        consolidate_outputs : bool
            If True, consolidate outputs in Phase 3 after all simulations complete
        verbose : bool
            If True, print progress messages

        Returns
        -------
        tuple[Path, str]
            (job_script_path, job_id) where job_id is the SLURM job ID

        Raises
        ------
        RuntimeError
            If sbatch submission fails

        Examples
        --------
        >>> analysis = TRITONSWMM_analysis(...)
        >>> script_path, job_id = analysis.submit_SLURM_job_array(
        ...     process_system_level_inputs=True,
        ...     compile_TRITON_SWMM=True,
        ...     consolidate_outputs=True,
        ... )
        >>> print(f"Job submitted with ID: {job_id}")
        >>> print(f"Monitor with: squeue -j {job_id}")
        """
        # Generate the three phase scripts
        setup_script = self.generate_setup_workflow_script(
            process_system_level_inputs=process_system_level_inputs,
            overwrite_system_inputs=overwrite_system_inputs,
            compile_TRITON_SWMM=compile_TRITON_SWMM,
            recompile_if_already_done_successfully=recompile_if_already_done_successfully,
            verbose=verbose,
        )

        ensemble_script = self.generate_ensemble_simulations_script(
            prepare_scenarios=prepare_scenarios,
            process_timeseries=process_timeseries,
            which=which,
            clear_raw_outputs=clear_raw_outputs,
            overwrite_if_exist=overwrite_if_exist,
            compression_level=compression_level,
            pickup_where_leftoff=pickup_where_leftoff,
            overwrite_scenario=overwrite_scenario,
            rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            verbose=verbose,
        )

        consolidation_script = self.generate_consolidation_workflow_script(
            consolidate_outputs=consolidate_outputs,
            overwrite_if_exist=overwrite_if_exist,
            compression_level=compression_level,
            verbose=verbose,
        )

        # Submit the three scripts with dependencies
        if verbose:
            print(
                "Submitting three-phase workflow with SLURM dependencies...",
                flush=True,
            )

        try:
            # Phase 1: Setup
            result1 = subprocess.run(
                ["sbatch", str(setup_script)],
                capture_output=True,
                text=True,
                check=True,
            )
            job_id_1 = result1.stdout.strip().split()[-1]
            if verbose:
                print(f"Phase 1 (Setup) submitted with job ID: {job_id_1}", flush=True)

            # Phase 2: Ensemble (depends on Phase 1)
            result2 = subprocess.run(
                ["sbatch", f"--dependency=afterok:{job_id_1}", str(ensemble_script)],
                capture_output=True,
                text=True,
                check=True,
            )
            job_id_2 = result2.stdout.strip().split()[-1]
            if verbose:
                print(
                    f"Phase 2 (Ensemble) submitted with job ID: {job_id_2} "
                    f"(depends on {job_id_1})",
                    flush=True,
                )

            # Phase 3: Consolidation (depends on Phase 2)
            result3 = subprocess.run(
                [
                    "sbatch",
                    f"--dependency=afterok:{job_id_2}",
                    str(consolidation_script),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            job_id_3 = result3.stdout.strip().split()[-1]
            if verbose:
                print(
                    f"Phase 3 (Consolidation) submitted with job ID: {job_id_3} "
                    f"(depends on {job_id_2})",
                    flush=True,
                )
                print(
                    f"Complete workflow submitted! Monitor with: squeue -j {job_id_1},{job_id_2},{job_id_3}",
                    flush=True,
                )

            # Return the ensemble script path and the final job ID
            return ensemble_script, job_id_3

        except subprocess.CalledProcessError as e:
            error_msg = f"sbatch submission failed: {e.stderr}"
            if verbose:
                print(error_msg, flush=True)
            raise RuntimeError(error_msg)

    def compile_TRITON_SWMM(
        self,
        recompile_if_already_done_successfully: bool = False,
        verbose: bool = False,
    ):
        self.analysis_paths.compiled_TRITONSWMM_directory.mkdir(
            parents=True, exist_ok=True
        )

        if self.compilation_successful and not recompile_if_already_done_successfully:
            print("TRITON-SWMM already compiled", flush=True)
            return
        # TODO ADD TOGGLE TO ONLY DO THIS IF NOT ALREADY COMPILED
        compiled_TRITONSWMM_directory = (
            self.analysis_paths.compiled_TRITONSWMM_directory
        )
        compilation_script = self.analysis_paths.compilation_script
        TRITONSWMM_software_directory = (
            self._system.cfg_system.TRITONSWMM_software_directory
        )
        TRITON_SWMM_make_command = self.cfg_analysis.TRITON_SWMM_make_command
        TRITON_SWMM_software_compilation_script = (
            self._system.cfg_system.TRITON_SWMM_software_compilation_script
        )
        if compiled_TRITONSWMM_directory.exists():
            shutil.rmtree(compiled_TRITONSWMM_directory)
        shutil.copytree(TRITONSWMM_software_directory, compiled_TRITONSWMM_directory)
        mapping = dict(
            COMPILED_MODEL_DIR=compiled_TRITONSWMM_directory,
            MAKE_COMMAND=TRITON_SWMM_make_command,
        )
        comp_script_content = create_from_template(
            TRITON_SWMM_software_compilation_script,
            mapping,
            compilation_script,
        )
        compilation_logfile = self.analysis_paths.compilation_logfile

        with open(compilation_logfile, "w") as logfile:
            proc = subprocess.run(  # type: ignore
                ["/bin/bash", str(compilation_script)],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                check=True,
            )

        import time

        start_time = time.time()
        compilation_log = read_text_file_as_string(compilation_logfile)
        while "Building finished: triton" not in compilation_log:
            time.sleep(0.1)
            compilation_log = read_text_file_as_string(compilation_logfile)
            elapsed = time.time() - start_time
            time.sleep(0.1)
            if elapsed > 5:
                break
        self.compilation_log = compilation_log
        success = self.compilation_successful
        self.log.TRITONSWMM_compiled_successfully.set(success)
        if not success:
            if verbose:
                print(
                    "warning: TRITON-SWMM did not compile successfully.\
    You can load compilation log as string using\
    retrieve_compilation_log or print it to the\
    terminal using the method print_compilation_log",
                    flush=True,
                )
        return

    def retrieve_compilation_log(self):
        if self.analysis_paths.compilation_logfile.exists():
            return read_text_file_as_string(self.analysis_paths.compilation_logfile)
        return "no sim logfile created"

    def print_compilation_log(self):
        print(self.retrieve_compilation_log(), flush=True)

    def _validate_compilation(self):
        compilation_log = self.retrieve_compilation_log()
        swmm_check = "[100%] Built target runswmm" in compilation_log
        triton_check = "Building finished: triton" in compilation_log
        success = swmm_check and triton_check
        return success

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
    def TRITON_summary(self):
        return self.process.TRITON_summary


# %%
def minutes_to_hhmmss(minutes: int) -> str:
    secs = 0
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
