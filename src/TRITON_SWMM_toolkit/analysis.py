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
        compiled_software_directory: Optional[Path] = None,
    ) -> None:
        self._system = system
        self.analysis_config_yaml = analysis_config_yaml
        cfg_analysis = load_analysis_config(analysis_config_yaml)
        self.cfg_analysis = cfg_analysis
        # define additional paths not defined in cfg
        if compiled_software_directory is None:
            compiled_software_directory = (
                self._system.cfg_system.system_directory
                / self.cfg_analysis.analysis_id
                / "compiled_software"
            )
            compiled_software_directory.mkdir(parents=True, exist_ok=True)
        if analysis_dir is None:
            analysis_dir = (
                self._system.cfg_system.system_directory / self.cfg_analysis.analysis_id
            )

        self.analysis_paths = AnalysisPaths(
            f_log=analysis_dir / "log.json",
            analysis_dir=analysis_dir,
            compiled_software_directory=compiled_software_directory,
            TRITON_build_dir=compiled_software_directory / "build",
            compilation_script=compiled_software_directory / "compile.sh",
            simulation_directory=analysis_dir / "sims",
            compilation_logfile=compiled_software_directory / f"compilation.log",
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
        self.scenarios = {}
        self._sim_run_objects = {}
        self._sim_run_processing_objects = {}
        self._simulation_run_statuses = {}
        self.run_modes = Mode
        # self.compilation_successful = False
        self._refresh_log()
        self.in_slurm = "SLURM_JOB_ID" in os.environ.copy()

        if self.analysis_paths.compilation_logfile.exists():
            self._validate_compilation()
        self._add_all_scenarios()
        self.process = TRITONSWMM_analysis_post_processing(self)
        self.plot = TRITONSWMM_analysis_plotting(self)
        if self.cfg_analysis.toggle_sensitivity_analysis == True:
            self.sensitivity = TRITONSWMM_sensitivity_analysis(self)

        cfg_anlysys_yaml = analysis_dir / f"analysis.yaml"
        cfg_anlysys_yaml.write_text(
            yaml.safe_dump(
                self.cfg_analysis.model_dump(mode="json"),
                sort_keys=False,  # .dict() for pydantic v1
            )
        )
        self._update_log()

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
    ):
        self.consolidate_TRITON_simulation_summaries(
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
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
            print("=== System Configuration ===")
            self._system.cfg_system.display_tabulate_cfg()
        if which == "both":
            print("\n")
        if which in ["analysis", "both"]:
            print("=== analysis Configuration ===")
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
        dic_sim_paths = self.scenarios[event_iloc].scen_paths.as_dict()
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

    def _add_scenario(self, event_iloc: int):
        scen = TRITONSWMM_scenario(event_iloc, self)
        self.scenarios[event_iloc] = scen
        return scen

    def _add_all_scenarios(self):
        for event_iloc in self.df_sims.index:
            self._add_scenario(event_iloc)
        return

    @property
    def scenarios_not_created(self):
        scens_not_created = []
        for event_iloc in self.df_sims.index:
            scen = self.scenarios[event_iloc]
            scen.log.refresh()
            if scen.log.scenario_creation_complete.get() != True:
                scens_not_created.append(str(scen.log.logfile.parent))
        return scens_not_created

    @property
    def scenarios_not_run(self):
        scens_not_run = []
        for event_iloc in self.df_sims.index:
            scen = self.scenarios[event_iloc]
            if scen.sim_run_completed != True:
                scens_not_run.append(str(scen.log.logfile.parent))
        return scens_not_run

    def _update_log(self):
        # dict_all_logs = {}
        all_scens_created = True
        all_sims_run = True
        all_TRITON_outputs_processed = True
        all_SWMM_outputs_processed = True
        all_raw_TRITON_outputs_cleared = True
        all_raw_SWMM_outputs_cleared = True
        for event_iloc in self.df_sims.index:
            scen = self.scenarios[event_iloc]
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
    ):
        prepare_scenario_launchers = []
        for event_iloc in self.df_sims.index:
            scenario = self.scenarios[event_iloc]
            launcher = scenario.prepare_scenario(
                overwrite_scenario=overwrite_scenario,
                rerun_swmm_hydro_if_outputs_exist=rerun_swmm_hydro_if_outputs_exist,
            )
            prepare_scenario_launchers.append(launcher)

        return prepare_scenario_launchers

    def calculate_effective_max_parallel(
        self,
        min_memory_per_function_MiB: int | None = None,
        max_parallel_cpu: int | None = None,
        max_parallel_gpu: int | None = None,
        verbose: bool = False,
    ) -> int:
        """
        Calculate the effective maximum parallelism based on CPU, GPU, and memory constraints.

        Parameters
        ----------
        min_memory_per_function_MiB : int | None
            Minimum memory required per function (MiB).
            If provided, concurrency is reduced to avoid oversubscription.
        max_parallel_cpu : int | None
            CPU-based upper bound on parallelism (e.g., based on cores/threads per task).
            If None, defaults to physical CPU count - 1.
        max_parallel_gpu : int | None
            GPU-based upper bound on parallelism (e.g., total GPUs / GPUs per task).
            If None, GPU constraints are not applied.
        verbose : bool
            Print progress messages.

        Returns
        -------
        int
            The effective maximum number of parallel tasks.
        """
        # ----------------------------
        # CPU-based limit
        # ----------------------------
        if max_parallel_cpu is None:
            physical_cores = psutil.cpu_count(logical=False)
            if isinstance(physical_cores, int) and physical_cores > 1:
                physical_cores -= 1  # more conservative process count
            max_parallel_cpu = physical_cores or 1

        # ----------------------------
        # Memory-based limit
        # ----------------------------
        mem_limit = max_parallel_cpu
        if min_memory_per_function_MiB is not None:
            available_mem_MiB = psutil.virtual_memory().available // (1024**2)
            mem_limit = max(1, available_mem_MiB // min_memory_per_function_MiB)

            if verbose:
                print(
                    f"Memory-based limit: {mem_limit} "
                    f"(available {available_mem_MiB} MiB, "
                    f"{min_memory_per_function_MiB} MiB per task)"
                )

        # ----------------------------
        # Final concurrency (apply all constraints)
        # ----------------------------
        limits = [max_parallel_cpu, mem_limit]
        if max_parallel_gpu is not None:
            limits.append(max_parallel_gpu)

        effective_max_parallel = min(limits)

        return effective_max_parallel

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
            max_parallel_cpu=max_parallel,
            verbose=verbose,
        )

        if verbose:
            print(
                f"Running {len(function_launchers)} functions "
                f"(max parallel = {effective_max_parallel})"
            )

        results: List[int] = []

        def wrapper(idx: int, launcher: Callable[[], None]):
            start = time.time()
            launcher()
            elapsed = time.time() - start
            if verbose:
                print(f"Function {idx} completed in {elapsed:.2f} s")
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
    ):
        for event_iloc in self.df_sims.index:
            scen = self.scenarios[event_iloc]
            launcher = scen.prepare_scenario(
                overwrite_scenarios, rerun_swmm_hydro_if_outputs_exist
            )
            launcher()
            self._update_log()  # update logs
        return

    def print_logfile_for_scenario(self, event_iloc):
        scen = self.scenarios[event_iloc]
        scen.log.print()

    def retrieve_sim_command_text(
        self,
        pickup_where_leftoff: bool = False,
        in_slurm: Optional[bool] = None,
        verbose: bool = False,
        extra_env: Optional[dict] = None,
    ):
        sim_commands = []
        for event_iloc in self.df_sims.index:
            run = self._retreive_sim_runs(event_iloc)
            cmd, env, tritonswmm_logfile, sim_start_reporting_tstep = (  # type: ignore
                run.prepare_simulation_command(
                    pickup_where_leftoff=pickup_where_leftoff,
                    in_slurm=in_slurm,
                    verbose=verbose,
                )
            )
            env_lines = [f"export {k}={v}" for k, v in env.items()]  # type: ignore
            cmd_line = " ".join(cmd) + f" > {tritonswmm_logfile} 2>&1 &"  # type: ignore
            sim_commands.append(cmd_line)
        env_text = "\n".join(env_lines)
        command_text = env_text + "\n" * 2 + "\n".join(sim_commands) + "\n\nwait\n"
        return command_text

    # analysis functions
    # def create_ensemble_bash_script(self, run_command: str):
    #     """
    #     Generates one bash script that launches all simulations in the ensemble.
    #     """
    #     run_mode = self.cfg_analysis.run_mode
    #     if run_mode == "gpu":
    #         gpu_toggle = ""
    #     else:
    #         gpu_toggle = " "
    #     mapping = dict(
    #         allocation=self.cfg_analysis.hpc_allocation,
    #         time=minutes_to_hhmmss(self.cfg_analysis.hpc_time_min),  # type: ignore
    #         partition=self.cfg_analysis.hpc_partition,
    #         nodes=self.cfg_analysis.hpc_n_nodes,
    #         gpu_toggle=gpu_toggle,
    #         gres=self.cfg_analysis.hpc_gpus_requested,
    #         run_command=run_command,
    #     )

    #     create_from_template(
    #         self.cfg_analysis.hpc_bash_script_ensemble_template,  # type: ignore
    #         mapping,
    #         self.analysis_paths.bash_script_path,  # type: ignore
    #     )

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
        ts_scenario = self.scenarios[event_iloc]

        if not ts_scenario.log.scenario_creation_complete.get():
            print("Log file:")
            print(ts_scenario.log.print())
            raise ValueError("scenario_creation_complete must be 'success'")
        if not self.compilation_successful:
            print("Log file:")
            print(ts_scenario.log.print())
            raise ValueError("TRITONSWMM has not been compiled")
        run = self._retreive_sim_runs(event_iloc)
        if verbose:
            print("run instance instantiated")

        run.run_sim(pickup_where_leftoff=pickup_where_leftoff, verbose=verbose)
        self.sim_run_status(event_iloc)
        self._update_log()  # updates analysis log
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
    ):
        scenario_timeseries_processing_launchers = []
        for event_iloc in self.df_sims.index:
            proc = self._retrieve_sim_run_processing_object(event_iloc=event_iloc)
            launcher = proc.write_timeseries_outputs(
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
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
        launcher = proc.write_timeseries_outputs(
            which=which,
            clear_raw_outputs=clear_raw_outputs,
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )
        launcher()

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
        weather_event_indexers = self._retrieve_weather_indexer_using_integer_index(
            event_iloc
        )
        ts_scenario = self.scenarios[event_iloc]
        run = TRITONSWMM_run(weather_event_indexers, ts_scenario)
        self._sim_run_objects[event_iloc] = run
        return run

    def _retrieve_sim_run_processing_object(self, event_iloc):
        run = self._retreive_sim_runs(event_iloc)
        proc = TRITONSWMM_sim_post_processing(run)
        self._sim_run_processing_objects[event_iloc] = proc
        return proc

    def _create_launchable_sims(
        self,
        pickup_where_leftoff: bool = False,
        verbose: bool = False,
    ):
        in_slurm = self.in_slurm
        launch_functions = []
        for event_iloc in self.df_sims.index:
            run = self._retreive_sim_runs(event_iloc)
            launch_function = run.retrieve_sim_launcher(
                pickup_where_leftoff=pickup_where_leftoff,
                verbose=verbose,
                in_slurm=in_slurm,
            )
            if launch_function is None:
                continue
            launch_functions.append(launch_function)
        return launch_functions

    def run_simulations_concurrently(
        self,
        launch_functions: list[Callable[[], tuple]],
        verbose: bool = True,
    ):
        """
        Docstring for run_simulations_concurrently

        :param self: automatically chooses whether to use SLURM or ThreadPoolExecutor for concurrent runs
        """
        if self.in_slurm:
            self.run_simulations_concurrently_on_SLURM_HPC(
                launch_functions=launch_functions, verbose=verbose
            )
        else:
            self.run_simulations_concurrently_on_desktop(
                launch_functions=launch_functions, verbose=verbose
            )

    def run_simulations_concurrently_on_SLURM_HPC(
        self,
        launch_functions: list[Callable[[], tuple]],
        max_concurrent_srun: Optional[int] = None,
        verbose: bool = True,
    ) -> list[str]:
        """
        Launch simulations concurrently on an HPC system using SLURM.
        Uses a pool-based approach to limit concurrent srun tasks and avoid
        resource contention.

        Parameters
        ----------
        launch_functions : list of callables
            Each function launches a simulation and returns a tuple:
            (proc, log_file_handle, start_time, log_dict, run_obj)
        max_concurrent_srun : int | None
            Maximum number of concurrent srun tasks. If None, defaults to
            the number of physical cores available divided by threads per task.
        verbose : bool
            If True, prints progress messages.

        Returns
        -------
        list[str]
            List of simulation statuses, in completion order.
        """
        # ----------------------------
        # Determine max concurrency
        # ----------------------------
        if max_concurrent_srun is None:
            # Conservative default: use physical cores / threads per task
            physical_cores = psutil.cpu_count(logical=False) or 1
            threads_per_sim = self.cfg_analysis.n_omp_threads or 1
            max_concurrent_srun = max(1, physical_cores // threads_per_sim)

            # For interactive jobs, be even more conservative
            if verbose:
                print(
                    f"[SLURM] Auto-detected max_concurrent_srun = {max_concurrent_srun}",
                    flush=True,
                )

        if verbose:
            print(
                f"[SLURM] Running {len(launch_functions)} simulations "
                f"(max {max_concurrent_srun} concurrent srun tasks)",
                flush=True,
            )

        # ----------------------------
        # Pool-based execution
        # ----------------------------
        running: list[tuple] = []
        results: list[str] = []
        launch_iter = iter(launch_functions)
        completed_count = 0

        def launch_next():
            """Launch the next simulation if available."""
            try:
                launch = next(launch_iter)
            except StopIteration:
                return False

            proc, lf, start, log_dic, run = launch()
            running.append((proc, lf, start, log_dic, run))

            if verbose:
                print(
                    f"[SLURM] Launched sim for scenario {run._scenario.event_iloc} "
                    f"(PID {proc.pid}, {len(running)} running)",
                    flush=True,
                )
            return True

        # Prime the pool with initial tasks
        for _ in range(min(max_concurrent_srun, len(launch_functions))):
            launch_next()

        # Main polling loop
        while running:
            for entry in list(running):
                proc, lf, start, log_dic, run = entry

                # Check if process has completed
                if proc.poll() is None:
                    continue  # Still running

                # Process has completed
                lf.close()
                end_time = time.time()
                elapsed = end_time - start

                status, _ = run._check_simulation_run_status()

                log_dic.update(time_elapsed_s=elapsed, status=status)
                run.log.add_sim_entry(**log_dic)
                results.append(status)
                running.remove(entry)
                completed_count += 1

                if verbose:
                    print(
                        f"[SLURM] Scenario {run._scenario.event_iloc} completed: {status} "
                        f"({elapsed:.1f}s, {completed_count}/{len(launch_functions)} done)",
                        flush=True,
                    )

                # Launch next task if available
                launch_next()

            # Small sleep to prevent busy-waiting
            time.sleep(0.1)

        self._update_log()
        return results

    def run_simulations_concurrently_on_desktop(
        self,
        launch_functions: List[Callable[[], tuple]],
        use_gpu: bool = False,
        total_gpus_available: Optional[int] = 0,
        min_memory_per_sim_MiB: int | None = 1024,
        verbose: bool = True,
    ):
        # ----------------------------
        # Determine CPU parallelism
        # ----------------------------
        total_cores = os.cpu_count() or 1
        threads_per_sim = self.cfg_analysis.n_omp_threads or 1
        mpi_ranks = self.cfg_analysis.n_mpi_procs or 1
        n_gpus = self.cfg_analysis.n_gpus or 0

        cores_per_sim = threads_per_sim * mpi_ranks
        max_parallel_cpu = max(1, total_cores // cores_per_sim)

        # ----------------------------
        # Determine GPU parallelism
        # ----------------------------
        max_parallel_gpu = None
        if use_gpu and n_gpus and total_gpus_available:
            max_parallel_gpu = max(1, total_gpus_available // n_gpus)

        # ----------------------------
        # Calculate effective max parallel with all constraints
        # ----------------------------
        max_parallel = self.calculate_effective_max_parallel(
            min_memory_per_function_MiB=min_memory_per_sim_MiB,
            max_parallel_cpu=max_parallel_cpu,
            max_parallel_gpu=max_parallel_gpu,
            verbose=verbose,
        )

        if verbose:
            print(f"Running up to {max_parallel} simulations concurrently")

        # ----------------------------
        # Launch + monitor loop
        # ----------------------------
        running = []
        results = []

        launch_iter = iter(launch_functions)

        def launch_next():
            try:
                launch = next(launch_iter)
            except StopIteration:
                return False

            proc, lf, start, log_dic, run = launch()
            running.append((proc, lf, start, log_dic, run))
            return True

        # Prime the pool
        for _ in range(min(max_parallel, len(launch_functions))):
            launch_next()

        # Main loop
        while running:
            for entry in list(running):
                proc, lf, start, log_dic, run = entry

                if proc.poll() is None:
                    continue  # still running

                # Process finished
                lf.close()
                end_time = time.time()
                elapsed = end_time - start

                status, _ = run._check_simulation_run_status()

                log_dic["time_elapsed_s"] = elapsed
                log_dic["status"] = status
                run.log.add_sim_entry(**log_dic)

                results.append(status)
                running.remove(entry)

                if verbose:
                    print(f"Simulation finished: {status}")

                # Launch next job if available
                launch_next()

            time.sleep(0.1)  # prevent busy-waiting

        self._update_log()
        return results

    def run_all_sims_in_serially(
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
            print("Running all sims in series...")
        for event_iloc in self.df_sims.index:
            if verbose:
                print(
                    f"Running sim {event_iloc} and pickup_where_leftoff = {pickup_where_leftoff}"
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

    def compile_TRITON_SWMM(
        self, recompile_if_already_done_successfully: bool = True, verbose: bool = False
    ):
        if self.compilation_successful and not recompile_if_already_done_successfully:
            print("TRITON-SWMM already compiled")
            return
        # TODO ADD TOGGLE TO ONLY DO THIS IF NOT ALREADY COMPILED
        compiled_software_directory = self.analysis_paths.compiled_software_directory
        compilation_script = self.analysis_paths.compilation_script
        TRITONSWMM_software_directory = (
            self._system.cfg_system.TRITONSWMM_software_directory
        )
        TRITON_SWMM_make_command = self.cfg_analysis.TRITON_SWMM_make_command
        TRITON_SWMM_software_compilation_script = (
            self._system.cfg_system.TRITON_SWMM_software_compilation_script
        )
        if compiled_software_directory.exists():
            shutil.rmtree(compiled_software_directory)
        shutil.copytree(TRITONSWMM_software_directory, compiled_software_directory)
        mapping = dict(
            COMPILED_MODEL_DIR=compiled_software_directory,
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
    terminal using the method print_compilation_log"
                )
        return

    def retrieve_compilation_log(self):
        if self.analysis_paths.compilation_logfile.exists():
            return read_text_file_as_string(self.analysis_paths.compilation_logfile)
        return "no sim logfile created"

    def print_compilation_log(self):
        print(self.retrieve_compilation_log())

    def _validate_compilation(self):
        compilation_log = self.retrieve_compilation_log()
        swmm_check = "[100%] Built target runswmm" in compilation_log
        triton_check = "Building finished: triton" in compilation_log
        success = swmm_check and triton_check
        # self.compilation_successful = success
        self.log.TRITONSWMM_compiled_successfully.set(success)
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
