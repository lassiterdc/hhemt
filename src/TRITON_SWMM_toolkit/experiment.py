# %%
import subprocess
import shutil
from TRITON_SWMM_toolkit.utils import (
    create_from_template,
    read_text_file_as_string,
)
from pathlib import Path
from TRITON_SWMM_toolkit.config import load_experiment_config
import pandas as pd
from typing import Literal, List
from TRITON_SWMM_toolkit.paths import ExpPaths
from pprint import pprint
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.running_a_simulation import TRITONSWMM_run
from TRITON_SWMM_toolkit.processing_simulation import TRITONSWMM_sim_post_processing

from TRITON_SWMM_toolkit.processing_experiment import TRITONSWMM_exp_post_processing
from TRITON_SWMM_toolkit.constants import Mode
from TRITON_SWMM_toolkit.plot import print_json_file_tree
from TRITON_SWMM_toolkit.logging import TRITONSWMM_experiment_log

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import TRITONSWMM_system

    # from .processing_experiment import TRITONSWMM_exp_post_processing


class TRITONSWMM_experiment:
    def __init__(
        self,
        experiment_config_yaml: Path,
        system: "TRITONSWMM_system",
    ) -> None:
        self._system = system
        self.experiment_config_yaml = experiment_config_yaml
        cfg_exp = load_experiment_config(experiment_config_yaml)
        self.cfg_exp = cfg_exp
        # define additional paths not defined in cfg
        compiled_software_directory = (
            self._system.cfg_system.system_directory
            / self.cfg_exp.experiment_id
            / "compiled_software"
        )
        compiled_software_directory.mkdir(parents=True, exist_ok=True)
        experiment_dir = (
            self._system.cfg_system.system_directory / self.cfg_exp.experiment_id
        )
        self.exp_paths = ExpPaths(
            f_log=experiment_dir / "log.json",
            experiment_dir=experiment_dir,
            compiled_software_directory=compiled_software_directory,
            TRITON_build_dir=compiled_software_directory / "build",
            compilation_script=compiled_software_directory / "compile.sh",
            simulation_directory=experiment_dir / "sims",
            compilation_logfile=compiled_software_directory / f"compilation.log",
            output_triton_summary=experiment_dir
            / f"TRITON.{self.cfg_exp.TRITON_processed_output_type}",
            output_swmm_links_summary=experiment_dir
            / f"SWMM_links.{self.cfg_exp.TRITON_processed_output_type}",
            output_swmm_node_summary=experiment_dir
            / f"SWMM_nodes.{self.cfg_exp.TRITON_processed_output_type}",
        )
        self.df_sims = pd.read_csv(self.cfg_exp.weather_events_to_simulate).loc[
            :, self.cfg_exp.weather_event_indices
        ]
        self.scenarios = {}
        self._sim_run_objects = {}
        self._sim_run_processing_objects = {}
        self._simulation_run_statuses = {}
        self.run_modes = Mode
        self.compilation_successful = False

        if self.exp_paths.f_log.exists():
            self.log = TRITONSWMM_experiment_log.from_json(self.exp_paths.f_log)
        else:
            self.log = TRITONSWMM_experiment_log(logfile=self.exp_paths.f_log)

        if self.exp_paths.compilation_logfile.exists():
            self._validate_compilation()
        self._add_all_scenarios()
        self.process = TRITONSWMM_exp_post_processing(self)

    def consolidate_TRITON_simulation_summaries(
        self,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        self.process.consolidate_TRITON_outputs_for_experiment(
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
        self.process.consolidate_SWMM_outputs_for_experiment(
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )

    def print_cfg(self, which: Literal["system", "experiment", "both"] = "both"):
        if which == ["system", "both"]:
            print("=== System Configuration ===")
            self._system.cfg_system.display_tabulate_cfg()
        if which == "both":
            print("\n")
        if which in ["experiment", "both"]:
            print("=== Experiment Configuration ===")
            self.cfg_exp.display_tabulate_cfg()

    def print_all_yaml_defined_input_files(self):
        print_json_file_tree(self.dict_of_exp_and_sys_config())

    def dict_of_exp_and_sys_config(self):
        dic_exp = self._system.cfg_system.model_dump()
        dic_sys = self.cfg_exp.model_dump()
        return dic_exp | dic_sys

    def dict_of_all_sim_files(self, sim_iloc):
        dic_syspaths = self._system.sys_paths.as_dict()
        dic_exp_paths = self.exp_paths.as_dict()
        dic_sim_paths = self.scenarios[sim_iloc].scen_paths.as_dict()
        dic_all_paths = dic_syspaths | dic_exp_paths | dic_sim_paths
        return dic_all_paths

    def print_all_sim_files(self, sim_iloc):
        dic_all_paths = self.dict_of_all_sim_files(sim_iloc)
        print_json_file_tree(dic_all_paths)

    def _retrieve_weather_indexer_using_integer_index(
        self,
        sim_iloc,
    ):
        row = self.df_sims.loc[sim_iloc, self.cfg_exp.weather_event_indices]
        weather_event_indexers = row.to_dict()
        return weather_event_indexers

    def _add_scenario(self, sim_iloc: int):
        scen = TRITONSWMM_scenario(sim_iloc, self)
        self.scenarios[sim_iloc] = scen
        return scen

    def _add_all_scenarios(self):
        all_scens_created = True
        all_sims_run = True
        all_TRITON_outputs_processed = True
        all_SWMM_outputs_processed = True
        all_raw_TRITON_outputs_cleared = True
        all_raw_SWMM_outputs_cleared = True
        for sim_iloc in self.df_sims.index:
            # sim run status
            scen = self._add_scenario(sim_iloc)
            all_sims_run = all_sims_run and scen.sim_run_completed
            # sim creation status
            scen_created = bool(scen.log.scenario_creation_complete.get())
            all_scens_created = all_scens_created and scen_created
            # sim output processing status
            proc = self._retrieve_sim_run_processing_object(sim_iloc)
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
        self.log.all_sims_run.set(all_scens_created)
        self.log.all_TRITON_timeseries_processed.set(all_TRITON_outputs_processed)
        self.log.all_SWMM_timeseries_processed.set(all_SWMM_outputs_processed)
        self.log.all_raw_TRITON_outputs_cleared.set(all_raw_TRITON_outputs_cleared)
        self.log.all_raw_SWMM_outputs_cleared.set(all_raw_SWMM_outputs_cleared)
        return

    def _prepare_scenario(
        self,
        sim_iloc,
        overwrite_sim: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
    ):
        scen = self.scenarios[sim_iloc]
        scen._prepare_simulation(
            overwrite_sim, rerun_swmm_hydro_if_outputs_exist, verbose
        )
        return

    def prepare_all_scenarios(
        self,
        overwrite_sims: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
    ):
        self._add_all_scenarios()
        for sim_iloc in self.df_sims.index:
            self._prepare_scenario(
                sim_iloc, overwrite_sims, rerun_swmm_hydro_if_outputs_exist, verbose
            )
        return

    def print_logfile_for_scenario(self, sim_iloc):
        scen = self.scenarios[sim_iloc]
        scen.log.print()

    def run_sim(
        self,
        sim_iloc: int,
        mode: Mode | Literal["single_core"],
        pickup_where_leftoff,
        process_outputs_after_sim_completion: bool,
        which: Literal["TRITON", "SWMM", "both"],
        clear_raw_outputs: bool,
        overwrite_if_exist: bool,
        compression_level: int,
        verbose=False,
    ):
        ts_scenario = self.scenarios[sim_iloc]

        if not ts_scenario.log.scenario_creation_complete.get():
            print("Log file:")
            print(ts_scenario.log.print())
            raise ValueError("scenario_creation_complete must be 'success'")
        if not self.compilation_successful:
            print("Log file:")
            print(ts_scenario.log.print())
            raise ValueError("TRITONSWMM has not been compiled")
        run = self._retreive_sim_run_object(sim_iloc)
        if verbose:
            print("run instance instantiated")
        run.run_sim(mode, pickup_where_leftoff, verbose)
        self.sim_run_status(sim_iloc)
        self._add_all_scenarios()  # updates experiment log
        if process_outputs_after_sim_completion and run._scenario.sim_run_completed:
            self.process_sim_output(
                sim_iloc,
                which,
                clear_raw_outputs,
                overwrite_if_exist,
                verbose,
                compression_level,
            )
        return

    def process_sim_output(
        self,
        sim_iloc,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        proc = self._retrieve_sim_run_processing_object(sim_iloc=sim_iloc)
        proc.write_timeseries_outputs(
            which=which,
            clear_raw_outputs=clear_raw_outputs,
            overwrite_if_exist=overwrite_if_exist,
            verbose=verbose,
            compression_level=compression_level,
        )

    def process_all_sim_outputs(
        self,
        which: Literal["TRITON", "SWMM", "both"] = "both",
        clear_raw_outputs: bool = True,
        overwrite_if_exist: bool = False,
        verbose: bool = False,
        compression_level: int = 5,
    ):
        for sim_iloc in self.df_sims.index:
            self.process_sim_output(
                sim_iloc=sim_iloc,
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                verbose=verbose,
                compression_level=compression_level,
            )
        self._add_all_scenarios()

        return

    def sim_run_status(self, sim_iloc):
        run = self._retreive_sim_run_object(sim_iloc)
        status = run._scenario.latest_simlog
        self._simulation_run_statuses[sim_iloc] = status
        return status

    def _retreive_sim_run_object(self, sim_iloc):
        weather_event_indexers = self._retrieve_weather_indexer_using_integer_index(
            sim_iloc
        )
        ts_scenario = self.scenarios[sim_iloc]
        run = TRITONSWMM_run(weather_event_indexers, ts_scenario)
        self._sim_run_objects[sim_iloc] = run
        return run

    def _retrieve_sim_run_processing_object(self, sim_iloc):
        run = self._retreive_sim_run_object(sim_iloc)
        proc = TRITONSWMM_sim_post_processing(run)
        self._sim_run_processing_objects[sim_iloc] = proc
        return proc

    def run_all_sims_in_series(
        self,
        mode: Mode | Literal["single_core"],
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
        Arguments passed to processing process_sim_outputs (and only needed if process_outputs_after_sim_completion=True):
            - which: Literal["TRITON", "SWMM", "both"]
            - clear_raw_outputs: bool
            - overwrite_if_exist: bool
            - compression_level: int
        """
        if verbose:
            print("Running all sims in series...")
        for sim_iloc in self.df_sims.index:
            if verbose:
                print(
                    f"Running sim {sim_iloc} with mode = {mode} and pickup_where_leftoff = {pickup_where_leftoff}"
                )
            self.run_sim(
                sim_iloc=sim_iloc,
                mode=mode,
                pickup_where_leftoff=pickup_where_leftoff,
                verbose=verbose,
                process_outputs_after_sim_completion=process_outputs_after_sim_completion,
                which=which,
                clear_raw_outputs=clear_raw_outputs,
                overwrite_if_exist=overwrite_if_exist,
                compression_level=compression_level,
            )
        self._add_all_scenarios()  # updates log

    def compile_TRITON_SWMM(
        self, recompile_if_already_done_successfully: bool = True, verbose: bool = False
    ):
        if self.compilation_successful and not recompile_if_already_done_successfully:
            print("TRITON-SWMM already compiled")
            return
        # TODO ADD TOGGLE TO ONLY DO THIS IF NOT ALREADY COMPILED
        compiled_software_directory = self.exp_paths.compiled_software_directory
        compilation_script = self.exp_paths.compilation_script
        TRITONSWMM_software_directory = (
            self._system.cfg_system.TRITONSWMM_software_directory
        )
        TRITON_SWMM_make_command = self.cfg_exp.TRITON_SWMM_make_command
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
        compilation_logfile = self.exp_paths.compilation_logfile

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
        success = self._validate_compilation()
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
        return read_text_file_as_string(self.exp_paths.compilation_logfile)

    def print_compilation_log(self):
        print(self.retrieve_compilation_log())

    def _validate_compilation(self):
        compilation_log = self.retrieve_compilation_log()
        swmm_check = "[100%] Built target runswmm" in compilation_log
        triton_check = "Building finished: triton" in compilation_log
        success = swmm_check and triton_check
        self.compilation_successful = success
        self.log.TRITONSWMM_compiled_successfully.set(success)
        return success

    @property
    def TRITON_experiment_summary_created(self):
        return bool(self.log.TRITON_experiment_summary_created.get())

    @property
    def SWMM_node_experiment_summary_created(self):
        return bool(self.log.SWMM_node_experiment_summary_created.get())

    @property
    def SWMM_link_experiment_summary_created(self):
        return bool(self.log.SWMM_link_experiment_summary_created.get())

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
