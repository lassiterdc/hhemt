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
from typing import Literal
from TRITON_SWMM_toolkit.paths import ExpPaths
from pprint import pprint
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
from TRITON_SWMM_toolkit.running_a_simulation import TRITONSWMM_run
from TRITON_SWMM_toolkit.constants import Mode
from TRITON_SWMM_toolkit.plot import print_json_file_tree


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import TRITONSWMM_system


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
        self.exp_paths = ExpPaths(
            compiled_software_directory=compiled_software_directory,
            TRITON_build_dir=compiled_software_directory / "build",
            compilation_script=compiled_software_directory / "compile.sh",
            simulation_directory=self._system.cfg_system.system_directory
            / self.cfg_exp.experiment_id
            / "sims",
            compilation_logfile=compiled_software_directory / f"compilation.log",
        )
        self.df_sims = pd.read_csv(self.cfg_exp.weather_events_to_simulate)
        self.scenarios = {}
        self._sim_run_objects = {}
        self._simulation_run_statuses = {}
        self.run_modes = Mode
        self.compilation_successful = False
        if self.exp_paths.compilation_logfile.exists():
            self._validate_compilation()
        self._add_all_scenarios()

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
        dic_exp = self._system.cfg_system.model_dump()
        dic_sys = self.cfg_exp.model_dump()
        print_json_file_tree(dic_exp | dic_sys)

    def print_all_sim_files(self, sim_iloc):
        dic_syspaths = self._system.sys_paths.as_dict()
        dic_exp_paths = self.exp_paths.as_dict()
        dic_sim_paths = self.scenarios[sim_iloc].sim_paths.as_dict()
        dic_all_paths = dic_syspaths | dic_exp_paths | dic_sim_paths
        print_json_file_tree(dic_all_paths)

    def _retrieve_weather_indexer_using_integer_index(
        self,
        sim_iloc,
    ):
        row = self.df_sims.loc[sim_iloc, self.cfg_exp.weather_event_indices]
        weather_event_indexers = row.to_dict()
        return weather_event_indexers

    def _add_scenario(self, sim_iloc: int):
        self.scenarios[sim_iloc] = TRITONSWMM_scenario(sim_iloc, self)
        self.scenarios[
            sim_iloc
        ].log.TRITONSWMM_compiled_successfully_for_experiment.set(
            self.compilation_successful
        )
        return

    def _add_all_scenarios(self):
        for sim_iloc in self.df_sims.index:
            self._add_scenario(sim_iloc)
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
        mode: Mode | str,
        pickup_where_leftoff,
        verbose=False,
    ):
        weather_event_indexers = self._retrieve_weather_indexer_using_integer_index(
            sim_iloc
        )
        ts_scenario = self.scenarios[sim_iloc]

        if not ts_scenario.log.scenario_creation_complete:
            print("Log file:")
            print(ts_scenario.log.print())
            raise ValueError("scenario_creation_complete must be 'success'")
        if not self.compilation_successful:
            print("Log file:")
            print(ts_scenario.log.print())
            raise ValueError("TRITONSWMM has not been compiled")
        run = self._retreive_sim_run_object(sim_iloc)
        if mode == "single_core":
            run.run_singlecore_simulation(pickup_where_leftoff, verbose)
        self.sim_run_status(sim_iloc)
        return

    def sim_run_status(self, sim_iloc):
        run = self._retreive_sim_run_object(sim_iloc)
        status = run.latest_sim_status()
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

    def run_all_sims_in_series(self, mode: Mode | str, pickup_where_leftoff: bool):
        for sim_iloc in self.df_sims.index:
            self.run_sim(sim_iloc, mode, pickup_where_leftoff)

    def compile_TRITON_SWMM(self, recompile_if_already_done_successfully: bool = True):
        if self.compilation_successful and not recompile_if_already_done_successfully:
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
        if not success:
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
        return success


# %%
