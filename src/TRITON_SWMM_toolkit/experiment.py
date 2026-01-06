# %%
import subprocess
import shutil
from TRITON_SWMM_toolkit.utils import create_from_template, read_text_file_as_string
from pathlib import Path
from TRITON_SWMM_toolkit.config import load_experiment_config
from dataclasses import dataclass
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
import pandas as pd
from typing import Literal
from TRITON_SWMM_toolkit.paths import ExpPaths

from TRITON_SWMM_toolkit.simulation import TRITONSWMM_sim

# from TRITON_SWMM_toolkit.system import TRITONSWMM_system

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import TRITONSWMM_system


class TRITONSWMM_experiment:

    def __init__(
        self,
        experiment_config_yaml: Path,
        ts_system: "TRITONSWMM_system",
    ) -> None:
        self._sys_paths = ts_system.sys_paths
        self._cfg_system = ts_system.cfg_system
        self.experiment_config_yaml = experiment_config_yaml
        cfg_exp = load_experiment_config(experiment_config_yaml)
        self.cfg_exp = cfg_exp
        # define additional paths not defined in cfg
        compiled_software_directory = (
            self._cfg_system.system_directory
            / self.cfg_exp.experiment_id
            / "compiled_software"
        )
        compiled_software_directory.mkdir(parents=True, exist_ok=True)
        self.exp_paths = ExpPaths(
            compiled_software_directory=compiled_software_directory,
            TRITON_build_dir=compiled_software_directory / "build",
            compilation_script=compiled_software_directory / "compile.sh",
            simulation_directory=self._cfg_system.system_directory
            / self.cfg_exp.experiment_id
            / "sims",
        )
        self.df_sims = pd.read_csv(self.cfg_exp.weather_events_to_simulate)
        self._scenario_setups = {}
        self._simulations = {}

    def _retrieve_weather_indexer_using_integer_index(
        self,
        sim_iloc,
    ):
        row = self.df_sims.loc[sim_iloc, self.cfg_exp.weather_event_indices]
        weather_event_indexers = row.to_dict()
        return weather_event_indexers

    def _add_scenario(self, sim_iloc: int):
        weather_event_indexers = self._retrieve_weather_indexer_using_integer_index(
            sim_iloc
        )
        self._scenario_setups[sim_iloc] = TRITONSWMM_scenario(
            weather_event_indexers, self
        )
        return

    def _add_all_scenarios(self):
        for sim_iloc in self.df_sims.index:
            self._add_simulation(self, sim_iloc)  # type: ignore
        return

    def _prepare_scenario(
        self, sim_iloc, overwrite_sim, rerun_swmm_hydro_if_outputs_exist
    ):
        self._scenario_setups[sim_iloc]._prepare_simulation(
            overwrite_sim, rerun_swmm_hydro_if_outputs_exist
        )

    def prepare_all_scenarios(self, overwrite_sims, rerun_swmm_hydro_if_outputs_exist):
        self._add_all_scenarios()
        for sim_iloc in self.df_sims.index:
            self._prepare_scenario(
                sim_iloc, overwrite_sims, rerun_swmm_hydro_if_outputs_exist
            )

    def run_sim(
        self,
        sim_iloc,
        mode: Literal["single_core"],
        pickup_where_leftoff,
        verbose=False,
    ):
        weather_event_indexers = self._retrieve_weather_indexer_using_integer_index(
            sim_iloc
        )
        ts_scenario = self._scenario_setups[sim_iloc]
        log = ts_scenario._initialize_sim_logfile()
        if log["simulation_creation_status"] == "complete":
            run = TRITONSWMM_sim(weather_event_indexers, ts_scenario)
            if mode == "single_core":
                run.run_singlecore_simulation(pickup_where_leftoff, verbose)
        else:
            print("Log file:")
            print(log)
            raise ValueError("simulation_creation_status must be 'complete'")
        return

    def run_all_sims_in_series(self, mode, pickup_where_leftoff):
        for sim_iloc in self.df_sims.index:
            self.run_sim(sim_iloc, mode, pickup_where_leftoff)

    def compile_TRITON_SWMM(self):
        # TODO ADD TOGGLE TO ONLY DO THIS IF NOT ALREADY COMPILED
        compiled_software_directory = self.exp_paths.compiled_software_directory
        compilation_script = self.exp_paths.compilation_script
        TRITONSWMM_software_directory = self._cfg_system.TRITONSWMM_software_directory
        TRITON_SWMM_make_command = self.cfg_exp.TRITON_SWMM_make_command
        TRITON_SWMM_software_compilation_script = (
            self._cfg_system.TRITON_SWMM_software_compilation_script
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
        compilation_logfile = compiled_software_directory / f"compilation.log"

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
        return compilation_log
