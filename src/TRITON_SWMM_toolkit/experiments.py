# %%
import subprocess
import shutil
from TRITON_SWMM_toolkit.utils import create_from_template, read_text_file_as_string
from pathlib import Path
from TRITON_SWMM_toolkit.system import TRITONSWMM_system
from types import SimpleNamespace
from TRITON_SWMM_toolkit.config import load_experiment_config
from dataclasses import dataclass
from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
import pandas as pd
from TRITON_SWMM_toolkit.config import system_config
from TRITON_SWMM_toolkit.system import SysPaths


@dataclass
class ExpPaths:
    compiled_software_directory: Path
    TRITON_build_dir: Path
    compilation_script: Path
    simulation_directory: Path


class TRITONSWMM_experiment:
    def __init__(
        self,
        experiment_config_yaml: Path,
        sys_paths: SysPaths,
        cfg_system: system_config,
    ) -> None:
        self._sys_paths = sys_paths
        self._cfg_system = cfg_system
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
        self._sim_setups = {}
        # add all simulations
        self._add_all_sims()

    def _add_simulation(self, sim_iloc: int):
        weather_event_indexers = self._retrieve_weather_indexer_using_integer_index(
            sim_iloc
        )
        self._sim_setups[sim_iloc] = TRITONSWMM_scenario(weather_event_indexers, self)
        return

    def _add_all_sims(self):
        for sim_iloc in self.df_sims.index:
            self._add_simulation(self, sim_iloc)  # type: ignore
        return

    def _retrieve_weather_indexer_using_integer_index(
        self,
        sim_iloc,
    ):
        row = self.df_sims.loc[sim_iloc, self.cfg_exp.weather_event_indices]
        weather_event_indexers = row.to_dict()
        return weather_event_indexers

    def _prepare_sim(self, sim_iloc, overwrite_sim, rerun_swmm_hydro_if_outputs_exist):
        self._sim_setups[sim_iloc]._prepare_simulation(
            overwrite_sim, rerun_swmm_hydro_if_outputs_exist
        )

    def prepare_all_simulations(
        self, overwrite_sims, rerun_swmm_hydro_if_outputs_exist
    ):
        self._add_all_sims()
        for sim_iloc in self.df_sims.index:
            self._prepare_sim(
                sim_iloc, overwrite_sims, rerun_swmm_hydro_if_outputs_exist
            )

    def compile_TRITON_SWMM(self):
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
