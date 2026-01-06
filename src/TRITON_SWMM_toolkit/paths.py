from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from dataclasses import dataclass, asdict
from TRITON_SWMM_toolkit.plot import print_json_file_tree


@dataclass
class MainDataClass:
    def as_dict(self) -> Dict[str, Path]:
        """
        Return the dataclass fields as a dictionary.
        """
        return asdict(self)

    def print_existing_files_and_nonempty_directories(self):
        print_json_file_tree(self.as_dict())

    def print_all_paths(self):
        print_json_file_tree(self.as_dict())


@dataclass
class SysPaths(MainDataClass):
    dem_processed: Path
    mannings_processed: Path


@dataclass
class ExpPaths(MainDataClass):
    compiled_software_directory: Path
    TRITON_build_dir: Path
    compilation_script: Path
    simulation_directory: Path
    compilation_logfile: Path


@dataclass
class SimPaths(MainDataClass):
    f_log: Path
    sim_folder: Path
    dir_weather_datfiles: Path
    inp_hydro: Path
    inp_hydraulics: Path
    inp_full: Path
    extbc_tseries: Path
    extbc_loc: Path
    hyg_timeseries: Path
    hyg_locs: Path
    triton_swmm_cfg: Path
    sim_tritonswmm_executable: Path
    tritonswmm_logfile_dir: Path
