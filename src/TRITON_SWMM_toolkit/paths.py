from dataclasses import dataclass
from pathlib import Path


@dataclass
class SysPaths:
    dem_processed: Path
    mannings_processed: Path


@dataclass
class ExpPaths:
    compiled_software_directory: Path
    TRITON_build_dir: Path
    compilation_script: Path
    simulation_directory: Path


@dataclass
class SimPaths:
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
