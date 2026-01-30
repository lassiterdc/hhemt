from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional
from TRITON_SWMM_toolkit.plot_utils import print_json_file_tree


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

    # Split build directories by backend
    TRITON_build_dir_cpu: Path  # Always present
    TRITON_build_dir_gpu: Optional[Path]  # Only if GPU configured

    # Split compilation artifacts by backend
    compilation_script_cpu: Path
    compilation_script_gpu: Optional[Path]
    compilation_logfile_cpu: Path
    compilation_logfile_gpu: Optional[Path]

    # Backwards compatibility aliases (point to CPU versions)
    TRITON_build_dir: Optional[Path] = None
    compilation_script: Optional[Path] = None
    compilation_logfile: Optional[Path] = None


@dataclass
class AnalysisPaths(MainDataClass):
    f_log: Path
    analysis_dir: Path
    simulation_directory: Path
    output_triton_summary: Path
    output_swmm_links_summary: Path
    output_swmm_node_summary: Path
    output_tritonswmm_performance_summary: Path
    bash_script_path: Optional[Path] = None


@dataclass
class ScenarioPaths(MainDataClass):
    f_log: Path
    sim_folder: Path
    weather_timeseries: Path
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
    output_tritonswmm_performance_timeserie: Path
    output_tritonswmm_performance_summary: Path
    output_triton_timeseries: Path
    output_swmm_link_time_series: Path
    output_swmm_node_time_series: Path
    output_triton_summary: Path
    output_swmm_node_summary: Path
    output_swmm_link_summary: Path
