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
    TRITONSWMM_build_dir_cpu: Path  # Always present
    TRITONSWMM_build_dir_gpu: Optional[Path]  # Only if GPU configured

    # Split build directories by backend
    TRITON_build_dir_cpu: Path  # Always present
    TRITON_build_dir_gpu: Optional[Path]  # Only if GPU configured

    SWMM_build_dir: Optional[Path]

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

    # TRITON-SWMM Coupled Model consolidated outputs
    output_tritonswmm_triton_summary: Optional[Path] = None
    output_tritonswmm_node_summary: Optional[Path] = None
    output_tritonswmm_link_summary: Optional[Path] = None
    output_tritonswmm_performance_summary: Optional[Path] = None

    # TRITON-only consolidated outputs
    output_triton_only_summary: Optional[Path] = None
    output_triton_only_performance_summary: Optional[Path] = None

    # SWMM-only consolidated outputs
    output_swmm_only_node_summary: Optional[Path] = None
    output_swmm_only_link_summary: Optional[Path] = None

    bash_script_path: Optional[Path] = None


@dataclass
class ScenarioPaths(MainDataClass):
    scenario_prep_log: Path
    sim_folder: Path
    weather_timeseries: Path
    dir_weather_datfiles: Path
    swmm_hydro_inp: Path
    swmm_hydraulics_inp: Path
    swmm_hydraulics_rpt: Optional[Path]
    swmm_full_inp: Path
    swmm_full_rpt_file: Path
    swmm_full_out_file: Path
    extbc_tseries: Path
    extbc_loc: Path
    hyg_timeseries: Path
    hyg_locs: Path

    # Model-specific CFG files
    triton_swmm_cfg: Path  # TRITON-SWMM coupled model CFG
    triton_cfg: Optional[Path] = None  # TRITON-only CFG (no SWMM)

    # Centralized logs directory
    logs_dir: Optional[Path] = None

    # Model-specific output directories
    out_triton: Optional[Path] = None  # TRITON-only outputs
    out_tritonswmm: Optional[Path] = None  # Coupled model outputs

    # Model-specific log files
    log_run_triton: Optional[Path] = None
    log_run_tritonswmm: Optional[Path] = None
    log_run_swmm: Optional[Path] = None

    # Executables
    sim_tritonswmm_executable: Optional[Path] = None  # Coupled model executable
    sim_triton_executable: Optional[Path] = None  # TRITON-only executable
    sim_swmm_executable: Optional[Path] = None  # SWMM standalone executable

    # Outputs - TRITON-SWMM Coupled Model
    tritonswmm_logfile_dir: Optional[Path] = None
    output_tritonswmm_performance_timeseries: Optional[Path] = None
    output_tritonswmm_performance_summary: Optional[Path] = None
    output_tritonswmm_triton_timeseries: Optional[Path] = None
    output_tritonswmm_triton_summary: Optional[Path] = None
    output_tritonswmm_link_time_series: Optional[Path] = None
    output_tritonswmm_link_summary: Optional[Path] = None
    output_tritonswmm_node_time_series: Optional[Path] = None
    output_tritonswmm_node_summary: Optional[Path] = None

    # Outputs - TRITON-only Model
    output_triton_only_performance_timeseries: Optional[Path] = None
    output_triton_only_performance_summary: Optional[Path] = None
    output_triton_only_timeseries: Optional[Path] = None
    output_triton_only_summary: Optional[Path] = None

    # Outputs - SWMM-only Standalone Model
    output_swmm_only_link_time_series: Optional[Path] = None
    output_swmm_only_link_summary: Optional[Path] = None
    output_swmm_only_node_time_series: Optional[Path] = None
    output_swmm_only_node_summary: Optional[Path] = None
