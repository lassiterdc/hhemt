from dataclasses import asdict, dataclass
from pathlib import Path

from hhemt.plot_utils import print_json_file_tree


@dataclass
class MainDataClass:
    def as_dict(self) -> dict[str, Path]:
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
    TRITONSWMM_build_dir_gpu: Path | None  # Only if GPU configured

    # Split build directories by backend
    TRITON_build_dir_cpu: Path  # Always present
    TRITON_build_dir_gpu: Path | None  # Only if GPU configured

    SWMM_build_dir: Path | None

    # Split compilation artifacts by backend
    compilation_script_cpu: Path
    compilation_script_gpu: Path | None
    compilation_logfile_cpu: Path
    compilation_logfile_gpu: Path | None

    # Backwards compatibility aliases (point to CPU versions)
    TRITON_build_dir: Path | None = None
    compilation_script: Path | None = None
    compilation_logfile: Path | None = None

    system_datatree_zarr: Path | None = None


@dataclass
class AnalysisPaths(MainDataClass):
    f_log: Path
    analysis_dir: Path
    simulation_directory: Path
    analysis_log_directory: Path
    simlog_directory: Path

    # TRITON-SWMM Coupled Model consolidated outputs
    output_tritonswmm_triton_summary: Path | None = None
    output_tritonswmm_node_summary: Path | None = None
    output_tritonswmm_link_summary: Path | None = None
    output_tritonswmm_performance_summary: Path | None = None

    # TRITON-only consolidated outputs
    output_triton_only_summary: Path | None = None
    output_triton_only_performance_summary: Path | None = None

    # SWMM-only consolidated outputs
    output_swmm_only_node_summary: Path | None = None
    output_swmm_only_link_summary: Path | None = None

    # Hierarchical DataTree zarr — aggregates all enabled modes into one store.
    analysis_datatree_zarr: Path | None = None

    # Sensitivity-level hierarchical DataTree zarr (aggregates member trees).
    sensitivity_datatree_zarr: Path | None = None

    bash_script_path: Path | None = None


@dataclass
class ScenarioPaths(MainDataClass):
    scenario_prep_log: Path
    sim_folder: Path
    weather_timeseries: Path
    dir_weather_datfiles: Path
    swmm_hydro_inp: Path
    swmm_hydraulics_inp: Path
    swmm_hydraulics_rpt: Path | None
    swmm_full_inp: Path
    swmm_full_rpt_file: Path | None
    swmm_full_out_file: Path | None
    extbc_tseries: Path
    extbc_loc: Path
    hyg_timeseries: Path
    hyg_locs: Path

    # Model-specific CFG files
    triton_swmm_cfg: Path  # TRITON-SWMM coupled model CFG
    triton_cfg: Path | None = None  # TRITON-only CFG (no SWMM)

    # Centralized logs directory
    logs_dir: Path | None = None

    # Model-specific output directories
    out_triton: Path | None = None  # TRITON-only outputs
    out_tritonswmm: Path | None = None  # Coupled model outputs

    # Model-specific log files: RETIRED. `log_run_triton` / `log_run_tritonswmm` /
    # `log_run_swmm` declared `{sim_folder}/logs/run_{model}.log` — a path NOTHING has
    # ever written. The real per-sim runtime log is analysis-level and is resolved by
    # `run_simulation.model_logfile_for` (the single source of truth). The dead fields
    # were removed because they were not merely unused: they were a TRAP.
    # `analysis_validation.check_coupled_resume_validity` hand-inlined
    # `Path(scenario_directory)/"logs"/"run_tritonswmm.log"` — byte-identical to the
    # retired `log_run_tritonswmm` — so every read raised, every row was skipped, and the
    # check passed VACUOUSLY on every experiment. A declared-but-never-written path field
    # reads as ground truth to authors and reviewers alike; do not reintroduce one.

    # Executables
    sim_tritonswmm_executable: Path | None = None  # Coupled model executable
    sim_triton_executable: Path | None = None  # TRITON-only executable
    sim_swmm_executable: Path | None = None  # SWMM standalone executable

    # Outputs - TRITON-SWMM Coupled Model
    output_tritonswmm_performance_timeseries: Path | None = None
    output_tritonswmm_performance_summary: Path | None = None
    output_tritonswmm_triton_timeseries: Path | None = None
    output_tritonswmm_triton_summary: Path | None = None
    output_tritonswmm_link_time_series: Path | None = None
    output_tritonswmm_link_summary: Path | None = None
    output_tritonswmm_node_time_series: Path | None = None
    output_tritonswmm_node_summary: Path | None = None

    # Outputs - TRITON-only Model
    output_triton_only_performance_timeseries: Path | None = None
    output_triton_only_performance_summary: Path | None = None
    output_triton_only_timeseries: Path | None = None
    output_triton_only_summary: Path | None = None

    # Outputs - SWMM-only Standalone Model
    output_swmm_only_link_time_series: Path | None = None
    output_swmm_only_link_summary: Path | None = None
    output_swmm_only_node_time_series: Path | None = None
    output_swmm_only_node_summary: Path | None = None
