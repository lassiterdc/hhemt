# %%
import pandas as pd
import rioxarray as rxr
import numpy as np
import xarray as xr
import sys
import shutil
import swmmio
import warnings
import TRITON_SWMM_toolkit.utils as utils
from datetime import datetime
from pathlib import Path
from TRITON_SWMM_toolkit.exceptions import CompilationError, ConfigurationError
from TRITON_SWMM_toolkit.log import (
    TRITONSWMM_scenario_log,
    TRITONSWMM_model_log,
    LogField,
)
from TRITON_SWMM_toolkit.paths import ScenarioPaths
from typing import TYPE_CHECKING, Literal, Optional
import threading
from TRITON_SWMM_toolkit.subprocess_utils import run_subprocess_with_tee
from TRITON_SWMM_toolkit.scenario_inputs import ScenarioInputGenerator
from TRITON_SWMM_toolkit.swmm_runoff_modeling import SWMMRunoffModeler
from TRITON_SWMM_toolkit.swmm_full_model import SWMMFullModelBuilder


lock = threading.Lock()

if TYPE_CHECKING:
    from .analysis import TRITONSWMM_analysis


class TRITONSWMM_scenario:
    log: TRITONSWMM_scenario_log

    def __init__(self, event_iloc: int, analysis: "TRITONSWMM_analysis") -> None:
        self.event_iloc = event_iloc
        self._analysis = analysis
        self._system = analysis._system
        self.weather_event_indexers = (
            self._analysis._retrieve_weather_indexer_using_integer_index(event_iloc)
        )
        from TRITON_SWMM_toolkit.run_simulation import TRITONSWMM_run

        # define sim specific filepaths
        analysis_simulations_folder = self._analysis.analysis_paths.simulation_directory
        self.sim_id_str = self._retrieve_sim_id_str()
        sim_folder = analysis_simulations_folder / self.sim_id_str
        processed_output_folder = sim_folder / "processed"
        processed_output_folder.mkdir(parents=True, exist_ok=True)
        swmm_folder = sim_folder / "swmm"
        swmm_folder.mkdir(parents=True, exist_ok=True)
        self.backend = analysis.backend

        # Model toggles from system config
        cfg_sys = self._system.cfg_system
        out_type = self._analysis.cfg_analysis.TRITON_processed_output_type

        # Centralized logs directory
        logs_dir = sim_folder / "logs"

        # Model-specific output directories
        out_triton = sim_folder / "out_triton" if cfg_sys.toggle_triton_model else None
        out_tritonswmm = (
            sim_folder / "out_tritonswmm" if cfg_sys.toggle_tritonswmm_model else None
        )

        self.scen_paths = ScenarioPaths(
            sim_folder=sim_folder,
            scenario_prep_log=sim_folder / "scenario_prep_log.json",
            weather_timeseries=sim_folder / "sim_weather.nc",
            # swmm time series
            dir_weather_datfiles=sim_folder / "dats",
            # swmm-related
            swmm_hydro_inp=swmm_folder / "hydro.inp",  # runoff input generation
            swmm_hydraulics_inp=swmm_folder
            / "hydraulics.inp",  # TRITON-SWMM .inp for modeling hydraulics
            swmm_hydraulics_rpt=(
                out_tritonswmm / "swmm" / "hydraulics.rpt" if out_tritonswmm else None
            ),  # runoff generation output
            swmm_full_inp=swmm_folder / "full.inp",  # full SWMM model
            swmm_full_rpt_file=swmm_folder / "full.rpt",  # full swmm RPT
            swmm_full_out_file=swmm_folder / "full.out",  # full swmm binary output file
            # external boundary conditions
            extbc_tseries=sim_folder / "extbc" / f"tseries.txt",
            extbc_loc=sim_folder / "extbc" / f"loc.extbc",
            # inflow hydrographs
            hyg_timeseries=sim_folder / "strmflow" / "tseries.hyg",
            hyg_locs=sim_folder / "strmflow" / "loc.txt",
            # Model-specific CFG files
            triton_swmm_cfg=sim_folder / "TRITONSWMM.cfg",
            triton_cfg=(
                sim_folder / "TRITON.cfg" if cfg_sys.toggle_triton_model else None
            ),
            # Centralized logs
            logs_dir=logs_dir,
            # Model-specific output directories
            out_triton=out_triton,
            out_tritonswmm=out_tritonswmm,
            # Model-specific log files
            log_run_triton=(
                logs_dir / "run_triton.log" if cfg_sys.toggle_triton_model else None
            ),
            log_run_tritonswmm=(
                logs_dir / "run_tritonswmm.log"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            log_run_swmm=(
                logs_dir / "run_swmm.log" if cfg_sys.toggle_swmm_model else None
            ),
            # Executables
            sim_tritonswmm_executable=sim_folder / "build" / "triton.exe",
            sim_triton_executable=(
                sim_folder / "build_triton" / "triton.exe"
                if cfg_sys.toggle_triton_model
                else None
            ),
            sim_swmm_executable=(
                self._system.swmm_executable if cfg_sys.toggle_swmm_model else None
            ),
            tritonswmm_logfile_dir=sim_folder / "tritonswmm_sim_logfiles",
            # TRITON-SWMM Coupled Model Outputs
            output_tritonswmm_performance_timeseries=(
                processed_output_folder / f"TRITONSWMM_perf_tseries.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_performance_summary=(
                processed_output_folder / f"TRITONSWMM_perf_summary.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_triton_timeseries=(
                processed_output_folder / f"TRITONSWMM_TRITON_tseries.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_triton_summary=(
                processed_output_folder / f"TRITONSWMM_TRITON_summary.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_link_time_series=(
                processed_output_folder / f"TRITONSWMM_SWMM_link_tseries.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_link_summary=(
                processed_output_folder / f"TRITONSWMM_SWMM_link_summary.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_node_time_series=(
                processed_output_folder / f"TRITONSWMM_SWMM_node_tseries.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            output_tritonswmm_node_summary=(
                processed_output_folder / f"TRITONSWMM_SWMM_node_summary.{out_type}"
                if cfg_sys.toggle_tritonswmm_model
                else None
            ),
            # TRITON-only Model Outputs
            output_triton_only_performance_timeseries=(
                processed_output_folder / f"TRITON_only_perf_tseries.{out_type}"
                if cfg_sys.toggle_triton_model
                else None
            ),
            output_triton_only_performance_summary=(
                processed_output_folder / f"TRITON_only_perf_summary.{out_type}"
                if cfg_sys.toggle_triton_model
                else None
            ),
            output_triton_only_timeseries=(
                processed_output_folder / f"TRITON_only_tseries.{out_type}"
                if cfg_sys.toggle_triton_model
                else None
            ),
            output_triton_only_summary=(
                processed_output_folder / f"TRITON_only_summary.{out_type}"
                if cfg_sys.toggle_triton_model
                else None
            ),
            # SWMM-only Standalone Model Outputs (in swmm/ folder)
            output_swmm_only_link_time_series=(
                processed_output_folder / f"SWMM_only_link_tseries.{out_type}"
                if cfg_sys.toggle_swmm_model
                else None
            ),
            output_swmm_only_link_summary=(
                processed_output_folder / f"SWMM_only_link_summary.{out_type}"
                if cfg_sys.toggle_swmm_model
                else None
            ),
            output_swmm_only_node_time_series=(
                processed_output_folder / f"SWMM_only_node_tseries.{out_type}"
                if cfg_sys.toggle_swmm_model
                else None
            ),
            output_swmm_only_node_summary=(
                processed_output_folder / f"SWMM_only_node_summary.{out_type}"
                if cfg_sys.toggle_swmm_model
                else None
            ),
        )
        self._create_directories()
        if self.scen_paths.scenario_prep_log.exists():
            self.log = TRITONSWMM_scenario_log.from_json(
                self.scen_paths.scenario_prep_log
            )
        else:
            self.log = TRITONSWMM_scenario_log(
                event_iloc=self.event_iloc,
                event_idx=self.weather_event_indexers,
                simulation_folder=self.scen_paths.sim_folder,
                logfile=self.scen_paths.scenario_prep_log,
            )
        self.run = TRITONSWMM_run(self)

        # Initialize scenario preparation components
        self._input_generator = ScenarioInputGenerator(self)
        self._runoff_modeler = SWMMRunoffModeler(self)
        self._full_model_builder = SWMMFullModelBuilder(self)

    def get_log(
        self, model_type: Literal["triton", "tritonswmm", "swmm"]
    ) -> TRITONSWMM_model_log:
        """
        Get the log for a specific model type.

        Each model type has its own log file to avoid race conditions in
        multi-model concurrent execution.

        Args:
            model_type: Which model's log to retrieve

        Returns:
            Model-specific log with only relevant fields initialized
        """
        log_file = self.scen_paths.sim_folder / f"log_{model_type}.json"

        # Load existing log if it exists, otherwise create new one
        if log_file.exists():
            log = TRITONSWMM_model_log.from_json(log_file)
        else:
            log = TRITONSWMM_model_log(
                event_iloc=self.event_iloc,
                event_idx=self.weather_event_indexers,
                simulation_folder=self.scen_paths.sim_folder,
                logfile=log_file,
            )

        # Initialize model-specific fields if they are None
        # (Handles both new logs and existing logs that haven't been fully populated yet)
        if model_type in ("triton", "tritonswmm"):
            # TRITON models need performance and TRITON output fields
            if log.performance_timeseries_written is None:
                log.performance_timeseries_written = LogField()
            if log.performance_summary_written is None:
                log.performance_summary_written = LogField()
            if log.TRITON_timeseries_written is None:
                log.TRITON_timeseries_written = LogField()
            if log.TRITON_summary_written is None:
                log.TRITON_summary_written = LogField()
            if log.raw_TRITON_outputs_cleared is None:
                log.raw_TRITON_outputs_cleared = LogField()
            if log.full_TRITON_timeseries_cleared is None:
                log.full_TRITON_timeseries_cleared = LogField()

        if model_type in ("swmm", "tritonswmm"):
            # SWMM models need SWMM output fields
            if log.SWMM_node_timeseries_written is None:
                log.SWMM_node_timeseries_written = LogField()
            if log.SWMM_link_timeseries_written is None:
                log.SWMM_link_timeseries_written = LogField()
            if log.SWMM_node_summary_written is None:
                log.SWMM_node_summary_written = LogField()
            if log.SWMM_link_summary_written is None:
                log.SWMM_link_summary_written = LogField()
            if log.raw_SWMM_outputs_cleared is None:
                log.raw_SWMM_outputs_cleared = LogField()
            if log.full_SWMM_timeseries_cleared is None:
                log.full_SWMM_timeseries_cleared = LogField()

        # Re-bind parent log reference after assigning optional LogField members
        # (needed so LogField.set() can call parent .write()).
        log.model_post_init(None)

        return log

    @property
    def model_types_enabled(self) -> list[str]:
        """Get list of enabled model types from system config."""
        enabled = []
        cfg_sys = self._system.cfg_system
        if cfg_sys.toggle_triton_model:
            enabled.append("triton")
        if cfg_sys.toggle_tritonswmm_model:
            enabled.append("tritonswmm")
        if cfg_sys.toggle_swmm_model:
            enabled.append("swmm")
        return enabled

    @property
    def sim_compute_time_min(self) -> float:
        """Simulation compute time in minutes.

        Returns 0.0 — runtime tracking via simlog was removed. Derive from
        log file timestamps or performance.txt if needed in the future.
        """
        return 0.0

    def model_run_completed(
        self, model_type: Literal["triton", "tritonswmm", "swmm"]
    ) -> bool:
        """Check completion status for a specific model type.

        Parameters
        ----------
        model_type : Literal["triton", "tritonswmm", "swmm"]
            Which model to check completion for

        Returns
        -------
        bool
            True if the specified model completed successfully
        """
        # Use log-file-based completion checking
        success = self.run.model_run_completed(model_type)

        return success

    def latest_sim_date(
        self,
        model_type: Literal["triton", "tritonswmm", "swmm"],
        astype: Literal["dt", "str"] = "dt",
    ) -> datetime | str:
        """Get the simulation datetime from the specified model's log.

        Returns datetime.min / "" — run timestamp is not currently persisted in the log.
        """
        return datetime.min if astype == "dt" else ""

    def _create_directories(self):
        """Create all required directories for the scenario."""
        self.scen_paths.dir_weather_datfiles.mkdir(parents=True, exist_ok=True)
        self.scen_paths.extbc_tseries.parent.mkdir(parents=True, exist_ok=True)
        self.scen_paths.hyg_timeseries.parent.mkdir(parents=True, exist_ok=True)

        # Centralized logs directory
        if self.scen_paths.logs_dir:
            self.scen_paths.logs_dir.mkdir(parents=True, exist_ok=True)

        # Model-specific output directories
        if self.scen_paths.out_triton:
            self.scen_paths.out_triton.mkdir(parents=True, exist_ok=True)
        if self.scen_paths.out_tritonswmm:
            self.scen_paths.out_tritonswmm.mkdir(parents=True, exist_ok=True)

        # Legacy logfile dir (for backwards compatibility)
        if self.scen_paths.tritonswmm_logfile_dir:
            self.scen_paths.tritonswmm_logfile_dir.mkdir(parents=True, exist_ok=True)
        return

    def _retrieve_sim_id_str(self):
        sim_id_str = "_".join(
            f"{idx}.{val}" for idx, val in self.weather_event_indexers.items()
        )
        return f"{self.event_iloc}-{sim_id_str}"

    def seconds_to_hhmm(self, seconds):
        seconds = int(seconds)
        h, rem = divmod(int(seconds), 3600)
        return f"{h}:{rem//60:02d}"

    def seconds_to_hhmmss(self, seconds: int | float) -> str:
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _generate_TRITON_SWMM_cfg(self):
        use_constant_mannings = self._system.cfg_system.toggle_use_constant_mannings
        dem_processed = self._system.sys_paths.dem_processed
        manhole_diameter = self._analysis.cfg_analysis.manhole_diameter
        manhole_loss_coefficient = self._analysis.cfg_analysis.manhole_loss_coefficient
        TRITON_raw_output_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        mannings_processed = self._system.sys_paths.mannings_processed
        constant_mannings = self._system.cfg_system.constant_mannings
        hydraulic_timestep_s = self._analysis.cfg_analysis.hydraulic_timestep_s
        TRITON_reporting_timestep_s = (
            self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        )
        open_boundaries = self._analysis.cfg_analysis.open_boundaries
        triton_swmm_configuration_template = (
            self._system.cfg_system.triton_swmm_configuration_template
        )

        if use_constant_mannings:
            const_man_toggle = ""
            man_file_toggle = "#"
        else:
            const_man_toggle = "#"
            man_file_toggle = ""

        swmmmodel = swmmio.Model(str(self.scen_paths.swmm_hydro_inp))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_options = swmmmodel.inp.options
        start_datetime = pd.to_datetime(
            sim_options.Value.START_DATE + " " + sim_options.Value.START_TIME
        )
        end_datetime = pd.to_datetime(
            sim_options.Value.END_DATE + " " + sim_options.Value.END_TIME
        )
        sim_dur_s = int((end_datetime - start_datetime) / np.timedelta64(1, "s"))

        df_extbc_loc = pd.read_csv(self.scen_paths.extbc_loc)
        num_ext_bc = len(df_extbc_loc)

        df_src_loc = pd.read_csv(self.scen_paths.hyg_locs)
        num_srcs = len(df_src_loc)

        sim_id_str = self.sim_id_str

        mapping = dict(
            CASE_DESC=sim_id_str,
            DEM=dem_processed,
            SWMM=self.scen_paths.swmm_hydraulics_inp,
            MH_DIAM=manhole_diameter,
            MH_LOSS=manhole_loss_coefficient,
            NUM_SOURCES=num_srcs,
            OUT_FORMAT=TRITON_raw_output_type.upper(),
            HYDROGRAPH=self.scen_paths.hyg_timeseries,
            HYDO_SRC_LOC=self.scen_paths.hyg_locs,
            MANNINGS=mannings_processed,
            CONST_MAN_TOGGLE=const_man_toggle,
            MAN_FILE_TOGGLE=man_file_toggle,
            CONST_MAN=constant_mannings,
            NUM_EXT_BC=num_ext_bc,
            EXTBC_DIR=str(self.scen_paths.extbc_loc.parent),
            EXTBC_FILE=self.scen_paths.extbc_loc,
            SIM_DUR_S=sim_dur_s,
            TSTEP_S=hydraulic_timestep_s,
            REPORTING_TSTEP_S=TRITON_reporting_timestep_s,
            OPEN_BOUNDARIES=open_boundaries,
        )
        utils.create_from_template(
            triton_swmm_configuration_template, mapping, self.scen_paths.triton_swmm_cfg
        )

        # Post-process to add output_folder for TRITON-SWMM outputs
        cfg_content = self.scen_paths.triton_swmm_cfg.read_text()
        if "output_folder" not in cfg_content:
            # Insert after dem_filename line
            cfg_content = cfg_content.replace(
                "\ndem_filename=", f'\noutput_folder="out_tritonswmm"\ndem_filename='
            )
            self.scen_paths.triton_swmm_cfg.write_text(cfg_content)

        self.log.triton_swmm_cfg_created.set(True)
        return

    def _generate_TRITON_cfg(self):
        """
        Generate TRITON-only configuration file (no SWMM coupling).

        This creates a TRITON.cfg with inp_filename commented out,
        enabling standalone 2D hydrodynamic simulations without SWMM.
        """
        if not self._system.cfg_system.toggle_triton_model:
            return  # Skip if TRITON-only not enabled

        if self.scen_paths.triton_cfg is None:
            return  # Path not configured

        use_constant_mannings = self._system.cfg_system.toggle_use_constant_mannings
        dem_processed = self._system.sys_paths.dem_processed
        TRITON_raw_output_type = self._analysis.cfg_analysis.TRITON_raw_output_type
        mannings_processed = self._system.sys_paths.mannings_processed
        constant_mannings = self._system.cfg_system.constant_mannings
        hydraulic_timestep_s = self._analysis.cfg_analysis.hydraulic_timestep_s
        TRITON_reporting_timestep_s = (
            self._analysis.cfg_analysis.TRITON_reporting_timestep_s
        )
        open_boundaries = self._analysis.cfg_analysis.open_boundaries
        triton_swmm_configuration_template = (
            self._system.cfg_system.triton_swmm_configuration_template
        )

        if use_constant_mannings:
            const_man_toggle = ""
            man_file_toggle = "#"
        else:
            const_man_toggle = "#"
            man_file_toggle = ""

        # Get simulation duration from SWMM model (same as TRITON-SWMM)
        swmmmodel = swmmio.Model(str(self.scen_paths.swmm_hydro_inp))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim_options = swmmmodel.inp.options
        start_datetime = pd.to_datetime(
            sim_options.Value.START_DATE + " " + sim_options.Value.START_TIME
        )
        end_datetime = pd.to_datetime(
            sim_options.Value.END_DATE + " " + sim_options.Value.END_TIME
        )
        sim_dur_s = int((end_datetime - start_datetime) / np.timedelta64(1, "s"))

        df_extbc_loc = pd.read_csv(self.scen_paths.extbc_loc)
        num_ext_bc = len(df_extbc_loc)

        df_src_loc = pd.read_csv(self.scen_paths.hyg_locs)
        num_srcs = len(df_src_loc)

        sim_id_str = self.sim_id_str

        # TRITON-only mapping - SWMM is commented out
        mapping = dict(
            CASE_DESC=f"{sim_id_str}_triton_only",
            DEM=dem_processed,
            SWMM="#DISABLED_FOR_TRITON_ONLY",  # Will be commented out in post-processing
            MH_DIAM=0,  # Not used in TRITON-only
            MH_LOSS=0,  # Not used in TRITON-only
            NUM_SOURCES=num_srcs,
            OUT_FORMAT=TRITON_raw_output_type.upper(),
            HYDROGRAPH=self.scen_paths.hyg_timeseries,
            HYDO_SRC_LOC=self.scen_paths.hyg_locs,
            MANNINGS=mannings_processed,
            CONST_MAN_TOGGLE=const_man_toggle,
            MAN_FILE_TOGGLE=man_file_toggle,
            CONST_MAN=constant_mannings,
            NUM_EXT_BC=num_ext_bc,
            EXTBC_DIR=str(self.scen_paths.extbc_loc.parent),
            EXTBC_FILE=self.scen_paths.extbc_loc,
            SIM_DUR_S=sim_dur_s,
            TSTEP_S=hydraulic_timestep_s,
            REPORTING_TSTEP_S=TRITON_reporting_timestep_s,
            OPEN_BOUNDARIES=open_boundaries,
        )

        # Create CFG from template
        utils.create_from_template(
            triton_swmm_configuration_template, mapping, self.scen_paths.triton_cfg
        )

        # Post-process to comment out inp_filename line and add output_folder
        cfg_content = self.scen_paths.triton_cfg.read_text()
        cfg_content = cfg_content.replace(
            'inp_filename="#DISABLED_FOR_TRITON_ONLY"',
            '#inp_filename=""  # TRITON-only mode (no SWMM coupling)',
        )

        # Add output_folder for TRITON-only outputs
        if "output_folder" not in cfg_content:
            # Insert after dem_filename line
            cfg_content = cfg_content.replace(
                "\ndem_filename=", f'\noutput_folder="out_triton"\ndem_filename='
            )

        self.scen_paths.triton_cfg.write_text(cfg_content)

        self.log.triton_cfg_created.set(True)
        return

    def _copy_tritonswmm_build_folder_to_sim(self):
        """
        Symlink TRITON-SWMM build folder into simulation directory.

        Parameters
        ----------
        backend : str
            Which backend build to symlink ("cpu" or "gpu")
        """
        # Select source build directory
        if self.backend == "cpu":
            src_build_fpath = self._system.sys_paths.TRITONSWMM_build_dir_cpu
        elif self.backend == "gpu":
            if self._system.sys_paths.TRITONSWMM_build_dir_gpu is None:
                raise ConfigurationError(
                    field="gpu_compilation_backend",
                    message="GPU backend requested but gpu_compilation_backend not set.\n"
                    "  Set gpu_compilation_backend='HIP' or 'CUDA' in system config YAML.",
                    config_path=self._system.system_config_yaml,
                )
            src_build_fpath = self._system.sys_paths.TRITONSWMM_build_dir_gpu
        else:
            raise ConfigurationError(
                field="backend",
                message=f"Unknown backend '{self.backend}'. Must be 'cpu' or 'gpu'.",
            )

        # Verify source exists and compilation successful
        if not src_build_fpath.exists():
            raise FileNotFoundError(
                f"{self.backend.upper()} build directory not found: {src_build_fpath}"
            )

        if self.backend == "cpu" and not self._system.compilation_cpu_successful:
            raise CompilationError(
                model_type="tritonswmm",
                backend="cpu",
                logfile=self._system.sys_paths.compilation_logfile_cpu,
                return_code=1,
            )
        elif self.backend == "gpu" and not self._system.compilation_gpu_successful:
            raise CompilationError(
                model_type="tritonswmm",
                backend="gpu",
                logfile=self._system.sys_paths.compilation_logfile_gpu,  # type: ignore
                return_code=1,
            )

        # Link into scenario (strict symlink; no fallback copy)
        sim_tritonswmm_executable = self.scen_paths.sim_tritonswmm_executable
        target_build_fpath = sim_tritonswmm_executable.parent  # type: ignore
        self._create_strict_dir_symlink(
            source_dir=src_build_fpath,
            target_link=target_build_fpath,
            label="TRITON-SWMM build",
        )

        # Update log
        self.log.sim_tritonswmm_executable_copied.set(True)
        self.log.triton_backend_used.set(self.backend)

    def _copy_triton_only_build_folder_to_sim(self):
        """Symlink TRITON-only build folder to scenario directory when enabled."""
        if not self._system.cfg_system.toggle_triton_model:
            return

        if self.scen_paths.sim_triton_executable is None:
            return

        if self.backend == "cpu":
            src_build_fpath = self._system.sys_paths.TRITON_build_dir_cpu
            compiled_ok = self._system.compilation_triton_only_cpu_successful
        elif self.backend == "gpu":
            if self._system.sys_paths.TRITON_build_dir_gpu is None:
                raise ConfigurationError(
                    field="gpu_compilation_backend",
                    message="GPU backend requested but gpu_compilation_backend not set in system config.",
                    config_path=self._system.system_config_yaml,
                )
            src_build_fpath = self._system.sys_paths.TRITON_build_dir_gpu
            compiled_ok = self._system.compilation_triton_only_gpu_successful
        else:
            raise ConfigurationError(
                field="backend",
                message=f"Unknown backend '{self.backend}'. Must be 'cpu' or 'gpu'.",
            )

        if not src_build_fpath.exists():
            raise FileNotFoundError(
                f"TRITON-only build directory not found: {src_build_fpath}"
            )

        if not compiled_ok:
            raise CompilationError(
                model_type="triton",
                backend=self.backend,
                logfile=src_build_fpath / "compilation.log",
                return_code=1,
            )

        target_build_fpath = self.scen_paths.sim_triton_executable.parent
        self._create_strict_dir_symlink(
            source_dir=src_build_fpath,
            target_link=target_build_fpath,
            label="TRITON-only build",
        )

    def _create_strict_dir_symlink(self, source_dir, target_link, label: str) -> None:
        """
        Create/replace a directory symlink and fail fast if not exactly correct.

        This intentionally does NOT fall back to copying build artifacts.
        """
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot create symlink for {label}: source directory missing or invalid: {source_dir}"
            )

        # Remove any existing target path (dir/file/symlink) so symlink creation is deterministic.
        if target_link.exists() or target_link.is_symlink():
            if target_link.is_symlink() or target_link.is_file():
                target_link.unlink()
            elif target_link.is_dir():
                shutil.rmtree(target_link)
            else:
                target_link.unlink()

        target_link.parent.mkdir(parents=True, exist_ok=True)

        try:
            target_link.symlink_to(source_dir, target_is_directory=True)
        except OSError as e:
            raise RuntimeError(
                f"Failed to create required symlink for {label}.\n"
                f"  source: {source_dir}\n"
                f"  target: {target_link}\n"
                f"This workflow requires symlinks and will not fall back to copying build directories."
            ) from e

        # Fail loud if symlink is not present or points somewhere unexpected.
        if not target_link.is_symlink():
            raise RuntimeError(
                f"Expected {label} target to be a symlink, but it is not: {target_link}"
            )

        resolved_target = target_link.resolve(strict=True)
        resolved_source = source_dir.resolve(strict=True)
        if resolved_target != resolved_source:
            raise RuntimeError(
                f"{label} symlink points to unexpected location.\n"
                f"  expected: {resolved_source}\n"
                f"  actual:   {resolved_target}\n"
                f"  link:     {target_link}"
            )

    def _write_sim_weather_nc(self):
        weather_timeseries = self._analysis.cfg_analysis.weather_timeseries
        weather_event_indexers = self.weather_event_indexers
        with lock:
            with xr.open_dataset(
                weather_timeseries, engine="h5netcdf"
            ) as ds_event_weather_series:
                ds_event_ts = ds_event_weather_series.sel(weather_event_indexers).load()
                utils.write_netcdf(
                    ds_event_ts,
                    self.scen_paths.weather_timeseries,
                    compression_level=5,
                    chunks="auto",
                )

    @property
    def ds_event_ts(self):
        if not self.scen_paths.weather_timeseries.exists():
            self._write_sim_weather_nc()
        return xr.open_dataset(self.scen_paths.weather_timeseries, engine="h5netcdf")

    def prepare_scenario(
        self,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
    ):
        """
        Prepare scenario for simulation.

        Parameters
        ----------
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenario
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        backend : Optional[str]
            Force specific backend ("cpu" or "gpu"). If None, auto-selects based on run_mode.
        """
        # Halt if scenario already complete
        if (
            self.log.scenario_creation_complete.get()
            and not overwrite_scenario_if_already_set_up
        ):
            print(  # type: ignore
                "Simulation already successfully created. "
                "If you wish to overwrite it, re-run with overwrite_scenario_if_already_set_up=True.",
                flush=True,
            )
            return

        # Validate backend is available
        if self.backend == "gpu" and not self._system.compilation_gpu_successful:
            logfile = self._system.sys_paths.compilation_logfile_gpu
            raise CompilationError(
                model_type="tritonswmm",
                backend="gpu",
                logfile=logfile if logfile else Path("missing"),
                return_code=1,
            )

        if self.backend == "cpu" and not self._system.compilation_cpu_successful:
            raise CompilationError(
                model_type="tritonswmm",
                backend="cpu",
                logfile=self._system.sys_paths.compilation_logfile_cpu,
                return_code=1,
            )

        print(
            f"[Scenario {self.event_iloc}] Using {self.backend.upper()} backend",
            flush=True,
        )

        # Main scenario setup
        self._write_sim_weather_nc()

        # SWMM runoff modeling - generates hydrograph inputs
        self._runoff_modeler.write_swmm_rainfall_dat_files()
        self._runoff_modeler.write_swmm_waterlevel_dat_files()

        # Create SWMM hydraulics model - direct TRITON-SWMM input
        self._input_generator.create_hydraulics_model_from_template(
            self._system.cfg_system.SWMM_hydraulics,
            self.scen_paths.swmm_hydraulics_inp,
        )
        self.log.inp_hydraulics_model_created_successfully.set(True)

        # Optional: Full SWMM model (standalone SWMM execution)
        if self._system.cfg_system.toggle_swmm_model:
            self._full_model_builder.create_full_model_from_template(
                self._system.cfg_system.SWMM_full,
                self.scen_paths.swmm_full_inp,
            )
            self.log.inp_full_model_created_successfully.set(True)

        # SWMM hydrology for runoff generation
        if self._system.cfg_system.toggle_use_swmm_for_hydrology:
            self._runoff_modeler.create_hydrology_model_from_template(
                self._system.cfg_system.SWMM_hydrology,
                self.scen_paths.swmm_hydro_inp,
            )
            self._runoff_modeler.run_swmm_hydro_model(
                rerun_if_exists=rerun_swmm_hydro_if_outputs_exist,
                verbose=False,
            )
            self.log.inp_hydro_model_created_successfully.set(True)

        # Create TRITON inputs
        self._input_generator.create_external_boundary_condition_files()
        self._runoff_modeler.write_hydrograph_files()
        self._input_generator.update_hydraulics_model_to_have_1_inflow_node_per_DEM_gridcell(
            verbose=False
        )

        # Generate model-specific CFG files
        self._generate_TRITON_SWMM_cfg()  # Coupled model CFG
        self._generate_TRITON_cfg()  # TRITON-only CFG (if enabled)

        # Copy build folders - FAIL FAST if toggle ON but not compiled
        # TRITON-SWMM: Check toggle and compilation status
        if self._system.cfg_system.toggle_tritonswmm_model:
            if not (
                self._system.log.compilation_tritonswmm_cpu_successful.get()
                or self._system.log.compilation_tritonswmm_gpu_successful.get()
            ):
                raise RuntimeError(
                    "toggle_tritonswmm_model is enabled but TRITON-SWMM was not successfully compiled. "
                    "Either compile TRITON-SWMM (system.compile_TRITON_SWMM()) or disable the toggle "
                    "(set toggle_tritonswmm_model=False in system config)."
                )
            self._copy_tritonswmm_build_folder_to_sim()

        # TRITON-only: Check toggle and compilation status
        if self._system.cfg_system.toggle_triton_model:
            if not (
                self._system.log.compilation_triton_cpu_successful.get()
                or self._system.log.compilation_triton_gpu_successful.get()
            ):
                raise RuntimeError(
                    "toggle_triton_model is enabled but TRITON-only was not successfully compiled. "
                    "Either compile TRITON-only (system.compile_TRITON_only()) or disable the toggle "
                    "(set toggle_triton_model=False in system config)."
                )
            self._copy_triton_only_build_folder_to_sim()

        # SWMM: Check toggle and compilation status
        # Note: SWMM doesn't need build folder copying - uses absolute path to executable
        if self._system.cfg_system.toggle_swmm_model:
            if not self._system.log.compilation_swmm_successful.get():
                raise RuntimeError(
                    "toggle_swmm_model is enabled but SWMM was not successfully compiled. "
                    "Either compile SWMM (system.compile_SWMM()) or disable the toggle "
                    "(set toggle_swmm_model=False in system config)."
                )

        self.log.scenario_creation_complete.set(True)
        print("Scenario preparation complete", flush=True)

        return

    def _create_subprocess_prepare_scenario_launcher(
        self,
        overwrite_scenario_if_already_set_up: bool = False,
        rerun_swmm_hydro_if_outputs_exist: bool = False,
        verbose: bool = False,
    ):
        """
        Create a launcher function that runs scenario preparation in a subprocess.

        This isolates PySwmm to a separate process, avoiding MultiSimulationError
        when preparing multiple scenarios concurrently.

        Parameters
        ----------
        event_iloc : int
            Integer index of the scenario to prepare
        overwrite_scenario_if_already_set_up : bool
            If True, overwrite existing scenario
        rerun_swmm_hydro_if_outputs_exist : bool
            If True, rerun SWMM hydrology model even if outputs exist
        verbose : bool
            If True, print progress messages

        Returns
        -------
        callable
            A launcher function that executes the subprocess
        """

        event_iloc = self.event_iloc
        scenario_logfile = self.log.logfile.parent / f"scenario_prep_{event_iloc}.log"

        # Build command - always use direct Python execution (no srun)
        cmd = [
            f"{self._analysis._python_executable}",
            "-m",
            "TRITON_SWMM_toolkit.prepare_scenario_runner",
            "--event-iloc",
            str(event_iloc),
            "--analysis-config",
            str(self._analysis.analysis_config_yaml),
            "--system-config",
            str(self._system.system_config_yaml),
        ]

        # Add optional flags
        if overwrite_scenario_if_already_set_up:
            cmd.append("--overwrite-scenario-if-already-set-up")
        if rerun_swmm_hydro_if_outputs_exist:
            cmd.append("--rerun-swmm-hydro")

        def launcher():
            """Execute scenario preparation in a subprocess."""
            if verbose:
                print(
                    f"[Scenario {event_iloc}] Launching subprocess: {' '.join(cmd)}",
                    flush=True,
                )

            # Use tee logging to write to both file and stdout
            proc = run_subprocess_with_tee(
                cmd=cmd,
                logfile=scenario_logfile,
                env=None,  # Uses os.environ by default
                echo_to_stdout=True,
            )

            rc = proc.returncode

            if verbose:
                if rc == 0:
                    print(
                        f"[Scenario {event_iloc}] Subprocess completed successfully",
                        flush=True,
                    )
                else:
                    print(
                        f"[Scenario {event_iloc}] Subprocess failed with return code {rc}",
                        flush=True,
                    )

        return launcher


def return_tstep_in_hrs(time_indexed_pd_obj):
    tstep_sim_tseries = pd.Series(time_indexed_pd_obj.index.diff()).mode()[0]
    tstep_sim_tseries_h = tstep_sim_tseries / np.timedelta64(1, "h")
    return tstep_sim_tseries_h


def return_tstep_in_hrs_for_weather_time_series(
    ds_tseries, weather_time_series_timestep_dimension_name
):
    time_indexed_pd_obj = ds_tseries[
        weather_time_series_timestep_dimension_name
    ].to_dataframe()
    return return_tstep_in_hrs(time_indexed_pd_obj)


def extract_vertex_coordinates(geometry):
    # Ensure the geometry is a LineString or MultiLineString
    if geometry.geom_type in ["LineString", "MultiLineString"]:
        return list(geometry.coords)
    else:
        return None


def infer_side(dem, min_x, max_x, min_y, max_y):
    dem_min_x = dem.x.values.min()
    dem_max_x = dem.x.values.max()
    dem_min_y = dem.y.values.min()
    dem_max_y = dem.y.values.max()
    if abs(min_x - max_x) > abs(min_y - max_y):
        loc = "top_or_bottom"
        if abs(max_y - dem_max_y) > abs(min_y - dem_min_y):
            loc = "bottom"
        else:
            loc = "top"
    else:
        loc = "left_or_right"
        if abs(max_x - dem_max_x) > abs(min_x - dem_min_x):
            loc = "left"
        else:
            loc = "right"
    return loc


def find_closest_dem_coord(x_val, y_val, BC_side, rds_dem):
    dem_xs = rds_dem.x.values  # + cellsize/2
    dem_ys = rds_dem.y.values
    if BC_side == "left":
        x_coord = min(dem_xs)
        y_coord = dem_ys[np.argmin(np.abs(dem_ys - y_val))]
    elif BC_side == "right":
        x_coord = max(dem_xs)
        y_coord = dem_ys[np.argmin(np.abs(dem_ys - y_val))]
    elif BC_side == "top":
        x_coord = dem_xs[np.argmin(np.abs(dem_xs - x_val))]
        y_coord = max(dem_ys)
    elif BC_side == "bottom":
        x_coord = dem_xs[np.argmin(np.abs(dem_xs - x_val))]
        y_coord = min(dem_ys)
    else:
        print("boundary condition location not defined")
    if (x_coord < min(dem_xs)) or (x_coord > max(dem_xs)):
        sys.exit("This x coordinate falls outside the domain of the DEM")
    if (y_coord < min(dem_ys)) or (y_coord > max(dem_ys)):
        sys.exit("This y coordinate falls outside the domain of the DEM")
    return x_coord, y_coord


def find_lowest_inv(node_to_keep, nodes):
    from scipy.stats import rankdata

    lst_invs = []
    for node_id in node_to_keep:
        row = nodes.loc[node_id, :]
        inv_elev = row.InvertElev
        lst_invs.append(inv_elev)
    ranks_inv = rankdata(lst_invs, method="min")
    # subset the nodes that have the lowest elevation
    node_to_keep = node_to_keep[ranks_inv == min(ranks_inv)]
    node_to_keep = list(np.unique(node_to_keep))  # type: ignore
    return node_to_keep


def return_df_of_nodes_grouped_by_DEM_gridcell(f_inp, dem_processed, verbose=False):
    rds_dem = rxr.open_rasterio(dem_processed)
    model = swmmio.Model(str(f_inp))
    warnings.filterwarnings(
        "ignore", category=UserWarning, module=r"swmmio\.utils\.dataframes"
    )
    node_coords = model.nodes.geodataframe["geometry"]
    dem_xs = rds_dem.x.values  # type: ignore
    dem_ys = rds_dem.y.values  # type: ignore
    d_node_locs = dict(node_key=[], dem_x_coord=[], dem_y_coord=[])
    lst_outfalls = list(model.nodes.geodataframe["OutfallType"].dropna().index)
    ## creating a row for each group of nodes associated with a single DEM cell (this is to make sure there is only 1 inflow node per gridcell)
    for node_id in node_coords.index:
        # verify that the node is within the dem
        node = node_coords[node_id]
        x_coord = node.x
        y_coord = node.y
        closest_dem_cell_x_ind = pd.Series(abs(dem_xs - x_coord)).idxmin()
        closest_dem_cell_y_ind = pd.Series(abs(dem_ys - y_coord)).idxmin()
        d_node_locs["node_key"].append(node_id)
        d_node_locs["dem_x_coord"].append(dem_xs[closest_dem_cell_x_ind])
        d_node_locs["dem_y_coord"].append(dem_ys[closest_dem_cell_y_ind])
        lst_out_of_bounds_nodes = []
        if (
            (x_coord < dem_xs.min())
            or (x_coord > dem_xs.max())
            or (y_coord < dem_ys.min())
            or (y_coord > dem_ys.max())
        ):
            if verbose:
                print("WARNING: node out bounds. Node ID: {}".format(node_id))
                print(
                    "dem lower left: ({},{}) | dem upper right: ({}, {})".format(
                        dem_xs.min(), dem_ys.min(), dem_xs.max(), dem_ys.max()
                    )
                )
                print("node coords: {}, {}".format(x_coord, y_coord))
            lst_out_of_bounds_nodes.append(node_id)
    ## create dataframe with node key and associated dem x and y coordinate for grouping
    df_node_locs = pd.DataFrame(d_node_locs)
    return df_node_locs, lst_outfalls


def calc_area(row):
    """calculate the cross-sectional area of a sewer segment. If the segment
    is multi-barrel, the area will reflect the total of all barrels"""
    if row.Shape == "ARCH":  # TREATING AS RECTANGULAR FOR SIMPLICITY
        h = row.Geom1
        w = row.Geom2
        area = h * w
        # print("Encountered arch cross sectional shape. Currently calculating a rectangular area assuming it's close enough.")
        return area * row.Barrels
    elif row.Shape in [
        "CIRCULAR",
        "HORIZ_ELLIPSE",
    ]:  # assuming horizontal ellipse is circular area
        d = row.Geom1
        area = 3.1415 * (d * d) / 4
        return round((area * row.Barrels), 2)
    elif "RECT" in row.Shape:
        # assume triangular bottom sections (geom3) deepens the excavated box
        return (row.Geom1 + row.Geom3) * float(row.Geom2) * row.Barrels
    elif row.Shape == "EGG":
        # assume geom1 is the span
        return row.Geom1 * 1.5 * row.Barrels
    else:
        print("shape not recognized in calc_area")
    return


# %%
