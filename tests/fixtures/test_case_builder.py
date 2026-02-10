"""
Test case builder for TRITON-SWMM toolkit.

This module provides the `retrieve_TRITON_SWMM_test_case` class for creating
isolated test environments with:
- Synthetic weather timeseries
- Short simulation durations
- Isolated test directories
- Platform-specific configurations

Example:
    test_case = retrieve_TRITON_SWMM_test_case(
        cfg_system_yaml=system_yaml_path,
        analysis_name="test_analysis",
        n_events=5,
        n_reporting_tsteps_per_sim=12,
        TRITON_reporting_timestep_s=10,
        test_system_dirname="tests",
        start_from_scratch=True,
    )
    system = test_case.system
"""

import pandas as pd
import numpy as np
import shutil
import yaml
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import TRITON_SWMM_toolkit.constants as cnst

# Import from production package
from TRITON_SWMM_toolkit.config.loaders import load_analysis_config
from TRITON_SWMM_toolkit.config.analysis import analysis_config
from TRITON_SWMM_toolkit.system import TRITONSWMM_system
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis

from TRITON_SWMM_toolkit.examples import TRITON_SWMM_example


class retrieve_TRITON_SWMM_test_case:
    """
    Create isolated TRITON-SWMM test case with synthetic weather data.

    This class:
    - Loads a base system configuration
    - Creates an isolated test directory
    - Generates synthetic weather timeseries (NetCDF format)
    - Creates shortened analysis configuration for fast testing
    - Processes system-level inputs (unless start_from_scratch=False)

    The test case is self-contained and can be run independently without
    affecting other tests or production data.

    Attributes:
        system: TRITONSWMM_system instance with configured analysis
    """

    # LOADING FROM SYSTEM CONFIG
    def __init__(
        self,
        example: TRITON_SWMM_example,
        # cfg_system_yaml: Path,
        analysis_name: str,
        n_events: int,
        n_reporting_tsteps_per_sim: int,
        TRITON_reporting_timestep_s: int,
        test_system_dirname: str,
        analysis_description: str = "",
        start_from_scratch: bool = False,
        additional_analysis_configs: Optional[dict] = None,
        additional_system_configs: Optional[dict] = None,
    ):
        """
        Initialize test case from system configuration.

        Args:
            cfg_system_yaml: Path to base system configuration YAML
            analysis_name: Name for the test analysis
            n_events: Number of weather events to simulate
            n_reporting_tsteps_per_sim: Number of timesteps per simulation
            TRITON_reporting_timestep_s: Reporting timestep in seconds
            test_system_dirname: Name for isolated test directory
            analysis_description: Optional description for analysis
            start_from_scratch: If True, remove existing test directory and reprocess inputs
            additional_analysis_configs: Dict of extra analysis config overrides
            additional_system_configs: Dict of extra system config overrides
        """
        # Fix mutable default arguments
        additional_analysis_configs = additional_analysis_configs or {}
        additional_system_configs = additional_system_configs or {}

        # load system
        self.system = example.system
        self.analysis = example.analysis

        for key, val in additional_system_configs.items():
            setattr(self.system.cfg_system, key, val)

        # update system directory
        self.system.cfg_system.system_directory = (
            self.system.cfg_system.system_directory.parent / test_system_dirname
        )
        anlysys_dir = self.system.cfg_system.system_directory / analysis_name

        if start_from_scratch and anlysys_dir.exists():
            shutil.rmtree(anlysys_dir)
        anlysys_dir.mkdir(parents=True, exist_ok=True)

        new_system_config_yaml = (
            self.system.cfg_system.system_directory / f"{test_system_dirname}.yaml"
        )

        new_system_config_yaml.write_text(
            yaml.safe_dump(
                self.system.cfg_system.model_dump(mode="json"),
                sort_keys=False,  # .dict() for pydantic v1
            )
        )
        # udpate system
        self.system = TRITONSWMM_system(new_system_config_yaml)

        # load single sime analysis
        # single_sim_anlysys_yaml = NorfolkIreneExample.load_analysis_template()
        cfg_analysis = example.analysis.cfg_analysis.model_copy()

        # update analysis attributes
        cfg_analysis.analysis_id = analysis_name
        cfg_analysis.analysis_description = analysis_description
        f_weather_indices = anlysys_dir / "weather_indices.csv"
        cfg_analysis.weather_events_to_simulate = f_weather_indices
        event_index_name = "event_id"
        cfg_analysis.weather_event_indices = [event_index_name]
        cfg_analysis.TRITON_reporting_timestep_s = TRITON_reporting_timestep_s

        # create weather indexer dataset
        df_weather_indices = pd.DataFrame({event_index_name: np.arange(n_events)})
        df_weather_indices.to_csv(f_weather_indices)

        # add additional fields
        for key, val in additional_analysis_configs.items():
            setattr(cfg_analysis, key, val)

        f_weather_tseries = anlysys_dir / "weather_tseries.nc"
        cfg_analysis.weather_timeseries = f_weather_tseries

        cfg_analysis = analysis_config.model_validate(cfg_analysis)
        # write analysis as yaml
        cfg_anlysys_yaml = anlysys_dir / f"{analysis_name}.yaml"
        cfg_anlysys_yaml.write_text(
            yaml.safe_dump(
                cfg_analysis.model_dump(mode="json"),
                sort_keys=False,  # .dict() for pydantic v1
            )
        )
        # update analysis
        self.analysis = TRITONSWMM_analysis(cfg_anlysys_yaml, self.system)
        # Link analysis back to system
        self.system._analysis = self.analysis

        self.create_short_intense_weather_timeseries(
            f_weather_tseries, n_reporting_tsteps_per_sim, n_events, event_index_name
        )
        self.system.process_system_level_inputs(
            overwrite_outputs_if_already_created=start_from_scratch
        )

    # create weather time series dataset
    def create_short_intense_weather_timeseries(
        self,
        f_out,
        n_reporting_tsteps_per_sim,
        n_events,
        event_index_name,
        rain_intensity=50,
        storm_tide=3,
    ):
        """
        Generate synthetic weather timeseries for testing.

        Creates a NetCDF file with:
        - Constant rainfall intensity across all rain gages
        - Constant storm tide water level
        - Multiple events with identical weather patterns

        Args:
            f_out: Output NetCDF file path
            n_reporting_tsteps_per_sim: Number of timesteps per event
            n_events: Number of weather events to generate
            event_index_name: Name of event index dimension
            rain_intensity: Rainfall rate (mm/hr) - default 50 mm/hr
            storm_tide: Water level (m) - default 3m
        """
        wlevel_name = self.analysis.cfg_analysis.weather_time_series_storm_tide_datavar
        tstep_coord_name = (
            self.analysis.cfg_analysis.weather_time_series_timestep_dimension_name
        )
        df_raingage_mapping = pd.read_csv(self.system.cfg_system.subcatchment_raingage_mapping)  # type: ignore
        gage_colname = (
            self.system.cfg_system.subcatchment_raingage_mapping_gage_id_colname
        )
        gages = df_raingage_mapping[gage_colname].unique()

        reporting_tstep_sec = self.analysis.cfg_analysis.TRITON_reporting_timestep_s

        timesteps = pd.date_range(
            start="2000-01-01",
            periods=n_reporting_tsteps_per_sim + 1,
            freq=f"{int(reporting_tstep_sec)}s",
        )
        columns = list(gages) + [wlevel_name]
        df_tseries = pd.DataFrame(index=timesteps, columns=columns)
        df_tseries.loc[:, wlevel_name] = storm_tide  # type: ignore
        df_tseries.loc[:, gages] = rain_intensity
        df_tseries.index.name = tstep_coord_name
        df_tseries.columns = df_tseries.columns.astype(str)
        lst_df = []
        for event_idx in np.arange(n_events):
            df = df_tseries.copy()
            df[event_index_name] = event_idx
            lst_df.append(df)
        df_tseries = pd.concat(lst_df)
        df_tseries = df_tseries.reset_index().set_index(
            [event_index_name, tstep_coord_name]
        )

        ds_weather_tseries = df_tseries.to_xarray()
        ds_weather_tseries.to_netcdf(f_out)
        return
