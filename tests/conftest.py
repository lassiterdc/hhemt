# %% load all libraries and constants
import xarray as xr
import pandas as pd
from TRITON_SWMM_toolkit.examples import (
    load_norfolk_system_config,
    load_norfolk_single_sim_experiment,
)
from pathlib import Path
from TRITON_SWMM_toolkit.system import TRITONSWMM_system
from TRITON_SWMM_toolkit.experiment import TRITONSWMM_experiment

# from TRITON_SWMM_toolkit.scenario import TRITONSWMM_scenario
# from TRITON_SWMM_toolkit.simulation import TRITONSWMM_sim

TST_DIR_SUFFIX = "test"
DUR_MIN = 10  # for testing


#  define test case class
class TRITON_SWMM_testcase:
    cfg_exp_1sim_yaml: Path
    dur_min: int = 10
    event_iloc_for_subsetting = 0

    # LOADING FROM SYSTEM CONFIG
    def __init__(
        self, cfg_system_yaml: Path, cfg_exp_1sim_yaml: Path, test_dirname: str
    ):
        self.ts_sys = TRITONSWMM_system(cfg_system_yaml)
        self.ts_sys.cfg_system.system_directory = (
            self.ts_sys.cfg_system.system_directory.parent / test_dirname
        )
        self.ts_exp = TRITONSWMM_experiment(cfg_exp_1sim_yaml, self.ts_sys)
        new_weather_timeseries = (
            self._create_reduced_weather_file_for_testing_if_it_does_not_exist()
        )
        self.ts_exp.cfg_exp.weather_timeseries = new_weather_timeseries  # type: ignore

    def _create_reduced_weather_file_for_testing_if_it_does_not_exist(self):
        og_weather_timeseries = self.ts_exp.cfg_exp.weather_timeseries
        new_weather_timeseries = (
            self.ts_sys.cfg_system.system_directory / "weather_subset.nc"
        )
        # weather_events_to_simulate = self.ts_exp.cfg_exp.weather_events_to_simulate
        # weather_event_indices = self.ts_exp.cfg_exp.weather_event_indices
        weather_time_series_timestep_dimension_name = (
            self.ts_exp.cfg_exp.weather_time_series_timestep_dimension_name
        )
        dur_min = self.dur_min

        weather_event_indexers = (
            self.ts_exp._retrieve_weather_indexer_using_integer_index(
                self.event_iloc_for_subsetting
            )
        )

        ds_event_weather_series = xr.open_dataset(og_weather_timeseries)
        ds_event_ts = ds_event_weather_series.sel(weather_event_indexers)

        peak_idx = ds_event_ts["mm_per_hr"].to_series().dropna().idxmax()
        # compute 6 min window around peak rainfall
        first_idx = peak_idx - pd.Timedelta(f"{dur_min/2} minutes")  # type: ignore
        last_idx = first_idx + pd.Timedelta(f"{dur_min} minutes")

        ds_event_weather_series = ds_event_weather_series.sel(
            {weather_time_series_timestep_dimension_name: slice(first_idx, last_idx)}
        )

        tsteps_new = ds_event_weather_series[
            weather_time_series_timestep_dimension_name
        ].to_series()

        new_weather_timeseries.parent.mkdir(parents=True, exist_ok=True)

        if new_weather_timeseries.exists():
            with xr.open_dataset(new_weather_timeseries) as ds_existing:
                tsteps_existing = ds_existing[
                    weather_time_series_timestep_dimension_name
                ].to_series()
            if len(tsteps_new) == len(tsteps_existing):
                if (
                    tsteps_new == tsteps_existing
                ).all():  # don't rewrite if it already matches
                    return new_weather_timeseries
            else:  # if they are not identical, remove the file and rerewrite
                new_weather_timeseries.unlink()
        ds_event_weather_series.to_netcdf(new_weather_timeseries)
        print(f"created weather netcdf {new_weather_timeseries}")
        return new_weather_timeseries


norfolk_system_yaml = load_norfolk_system_config(download_if_exists=False)
norfolk_1sim_1core_experiment_yaml = load_norfolk_single_sim_experiment()

single_sim_single_core = TRITON_SWMM_testcase(
    norfolk_system_yaml, norfolk_1sim_1core_experiment_yaml, "sys_test"
)
