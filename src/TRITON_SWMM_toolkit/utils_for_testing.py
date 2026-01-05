# %%
import xarray as xr
import pandas as pd
import numpy as np
from TRITON_SWMM_toolkit.prepare_a_simulation import (
    retrieve_weather_indexer_using_integer_index,
)


def create_reduced_weather_file_for_testing_if_it_does_not_exist(
    og_weather_timeseries,
    new_weather_timeseries,
    weather_events_to_simulate,
    weather_event_indices,
    weather_time_series_timestep_dimension_name,
    dur_min,
):

    weather_event_indexers = retrieve_weather_indexer_using_integer_index(
        0, weather_events_to_simulate, weather_event_indices
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
                return
        else:  # if they are not identical, remove the file and rerewrite
            new_weather_timeseries.unlink()
    ds_event_weather_series.to_netcdf(new_weather_timeseries)
    return
