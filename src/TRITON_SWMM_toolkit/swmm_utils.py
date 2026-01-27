"""
SWMM Utility Functions

This module contains shared utility functions for creating SWMM .inp files from templates.
These functions are used across multiple SWMM model creation modules to avoid code duplication.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING
import TRITON_SWMM_toolkit.utils as utils

if TYPE_CHECKING:
    from .scenario import TRITONSWMM_scenario


def create_swmm_inp_from_template(
    scenario: "TRITONSWMM_scenario",
    swmm_model_template: Path,
    destination: Path,
) -> None:
    """
    Create SWMM .inp file from template with scenario-specific values.

    This is a shared utility function used by all SWMM model creation methods
    (hydrology, hydraulics, and full models). It fills in template placeholders
    with scenario-specific values including time series data, rain gauges,
    simulation timing, and reporting intervals.

    Parameters
    ----------
    scenario : TRITONSWMM_scenario
        The scenario object containing configuration and paths
    swmm_model_template : Path
        Path to the SWMM template file
    destination : Path
        Path where the filled template should be written

    Raises
    ------
    ValueError
        If rainfall units are not 'mm/hr' or 'mm'
    SystemExit
        If required template keys are missing from the mapping
    """
    cfg_analysis = scenario._analysis.cfg_analysis
    weather_time_series_timestep_dimension_name = (
        cfg_analysis.weather_time_series_timestep_dimension_name
    )

    ds_event_ts = scenario.ds_event_ts

    # Calculate timestep interval
    tstep_seconds = (
        ds_event_ts[weather_time_series_timestep_dimension_name]
        .to_series()
        .diff()
        .mode()
        .iloc[0]
        .total_seconds()  # type:ignore
    )
    interval = scenario.seconds_to_hhmmss(tstep_seconds)

    # Determine rainfall format
    if cfg_analysis.rainfall_units == "mm/hr":
        format = "INTENSITY"
    elif cfg_analysis.rainfall_units == "mm":
        format = "DEPTH"
    else:
        raise ValueError(
            f"Invalid rainfall units specified. Expecting mm/hr or mm but got {cfg_analysis.rainfall_units}"
        )

    # Create time series and rain gauge sections
    fs = scenario.log.swmm_rainfall_dat_files.get()

    lst_tseries_section = []
    rain_gages_section = []
    for key, val in fs.items():
        lst_tseries_section.append(f'{key} FILE "{val}"')
        row = f"{key} {format} {interval} 1.0 TIMESERIES {key}"
        rain_gages_section.append(row)

    # Add storm tide boundary if enabled
    if cfg_analysis.toggle_storm_tide_boundary:
        lst_tseries_section.append(
            f'water_level FILE "{scenario.log.storm_tide_for_swmm.get()}"'
        )

    # Build template mapping
    mapping = dict()
    mapping["TIMESERIES"] = "\n\n".join(lst_tseries_section)
    mapping["RAINGAGES"] = "\n".join(rain_gages_section)
    template_keys = utils.find_all_keys_in_template(swmm_model_template)

    # Get simulation time bounds
    first_tstep = (
        ds_event_ts[weather_time_series_timestep_dimension_name].to_series().min()
    )
    last_tstep = (
        ds_event_ts[weather_time_series_timestep_dimension_name].to_series().max()
    )

    # Add timing parameters
    mapping["START_DATE"] = first_tstep.strftime("%m/%d/%Y")
    mapping["START_TIME"] = first_tstep.strftime("%H:%M:%S")
    mapping["REPORT_START_DATE"] = mapping["START_DATE"]
    mapping["REPORT_START_TIME"] = mapping["START_TIME"]
    mapping["END_DATE"] = last_tstep.strftime("%m/%d/%Y")
    mapping["END_TIME"] = last_tstep.strftime("%H:%M:%S")
    mapping["REPORT_STEP"] = scenario.seconds_to_hhmmss(
        cfg_analysis.TRITON_reporting_timestep_s
    )

    # Validate all template keys are present
    missing_keys = [key for key in template_keys if key not in mapping.keys()]
    if missing_keys:
        print(
            f"One or more keys were not found in the dictionary defining template fill values."
        )
        print(f"Missing keys: {missing_keys}")
        print(f"All expected keys: {template_keys}")
        print(f"All keys accounted for: {mapping.keys()}")
        sys.exit()

    # Create the .inp file from template
    utils.create_from_template(swmm_model_template, mapping, destination)
