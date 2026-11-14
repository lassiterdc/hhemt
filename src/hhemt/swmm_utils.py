"""
SWMM Utility Functions

This module contains shared utility functions for creating SWMM .inp files from templates.
These functions are used across multiple SWMM model creation modules to avoid code duplication.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import hhemt.utils as utils

if TYPE_CHECKING:
    from .scenario import TRITONSWMM_scenario

# SWMM copies file names into buffers of this many characters (consts.h:
# `#define MAXFNAME 259 // Max. # characters in file name`). For the two kinds this
# module measures, a longer name is truncated: a time-series FILE name is opened by its
# truncated copy, and the binary output is written under its truncated copy. The input
# and report names are opened from the caller's full strings and are not affected.
_SWMM_MAXFNAME = 259


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
    ProcessingError
        Before anything is written, if any time-series ``FILE`` path the input names
        (one per rain gauge, plus ``water_level`` when the storm-tide boundary is on)
        is over 259 bytes, SWMM's file-name limit. SWMM would open a truncated name and
        fail with an error that names no length.
    ProcessingError
        Before anything is written, if the binary output name SWMM will be handed for
        this input is over 259 bytes, since SWMM would write its results under a
        shortened name. That name is: for ``swmm_hydro_inp``, the destination with
        ``.inp`` replaced by ``.out``; for ``swmm_full_inp``, ``swmm_full_out_file``
        (not checked when it is ``None``); for ``swmm_hydraulics_inp``, when
        ``swmm_hydraulics_rpt`` is set (the coupled model is on) and the run is native,
        ``swmm_hydraulics_rpt`` with a ``.out`` suffix. It is not checked in container
        mode, where TRITON hands SWMM a name inside the image, nor for any other
        destination. Input and report names are never checked.
    ProcessingError
        If template keys are missing from the fill mapping.
    """
    cfg_analysis = scenario._analysis.cfg_analysis
    weather_time_series_timestep_dimension_name = cfg_analysis.weather_time_series_timestep_dimension_name

    ds_event_ts = scenario.ds_event_ts

    # Calculate timestep interval
    tstep_seconds = (
        ds_event_ts[weather_time_series_timestep_dimension_name].to_series().diff().mode().iloc[0].total_seconds()
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
    file_paths = list(fs.items())
    if cfg_analysis.toggle_storm_tide_boundary:
        storm_tide_path = scenario.log.storm_tide_for_swmm.get()
        lst_tseries_section.append(f'water_level FILE "{storm_tide_path}"')
        file_paths.append(("water_level", storm_tide_path))

    # Refuse before writing anything: SWMM truncates a FILE name longer than
    # _SWMM_MAXFNAME and then fails with "ERROR 361: could not open external file",
    # an error that names no length. Length is the byte count SWMM's strlen sees.
    from hhemt.exceptions import ProcessingError

    for key, path in file_paths:
        n_bytes = len(os.fsencode(str(path)))
        if n_bytes > _SWMM_MAXFNAME:
            raise ProcessingError(
                operation="create_swmm_inp_from_template",
                filepath=Path(path),
                reason=(
                    f"SWMM input {destination}: the FILE path for time series {key!r} is {n_bytes} bytes "
                    f"long, over SWMM's {_SWMM_MAXFNAME}-character file-name limit: {path}. SWMM cannot "
                    f"open a file name longer than {_SWMM_MAXFNAME} characters. Move the analysis "
                    f"directory to a shorter path."
                ),
            )

    # The binary output name SWMM is handed for this input is held in a buffer of the
    # same size, and an over-long one is written under a shortened name with no error.
    # The input and report names are opened from the caller's full strings, so they
    # are not measured here.
    scen_paths = scenario.scen_paths
    out_name = None
    if destination == scen_paths.swmm_hydro_inp:
        # pyswmm derives the output name this way when it is given none.
        out_name = str(destination).replace(".inp", ".out")
    elif destination == scen_paths.swmm_full_inp:
        if scen_paths.swmm_full_out_file is not None:
            out_name = str(scen_paths.swmm_full_out_file)
    elif destination == scen_paths.swmm_hydraulics_inp and scen_paths.swmm_hydraulics_rpt is not None:
        from hhemt.config.hpc_system import resolve_container_spec

        # In a container TRITON hands SWMM an output name under the image's install
        # root, not under the host analysis directory, so no host path is measured.
        in_container = (
            cfg_analysis.execution_environment == "container"
            and resolve_container_spec(scenario._analysis.cfg_hpc_system) is not None
        )
        if not in_container:
            out_name = str(scen_paths.swmm_hydraulics_rpt.with_suffix(".out"))
    if out_name is not None:
        n_bytes = len(os.fsencode(out_name))
        if n_bytes > _SWMM_MAXFNAME:
            raise ProcessingError(
                operation="create_swmm_inp_from_template",
                filepath=Path(out_name),
                reason=(
                    f"SWMM input {destination}: the binary output name SWMM will be handed is "
                    f"{n_bytes} bytes long, over SWMM's {_SWMM_MAXFNAME}-character file-name limit: "
                    f"{out_name}. SWMM would write its results under a shortened name. Move the "
                    f"analysis directory to a shorter path."
                ),
            )

    # Build template mapping
    mapping = dict()
    mapping["TIMESERIES"] = "\n\n".join(lst_tseries_section)
    mapping["RAINGAGES"] = "\n".join(rain_gages_section)
    template_keys = utils.find_all_keys_in_template(swmm_model_template)

    # The window is the USER'S, read back rather than derived. Taking min/max of the
    # clipped coordinate would give the same answer today, and that is precisely why it
    # is not done: it is a derivation whose agreement with the declared window is a
    # coincidence of the clip rather than a property anyone stated. It also keeps this
    # value INDEPENDENT of the forcing row count, which Spec 8's three-source guard
    # requires -- an "optimisation" back to a coordinate reduction here silently
    # collapses that guard into a tautology.
    first_tstep, last_tstep = scenario._resolve_event_window()

    # Add timing parameters
    mapping["START_DATE"] = first_tstep.strftime("%m/%d/%Y")
    mapping["START_TIME"] = first_tstep.strftime("%H:%M:%S")
    mapping["REPORT_START_DATE"] = mapping["START_DATE"]
    mapping["REPORT_START_TIME"] = mapping["START_TIME"]
    mapping["END_DATE"] = last_tstep.strftime("%m/%d/%Y")
    mapping["END_TIME"] = last_tstep.strftime("%H:%M:%S")
    mapping["REPORT_STEP"] = scenario.seconds_to_hhmmss(cfg_analysis.TRITON_reporting_timestep_s)

    # Validate all template keys are present
    missing_keys = [key for key in template_keys if key not in mapping.keys()]
    if missing_keys:
        print("One or more keys were not found in the dictionary defining template fill values.")
        print(f"Missing keys: {missing_keys}")
        print(f"All expected keys: {template_keys}")
        print(f"All keys accounted for: {mapping.keys()}")
        # RAISE, never sys.exit() -- and the bare form is the sharp edge: with no
        # argument it exits 0, so a scenario whose SWMM template could not be filled
        # reported SUCCESS to Snakemake and to every completion predicate downstream.
        raise ProcessingError(
            operation="create_swmm_inp_from_template",
            filepath=Path(swmm_model_template),
            reason=(
                f"template keys missing from the fill mapping: {sorted(missing_keys)}. "
                f"Expected: {sorted(template_keys)}."
            ),
        )

    # Create the .inp file from template
    utils.create_from_template(swmm_model_template, mapping, destination)
