# src/TRITON_SWMM_toolkit/config_model.py
from pydantic import BaseModel, Field, field_validator, constr
from pathlib import Path
import yaml
import numpy as np
from typing import Literal, Annotated
import re


class system_config(BaseModel):
    # FILEPATHS
    variable_boundary_condition: Path = Field(
        "n/a",
        description="Path to shapefile representing extent of variable water level boundary condition.",
    )
    system_directory: Path = Field(
        "n/a",
        description="Path where TRITON-SWMM system outputs will be stored.",
    )
    watershed_shapefile: Path = Field("n/a", description="Directory to store outputs")
    DEM_fullres: Path = Field(
        "n/a", description="DEM to be formatted and, if desired, coarsened, for TRITON"
    )
    landuse_lookup_file: Path = Field(
        "n/a",
        description="CSV file containing lookup table relating landuse categories to manning's roughness coefficients",
    )
    SWMM_hydraulics: Path = Field(
        "n/a",
        description="Hydraulics-only SWMM model (.inp) template with fillable fields based on input weather data. An event-specific scenario of this model will be input to TRITON-SWMM.",
    )
    SWMM_hydrology: Path = Field(
        "n/a",
        description="Hydrology-only SWMM model (.inp) template with fillable fields based on input weather data. This will be run prior to TRITON-SWMM to generate runoff time series in grid cells that overlap with subcatchment outlet nodes.",
    )
    SWMM_full: Path = Field(
        "n/a",
        description="Full SWMM model (.inp) template with fillable fields based on input weather data. Scenarios based on this can be run in addition to TRITON-SWMM to compare SWMM hydraulics results.",
    )
    landuse_raster: Path = Field(
        "n/a",
        description="Landuse raster used for creating manning's roughness input.",
    )
    TRITONSWMM_software_directory: Path = Field(
        "n/a",
        description="Folder containing the TRITON-SWMM model version used for a particular simulation.",
    )
    TRITON_SWMM_software_compilation_script: Path = Field(
        "n/a",
        description="Folder containing script to build experiment-specific version of TRITON-SWMM.",
    )
    weather_timeseries: Path = Field(
        "n/a",
        description="Netcdf containing weather event time series data. Events must share indices with weather_event_summary_csv.",
    )
    weather_event_summary_csv: Path = Field(
        "n/a",
        description="CSV file with weather event summary statistics. Events must share indices with weather_timeseries.",
    )
    subcatchment_raingage_mapping: Path = Field(
        "n/a",
        description="Lookup table relating spatially indexed rainfall time series to SWMM subcatchment IDs.",
    )
    triton_swmm_configuration_template: Path = Field(
        "n/a",
        description="Path to the template TRITON-SWMM cfg file that defines the variables and inputs per simulation.",
    )
    # ATTRIBUTES
    landuse_description_colname: str = Field(
        "original_description",
        description="column name in the landuse_lookup_file corresponding to landuse description.",
    )
    landuse_lookup_class_id_colname: str = Field(
        "CLASS_ID",
        description="column name in the landuse_lookup_file corresponding to landuse classification.",
    )
    landuse_lookup_mannings_colname: str = Field(
        "mannings",
        description="column name in the landuse_lookup_file corresponding to manning's coefficient.",
    )
    landuse_plot_color_colname: str = Field(
        "plot_color",
        description="column name in the landuse_lookup_file corresponding to target plot colors by landuse.",
    )
    weather_event_indices: list = Field(
        "unspecified",
        description="List of one or more strings corresponding to fields used for indexing unique weather events. These must match what is in weather_timeseries and weather_event_summary_csv.",
    )
    weather_time_series_storm_tide_datavar: str = Field(
        "unspecified",
        description="Data variables in weather_timeseries corresponding to storm tide.",
    )
    weather_time_series_timestep_dimension_name: str = Field(
        "unspecified",
        description="Dimension in weather_timeseries corresponding to timestep.",
    )
    subcatchment_raingage_mapping_gage_id_colname: str = Field(
        "unspecified",
        description="Column name in subcatchment_raingage_mapping_gage corresponding to the rain gage ids.",
    )
    # CONSTANTS
    dem_outside_watershed_height: float = Field(
        np.nan,
        description="DEM height applied to grid cells outside of the watershed boundary.",
    )
    dem_building_height: float = Field(
        np.nan,
        description="DEM height applied to DEM gridcells overlapping buildings.",
    )
    rainfall_units: str = Field(
        "unspecified",
        description="Rainfall units in weather_timeseries, e.g,. mm/hr, mm, in, in/hr. Must align with specifications in SWMM_hydrology model.",
    )
    storm_tide_units: str = Field(
        "unspecified",
        description="Storm tide units, e.g., ft, m. Must align with units used DEM.",
    )
    # PARAMETERS
    target_dem_resolution: float = Field(
        np.nan,
        description="Target DEM resolution for TRITON-SWMM in the native resolution of the provided DEM.",
    )
    TRITON_output_type: Literal["bin", "asc"] = Field(
        "bin",
        description="TRITON output type, asc or bin.",
    )
    use_constant_mannings: bool = Field(
        False,
        description="Whether or not to use a constant manning's coefficient.",
    )
    constant_mannings: float = Field(
        -9999,
        description="Constant manning's coefficient to use. Only applies if use_constant_mannings is set to True.",
    )
    manhole_diameter: float = Field(
        1.2,
        description="Manhole diameter of TRITON-SWMM interaction nodes.",
    )
    manhole_loss_coefficient: float = Field(
        0.1,
        description="Loss coefficient of TRITON-SWMM interactions occuring at manholes.",
    )
    hydraulic_timestep_s: float = Field(
        0.01,
        description="Timestep for hydraulic computations in seconds.",
    )
    TRITON_reporting_timestep_s: float = Field(
        120,
        description="Reporting timestep in seconds.",
    )
    open_boundaries: int = Field(
        1,
        description="0 for closed, 1 for open. This is affects all boundaries wherever external boundary conditions are not otherwise defined.",
    )


class experiment_config(BaseModel):
    experiment_id: Annotated[
        str,
        Field(
            default="default_experiment_name",
            description="Experiment identifier",
            pattern=r"^[A-Za-z][A-Za-z0-9_.]*$",
        ),
    ]
    benchmarking_experiment: Path = Field(
        "n/a",
        description="Benchmarking experimental design.",
    )
    weather_event_indexers: dict = Field(
        "n/a",
        description="Dictionary defining weather event index used for benchmarking. The keys must correspond to the sytem's weather_event_indices.",
    )
    event_description: str = Field(
        "n/a",
        description="Description of event used for benchmarking.",
    )
    experiment_folder: Path = Field(
        "n/a",
        description="Folder in which simulation scenarios will be generated and run.",
    )
    TRITON_SWMM_make_command: Path = Field(
        "hpc_swmm_omp",
        description="This should be one of the make commands listed in Makefile in the TRITONSWMM software directory.",
    )

    @field_validator("experiment_id")
    def validate_experiment_id(cls, v):
        if not re.match(r"^[A-Za-z][A-Za-z0-9_.]*$", v):
            raise ValueError(
                "experiment_id must start with a letter and contain only letters, digits, underscores, or periods"
            )
        return v


def load_system_config(cfg):
    cfg = system_config.model_validate(cfg)
    return cfg


def load_experiment_config(cfg):
    cfg = experiment_config.model_validate(cfg)
    return cfg


# def load_benchmarking_experiment_config_config(cfg):
#     cfg = benchmarking_experiment_config.model_validate(cfg)
#     return cfg
