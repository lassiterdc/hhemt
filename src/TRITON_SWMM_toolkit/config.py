# src/TRITON_SWMM_toolkit/config_model.py
# %%
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    constr,
    ValidationError,
    model_validator,
)
from typing import ClassVar, List, Dict
from pathlib import Path
import yaml
import numpy as np
from typing import Literal, Annotated
import re
import pandas as pd
from tabulate import tabulate
import pandas as pd
from typing import Optional


class cfgBaseModel(BaseModel):
    toggle_tests: ClassVar[List[Dict]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.toggle_tests = []  # fresh list for each subclass

    def __init__(self, **data):
        try:
            super().__init__(**data)
            get_tests = getattr(self, "get_toggle_tests", None)
            if callable(get_tests):
                get_tests()
        except ValidationError as e:
            # Extract field errors and messages
            messages = []
            for err in e.errors():
                loc = ".".join(str(l) for l in err["loc"])
                msg = err["msg"]
                messages.append(f"{loc}: {msg}")
            # Print clean message
            print("\n=== Validation Error ===")
            for m in messages:
                print(f"- {m}")
            print("========================\n")
            # Prevent full traceback
            raise

    @staticmethod
    def _get_field_descriptions(model_cls):
        data = {
            field_name: field_info.description or ""
            for field_name, field_info in model_cls.model_fields.items()
        }
        sr = pd.Series(data)
        sr.index.name = "attr_name"  # type: ignore
        sr.name = "desc"  # type: ignore
        return sr

    @staticmethod
    def _get_field_optionality(model_cls):
        """
        Returns a Series with field names as index and True/False for optionality
        """
        data = {}
        for name, field in model_cls.model_fields.items():
            is_optional = field.default is not ... or field.allow_none  # type: ignore
            data[name] = is_optional
        sr = pd.Series(data)
        sr.index.name = "attr_name"  # type: ignore
        sr.name = "optional"  # type: ignore
        return sr

    def cfg_dic_to_df(self):
        s_vals = pd.DataFrame(self, columns=["attr_name", "val"]).set_index(
            "attr_name"
        )["val"]
        s_descs = self._get_field_descriptions(self.__class__)
        df_vars = pd.concat([s_descs, s_vals], axis=1)
        return df_vars

    def display_tabulate_cfg(self, col1_width=25, col2_width=50, col3_width=50):
        data = self.cfg_dic_to_df()

        lst_rows = []
        for idx, row in data.iterrows():
            vals_as_list = [
                str(idx),
                str(row.desc),
                (  # even coerced as strings, True and False cause line splitting to fail so they need to be modified
                    str(row.val).lower()
                    if str(row.val) in ["True", "False"]
                    else str(row.val)
                ),
            ]
            lst_rows.append(vals_as_list)

        print(
            tabulate(
                lst_rows,  # type: ignore
                headers=[str(data.index.name)] + list(data.columns),  # type: ignore
                tablefmt="grid",
                maxcolwidths=[25, 60, 60],
            )
        )

    # VALIDATION
    @staticmethod
    def validate_from_toggle(
        values, toggle_varname, lst_rqrd_if_true, lst_rqrd_if_false
    ):
        failing_vars = []
        errors = []
        toggle = values.get(toggle_varname)
        if toggle:
            for var in lst_rqrd_if_true:
                # print(f"testing {var}")
                if values.get(var) is None:
                    errors.append(f"{var} must be provided if {toggle_varname} is True")
                    failing_vars.append(var)
        else:
            for var in lst_rqrd_if_false:
                # print(f"testing {var}")
                if values.get(var) is None:
                    errors.append(f"{var} must be provided if {toggle_varname} is True")
                    failing_vars.append(var)
        return failing_vars, errors

    @classmethod
    def append_errors_and_failing_vars(
        cls,
        values,
        failing_vars,
        errors,
        toggle_varname,
        lst_rqrd_if_true,
        lst_rqrd_if_false,
    ):
        additional_failing_vars, additional_errors = cls.validate_from_toggle(
            values, toggle_varname, lst_rqrd_if_true, lst_rqrd_if_false
        )
        failing_vars.extend(additional_failing_vars)
        errors.extend(additional_errors)
        return failing_vars, errors

    @model_validator(mode="before")
    def validate_toggle_dependencies(cls, values):
        """
        Validates that all fields whose dependency is determiend by toggles.
        """
        toggle_tests = cls.toggle_tests
        # print(f"validating using toggle tests: {toggle_tests}")
        errors = []
        failing_vars = []
        for test in toggle_tests:
            failing_vars, errors = cls.append_errors_and_failing_vars(
                values, failing_vars, errors, **test
            )
        ############
        if len(errors) > 0:
            # print(errors)
            raise ValueError("; ".join(errors))
        return values


class system_config(cfgBaseModel):
    # FILEPATHS
    storm_tide_boundary_line_gis: Optional[Path] = Field(
        None,
        description="Path to a line gis file spanning the extent of the dem boundary where the variable storm tide boundary condition should be applied.",
    )
    system_directory: Path = Field(
        ...,
        description="Path where TRITON-SWMM system outputs will be stored.",
    )
    watershed_gis_polygon: Path = Field(
        ..., description="Watershed or subcatchment gis used for plotting."
    )
    DEM_fullres: Path = Field(
        ..., description="DEM to be formatted and, if desired, coarsened, for TRITON"
    )
    landuse_lookup_file: Optional[Path] = Field(
        None,
        description="CSV file containing lookup table relating landuse categories to manning's roughness coefficients",
    )
    SWMM_hydraulics: Path = Field(
        ...,
        description="Hydraulics-only SWMM model (.inp) template with fillable fields based on input weather data. An event-specific scenario of this model will be input to TRITON-SWMM.",
    )
    SWMM_hydrology: Optional[Path] = Field(
        None,
        description="Hydrology-only SWMM model (.inp) template with fillable fields based on input weather data. This will be run prior to TRITON-SWMM to generate runoff time series in grid cells that overlap with subcatchment outlet nodes.",
    )
    SWMM_full: Optional[Path] = Field(
        None,
        description="Full SWMM model (.inp) template with fillable fields based on input weather data. Scenarios based on this can be run in addition to TRITON-SWMM to compare SWMM hydraulics results.",
    )
    landuse_raster: Optional[Path] = Field(
        None,
        description="Landuse raster used for creating manning's roughness input.",
    )
    TRITONSWMM_software_directory: Path = Field(
        ...,
        description="Folder containing the TRITON-SWMM model version used for a particular simulation.",
    )
    TRITON_SWMM_software_compilation_script: Path = Field(
        ...,
        description="Folder containing script to build experiment-specific version of TRITON-SWMM.",
    )
    weather_timeseries: Path = Field(
        ...,
        description="Netcdf containing weather event time series data. Events must share indices with weather_event_summary_csv.",
    )
    weather_event_summary_csv: Optional[Path] = Field(
        None,
        description="CSV file with weather event summary statistics. Events must share indices with weather_timeseries.",
    )
    subcatchment_raingage_mapping: Optional[Path] = Field(
        None,
        description="Lookup table relating spatially indexed rainfall time series to SWMM subcatchment IDs.",
    )
    triton_swmm_configuration_template: Path = Field(
        ...,
        description="Path to the template TRITON-SWMM cfg file that defines the variables and inputs per simulation.",
    )
    # ATTRIBUTES
    landuse_description_colname: Optional[str] = Field(
        None,
        description="column name in the landuse_lookup_file corresponding to landuse description.",
    )
    landuse_lookup_class_id_colname: Optional[str] = Field(
        None,
        description="column name in the landuse_lookup_file corresponding to landuse classification.",
    )
    landuse_lookup_mannings_colname: Optional[str] = Field(
        None,
        description="column name in the landuse_lookup_file corresponding to manning's coefficient.",
    )
    landuse_plot_color_colname: Optional[str] = Field(
        None,
        description="column name in the landuse_lookup_file corresponding to target plot colors by landuse.",
    )
    weather_event_indices: list = Field(
        ...,
        description="List of one or more strings corresponding to fields used for indexing unique weather events. These must match what is in weather_timeseries and weather_event_summary_csv.",
    )
    weather_time_series_storm_tide_datavar: Optional[str] = Field(
        None,
        description="Data variables in weather_timeseries corresponding to storm tide.",
    )
    weather_time_series_timestep_dimension_name: str = Field(
        ...,
        description="Dimension in weather_timeseries corresponding to timestep.",
    )
    subcatchment_raingage_mapping_gage_id_colname: Optional[str] = Field(
        None,
        description="Column name in subcatchment_raingage_mapping_gage corresponding to the rain gage ids.",
    )
    # CONSTANTS
    dem_outside_watershed_height: Optional[float] = Field(
        None,
        description="DEM height applied to grid cells outside of the watershed boundary. Used for scaling DEM plot colorbars.",
    )
    dem_building_height: Optional[float] = Field(
        None,
        description="DEM height applied to DEM gridcells overlapping buildings. Used for scaling DEM plot colorbars.",
    )
    rainfall_units: str = Field(
        ...,
        description="Rainfall units in weather_timeseries, e.g,. mm/hr, mm, in, in/hr. Must align with specifications in SWMM_hydrology model.",
    )
    storm_tide_units: Optional[str] = Field(
        None,
        description="Storm tide units, e.g., ft, m. Must align with units used DEM.",
    )
    # TOGGLES
    toggle_use_swmm_for_hydrology: bool = Field(
        ...,
        description="Determines whether a hydrology-only SWMM model will be used for rainfall-runoff calculations.",
    )
    toggle_storm_tide_boundary: bool = Field(
        ...,
        description="If True, a boundary condition representing storm tide will be applied to the model.",
    )
    toggle_use_constant_mannings: bool = Field(
        ...,
        description="Determines whether or not to use a constant manning's coefficient.",
    )
    toggle_full_swmm_model: bool = Field(
        ...,
        description="Determines whether or not a basic SWMM model will be run",
    )
    # PARAMETERS
    target_dem_resolution: float = Field(
        ...,
        description="Target DEM resolution for TRITON-SWMM in the native resolution of the provided DEM.",
    )
    TRITON_output_type: Literal["bin", "asc"] = Field(
        "bin",
        description="TRITON output type, asc or bin.",
    )
    constant_mannings: Optional[float] = Field(
        None,
        description="Constant manning's coefficient to use. Only applies if toggle_use_constant_mannings is set to True.",
    )
    manhole_diameter: float = Field(
        ...,
        description="Manhole diameter of TRITON-SWMM interaction nodes.",
    )
    manhole_loss_coefficient: float = Field(
        ...,
        description="Loss coefficient of TRITON-SWMM interactions occuring at manholes.",
    )
    hydraulic_timestep_s: float = Field(
        ...,
        description="Timestep for hydraulic computations in seconds.",
    )
    TRITON_reporting_timestep_s: float = Field(
        ...,
        description="Reporting timestep in seconds.",
    )
    open_boundaries: int = Field(
        ...,
        description="0 for closed, 1 for open. This is affects all boundaries wherever external boundary conditions are not otherwise defined.",
    )

    # VALIDATING DEPENDENCIES BASED ON TOGGLES
    @classmethod
    def get_toggle_tests(cls):
        ### toggle_use_constant_mannings
        mannings_test = dict(
            toggle_varname="toggle_use_constant_mannings",
            lst_rqrd_if_true=["constant_mannings"],
            lst_rqrd_if_false=[
                "landuse_lookup_file",
                "landuse_raster",
                "landuse_description_colname",
                "landuse_lookup_class_id_colname",
                "landuse_lookup_mannings_colname",
            ],
        )
        cls.toggle_tests.append(mannings_test)
        ### toggle_use_swmm_for_hydrology
        swmm_hydro_test = dict(
            toggle_varname="toggle_use_swmm_for_hydrology",
            lst_rqrd_if_true=[
                "SWMM_hydrology",
                "subcatchment_raingage_mapping",
                "subcatchment_raingage_mapping_gage_id_colname",
            ],
            lst_rqrd_if_false=[""],
        )
        cls.toggle_tests.append(swmm_hydro_test)
        ### toggle_storm_tide_boundary
        storm_tide_boundary_test = dict(
            toggle_varname="toggle_storm_tide_boundary",
            lst_rqrd_if_true=[
                "storm_tide_boundary_line_gis",
                "weather_time_series_storm_tide_datavar",
                "storm_tide_units",
            ],
            lst_rqrd_if_false=[""],
        )
        cls.toggle_tests.append(storm_tide_boundary_test)
        ### toggle_full_swmm_model
        full_swmm_model_test = dict(
            toggle_varname="toggle_full_swmm_model",
            lst_rqrd_if_true=["SWMM_full"],
            lst_rqrd_if_false=[""],
        )
        cls.toggle_tests.append(full_swmm_model_test)
        return


class experiment_config(cfgBaseModel):
    # REQUIRED PARAMETERS
    experiment_id: Annotated[
        str,
        Field(
            ...,
            description="Experiment identifier. Used for creating experiment folder if one with the same name does not exist.",
            pattern=r"^[A-Za-z][A-Za-z0-9_.]*$",
        ),
    ]
    # TOGGLES
    toggle_benchmarking_experiment: bool = Field(
        ...,
        description="Whether or not this is a benchmarking study. If so, a .csv file is required for input benchmarking_experiment defining the experimental setup.",
    )
    # OPTIONAL OR DEPENDENT PARAMETERS
    benchmarking_experiment: Optional[Path] = Field(
        None,
        description="Benchmarking experimental design csv file.",
    )
    weather_events_to_simulate: Path = Field(
        ...,
        description="Path to a .csv file defining weather event index used for benchmarking. The columns must correspond to the sytem's weather_event_indices.",
    )
    experiment_description: Optional[str] = Field(
        None,
        description="For readability.",
    )
    TRITON_SWMM_make_command: Path = Field(
        "hpc_swmm_omp",
        description="This should be one of the make commands listed in Makefile in the TRITONSWMM software directory.",
    )

    # VALIDATION - STRING REQUIREMENTS
    @field_validator("experiment_id")
    def validate_experiment_id(cls, v):
        if not re.match(r"^[A-Za-z][A-Za-z0-9_.]*$", v):
            raise ValueError(
                "experiment_id must start with a letter and contain only letters, digits, underscores, or periods"
            )
        return v

    # VALIDATING DEPENDENCIES BASED ON TOGGLES
    @classmethod
    def get_toggle_tests(cls):
        ### toggle_use_constant_mannings
        bm_test = dict(
            toggle_varname="toggle_use_constant_mannings",
            lst_rqrd_if_true=["benchmarking_experiment"],
            lst_rqrd_if_false=[],
        )
        cls.toggle_tests.append(bm_test)


def load_system_config(cfg):
    cfg = system_config.model_validate(cfg)
    return cfg


def load_experiment_config(cfg):
    cfg = experiment_config.model_validate(cfg)
    return cfg


# def load_benchmarking_experiment_config_config(cfg):
#     cfg = benchmarking_experiment_config.model_validate(cfg)
#     return cfg
