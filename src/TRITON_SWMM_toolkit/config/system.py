from pydantic import Field, model_validator
from typing import Optional, Literal
from pathlib import Path
from TRITON_SWMM_toolkit.config.base import cfgBaseModel


class system_config(cfgBaseModel):
    # FILEPATHS
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
    SWMM_software_directory: Optional[Path] = Field(
        None,
        description="Folder containing the SWMM model software.",
    )
    TRITONSWMM_software_directory: Path = Field(
        ...,
        description="Folder containing the TRITONSWMM model software.",
    )
    TRITONSWMM_git_URL: str = Field(
        ...,
        description="Git repository with TRITONSWMM",
    )
    TRITONSWMM_branch_key: Optional[str] = Field(
        None,
        description="TRITONSWMM branch to checkout. Known working branches: 02438b60613a7d913d884e7b836f9f5ff421fe7d",
    )
    SWMM_git_URL: str = Field(
        "https://github.com/USEPA/Stormwater-Management-Model.git",
        description="Git repository with SWMM",
    )
    SWMM_tag_key: Optional[str] = Field(
        "v5.2.4",
        description="SWMM tag to checkout.",
    )
    gpu_compilation_backend: Optional[Literal["HIP", "CUDA"]] = Field(
        None,
        description=(
            "GPU backend for compilation: 'HIP' for AMD GPUs (ROCm), 'CUDA' for NVIDIA GPUs. "
            "If None, only CPU (OPENMP) backend will be compiled. "
            "When set, both CPU and GPU backends are compiled into separate build directories."
        ),
    )
    gpu_hardware: Optional[str] = Field(
        None,
        description=(
            "GPU hardware identifier used to select compilation flags "
            "(e.g., 'a6000', 'rtx3090', 'a100', 'h100'). Required for CUDA "
            "single-arch GPU builds."
        ),
    )
    preferred_slurm_option_for_allocating_gpus: Optional[Literal["gres", "gpus"]] = (
        Field(
            "gpus",
            description=(
                "Preferred SLURM GPU allocation directive. "
                "Set to 'gres' to emit --gres=gpu:..., or 'gpus' to emit "
                "--gpus/--gpus-per-node when supported by the cluster."
            ),
        )
    )
    additional_modules_needed_to_run_TRITON_SWMM_on_hpc: Optional[str] = Field(
        None,
        description="Space separated list of modules to load using 'module load' prior to running each TRITON-SWMM simulatoin, e.g,. 'PrgEnv-amd Core/24.07 craype-accel-amd-gfx90a'",
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
    # TOGGLES
    toggle_use_swmm_for_hydrology: bool = Field(
        ...,
        description="Determines whether a hydrology-only SWMM model will be used for rainfall-runoff calculations.",
    )
    toggle_use_constant_mannings: bool = Field(
        ...,
        description="Determines whether or not to use a constant manning's coefficient.",
    )
    toggle_triton_model: bool = Field(
        ...,
        description="Determines whether or not a TRITON-only model will be compiled and run",
    )
    toggle_tritonswmm_model: bool = Field(
        ...,
        description="Determines whether or not a TRITON-SWMM coupled model will be compiled and run",
    )
    toggle_swmm_model: bool = Field(
        ...,
        description="Determines whether or not a standalone SWMM model will be compiled and run",
    )
    # PARAMETERS
    target_dem_resolution: float = Field(
        ...,
        description="Target DEM resolution for TRITON-SWMM in the native resolution of the provided DEM.",
    )
    constant_mannings: Optional[float] = Field(
        None,
        description="Constant manning's coefficient to use. Only applies if toggle_use_constant_mannings is set to True.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_toggle_dependencies(cls, values):
        errors = []

        _, additional_errors = cls.validate_from_toggle(
            values,
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
        errors.extend(additional_errors)

        _, additional_errors = cls.validate_from_toggle(
            values,
            toggle_varname="toggle_use_swmm_for_hydrology",
            lst_rqrd_if_true=[
                "SWMM_hydrology",
                "subcatchment_raingage_mapping",
                "subcatchment_raingage_mapping_gage_id_colname",
            ],
            lst_rqrd_if_false=[],
        )
        errors.extend(additional_errors)

        _, additional_errors = cls.validate_from_toggle(
            values,
            toggle_varname="toggle_swmm_model",
            lst_rqrd_if_true=["SWMM_full"],
            lst_rqrd_if_false=[],
        )
        errors.extend(additional_errors)

        if errors:
            raise ValueError("; ".join(errors))
        return values
