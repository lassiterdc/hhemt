import warnings
from pydantic import Field, model_validator
from typing import Optional
from pathlib import Path
from hhemt.config.base import cfgBaseModel


class CRSConfig(cfgBaseModel):
    """Composite horizontal + vertical CRS for the modeled system.

    The horizontal CRS controls map-axis units (Easting/Northing) and
    spatial reprojection. The vertical CRS controls elevation/WSE/depth
    colorbar labels and provides a datum reference for CF-1.13
    `vertical_datum` stamping on consolidated outputs.

    `vertical_epsg` defaults to EPSG:5703 (NAVD88 height, m), which is
    the dominant North American vertical datum. The datum-aware
    `units.dem_elev_label`/`wse_label`/`depth_label` functions consume
    this field to render colorbar suffixes like `"(m, NAVD88)"`.
    Override with EPSG:6360 (NAVD88 ftUS) or EPSG:8228 (NAVD88 ft) when
    the DEM is in non-metric vertical units.
    """

    horizontal_epsg: int = Field(
        ...,
        description=(
            "EPSG code for the horizontal CRS of the DEM. Must be a "
            "projected or geographic CRS. Validated against the DEM "
            "raster's .rio.crs at system processing time."
        ),
    )
    vertical_epsg: int = Field(
        5703,
        description=(
            "EPSG code for the vertical CRS / datum of DEM elevations. "
            "Defaults to 5703 (NAVD88 height, m). Drives elevation/WSE/"
            "depth colorbar labels in renderers and CF-1.13 "
            "vertical_datum stamping on consolidated outputs."
        ),
    )

    @model_validator(mode="after")
    def validate_crs_types(self):
        import pyproj

        horiz = pyproj.CRS.from_epsg(self.horizontal_epsg)
        if not (horiz.is_projected or horiz.is_geographic):
            raise ValueError(
                f"horizontal_epsg {self.horizontal_epsg} is neither projected "
                f"nor geographic; check the EPSG code."
            )
        vert = pyproj.CRS.from_epsg(self.vertical_epsg)
        if not vert.is_vertical:
            raise ValueError(
                f"vertical_epsg {self.vertical_epsg} is not a vertical CRS; "
                f"use EPSG:5703 (NAVD88 m), 6360 (NAVD88 ftUS), or 8228 "
                f"(NAVD88 ft)."
            )
        return self


class system_config(cfgBaseModel):
    """Pydantic model for TRITON-SWMM system configuration.

    Per-sub-analysis variation via prefixed sensitivity-CSV columns
    ----------------------------------------------------------------
    When used with hhemt's sensitivity-analysis workflow,
    every field of this SystemConfig may be varied per sub-analysis via
    sensitivity-CSV/XLSX columns of the form ``system.{field}``. The
    column is mutually exclusive with the ``system_config_yaml`` full-file
    column on a per-row basis. Overlay cells are synthesized into a
    per-``UniqueSystemTarget`` config via Pydantic ``model_validate``
    (re-fires field-level and cross-field validators), then materialized
    to ``{analysis_dir}/_generated/target_{target_id}.yaml`` so runner
    scripts can consume the resolved config via their existing
    ``--system-config`` argument. See the
    ``synthesized per target system yamls under generated`` stipulation
    for the derived-artifact invariant. Coordinated multi-field changes
    (e.g., Manning's-mode toggle + lookup file + colname coupling) should
    still use the ``system_config_yaml`` escape hatch; bare-column
    overlays cannot atomically express multi-field dependencies that
    Pydantic cross-field validators expect.
    """

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
        json_schema_extra={"toolkit_owned_output": True},
        description="Folder containing the SWMM model software (created by the clone/build gate at run/setup).",
    )
    TRITONSWMM_software_directory: Path = Field(
        ...,
        json_schema_extra={"toolkit_owned_output": True},
        description="Folder containing the TRITONSWMM model software (created by the clone/build gate at run/setup).",
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
    # Phase-4 (4c, hpc-system-profile-config): gpu_compilation_backend, gpu_hardware,
    # preferred_slurm_option_for_allocating_gpus, and
    # additional_modules_needed_to_run_TRITON_SWMM_on_hpc were RETIRED off system_config.
    # GPU hardware/backend now live per-partition on PartitionSpec (resolved via
    # config.hpc_system.resolve_gpu_target + constructor-injected into TRITONSWMM_system);
    # the alloc flavor is hpc_system_config.gpu_allocation_flavor; modules are
    # hpc_system_config.additional_modules (joined via resolve_additional_modules).
    # A one-cycle pop-and-warn shim in validate_toggle_dependencies (below) lets
    # un-migrated YAMLs still load. REMOVE the shim after <release>.
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
    processed_xllcorner: Optional[float] = Field(
        None,
        description=(
            "Optional lower-left x-coordinate to anchor processed DEM/Manning outputs "
            "(in DEM CRS units). When set, processed rasters are aligned to this origin."
        ),
    )
    processed_yllcorner: Optional[float] = Field(
        None,
        description=(
            "Optional lower-left y-coordinate to anchor processed DEM/Manning outputs "
            "(in DEM CRS units). When set, processed rasters are aligned to this origin."
        ),
    )
    ncols: Optional[int] = Field(
        None,
        description=(
            "Target number of columns in the processed DEM, mannings, and TRITON results"
        ),
    )
    nrows: Optional[int] = Field(
        None,
        description=(
            "Target number of rows in the processed DEM, mannings, and TRITON results"
        ),
    )
    crs: CRSConfig = Field(
        ...,
        description=(
            "Composite horizontal + vertical CRS for the modeled system. "
            "horizontal_epsg is required; vertical_epsg defaults to 5703 "
            "(NAVD88 m). See CRSConfig for full semantics."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def validate_toggle_dependencies(cls, values):
        # Backward-compat shim (one-cycle, removable after consumers migrate):
        # accept legacy flat `crs_epsg: <int>` and promote into the nested
        # `crs: {horizontal_epsg: <int>}` form. Vertical defaults to 5703.
        # NB: cfgBaseModel uses extra="forbid"; the legacy key must be
        # popped — not merely shadowed — before pydantic validates.
        if isinstance(values, dict) and "crs_epsg" in values and "crs" not in values:
            legacy = values.pop("crs_epsg")
            if legacy is not None:
                values["crs"] = {"horizontal_epsg": int(legacy)}

        # REMOVE after <release>: Phase-4 hpc-system-profile-config retired the four
        # HPC system_config fields to PartitionSpec / hpc_system_config. Pop-and-warn
        # so un-migrated YAMLs still LOAD (extra="forbid" would else reject them).
        _retired_hpc_keys = (
            "gpu_hardware",
            "gpu_compilation_backend",
            "preferred_slurm_option_for_allocating_gpus",
            "additional_modules_needed_to_run_TRITON_SWMM_on_hpc",
        )
        if isinstance(values, dict):
            for _k in _retired_hpc_keys:
                if _k in values:
                    values.pop(_k)
                    warnings.warn(
                        f"system_config field '{_k}' is retired (moved to the "
                        f"per-HPC-system config / PartitionSpec). It is ignored. "
                        f"Remove it from your system config YAML.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
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

        processed_xllcorner = values.get("processed_xllcorner")
        processed_yllcorner = values.get("processed_yllcorner")
        ncols = values.get("ncols")
        nrows = values.get("nrows")
        processed_fields = [processed_xllcorner, processed_yllcorner, ncols, nrows]
        if any(val is not None for val in processed_fields) and not all(
            val is not None for val in processed_fields
        ):
            errors.append(
                "processed_xllcorner, processed_yllcorner, ncols, and nrows must be provided together."
            )

        if errors:
            raise ValueError("; ".join(errors))
        return values
