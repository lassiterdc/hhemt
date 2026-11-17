import warnings
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from hhemt.config.base import cfgBaseModel, field_meta, when


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
            "EPSG code for the vertical CRS or datum of DEM elevations. Defaults to 5703, NAVD88 height in "
            "m. Drives elevation, WSE, and depth colorbar labels in renderers and CF-1.13 vertical_datum "
            "stamping on consolidated outputs."
        ),
    )

    @model_validator(mode="after")
    def validate_crs_types(self):
        import pyproj

        horiz = pyproj.CRS.from_epsg(self.horizontal_epsg)
        if not (horiz.is_projected or horiz.is_geographic):
            raise ValueError(
                f"horizontal_epsg {self.horizontal_epsg} is neither projected nor geographic; check the EPSG code."
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

    Per-member variation via prefixed sensitivity-CSV columns
    ----------------------------------------------------------------
    When used with hhemt's sensitivity-analysis workflow,
    every field of this SystemConfig may be varied per member via
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
        # A toolkit-owned OUTPUT: from_case_study mkdirs it AFTER the pre-download
        # validation, and for the shipped Norfolk case it is ${PACKAGE_DIR}/system,
        # ignored by .gitignore:243 and absent on a fresh clone. Exempt from the
        # load-time existence check for the same reason the two software dirs are.
        json_schema_extra={"toolkit_owned_output": True},
    )
    watershed_gis_polygon: Path = Field(..., description="Watershed or subcatchment gis used for plotting.")
    DEM_fullres: Path = Field(..., description="DEM to be formatted and, if desired, coarsened, for TRITON")
    observation_points: Path | None = Field(
        None,
        description="CSV of points where TRITON records depth and unit discharge through time. The header is "
        "`id,x,y`. Each row holds a unique string `id` and the point's `x` and `y` in the system's horizontal "
        "CRS, the same coordinates as the DEM. Every point must lie inside the processed DEM and on a cell that "
        "TRITON computes on, so not on the outside-watershed or building fill height. A point that fails is "
        "refused by the `id` of that point. The row order identifies the points in the results. Unset, TRITON "
        "records no observation series. The series are consolidated into the analysis zarr and into no separate "
        "store.",
    )
    landuse_lookup_file: Path | None = Field(
        None,
        description="CSV file containing lookup table relating landuse categories to manning's roughness coefficients",
        json_schema_extra=field_meta(required_when=[when("toggle_use_constant_mannings", False)]),
    )
    SWMM_model: Path = Field(
        ...,
        description="A plain runnable SWMM 5.2 model, a .inp file, in CMS flow units, with no placeholders. The "
        "toolkit derives the coupled hydraulics model, the hydrology model, and the standalone SWMM model from "
        "the file. For each event the toolkit rewrites the simulation window and the rain gauges. The toolkit "
        "also rewrites the rain time series and the water-level time series.",
    )
    swmm_outfalls_to_couple: list[str] = Field(
        default_factory=list,
        description="Names of [OUTFALLS] rows of SWMM_model to convert into junctions that exchange water with the "
        "TRITON surface, for free, normal, or fixed outfalls that no external-boundary water-level record names. "
        "An outfall a water-level record names is converted by that record and must not be listed here.",
    )
    swmm_coupling_mode: Literal["all_junctions", "list", "legacy"] = Field(
        "all_junctions",
        description=(
            "The SWMM nodes where the coupled TRITON-SWMM model may exchange water. The toolkit always "
            "gives TRITON an explicit list that the toolkit writes for each scenario, and this setting "
            "chooses the candidates that the list is selected from. The default, all_junctions, couples "
            "every junction that has a [COORDINATES] row. The list choice takes the nodes named in "
            "swmm_coupling_nodes_file. The third choice selects every node that carries a FLOW [INFLOWS] "
            "line."
        ),
        json_schema_extra=field_meta(
            applies_when=[when("toggle_tritonswmm_model", True)],
            options={
                "all_junctions": (
                    "The default. Every [JUNCTIONS] node of the hydraulics role of SWMM_model with a [COORDINATES] row."
                ),
                "list": (
                    "The nodes named in swmm_coupling_nodes_file. Each node must be a junction with a "
                    "[COORDINATES] row."
                ),
                "legacy": (
                    "An explicit opt-in that selects every node of the hydraulics role of SWMM_model that carries an "
                    "[INFLOWS] line with constituent FLOW, read by SWMM's own rule."
                ),
            },
        ),
    )
    swmm_coupling_nodes_file: Path | None = Field(
        None,
        description=(
            "A one-column file of coupling-node names, read when swmm_coupling_mode is 'list'. The file "
            "holds one name per line, with surrounding whitespace stripped, blank lines skipped, lines "
            "starting with `;` skipped, and no header."
        ),
        json_schema_extra=field_meta(
            applies_when=[when("swmm_coupling_mode", "list")],
            required_when=[when("swmm_coupling_mode", "list")],
        ),
    )
    landuse_raster: Path | None = Field(
        None,
        description="Landuse raster used for creating manning's roughness input.",
        json_schema_extra=field_meta(required_when=[when("toggle_use_constant_mannings", False)]),
    )
    SWMM_software_directory: Path | None = Field(
        None,
        json_schema_extra={"toolkit_owned_output": True, "toolkit_clones_into": True},
        description="Folder containing the SWMM model software, created by the clone/build gate at run/setup.",
    )
    # Optional[Path] (not required) ONLY so a portability-scrubbed render bundle's
    # cfg_system.yaml (which nulls this per bundle/_path_policy.py IS_NONE_ACCEPTABLE)
    # loads for bundle-local EDA/render; requiredness is enforced at the
    # TRITONSWMM_system constructor chokepoint, not at load time. Mirrors the
    # SWMM_software_directory sibling above.
    TRITONSWMM_software_directory: Path | None = Field(
        None,
        json_schema_extra={"toolkit_owned_output": True, "toolkit_clones_into": True},
        description="Folder containing the TRITONSWMM model software, created by the clone/build gate at "
        "run/setup. May be null in the synthesized system_config.yaml of a reconstituted "
        "reprex bundle, because the round-trip re-derives the value from the by-reference "
        "software.",
    )
    TRITONSWMM_git_URL: str = Field(
        ...,
        description="Git repository with TRITONSWMM",
    )
    TRITONSWMM_branch_key: str | None = Field(
        None,
        description=(
            "Git ref the TRITONSWMM clone is pinned to. In native mode the value may be a branch, a tag, "
            "or a commit sha. In container mode the value MUST be a full commit sha, because a container "
            "skips the compile and no clone exists to resolve a branch name against. The pin is re-checked "
            "on every run, and a run is refused when an existing clone sits at a different commit. The "
            "check is the only protection against stale numerics, because an existing clone is never "
            "re-checked-out and the cached build of the clone is never invalidated. Leave unset to build "
            "whatever the default branch of the clone resolves to."
        ),
    )
    TRITONSWMM_extra_c_flags: str = Field(
        default="",
        description=(
            "Extra C compiler flags for the coupled TRITON-SWMM build, appended after `-fopenmp` in the "
            "`-DCMAKE_C_FLAGS` argument of each of the 3 compile commands (CPU, CUDA, and the other GPU "
            "backend). The flags reach the SWMM C library that the TRITON-SWMM build compiles. The TRITON-only "
            "compile ignores the field, because the C-flag site of that compile is never used. The default empty "
            "string appends no flag, so the compile script of a system that leaves the field unset is "
            "byte-identical to a script rendered before the field existed. "
            "The value is part of the identity of the coupled compile, so a change of flags rebuilds the solver. "
            "Optimisation flags such as `-O3` can change floating-point results, so compare the output with the "
            "output of a build without the flags before you trust a run that uses them. Applies only to a native "
            "build. Preflight refuses a non-empty value under the container execution environment, where the SIF "
            "supplies the binary and no compile runs, and refuses a non-empty value when toggle_tritonswmm_model "
            "is off. The toolkit splices the text into a single-quoted shell argument, so load refuses a single "
            "quote or a line break."
        ),
        json_schema_extra=field_meta(applies_when=[when("toggle_tritonswmm_model", True)]),
    )
    SWMM_git_URL: str = Field(
        "https://github.com/USEPA/Stormwater-Management-Model.git",
        description="Git repository with SWMM",
    )
    SWMM_tag_key: str | None = Field(
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
    subcatchment_raingage_mapping: Path | None = Field(
        None,
        description="Lookup table relating spatially indexed rainfall time series to SWMM subcatchment IDs.",
        json_schema_extra=field_meta(required_when=[when("toggle_use_swmm_for_hydrology", True)]),
    )
    # ATTRIBUTES
    landuse_lookup_class_id_colname: str | None = Field(
        None,
        description="column name in the landuse_lookup_file corresponding to landuse classification.",
        json_schema_extra=field_meta(required_when=[when("toggle_use_constant_mannings", False)]),
    )
    landuse_lookup_mannings_colname: str | None = Field(
        None,
        description="column name in the landuse_lookup_file corresponding to manning's coefficient.",
        json_schema_extra=field_meta(required_when=[when("toggle_use_constant_mannings", False)]),
    )
    subcatchment_raingage_mapping_gage_id_colname: str | None = Field(
        None,
        description="Column name in subcatchment_raingage_mapping_gage corresponding to the rain gage ids.",
        json_schema_extra=field_meta(required_when=[when("toggle_use_swmm_for_hydrology", True)]),
    )
    # CONSTANTS
    dem_outside_watershed_height: float | None = Field(
        None,
        description="Elevation burned into the DEM outside the watershed boundary. Report figures never "
        "test cells against the threshold directly. Cells at or above the smaller of this "
        "value and dem_building_height, less wall_threshold_buffer_m, are painted as walls. "
        "When dem_building_height is unset, this value acts alone. A value below the real "
        "terrain of the raster paints terrain as wall and reports no error. Leave unset when "
        "the DEM carries no such sentinel.",
    )
    dem_building_height: float | None = Field(
        None,
        description="Elevation burned into DEM cells overlapping buildings. Report figures drop cells "
        "whose value equals this number from the DEM colorbar range, and paint cells at or "
        "above the smaller of this value and dem_outside_watershed_height, less "
        "wall_threshold_buffer_m, as walls. When dem_outside_watershed_height is unset, this "
        "value acts alone. A value that mismatches the raster leaves rooftops in the colorbar "
        "and reports no error. Leave unset when the DEM carries no such sentinel.",
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
        description=(
            "Determines whether a TRITON-only model is enabled. The build is not performed here. In native "
            "mode the setup rule asserts that an enabled model already has a successful build."
        ),
    )
    toggle_tritonswmm_model: bool = Field(
        ...,
        description=(
            "Determines whether a TRITON-SWMM coupled model is enabled. The build is not performed here. "
            "In native mode the setup rule asserts that an enabled model already has a successful build."
        ),
    )
    toggle_swmm_model: bool = Field(
        ...,
        description=(
            "Determines whether a standalone SWMM model is enabled. The build is not performed here. In "
            "native mode the setup rule asserts that an enabled model already has a successful build."
        ),
    )
    # PARAMETERS
    target_dem_resolution: float = Field(
        ...,
        description="Target DEM resolution for TRITON-SWMM in the native resolution of the provided DEM.",
    )
    constant_mannings: float | None = Field(
        None,
        description="Constant manning's coefficient to use. Only applies if toggle_use_constant_mannings is set to "
        "True.",
        json_schema_extra=field_meta(required_when=[when("toggle_use_constant_mannings", True)]),
    )
    processed_xllcorner: float | None = Field(
        None,
        description=(
            "Optional lower-left x-coordinate to anchor processed DEM/Manning outputs, in DEM CRS units. "
            "When set, processed rasters are aligned to this origin."
        ),
    )
    processed_yllcorner: float | None = Field(
        None,
        description=(
            "Optional lower-left y-coordinate to anchor processed DEM/Manning outputs, in DEM CRS units. "
            "When set, processed rasters are aligned to this origin."
        ),
    )
    ncols: int | None = Field(
        None,
        description=("Target number of columns in the processed DEM, mannings, and TRITON results"),
    )
    nrows: int | None = Field(
        None,
        description=("Target number of rows in the processed DEM, mannings, and TRITON results"),
    )
    crs: CRSConfig = Field(
        ...,
        description=(
            "Composite horizontal + vertical CRS for the modeled system. horizontal_epsg is required, and "
            "vertical_epsg defaults to 5703, NAVD88 m. See CRSConfig for full semantics."
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
        _retired_template_keys = {
            "SWMM_hydraulics": "SWMM_model",
            "SWMM_hydrology": "SWMM_model",
            "SWMM_full": "SWMM_model",
            "triton_swmm_configuration_template": "no config field: the toolkit renders the TRITON cfg itself",
        }
        if isinstance(values, dict):
            for _k, _use in _retired_template_keys.items():
                if _k in values:
                    raise ValueError(
                        f"'{_k}' is not a system config field: SWMM and TRITON templates are no longer used. "
                        f"Replacement: {_use}. For the SWMM models set SWMM_model to one plain runnable .inp and "
                        f"list the outfalls to convert in swmm_outfalls_to_couple."
                    )
        errors = []

        # Conditional requiredness for these ten fields is DECLARED on the fields
        # themselves via `field_meta(required_when=...)` and enforced generically by
        # `cfgBaseModel._enforce_required_when`. The hand-written `validate_from_toggle`
        # calls that used to live here were a SECOND enforcement site the Reproduction
        # Guide's Required cell could not see, so all ten rendered as "Optional" while
        # the config refused to load without them. Do not reintroduce them; see the
        # stipulation "config field descriptions options and conditional requiredness
        # are declared once on the field". Path existence is unaffected -- the base
        # `_check_paths_exist` field validator already covers every Path field.

        processed_xllcorner = values.get("processed_xllcorner")
        processed_yllcorner = values.get("processed_yllcorner")
        ncols = values.get("ncols")
        nrows = values.get("nrows")
        processed_fields = [processed_xllcorner, processed_yllcorner, ncols, nrows]
        if any(val is not None for val in processed_fields) and not all(val is not None for val in processed_fields):
            errors.append("processed_xllcorner, processed_yllcorner, ncols, and nrows must be provided together.")

        if errors:
            raise ValueError("; ".join(errors))
        return values

    @field_validator("TRITONSWMM_extra_c_flags", mode="after")
    @classmethod
    def _validate_extra_c_flags(cls, v):
        """Strip the flags, and refuse a single quote or a control character.

        The compile appends the text inside the single-quoted ``-DCMAKE_C_FLAGS='-fopenmp ...'`` argument of a
        generated shell script. A quote would close that argument early, and the rest of the string would run as
        shell text; a line break would end the command line. Refusing is deliberately NOT escaping: an escaped
        quote reaches the compiler as part of a flag, which is not what the user meant either.
        """
        value = v.strip()
        bad = sorted({c for c in value if c == "'" or ord(c) < 32 or ord(c) == 127})
        if bad:
            raise ValueError(
                f"TRITONSWMM_extra_c_flags must not contain a single quote or a control character; found "
                f"{[repr(c) for c in bad]}. The text is spliced into a single-quoted shell argument."
            )
        return value

    @model_validator(mode="after")
    def validate_coupling_declaration(self):
        """Refuse a coupling-node file given under a mode that does not read it.

        The file is read only under ``swmm_coupling_mode: list``. A file set under ``legacy``
        or ``all_junctions`` would be silently ignored, so a user who meant to restrict the
        coupled set would run with a different one and no error. The opposite case, ``list``
        with no file, is NOT checked here: it is the ``required_when`` declaration on
        ``swmm_coupling_nodes_file``, enforced by ``cfgBaseModel._enforce_required_when``, which
        is the one place a conditional requirement is enforced. A second check here would be a
        second enforcement site that the rendered "Required" cell cannot see.
        """
        if self.swmm_coupling_nodes_file is not None and self.swmm_coupling_mode != "list":
            raise ValueError(
                f"system_config.swmm_coupling_nodes_file is set ({self.swmm_coupling_nodes_file}) but "
                f"swmm_coupling_mode is '{self.swmm_coupling_mode}'; the file is read only when "
                "swmm_coupling_mode is 'list'. Set swmm_coupling_mode: list to use the file, or remove "
                "swmm_coupling_nodes_file."
            )
        return self

    @model_validator(mode="after")
    def validate_swmm_model_is_si(self):
        """Refuse, at config load, a SWMM_model whose FLOW_UNITS the toolkit cannot read correctly.

        The toolkit reads the coupled report's and the standalone output's flows as m3/s with no conversion,
        so SWMM_model must use CMS. The unit is read by `swmm_runoff_modeling.flow_units_from_inp_text` (the
        last FLOW_UNITS line, matched by prefix as SWMM matches it). A file with no [OPTION... section is
        skipped rather than judged, and an unreadable file is skipped: the runtime guard backstops both.
        """
        import re as _re

        from hhemt.swmm_runoff_modeling import flow_units_from_inp_text

        _text = None
        for _enc in ("utf-8", "latin-1"):
            try:
                _text = Path(self.SWMM_model).read_text(encoding=_enc)
                break
            except (OSError, UnicodeDecodeError):
                continue
        if _text is None or not _re.search(r"^\s*\[OPTION", _text, _re.IGNORECASE | _re.MULTILINE):
            return self
        _found = flow_units_from_inp_text(_text, default=None)
        if _found is None:
            _why = "declares no FLOW_UNITS; SWMM defaults to CFS (US)"
        elif _found != "CMS":
            _why = f"declares FLOW_UNITS {_found}"
        else:
            return self
        raise ValueError(
            f"SWMM_model ({self.SWMM_model}) {_why}; it must use CMS, because the toolkit reads its flows as "
            "m3/s without conversion."
        )
