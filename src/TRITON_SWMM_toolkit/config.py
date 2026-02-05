from pydantic import (
    BaseModel,
    Field,
    field_validator,
    ValidationError,
    model_validator,
)
from typing import ClassVar, List, Dict
from pathlib import Path
import yaml
from typing import Literal, Annotated, Any, Optional, Tuple
import re
import pandas as pd
from tabulate import tabulate
from TRITON_SWMM_toolkit.plot_utils import print_json_file_tree


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
        sr = pd.Series(data)  # type: ignore
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
        sr = pd.Series(data)  # type: ignore
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

    def print_files_defined_in_yaml(self):
        print_json_file_tree(self.model_dump())

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
        values: Dict[str, Any],
        toggle_varname: str,
        lst_rqrd_if_true: List[str],
        lst_rqrd_if_false: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Validate that required fields are provided depending on a toggle.

        Additionally, for fields that are Path-like, validate that the file exists.

        Returns:
            failing_vars: list of field names that failed
            errors: list of error messages
        """
        failing_vars: List[str] = []
        errors: List[str] = []
        toggle = values.get(toggle_varname)
        required_fields = lst_rqrd_if_true if toggle else lst_rqrd_if_false
        for var in required_fields:
            val = values.get(var)
            # Check for presence
            if val is None:
                errors.append(
                    f"{var} must be provided if {toggle_varname} is {'True' if toggle else 'False'}"
                )
                failing_vars.append(var)
                continue
            # Check if Path exists
            if isinstance(val, Path):
                p = val.expanduser()
                if not p.exists():
                    errors.append(f"{var} path does not exist: {p}")
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

    @field_validator("*", mode="before")
    @classmethod
    def _check_paths_exist(cls, v: Any, info) -> Any:
        """
        Validate that all Path-like fields exist.
        Skips non-path fields automatically.
        """
        if v is None:
            return v  # allow optional
        # Only handle Path or str values
        if isinstance(v, Path):
            p = Path(v).expanduser()
            if not p.exists():
                raise ValueError(f"File does not exist: {p}")
            return p  # convert str → Path
        # everything else is ignored
        return v


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
        ### toggle_swmm_model (standalone SWMM execution)
        swmm_model_test = dict(
            toggle_varname="toggle_swmm_model",
            lst_rqrd_if_true=["SWMM_full"],
            lst_rqrd_if_false=[],
        )
        cls.toggle_tests.append(swmm_model_test)
        return


class analysis_config(cfgBaseModel):
    # REQUIRED INPUTS
    analysis_id: Annotated[
        str,
        Field(
            ...,
            description="analysis identifier. Used for creating analysis folder if one with the same name does not exist.",
            pattern=r"^[A-Za-z][A-Za-z0-9_.]*$",
        ),
    ]
    weather_event_indices: list = Field(
        ...,
        description="List of one or more strings corresponding to fields used for indexing unique weather events. These must match what is in weather_timeseries and weather_event_summary_csv.",
    )
    weather_timeseries: Path = Field(
        ...,
        description="Netcdf containing weather event time series data. Events must share indices with weather_event_summary_csv.",
    )
    weather_time_series_timestep_dimension_name: str = Field(
        ...,
        description="Dimension in weather_timeseries corresponding to timestep.",
    )
    rainfall_units: Literal["mm", "mm/hr"] = Field(
        ...,
        description="Rainfall units in weather_timeseries mm or mm/hr.",
    )
    # COMPUTE CONFIG
    run_mode: Literal["serial", "openmp", "mpi", "hybrid", "gpu"] = Field(
        ..., description="Compute configuration"
    )
    n_mpi_procs: Optional[int] = Field(
        1, description="Number of MPI ranks per simulation."
    )
    n_omp_threads: Optional[int] = Field(1, description="Threads per rank")
    n_gpus: Optional[int] = Field(0, description="Number of GPUs per simulation")
    n_nodes: Optional[int] = Field(1, description="Number of nodes per simulation.")

    # MULTI-SIMULATION EXECUTION METHOD
    multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks"] = (
        Field(
            "local",
            description="Method for running multiple simulations: 'local' (ThreadPoolExecutor on desktop), 'batch_job' (SLURM job array with independent tasks), or '1_job_many_srun_tasks' (single SLURM job with multiple srun tasks respecting job allocation).",
        )
    )
    hpc_total_nodes: Optional[int] = Field(
        None,
        description="This is the total number of nodes that will be requested when multi_sim_run_method = 1_job_many_srun_tasks",
    )
    hpc_total_job_duration_min: Optional[int] = Field(
        None,
        description="This is the job duration when multi_sim_run_method = 1_job_many_srun_tasks",
    )
    hpc_gpus_per_node: Optional[int] = Field(
        None,
        description="GPUs per node on the HPC cluster. Required when using GPUs with "
        "multi_sim_run_method = 1_job_many_srun_tasks. "
        "Used with --gres=gpu:{hpc_gpus_per_node} directive. ",
    )
    hpc_cpus_per_node: Optional[int] = Field(
        None,
        description="CPUs per node on the HPC cluster. Required for dry runs using "
        "multi_sim_run_method = 1_job_many_srun_tasks.",
    )
    # local run constraints
    local_cpu_cores_for_workflow: Optional[int] = Field(
        None,
        description="This is passed to Snakemake to let it know how many CPU cores its allowed to use on your computer",
    )
    # HPC JOB ARRAY PARAMETERS
    mem_gb_per_cpu: int = Field(2, description="Memory per CPU in GB. Defaults to 2GB.")
    hpc_time_min_per_sim: Optional[int] = Field(
        60,
        description="Time in minutes per simulation for SLURM job array. Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    hpc_max_simultaneous_sims: Optional[int] = Field(
        None,
        description="Maximum number of concurrent simulations. "
        "NOTE: Not required for multi_sim_run_method=1_job_many_srun_tasks "
        "(concurrency determined dynamically from SLURM allocation). "
        "Required for setting an upper limit on the number of concurrent jobs submitted using sbatch for multi_sim_run_method=batch_job",
    )
    hpc_ensemble_partition: Optional[str] = Field(
        None,
        description="SLURM partition name (e.g., 'standard', 'gpu', 'high-memory') for running simulations. Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    hpc_setup_and_analysis_processing_partition: Optional[str] = Field(
        None,
        description="SLURM partition name for simulation setup and analysis output consolidation (single node, single core processing). Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    hpc_account: Optional[str] = Field(
        None,
        description="SLURM allocation/account name. Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    python_path: Optional[Path] = Field(
        None,
        description="Optional path to Python executable (e.g., /home/user/.conda/envs/myenv/bin/python). If provided, this will be used instead of 'python' in SLURM scripts. Useful for specifying a conda environment's Python on HPC systems.",
    )
    additional_SBATCH_params: Optional[List[str]] = Field(
        None,
        description="Optional list of SBATCH arguments (omit #SBATCH). Really only relevant for when multi_sim_run_method = 1_job_many_srun_tasks.",
    )
    # TOGGLES
    # TODO - create validatoin checks for bash script toggle
    # toggle_run_ensemble_with_bash_script: bool = Field(
    #     ...,
    #     description="If true, a bash script will be generated using a template and submitted the the HPC to run the ensemble.",
    # )
    toggle_sensitivity_analysis: bool = Field(
        ...,
        description="Whether or not this is a sensitivity study. If so, a .csv file is required for input sensitivity_analysis defining the analysisal setup.",
    )
    toggle_storm_tide_boundary: bool = Field(
        ...,
        description="If True, a boundary condition representing storm tide will be applied to the model.",
    )
    # OPTIONAL OR DEPENDENT
    # hpc_bash_script_ensemble_template: Optional[Path] = Field(
    #     None,
    #     description="Bash script template filled with other user defined variables in the analysis configuration yaml.",
    # )
    # hpc_n_nodes: Optional[int] = Field(
    #     None, description="Number of HPC nodes to request."
    # )
    # hpc_cpus_per_task: Optional[int] = Field(
    #     None, description="CPUs per task (threads per MPI rank)."
    # )
    # hpc_ntasks_per_node: Optional[int] = Field(
    #     None, description="Number of tasks per node (MPI ranks per node)"
    # )
    # hpc_gpus_requested: Optional[int] = Field(
    #     None, description="Number of GPUs requested."
    # )

    storm_tide_boundary_line_gis: Optional[Path] = Field(
        None,
        description="Path to a line gis file spanning the extent of the dem boundary where the variable storm tide boundary condition should be applied.",
    )
    storm_tide_units: Optional[str] = Field(
        None,
        description="Storm tide units, e.g., ft, m. Must align with units used DEM.",
    )
    weather_event_summary_csv: Optional[Path] = Field(
        None,
        description="CSV file with weather event summary statistics. Events must share indices with weather_timeseries.",
    )
    weather_time_series_storm_tide_datavar: Optional[str] = Field(
        None,
        description="Data variables in weather_timeseries corresponding to storm tide.",
    )
    sensitivity_analysis: Optional[Path] = Field(
        None,
        description="sensitivity analysisal design csv file.",
    )
    weather_events_to_simulate: Path = Field(
        ...,
        description="Path to a .csv file defining weather event index used for sensitivity. The columns must correspond to the sytem's weather_event_indices.",
    )
    analysis_description: Optional[str] = Field(
        None,
        description="For readability.",
    )

    # TRITON-SWMM PARAMETERS
    TRITON_processed_output_type: Literal["zarr", "nc"] = Field(
        "zarr",
        description="TRITON processed output type, zarr or nc.",
    )
    TRITON_raw_output_type: Literal["bin", "asc"] = Field(
        "bin",
        description="TRITON raw output type, asc or bin.",
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

    # extra inputs (currently only used by sensitivity analysis)
    analysis_dir: Optional[Path] = Field(
        None,
        description="Optional path to analysis directory. If not specified, the analysis directory will be placed within the system directory named named with the analysis_id",
    )

    # VALIDATION - STRING REQUIREMENTS
    @field_validator("analysis_id")
    def validate_analysis_id(cls, v):
        if not re.match(r"^[A-Za-z0-9_.]*$", v):
            raise ValueError(
                "analysis_id must contain only letters, digits, underscores, or periods"
            )
        return v

    # VALIDATING DEPENDENCIES BASED ON TOGGLES
    @classmethod
    def get_toggle_tests(cls):
        ### toggle_sensitivity_analysis
        bm_test = dict(
            toggle_varname="toggle_sensitivity_analysis",
            lst_rqrd_if_true=["sensitivity_analysis"],
            lst_rqrd_if_false=[],
        )
        cls.toggle_tests.append(bm_test)
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

    @model_validator(mode="before")
    @classmethod
    def check_consistency(cls, values):
        mode = values.get("run_mode")
        mpi = values.get("n_mpi_procs")
        omp = values.get("n_omp_threads")
        gpus = values.get("n_gpus")
        nodes = values.get("n_nodes")

        # -------------------------------
        # Validation rules per mode
        # -------------------------------
        if mode == "serial":
            if mpi is not None and mpi != 1:
                raise ValueError(
                    f"n_mpi_procs is set to {mpi}.\nn_mpi_procs must be None or 1 for serial mode"
                )
            if omp is not None and omp != 1:
                raise ValueError("n_omp_threads must be 1 or None for serial mode")
            if gpus not in (None, 0):
                raise ValueError("n_gpus must be None or 0 for serial mode")
            if nodes is not None and nodes != 1:
                raise ValueError(
                    "n_nodes must be 1 or None for serial mode (single task cannot span multiple nodes)"
                )

        elif mode == "openmp":
            if mpi not in (None, 1):
                raise ValueError("n_mpi_procs must be None or 1 for OpenMP mode")
            if omp is None or omp < 2:
                raise ValueError("n_omp_threads must be >1 for OpenMP mode")
            if gpus not in (None, 0):
                raise ValueError("n_gpus must be None or 0 for OpenMP mode")
            if nodes is not None and nodes != 1:
                raise ValueError(
                    "n_nodes must be 1 or None for OpenMP mode (single task cannot span multiple nodes)"
                )

        elif mode == "mpi":
            if mpi is None or mpi < 2:
                raise ValueError("n_mpi_procs must be >1 for MPI mode")
            if omp not in (None, 1):
                raise ValueError("n_omp_threads must be 1 or None for MPI-only mode")
            if gpus not in (None, 0):
                raise ValueError("n_gpus must be None or 0 for MPI-only mode")
            # Validate ntasks >= nnodes
            if nodes is not None and mpi is not None and mpi < nodes:
                raise ValueError(
                    f"n_mpi_procs must be >= n_nodes for MPI mode. "
                    f"You specified n_mpi_procs={mpi} and n_nodes={nodes}. "
                    f"Each node requires at least one MPI rank to run on it."
                )

        elif mode == "hybrid":
            if mpi is None or mpi < 2:
                raise ValueError("n_mpi_procs must be >1 for hybrid mode")
            if omp is None or omp < 2:
                raise ValueError("n_omp_threads must be >1 for hybrid mode")
            if gpus not in (None, 0):
                raise ValueError("n_gpus must be None or 0 for hybrid CPU mode")
            # Validate ntasks >= nnodes
            if nodes is not None and mpi is not None and mpi < nodes:
                raise ValueError(
                    f"n_mpi_procs must be >= n_nodes for hybrid mode. "
                    f"You specified n_mpi_procs={mpi} and n_nodes={nodes}. "
                    f"Each node requires at least one MPI rank to run on it."
                )

        elif mode == "gpu":
            if gpus is None or gpus < 1:
                raise ValueError("n_gpus must be >=1 for GPU mode")
            if mpi is not None and mpi < 1:
                raise ValueError("n_mpi_procs must be >=1 if using MPI with GPU")
            if omp is not None and omp < 1:
                raise ValueError("n_omp_threads must be >=1 if using OpenMP with GPU")
            # Validate ntasks >= nnodes (if using MPI with GPU)
            if mpi is not None and mpi > 1 and nodes is not None and mpi < nodes:
                raise ValueError(
                    f"n_mpi_procs must be >= n_nodes for GPU mode with MPI. "
                    f"You specified n_mpi_procs={mpi} and n_nodes={nodes}. "
                    f"Each node requires at least one MPI rank to run on it."
                )

        else:
            raise ValueError(f"Unknown run_mode: {mode}")

        return values


def load_system_config_from_dict(cfg_dict):
    cfg = system_config.model_validate(cfg_dict)
    return cfg


def load_system_config(cfg_yaml: Path):
    cfg = yaml.safe_load(cfg_yaml.read_text())
    cfg = system_config.model_validate(cfg)
    return cfg


def load_analysis_config(cfg_yaml: Path):
    cfg = yaml.safe_load(cfg_yaml.read_text())
    cfg = analysis_config.model_validate(cfg)
    return cfg


# def load_sensitivity_analysis_config_config(cfg):
#     cfg = sensitivity_analysis_config.model_validate(cfg)
#     return cfg
