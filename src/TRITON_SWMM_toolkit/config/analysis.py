from pydantic import Field, field_validator, model_validator
from typing import List, Optional, Literal, Annotated
from pathlib import Path
import re
from TRITON_SWMM_toolkit.config.base import cfgBaseModel


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
    # SWMM only
    n_threads_swmm: Optional[int] = Field(
        1, description="Threads per rank for SWMM-only simulations"
    )

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
    gpu_hardware: Optional[str] = Field(
        None,
        description=(
            "Optional GPU hardware selector (e.g., 'a100', 'h200', 'rtx3090'). "
            "If provided, SLURM GPU requests will qualify the GPU type using "
            "--gpus (batch_job) or --gres (1_job_many_srun_tasks)."
        ),
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
    local_gpus_for_workflow: Optional[int] = Field(
        None,
        description="This is passed to Snakemake to let it know how many GPUS its allowed to use on your computer",
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
    toggle_sensitivity_analysis: bool = Field(
        ...,
        description="Whether or not this is a sensitivity study. If so, a .csv file is required for input sensitivity_analysis defining the analysisal setup.",
    )
    toggle_storm_tide_boundary: bool = Field(
        ...,
        description="If True, a boundary condition representing storm tide will be applied to the model.",
    )

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
    is_subanalysis: Optional[bool] = Field(
        False,
        description="This is used in the backend to help route subanalyses to appropriate processes.",
    )

    # VALIDATION - STRING REQUIREMENTS
    @field_validator("analysis_id")
    def validate_analysis_id(cls, v):
        if not re.match(r"^[A-Za-z0-9_.]*$", v):
            raise ValueError(
                "analysis_id must contain only letters, digits, underscores, or periods"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_toggle_dependencies(cls, values):
        errors = []

        _, additional_errors = cls.validate_from_toggle(
            values,
            toggle_varname="toggle_sensitivity_analysis",
            lst_rqrd_if_true=["sensitivity_analysis"],
            lst_rqrd_if_false=[],
        )
        errors.extend(additional_errors)

        _, additional_errors = cls.validate_from_toggle(
            values,
            toggle_varname="toggle_storm_tide_boundary",
            lst_rqrd_if_true=[
                "storm_tide_boundary_line_gis",
                "weather_time_series_storm_tide_datavar",
                "storm_tide_units",
            ],
            lst_rqrd_if_false=[],
        )
        errors.extend(additional_errors)

        if errors:
            raise ValueError("; ".join(errors))
        return values

    @model_validator(mode="before")
    @classmethod
    def check_consistency(cls, values):
        mode = values.get("run_mode")
        mpi = values.get("n_mpi_procs")
        omp = values.get("n_omp_threads")
        gpus = values.get("n_gpus")
        nodes = values.get("n_nodes")
        multi_sim_method = values.get("multi_sim_run_method")
        hpc_gpus_per_node = values.get("hpc_gpus_per_node")

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

            if multi_sim_method == "1_job_many_srun_tasks" and not hpc_gpus_per_node:
                raise ValueError(
                    "hpc_gpus_per_node is required for 1_job_many_srun_tasks when using GPUs"
                )

        return values
