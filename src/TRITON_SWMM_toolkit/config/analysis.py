import math
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from TRITON_SWMM_toolkit.config.base import cfgBaseModel

# One-way import: config/analysis.py imports report_config; config/report.py
# must not import from config/analysis.py to avoid circular import.
from TRITON_SWMM_toolkit.config.report import report_config as _report_config_model

ClearRawValue = Literal["all", "none"] | list[Literal["tritonswmm", "triton", "swmm"]]
ForceRerunValue = Literal["all", "none"] | dict[Literal["sa_id", "event_iloc"], list[int | str]]


def _read_cgroup_memory_limit_mib() -> float | None:
    """Best-effort read of the process's cgroup memory ceiling, in MiB.

    Returns None when the limit is unknown or unlimited (so callers fall back to
    the declared config value). Non-fatal by contract — never raises.
    """
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as fh:
                raw = fh.read().strip()
            if raw in ("max", ""):
                return None
            val = int(raw)
            if val >= 2**62:  # cgroup v1 'unlimited' sentinel near 2**63
                return None
            return val / (1024**2)
        except (OSError, ValueError):
            continue
    return None


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
    weather_time_series_spatial_mean_rainfall_datavar: str = Field(
        ...,
        description=(
            "Data variable in weather_timeseries corresponding to the "
            "spatially-averaged rainfall time series the report renderers "
            "(per_sim_peak_flood_depth / per_sim_conduit_flow event-hydrology "
            "panels) read for the rainfall sub-panel. Required."
        ),
    )
    rainfall_units: Literal["mm", "mm/hr"] = Field(
        ...,
        description="Rainfall units in weather_timeseries mm or mm/hr.",
    )
    # COMPUTE CONFIG
    run_mode: Literal["serial", "openmp", "mpi", "hybrid", "gpu"] = Field(..., description="Compute configuration")
    n_mpi_procs: int | None = Field(1, description="Number of MPI ranks per simulation.")
    n_omp_threads: int | None = Field(
        1,
        description=(
            "Number of OpenMP threads for simulation execution. For TRITON/TRITON-SWMM models, "
            "controls OpenMP threading in the executable. For SWMM standalone models, dynamically "
            "updates the THREADS parameter in the [OPTIONS] section of .inp files."
        ),
    )
    n_gpus: int | None = Field(0, description="Number of GPUs per simulation")
    n_nodes: int | None = Field(1, description="Number of nodes per simulation.")

    # MULTI-SIMULATION EXECUTION METHOD
    multi_sim_run_method: Literal["local", "batch_job", "1_job_many_srun_tasks"] = Field(
        "local",
        description="Method for running multiple simulations: 'local' (ThreadPoolExecutor on desktop), 'batch_job' (tmux session running Snakemake on login node - recommended for HPC), or '1_job_many_srun_tasks' (single SLURM job with multiple srun tasks respecting job allocation).",
    )
    hpc_total_nodes: int | None = Field(
        None,
        description="This is the total number of nodes that will be requested when multi_sim_run_method = 1_job_many_srun_tasks",
    )
    hpc_total_job_duration_min: int | None = Field(
        None,
        description="This is the job duration when multi_sim_run_method = 1_job_many_srun_tasks",
    )
    # TODO - hpc_gpus_per_node should be used in the 1 big job approach and in the batch job approach
    # TODO - hpc_cpus_per_node should be used in a similar way. With both these arguments,
    # specifying n_nodes should no longer be necessary.
    hpc_gpus_per_node: int | None = Field(
        None,
        description=(
            "GPUs per node on the HPC cluster. Required when using GPUs with "
            "multi_sim_run_method = 1_job_many_srun_tasks or batch_job. "
            "Used to populate per-node GPU allocation for Snakemake and to "
            "generate --gres or --gpus-per-node directives."
        ),
    )
    hpc_cpus_per_node: int | None = Field(
        None,
        description="CPUs per node on the HPC cluster. Required for dry runs using "
        "multi_sim_run_method = 1_job_many_srun_tasks.",
    )
    hpc_mem_allocation_for_sim_output_processing_mb: int = Field(
        12000,
        description="Memory allocation for creating simulation time series.",
    )
    hpc_mem_allocation_for_analysis_output_consolidation_mb: int = Field(
        12000,
        description="Memory allocation for consolidating simulation summaries across all scenarios.",
    )
    hpc_mem_allocation_for_setup_mb: int = Field(
        default=12000,
        gt=0,
        description=(
            "Memory allocation (in MB) for the setup_target SLURM rule that runs "
            "system-input processing (DEM coarsening, Manning's raster) and TRITON-SWMM "
            "compilation. Default 12 GB covers 0.35 m DEM processing (empirical peak "
            "~5.15 GB parent-process RSS) with 2.3x headroom and the compile-side peak "
            "(~1.34 GB) ~9x. Increase for higher-resolution DEMs or larger watersheds."
        ),
    )
    hpc_runtime_min_for_setup: int = Field(
        default=60,
        gt=0,
        description=(
            "Time allocation (in minutes) for the setup_target SLURM rule. Default 60 "
            "covers 0.35 m DEM processing (empirical wall time ~2:24) plus a -j4 GPU "
            "compile (~3 min) with headroom. Increase for higher-resolution DEMs or "
            "slower nodes."
        ),
    )
    hpc_max_wait_for_inflight_min: int = Field(
        480,
        ge=60,
        le=10080,
        description=(
            "OVERRIDE CEILING (minutes) on the v2 wait-on-sentinel rule's poll "
            "cap. As of v2-post-death-recovery-hardening, the wait-rule cap is "
            "DERIVED per-rule from the waited-on sim's own walltime "
            "(hpc_total_job_duration_min + 30 min slack); this field caps that "
            "derived value from above (min(derived, this)). Default 480 (8h). "
            "Bounds: 60 (1h) to 10080 (1 week). Set below the job walltime only "
            "to force an earlier give-up on a still-running wait."
        ),
    )
    # local run constraints
    local_cpu_cores_for_workflow: int | None = Field(
        None,
        description="This is passed to Snakemake to let it know how many CPU cores its allowed to use on your computer",
    )
    local_gpus_for_workflow: int | None = Field(
        None,
        description="This is passed to Snakemake to let it know how many GPUS its allowed to use on your computer",
    )
    # HPC JOB ARRAY PARAMETERS
    mem_gb_per_cpu: int = Field(2, description="Memory per CPU in GB. Defaults to 2GB.")
    hpc_time_min_per_sim: int | None = Field(
        60,
        description="Time in minutes per simulation for SLURM job array. Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    hpc_max_simultaneous_sims: int | None = Field(
        None,
        description="Maximum number of concurrent simulations. "
        "NOTE: Not required for multi_sim_run_method=1_job_many_srun_tasks "
        "(concurrency determined dynamically from SLURM allocation). "
        "Required for setting an upper limit on the number of concurrent jobs submitted using sbatch for multi_sim_run_method=batch_job",
    )
    hpc_ensemble_partition: str | None = Field(
        None,
        description="SLURM partition name (e.g., 'standard', 'gpu', 'high-memory') for running simulations. Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    hpc_setup_and_analysis_processing_partition: str | None = Field(
        None,
        description="SLURM partition name for simulation setup and analysis output consolidation (single node, single core processing). Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    hpc_account: str | None = Field(
        None,
        description="SLURM allocation/account name. Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    hpc_login_node: str | None = Field(
        None,
        description=(
            "Specific HPC login node hostname for tmux session reattach (e.g., 'login1.hpc.virginia.edu'). "
            "Only needed if the cluster uses round-robin login load balancing. "
            "If unset, the toolkit auto-detects and stores the submission node hostname at launch time. "
            "When set, reattach hints will use ssh to this node directly."
        ),
    )
    python_path: Path | None = Field(
        None,
        description="Optional path to Python executable (e.g., /home/user/.conda/envs/myenv/bin/python). If provided, this will be used instead of 'python' in SLURM scripts. Useful for specifying a conda environment's Python on HPC systems.",
    )
    additional_SBATCH_params: list[str] | None = Field(
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

    storm_tide_boundary_line_gis: Path | None = Field(
        None,
        description="Path to a line gis file spanning the extent of the dem boundary where the variable storm tide boundary condition should be applied.",
    )
    storm_tide_units: str | None = Field(
        None,
        description="Storm tide units, e.g., ft, m. Must align with units used DEM.",
    )
    weather_event_summary_csv: Path | None = Field(
        None,
        description="CSV file with weather event summary statistics. Events must share indices with weather_timeseries.",
    )
    weather_time_series_storm_tide_datavar: str | None = Field(
        None,
        description="Data variables in weather_timeseries corresponding to storm tide.",
    )
    sensitivity_analysis: Path | None = Field(
        None,
        description="sensitivity analysisal design csv file.",
    )
    weather_events_to_simulate: Path = Field(
        ...,
        description="Path to a .csv file defining weather event index used for sensitivity. The columns must correspond to the sytem's weather_event_indices.",
    )
    analysis_description: str | None = Field(
        None,
        description="For readability.",
    )

    # TRITON-SWMM PARAMETERS
    target_processed_output_type: Literal["zarr", "nc"] = Field(
        "zarr",
        description="TRITON processed output type, zarr or nc.",
    )
    process_output_target_chunksize_mb: int = Field(
        200,
        description="Target memory budget (MiB) PER LOAD CHUNK for streaming-chunked operations on per-scenario timeseries output. This is the in-memory RSS guard ONLY; it does NOT govern zarr-append granularity (see process_append_batch_timesteps). Consumed by both write_timeseries_outputs (raw-to-zarr chunked LOAD at process_simulation.py L544/L736) AND summarize_triton_simulation_results' _streaming_argmax_with_companions helper (per-cell argmax+companion reduction). On fine grids a single float64 timestep can meet/exceed this budget, flooring the load chunk to 1 timestep — that is a correct memory guard, NOT a performance bug, because append granularity is decoupled via process_append_batch_timesteps. See Gotcha #23/#24.",
    )
    process_append_batch_timesteps: int = Field(
        128,
        description="Number of LOADED timesteps to accumulate before emitting ONE zarr append in write_timeseries_outputs. Decouples zarr-append granularity from the in-memory load-chunk size (process_output_target_chunksize_mb), so fine grids that floor the load chunk to 1 timestep still emit only ceil(N_timesteps / this) appends instead of O(N_timesteps) tiny appends. Independent of the streaming-summary reduction (which does not append). Buffer RSS is additionally byte-capped at 2x the load budget at write time, so raising this is safe.",
    )
    process_append_batch_memory_budget_mb: int | None = Field(
        None,
        description=(
            "Memory budget (MiB) governing BOTH the zarr-append batch byte cap in "
            "write_timeseries_outputs AND the streaming-argmax summary reduction in "
            "summarize_triton_simulation_results. Distinct from "
            "process_output_target_chunksize_mb (the small per-LOAD-chunk RSS guard, "
            "~200 MiB): this larger budget lets fine grids accumulate a bigger pending "
            "batch / argmax chunk inside the process job's real RAM allocation. When None "
            "(default), resolved at config-load to a fraction (0.35) of "
            "hpc_mem_allocation_for_sim_output_processing_mb (the field that sets the "
            "process rule's SLURM mem_mb), clamped to the actual cgroup limit when "
            "readable — see the _resolve_process_batch_budget validator. A concrete int "
            "overrides the fraction but is still ceiling-checked at <= 0.5*job_RAM. The "
            "0.35 fraction reserves headroom for the peak-RSS inequality 2*B + per_ts "
            "<= job_RAM (the flush transiently holds the pending batch B, its xr.concat "
            "copy ~B, and one live load chunk per_ts), accounting for the post-append "
            "trigger overshoot. Consumed at process_simulation.py write-flush cap and "
            "argmax budget."
        ),
    )
    process_store_float32: bool = Field(
        True,
        description=(
            "Store per-scenario spatial timeseries (H/QX/QY/MH -> wlevel/velocity) as "
            "float32 in the processed zarr instead of float64, ~halving on-disk size and "
            "I/O. Default True. Set False for precision-sensitive analyses (e.g. tight "
            "mass-balance). Does NOT alter CF attributes — dtype lives in the zarr encoding "
            "dict, orthogonal to cf_conventions.py. Consumed by utils.return_dic_zarr_encodings."
        ),
    )
    process_timestep_chunk: int | None = Field(
        None,
        description=(
            "Explicit on-disk `timestep_min` zarr chunk size for the per-scenario "
            "spatial timeseries. When None (default), preserves the current "
            "first-write-extent chunking behavior. Decouples read-locality from the "
            "write append-batch size. Consumed by utils.return_dic_zarr_encodings."
        ),
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
    TRITON_reporting_timestep_s: int | float = Field(
        ...,
        description="Reporting timestep in seconds.",
    )
    open_boundaries: int = Field(
        ...,
        description="0 for closed, 1 for open. This is affects all boundaries wherever external boundary conditions are not otherwise defined.",
    )

    # extra inputs (currently only used by sensitivity analysis)
    analysis_dir: Path | None = Field(
        None,
        description="Optional path to analysis directory. If not specified, the analysis directory will be placed within the system directory named named with the analysis_id",
    )
    is_subanalysis: bool = Field(
        False,
        description="This is used in the backend to help route subanalyses to appropriate processes.",
    )
    master_analysis_cfg_yaml: Path | None = Field(
        None,
        description="Path to the configuration file of the master analysis.",
    )
    report: _report_config_model = Field(
        ...,
        description=(
            "Required inline report-rendering config (formerly a separate "
            "report_config.yaml referenced by absolute path in Snakefile shell "
            "lines, eliminated post-F2). The canonical source of truth for "
            "renderer parameters including `interactive.static_backend`. A "
            "cfg_analysis.yaml file without a `report:` block raises pydantic "
            "ValidationError at load time. Callers may still pass an explicit "
            "`report_config=` argument to `analysis.run()` to override."
        ),
    )

    brand_theme: Path | None = Field(
        None,
        description=(
            "Optional path to a brand-theme YAML (ADR-7 layer 2 — institutional "
            "identity: report.css :root palette + HTML-table primary/accent + "
            "navbar upper-left text). When None (default), the code-frozen "
            "DEFAULT_BRAND_THEME (config/brand_theme.py) applies. Mirrors the "
            "sensitivity_analysis / storm_tide_boundary_line_gis path-field "
            "precedent. Callers may pass an explicit `override_brand_theme=` Path "
            "to `analysis.run()` to override for one invocation, mirroring the "
            "`report_config=` runtime-override precedent. Automatically "
            "per-sub-analysis overlayable via an `analysis.brand_theme` "
            "sensitivity column."
        ),
    )

    # CLEANUP / FORCE-RERUN POLICY (cleanup-rerun-delete-redesign Phase 1)
    clear_raw: ClearRawValue = Field(
        "none",
        description=(
            'Post-processing cleanup policy. "all" deletes all raw outputs '
            'for every enabled model type. "none" deletes nothing. A list '
            'of model type strings (subset of "tritonswmm", "triton", "swmm") '
            "deletes raw outputs only for the listed model types. Defaults "
            'to "none" — yamls written before this field was introduced '
            "load cleanly with the strict-safe (delete-nothing) default."
        ),
    )
    force_rerun: ForceRerunValue = Field(
        "none",
        description=(
            'Force-rerun policy. "all" re-runs everything. "none" runs no '
            'forced re-runs. A dict with exactly one key — "sa_id" (sensitivity '
            'only) or "event_iloc" (non-sensitivity only) — and a list of int '
            "or string identifiers re-runs only the named sub-analyses or "
            'events. Defaults to "none" — yamls written before this field '
            "was introduced load cleanly with the strict-safe (re-run-nothing) "
            "default."
        ),
    )

    # VALIDATION - STRING REQUIREMENTS
    @field_validator("analysis_id")
    def validate_analysis_id(cls, v):
        if not re.match(r"^[A-Za-z0-9_.]*$", v):
            raise ValueError("analysis_id must contain only letters, digits, underscores, or periods")
        return v

    @field_validator("clear_raw", mode="after")
    @classmethod
    def _validate_clear_raw(cls, v):
        if isinstance(v, list):
            if not v:
                raise ValueError("clear_raw list form cannot be empty; use 'none' to delete nothing")
            if len(v) != len(set(v)):
                raise ValueError(f"clear_raw list contains duplicates: {v}")
            for item in v:
                if item in ("all", "none"):
                    raise ValueError(
                        f"clear_raw list cannot contain sentinel value {item!r}; "
                        f"use the sentinel as a bare string (clear_raw: {item})"
                    )
        return v

    @field_validator("force_rerun", mode="after")
    @classmethod
    def _validate_force_rerun(cls, v):
        _SA_ID_RE = re.compile(r"^[A-Za-z0-9_.]+$")
        if isinstance(v, dict):
            if len(v) != 1:
                raise ValueError(
                    f"force_rerun dict form must have exactly one key (either "
                    f"'sa_id' or 'event_iloc'); got {len(v)} keys: {list(v.keys())}"
                )
            key = next(iter(v))
            if key not in ("sa_id", "event_iloc"):
                raise ValueError(f"force_rerun dict key must be 'sa_id' or 'event_iloc'; got {key!r}")
            values = v[key]
            if not isinstance(values, list) or not values:
                raise ValueError(f"force_rerun.{key} value must be a non-empty list; got {values!r}")
            if len(values) != len(set(map(str, values))):
                raise ValueError(f"force_rerun.{key} list contains duplicates: {values}")
            if key == "sa_id":
                bad = [str(x) for x in values if not _SA_ID_RE.match(str(x))]
                if bad:
                    raise ValueError(
                        f"force_rerun.sa_id values must match ^[A-Za-z0-9_.]+$ "
                        f"(per accepted decision 'All user-provided identifiers that "
                        f"become Snakemake wildcards must match ^[A-Za-z0-9_.]+$'); "
                        f"got invalid: {bad}"
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
    def validate_subanalysis_requirements(cls, values):
        errors = []
        if values.get("is_subanalysis") is True:
            if values.get("master_analysis_cfg_yaml") is None:
                errors.append("master_analysis_cfg_yaml must be provided when is_subanalysis=True")
            if values.get("analysis_dir") is None:
                errors.append("analysis_dir must be provided when is_subanalysis=True")

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
        hpc_max_simultaneous_sims = values.get("hpc_max_simultaneous_sims")
        hpc_total_job_duration_min = values.get("hpc_total_job_duration_min")
        hpc_time_min_per_sim = values.get("hpc_time_min_per_sim")

        # -------------------------------
        # Validation rules per mode
        # -------------------------------
        if mode == "serial":
            if mpi is not None and mpi != 1:
                raise ValueError(f"n_mpi_procs is set to {mpi}.\nn_mpi_procs must be None or 1 for serial mode")
            if omp is not None and omp != 1:
                raise ValueError("n_omp_threads must be 1 or None for serial mode")
            if gpus not in (None, 0):
                raise ValueError("n_gpus must be None or 0 for serial mode")
            if nodes is not None and nodes != 1:
                raise ValueError("n_nodes must be 1 or None for serial mode (single task cannot span multiple nodes)")

        elif mode == "openmp":
            if mpi not in (None, 1):
                raise ValueError("n_mpi_procs must be None or 1 for OpenMP mode")
            if omp is None or omp < 2:
                raise ValueError("n_omp_threads must be >1 for OpenMP mode")
            if gpus not in (None, 0):
                raise ValueError("n_gpus must be None or 0 for OpenMP mode")
            if nodes is not None and nodes != 1:
                raise ValueError("n_nodes must be 1 or None for OpenMP mode (single task cannot span multiple nodes)")

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

            if multi_sim_method in {"1_job_many_srun_tasks", "batch_job"} and not hpc_gpus_per_node:
                raise ValueError(
                    "hpc_gpus_per_node is required when using GPUs with batch_job or 1_job_many_srun_tasks"
                )

        if multi_sim_method == "batch_job" and (hpc_max_simultaneous_sims is None or hpc_max_simultaneous_sims < 1):
            raise ValueError("hpc_max_simultaneous_sims is required and must be > 0 for multi_sim_run_method=batch_job")

        if multi_sim_method == "batch_job" and (hpc_total_job_duration_min is None or hpc_total_job_duration_min < 1):
            raise ValueError(
                "hpc_total_job_duration_min is required and must be > 0 for multi_sim_run_method=batch_job"
            )

        if multi_sim_method == "batch_job":
            if hpc_time_min_per_sim is None:
                raise ValueError("hpc_time_min_per_sim is required and must be >= 1 for multi_sim_run_method=batch_job")
            if isinstance(hpc_time_min_per_sim, float) and math.isnan(hpc_time_min_per_sim):
                raise ValueError(
                    "hpc_time_min_per_sim must be a valid integer >= 1 for multi_sim_run_method=batch_job (NaN detected)"
                )
            if hpc_time_min_per_sim < 1:
                raise ValueError("hpc_time_min_per_sim must be >= 1 for multi_sim_run_method=batch_job")

        return values

    @model_validator(mode="after")
    def _validate_force_rerun_against_sensitivity_toggle(self):
        if isinstance(self.force_rerun, dict):
            key = next(iter(self.force_rerun))
            if key == "sa_id" and not self.toggle_sensitivity_analysis:
                raise ValueError("force_rerun.sa_id requires toggle_sensitivity_analysis=True")
            if key == "event_iloc" and self.toggle_sensitivity_analysis:
                raise ValueError(
                    "force_rerun.event_iloc requires toggle_sensitivity_analysis=False; "
                    "sensitivity-toggled analyses must use force_rerun.sa_id instead"
                )
        return self

    @model_validator(mode="after")
    def _validate_inflight_wait_vs_total_runtime(self):
        if (
            self.hpc_max_wait_for_inflight_min is not None
            and self.hpc_total_job_duration_min is not None
            and self.hpc_max_wait_for_inflight_min < self.hpc_total_job_duration_min
        ):
            import warnings

            warnings.warn(
                f"hpc_max_wait_for_inflight_min={self.hpc_max_wait_for_inflight_min} is less than "
                f"hpc_total_job_duration_min={self.hpc_total_job_duration_min}. Wait-rule will time out "
                f"before in-flight sims can finish; consider raising the wait cap.",
                UserWarning,
                stacklevel=2,
            )
        return self

    # Fraction of the declared process SLURM allocation used as the append/argmax
    # budget when process_append_batch_memory_budget_mb is left None. 0.35 keeps
    # headroom for the xr.concat batch copy (~2x pending) + one live load chunk
    # inside the declared mem, accounting for the post-append trigger overshoot.
    _PROCESS_BATCH_BUDGET_FRACTION = 0.35

    @model_validator(mode="after")
    def _resolve_process_batch_budget(self):
        declared_job_ram = self.hpc_mem_allocation_for_sim_output_processing_mb
        if self.process_append_batch_memory_budget_mb is None:
            self.process_append_batch_memory_budget_mb = round(self._PROCESS_BATCH_BUDGET_FRACTION * declared_job_ram)
        # R4 guard 1: never exceed half the declared job RAM (the 2*B <= job_RAM inequality).
        ceiling = round(0.5 * declared_job_ram)
        if self.process_append_batch_memory_budget_mb > ceiling:
            raise ValueError(
                f"process_append_batch_memory_budget_mb "
                f"({self.process_append_batch_memory_budget_mb}) exceeds 0.5 * "
                f"hpc_mem_allocation_for_sim_output_processing_mb ({ceiling}); the "
                f"2*B + per_ts <= job_RAM peak-RSS inequality requires B <= ~0.5*job_RAM."
            )
        # R4 guard 2: best-effort clamp to the ACTUAL cgroup limit when readable, so a
        # SLURM under-allocation (declared > granted) cannot drive the cap above the real
        # envelope (the declared-vs-actual OOM hazard, D6). No-op once declared == actual.
        actual = _read_cgroup_memory_limit_mib()
        if actual is not None:
            self.process_append_batch_memory_budget_mb = min(
                self.process_append_batch_memory_budget_mb,
                round(self._PROCESS_BATCH_BUDGET_FRACTION * actual),
            )
        return self
