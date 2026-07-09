import math
import re
import warnings
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from hhemt.config.base import cfgBaseModel
from hhemt.config.eda import eda_config

# One-way import: config/analysis.py imports report_config; config/report.py
# must not import from config/analysis.py to avoid circular import.
from hhemt.config.report import report_config as _report_config_model

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
    # DATASET PUBLISHING
    dataset_license: Literal["CC0-1.0", "CC-BY-NC-4.0"] = Field(
        default="CC0-1.0",
        description=(
            "SPDX identifier for the published DATASET license (frozen 2-entry vocab, "
            "ADR-8). Baked into the RO-Crate root Dataset.license at consolidation and "
            "read back for the DataCite rightsList at publish time (rightsIdentifierScheme "
            "'SPDX'). CC0-1.0 default is the open, regret-safe choice across immutable DOIs. "
            "CC-BY-NC-4.0 is the research/education-leaning slot; note CC 'NonCommercial' is "
            "broader than 'education only' and does not turn on user type."
        ),
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
    # Phase-4 (4d, hpc-system-profile-config): hpc_gpus_per_node + hpc_cpus_per_node
    # RETIRED off analysis_config. Per-node GPU/CPU topology now lives per-partition
    # on PartitionSpec.gpus_per_node / .cpus_per_node, resolved via the workflow
    # builder's _resolve_gpus_per_node / _resolve_cpus_per_node from the named
    # partition. A pop-and-warn shim (check_consistency, below) lets un-migrated
    # YAMLs still load. REMOVE the shim after <release>.
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
        10080,
        ge=60,
        le=10080,
        description=(
            "Backstop cap (minutes) on the v2 wait-on-sentinel rule's poll loop. "
            "As of the wait-rule in-loop-liveness change, the wait-rule detects "
            "job death in-loop (per-job squeue/sacct probe every ~5 min) and "
            "writes a _failed marker on confirmed death, so this cap is now a "
            "pure safety backstop (NOT walltime-derived) — it fires only if a "
            "job is stuck-but-alive past the cap. Default 10080 (1 week, the "
            "field max) makes waits effectively indefinite, safe because in-loop "
            "detection — not the timer — terminates a dead-job wait. Bounds: 60 "
            "(1h) to 10080 (1 week). Lower it only to force an earlier give-up."
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
    hpc_restart_times_simulate: int = Field(
        2,
        ge=0,
        description=(
            "Per-rule Snakemake `retries:` for the simulation rules "
            "(run_triton/run_tritonswmm/run_swmm/simulation_sa_*). A walltime "
            "kill is a SLURM TIMEOUT (retriable); raise this high (e.g. 20) for a "
            "hotstart-resume sweep so a killed sim re-dispatches from its latest "
            "config_NNNN.cfg checkpoint within ONE analysis.run(). Default 2."
        ),
    )
    hpc_restart_times_other: int = Field(
        2,
        ge=0,
        description=(
            "Per-rule Snakemake `retries:` baseline for the non-simulation rules "
            "(prepare/process/consolidate/plot/render), emitted as the GLOBAL "
            "restart-times so directive-less rules inherit it. Idempotent "
            "re-derivations, so a low count suffices. Default 2."
        ),
    )
    # Phase-4 (4d): hpc_max_simultaneous_sims RETIRED off analysis_config — it MOVED
    # to hpc_system_config.max_concurrent_jobs (D-D: a cluster-throughput cap belongs
    # on the per-HPC-system config, not the per-analysis config). Readers resolve it
    # from cfg_hpc_system.max_concurrent_jobs. Popped by the check_consistency shim.
    #
    # KEPT (D-A): the two partition SELECTORS stay on analysis_config — they are the
    # partition-NAME axis lookup keys the resolution helpers + preflight read to index
    # cfg_hpc_system.partitions[name], and partition-as-sensitivity-axis requires them
    # as the per-CSV-row overlay column. They are NOT retired despite the hpc_* prefix.
    hpc_ensemble_partition: str | None = Field(
        None,
        description="SLURM partition name (e.g., 'standard', 'gpu', 'high-memory') for running simulations. Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    hpc_setup_and_analysis_processing_partition: str | None = Field(
        None,
        description="SLURM partition name for simulation setup and analysis output consolidation (single node, single core processing). Required if using generate_SLURM_job_array_script() or submit_SLURM_job_array().",
    )
    # Phase-4 (4d): hpc_account, hpc_login_node, python_path RETIRED off analysis_config.
    # account -> hpc_system_config.default_account (via _resolve_account); login_node ->
    # hpc_system_config.login_node; python_path -> sys.executable fallback (no
    # hpc_system_config home — re-add if a cluster needs a bespoke interpreter).
    # Popped by the check_consistency shim.
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
            "`report_config=` argument to `analysis.run()` to override. "
            "This inline field IS ADR-7 reporting-config layer 3 "
            "(report-composition): the frozen-default-field whose optional "
            "runtime override is the `report_config=` Path kwarg on run() "
            "(resolved at analysis.py:1746-1757). Layer-3 precedence: explicit "
            "`report_config=` Path > inline cfg_analysis.report. It is "
            "deliberately INLINE (not a path field) per the post-F2 decision "
            "recorded above; ADR-7's 'path field' wording describes the default "
            "shape it imagined, not a functional contract — the inline-default + "
            "path-override form satisfies ADR-7's 'frozen-default-field + "
            "optional runtime override' requirement."
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

    static_plot_configs: list[Path] = Field(
        default_factory=list,
        description=(
            "ADR-7 reporting-config layer 4: per-plot static-config YAML paths. "
            "Each path is a standalone publication-static plot spec. Default [] "
            "(no static plots) — strict-safe; old yamls load cleanly. Each element "
            "is existence-validated at config-load via a dedicated "
            "@field_validator('static_plot_configs') (the base * validator "
            "cfgBaseModel._check_paths_exist only existence-checks SCALAR Path fields "
            "and passes list[Path] through, so a list-aware validator is required). "
            "REFERENCE + VALIDATION ONLY in this plan: the static_plots() generation "
            "this field triggers is built downstream in "
            "reporting-system_static-plots-entrypoint-and-distribution; the field is "
            "inert (settable but unconsumed) until that plan lands."
        ),
    )

    eda: eda_config = Field(
        default_factory=eda_config,
        description=(
            "Optional inline EDA-loop config (ADR-10): selects which EDA plots "
            "appear in the standalone eda_report.html. Default member set (the "
            "cross-sim byte-identity plot) applies when absent. Deliberately INLINE "
            "(not a path field) so it travels in cfg_analysis.yaml and Bundle.eda() "
            "reads it with zero extra carry/repoint wiring — the same rationale as "
            "the `report` field above. Runtime override via eda(override_eda_config=<Path>)."
        ),
    )

    execution_environment: Literal["native", "container"] = Field(
        "native",
        description=(
            "ADR-1: 'native' runs compile+sim+processing on the host (today's "
            "behavior, byte-identical); 'container' wraps the innermost sim {exe} and "
            "the process_{model} runners in `apptainer exec {sif}`, where the cluster "
            "SIF is described by hpc_system_config.container (ContainerSpec). Additive "
            "default-valued field so pre-container configs load as native. The "
            "native|container SELECTOR is experiment-scoped (C-HPC-FIELD-PLACEMENT); "
            "the cluster-coupled 'how to exec' lives on ContainerSpec."
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

    # VALIDATION - PATH-LIST EXISTENCE
    @field_validator("static_plot_configs", mode="after")
    @classmethod
    def _check_static_plot_configs_exist(cls, v: "list[Path]") -> "list[Path]":
        """Element-wise existence check for the layer-4 static-plot config list.

        The base ``*`` validator ``_check_paths_exist`` only handles scalar
        ``Path`` values and silently passes a ``list[Path]`` through, so list
        elements need their own existence validation (R-7 / V-8).
        """
        normed: list[Path] = []
        for elem in v:
            p = Path(elem).expanduser()
            if not p.exists():
                raise ValueError(f"static_plot_configs path does not exist: {p}")
            normed.append(p)
        return normed

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
        # REMOVE after <release>: Phase-4 (4d) pop-and-warn shim for the 6 retired
        # analysis_config HPC fields (moved to hpc_system_config.max_concurrent_jobs /
        # default_account / login_node, the partition-axis PartitionSpec topology, and
        # sys.executable). Pop-and-DROP so un-migrated YAMLs still load (extra="forbid"
        # would else reject). The two partition selectors are KEPT (D-A).
        if isinstance(values, dict):
            for _k in (
                "hpc_account",
                "hpc_login_node",
                "hpc_gpus_per_node",
                "hpc_cpus_per_node",
                "python_path",
                "hpc_max_simultaneous_sims",
            ):
                if _k in values:
                    values.pop(_k)
                    warnings.warn(
                        f"analysis_config field '{_k}' is retired (moved to the "
                        f"per-HPC-system config / partition axis / sys.executable). "
                        f"It is ignored. Remove it from your analysis config YAML.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
        mode = values.get("run_mode")
        mpi = values.get("n_mpi_procs")
        omp = values.get("n_omp_threads")
        gpus = values.get("n_gpus")
        nodes = values.get("n_nodes")
        multi_sim_method = values.get("multi_sim_run_method")
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

            # Phase-4 (4d): the hpc_gpus_per_node requirement (GPU mode) is retired
            # here — per-node GPU topology is resolved from the ensemble partition's
            # PartitionSpec and the workflow emitter asserts a positive count at
            # Snakefile-generation time. The hpc_max_simultaneous_sims requirement
            # (batch_job) moved to hpc_system_config.max_concurrent_jobs validation.

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
