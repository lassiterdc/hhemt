import math
import re
import warnings
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from hhemt.config.base import cfgBaseModel, field_meta, when
from hhemt.config.eda import eda_config

# One-way import: config/analysis.py imports report_config; config/report.py
# must not import from config/analysis.py to avoid circular import.
from hhemt.config.report import report_config as _report_config_model

# The DISTINGUISHED value the two weather_event_windows_csv column-name fields carry
# until the user names their own columns. It is a real default rather than None so the
# fields stay plainly Optional -- the window CSV itself is optional, so a config that
# omits it must not be required to answer a question about its columns.
#
# It is a SENTINEL, not a name chosen to be improbable, and every consumer MUST test
# for it BEFORE testing whether the named column exists. A CSV that happened to carry
# a column literally named "unspecified" would otherwise pass an existence check and be
# read as the window -- the exact silent-wrong-column accident the sentinel prevents.
#
# Declared once, here, so the Field default and every check share one source; a repeated
# string literal is how the default and the check drift apart.
WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED = "unspecified"

ClearRawValue = Literal["all", "none"] | list[Literal["tritonswmm", "triton", "swmm"]]
# Artifact-class vocabulary, ordered by REGENERATION COST rather than by filename.
# The partition rule against `clear_raw`, stated so a future artifact classifies
# itself without a lookup table: `clear_raw` decides which MODEL's raw solver
# outputs to delete (its members ARE model types); `remove_after_processing`
# decides which ARTIFACT CLASSES to remove once the per-model summaries are
# provably intact. An artifact identified by the model that produced it is a
# clear_raw concern; anything else is a class here.
#
# ALIAS RENAMED WITH THE FIELD (round 5, user directive): RemoveValue ->
# RemoveValue. The alias IS this field's type and appears in public signatures,
# so leaving it behind would put two names on one concept in the file the rename
# exists to clean. The four PRIVATE helpers keep `_reclaim_*` names -- see the
# rename-boundary ruling in the round-5 amendment.
#
#   T0 (regenerable by template-fill, no solver): prep_inputs
#   T1 (regenerable by re-running the CHEAP SWMM hydrology solver): hydrographs,
#      hydro_out -- BUT ONLY while the other of the pair survives; electing BOTH
#      makes a re-prep hard-fail at hydrograph_outputs_gate and recovery requires
#      clearing hydro_swmm_sim_completed and re-simulating hydrology by hand.
#   T2 (regenerable only by re-running an EXPENSIVE solver): timeseries,
#      raw_swmm_binaries, coupled_rpt, standalone_rpt
RemoveValue = Literal["all", "none"] | list[
    Literal[
        "timeseries",
        "raw_swmm_binaries",
        "coupled_rpt",
        "hydro_out",
        "prep_inputs",
        "hydrographs",
        "standalone_rpt",
    ]
]
# Deliberately NOT a widening of ClearRawValue: that alias is ALSO ForceRerunSpec.subject's
# type (see the comment directly below), so adding an artifact-class member to it would make
# that member a legal force-rerun subject, where it means nothing. Different axis --
# ClearRawValue enumerates MODEL TYPES over raw sim outputs; RemoveValue enumerates
# ARTIFACT CLASSES over post-processing redundancy -- so a different alias.
# Legacy shape, RETAINED as the accepted input form and as `ForceRerunSpec.subject`'s type.
# Every existing config value stays valid and keeps its exact present meaning; the
# `mode="before"` coercion below maps it to `stage="simulate"`, which is what force_rerun
# has always done (it deletes an upstream sim flag and lets Snakemake cascade downstream --
# see workflow.py::_delete_flags_for_force_rerun).
ForceRerunSubject = Literal["all", "none"] | dict[Literal["sa_id", "event_iloc"], list[int | str]]


class ForceRerunSpec(cfgBaseModel):
    """WHICH simulations to force, and from WHICH stage down.

    Two ORTHOGONAL axes as separate fields rather than keys in one dict. The subject keys
    are mutually exclusive by construction -- `member_id` requires `toggle_sensitivity_analysis`
    True and `event_iloc` requires it False -- which is why the shipped validator's
    `next(iter(...))` is correct. A `stage` key added to that same dict would make the
    first-key read insertion-order-dependent and silently skip subject validation.

    `stage` is a FLOOR, not a set: forcing from a stage re-runs it and everything
    downstream. That is the shipped semantics twice over -- by Snakemake cascade in
    `force_rerun`, and by strict containment in
    `reprocess_snakefile_generator.START_STAGES`.

    Base is `cfgBaseModel`, which already supplies `extra="forbid"`; its `_check_paths_exist`
    field validator skips non-Path fields and is therefore inert here.
    """

    subject: ForceRerunSubject = "none"
    stage: Literal["simulate", "process", "consolidate", "render"] = Field(
        "simulate",
        description=(
            "Earliest stage to force; that stage and everything downstream re-run. "
            "'simulate' (default) preserves the historical force_rerun meaning. "
            "'render' -- the plot + export + render rule family that produces "
            "analysis_report.html. Bundling is NOT on this axis: it emits no Snakemake "
            "rule and is always explicit and user-invoked."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy(cls, value):
        """Accept every historical value unchanged.

        `"all"` / `"none"` / `{"sa_id": [...]}` / `{"event_iloc": [...]}` all map to
        `subject=<value>, stage="simulate"` -- fixed points of the coercion. Only a mapping
        that already carries `subject` or `stage` is treated as the new form, so a legacy
        `{"sa_id": [...]}` can never be misread as it.
        """
        if isinstance(value, ForceRerunSpec):
            return value
        if isinstance(value, dict) and ("subject" in value or "stage" in value):
            return value
        return {"subject": value}

    @model_validator(mode="after")
    def _validate_subject_shape(self):
        """The dict-shape rules, lifted from `analysis_config._validate_force_rerun`.

        They live HERE rather than on analysis_config because they are properties of the
        SUBJECT VALUE and need no sibling-field context -- so one implementation covers both
        the legacy flat form and the two-axis form, which are the same `subject` by the time
        this runs. Left on analysis_config they validated only the shape analysis_config
        happened to inspect, which is how they went silent when the field type changed.

        Message text is PRESERVED VERBATIM from the original validator: the
        test_force_rerun_rejects_invalid_dict_shapes params match on these strings.

        DEFINITION ORDER IS LOAD-BEARING: this runs before _reject_subject_scoped_render so
        a malformed subject reports its own shape error rather than the render error.
        """
        v = self.subject
        MEMBER_ID_RE = re.compile(r"^[A-Za-z0-9_.]+$")
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
                bad = [str(x) for x in values if not MEMBER_ID_RE.match(str(x))]
                if bad:
                    raise ValueError(
                        f"force_rerun.member_id values must match ^[A-Za-z0-9_.]+$ "
                        f"(per accepted decision 'All user-provided identifiers that "
                        f"become Snakemake wildcards must match ^[A-Za-z0-9_.]+$'); "
                        f"got invalid: {bad}"
                    )
        return self

    @model_validator(mode="after")
    def _reject_subject_scoped_render(self):
        """A subject-scoped ``render`` floor is not a capability of this design.

        `plots/` is not partitioned by subject: alongside the per-sim figures that carry
        `__member.{member_id}__` it holds CROSS-SUB aggregates (`b4b_clean_identity`,
        `config_diff_maps`, `eda_cross_hardware_magnitude`) whose inputs span every
        member and which carry no subject token at all. Honouring a subject-scoped
        render would refresh some figures and leave the aggregates stale -- an artifact
        inconsistent with its own inputs, which is the defect class the stage axis exists
        to remove. So the render floor is subject-blind, and asking for a subset must FAIL
        LOUD rather than silently re-render more than the caller requested.

        The other three floors ARE subject-composable -- their flags are per-subject and
        Snakemake's cascade re-derives everything downstream consistently.
        """
        if self.stage == "render" and isinstance(self.subject, dict):
            requested = ", ".join(f"{k}={v!r}" for k, v in self.subject.items())
            raise ValueError(
                "force_rerun stage='render' cannot be scoped to a subject "
                f"({requested}): the render stage's outputs are not partitioned by "
                "subject -- plots/ also holds cross-sub aggregate figures with no subject "
                "token -- so a scoped re-render would leave those aggregates stale. Use "
                "{'subject': 'all', 'stage': 'render'} to re-render everything, or choose "
                "an earlier floor ('process' / 'consolidate'), which ARE subject-scopable "
                "and whose cascade re-renders consistently."
            )
        return self


# Back-compat alias; the coercion accepts every legacy form. Deliberately NOT a union with
# ForceRerunSubject: measured, smart-union resolves heterogeneously (str | dict | Spec by
# input), which is the multi-shape problem this change exists to remove.
ForceRerunValue = ForceRerunSpec


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
        json_schema_extra=field_meta(
            options={
                "mm": "Depth accumulated over each timestep (an incremental depth).",
                "mm/hr": "Instantaneous intensity at each timestep (a rate).",
            }
        ),
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
        json_schema_extra=field_meta(
            options={
                "CC0-1.0": (
                    "Public-domain dedication. No attribution required; the "
                    "regret-safe choice across immutable DOIs."
                ),
                "CC-BY-NC-4.0": (
                    "Attribution required, commercial use prohibited. "
                    "'NonCommercial' is broader than 'education only' and does "
                    "not turn on who the user is."
                ),
            }
        ),
    )
    # COMPUTE CONFIG
    run_mode: Literal["serial", "openmp", "mpi", "hybrid", "gpu"] = Field(
        ...,
        description=(
            "Per-simulation compute configuration. Selects which parallelism the "
            "solver uses, and therefore which of n_mpi_procs / n_omp_threads / "
            "n_gpus must be set."
        ),
        json_schema_extra=field_meta(
            options={
                "serial": "One process, one thread, no GPU. n_mpi_procs and n_omp_threads must be 1 or unset.",
                "openmp": "Shared-memory threading on one node. n_omp_threads must be > 1; no MPI, no GPU.",
                "mpi": "Distributed MPI ranks, one thread each. n_mpi_procs must be > 1 and >= n_nodes; no GPU.",
                "hybrid": "MPI ranks each running OpenMP threads. Both n_mpi_procs and n_omp_threads must be > 1; no GPU.",
                "gpu": "GPU-accelerated solve. n_gpus must be >= 1; MPI and OpenMP are optional alongside it.",
            }
        ),
    )
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
        # The per-option glossary moved OUT of this prose string and INTO the
        # `options` declaration below, which the report renders deterministically
        # and which `__pydantic_init_subclass__` holds in parity with the Literal.
        description="How the simulation ensemble is dispatched across scenarios.",
        json_schema_extra=field_meta(
            options={
                "local": "ThreadPoolExecutor in-process on the local machine. No SLURM.",
                "batch_job": "tmux session running Snakemake on the login node, submitting one SLURM job per scenario. Recommended for HPC.",
                "1_job_many_srun_tasks": "A single SLURM job that runs the scenarios as concurrent srun tasks inside one allocation.",
            }
        ),
    )
    hpc_total_nodes: int | None = Field(
        None,
        description="This is the total number of nodes that will be requested when multi_sim_run_method = 1_job_many_srun_tasks",
    )
    hpc_total_job_duration_min: int | None = Field(
        None,
        # Description CORRECTED: the previous text named `1_job_many_srun_tasks`
        # while the validator required this field under `batch_job`. The two
        # disagreed, and the report printed the description. Applicability and
        # requiredness are now DECLARED below and enforced from that declaration,
        # so the two cannot diverge again.
        description=(
            "Total wall-clock duration, in minutes, requested for the SLURM job "
            "that runs the simulation ensemble. Also used as the sim job's "
            "plausible-lifetime bound by the wait-on-sentinel poll cap and the "
            "stale-token fail-safe; when unset those fall back to "
            "hpc_max_wait_for_inflight_min."
        ),
        json_schema_extra=field_meta(
            applies_when=[when("multi_sim_run_method", "batch_job", "1_job_many_srun_tasks")],
            required_when=[when("multi_sim_run_method", "batch_job")],
        ),
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
    hpc_no_progress_timeout_min: int | None = Field(
        None,
        ge=10,
        le=10080,
        description=(
            "Backstop cap (minutes) on the tmux orchestrator's no-progress stall "
            "watchdog (_wait_for_tmux_session_completion). As of the watchdog "
            "liveness-gate change, the stall timer is SUSPENDED while SLURM "
            "reports at least one live job for the run, so this cap is a pure "
            "safety backstop (NOT a queue-wait tolerance) - it fires only when "
            "Snakemake is alive AND SLURM has no live work AND _status/ has not "
            "advanced. None (default) keeps the legacy walltime-derived fallback "
            "max(30, 6 * hpc_time_min_per_sim), which is NOT queue-aware on its "
            "own and is retained only so existing configs are unchanged. Bounds: "
            "10 (10 min) to 10080 (1 week). Mirrors hpc_max_wait_for_inflight_min."
        ),
    )
    hpc_max_queue_wait_min: int | None = Field(
        None,
        ge=30,
        le=10080,
        description=(
            "Cap (minutes) on how long the tmux orchestrator tolerates a live SLURM "
            "job set that is continuously ALL-PENDING (never RUNNING). Distinct from "
            "hpc_no_progress_timeout_min by construction: that one bounds the "
            "SLURM-SILENT state (a hang), this one bounds the QUEUE-STARVED state "
            "(partition contention). Reusing one number for both would recreate the "
            "walltime-coupling defect the liveness gate exists to remove. Any RUNNING "
            "observation resets the tracker. On expiry the session is killed - which "
            "cancels the queued jobs, since the SLURM executor runs scancel on "
            "interrupt - and a distinct queue-starved verdict is returned. None "
            "(default) means 720 (12 h). Bounds: 30 (30 min) to 10080 (1 week)."
        ),
    )
    resume_interruption_schedule: tuple[int, ...] | None = Field(
        default=None,
        description=(
            "Resume-test harness ONLY. A STRICTLY INCREASING tuple of ABSOLUTE "
            "hotstart-checkpoint indices at which a fresh TRITON/TRITON-SWMM sim is "
            "interrupted, producing len(schedule) hotstart resumes. The unit is the "
            "one the watcher already counts (config_NNNN.cfg files) and the one "
            "return_the_reporting_step_from_a_cfg returns, so no conversion exists "
            "anywhere in the mechanism and interval alignment is vacuous. An entry "
            "beyond the sim's checkpoint count degrades gracefully (no kill fires). "
            "Empty tuples, duplicates, non-positive entries and non-increasing "
            "sequences are rejected. The runner arms each kill "
            "from the PERSISTED n_resumes counter relative to the attempt's own "
            "baseline, never from an absolute count over the cfg directory (that "
            "directory accumulates across attempts, so an absolute predicate is "
            "already true on attempt 2 before any forward progress). Default None "
            "DISABLES the harness — production and clean-arm runs leave it unset and "
            "the runner path is byte-identical to a plain proc.wait(). Incompatible "
            "with multi_sim_run_method='1_job_many_srun_tasks' (rejected at "
            "preflight): that mode does not get the job-end cgroup reap that makes "
            "repeated per-attempt step teardown structurally safe under batch_job."
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
            "(run_triton/run_tritonswmm/run_swmm/simulation_member_*). A walltime "
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
    hpc_cpu_sim_partition: str | None = Field(
        None,
        description=(
            "SLURM partition for CPU-ONLY simulation rules (currently run_swmm). When None "
            "this resolves to hpc_setup_and_analysis_processing_partition, then to "
            "hpc_ensemble_partition. Exists because a CPU-only sim carries no --gres, so a "
            "GPU partition whose QOS enforces a GRES minimum rejects it at submit "
            "(UVA Rivanna -p gpu -> `sbatch: error: QOSMinGRES`). Set this only when the "
            "CPU sim rung needs a different partition from the setup/processing rungs."
        ),
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
        json_schema_extra=field_meta(required_when=[when("toggle_storm_tide_boundary", True)]),
    )
    storm_tide_units: str | None = Field(
        None,
        description="Storm tide units, e.g., ft, m. Must align with units used DEM.",
        json_schema_extra=field_meta(required_when=[when("toggle_storm_tide_boundary", True)]),
    )
    weather_event_summary_csv: Path | None = Field(
        None,
        description="CSV file with weather event summary statistics. Events must share indices with weather_timeseries.",
    )
    weather_time_series_storm_tide_datavar: str | None = Field(
        None,
        description="Data variables in weather_timeseries corresponding to storm tide.",
        json_schema_extra=field_meta(required_when=[when("toggle_storm_tide_boundary", True)]),
    )
    sensitivity_analysis: Path | None = Field(
        None,
        description="sensitivity analysisal design csv file.",
        json_schema_extra=field_meta(required_when=[when("toggle_sensitivity_analysis", True)]),
    )
    weather_events_to_simulate: Path = Field(
        ...,
        description="Path to a .csv file defining weather event index used for sensitivity. The columns must correspond to the sytem's weather_event_indices.",
    )
    weather_event_windows_csv: Path | None = Field(
        None,
        description=(
            "CSV declaring the simulation window for each weather event. Its index columns "
            "must be exactly weather_event_indices, plus a start and an end column whose "
            "names are given by weather_event_start_column / weather_event_end_column. The "
            "toolkit CLIPS each event to the declared window and never inspects the weather "
            "data to decide one. WHEN ABSENT, each event's window is the full extent of its "
            "time coordinate -- the file as given. The toolkit still refuses to run an event "
            "whose forcing is incomplete over that extent; supplying this CSV is how you "
            "declare a shorter window that IS complete."
        ),
    )
    weather_event_start_column: str = Field(
        WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED,
        description=(
            "Column in weather_event_windows_csv carrying each event's window START, parsed "
            "with pandas.to_datetime. The NAME is a config input so the toolkit binds no "
            "column name of its own: the default is the SENTINEL 'unspecified', which names "
            "no real column and is rejected before any column is read. Only relevant when "
            "weather_event_windows_csv is supplied -- without that file there is no CSV to "
            "name a column in, and the sentinel simply goes unread. When the file IS "
            "supplied, both column names must be stated and both must exist in it."
        ),
    )
    weather_event_end_column: str = Field(
        WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED,
        description=(
            "Column in weather_event_windows_csv carrying each event's window END, parsed "
            "with pandas.to_datetime. Inclusive: the clipped slice contains the end stamp. "
            "Sentinel-defaulted and validated exactly as weather_event_start_column."
        ),
    )

    @model_validator(mode="after")
    def _windows_csv_is_not_the_summary_csv(self):
        """Reject the single most probable misconfiguration.

        A tripwire on the likeliest operator action, not a fix: a COPY of the summary
        file under another name walks past it. The real protection is that the two
        column-name fields above default to a SENTINEL rather than to a plausible column
        name, so the summary CSV's own headers are never swept in by default -- together
        with the endpoint-equality assertion at the clip.
        """
        w, s = self.weather_event_windows_csv, self.weather_event_summary_csv
        if w is not None and s is not None and Path(w).resolve() == Path(s).resolve():
            raise ValueError(
                "weather_event_windows_csv and weather_event_summary_csv are the same "
                "file. The summary CSV describes events in real calendar dates; the "
                "window CSV must carry stamps on the weather file's own time axis."
            )
        return self

    weather_event_label_column: str | None = Field(
        None,
        description=(
            "Optional column name in weather_events_to_simulate carrying a human-readable "
            "display name for each event (e.g. '2003-09-17 05:44 - Hurricane Isabel'). "
            "Presentation only: the value NEVER enters weather_event_indices, the event_id "
            "slug, a figure stem, or any path segment. The column is renamed to the toolkit "
            "canonical name at projection, so downstream consumers never see this value. "
            "None disables labelling and every report surface falls back to its current text."
        ),
    )
    analysis_description: str | None = Field(
        None,
        description="For readability.",
    )

    # TRITON-SWMM PARAMETERS
    target_processed_output_type: Literal["zarr", "nc"] = Field(
        "zarr",
        description="TRITON processed output type, zarr or nc.",
        json_schema_extra=field_meta(
            options={
                "zarr": "Chunked Zarr store. Default and preferred -- supports lazy and partial reads.",
                "nc": "One NetCDF file per scenario. Use when a downstream consumer requires NetCDF.",
            }
        ),
    )
    toggle_consolidate_timeseries: bool = Field(
        default=False,
        description=(
            "When True, the per-scenario SWMM node (wlevel(t)) and SWMM link (flow(t)) "
            "TIMESERIES are consolidated into the analysis/sensitivity DataTree under "
            "tritonswmm/swmm_node_timeseries and tritonswmm/swmm_link_timeseries "
            "(concatenated along event_iloc). Default False (summaries only) — enable to "
            "unblock over-time clean-vs-resume difference figures. The TRITON gridded "
            "timeseries is NOT consolidated: it is ~83x larger per scenario by payload "
            "(6.6 MB vs 0.08 MB measured), would inflate a 28-scenario master tree 7.9x, "
            "and is GB-scale per scenario on a production fine grid. It is excluded on "
            "COST, not for lack of a consumer — the clean-vs-resume over-time figure does "
            "read wlevel_m from the per-scenario gridded store, via a documented estate "
            "reproduction script. What that figure consumes is a per-timestep "
            "max-over-(y,x) reduction (144 values/scenario), not the field; consolidating "
            "the reduction is the cheap, grid-independent extension. "
            "LAYOUT_VERSION bump (config/analysis.py is allowlisted; the tree-node add is "
            "additive-read-compatible)."
        ),
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
        json_schema_extra=field_meta(
            options={
                "bin": "Raw binary rasters. Default -- smaller and faster to write and parse.",
                "asc": "ASCII grid rasters. Human-readable, substantially larger and slower.",
            }
        ),
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
    is_experiment_member: bool = Field(
        False,
        description="This is used in the backend to help route members to appropriate processes.",
    )
    experiment_cfg_yaml: Path | None = Field(
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
            "per-member overlayable via an `analysis.brand_theme` "
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
        json_schema_extra=field_meta(
            options={
                "native": "Compile, simulate and process on the host. Today's behavior, byte-identical.",
                "container": (
                    "Wrap the sim executable and the process_{model} runners in "
                    "`apptainer exec {sif}`, per hpc_system_config.container."
                ),
            }
        ),
    )

    # CLEANUP / FORCE-RERUN POLICY (cleanup-rerun-delete-redesign Phase 1)
    #
    # TWO KNOBS, TWO AXES -- do not conflate them. `clear_raw` is per-MODEL-TYPE over RAW
    # SIMULATION outputs and fires inside write_timeseries_outputs.
    # `remove_after_processing` is per-ARTIFACT-CLASS over POST-PROCESSING REDUNDANCY and
    # fires in the process runner only after the summaries are provably intact.
    remove_after_processing: RemoveValue = Field(
        "none",
        description=(
            "Post-processing reclaim policy. Fires ONLY after the per-model summary "
            "outputs are verified present AND openable on disk. \"none\" reclaims "
            "nothing. \"all\" reclaims every artifact class. A list selects classes: "
            "\"timeseries\" drops the per-scenario *_tseries zarr/nc set (the paired "
            "complement of summary_paths._SUMMARY_STEMS_BY_MODEL -- no renderer and no "
            "default consolidation path reads them); \"raw_swmm_binaries\" drops "
            "out_tritonswmm/swmm/*.out (hydraulics.out plus the per-node *.out set), and "
            "NEVER *.rpt, which is a live completion predicate via "
            "run_simulation._coupled_swmm_report_finalized. \"raw_swmm_binaries\" "
            "TRUNCATES out_tritonswmm/swmm/hydraulics.rpt in place to its header, "
            "continuity and summary tables plus its trailer -- truncation, NEVER deletion, "
            "because that file is a live completion predicate via "
            "run_simulation._coupled_swmm_report_finalized. \"hydro_out\" drops the SWMM "
            "hydrology output swmm/hydro.out, which write_hydrograph_files reads; that is "
            "safe only because write_hydrograph_files carries an already-written gate. "
            "\"raw_swmm_binaries\" "
            "no-ops (with a logged reason) when clear_raw == \"none\", because its only "
            "consumer, eda.raw_resume_identity.compare_swmm_raw, also needs the raw "
            "H/QX/QY/MH set that clear_raw governs. Every reclaim is recorded per "
            "scenario in the per-model log and surfaced by "
            "analysis_validation.check_data_availability. Defaults to \"none\" -- "
            "existing yamls load cleanly with the strict-safe (reclaim-nothing) default. "
            "Reclaiming makes REPROCESSING those scenarios impossible without re-running "
            "the simulations."
        ),
    )
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
            "or string identifiers re-runs only the named members or "
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

    @field_validator("remove_after_processing", mode="after")
    @classmethod
    def _validate_remove_after_processing(cls, v):
        """Mirror of _validate_clear_raw's three reject arms, on the artifact-class axis."""
        if isinstance(v, list):
            if not v:
                raise ValueError(
                    "remove_after_processing list form cannot be empty; use 'none' to reclaim nothing"
                )
            if len(v) != len(set(v)):
                raise ValueError(f"remove_after_processing list contains duplicates: {v}")
            for item in v:
                if item in ("all", "none"):
                    raise ValueError(
                        f"remove_after_processing list cannot contain sentinel value {item!r}; "
                        f"use the sentinel as a bare string (remove_after_processing: {item})"
                    )
        return v

    @field_validator("resume_interruption_schedule", mode="after")
    @classmethod
    def _validate_resume_interruption_schedule(cls, v):
        if v is None:
            return v
        if not v:
            raise ValueError(
                "resume_interruption_schedule cannot be an empty tuple; use None to disable the harness"
            )
        if len(v) != len(set(v)):
            raise ValueError(f"resume_interruption_schedule contains duplicates: {v}")
        if any(entry < 1 for entry in v):
            raise ValueError(
                f"resume_interruption_schedule entries must be >= 1 (absolute checkpoint indices): {v}"
            )
        if any(b <= a for a, b in zip(v, v[1:], strict=False)):
            raise ValueError(
                f"resume_interruption_schedule must be strictly increasing with no duplicates: {v}"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_toggle_dependencies(cls, values):
        errors = []

        # Conditional requiredness for these four fields is DECLARED on the fields via
        # `field_meta(required_when=...)` and enforced by
        # `cfgBaseModel._enforce_required_when`. See the sibling note in config/system.py
        # for why the hand-written second enforcement site was removed.

        if errors:
            raise ValueError("; ".join(errors))
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_analysis_requirements(cls, values):
        errors = []
        if values.get("is_experiment_member") is True:
            if values.get("experiment_cfg_yaml") is None:
                errors.append("experiment_cfg_yaml must be provided when is_experiment_member=True")
            if values.get("analysis_dir") is None:
                errors.append("analysis_dir must be provided when is_experiment_member=True")

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

        # PRESENCE is enforced by the declaration on the field itself
        # (json_schema_extra=field_meta(required_when=[when("multi_sim_run_method",
        # "batch_job")]), consumed by cfgBaseModel._enforce_required_when). Do NOT
        # re-check `is None` here: a second enforcement site is exactly what let the
        # rendered "Required" cell drift from what was enforced. Only the VALUE
        # bound, which the declaration grammar does not express, remains here.
        if (
            multi_sim_method == "batch_job"
            and hpc_total_job_duration_min is not None
            and hpc_total_job_duration_min < 1
        ):
            raise ValueError("hpc_total_job_duration_min must be > 0 for multi_sim_run_method=batch_job")

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
        # Read the SUBJECT axis. With force_rerun pinned to ForceRerunSpec the outer value
        # is never a dict, so `isinstance(self.force_rerun, dict)` would be always False and
        # this guard would silently validate NOTHING -- accepting {"sa_id": [...]} under
        # toggle_sensitivity_analysis=False, which is precisely what it exists to reject.
        # getattr with a fallback keeps this correct if the value ever arrives unwrapped.
        subject = getattr(self.force_rerun, "subject", self.force_rerun)
        if isinstance(subject, dict):
            key = next(iter(subject))
            if key == "sa_id" and not self.toggle_sensitivity_analysis:
                raise ValueError("force_rerun.member_id requires toggle_sensitivity_analysis=True")
            if key == "event_iloc" and self.toggle_sensitivity_analysis:
                raise ValueError(
                    "force_rerun.event_iloc requires toggle_sensitivity_analysis=False; "
                    "sensitivity-toggled analyses must use force_rerun.member_id instead"
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
