"""The per-analysis configuration an hhemt run reads from its analysis YAML.

One analysis config describes how a set of simulations is run: the compute shape of
each simulation (run mode, ranks, threads, GPUs, nodes), how the ensemble is dispatched
(in-process, one SLURM job per scenario from a login-node tmux session, or many srun
tasks inside one allocation), the partition selectors and allocations that dispatch
needs, the model toggles and their inputs, the cleanup and force-rerun policy, and the
report settings. Validation here rejects an inconsistent combination when the YAML
loads, before any job is submitted. Most fields that are required only under some
other setting declare that condition on the field itself
(`field_meta(required_when=...)`), which is both what enforces it and what the report's
Reproduction Guide reads. Two groups do not, and the Guide does not see them: the
compute-shape counts each `run_mode` requires (e.g. `n_omp_threads` under `openmp`),
enforced in `check_consistency`, and `experiment_cfg_yaml` / `analysis_dir` under
`is_experiment_member=True`, enforced in `validate_analysis_requirements`.
"""

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

#: The depth (m) at or below which the TRITON flood velocity is undefined. The solver zeroes its
#: unit discharge below its own ``hextra`` (0.001 m in every run measured), so |q| / h is 0 / 0
#: there. Shared by the ``flood_velocity_min_depth_m`` config field and the summary function's
#: default, so a caller that passes nothing gets the configured default and not a second number.
DEFAULT_FLOOD_VELOCITY_MIN_DEPTH_M = 0.001

ClearRawValue = Literal["all", "none"] | list[Literal["tritonswmm", "triton", "swmm"]]
# Artifact-class vocabulary, ordered by REGENERATION COST rather than by filename.
# The partition rule against `clear_raw`, stated so a future artifact classifies
# itself without a lookup table: `clear_raw` decides which MODEL's raw solver
# outputs to delete (its members ARE model types); `remove_after_processing`
# decides which ARTIFACT CLASSES to remove once the per-model summaries are
# provably intact. An artifact identified by the model that produced it is a
# clear_raw concern; anything else is a class here.
#
# The alias is named for its field, `remove_after_processing`. The process-side helpers
# that apply it (`_reclaim_classes`, `_reclaim_attrs`, `_reclaim_paths` in
# process_simulation.py) keep `_reclaim_*` names.
#
#   T0 (regenerable by template-fill, no solver): prep_inputs
#   T1 (regenerable by re-running the CHEAP SWMM hydrology solver): hydrographs,
#      hydro_out -- BUT ONLY while the other of the pair survives; electing BOTH
#      makes a re-prep hard-fail at hydrograph_outputs_gate and recovery requires
#      clearing hydro_swmm_sim_completed and re-simulating hydrology by hand.
#   T2 (regenerable only by re-running an EXPENSIVE solver): timeseries,
#      raw_swmm_binaries, coupled_rpt, standalone_rpt
RemoveValue = (
    Literal["all", "none"]
    | list[
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
)
#: The reclaim policy a config gets when it names none: every class whose content the processed output also
#: holds. "timeseries" is excluded because the consolidation step copies those stores before clearing them,
#: and "hydro_out" because electing it with "hydrographs" makes a re-prep hard-fail.
_DEFAULT_REMOVE_AFTER_PROCESSING: list[str] = [
    "raw_swmm_binaries",
    "coupled_rpt",
    "prep_inputs",
    "hydrographs",
    "standalone_rpt",
]
# Deliberately NOT a widening of ClearRawValue: that alias types `clear_raw` and every
# `override_clear_raw` parameter, so adding an artifact-class member to it would make that
# member a legal `clear_raw` value, where it means nothing. Different axis --
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

    Two ORTHOGONAL axes as separate fields rather than keys in one dict. A dict subject
    must hold exactly one key, and `_validate_subject_shape` rejects any other length
    (`len(v) != 1`) before it reads the key with `next(iter(...))`, so that read is
    unambiguous. Which key is allowed also depends on the analysis -- an `sa_id` key
    requires `toggle_sensitivity_analysis` True and `event_iloc` requires it False --
    and that is enforced on the outer `analysis_config`, not here. Putting `stage` in the
    same dict would need the one-key rule relaxed, and the first-key read would then
    depend on insertion order and could silently skip subject validation.

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
            "Earliest stage to force. The stage and everything downstream re-run. 'simulate', the default, "
            "preserves the historical force_rerun meaning. 'render' covers the plot, export, and render "
            "rule family that produces analysis_report.html. Bundling is NOT on this axis, because "
            "bundling emits no Snakemake rule and is always explicit and invoked by the caller."
        ),
    )
    models: list[Literal["triton", "tritonswmm", "swmm"]] | None = Field(
        None,
        description=(
            "Model arms to force. None, the default, means every model type enabled for this analysis, as "
            "in the historical behaviour. Naming a subset leaves the unnamed arms' completion state "
            "untouched, including their flags, their per-model processing-log records, and their chapter "
            "sets, so a correct, complete arm is not invalidated by a force aimed at a damaged arm."
        ),
    )

    @model_validator(mode="after")
    def _validate_models_shape(self):
        """`models` is None or a non-empty list of distinct model types.

        An EMPTY list is rejected rather than treated as "force nothing": it is
        indistinguishable in effect from `subject="none"` and is far more likely a
        mistake than an intent. Duplicates are rejected because they cannot change the
        result and their presence signals the caller believes the field means something
        it does not.

        Whether the named models are ENABLED for this analysis is deliberately NOT
        checked here -- that needs sibling-field context this model does not hold (it is
        `system_config`'s toggles), and it is validated in the orchestrator module
        `src/hhemt/analysis.py` (not this config module), beside the override subject's
        toggle cross-check.
        """
        if self.models is None:
            return self
        if not self.models:
            raise ValueError(
                "force_rerun.models must be omitted (meaning every enabled model) or a "
                "non-empty list; an empty list is not a valid subject."
            )
        if len(set(self.models)) != len(self.models):
            raise ValueError(f"force_rerun.models contains duplicate entries: {self.models}")
        return self

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
        """The dict-shape rules for `subject`, which once lived on `analysis_config`.

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
                        # `{key}` rather than a literal: this branch is entered only when
                        # key == "sa_id", and interpolating the key the type actually
                        # admits makes a wrong-key message structurally unwritable here.
                        f"force_rerun.{key} values must match ^[A-Za-z0-9_.]+$ "
                        f"(per accepted decision 'All user-provided identifiers that "
                        f"become Snakemake wildcards must match ^[A-Za-z0-9_.]+$'); "
                        f"got invalid: {bad}"
                    )
        return self

    @model_validator(mode="after")
    def _reject_subject_scoped_render(self):
        """A subject-scoped ``render`` floor is not a capability of this design.

        `plots/` is not partitioned by subject: alongside the per-sim figures that carry
        a `__member.{member_id}` segment it holds CROSS-SUB aggregates (`raw_byte_identity`,
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
            description="Analysis identifier. Used for creating the analysis folder when no folder with "
            "the same name exists.",
            pattern=r"^[A-Za-z][A-Za-z0-9_.]*$",
        ),
    ]
    weather_event_indices: list = Field(
        ...,
        description="List of one or more strings corresponding to fields used for indexing unique weather "
        "events. The strings must match the fields in weather_timeseries and "
        "weather_event_summary_csv.",
    )
    weather_timeseries: Path = Field(
        ...,
        description="Netcdf containing weather event time series data. Events must share indices with "
        "weather_event_summary_csv.",
    )
    weather_time_series_timestep_dimension_name: str = Field(
        ...,
        description="Dimension in weather_timeseries corresponding to timestep.",
    )
    weather_time_series_spatial_mean_rainfall_datavar: str = Field(
        ...,
        description=(
            "Data variable in weather_timeseries corresponding to the spatially-averaged rainfall time "
            "series that the report renderers read for the rainfall sub-panel, namely the event-hydrology "
            "panels of per_sim_peak_flood_depth and per_sim_conduit_flow. Required."
        ),
    )
    rainfall_units: Literal["mm", "mm/hr"] = Field(
        ...,
        description="Rainfall units in weather_timeseries. The value is mm or mm/hr.",
        json_schema_extra=field_meta(
            options={
                "mm": "Depth accumulated over each timestep, an incremental depth.",
                "mm/hr": "Instantaneous intensity at each timestep, a rate.",
            }
        ),
    )
    # DATASET PUBLISHING
    dataset_license: (
        Literal[
            "CC0-1.0",
            "CC-BY-4.0",
            "CC-BY-SA-4.0",
            "CC-BY-NC-4.0",
            "CC-BY-NC-SA-4.0",
            "CC-BY-ND-4.0",
            "CC-BY-NC-ND-4.0",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "SPDX identifier of the license for the dataset this analysis PUBLISHES, meaning the "
            "consolidated outputs. No default exists, and only publishing needs a license. A config "
            "without a license loads, runs, consolidates, and bundles normally, and `publish()` refuses to "
            "deposit until a license is chosen. The value is written into the RO-Crate root "
            "Dataset.license when the outputs are consolidated and read back at publish time, so choosing "
            "a license after consolidating means re-running `reprocess(start_with='consolidate', "
            "regenerate_existing=True)` before publishing. Mutually exclusive with dataset_license_file, "
            "the field for a license outside the listed ids. Set either field and never both. "
            "'NonCommercial' is broader than 'education only' and applies to every reuser."
        ),
        json_schema_extra=field_meta(
            options={
                "CC0-1.0": (
                    "Public-domain dedication with no conditions at all. Commercial use and derivatives are "
                    "allowed, and no attribution is required."
                ),
                "CC-BY-4.0": "Attribution required. Commercial use and derivatives allowed.",
                "CC-BY-SA-4.0": (
                    "Attribution required. Commercial use and derivatives allowed, and derivatives must be "
                    "shared under this same license."
                ),
                "CC-BY-NC-4.0": "Attribution required. Derivatives allowed. Commercial use prohibited.",
                "CC-BY-NC-SA-4.0": (
                    "Attribution required. Commercial use prohibited. Derivatives allowed and must be shared "
                    "under this same license."
                ),
                "CC-BY-ND-4.0": "Attribution required. Commercial use allowed. No derivatives may be shared.",
                "CC-BY-NC-ND-4.0": ("Attribution required. Commercial use prohibited. No derivatives may be shared."),
            }
        ),
    )
    dataset_license_file: Path | None = Field(
        default=None,
        description=(
            "Path to a YAML license-details file, for a dataset license outside the SPDX ids that "
            "dataset_license accepts. The file is a YAML mapping carrying a required 'name', the human "
            "title of the license, and an optional 'link', the address where the owner publishes the "
            "license. No other key is read. Mutually exclusive with dataset_license. Set either field and "
            "never both, and neither is required at load because only publishing needs a license. Like "
            "dataset_license, the value is written into the RO-Crate root Dataset.license when the outputs "
            "are consolidated, so choosing the file after consolidating means re-running "
            "`reprocess(start_with='consolidate', regenerate_existing=True)` before publishing. The terms "
            "live in the file and not in this config, so the license of a deposit is reproducible only if "
            "the file is carried alongside the deposit. The bundle carries the file."
        ),
        json_schema_extra=field_meta(),
    )

    @model_validator(mode="after")
    def _at_most_one_dataset_license_source(self):
        """Refuse a config that declares its dataset license two ways.

        Two sources would make the crate, the published sidecar and the rights lists
        disagree about one dataset's terms with no rule saying which wins, and whichever
        one lost would be invisible in the deposit. Neither field is required: an analysis
        that never publishes carries neither, which is why this is a mutual exclusion and
        not a requiredness rule (D-106).
        """
        if self.dataset_license is not None and self.dataset_license_file is not None:
            raise ValueError(
                "dataset_license and dataset_license_file are both set. They are two ways of "
                "declaring the SAME dataset license and a deposit can carry only one: clear "
                "whichever does not describe the license, or clear both if this analysis will "
                "not be published."
            )
        return self

    # COMPUTE CONFIG
    run_mode: Literal["serial", "openmp", "mpi", "hybrid", "gpu"] = Field(
        ...,
        description=(
            "Per-simulation compute configuration. Selects the parallelism the solver uses, and therefore "
            "the fields among n_mpi_procs, n_omp_threads, and n_gpus that must be set."
        ),
        json_schema_extra=field_meta(
            options={
                "serial": "One process, one thread, no GPU. n_mpi_procs and n_omp_threads must be 1 or unset.",
                "openmp": "Shared-memory threading on one node. n_omp_threads must be > 1. MPI and GPU stay off.",
                "mpi": "Distributed MPI ranks, one thread each. n_mpi_procs must be > 1 and >= n_nodes. GPU stays off.",
                "hybrid": "MPI ranks each running OpenMP threads. Both n_mpi_procs and "
                "n_omp_threads must be > 1. GPU stays off.",
                "gpu": "GPU-accelerated solve. n_gpus must be >= 1. MPI and OpenMP are optional alongside the GPU.",
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
                "batch_job": "tmux session running Snakemake on the login node, submitting "
                "one SLURM job per scenario. Recommended for HPC.",
                "1_job_many_srun_tasks": "A single SLURM job that runs the scenarios as "
                "concurrent srun tasks inside one allocation.",
            }
        ),
    )
    hpc_total_nodes: int | None = Field(
        None,
        description="This is the total number of nodes that will be requested when multi_sim_run_method = "
        "1_job_many_srun_tasks",
    )
    hpc_total_job_duration_min: int | None = Field(
        None,
        # Description CORRECTED: the previous text named `1_job_many_srun_tasks`
        # while the validator required this field under `batch_job`. The two
        # disagreed, and the report printed the description. Applicability and
        # requiredness are now DECLARED below and enforced from that declaration,
        # so the two cannot diverge again.
        description=(
            "Total wall-clock duration, in minutes, requested for the SLURM job that runs the simulation "
            "ensemble. Also the plausible-lifetime bound of the sim job for the wait-on-sentinel poll cap "
            "and the stale-token fail-safe. When unset, both the poll cap and the fail-safe fall back to "
            "hpc_max_wait_for_inflight_min."
        ),
        json_schema_extra=field_meta(
            applies_when=[when("multi_sim_run_method", "batch_job", "1_job_many_srun_tasks")],
            required_when=[when("multi_sim_run_method", "batch_job")],
        ),
    )
    # hpc_gpus_per_node and hpc_cpus_per_node are RETIRED from analysis_config. Per-node
    # GPU/CPU topology now lives per-partition
    # on PartitionSpec.gpus_per_node / .cpus_per_node, resolved via the workflow
    # builder's _resolve_gpus_per_node / _resolve_cpus_per_node from the named
    # partition. A pop-and-warn shim (check_consistency, below) lets un-migrated
    # YAMLs still load. REMOVE the shim once un-migrated YAMLs no longer need to load; no
    # removal release has been set.
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
            "Memory allocation, in MB, for the setup_target SLURM rule. The rule runs system-input "
            "processing for DEM coarsening and the Manning's raster and, in native mode, asserts that "
            "every enabled model already has a successful build. The default of 12 GB covers 0.35 m DEM "
            "processing with 2.3x headroom over the empirical peak of about 5.15 GB parent-process RSS. "
            "The rule performs no compile and needs no compile-side headroom. Increase for "
            "higher-resolution DEMs or larger watersheds."
        ),
    )
    hpc_runtime_min_for_setup: int = Field(
        default=60,
        gt=0,
        description=(
            "Time allocation, in minutes, for the setup_target SLURM rule. The default 60 covers 0.35 m "
            "DEM processing with headroom over the empirical wall time of about 2:24. The rule performs no "
            "compile. Increase for higher-resolution DEMs or slower nodes."
        ),
    )

    hpc_runtime_min_for_sim_output_processing: int = Field(
        default=240,
        gt=0,
        description=(
            "Time allocation, in minutes, for the per-scenario output-processing SLURM rule whose MEMORY "
            "is set by hpc_mem_allocation_for_sim_output_processing_mb. The default 240 is the value every "
            "workflow path reads from this field. Raise the value for fine-grid or long simulations, "
            "because processing time scales with output VOLUME, the product of timesteps, grid cells, and "
            "model count. The field is a plain number and no value derived from the memory allocation, "
            "because no physical relation ties walltime to job RAM and inventing one would state an "
            "assumption as a computed value. The field is distinct from hpc_runtime_min_for_setup, a "
            "different rule for DEM processing and the build assertion, and from every process_* knob. The "
            "process_* knobs are in-runner byte and count budgets and no SLURM allocations."
        ),
    )

    hpc_runtime_min_for_analysis_output_consolidation: int = Field(
        default=30,
        gt=0,
        description=(
            "Time allocation, in minutes, for the analysis-level `consolidate` SLURM rule whose MEMORY is "
            "set by hpc_mem_allocation_for_analysis_output_consolidation_mb. The default 30 reproduces the "
            "value that the rule carried as a hardcoded literal, so an analysis config that omits this "
            "field is unchanged. Raise the value when the consolidation fan-in is large, because the cost "
            "of the rule is dominated by whole-population walks whose length scales with the number of "
            "scenarios and not with the size of any single scenario. The field is a plain number and no "
            "value derived from the memory allocation. The field is distinct from "
            "hpc_runtime_min_for_setup, a different rule for DEM processing and compilation, and from "
            "hpc_runtime_min_for_sim_output_processing, the per-scenario processing rule."
        ),
    )
    hpc_max_wait_for_inflight_min: int = Field(
        10080,
        ge=60,
        le=10080,
        description=(
            "Backstop cap, in minutes, on the poll loop of the v2 wait-on-sentinel rule. After the "
            "wait-rule in-loop-liveness change, the wait-rule detects job death in-loop with a per-job "
            "squeue/sacct probe about every 5 min and writes a _failed marker on confirmed death, so the "
            "cap is a pure safety backstop and no walltime-derived limit. The cap fires only if a job is "
            "stuck-but-alive past the cap. The default 10080 min is 1 week and the field maximum, so waits "
            "are effectively indefinite. The indefinite wait is safe because in-loop detection and not the "
            "timer terminates a dead-job wait. The valid range is 60 min, equal to 1 hour, to 10080 min, "
            "equal to 1 week. Lower the cap only to force an earlier give-up."
        ),
    )
    hpc_no_progress_timeout_min: int | None = Field(
        None,
        ge=10,
        le=10080,
        description=(
            "Backstop cap, in minutes, on the no-progress stall watchdog of the tmux orchestrator, "
            "_wait_for_tmux_session_completion. After the watchdog liveness-gate change, the stall timer "
            "is SUSPENDED while SLURM reports at least one live job for the run, so the cap is a pure "
            "safety backstop and no queue-wait tolerance. The cap fires only when Snakemake is alive AND "
            "SLURM has no live work AND _status/ has not advanced. None, the default, keeps the "
            "walltime-derived fallback `max(30, 6 * hpc_time_min_per_sim)`. The fallback is NOT "
            "queue-aware by itself and is retained only so existing configs are unchanged. The valid range "
            "is 10 min to 10080 min, equal to 1 week. Mirrors hpc_max_wait_for_inflight_min."
        ),
    )
    hpc_max_queue_wait_min: int | None = Field(
        None,
        ge=30,
        le=10080,
        description=(
            "Cap, in minutes, on how long the tmux orchestrator tolerates a live SLURM job set that is "
            "continuously ALL-PENDING and never RUNNING. The cap is distinct from "
            "hpc_no_progress_timeout_min by construction. hpc_no_progress_timeout_min bounds the "
            "SLURM-SILENT state, a hang, and this cap bounds the QUEUE-STARVED state, partition "
            "contention. Reusing a single number for both would recreate the walltime-coupling defect that "
            "the liveness gate exists to remove. Any RUNNING observation resets the tracker. On expiry the "
            "session is killed, cancelling the queued jobs because the SLURM executor runs scancel on "
            "interrupt, and a distinct queue-starved verdict is returned. None, the default, means 720 "
            "min, equal to 12 h. The valid range is 30 min to 10080 min, equal to 1 week."
        ),
    )
    resume_interruption_schedule: tuple[int, ...] | None = Field(
        default=None,
        description=(
            "Resume-test harness ONLY. A STRICTLY INCREASING tuple of ABSOLUTE hotstart-checkpoint "
            "indices. A fresh TRITON/TRITON-SWMM sim is interrupted at each index, producing "
            "`len(schedule)` hotstart resumes. The unit is the unit that the watcher already counts as "
            "config_NNNN.cfg files, and return_the_reporting_step_from_a_cfg returns the same unit, so no "
            "conversion exists anywhere in the mechanism and interval alignment is vacuous. An entry "
            "beyond the checkpoint count of the sim degrades gracefully, and no kill fires. Empty tuples, "
            "duplicates, non-positive entries, and non-increasing sequences are rejected. The runner arms "
            "each kill from the PERSISTED n_resumes counter relative to the baseline of the attempt and "
            "never from an absolute count over the cfg directory, because that directory accumulates "
            "across attempts and an absolute predicate is already true on attempt 2 before any forward "
            "progress. Default None DISABLES the harness. Production and clean-arm runs leave the field "
            "unset, and the runner path is byte-identical to a plain `proc.wait()`. Incompatible with "
            "multi_sim_run_method='1_job_many_srun_tasks', rejected at preflight, because that mode lacks "
            "the job-end cgroup reap that makes repeated per-attempt step teardown structurally safe under "
            "batch_job."
        ),
    )
    swmm_snapshot_disable: bool | None = Field(
        default=None,
        description=(
            "Resume arm selector for the COUPLED model only. When left unset with the default None, the "
            "toolkit decides from the model. If the coupled hydraulics model declares at least one "
            "pollutant and the IGNORE_QUALITY option in [OPTIONS] is not YES, the toolkit behaves as if "
            "the value were True, because a snapshot resume at an older TRITON build lacks the SWMM node "
            "and link water-quality state and exact replay carries the state. If the coupled build is a "
            "TRITON commit listed in `SWMM_QUALITY_SNAPSHOT_SHAS` in `hhemt.scenario`, whose snapshot "
            "carries that state, the toolkit behaves as if the value were False and the quality-routed run "
            "resumes from the snapshot. The coupled build is the sha that the system log records, namely "
            "the sha of the compiled coupled target natively or the image label in a container. An "
            "unlisted, abbreviated, or unrecorded sha counts as not listed. For every other model the "
            "toolkit behaves as if the value were False and writes the cfg byte for byte as before this "
            "field became optional. An explicit True or False overrides the decision. An explicit False "
            "keeps the snapshot route for a quality-routed model, and at a build NOT listed in that set "
            "the processed water-quality statistics of an event that resumed that way are withheld. When "
            "True, the toolkit appends the line `swmm_snapshot_disable=1` to the TRITON cfg of each "
            "coupled scenario, making the solver leave snapshot_path_stem EMPTY and forcing a coupled "
            "hotstart resume onto the OLD exchange-replay fallback in place of the full-precision state "
            "snapshot. The field makes the replay path measurable at a pin that also carries the snapshot "
            "path, so a replay arm and a snapshot arm can be compared without a solver-version confound. "
            "An explicit False appends no line, and a run whose model is not quality-routed and that "
            "leaves the field unset emits a cfg BYTE-IDENTICAL to a cfg emitted before this field existed. "
            "The TRITON-only cfg never carries the key, because the key gates the SWMM coupling surface "
            "and is meaningless there. THE SPELLING IS LOAD-BEARING IN OTHER REPOSITORIES. The solver "
            'reads the key through `argsd("swmm_snapshot_disable", argmap, "0")`, and arm membership '
            "for the bit-for-bit re-run is established iff the `config_{k}.cfg` of a member carries the "
            "literal `swmm_snapshot_disable=1`, carried there by the unknown-key cfg passthrough of the "
            "solver. Renaming the emitted key without renaming all of the reads in the same change "
            "silently unassigns every member from the arm with no error anywhere. THE PIN IS PART OF THE "
            "CONTRACT AND NO CHECK COVERS THE PIN. A solver that predates the key ignores the key, and no "
            "TRITON cfg parse rejects an unknown key, so setting the field against such a solver emits the "
            "line, runs the SNAPSHOT path, and leaves the armed-looking cfg behind as evidence. Verify "
            "that the key is present in the argsd table of the solver at the pin you run."
        ),
    )
    swmm_snapshot_keep_all: bool = Field(
        default=False,
        description=(
            "Resume-EXPERIMENT arm selector for the COUPLED model only. When True, the toolkit appends the "
            "line `swmm_snapshot_keep_all=1` to the TRITON cfg of each coupled scenario, disarming the "
            "snapshot PRUNE gate of the solver so every checkpoint index retains a restorable SWMM state "
            "snapshot in place of only the most recent snapshots. The field buys uniformity and no extra "
            "snapshots for their own sake. Under the keep-2 default, a member interrupted at a checkpoint "
            "older than the most recent retained snapshots finds NO snapshot and silently falls back to "
            "the exchange-replay path, so a single experiment mixes different resume mechanisms across the "
            "members and the cross-experiment comparison carries an undeclared confound. Setting this "
            "field makes the resume mechanism UNIFORM across every member of an arm, so a run against a "
            "solver pin can be compared to a run against another pin with the resume path held fixed. The "
            "field is INDEPENDENT of swmm_snapshot_disable and is not the inverse of swmm_snapshot_disable. "
            "swmm_snapshot_disable empties snapshot_path_stem and removes the snapshot path entirely, and "
            "this field changes the RETENTION DEPTH of a snapshot path that exists. Both False is the "
            "shipped default. Both True is expressible and is resolved BY THE SOLVER, where disable wins "
            "because no stem exists to retain into, and never by the toolkit. The toolkit emits the "
            "requested configuration and arbitrates no conflict. The default False appends no line, so "
            "every cfg emitted for a run that leaves the field unset is BYTE-IDENTICAL to a cfg emitted "
            "before the field existed. The TRITON-only cfg never carries the key, because the key gates "
            "the SWMM coupling surface and is meaningless there. THE SPELLING IS LOAD-BEARING IN ANOTHER "
            'REPOSITORY. The solver reads the key through `argsd("swmm_snapshot_keep_all", argmap, '
            '"0")`, and arm membership is established iff the `config_{k}.cfg` of a member carries the '
            "literal `swmm_snapshot_keep_all=1`, carried there by the unknown-key cfg passthrough of the "
            "solver. Renaming the emitted key without renaming the argsd read in the same change silently "
            "unassigns every member from the arm with no error anywhere. THE PIN IS PART OF THE CONTRACT "
            "AND NO CHECK COVERS THE PIN. A solver that predates the key ignores the key, and no TRITON "
            "cfg parse rejects an unknown key, so setting the field against such a solver emits the line, "
            "prunes to keep-2 anyway, and leaves the armed-looking cfg behind as evidence. Verify that the "
            "key is present in the argsd table of the solver at the pin you run."
        ),
    )
    swmm_snapshot_failed_replay: bool = Field(
        default=False,
        description=(
            "COUPLED model only. When True, the toolkit writes the line `swmm_snapshot_failed_replay=1` "
            "into the TRITON cfg that each coupled launch passes to the solver, including a resume, and "
            "when False the toolkit removes the line. The change takes effect at the NEXT launch, so the "
            "field can be set after a run has stopped at a failed snapshot write. Re-run with "
            "pickup_where_leftoff and the resume replays the SWMM exchange history at that checkpoint and "
            "continues, with correct results, in place of stopping with the write-failed error of the "
            "solver. The failure is still recorded, as a replay resume with reason write-failed, and "
            "raised as a write-failed alarm in the resume-event report. The default False writes no line "
            "into a cfg lacking the key. The TRITON-only cfg never carries the key. A solver that predates "
            "the key ignores the key and stops at the failed write, as the solver does without the key."
        ),
    )
    # local run constraints
    local_cpu_cores_for_workflow: int | None = Field(
        None,
        description="This is passed to Snakemake to set how many CPU cores Snakemake may use on your computer",
    )
    local_gpus_for_workflow: int | None = Field(
        None,
        description="This is passed to Snakemake to set how many GPUs Snakemake may use on your computer",
    )
    # HPC RESOURCES, RETRIES AND PARTITION SELECTORS
    mem_gb_per_cpu: int = Field(2, description="Memory per CPU in GB. Defaults to 2GB.")
    hpc_time_min_per_sim: int | None = Field(
        60,
        description="Walltime in minutes requested for each simulation. The Snakemake `runtime` of a "
        "simulation rule is this value, plus hpc_runtime_min_for_sim_output_processing when "
        "process_in_sim_rule runs output processing inside the simulation rule. Must be >= 1 "
        "under multi_sim_run_method='batch_job'. When an hpc_system_config is supplied, "
        "preflight checks the value against the max_runtime of the ensemble partition.",
        json_schema_extra=field_meta(required_when=[when("multi_sim_run_method", "batch_job")]),
    )
    hpc_restart_times_simulate: int = Field(
        2,
        ge=0,
        description=(
            "Per-rule Snakemake `retries:` for the simulation rules run_triton, run_tritonswmm, run_swmm, "
            "and simulation_member_*. A walltime kill is a SLURM TIMEOUT and is retriable. Raise this "
            "value high, such as 20, for a hotstart-resume sweep so a killed sim re-dispatches from the "
            "latest config_NNNN.cfg checkpoint within a single `analysis.run()`. Default 2. THIS NUMBER IS "
            "PER SNAKEMAKE PROCESS AND IS NO LIMIT ON THE NUMBER OF TIMES THE SOLVER CAN RUN. Snakemake "
            "compares the budget against `job.attempt`, an attribute of an in-memory job object "
            "initialised to 1 when the process builds the DAG and persisted nowhere. Under `--executor "
            "slurm` more than one such process exists per rule execution, namely the driver that submits "
            "the SLURM job and the Snakemake instance running inside the allocation, so the budget "
            "restarts at each nesting level and the REALISED ceiling is the PRODUCT across the levels and "
            "no single value. Measured on member 39 of the a6000 benchmarking arm, a declared 3 produced "
            "10 solver launches across 4 SLURM jobs. The realised TOTAL is bounded instead by "
            "`hpc_consecutive_failure_budget_simulate`, a durable cap per member, event, and model_type "
            "that the runner enforces before each launch. The knob makes a ceiling additive, and the "
            "retries field governs only how many restarts a SINGLE process will attempt."
        ),
    )
    hpc_restart_times_other: int = Field(
        2,
        ge=0,
        description=(
            "Per-rule Snakemake `retries:` baseline for the non-simulation rules prepare, process, "
            "consolidate, plot, and render, emitted as the GLOBAL restart-times so directive-less rules "
            "inherit the baseline. The rules are idempotent re-derivations, so a low count suffices. "
            "Default 2."
        ),
    )
    hpc_consecutive_failure_budget_simulate: int = Field(
        3,
        ge=0,
        description=(
            "DURABLE cap on CONSECUTIVE TERMINAL simulation failures per member, event, and model_type, "
            "enforced in the simulation runner BEFORE the solver is launched. The cap makes a retry "
            "ceiling REAL. `hpc_restart_times_simulate` bounds restarts within a single Snakemake process "
            "and persists across no process, so under `--executor slurm` the realised ceiling MULTIPLIES "
            "across nesting levels. The budget persists in the per-model log and so bounds the TOTAL. A "
            "TERMINAL failure is a failure that the runner itself concluded, with the runner writing "
            "`_status/_failed/{rule_token}.json`. A WALLTIME KILL is NOT terminal and is not counted, so a "
            "hotstart-resume sweep is unaffected however many times the sweep resumes. The count resets on "
            "a completing attempt and on a forced re-run, because the count is era-relative. Reaching the "
            "budget writes the failed marker and exits non-zero WITHOUT launching, a terminal state that "
            "an operator then clears. Set 0 to DISABLE the gate entirely. Default 3."
        ),
    )
    # hpc_max_simultaneous_sims is RETIRED from analysis_config: it MOVED to
    # hpc_system_config.max_concurrent_jobs, because a cluster-throughput cap belongs on
    # the per-HPC-system config, not the per-analysis config. Readers resolve it from
    # cfg_hpc_system.max_concurrent_jobs. Popped by the check_consistency shim.
    #
    # KEPT: the two partition SELECTORS stay on analysis_config — they are the
    # partition-NAME axis lookup keys the resolution helpers + preflight read to index
    # cfg_hpc_system.partitions[name], and partition-as-sensitivity-axis requires them
    # as the per-CSV-row overlay column. They are NOT retired despite the hpc_* prefix.
    hpc_ensemble_partition: str | None = Field(
        None,
        description="SLURM partition name, such as 'standard', 'gpu', or 'high-memory', for running "
        "simulations. The name sets the partition of the simulation rules, where a CPU-only "
        "simulation rule resolves hpc_cpu_sim_partition first, the default slurm_partition of "
        "the generated Snakemake executor profile, and the #SBATCH --partition of the single "
        "allocation under multi_sim_run_method='1_job_many_srun_tasks'. The name is also the "
        "key into hpc_system_config.partitions that resolves the GPU hardware of a simulation, "
        "and a sensitivity sweep may vary the partition per row through the hpc.partition "
        "column.",
        json_schema_extra=field_meta(
            required_when=[when("multi_sim_run_method", "batch_job", "1_job_many_srun_tasks")],
        ),
    )
    hpc_setup_and_analysis_processing_partition: str | None = Field(
        None,
        description="SLURM partition name for the single-node, single-core rules. The rules cover system "
        "setup and compilation, scenario preparation, output processing, consolidation, report "
        "rendering, the plots of the report, and deletion. Also the first choice for the "
        "batch_job orchestration job, with hpc_ensemble_partition as the fallback. Also a "
        "fallback for CPU-only simulation rules, per hpc_cpu_sim_partition.",
        json_schema_extra=field_meta(
            required_when=[when("multi_sim_run_method", "batch_job", "1_job_many_srun_tasks")],
        ),
    )
    hpc_cpu_sim_partition: str | None = Field(
        None,
        description=(
            "SLURM partition for CPU-ONLY simulation rules, currently run_swmm. When None, the partition "
            "resolves to hpc_setup_and_analysis_processing_partition and then to hpc_ensemble_partition. "
            "The field exists because a CPU-only sim carries no --gres, so a GPU partition whose QOS "
            "enforces a GRES minimum rejects the sim at submit, and `sbatch -p` naming that partition "
            "prints `sbatch: error: QOSMinGRES`. Set the field only when the CPU sim rung needs a "
            "different partition from the setup and processing rungs."
        ),
    )
    # hpc_account, hpc_login_node and python_path are RETIRED from analysis_config.
    # account -> hpc_system_config.default_account (via _resolve_account); login_node ->
    # hpc_system_config.login_node; python_path -> sys.executable fallback (no
    # hpc_system_config home — re-add if a cluster needs a bespoke interpreter).
    # Popped by the check_consistency shim.
    additional_SBATCH_params: list[str] | None = Field(
        None,
        description="Optional list of SBATCH arguments, written without the #SBATCH prefix. Relevant only "
        "when multi_sim_run_method = 1_job_many_srun_tasks.",
    )
    # TOGGLES
    toggle_sensitivity_analysis: bool = Field(
        ...,
        description="Whether or not this is a sensitivity study. If so, a .csv file is required for input "
        "sensitivity_analysis "
        "defining the analysisal setup.",
    )
    toggle_storm_tide_boundary: bool = Field(
        ...,
        description="If True, a boundary condition representing storm tide will be applied to the model.",
    )

    storm_tide_boundary_line_gis: Path | None = Field(
        None,
        description="Path to a line gis file spanning the extent of the dem boundary where the variable storm tide "
        "boundary "
        "condition should be applied.",
        json_schema_extra=field_meta(required_when=[when("toggle_storm_tide_boundary", True)]),
    )
    storm_tide_units: str | None = Field(
        None,
        description="Storm tide units, e.g., ft, m. Must align with units used DEM.",
        json_schema_extra=field_meta(required_when=[when("toggle_storm_tide_boundary", True)]),
    )
    weather_event_summary_csv: Path | None = Field(
        None,
        description="CSV file with weather event summary statistics. Events must share "
        "indices with weather_timeseries.",
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
        description="Path to a .csv file defining weather event index used for sensitivity. "
        "The columns must correspond "
        "to the "
        "sytem's weather_event_indices.",
    )
    weather_event_windows_csv: Path | None = Field(
        None,
        description=(
            "CSV declaring the simulation window for each weather event. The index columns must be "
            "weather_event_indices, plus a start column and an end column whose names are given by "
            "weather_event_start_column and weather_event_end_column. The toolkit CLIPS each event to the "
            "declared window and never inspects the weather data to decide a window. WHEN ABSENT, the "
            "window of each event is the full extent of the time coordinate of the event, the file as "
            "given. The toolkit still refuses to run an event whose forcing is incomplete over that "
            "extent. Supplying this CSV is how you declare a shorter window that IS complete."
        ),
    )
    weather_event_start_column: str = Field(
        WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED,
        description=(
            "Column in weather_event_windows_csv carrying the window START of each event, parsed with "
            "pandas.to_datetime. The NAME is a config input so the toolkit binds no column name. The "
            "default is the SENTINEL 'unspecified', a value that names no real column and is rejected "
            "before any column is read. Only relevant when weather_event_windows_csv is supplied, because "
            "without that file no CSV exists to name a column in and the sentinel simply goes unread. When "
            "the file IS supplied, both column names must be stated and both must exist in the file."
        ),
    )
    weather_event_end_column: str = Field(
        WEATHER_EVENT_WINDOW_COLUMN_UNSPECIFIED,
        description=(
            "Column in weather_event_windows_csv carrying the window END of each event, parsed with "
            "pandas.to_datetime. Inclusive, so the clipped slice contains the end stamp. "
            "Sentinel-defaulted and validated as weather_event_start_column is."
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
            "Optional column name in weather_events_to_simulate carrying a human-readable display name for "
            "each event, such as '2003-09-17 05:44 - Hurricane Isabel'. Presentation only. The value NEVER "
            "enters weather_event_indices, the event_id slug, a figure stem, or any path segment. The "
            "column becomes the toolkit canonical name at projection, so downstream consumers never see "
            "the value. None disables labelling, and every report surface falls back to the current text."
        ),
    )
    analysis_description: str | None = Field(
        None,
        description="For readability.",
    )
    hhemt_sha: Annotated[
        str,
        Field(
            ...,
            pattern=r"^[0-9a-f]{40}$",
            description="The hhemt commit this experiment runs, as the FULL 40-hex lowercase sha only, "
            "with no tag, no branch, and no short sha. REQUIRED, with no default. Every "
            "process that constructs an analysis, namely the driver, the simulation runner, "
            "the in-image processing runner, and the consolidator, compares the sha to the "
            "identity of the running toolkit, `validation.running_identity`, before writing "
            "anything into the analysis tree, and refuses on a mismatch with exit 2, so a "
            "refused construction leaves no directory behind.",
            json_schema_extra=field_meta(),
        ),
    ]

    # TRITON-SWMM PARAMETERS
    target_processed_output_type: Literal["zarr", "nc"] = Field(
        "zarr",
        description="TRITON processed output type. The value is zarr or nc.",
        json_schema_extra=field_meta(
            options={
                "zarr": "Chunked Zarr store. Default and preferred -- supports lazy and partial reads.",
                "nc": "One NetCDF file per scenario for each artifact the output type governs. The artifacts "
                "are the gridded TRITON output, the SWMM node timeseries, the SWMM link timeseries, the "
                "performance timeseries, and their summaries. The hydrology capture summaries stay Zarr. "
                "They are hydrology_inflow_summary and hydrology_rpt_summary. "
                "Use when a downstream consumer requires NetCDF.",
            }
        ),
    )
    toggle_consolidate_timeseries: bool = Field(
        default=False,
        description=(
            "When True, the per-scenario SWMM node and SWMM link TIMESERIES are consolidated into the "
            "`timeseries` branch of their group inside analysis_datatree.zarr, beside the `summary` branch of "
            "the same group. A sensitivity master holds each member's series in the `timeseries` branches of "
            "experiment_datatree.zarr. Every model family that writes series is consolidated. The node and link "
            "groups are tritonswmm/swmm_node/timeseries and tritonswmm/swmm_link/timeseries for the coupled model, "
            "and swmm_only/swmm_node/timeseries and swmm_only/swmm_link/timeseries for the SWMM-only model. The "
            "SWMM-only model also writes swmm_only/system/timeseries with 6 system series of flood and "
            "water-balance quantities, and swmm_only/system/summary with the maximum and final value of each "
            "series. Each scenario is appended in "
            "turn along event_iloc, on an "
            "elapsed_s axis of seconds since the start of that event. The node series holds `wlevel(t)` and the "
            "link series holds `flow(t)`. Default False, summaries only. Enable to unblock over-time "
            "clean-vs-resume difference figures. The gridded TRITON store is also consolidated, "
            "into tritonswmm/triton/timeseries for the coupled model and triton_only/triton/timeseries for the "
            "TRITON-only model. Each gridded group has a separate elapsed_s axis at the TRITON reporting "
            "step, chunks of about 32 MB, a streamed write, and a read-back that compares the stored slice with "
            "the source block by block. The variables kept are the ones the timeseries_keep_triton_gridded field "
            "selects. The per-scenario series stores of every family are cleared under one switch, "
            "toggle_clear_scenario_timeseries_after_consolidation, once the scenario slice has passed the read-back."
        ),
    )
    toggle_coupling_exchange_flow: bool = Field(
        default=False,
        description=(
            "When True, the coupled model's per-scenario SWMM node time-series store also carries "
            "coupling_exchange_flow, the signed flow in m3/s that TRITON exchanges with SWMM at each coupling "
            "node, averaged over each reporting interval. The sign is positive when water drains from the street "
            "surface into the sewer and negative when the sewer surcharges onto the street. The store also "
            "carries 8 per-node statistics from the step series of the solver. The statistics are the peak and the "
            "volume in each direction, the net volume, the time of each peak, and the duration of "
            "sewer-to-surface flow. The values are decoded from the exchange side-file of the solver, and the "
            "side-file stays on disk until the node store is written. Applies only to a coupled system with "
            "toggle_tritonswmm_model on, and preflight refuses True when the coupled model is off. The "
            "coupling-node flag is_coupling_node needs no field, because the coupled model always writes the "
            "flag. Default False. A store written with False carries neither the flux series nor the "
            "statistics."
        ),
        json_schema_extra=field_meta(applies_when=[when("toggle_tritonswmm_model", True)]),
    )
    toggle_clear_scenario_timeseries_after_consolidation: bool | None = Field(
        default=None,
        description=(
            "When True, a scenario's own time-series stores are deleted as soon as the "
            "slice of the scenario has been appended to the `timeseries` branch of analysis_datatree.zarr, read back "
            "from the store, and "
            "found equal, NaN-aware, to the data of the scenario. Peak disk use then stays near one copy of the "
            "series, and the consolidated store holds the only copy. The stores are the SWMM node and link stores "
            "of the coupled and SWMM-only models, the system series store of the SWMM-only model, and the gridded "
            "TRITON store of the coupled and TRITON-only models. The per-scenario summary stores of every model are "
            "cleared too, after each slice of the scenario has been read back from the `summary` branches of "
            "analysis_datatree.zarr, and the branches then hold the only copy of the summaries and only ever "
            "gain events. A cleared summary is read from the branch and is never re-derived, and a forced "
            "re-process of a cleared scenario needs the raw output of the scenario. "
            "Requires toggle_consolidate_timeseries, and "
            "preflight refuses True without the toggle. The deletion is recorded in the model log of the "
            "scenario. A cleared store reads as processed and is never re-derived. Left unset, the field "
            "takes the value of toggle_consolidate_timeseries, so the field is True when consolidation is on and the "
            "per-scenario copies are cleared, and False when consolidation is off. An explicit False keeps the "
            "per-scenario copies after consolidation. A written cfg_analysis.yaml carries the resolved "
            "value, never None. Default unset."
        ),
        json_schema_extra=field_meta(applies_when=[when("toggle_consolidate_timeseries", True)]),
    )
    timeseries_keep_swmm_node: str | list[str] | None = Field(
        default=None,
        description=(
            "The SWMM node time-series variables to KEEP in the per-scenario store after the summary is "
            "written. The store drops the rest, and the summary keeps every statistic. None or blank keeps "
            "everything. A string names a toolkit set. The set 'all' holds every variable of the class, and the "
            "set 'default' holds every variable except flooding for the coupled model, because flooding there "
            "is the growth rate of stored volume above full volume and not water lost. For standalone SWMM the "
            "set 'default' holds every variable. "
            "A list names variables explicitly and may force in a variable that a set excludes. Variables with "
            "no time dimension are static and always kept. They are the coupling-node flag, the per-node "
            "statistics, and the run-level variables. The kept names are the variables that "
            "the `timeseries` branches of analysis_datatree.zarr and the bundle carry. Preflight refuses a name "
            "outside the domain of every "
            "enabled model family. The field applies even when toggle_consolidate_timeseries is off. A pruned "
            "variable is gone, and recovering the variable takes a re-run."
        ),
    )
    timeseries_keep_swmm_link: str | list[str] | None = Field(
        default=None,
        description=(
            "The SWMM link time-series variables to KEEP in the per-scenario store after the summary is "
            "written. The store drops the rest, and the summary keeps every statistic. None or blank keeps "
            "everything. A string names a toolkit set. The set 'all' holds every variable of the class, and the "
            "set 'default' holds every variable. "
            "A list names variables explicitly and may force in a variable that a set excludes. Variables with "
            "no time dimension are static and always kept. They are the coupling-node flag, the per-node "
            "statistics, and the run-level variables. The kept names are the variables that "
            "the `timeseries` branches of analysis_datatree.zarr and the bundle carry. Preflight refuses a name "
            "outside the domain of every "
            "enabled model family. The field applies even when toggle_consolidate_timeseries is off. A pruned "
            "variable is gone, and recovering the variable takes a re-run."
        ),
    )
    timeseries_keep_triton_gridded: str | list[str] | None = Field(
        default=None,
        description=(
            "The gridded TRITON surface time-series variables to KEEP in the per-scenario store after the "
            "summary is written. The store drops the rest, and the summary keeps every statistic. None or blank "
            "keeps everything. A string names a toolkit set. The set 'all' holds every variable of the class, "
            "and the set 'default' holds every variable except max_flood_depth, unit_discharge_x, and "
            "unit_discharge_y, and the summary keeps the peaks of the excluded variables. "
            "A list names variables explicitly and may force in a variable that a set excludes. Variables with "
            "no time dimension are static and always kept. They are the coupling-node flag, the per-node "
            "statistics, and the run-level variables. The kept names are the variables that "
            "the `timeseries` branches of analysis_datatree.zarr and the bundle carry. Preflight refuses a name "
            "outside the domain of every "
            "enabled model family. The field applies even when toggle_consolidate_timeseries is off. A pruned "
            "variable is gone, and recovering the variable takes a re-run."
        ),
    )
    process_output_target_chunksize_mb: int = Field(
        200,
        description="Target memory budget, in MiB, PER LOAD CHUNK for streaming-chunked operations on "
        "per-scenario timeseries output. The field is the in-memory RSS guard ONLY. The number "
        "of timesteps in one output chapter is set by process_append_batch_timesteps. Consumed "
        "by the chunked raw-to-zarr LOAD inside write_timeseries_outputs AND by the "
        "_streaming_argmax_with_companions helper of summarize_triton_simulation_results, the "
        "per-cell argmax-with-companions reduction. On fine grids a single float64 timestep "
        "can meet or exceed the budget, flooring the load chunk to 1 timestep. The floor is a "
        "correct memory guard and no performance bug, because write granularity is decoupled "
        "via process_append_batch_timesteps.",
    )
    process_append_batch_timesteps: int = Field(
        128,
        description="Number of LOADED timesteps to accumulate before emitting ONE output chapter in "
        "write_timeseries_outputs. The field decouples write granularity from the in-memory "
        "load-chunk size, process_output_target_chunksize_mb, so fine grids that floor the "
        "load chunk to 1 timestep still emit only `ceil(N_timesteps / this)` chapters in place "
        "of `O(N_timesteps)` tiny chapters. Independent of the streaming-summary reduction, "
        "and that reduction writes no chapter. Buffer RSS is bounded by "
        "process_append_batch_memory_budget_mb, because the buffer is flushed once the buffer "
        "reaches that budget, so the buffer holds less than that budget plus a single load "
        "chunk before each flush.",
    )
    process_append_batch_memory_budget_mb: int | None = Field(
        None,
        description=(
            "Memory budget, in MiB, governing BOTH the output-chapter batch byte cap in "
            "write_timeseries_outputs AND the streaming-argmax summary reduction in "
            "summarize_triton_simulation_results. The budget is distinct from "
            "process_output_target_chunksize_mb, the small per-LOAD-chunk RSS guard of about 200 MiB. The "
            "larger budget lets fine grids accumulate a bigger pending batch or argmax chunk inside the "
            "real RAM allocation of the process job. When None, the default, the budget resolves at "
            "config-load to the fraction 0.35 of hpc_mem_allocation_for_sim_output_processing_mb, the "
            "field that sets the process-rule mem_mb in SLURM, clamped to the actual cgroup limit when "
            "readable. See the _resolve_process_batch_budget validator. A concrete int overrides the "
            "fraction and is still ceiling-checked at <= 0.5*job_RAM. The 0.35 fraction reserves headroom "
            "for the peak-RSS inequality `2*B + per_ts <= job_RAM`, because the flush transiently holds "
            "the pending batch B, the xr.concat copy of about B, and a live load chunk per_ts, with "
            "allowance for the post-append trigger overshoot. Consumed at the process_simulation.py "
            "write-flush cap and argmax budget."
        ),
    )
    flood_velocity_min_depth_m: float = Field(
        DEFAULT_FLOOD_VELOCITY_MIN_DEPTH_M,
        gt=0,
        description=(
            "Wet-depth threshold, in m, for the TRITON flood velocity. The summary stores the maximum, "
            "over output prints, of the ratio `|q| / h`. Here q is the solver unit discharge from the QX "
            "and QY outputs and h is the depth of the SAME print. At a print where h is at or below the "
            "threshold, or NaN, the flood velocity is NaN, because the solver zeroes q below the hextra of "
            "the solver and `q / h` is then `0 / 0`. The default 0.001 m equals the hextra of the solver "
            "in every measured run. The toolkit takes no hextra value from the cfg, so raise this value "
            "yourself if a case study sets a larger hextra. Read at summary time and stamped into the "
            "`comment` attribute of each flood-velocity variable."
        ),
    )
    process_store_float32: bool = Field(
        True,
        description=(
            "Store per-scenario spatial timeseries as float32 in the processed zarr in place of float64, "
            "roughly halving on-disk size and I/O. The affected quantities are H/QX/QY/MH and the derived "
            "flood depth/unit discharge. Default True. Set False for precision-sensitive analyses, such as "
            "tight mass-balance. The setting leaves CF attributes unchanged, because the dtype lives in "
            "the zarr encoding dict, orthogonal to cf_conventions.py. Consumed by "
            "utils.return_dic_zarr_encodings."
        ),
    )
    process_timestep_chunk: int | None = Field(
        None,
        description=(
            "Explicit on-disk `timestep` zarr chunk size for the per-scenario spatial timeseries. A "
            "READ-LOCALITY OVERRIDE ONLY. The field replaces the TIME component of the grid that "
            "utils.resolve_chunk_grid derives and leaves the spatial components derived, so no value here "
            "can re-introduce an extent-dependent grid. When None, the default, the derived time component "
            "is used. The chunk grid is made safe by the once-per-scenario call of the writer to "
            "resolve_chunk_grid, and the field plays no part in that safety. With no grid declared at all, "
            "zarr sizes each store from the FULL shape, making the spatial grid of a chapter a function of "
            "the time extent of the chapter. Consumed by utils.return_dic_zarr_encodings and "
            "utils.resolve_chunk_grid."
        ),
    )
    allow_mixed_version_chapters: bool = Field(
        False,
        description=(
            "Bypass the chapter-set build guard. When False, the default, and a resume would extend a "
            "per-scenario chapter set that a DIFFERENT hhemt build wrote, processing refuses loudly in "
            "place of publishing a unified store built by different processing builds. Set True only if no "
            "material change exists between the builds. The run then proceeds, prints the mismatch to "
            "standard output, kept in the processing step log when the workflow runs the step, and, once "
            "the guard flags a chapter, adds the build to provenance_history.json in the directory of the "
            "chapter set beside the earlier builds. The guard identifies a build by processing_build_key. "
            "The key is the identity of the running toolkit. Inside a container image the identity is the "
            "staged HHEMT_SHA of the image. When the identity is unconfirmed, as with a checkout mounted "
            "over the source of the image, the key is the org.hhemt.hhemt_sha label of the image, read "
            "from /.singularity.d/labels.json. The label names the image and not a checkout mounted over "
            "the image, so changes to such a checkout are undetected. The history records the same "
            "identity, so a label-identified run is compared on the next resume. A build that neither "
            "source resolves is recorded as 'unknown' and is not compared. The detector is therefore live "
            "under execution_environment='container' as well as natively. The guard prevents a chapter set "
            "written with a declared chunk grid from being extended by a build that declares another grid, "
            "so leaving the field False is load-bearing. Consumed by "
            "provenance.assert_chapters_match_running_build."
        ),
    )
    TRITON_raw_output_type: Literal["bin", "asc"] = Field(
        "bin",
        description="TRITON raw output type. The value is asc or bin.",
        json_schema_extra=field_meta(
            options={
                "bin": "Raw binary rasters. Default -- smaller and faster to write and parse.",
                "asc": "ASCII grid rasters that are human-readable but far larger and slower.",
            }
        ),
    )
    manhole_diameter: float = Field(
        ...,
        description="Manhole diameter of TRITON-SWMM interaction nodes, in meters.",
    )
    manhole_loss_coefficient: float = Field(
        ...,
        description="Loss coefficient of TRITON-SWMM interactions occuring at manholes.",
    )
    swmm_node_rim_from_dem: bool = Field(
        default=False,
        description=(
            "When True, the toolkit writes the line `swmm_node_rim_from_dem=1` into the TRITON cfg of each "
            "coupled scenario, and TRITON then takes the rim of a coupled node, the height of the surface "
            "connection of the node, as the DEM cell elevation minus the invert of the node, in place of "
            "the depth that SWMM itself uses for the node. When False, the default, the line is not "
            "written and the rim is the SWMM depth. Only TRITON builds that read the key honour the "
            "setting, so preparation refuses a True value unless the build the run uses is known to read "
            "the key. In native mode the TRITON clone src/config_utils.h must carry the read of the key, "
            "or the clone must descend from the commit that added the key. In container mode the recorded "
            "TRITON commit of the image must equal TRITONSWMM_branch_key and be one of the commits known "
            "to read the key. The setting changes hydraulic results when True."
        ),
    )
    one_coupled_inflow_node_per_dem_cell: bool = Field(
        default=False,
        description=(
            "Applies only to a coupled system with toggle_tritonswmm_model on and "
            "system_config.swmm_coupling_mode set to the third choice. A coupled system under any other "
            "mode with the option on is refused at run entry, and a system whose coupled model is off "
            "writes no coupling-node list, so the option is inert there. When True, scenario preparation "
            "writes at most ONE candidate junction per DEM cell to the coupling-node list, the limit that "
            "TRITON-SWMM required before TRITON-SWMM coupled any number of nodes per cell. The kept "
            "junction is the junction whose own outgoing link has the largest flow area. On a tie, the "
            "kept junction is the junction of the cell whose outgoing link leaves the cell, then the "
            "junction with the lowest invert. A junction with no outgoing pipe is ranked by the largest "
            "incoming pipe. No [INFLOWS] line is removed. Each prepared scenario records the cells that "
            "the scenario reduced in _status/inflow_node_selection.json, and the inflow-node reduction "
            "check of the report lists that file. When False, the default, every candidate junction is "
            "written, so the number of surface exchange points is independent of the DEM resolution. In "
            "both cases outfalls, storage nodes, and dividers are not coupled."
        ),
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
        description="0 for closed and 1 for open. The value applies to all boundaries wherever external "
        "boundary conditions are otherwise undefined.",
    )

    # OPTIONAL PATHS, EXPERIMENT MEMBERSHIP, REPORTING AND EXECUTION SETTINGS
    analysis_dir: Path | None = Field(
        None,
        description="Optional path to analysis directory. If not specified, the analysis directory will be placed "
        "within the "
        "system directory named named with the analysis_id",
        # A toolkit-owned OUTPUT: the analysis tree is created by the run, not supplied
        # by the user. Also the FORCED_DOT "." bundle-root marker in a bundle's
        # cfg_analysis (bundle/_path_policy.py). Exempt from the load-time check.
        json_schema_extra={"toolkit_owned_output": True},
    )
    is_experiment_member: bool = Field(
        False,
        description="This is used in the backend to help route members to appropriate processes.",
    )
    experiment_cfg_yaml: Path | None = Field(
        None,
        description="Path to the configuration file of the parent experiment that contains this analysis "
        "as a member. Required when is_experiment_member is true.",
    )
    report: _report_config_model = Field(
        ...,
        description=(
            "Required inline report-rendering config. The config replaces a separate report_config.yaml "
            "that Snakefile shell lines referenced by absolute path. The canonical source of truth for "
            "renderer parameters, including `interactive.static_backend`. A cfg_analysis.yaml file without "
            "a `report:` block raises pydantic ValidationError at load time. Callers may still pass an "
            "explicit `report_config=` argument to `analysis.run()` to override. The precedence rule is "
            "that an explicit `report_config=` path passed to `analysis.run()` wins over this inline "
            "block. The field is deliberately inline and is no path to a separate file, so a "
            "cfg_analysis.yaml carries the report configuration with no repointing."
        ),
    )

    brand_theme: Path | None = Field(
        None,
        description=(
            "Optional path to a brand-theme YAML for institutional identity, The theme covers the "
            "report.css :root palette, the HTML-table primary color, the HTML-table accent color, and the "
            "navbar upper-left text. When None, the default, the code-frozen DEFAULT_BRAND_THEME in "
            "config/brand_theme.py applies. Mirrors the sensitivity_analysis and "
            "storm_tide_boundary_line_gis path-field precedent. Callers may pass an explicit "
            "`override_brand_theme=` Path to `analysis.run()` to override for a single invocation, "
            "mirroring the `report_config=` runtime-override precedent. Automatically per-member "
            "overlayable via an `analysis.brand_theme` sensitivity column."
        ),
    )

    static_plot_configs: list[Path] = Field(
        default_factory=list,
        description=(
            "Per-plot static-config YAML paths. Each path is a standalone publication-static plot spec. "
            "Default [], meaning no static plots. The field is strict-safe, and old yamls load cleanly. "
            "Each element is existence-validated at config-load via a dedicated "
            "`@field_validator('static_plot_configs')`, because the base * validator "
            "`cfgBaseModel._check_paths_exist` only existence-checks SCALAR Path fields and passes "
            "list[Path] through, so a list-aware validator is required. Consumed by "
            "`analysis.static_plots()`, the method that renders one figure per config via a dedicated "
            "Snakemake workflow. A single config may also be selected by ID at call time. Leaving this "
            "list empty is an error unless a config is supplied at call time."
        ),
    )

    eda: eda_config = Field(
        default_factory=eda_config,
        description=(
            "Optional inline EDA-loop config that selects the EDA plots in the standalone eda_report.html. "
            "The default member set, the cross-sim byte-identity plot, applies when absent. The config is "
            "deliberately INLINE and no path field, so the config travels in cfg_analysis.yaml and "
            "`Bundle.eda()` reads the config with zero extra carry/repoint wiring, for the same reason as "
            "the `report` field above. Runtime override via `eda(override_eda_config=<Path>)`."
        ),
    )

    execution_environment: Literal["native", "container"] = Field(
        "native",
        description=(
            "'native' runs sim+processing on the host, the default. 'container' wraps the innermost sim "
            "{exe} and the process_{model} runners in `apptainer exec {sif}`, with the cluster SIF "
            "described by hpc_system_config.container, a ContainerSpec. Additive default-valued field so "
            "pre-container configs load as native. The native|container SELECTOR is experiment-scoped, and "
            "the cluster-coupled 'how to exec' lives on ContainerSpec."
        ),
        json_schema_extra=field_meta(
            options={
                "native": "Simulate and process on the host.",
                "container": (
                    "Wrap the sim executable and the process_{model} runners in "
                    "`apptainer exec {sif}`, per hpc_system_config.container."
                ),
            }
        ),
    )
    process_in_sim_rule: bool = Field(
        False,
        description=(
            "Run the per-scenario output processing of each simulation INSIDE the simulation rule, as one "
            "Snakemake rule and one SLURM job with the sim runner and then the process runner in one "
            "shell, in place of a separate `process_{model}` rule. Default False reproduces today's "
            "Snakefile byte-for-byte. True removes the processing job from the SLURM queue entirely, so "
            "reclaim of raw output can never wait behind a backlog of older simulation jobs, as in the "
            "2026-09-12 stochastic-ensemble stall. The combined rule declares BOTH the c_run and d_process "
            "flags as outputs and requests `mem_mb = max(sim, processing)` with `runtime = sim + "
            "processing`. The rule is safe under retries because the sim runner short-circuits an "
            "already-completed simulation, since run_simulation.prepare_simulation_command returns None on "
            "model_run_completed, so a processing failure re-runs only the processing pass. Applies to "
            "multi_sim_run_method='batch_job' and '1_job_many_srun_tasks'. REFUSED on a sensitivity "
            "analysis, where the sensitivity-primary generator raises ConfigurationError, until the member "
            "rules carry the same branch."
        ),
        json_schema_extra=field_meta(
            applies_when=[when("multi_sim_run_method", "batch_job", "1_job_many_srun_tasks")],
        ),
    )

    # CLEANUP / FORCE-RERUN POLICY
    #
    # TWO KNOBS, TWO AXES -- do not conflate them. `clear_raw` is per-MODEL-TYPE over RAW
    # SIMULATION outputs and fires inside write_timeseries_outputs.
    # `remove_after_processing` is per-ARTIFACT-CLASS over POST-PROCESSING REDUNDANCY and
    # fires only after the summaries are provably intact: the per-model classes
    # (timeseries, raw_swmm_binaries, coupled_rpt) in the process runner, and the four
    # scenario-scoped classes (hydro_out, prep_inputs, hydrographs, standalone_rpt) in
    # consolidate_workflow: at per-scenario consolidation, and in the analysis-level
    # fallback reclaim_unconsolidated_scenarios for events with no per-scenario
    # consolidation flag, gated there on scenario_summaries_present.
    remove_after_processing: RemoveValue = Field(
        _DEFAULT_REMOVE_AFTER_PROCESSING,
        description=(
            "Post-processing reclaim policy. Fires ONLY after the per-model summary outputs are verified "
            'present AND openable on disk. "none" reclaims no artifact. "all" reclaims every artifact '
            'class. A list selects classes. "timeseries" drops the per-scenario *_tseries zarr/nc set, '
            "the paired complement of summary_paths._SUMMARY_STEMS_BY_MODEL, and neither a renderer nor a "
            'default consolidation path reads the set. "raw_swmm_binaries" drops '
            "out_tritonswmm/swmm/*.out, covering hydraulics.out plus the per-node *.out set, and NEVER "
            "*.rpt, a live completion predicate via run_simulation._coupled_swmm_report_finalized. "
            '"coupled_rpt" TRUNCATES out_tritonswmm/swmm/hydraulics.rpt in place to the header, the '
            "continuity table, the summary table, and the trailer. The class truncates and NEVER deletes, "
            "because the file is a live completion predicate via "
            'run_simulation._coupled_swmm_report_finalized. "hydro_out" drops the SWMM hydrology output '
            "swmm/hydro.out, the file that write_hydrograph_files reads, and is safe only because "
            'write_hydrograph_files carries an already-written gate. "prep_inputs" drops the weather '
            ".dat directory, the external-boundary timeseries directory, and the weather timeseries file, "
            'all regenerable by re-running scenario preparation with no solver. "hydrographs" drops the '
            "hydrograph timeseries and locations files ONLY once processed/hydrology_inflow_summary.zarr "
            'is present and openable, and otherwise declines. "standalone_rpt" TRUNCATES the standalone '
            "full.rpt in place and never deletes the file, and deletes the hydrology .rpt only once a "
            "processed/hydrology_rpt_summary.zarr capture carrying every subcatchment that the hydrology "
            'model declares has been written. "raw_swmm_binaries" no-ops with a logged reason when '
            'clear_raw == "none", because the only consumer of the class, '
            "eda.raw_byte_identity.compare_swmm_raw, also needs the raw H/QX/QY/MH set that clear_raw "
            "governs. Every reclaim is recorded per scenario in the per-model log and surfaced by "
            "analysis_validation.check_data_availability. The default reclaims every class except "
            '"timeseries" and "hydro_out", so a run with nothing configured leaves no raw or intermediate '
            'copy of data that the processed output holds. "timeseries" is left out because its stores are '
            "cleared only after a verified copy into the consolidated tree, by "
            "toggle_clear_scenario_timeseries_after_consolidation, and electing the class here would delete "
            'them before that copy. "hydro_out" is left out because electing it together with '
            '"hydrographs" makes a re-prep hard-fail. Set "none" to keep every artifact. Reclaiming makes '
            "REPROCESSING the affected scenarios impossible without re-running the simulations."
        ),
    )
    clear_raw: ClearRawValue = Field(
        "none",
        description=(
            'Post-processing cleanup policy. "all" deletes all raw outputs for every enabled model type. '
            '"none" deletes no output. A list of model type strings, a subset of "tritonswmm", '
            '"triton", and "swmm", deletes raw outputs only for the listed model types. Defaults to '
            '"none", so yamls written before this field was introduced load cleanly with the strict-safe '
            "default of no deletion."
        ),
    )
    # `validate_default=True`, because pydantic does not validate a default and this field's
    # `mode="before"` coercion is what turns the literal "none" into a ForceRerunSpec.
    # Without it an omitted `force_rerun` holds the raw string in a field typed as a spec,
    # and every `model_dump` of the config emits a serializer warning. Deliberately NOT a
    # `default_factory`: the JSON-schema default the config reference renders stays "none".
    force_rerun: ForceRerunValue = Field(
        "none",
        validate_default=True,
        description=(
            'Force-rerun policy. "all" re-runs everything. "none" runs no forced re-runs. A dict with '
            'a single key, "sa_id" for a sensitivity analysis or "event_iloc" otherwise, and a list of '
            'int or string identifiers re-runs only the named members or events. Defaults to "none", so '
            "yamls written before this field was introduced load cleanly with the strict-safe default of "
            "no re-runs."
        ),
    )

    # VALIDATION - PATH-LIST EXISTENCE
    @field_validator("static_plot_configs", mode="after")
    @classmethod
    def _check_static_plot_configs_exist(cls, v: "list[Path]") -> "list[Path]":
        """Element-wise existence check for the `static_plot_configs` list.

        The base ``*`` validator ``_check_paths_exist`` only handles scalar
        ``Path`` values and silently passes a ``list[Path]`` through, so list
        elements need their own existence validation.
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
                raise ValueError("remove_after_processing list form cannot be empty; use 'none' to reclaim nothing")
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
            raise ValueError("resume_interruption_schedule cannot be an empty tuple; use None to disable the harness")
        if len(v) != len(set(v)):
            raise ValueError(f"resume_interruption_schedule contains duplicates: {v}")
        if any(entry < 1 for entry in v):
            raise ValueError(f"resume_interruption_schedule entries must be >= 1 (absolute checkpoint indices): {v}")
        if any(b <= a for a, b in zip(v, v[1:], strict=False)):
            raise ValueError(f"resume_interruption_schedule must be strictly increasing with no duplicates: {v}")
        return v

    @model_validator(mode="before")
    @classmethod
    def validate_toggle_dependencies(cls, values):
        errors = []

        # This validator enforces nothing itself. Conditional requiredness for the four
        # toggle-dependent fields (storm_tide_boundary_line_gis, storm_tide_units and
        # weather_time_series_storm_tide_datavar under toggle_storm_tide_boundary, and
        # sensitivity_analysis under toggle_sensitivity_analysis) is DECLARED on the fields
        # via `field_meta(required_when=...)` and enforced by
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
        # REMOVE once un-migrated YAMLs no longer need to load (no removal release has
        # been set): pop-and-warn shim for the 6 retired
        # analysis_config HPC fields (moved to hpc_system_config.max_concurrent_jobs /
        # default_account / login_node, the partition-axis PartitionSpec topology, and
        # sys.executable). Pop-and-DROP so un-migrated YAMLs still load (extra="forbid"
        # would else reject). The two partition selectors are KEPT.
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
            if mpi is not None and mpi > 1 and nodes is not None and mpi < nodes:
                raise ValueError(
                    f"n_mpi_procs must be >= n_nodes for GPU mode with MPI. "
                    f"You specified n_mpi_procs={mpi} and n_nodes={nodes}. "
                    f"Each node requires at least one MPI rank to run on it."
                )

            # The hpc_gpus_per_node requirement (GPU mode) is retired
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

        if multi_sim_method == "batch_job" and hpc_time_min_per_sim is not None:
            if isinstance(hpc_time_min_per_sim, float) and math.isnan(hpc_time_min_per_sim):
                raise ValueError(
                    "hpc_time_min_per_sim must be a valid integer >= 1 for multi_sim_run_method=batch_job (NaN "
                    "detected)"
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
                raise ValueError("force_rerun.sa_id requires toggle_sensitivity_analysis=True")
            if key == "event_iloc" and self.toggle_sensitivity_analysis:
                raise ValueError(
                    "force_rerun.event_iloc requires toggle_sensitivity_analysis=False; "
                    "sensitivity-toggled analyses must use force_rerun.sa_id instead"
                )
        return self

    @model_validator(mode="after")
    def _resolve_clear_scenario_timeseries(self):
        """Resolve an unset clearing toggle to the consolidation toggle, so every reader sees a bool.

        Once the consolidated store holds the verified slice the per-scenario copies are redundant, so an
        analysis that turns consolidation on and says nothing about clearing clears them. Resolving here
        rather than at each reader means ``processing_analysis`` and ``validation`` need no ``None`` arm,
        and a written ``cfg_analysis.yaml`` carries the resolved value. Resolving to ``False`` when
        consolidation is off keeps ``validation._validate_clear_scenario_timeseries`` quiet for an
        analysis that mentions neither key. An explicit ``True`` without consolidation is not touched
        here and stays refused by that check.
        """
        if self.toggle_clear_scenario_timeseries_after_consolidation is None:
            self.toggle_clear_scenario_timeseries_after_consolidation = bool(self.toggle_consolidate_timeseries)
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
        # Guard 1: never exceed half the declared job RAM (the 2*B <= job_RAM inequality).
        ceiling = round(0.5 * declared_job_ram)
        if self.process_append_batch_memory_budget_mb > ceiling:
            raise ValueError(
                f"process_append_batch_memory_budget_mb "
                f"({self.process_append_batch_memory_budget_mb}) exceeds 0.5 * "
                f"hpc_mem_allocation_for_sim_output_processing_mb ({ceiling}); the "
                f"2*B + per_ts <= job_RAM peak-RSS inequality requires B <= ~0.5*job_RAM."
            )
        # Guard 2: best-effort clamp to the ACTUAL cgroup limit when readable, so a
        # SLURM under-allocation (declared > granted) cannot drive the cap above the real
        # envelope, where the job would be OOM-killed. No-op once declared == actual.
        actual = _read_cgroup_memory_limit_mib()
        if actual is not None:
            self.process_append_batch_memory_budget_mb = min(
                self.process_append_batch_memory_budget_mb,
                round(self._PROCESS_BATCH_BUDGET_FRACTION * actual),
            )
        return self
