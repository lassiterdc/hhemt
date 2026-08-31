from hhemt.utils import write_json
from hhemt.exceptions import ProcessingError
from hhemt._filelock_compat import resolve_filelock
from filelock import Timeout
from pathlib import Path
from pydantic import BaseModel, Field, field_serializer, PrivateAttr, field_validator
import json
from typing import Type, Optional, Generic, TypeVar, Any, Dict
import logging


T = TypeVar("T")  # Generic type variable


# ----------------------------
# Custom types with generics
# ----------------------------
class LogField(Generic[T]):
    """
    A field that automatically writes to the parent log when updated.

    Usage:
        my_field: LogField[bool] = LogField()

    The type annotation is used for auto-registration of validators and serializers.
    No manual registration needed in most cases.
    """

    _log: "TRITONSWMM_log" = PrivateAttr()

    def __init__(
        self,
        value: Optional[T] = None,
        expected_type: Optional[Type[T]] = None,
    ):
        self.value: Optional[T] = value
        self._expected_type = expected_type

    def set_log(self, log: "TRITONSWMM_log"):
        self._log = log

    def set_type(self, expected_type: Type[T]):
        self._expected_type = expected_type

    def set(self, new_value: T):
        if self._expected_type:
            new_value = self._expected_type(new_value)  # type: ignore

        self.value = new_value
        self._log.write()

    def clear(self):
        """Set this field to None (NOT MEASURED) and persist.

        `set(None)` CANNOT express this. On any log loaded via `from_json` ->
        `model_validate`, the pydantic before-validator constructs
        `LogField(v, expected_type=bool)`, so `set(None)` coerces to
        `bool(None)` == False and rewrites the very value it was asked to clear.
        A log built fresh (default_factory, no expected_type) does NOT coerce —
        which is why the failure is invisible in any test that constructs a bare
        LogField and only appears on a hydrated one. MEASURED 2026-08-10.

        `write()` persists None correctly: its delta keys on
        `mine[k] != baseline[k]`, and None != False, so the field lands in
        changed_keys and the lost-update overlay does not restore the old value.

        Deliberately a separate verb rather than widening `set()`: `set()` is
        called throughout the codebase and its coercion contract is relied upon.
        Making `set(None)` mean "clear" is the more general fix and is a tracked
        follow-up needing its own blast-radius review.
        """
        self.value = None
        self._log.write()

    def get(self) -> Optional[T]:
        if self.value is None or self._expected_type is None:
            return self.value

        try:
            return self._expected_type(self.value)  # type: ignore
        except Exception:
            raise TypeError(
                f"Cannot coerce {self.value!r} " f"to {self._expected_type.__name__}"
            )

    def as_dict(self):
        return {"value": self.value}

    def __repr__(self):
        return f"LogField({self.value!r})"


class LogFieldDict(Generic[T]):
    """
    A dictionary field that automatically writes to the parent log when updated.

    Usage:
        my_dict: LogFieldDict[Path] = LogFieldDict()
    """

    _log: "TRITONSWMM_log" = PrivateAttr()

    def __init__(
        self, d: Optional[Dict[Any, T]] = None, expected_type: Optional[Type[T]] = None
    ):
        self.value: Dict[Any, T] = d or {}
        self._expected_type = expected_type

    def set_log(self, log: "TRITONSWMM_log"):
        self._log = log

    def set(self, new_dict: Dict[Any, Any]):
        for k, v in new_dict.items():
            if self._expected_type:
                v = self._expected_type(v)  # type: ignore
            self.value[k] = v
        self._log.write()

    def get(self) -> Dict[Any, T]:
        if self._expected_type:
            return {k: self._expected_type(v) for k, v in self.value.items()}  # type: ignore
        return self.value

    def as_dict(self):
        return {"value": self.value}

    def __repr__(self):
        return f"LogFieldDict({self.value!r})"


# ----------------------------
# Simulation Processing
# ----------------------------
class ProcessingEntry(BaseModel):
    filepath: Path
    size_MiB: float
    time_elapsed_s: float
    success: bool
    notes: str = ""
    warnings: str = ""


class Processing(BaseModel):
    _log: "TRITONSWMM_log" = PrivateAttr()
    outputs: Dict[str, ProcessingEntry] = Field(default_factory=dict)

    def set_log(self, log: "TRITONSWMM_log"):
        self._log = log

    def update(self, entry: ProcessingEntry):
        self.outputs[entry.filepath.name] = entry
        self._log.write()


# ----------------------------
# Helper function to create validators and serializers
# ----------------------------
def _create_logfield_validator(expected_type: Optional[Type] = None):
    """Creates a validator function for LogField with optional type coercion."""

    def validator(cls, v: Any):
        if isinstance(v, LogField):
            return v
        return LogField(v, expected_type=expected_type)

    return validator


def _create_logfielddict_validator(expected_type: Optional[Type] = None):
    """Creates a validator function for LogFieldDict with optional type coercion."""

    def validator(cls, v: Any):
        if isinstance(v, LogFieldDict):
            return v
        if v is None:
            return LogFieldDict(expected_type=expected_type)
        return LogFieldDict(v, expected_type=expected_type)

    return validator


def _logfield_serializer(v):
    """Serializer for LogField and LogFieldDict."""
    if isinstance(v, (LogField, LogFieldDict)):
        return v.get()
    if isinstance(v, Path):
        return str(v)
    return v


# ----------------------------
# TRITONSWMM_log base model (used for creating sim and analysis logs)
# ----------------------------
class TRITONSWMM_log(BaseModel):
    logfile: Path
    # ----------------------------
    # Pydantic config
    # ----------------------------
    model_config = {"arbitrary_types_allowed": True}
    # Snapshot of the on-disk state this instance last synced with (set at load
    # and after each write). write() overlays ONLY fields changed since this
    # baseline onto the latest disk state, so a concurrent writer's updates to
    # OTHER fields are never clobbered (lost-update prevention).
    _baseline: dict = PrivateAttr(default_factory=dict)

    # ----------------------------
    # Parent injection
    # ----------------------------
    def model_post_init(self, __context):
        """
        Bind this TRITONSWMM_scenario_log instance to all child log-aware objects.
        """
        for value in self.__dict__.values():
            if hasattr(value, "set_log"):
                value.set_log(self)
        # Baseline = this instance's current serialized state (model defaults for
        # a fresh log; overwritten with loaded data by from_json / refresh).
        self._baseline = self.as_dict()

    # ----------------------------
    # Persistence helpers
    # ----------------------------
    def as_dict(self):
        return self.model_dump(mode="json")

    def write(self):
        """Persist this log with concurrency-safe, lost-update-free semantics.

        Multiple processes (per-sim jobs, consolidate jobs, analysis
        constructions) may write the SAME log file concurrently. Under an
        exclusive per-log file lock we reload the latest on-disk state and
        overlay ONLY the fields THIS instance changed since it last synced
        (compared against the merge baseline), so a concurrent writer's updates
        to OTHER fields survive. write_json itself is atomic
        (temp + fsync + os.replace).
        """
        lock_path = self.logfile.with_suffix(f"{self.logfile.suffix}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # Bounded timeout: the merge is sub-second; a stale/dead lock holder must
        # not deadlock every writer (SE-F-I-2). A filelock timeout is translated
        # to a ProcessingError naming this log file.
        try:
            with resolve_filelock(str(lock_path), timeout=30):
                disk: dict = {}
                if self.logfile.exists():
                    try:
                        with self.logfile.open() as f:
                            disk = json.load(f)
                    except (json.JSONDecodeError, OSError) as exc:
                        # An EXISTING log that will not parse/read is NOT the same as
                        # an absent log. Degrading to {} here silently discards every
                        # field this instance did not change, which is how a
                        # reconstruction can emit an all-defaults document over a log
                        # that carried irreplaceable provenance. Preserve the bytes
                        # beside the log and say so, then proceed (write() must never
                        # abort a run over a diagnostic).
                        _quarantine = self.logfile.with_suffix(
                            f"{self.logfile.suffix}.unreadable"
                        )
                        try:
                            _quarantine.write_bytes(self.logfile.read_bytes())
                        except OSError:
                            _quarantine = None
                        logging.getLogger(__name__).warning(
                            "Log file %s exists but could not be read (%s: %s); "
                            "this write will NOT preserve its unchanged fields. "
                            "Original bytes preserved at %s.",
                            self.logfile,
                            type(exc).__name__,
                            exc,
                            _quarantine if _quarantine is not None else "<preserve failed>",
                        )
                        disk = {}
                mine = self.as_dict()
                changed_keys = {
                    k for k, v in mine.items() if v != self._baseline.get(k)
                }
                # Overlay disk's value for every field I did NOT change, so a
                # concurrent writer's updates win on those fields; my changed
                # fields and required fields (e.g. logfile) come from mine.
                # `k in mine` (SE-F-I-1 Spec 2) drops undeclared keys so removed
                # all_* fields are NOT resurrected on write — closes the
                # resurrection vector for any future field removal.
                overlay = {
                    k: v
                    for k, v in disk.items()
                    if k not in changed_keys and k in mine
                }
                merged = {**mine, **overlay}
                # A write must never turn a NON-NULL on-disk value into null/absent
                # for a field this instance did not deliberately change. The overlay
                # above restores every unchanged declared field, so the only way that
                # happens is the `k in mine` filter: THIS process's model does not
                # DECLARE a field the on-disk log carries — a version-skewed writer.
                # Warn loudly and name the fields; the write still proceeds, because
                # refusing it would strand a run over a diagnostic. A deliberate
                # field REMOVAL shipped with a migration also lands here; that is
                # why this is a warning and never a raise.
                _dropped = sorted(
                    k
                    for k, v in disk.items()
                    if v is not None and k not in changed_keys and merged.get(k) is None
                )
                if _dropped:
                    logging.getLogger(__name__).warning(
                        "Log write to %s DROPS %d non-null on-disk field(s) this "
                        "model does not declare: %s. The writing process is running "
                        "an hhemt version older than the one that wrote the log; the "
                        "dropped values are NOT recoverable from the rewritten file.",
                        self.logfile,
                        len(_dropped),
                        ", ".join(_dropped),
                    )
                write_json(merged, self.logfile)
                self._baseline = merged
        except Timeout as exc:
            raise ProcessingError(
                "log write (file lock acquisition)",
                filepath=self.logfile,
                reason=f"timed out after 30s acquiring {lock_path}",
            ) from exc

    def _dict_for_json(self, obj):
        if isinstance(obj, dict):
            return {k: self._dict_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._dict_for_json(x) for x in obj]
        elif isinstance(obj, (LogField, LogFieldDict)):
            return self._dict_for_json(obj.get())
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def _as_json(self, indent: int = 4):
        return json.dumps(self._dict_for_json(self.as_dict()), indent=indent)

    def print(self, indent: int = 4):
        print(self._as_json(indent))

    def refresh(self):
        """Reload the log from disk, updating all instance attributes."""
        if self.logfile.exists():
            reloaded = self.from_json(self.logfile)
            # Copy all attributes from reloaded instance to self
            for key, value in reloaded.__dict__.items():
                setattr(self, key, value)
            # Re-sync the merge baseline to the freshly reloaded disk state.
            self._baseline = self.as_dict()
        else:
            pass

    @classmethod
    def from_json(cls, path: Path | str):
        path = Path(path)
        try:
            with path.open() as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logging.getLogger(__name__).warning(
                "Log file %s was empty or corrupted; rebuilding log from defaults.",
                path,
            )
            data = None

        if data:
            log = cls.model_validate(data)
        else:
            log = cls(logfile=path)

        # Ensure future writes go back to the same file
        log.logfile = path
        # Baseline = the on-disk state just loaded, so write() overlays only the
        # fields THIS instance subsequently changes (lost-update prevention).
        log._baseline = log.as_dict()

        return log


class TRITONSWMM_scenario_log(TRITONSWMM_log):
    event_iloc: int = 0
    event_idx: Dict = Field(default_factory=dict)
    simulation_folder: Path = Path(".")
    logfile: Path

    # ----------------------------
    # Log fields
    # ----------------------------
    # SWMM stuff
    swmm_rainfall_dat_files: LogFieldDict[Path] = Field(default_factory=LogFieldDict)
    storm_tide_for_swmm: LogField[Path] = Field(default_factory=LogField)
    # scenario creation
    scenario_creation_complete: LogField[bool] = Field(default_factory=LogField)
    inp_hydraulics_model_created_successfully: LogField[bool] = Field(
        default_factory=LogField
    )
    inp_full_model_created_successfully: LogField[bool] = Field(
        default_factory=LogField
    )
    inp_hydro_model_created_successfully: LogField[bool] = Field(
        default_factory=LogField
    )
    hydro_swmm_sim_completed: LogField[bool] = Field(default_factory=LogField)
    extbc_tseries_created: LogField[bool] = Field(default_factory=LogField)
    extbc_loc_created: LogField[bool] = Field(default_factory=LogField)
    hyg_timeseries_created: LogField[bool] = Field(default_factory=LogField)
    hyg_locs_created: LogField[bool] = Field(default_factory=LogField)
    inflow_nodes_in_hydraulic_inp_assigned: LogField[bool] = Field(
        default_factory=LogField
    )
    triton_swmm_cfg_created: LogField[bool] = Field(default_factory=LogField)
    triton_cfg_created: LogField[bool] = Field(
        default_factory=LogField
    )  # TRITON-only CFG
    sim_tritonswmm_executable_copied: LogField[bool] = Field(default_factory=LogField)
    # Track which backend was used for this scenario
    triton_backend_used: LogField[str] = Field(
        default_factory=LogField
    )  # "cpu" or "gpu"
    # This log tracks scenario preparation state only.

    # ----------------------------
    # Consolidated validators using helper functions
    # ----------------------------
    # Boolean LogFields (scenario preparation only, no model-specific fields)
    _validate_bool_fields = field_validator(
        "scenario_creation_complete",
        "inp_hydraulics_model_created_successfully",
        "inp_full_model_created_successfully",
        "inp_hydro_model_created_successfully",
        "hydro_swmm_sim_completed",
        "extbc_tseries_created",
        "extbc_loc_created",
        "hyg_timeseries_created",
        "hyg_locs_created",
        "inflow_nodes_in_hydraulic_inp_assigned",
        "triton_swmm_cfg_created",
        "triton_cfg_created",  # TRITON-only CFG
        "sim_tritonswmm_executable_copied",
        mode="before",
    )(_create_logfield_validator(bool))

    # String LogFields
    _validate_string_fields = field_validator(
        "triton_backend_used",
        mode="before",
    )(_create_logfield_validator(str))

    # Path LogFields
    _validate_path_field = field_validator(
        "storm_tide_for_swmm",
        mode="before",
    )(_create_logfield_validator(Path))

    # LogFieldDict
    _validate_dict_field = field_validator(
        "swmm_rainfall_dat_files",
        mode="before",
    )(_create_logfielddict_validator(Path))

    # ----------------------------
    # Consolidated serializer (scenario preparation only, no model-specific fields)
    # ----------------------------
    _serialize_logfields = field_serializer(
        "swmm_rainfall_dat_files",
        "storm_tide_for_swmm",
        "scenario_creation_complete",
        "inp_hydraulics_model_created_successfully",
        "inp_full_model_created_successfully",
        "inp_hydro_model_created_successfully",
        "hydro_swmm_sim_completed",
        "extbc_tseries_created",
        "extbc_loc_created",
        "hyg_timeseries_created",
        "hyg_locs_created",
        "inflow_nodes_in_hydraulic_inp_assigned",
        "triton_swmm_cfg_created",
        "triton_cfg_created",  # TRITON-only CFG
        "sim_tritonswmm_executable_copied",
        "triton_backend_used",
    )(_logfield_serializer)


class TRITONSWMM_model_log(TRITONSWMM_log):
    """
    Processing log for a single model type (triton, tritonswmm, or swmm).

    Fields are Optional and only populated for relevant model types.
    This eliminates race conditions by giving each model its own log file.

    Field population by model type:
    - Common fields (all): simulation_completed, sim_run_time_minutes, processing_log
    - Performance fields (triton, tritonswmm): performance_timeseries_written, performance_summary_written
    - TRITON fields (triton, tritonswmm): TRITON_timeseries_written, TRITON_summary_written, raw_TRITON_outputs_cleared, full_TRITON_timeseries_cleared
    - SWMM fields (swmm, tritonswmm): SWMM_node/link_timeseries_written, SWMM_node/link_summary_written, raw_SWMM_outputs_cleared, full_SWMM_timeseries_cleared, raw_SWMM_binaries_reclaimed, coupled_rpt_truncated, hydro_out_reclaimed
    """

    event_iloc: int = 0
    event_idx: Dict = Field(default_factory=dict)
    simulation_folder: Path = Path(".")
    logfile: Path

    # Common fields (all model types)
    # Simulation execution
    simulation_completed: LogField[bool] = Field(default_factory=LogField)
    sim_run_time_minutes: LogField[float] = Field(default_factory=LogField)
    # n_resumes counts hotstart resumes for this (model_type, event). Incremented
    # at the resume-decision site in run_simulation.py. Unset on legacy logs;
    # consumers MUST coalesce None -> 0 (LogField.get() returns None when unset).
    n_resumes: LogField[int] = Field(default_factory=LogField)
    # resume_reporting_tsteps records the REALIZED resume reporting-step (config_NNNN.cfg
    # NNNN = the reporting step, return_the_reporting_step_from_a_cfg) per resume attempt,
    # appended in run_simulation.py's hotstart branch. First-class durable record of WHERE
    # each config actually resumed, so the b4b same-timestep-interruption experiment is
    # answerable from the data forever (KR-b). Legacy logs omit it; consumers coalesce None -> [].
    resume_reporting_tsteps: LogField[list[int]] = Field(default_factory=LogField)
    processing_log: Processing = Field(default_factory=Processing)

    # Performance timeseries (triton and tritonswmm only)
    performance_timeseries_written: Optional[LogField[bool]] = None
    performance_summary_written: Optional[LogField[bool]] = None

    # TRITON outputs (triton and tritonswmm only)
    TRITON_timeseries_written: Optional[LogField[bool]] = None
    TRITON_summary_written: Optional[LogField[bool]] = None
    raw_TRITON_outputs_cleared: Optional[LogField[bool]] = None
    full_TRITON_timeseries_cleared: Optional[LogField[bool]] = None

    # SWMM outputs (swmm and tritonswmm only)
    SWMM_node_timeseries_written: Optional[LogField[bool]] = None
    SWMM_link_timeseries_written: Optional[LogField[bool]] = None
    SWMM_node_summary_written: Optional[LogField[bool]] = None
    SWMM_link_summary_written: Optional[LogField[bool]] = None
    raw_SWMM_outputs_cleared: Optional[LogField[bool]] = None
    full_SWMM_timeseries_cleared: Optional[LogField[bool]] = None
    # Post-processing reclaim (analysis_config.remove_after_processing) of
    # out_tritonswmm/swmm/*.out -- the coupled-SWMM binary outputs, NEVER the .rpt, which
    # is a live completion predicate. Distinct from raw_SWMM_outputs_cleared, which records
    # the clear_raw pass over the RAW SIM tree; these are two axes and two records.
    # Additive + defaulted None, so every pre-existing log_{model}.json deserialises
    # unchanged and consumers coalesce None -> not reclaimed.
    raw_SWMM_binaries_reclaimed: Optional[LogField[bool]] = None
    # Post-processing TRUNCATION of out_tritonswmm/swmm/hydraulics.rpt to its header,
    # continuity/summary tables and trailer. TRUNCATION, never deletion -- the file remains
    # the coupled-run completion signal (Gotcha 70).
    coupled_rpt_truncated: Optional[LogField[bool]] = None
    # Post-processing reclaim of swmm/hydro.out, the SWMM hydrology output that
    # write_hydrograph_files reads. Distinct from raw_SWMM_outputs_cleared (the clear_raw
    # pass over the RAW SIM tree) and from raw_SWMM_binaries_reclaimed (out_tritonswmm/swmm
    # binaries): three axes, three records.
    hydro_out_reclaimed: Optional[LogField[bool]] = None
    # Three ADDITIVE records for the regeneration-cost classes. Same change class as the
    # allowlisted additive fields above: Optional and defaulted None, so every
    # pre-existing log_{model}.json deserialises unchanged and consumers coalesce
    # None -> not reclaimed. These are NEW keys, never renames of existing ones --
    # renaming an on-disk log key is the break that WOULD owe a LAYOUT_VERSION bump.
    prep_inputs_reclaimed: Optional[LogField[bool]] = None
    hydrographs_reclaimed: Optional[LogField[bool]] = None
    standalone_rpt_reclaimed: Optional[LogField[bool]] = None

    # Validators for LogField types
    _validate_bool_fields = field_validator(
        "simulation_completed",
        "performance_timeseries_written",
        "performance_summary_written",
        "TRITON_timeseries_written",
        "TRITON_summary_written",
        "raw_TRITON_outputs_cleared",
        "full_TRITON_timeseries_cleared",
        "SWMM_node_timeseries_written",
        "SWMM_link_timeseries_written",
        "SWMM_node_summary_written",
        "SWMM_link_summary_written",
        "raw_SWMM_outputs_cleared",
        "full_SWMM_timeseries_cleared",
        "raw_SWMM_binaries_reclaimed",
        "coupled_rpt_truncated",
        "hydro_out_reclaimed",
        mode="before",
    )(_create_logfield_validator(bool))

    _validate_float_fields = field_validator(
        "sim_run_time_minutes",
        mode="before",
    )(_create_logfield_validator(float))

    _validate_int_fields = field_validator(
        "n_resumes",
        mode="before",
    )(_create_logfield_validator(int))

    # resume_reporting_tsteps is LogField[list[int]] (KR-b) — a list-typed LogField, so it
    # gets its own validator group rather than the scalar-int coercion above.
    _validate_list_fields = field_validator(
        "resume_reporting_tsteps",
        mode="before",
    )(_create_logfield_validator(list))

    # Serializers
    _serialize_bool_fields = field_serializer(
        "simulation_completed",
        "performance_timeseries_written",
        "performance_summary_written",
        "TRITON_timeseries_written",
        "TRITON_summary_written",
        "raw_TRITON_outputs_cleared",
        "full_TRITON_timeseries_cleared",
        "SWMM_node_timeseries_written",
        "SWMM_link_timeseries_written",
        "SWMM_node_summary_written",
        "SWMM_link_summary_written",
        "raw_SWMM_outputs_cleared",
        "full_SWMM_timeseries_cleared",
        "raw_SWMM_binaries_reclaimed",
        "coupled_rpt_truncated",
        "hydro_out_reclaimed",
        when_used="json",
    )(lambda self, v: v.get() if v is not None else None)

    _serialize_float_fields = field_serializer(
        "sim_run_time_minutes",
        when_used="json",
    )(lambda self, v: v.get() if v is not None else None)

    _serialize_int_fields = field_serializer(
        "n_resumes",
        when_used="json",
    )(lambda self, v: v.get() if v is not None else None)

    _serialize_list_fields = field_serializer(
        "resume_reporting_tsteps",
        when_used="json",
    )(lambda self, v: v.get() if v is not None else None)

    def model_post_init(self, __context):
        """Bind this model log to all nested log-aware fields."""
        super().model_post_init(__context)

    def add_sim_processing_entry(
        self,
        filepath: Path,
        size_MiB: float,
        time_elapsed_s: float,
        success: bool,
        notes: str = "",
        warnings: str = "",
    ):
        """Add a processing entry to the processing log."""
        simlog = ProcessingEntry(
            filepath=filepath,
            size_MiB=size_MiB,
            time_elapsed_s=time_elapsed_s,
            success=success,
            notes=notes,
            warnings=warnings,
        )
        self.processing_log.update(simlog)


class TRITONSWMM_system_log(TRITONSWMM_log):
    """System-level log tracking compilation and preprocessing status."""

    # DEM and Manning's preprocessing
    dem_processed: LogField[bool] = Field(default_factory=LogField)
    dem_shape: LogField[tuple] = Field(default_factory=LogField)
    mannings_processed: LogField[bool] = Field(default_factory=LogField)
    mannings_shape: LogField[tuple] = Field(default_factory=LogField)

    # TRITON-SWMM compilation
    compilation_tritonswmm_cpu_successful: LogField[bool] = Field(
        default_factory=LogField
    )
    compilation_tritonswmm_gpu_successful: LogField[bool] = Field(
        default_factory=LogField
    )

    # TRITON-only compilation
    compilation_triton_cpu_successful: LogField[bool] = Field(default_factory=LogField)
    compilation_triton_gpu_successful: LogField[bool] = Field(default_factory=LogField)

    # SWMM compilation
    compilation_swmm_successful: LogField[bool] = Field(default_factory=LogField)

    # TRITON provenance capture (D2). Captured at compile time in system.py against the
    # ACTUAL cloned TRITON tree, immediately after _verify_tritonswmm_pin. This is the named
    # persistence carrier between the compile process (setup_workflow) and the consolidation
    # process (consolidate_workflow) — different SLURM jobs on HPC — read by the two
    # consolidation stamp sites via analysis._system.log.triton_head_sha.get() and stamped
    # onto the consolidated tree root as `triton_producing_sha` (the rename is deliberate;
    # grepping either name alone finds only half the chain).
    #
    # ONE FIELD, NOT THREE. The two sibling booleans that lived here
    # (triton_has_coupled_resume_fix, triton_has_swmm_depth_scatter_fix) are RETIRED: they were
    # per-defect verdicts DERIVED from a toolkit constant and frozen into durable state at run
    # time, so a later toolkit change could not correct them. Dated instance: the scatter
    # constant was introduced as None in 463ab9e (2026-08-02) beside an `is None -> set(False)`
    # branch, the synth_cc campaign ran 2026-08-04 inside that window and was stamped False on
    # every arm, and the constant received its real value at 7b7b573 (2026-08-05) — after the
    # sims were on disk. The stamp is refuted by direct measurement (the producing sha IS the
    # fix commit) and no re-consolidation can correct it, because the flag is re-derived only on
    # the compile path. Defect applicability is now DERIVED at read time from this sha by
    # `model_defects.resolve_for_tree_attrs`, so nothing durable is written that a later registry
    # edit can contradict. Retiring the two keys is safe on legacy logs: TRITONSWMM_log sets no
    # `extra` policy, so pydantic's default `ignore` absorbs them on from_json.
    triton_head_sha: LogField[str] = Field(default_factory=LogField)

    # PER-BUILD-TARGET TRITON provenance (contract item i). `triton_head_sha` above is
    # RETAINED with its exact meaning and every one of its consumers is untouched --
    # including model_defects.resolve_for_tree_attrs, which keys every registered defect
    # verdict on it. The two below are ADDITIVE and answer a different question: not "what
    # source did this analysis come from" but "were the two TRITON binaries built from the
    # SAME source state". They are written ONLY by a compile invocation that actually built
    # that target (the *_locked helpers now return a build-executed bool), so a target
    # cached from an older clone state legitimately retains an OLDER sha than the clone --
    # that divergence is the point, not a defect. Under the ordinary case of one clone and
    # one run they are equal, and their equality is then a MEASUREMENT over two independent
    # stamps rather than one value written twice under two names.
    tritonswmm_producing_sha: LogField[str] = Field(default_factory=LogField)
    triton_only_producing_sha: LogField[str] = Field(default_factory=LogField)

    # Standalone-SWMM provenance capture. The FOURTH contract axis, and the only one that
    # had no carrier at all: `swmm_version` appeared in exactly two files, both
    # container-preflight, and in no writer. Captured in system.py at the tail of
    # `_compile_SWMM_locked`'s SUCCESS branch -- deliberately NOT at compile entry the way
    # `triton_head_sha` is, per `provenance.producing_stamp`'s corollary that a stage which
    # did not execute must not stamp. The already-compiled gate returns early above the
    # capture and the persisted log retains the earlier run's value, so a cached build
    # keeps a true stamp rather than acquiring a false one.
    #
    # The `standalone_` prefix is load-bearing TWICE. It names the thing (the COUPLED
    # model's SWMM is vendored inside TRITON and travels with the TRITON pin, so it is NOT
    # this field), and it keeps the identifier from being a SUFFIX of the coupled sha
    # declared above -- a bare `swmm_producing_sha` is a substring of it, so every future
    # grep for one would silently match the other.
    #
    # TWO FIELDS, ONE KIND EACH. The version field is tag-shaped and available in BOTH
    # modes (native: the built clone's CMakeLists `VERSION`; container: the
    # org.hhemt.swmm_version label), normalized by stripping a leading "v" so the two modes
    # cannot disagree on shape alone. The sha field is native-only and is left ABSENT in
    # container mode -- a SIF label carries no sha, and inferring one from the tag would
    # launder an assumption into an apparent measurement.
    standalone_swmm_producing_version: LogField[str] = Field(default_factory=LogField)
    standalone_swmm_producing_sha: LogField[str] = Field(default_factory=LogField)

    # System-level DataTree consolidation
    system_datatree_consolidation_complete: LogField[bool] = Field(
        default_factory=LogField
    )
    dem_crs_epsg: LogField[int] = Field(default_factory=LogField)
    vertical_crs_epsg: LogField[int] = Field(default_factory=LogField)

    # ----------------------------
    # Consolidated validators
    # ----------------------------
    _validate_bool_fields = field_validator(
        "dem_processed",
        "mannings_processed",
        "compilation_tritonswmm_cpu_successful",
        "compilation_tritonswmm_gpu_successful",
        "compilation_triton_cpu_successful",
        "compilation_triton_gpu_successful",
        "compilation_swmm_successful",
        "system_datatree_consolidation_complete",
        mode="before",
    )(_create_logfield_validator(bool))

    _validate_tuple_fields = field_validator(
        "dem_shape",
        "mannings_shape",
        mode="before",
    )(_create_logfield_validator(tuple))

    _validate_string_fields = field_validator(
        "triton_head_sha",
        "tritonswmm_producing_sha",
        "triton_only_producing_sha",
        "standalone_swmm_producing_version",
        "standalone_swmm_producing_sha",
        mode="before",
    )(_create_logfield_validator(str))

    _validate_int_fields = field_validator(
        "dem_crs_epsg",
        "vertical_crs_epsg",
        mode="before",
    )(_create_logfield_validator(int))

    # ----------------------------
    # Consolidated serializer
    # ----------------------------
    _serialize_logfields = field_serializer(
        "dem_processed",
        "dem_shape",
        "mannings_processed",
        "mannings_shape",
        "compilation_tritonswmm_cpu_successful",
        "compilation_tritonswmm_gpu_successful",
        "compilation_triton_cpu_successful",
        "compilation_triton_gpu_successful",
        "compilation_swmm_successful",
        "system_datatree_consolidation_complete",
        "triton_head_sha",
        "tritonswmm_producing_sha",
        "triton_only_producing_sha",
        "standalone_swmm_producing_version",
        "standalone_swmm_producing_sha",
        "dem_crs_epsg",
        "vertical_crs_epsg",
    )(_logfield_serializer)


class TRITONSWMM_analysis_log(TRITONSWMM_log):
    # Hierarchical DataTree consolidation (Phase 2)
    datatree_consolidation_complete: LogField[bool] = Field(
        default_factory=LogField
    )
    consolidation_version: LogField[int] = Field(default_factory=LogField)
    # Fingerprint of the inputs that determine the SHAPE of the consolidated tree
    # (see processing_analysis.py::_consolidation_inputs_fingerprint). The
    # consolidate guard treats a mismatch OR an absent stamp as stale and rebuilds,
    # so a consolidation-affecting config change (e.g. toggle_consolidate_timeseries)
    # invalidates an otherwise-complete tree without any operator action.
    consolidation_inputs_fingerprint: LogField[str] = Field(default_factory=LogField)
    # Sensitivity-level DataTree consolidation (Phase 3)
    sensitivity_datatree_consolidation_complete: LogField[bool] = Field(
        default_factory=LogField
    )
    # Track which backends are available at analysis creation time
    cpu_backend_available: LogField[bool] = Field(default_factory=LogField)
    gpu_backend_available: LogField[bool] = Field(default_factory=LogField)
    processing_log: Processing = Field(default_factory=Processing)

    # Workflow submission tracking (for tmux-based orchestration)
    tmux_session_name: LogField[str] = Field(default_factory=LogField)
    snakemake_pid: LogField[int] = Field(default_factory=LogField)
    workflow_submission_time: LogField[str] = Field(default_factory=LogField)
    workflow_submission_mode: LogField[str] = Field(default_factory=LogField)  # "tmux", "batch_job", etc.
    workflow_submission_node: LogField[str] = Field(default_factory=LogField)  # login node hostname at submission time
    orchestrator_slurm_jobid: LogField[str] = Field(default_factory=LogField)  # E2: single-job orchestrator jobid
    workflow_canceled: LogField[bool] = Field(default_factory=LogField)
    workflow_cancellation_time: LogField[str] = Field(default_factory=LogField)
    # multi_allocation_in_progress is an INERT defense-in-depth backstop
    # (resume-retry-resilience P3): the _clear_raw_outputs guard raises if this field
    # is ever True, refusing a raw-output delete that would strip pre-resume
    # performance{N}.txt checkpoints the V0008 aggregator depends on. Production
    # auto-wiring (when to set True/False) is DEFERRED: by DAG construction the per-sim
    # clear already fires only after that sim's c_run completion flag (i.e.
    # post-final-allocation), so an analysis-level set-True would over-block finished
    # sims while siblings resume AND poison the detached-batch_job processing path
    # (which reloads a fresh disk log). Correct per-sim enforcement needs a real
    # two-allocation batch_job run to validate (see the deferred follow-up). Unset by
    # default -> the guard never fires in production today; consumers coalesce
    # None -> not-in-progress.
    multi_allocation_in_progress: LogField[bool] = Field(default_factory=LogField)

    # ----------------------------
    # Consolidated validators using helper functions
    # ----------------------------
    _validate_bool_fields = field_validator(
        "datatree_consolidation_complete",
        "sensitivity_datatree_consolidation_complete",
        "cpu_backend_available",
        "gpu_backend_available",
        "workflow_canceled",
        "multi_allocation_in_progress",
        mode="before",
    )(_create_logfield_validator(bool))

    _validate_workflow_str_fields = field_validator(
        "tmux_session_name",
        "workflow_submission_time",
        "workflow_submission_mode",
        "workflow_cancellation_time",
        "workflow_submission_node",
        "orchestrator_slurm_jobid",
        "consolidation_inputs_fingerprint",
        mode="before",
    )(_create_logfield_validator(str))

    _validate_workflow_int_fields = field_validator(
        "snakemake_pid",
        "consolidation_version",
        mode="before",
    )(_create_logfield_validator(int))

    # ----------------------------
    # Consolidated serializer
    # ----------------------------
    _serialize_logfields = field_serializer(
        "datatree_consolidation_complete",
        "consolidation_version",
        "consolidation_inputs_fingerprint",
        "sensitivity_datatree_consolidation_complete",
        "cpu_backend_available",
        "gpu_backend_available",
        "tmux_session_name",
        "snakemake_pid",
        "workflow_submission_time",
        "workflow_submission_mode",
        "workflow_canceled",
        "workflow_cancellation_time",
        "workflow_submission_node",
        "orchestrator_slurm_jobid",
        "multi_allocation_in_progress",
    )(_logfield_serializer)

    # ----------------------------
    # Processing entries
    # ----------------------------
    def add_sim_processing_entry(
        self,
        filepath: Path,
        size_MiB: float,
        time_elapsed_s: float,
        success: bool,
        notes: str = "",
        warnings: str = "",
    ):
        simlog = ProcessingEntry(
            filepath=filepath,
            size_MiB=round(size_MiB, 2),
            time_elapsed_s=round(time_elapsed_s, 2),
            success=success,
            notes=notes,
            warnings=warnings,
        )
        self.processing_log.update(simlog)


def log_function_to_file(logfile_path: Path):
    """Decorator to log Python messages and exceptions to a file safely for concurrent runs."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # ---------- Change 1: Create a unique logger for this function call ----------
            logger = logging.getLogger(f"{func.__name__}_{id(func)}")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            logger.handlers.clear()  # Remove any existing handlers

            # ---------- Change 2: Ensure logfile directory exists and file exists ----------
            logfile_path.parent.mkdir(parents=True, exist_ok=True)
            logfile_path.touch(exist_ok=True)  # Make sure file exists immediately

            # ---------- Change 3: FileHandler with explicit flush ----------
            fh = logging.FileHandler(logfile_path, mode="a", encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            try:
                # Pass logger into the wrapped function
                return func(*args, logger=logger, **kwargs)
            except Exception:
                logger.exception("Exception occurred during function execution")
                raise
            finally:
                # ---------- Change 4: Explicit flush before closing ----------
                fh.flush()  # Ensures all logs are written to disk before checking exists()
                logger.removeHandler(fh)
                fh.close()

        return wrapper

    return decorator
