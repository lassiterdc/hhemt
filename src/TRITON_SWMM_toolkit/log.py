from TRITON_SWMM_toolkit.utils import write_json
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
# Simulation log entries (DEPRECATED - kept for backward compatibility)
# ----------------------------
# NOTE: These classes are deprecated. Simulation completion is now tracked via log files
# (run_triton.log, run_tritonswmm.log, run_swmm.log) instead of simlog entries.
# These are kept only to prevent errors when loading existing log files.


class SimEntry(BaseModel):
    sim_datetime: str
    sim_start_reporting_tstep: int | float
    tritonswmm_logfile: Path
    time_elapsed_s: float
    status: str
    run_mode: str
    cmd: str
    n_mpi_procs: int | float | None
    n_omp_threads: int | float | None
    n_gpus: int | float | None
    env: dict


class SimLog(BaseModel):
    _log: "TRITONSWMM_log" = PrivateAttr()
    run_attempts: Dict[str, SimEntry] = Field(default_factory=dict)

    def set_log(self, log: "TRITONSWMM_log"):
        self._log = log

    def update(self, entry: SimEntry):
        # DEPRECATED: No longer persists to log file
        # Completion tracking moved to log files (run_*.log)
        self.run_attempts[entry.sim_datetime] = entry
        # self._log.write()  # Commented out - no longer persist


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

    # ----------------------------
    # Persistence helpers
    # ----------------------------
    def as_dict(self):
        return self.model_dump()

    def write(self):
        write_json(self.as_dict(), self.logfile)

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
    # RUNNING SIMULATIONS
    # DEPRECATED: Completion status now tracked via log files (run_triton.log, run_tritonswmm.log, run_swmm.log)
    # These fields kept for backward compatibility with existing log files
    simulation_completed: LogField[bool] = Field(default_factory=LogField)
    sim_log: SimLog = Field(default_factory=SimLog)
    # POST PROCESSING
    TRITONSWMM_performance_timeseries_written: LogField[bool] = Field(
        default_factory=LogField
    )
    TRITONSWMM_performance_summary_written: LogField[bool] = Field(
        default_factory=LogField
    )
    TRITON_only_performance_timeseries_written: LogField[bool] = Field(
        default_factory=LogField
    )
    TRITON_only_performance_summary_written: LogField[bool] = Field(
        default_factory=LogField
    )
    TRITON_timeseries_written: LogField[bool] = Field(default_factory=LogField)
    TRITON_only_timeseries_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_node_timeseries_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_link_timeseries_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_only_node_timeseries_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_only_link_timeseries_written: LogField[bool] = Field(default_factory=LogField)
    raw_TRITON_outputs_cleared: LogField[bool] = Field(default_factory=LogField)
    raw_SWMM_outputs_cleared: LogField[bool] = Field(default_factory=LogField)
    raw_TRITON_only_outputs_cleared: LogField[bool] = Field(default_factory=LogField)
    raw_SWMM_only_outputs_cleared: LogField[bool] = Field(default_factory=LogField)
    # SUMMARY PROCESSING
    TRITON_summary_written: LogField[bool] = Field(default_factory=LogField)
    TRITON_only_summary_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_node_summary_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_link_summary_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_only_node_summary_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_only_link_summary_written: LogField[bool] = Field(default_factory=LogField)
    # FULL TIMESERIES CLEANUP
    full_TRITON_timeseries_cleared: LogField[bool] = Field(default_factory=LogField)
    full_SWMM_timeseries_cleared: LogField[bool] = Field(default_factory=LogField)
    processing_log: Processing = Field(default_factory=Processing)

    # ----------------------------
    # Consolidated validators using helper functions
    # ----------------------------
    # Boolean LogFields
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
        "simulation_completed",  # DEPRECATED but kept for backward compatibility
        "TRITON_timeseries_written",
        "TRITON_only_timeseries_written",
        "TRITONSWMM_performance_timeseries_written",
        "TRITONSWMM_performance_summary_written",
        "TRITON_only_performance_timeseries_written",
        "TRITON_only_performance_summary_written",
        "SWMM_node_timeseries_written",
        "SWMM_link_timeseries_written",
        "SWMM_only_node_timeseries_written",
        "SWMM_only_link_timeseries_written",
        "raw_TRITON_outputs_cleared",
        "raw_SWMM_outputs_cleared",
        "raw_TRITON_only_outputs_cleared",
        "raw_SWMM_only_outputs_cleared",
        "TRITON_summary_written",
        "TRITON_only_summary_written",
        "SWMM_node_summary_written",
        "SWMM_link_summary_written",
        "SWMM_only_node_summary_written",
        "SWMM_only_link_summary_written",
        "full_TRITON_timeseries_cleared",
        "full_SWMM_timeseries_cleared",
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
    # Consolidated serializer
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
        "simulation_completed",  # DEPRECATED but kept for backward compatibility
        "TRITON_timeseries_written",
        "TRITON_only_timeseries_written",
        "TRITONSWMM_performance_timeseries_written",
        "TRITONSWMM_performance_summary_written",
        "TRITON_only_performance_timeseries_written",
        "TRITON_only_performance_summary_written",
        "SWMM_node_timeseries_written",
        "SWMM_link_timeseries_written",
        "SWMM_only_node_timeseries_written",
        "SWMM_only_link_timeseries_written",
        "raw_TRITON_outputs_cleared",
        "raw_SWMM_outputs_cleared",
        "raw_TRITON_only_outputs_cleared",
        "raw_SWMM_only_outputs_cleared",
        "TRITON_summary_written",
        "TRITON_only_summary_written",
        "SWMM_node_summary_written",
        "SWMM_link_summary_written",
        "SWMM_only_node_summary_written",
        "SWMM_only_link_summary_written",
        "full_TRITON_timeseries_cleared",
        "full_SWMM_timeseries_cleared",
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
            size_MiB=size_MiB,
            time_elapsed_s=time_elapsed_s,
            success=success,
            notes=notes,
            warnings=warnings,
        )
        self.processing_log.update(simlog)


class TRITONSWMM_model_log(TRITONSWMM_log):
    """
    Processing log for a single model type (triton, tritonswmm, or swmm).

    Fields are Optional and only populated for relevant model types.
    This eliminates race conditions by giving each model its own log file.

    Field population by model type:
    - Common fields (all): simulation_completed, sim_run_time_minutes, processing_log, sim_log
    - Performance fields (triton, tritonswmm): performance_timeseries_written, performance_summary_written
    - TRITON fields (triton, tritonswmm): TRITON_timeseries_written, TRITON_summary_written, raw_TRITON_outputs_cleared, full_TRITON_timeseries_cleared
    - SWMM fields (swmm, tritonswmm): SWMM_node/link_timeseries_written, SWMM_node/link_summary_written, raw_SWMM_outputs_cleared, full_SWMM_timeseries_cleared
    """

    event_iloc: int = 0
    event_idx: Dict = Field(default_factory=dict)
    simulation_folder: Path = Path(".")
    logfile: Path

    # Common fields (all model types)
    simulation_completed: LogField[bool] = Field(default_factory=LogField)
    sim_run_time_minutes: LogField[float] = Field(default_factory=LogField)
    sim_log: SimLog = Field(default_factory=SimLog)
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
        mode="before",
    )(_create_logfield_validator(bool))

    _validate_float_fields = field_validator(
        "sim_run_time_minutes",
        mode="before",
    )(_create_logfield_validator(float))

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
        when_used="json",
    )(lambda self, v: v.get() if v is not None else None)

    _serialize_float_fields = field_serializer(
        "sim_run_time_minutes",
        when_used="json",
    )(lambda self, v: v.get() if v is not None else None)

    def model_post_init(self, __context):
        """Set parent log reference for nested objects after initialization."""
        if hasattr(self, "sim_log"):
            self.sim_log.set_log(self)
        if hasattr(self, "processing_log"):
            self.processing_log.set_log(self)

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
        mode="before",
    )(_create_logfield_validator(bool))

    _validate_tuple_fields = field_validator(
        "dem_shape",
        "mannings_shape",
        mode="before",
    )(_create_logfield_validator(tuple))

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
    )(_logfield_serializer)


class TRITONSWMM_analysis_log(TRITONSWMM_log):
    all_scenarios_created: LogField[bool] = Field(default_factory=LogField)
    all_sims_run: LogField[bool] = Field(default_factory=LogField)
    all_TRITON_timeseries_processed: LogField[bool] = Field(default_factory=LogField)
    all_SWMM_timeseries_processed: LogField[bool] = Field(default_factory=LogField)
    all_TRITONSWMM_performance_timeseries_processed: LogField[bool] = Field(
        default_factory=LogField
    )
    all_raw_TRITON_outputs_cleared: LogField[bool] = Field(default_factory=LogField)
    all_raw_SWMM_outputs_cleared: LogField[bool] = Field(default_factory=LogField)
    # TRITON-SWMM coupled model consolidated summaries
    tritonswmm_triton_analysis_summary_created: LogField[bool] = Field(
        default_factory=LogField
    )
    tritonswmm_node_analysis_summary_created: LogField[bool] = Field(
        default_factory=LogField
    )
    tritonswmm_link_analysis_summary_created: LogField[bool] = Field(
        default_factory=LogField
    )
    tritonswmm_performance_analysis_summary_created: LogField[bool] = Field(
        default_factory=LogField
    )
    # TRITON-only consolidated summaries
    triton_only_analysis_summary_created: LogField[bool] = Field(
        default_factory=LogField
    )
    triton_only_performance_analysis_summary_created: LogField[bool] = Field(
        default_factory=LogField
    )
    # SWMM-only consolidated summaries
    swmm_only_node_analysis_summary_created: LogField[bool] = Field(
        default_factory=LogField
    )
    swmm_only_link_analysis_summary_created: LogField[bool] = Field(
        default_factory=LogField
    )
    # Track which backends are available at analysis creation time
    cpu_backend_available: LogField[bool] = Field(default_factory=LogField)
    gpu_backend_available: LogField[bool] = Field(default_factory=LogField)
    processing_log: Processing = Field(default_factory=Processing)

    # ----------------------------
    # Consolidated validators using helper functions
    # ----------------------------
    _validate_bool_fields = field_validator(
        "all_scenarios_created",
        "all_sims_run",
        "all_TRITON_timeseries_processed",
        "all_SWMM_timeseries_processed",
        "all_TRITONSWMM_performance_timeseries_processed",
        "all_raw_TRITON_outputs_cleared",
        "all_raw_SWMM_outputs_cleared",
        "tritonswmm_triton_analysis_summary_created",
        "tritonswmm_node_analysis_summary_created",
        "tritonswmm_link_analysis_summary_created",
        "tritonswmm_performance_analysis_summary_created",
        "triton_only_analysis_summary_created",
        "triton_only_performance_analysis_summary_created",
        "swmm_only_node_analysis_summary_created",
        "swmm_only_link_analysis_summary_created",
        "cpu_backend_available",
        "gpu_backend_available",
        mode="before",
    )(_create_logfield_validator(bool))

    # ----------------------------
    # Consolidated serializer
    # ----------------------------
    _serialize_logfields = field_serializer(
        "all_scenarios_created",
        "all_sims_run",
        "all_TRITON_timeseries_processed",
        "all_SWMM_timeseries_processed",
        "all_TRITONSWMM_performance_timeseries_processed",
        "all_raw_TRITON_outputs_cleared",
        "all_raw_SWMM_outputs_cleared",
        "tritonswmm_triton_analysis_summary_created",
        "tritonswmm_node_analysis_summary_created",
        "tritonswmm_link_analysis_summary_created",
        "tritonswmm_performance_analysis_summary_created",
        "triton_only_analysis_summary_created",
        "triton_only_performance_analysis_summary_created",
        "swmm_only_node_analysis_summary_created",
        "swmm_only_link_analysis_summary_created",
        "cpu_backend_available",
        "gpu_backend_available",
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
