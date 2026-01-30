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
# Simulation log entries
# ----------------------------
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
    _log: "TRITONSWMM_scenario_log" = PrivateAttr()
    run_attempts: Dict[str, SimEntry] = Field(default_factory=dict)

    def set_log(self, log: "TRITONSWMM_scenario_log"):
        self._log = log

    def update(self, entry: SimEntry):
        self.run_attempts[entry.sim_datetime] = entry
        self._log.write()


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
    _log: "TRITONSWMM_scenario_log" = PrivateAttr()
    outputs: Dict[str, ProcessingEntry] = Field(default_factory=dict)

    def set_log(self, log: "TRITONSWMM_scenario_log"):
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

        with path.open() as f:
            data = json.load(f)

        log = cls.model_validate(data)

        # Ensure future writes go back to the same file
        log.logfile = path

        return log


class TRITONSWMM_scenario_log(TRITONSWMM_log):
    event_iloc: int
    event_idx: Dict
    simulation_folder: Path
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
    sim_tritonswmm_executable_copied: LogField[bool] = Field(default_factory=LogField)
    # Track which backend was used for this scenario
    triton_backend_used: LogField[str] = Field(default_factory=LogField)  # "cpu" or "gpu"
    # RUNNING SIMULATIONS
    simulation_completed: LogField[bool] = Field(default_factory=LogField)
    sim_log: SimLog = Field(default_factory=SimLog)
    # POST PROCESSING
    TRITONSWMM_performance_timeseries_written: LogField[bool] = Field(
        default_factory=LogField
    )
    TRITONSWMM_performance_summary_written: LogField[bool] = Field(
        default_factory=LogField
    )
    TRITON_timeseries_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_node_timeseries_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_link_timeseries_written: LogField[bool] = Field(default_factory=LogField)
    raw_TRITON_outputs_cleared: LogField[bool] = Field(default_factory=LogField)
    raw_SWMM_outputs_cleared: LogField[bool] = Field(default_factory=LogField)
    # SUMMARY PROCESSING
    TRITON_summary_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_node_summary_written: LogField[bool] = Field(default_factory=LogField)
    SWMM_link_summary_written: LogField[bool] = Field(default_factory=LogField)
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
        "sim_tritonswmm_executable_copied",
        "simulation_completed",
        "TRITON_timeseries_written",
        "TRITONSWMM_performance_timeseries_written",
        "TRITONSWMM_performance_summary_written",
        "SWMM_node_timeseries_written",
        "SWMM_link_timeseries_written",
        "raw_TRITON_outputs_cleared",
        "raw_SWMM_outputs_cleared",
        "TRITON_summary_written",
        "SWMM_node_summary_written",
        "SWMM_link_summary_written",
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
        "sim_tritonswmm_executable_copied",
        "triton_backend_used",
        "simulation_completed",
        "TRITON_timeseries_written",
        "TRITONSWMM_performance_timeseries_written",
        "TRITONSWMM_performance_summary_written",
        "SWMM_node_timeseries_written",
        "SWMM_link_timeseries_written",
        "raw_TRITON_outputs_cleared",
        "raw_SWMM_outputs_cleared",
        "TRITON_summary_written",
        "SWMM_node_summary_written",
        "SWMM_link_summary_written",
        "full_TRITON_timeseries_cleared",
        "full_SWMM_timeseries_cleared",
    )(_logfield_serializer)

    # ----------------------------
    # Simulation entries
    # ----------------------------
    def add_sim_entry(
        self,
        sim_datetime: str,
        sim_start_reporting_tstep: int | float,
        tritonswmm_logfile: Path,
        time_elapsed_s: float,
        status: str,
        run_mode: str,
        cmd: str,
        n_mpi_procs: int | float | None,
        n_omp_threads: int | float | None,
        n_gpus: int | float | None,
        env: dict,
    ):
        simlog = SimEntry(
            sim_datetime=sim_datetime,
            sim_start_reporting_tstep=sim_start_reporting_tstep,
            tritonswmm_logfile=tritonswmm_logfile,
            time_elapsed_s=time_elapsed_s,
            status=status,
            run_mode=run_mode,
            cmd=cmd,
            n_mpi_procs=n_mpi_procs,
            n_omp_threads=n_omp_threads,
            n_gpus=n_gpus,
            env=env,
        )
        self.sim_log.update(simlog)

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
    TRITON_analysis_summary_created: LogField[bool] = Field(default_factory=LogField)
    SWMM_node_analysis_summary_created: LogField[bool] = Field(default_factory=LogField)
    SWMM_link_analysis_summary_created: LogField[bool] = Field(default_factory=LogField)
    TRITONSWMM_performance_analysis_summary_created: LogField[bool] = Field(
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
        "TRITON_analysis_summary_created",
        "SWMM_node_analysis_summary_created",
        "SWMM_link_analysis_summary_created",
        "TRITONSWMM_performance_analysis_summary_created",
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
        "TRITON_analysis_summary_created",
        "SWMM_node_analysis_summary_created",
        "SWMM_link_analysis_summary_created",
        "TRITONSWMM_performance_analysis_summary_created",
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
