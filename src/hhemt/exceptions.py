"""Custom exception hierarchy for TRITON-SWMM toolkit.

All toolkit-specific exceptions inherit from TRITONSWMMError, allowing
users to catch all toolkit errors with a single except clause while
still providing specific error types for different failure modes.

Each exception stores contextual attributes (file paths, return codes,
model types) to enable programmatic error handling and detailed
error reporting.
"""

from pathlib import Path


class TRITONSWMMError(Exception):
    """Base exception for all TRITON-SWMM toolkit errors.

    Catch this to handle all toolkit-specific exceptions.
    """

    pass


class StalePlotsError(TRITONSWMMError):
    """A report is being rendered from figures produced by a different toolkit build.

    Equality on the (sha, dirty) tuple, never an ordering: shas are unordered in general
    and the read path often has no object DB. Absent is never equal -- including absent
    on both sides -- so a regression in the capture site fails LOUD.
    """

    pass


class StaleReadModelError(TRITONSWMMError):
    """A rendered figure predates the persisted read-model it transcribes.

    Raised at bundle-emission time, before any staging copy, when the
    Errors-and-Warnings figure is OLDER than validation_report.json. Shipping that
    pair publishes a figure one generation behind its own data.

    Subclasses TRITONSWMMError directly rather than ProcessingError: the latter takes
    (operation, filepath, reason) and this error carries a single composed message.
    """

    pass


class ConfigurationError(TRITONSWMMError):
    """Invalid configuration values or toggle conflicts.

    Raised when:
    - Required fields are missing based on toggle states
    - Mutually exclusive options are both enabled
    - Configuration values fail validation rules

    Attributes
    ----------
    field : str
        The configuration field that failed validation.
    config_path : Path or None
        Path to the configuration file, when one was given.
    """

    def __init__(self, field: str, message: str, config_path: Path | None = None, fix_hint: str = ""):
        self.field = field
        self.config_path = config_path
        self.fix_hint = fix_hint

        lines = [f"Configuration error in field '{field}'"]
        if config_path:
            lines.append(f"  Config: {config_path}")
        lines.append(f"  {message}")
        if fix_hint:
            lines.append(f"  Fix: {fix_hint}")

        super().__init__("\n".join(lines))


class CompilationError(TRITONSWMMError):
    """TRITON/SWMM compilation failure.

    Raised when CMake build or make compilation fails for any model type.

    Attributes
    ----------
    model_type : str
        Which model failed: ``triton``, ``tritonswmm`` or ``swmm``.
    backend : str
        Compilation backend, such as ``cpu``, ``gpu`` or ``openmp``.
    logfile : Path
        Path to the compilation log file carrying the detailed error output.
    return_code : int
        Process return code from the compilation command.
    """

    def __init__(self, model_type: str, backend: str, logfile: Path, return_code: int):
        self.model_type = model_type
        self.backend = backend
        self.logfile = logfile
        self.return_code = return_code

        super().__init__(
            f"{model_type.upper()} {backend.upper()} compilation failed\n"
            f"  Return code: {return_code}\n"
            f"  Log: {logfile}\n"
            f"  Run: cat {logfile}"
        )


class SimulationError(TRITONSWMMError):
    """Simulation execution failure.

    Raised when a TRITON/SWMM simulation process fails during execution.

    Attributes
    ----------
    event_iloc : int
        Index of the weather event that failed.
    model_type : str
        Which model failed: ``triton``, ``tritonswmm`` or ``swmm``.
    logfile : Path or None
        Path to the simulation log file, when one was given.
    """

    def __init__(self, event_iloc: int, model_type: str, logfile: Path | None = None):
        self.event_iloc = event_iloc
        self.model_type = model_type
        self.logfile = logfile

        lines = [f"Simulation failed for event_iloc={event_iloc} (model={model_type})"]
        if logfile:
            lines.append(f"  Log: {logfile}")

        super().__init__("\n".join(lines))


class ProcessingError(TRITONSWMMError):
    """Output processing failure.

    Raised when post-simulation processing operations fail (parsing
    outputs, compressing files, generating summaries).

    Attributes
    ----------
    operation : str
        Description of the operation that failed.
    filepath : Path or None
        Path to the file being processed, when one was given.
    reason : str
        Detailed error reason. Empty when none was given.
    """

    def __init__(self, operation: str, filepath: Path | None = None, reason: str = ""):
        self.operation = operation
        self.filepath = filepath
        self.reason = reason

        lines = [f"Output processing failed: {operation}"]
        if filepath:
            lines.append(f"  File: {filepath}")
        if reason:
            lines.append(f"  Reason: {reason}")

        super().__init__("\n".join(lines))


class WorkflowError(TRITONSWMMError):
    """Snakemake workflow failure.

    Raised when Snakemake workflow execution fails during any phase
    (setup, preparation, execution, processing, consolidation).

    Attributes
    ----------
    phase : str
        Which workflow phase failed.
    return_code : int
        Process return code from Snakemake.
    stderr : str
        Standard-error output from Snakemake. Empty when none was captured.
    """

    def __init__(self, phase: str, return_code: int, stderr: str = ""):
        self.phase = phase
        self.return_code = return_code
        self.stderr = stderr

        lines = [f"Workflow failed during {phase} phase", f"  Return code: {return_code}"]
        if stderr.strip():
            lines.append(f"  Error output:\n{self._indent(stderr)}")

        super().__init__("\n".join(lines))

    @staticmethod
    def _indent(text: str, prefix: str = "    ") -> str:
        """Indent multi-line text for error message formatting."""
        return "\n".join(prefix + line for line in text.split("\n"))


class SLURMError(TRITONSWMMError):
    """SLURM job submission or execution failure.

    Raised when SLURM operations fail (job submission, resource
    allocation, job monitoring).

    Attributes
    ----------
    operation : str
        Which SLURM operation failed: ``submit``, ``monitor`` or ``allocate``.
    job_id : str or None
        SLURM job ID, when one was assigned.
    reason : str
        Detailed error reason. Empty when none was given.
    """

    def __init__(self, operation: str, job_id: str | None = None, reason: str = ""):
        self.operation = operation
        self.job_id = job_id
        self.reason = reason

        lines = [f"SLURM operation failed: {operation}"]
        if job_id:
            lines.append(f"  Job ID: {job_id}")
        if reason:
            lines.append(f"  Reason: {reason}")

        super().__init__("\n".join(lines))


class ResourceAllocationError(TRITONSWMMError):
    """Resource allocation failure for simulations.

    Raised when CPU/GPU/memory resource allocation fails or is
    inconsistent with configuration.

    Attributes
    ----------
    resource_type : str
        Which resource failed: ``cpu``, ``gpu`` or ``memory``.
    requested : str
        Requested resource amount.
    available : str or None
        Available resource amount, when it is known.
    """

    def __init__(self, resource_type: str, requested: str, available: str | None = None):
        self.resource_type = resource_type
        self.requested = requested
        self.available = available

        lines = [f"Resource allocation failed for {resource_type}", f"  Requested: {requested}"]
        if available:
            lines.append(f"  Available: {available}")

        super().__init__("\n".join(lines))


class CLIValidationError(TRITONSWMMError):
    """CLI argument validation failure (exit code 2).

    Raised when command-line arguments fail business logic validation,
    such as mutually exclusive flags, conditional requirements, or
    invalid argument combinations.

    Attributes
    ----------
    argument : str
        The argument or arguments that failed validation.
    fix_hint : str
        Hint for how to fix the issue. Empty when none was given.
    """

    def __init__(self, argument: str, message: str, fix_hint: str = ""):
        self.argument = argument
        self.fix_hint = fix_hint

        lines = [f"Invalid argument: {argument}", f"  {message}"]
        if fix_hint:
            lines.append(f"  Fix: {fix_hint}")

        super().__init__("\n".join(lines))


class GlobusTransferError(TRITONSWMMError):
    """Globus transfer failure.

    Raised when a Globus transfer task fails, is cancelled, or times out.

    Attributes
    ----------
    task_id : str
        Globus task ID.
    status : str
        Terminal task status, such as ``FAILED`` or ``CANCELLED``.
    detail_url : str
        URL to view the task details on app.globus.org, derived from ``task_id``.
    """

    def __init__(
        self,
        task_id: str,
        status: str,
        message: str = "",
    ):
        self.task_id = task_id
        self.status = status
        self.detail_url = f"https://app.globus.org/activity/{task_id}"

        lines = [f"Globus transfer {task_id} ended with status={status}"]
        if message:
            lines.append(f"  {message}")
        lines.append(f"  Details: {self.detail_url}")

        super().__init__("\n".join(lines))


class WorkflowPlanningError(TRITONSWMMError):
    """Workflow planning/build failure (exit code 3).

    Raised when Snakemake workflow generation or DAG planning fails,
    typically due to invalid target specifications or missing dependencies.

    Attributes
    ----------
    phase : str
        The planning phase that failed.
    """

    def __init__(self, phase: str, reason: str):
        self.phase = phase

        super().__init__(f"Workflow planning failed during {phase}\n  Reason: {reason}")


class PublishError(TRITONSWMMError):
    """Raised on a publishing-adapter failure (HydroShare/Zenodo deposit or DOI step)."""

    def __init__(self, target: str, doi: str | None = None, status: str = ""):
        self.target = target
        self.doi = doi
        self.status = status
        super().__init__(f"publish to {target} failed (doi={doi}): {status}")


# Standalone errors that do NOT extend TRITONSWMMError (they predate / stand apart from
# the toolkit hierarchy). BundleSchemaError lives HERE (not in bundle/__init__) so that
# cli_utils.py can import it for exit-code mapping without dragging the eager
# bundle -> _emit/_combine -> xarray/zarr import graph into `hhemt --help`. It is
# re-exported from hhemt.bundle for the existing `from hhemt.bundle import
# BundleSchemaError` call sites.
class BundleSchemaError(ValueError):
    """A bundle's bundle_schema_version does not match the locally-installed
    toolkit's BUNDLE_SCHEMA_VERSION. Distinct from generic malformed-manifest
    errors so callers can branch on schema-version mismatch specifically."""
