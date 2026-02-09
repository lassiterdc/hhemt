"""Custom exception hierarchy for TRITON-SWMM toolkit.

All toolkit-specific exceptions inherit from TRITONSWMMError, allowing
users to catch all toolkit errors with a single except clause while
still providing specific error types for different failure modes.

Each exception stores contextual attributes (file paths, return codes,
model types) to enable programmatic error handling and detailed
error reporting.
"""

from pathlib import Path
from typing import Optional


class TRITONSWMMError(Exception):
    """Base exception for all TRITON-SWMM toolkit errors.

    Catch this to handle all toolkit-specific exceptions.
    """
    pass


class ConfigurationError(TRITONSWMMError):
    """Invalid configuration values or toggle conflicts.

    Raised when:
    - Required fields are missing based on toggle states
    - Mutually exclusive options are both enabled
    - Configuration values fail validation rules

    Attributes:
        field: The configuration field that failed validation
        config_path: Optional path to the configuration file
    """
    def __init__(
        self,
        field: str,
        message: str,
        config_path: Optional[Path] = None
    ):
        self.field = field
        self.config_path = config_path

        lines = [f"Configuration error in field '{field}'"]
        if config_path:
            lines.append(f"  Config: {config_path}")
        lines.append(f"  {message}")

        super().__init__("\n".join(lines))


class CompilationError(TRITONSWMMError):
    """TRITON/SWMM compilation failure.

    Raised when CMake build or make compilation fails for any model type.

    Attributes:
        model_type: Which model failed ('triton', 'tritonswmm', 'swmm')
        backend: Compilation backend ('cpu', 'gpu', 'openmp', etc.)
        logfile: Path to compilation log file with detailed error output
        return_code: Process return code from compilation command
    """
    def __init__(
        self,
        model_type: str,
        backend: str,
        logfile: Path,
        return_code: int
    ):
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

    Attributes:
        event_iloc: Index of the weather event that failed
        model_type: Which model failed ('triton', 'tritonswmm', 'swmm')
        logfile: Optional path to simulation log file
    """
    def __init__(
        self,
        event_iloc: int,
        model_type: str,
        logfile: Optional[Path] = None
    ):
        self.event_iloc = event_iloc
        self.model_type = model_type
        self.logfile = logfile

        lines = [
            f"Simulation failed for event_iloc={event_iloc} (model={model_type})"
        ]
        if logfile:
            lines.append(f"  Log: {logfile}")

        super().__init__("\n".join(lines))


class ProcessingError(TRITONSWMMError):
    """Output processing failure.

    Raised when post-simulation processing operations fail (parsing
    outputs, compressing files, generating summaries).

    Attributes:
        operation: Description of the operation that failed
        filepath: Optional path to the file being processed
        reason: Optional detailed error reason
    """
    def __init__(
        self,
        operation: str,
        filepath: Optional[Path] = None,
        reason: str = ""
    ):
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

    Attributes:
        phase: Which workflow phase failed
        return_code: Process return code from Snakemake
        stderr: Optional stderr output from Snakemake
    """
    def __init__(
        self,
        phase: str,
        return_code: int,
        stderr: str = ""
    ):
        self.phase = phase
        self.return_code = return_code
        self.stderr = stderr

        lines = [
            f"Workflow failed during {phase} phase",
            f"  Return code: {return_code}"
        ]
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

    Attributes:
        operation: Which SLURM operation failed ('submit', 'monitor', 'allocate')
        job_id: Optional SLURM job ID
        reason: Optional detailed error reason
    """
    def __init__(
        self,
        operation: str,
        job_id: Optional[str] = None,
        reason: str = ""
    ):
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

    Attributes:
        resource_type: Which resource failed ('cpu', 'gpu', 'memory')
        requested: Requested resource amount
        available: Available resource amount (if known)
    """
    def __init__(
        self,
        resource_type: str,
        requested: str,
        available: Optional[str] = None
    ):
        self.resource_type = resource_type
        self.requested = requested
        self.available = available

        lines = [
            f"Resource allocation failed for {resource_type}",
            f"  Requested: {requested}"
        ]
        if available:
            lines.append(f"  Available: {available}")

        super().__init__("\n".join(lines))


class CLIValidationError(TRITONSWMMError):
    """CLI argument validation failure (exit code 2).

    Raised when command-line arguments fail business logic validation,
    such as mutually exclusive flags, conditional requirements, or
    invalid argument combinations.

    Attributes:
        argument: The argument(s) that failed validation
        fix_hint: Optional hint for how to fix the issue
    """
    def __init__(
        self,
        argument: str,
        message: str,
        fix_hint: str = ""
    ):
        self.argument = argument
        self.fix_hint = fix_hint

        lines = [f"Invalid argument: {argument}", f"  {message}"]
        if fix_hint:
            lines.append(f"  Fix: {fix_hint}")

        super().__init__("\n".join(lines))


class WorkflowPlanningError(TRITONSWMMError):
    """Workflow planning/build failure (exit code 3).

    Raised when Snakemake workflow generation or DAG planning fails,
    typically due to invalid target specifications or missing dependencies.

    Attributes:
        phase: The planning phase that failed
    """
    def __init__(self, phase: str, reason: str):
        self.phase = phase

        super().__init__(
            f"Workflow planning failed during {phase}\n"
            f"  Reason: {reason}"
        )
