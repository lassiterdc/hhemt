"""Top-level package for H&H Ensemble Modeling Toolkit."""

__author__ = """Daniel Lassiter"""
__email__ = "daniel.lassiter@outlook.com"

# Export high-level API
from .toolkit import Toolkit

# Export version_migration subpackage
from hhemt import version_migration  # noqa: F401

# Export custom exceptions for convenient access
from .exceptions import (
    TRITONSWMMError,
    ConfigurationError,
    CompilationError,
    SimulationError,
    ProcessingError,
    WorkflowError,
    SLURMError,
    ResourceAllocationError,
    CLIValidationError,
    WorkflowPlanningError,
)

__all__ = [
    "Toolkit",
    "TRITONSWMMError",
    "ConfigurationError",
    "CompilationError",
    "SimulationError",
    "ProcessingError",
    "WorkflowError",
    "SLURMError",
    "ResourceAllocationError",
    "CLIValidationError",
    "WorkflowPlanningError",
]

# from .hhemt import run_model
# from .hhemt import examples
