"""Top-level package for TRITON-SWMM Toolkit."""

__author__ = """Daniel Lassiter"""
__email__ = "daniel.lassiter@outlook.com"

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
)

__all__ = [
    "TRITONSWMMError",
    "ConfigurationError",
    "CompilationError",
    "SimulationError",
    "ProcessingError",
    "WorkflowError",
    "SLURMError",
    "ResourceAllocationError",
]

# from .TRITON_SWMM_toolkit import run_model
# from .TRITON_SWMM_toolkit import examples
