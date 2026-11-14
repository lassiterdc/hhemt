"""Top-level package for H&H Ensemble Modeling Toolkit."""

from typing import TYPE_CHECKING

__author__ = """Daniel Lassiter"""
__email__ = "daniel.lassiter@outlook.com"

# Export high-level API
# Export version_migration subpackage
from hhemt import version_migration  # noqa: F401

# Export custom exceptions for convenient access
from .exceptions import (
    CLIValidationError,
    CompilationError,
    ConfigurationError,
    ProcessingError,
    ResourceAllocationError,
    SimulationError,
    SLURMError,
    TRITONSWMMError,
    WorkflowError,
    WorkflowPlanningError,
)

if TYPE_CHECKING:
    # Kept so static loaders (griffe, which the docs gates and the published-surface check use,
    # and type checkers) still resolve ``hhemt.Toolkit``; at runtime it is loaded lazily below.
    from .toolkit import Toolkit

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


def __getattr__(name: str):
    """Load ``Toolkit`` on first access (PEP 562).

    ``hhemt.toolkit`` pulls in the whole system and analysis stack (about 1.9 s), which every
    ``import hhemt.<submodule>`` -- including each Snakemake runner job -- used to pay at
    package import. Resolving it on demand keeps ``from hhemt import Toolkit`` and
    ``hhemt.Toolkit`` working while a submodule import no longer imports ``hhemt.toolkit``.
    """
    if name == "Toolkit":
        from .toolkit import Toolkit

        globals()["Toolkit"] = Toolkit
        return Toolkit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), "Toolkit"})
