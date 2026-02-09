"""CLI utility functions for TRITON-SWMM toolkit.

This module provides helper functions for CLI implementation,
including exception-to-exit-code mapping and argument validation.
"""

from typing import Type
from .exceptions import (
    CLIValidationError,
    ConfigurationError,
    CompilationError,
    SimulationError,
    ProcessingError,
    WorkflowError,
    WorkflowPlanningError,
)


# Exit code mapping per CLI specification
# Exit codes:
#   0: success
#   2: argument/config validation errors
#   3: workflow planning/build errors
#   4: simulation execution failure
#   5: output processing/summarization failure
#  10+: unexpected internal errors

EXIT_CODE_MAP: dict[Type[Exception] | str, int] = {
    "success": 0,
    CLIValidationError: 2,
    ConfigurationError: 2,
    WorkflowPlanningError: 3,
    WorkflowError: 3,
    CompilationError: 3,
    SimulationError: 4,
    ProcessingError: 5,
    Exception: 10,  # Catch-all for unexpected errors
}


def map_exception_to_exit_code(exc: Exception) -> int:
    """Map exception to CLI exit code.

    Args:
        exc: Exception instance to map

    Returns:
        Exit code integer (0-10+)

    Examples:
        >>> map_exception_to_exit_code(CLIValidationError("test", "msg"))
        2
        >>> map_exception_to_exit_code(SimulationError(0, "triton"))
        4
        >>> map_exception_to_exit_code(ValueError("unexpected"))
        10
    """
    for exc_type, code in EXIT_CODE_MAP.items():
        if exc_type == "success" or isinstance(exc_type, str):
            continue
        if isinstance(exc, exc_type):
            return code

    # Default to 10 for unexpected errors
    return EXIT_CODE_MAP[Exception]
