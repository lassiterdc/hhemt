"""CLI utility functions for TRITON-SWMM toolkit.

This module provides helper functions for CLI implementation,
including exception-to-exit-code mapping and argument validation.
"""

from typing import Type
from .exceptions import (
    BundleSchemaError,
    CLIValidationError,
    ConfigurationError,
    CompilationError,
    SimulationError,
    ProcessingError,
    WorkflowError,
    WorkflowPlanningError,
)

# ADR-19's build-unavailable signal lives in container_build (not exceptions.py) because
# it is a structured branch signal rather than an error taxonomy member. Imported here so
# EXIT_CODE_MAP can give it an explicit code; container_build imports only
# hhemt._filelock_compat + hhemt.exceptions, so this introduces no cycle (verified).
from .container_build import SifBuildUnavailable as _SifBuildUnavailable


# Exit code mapping per CLI specification
# Exit codes:
#   0: success
#   2: argument/config validation errors
#   3: workflow planning/build errors
#   4: simulation execution failure
#   5: output processing/summarization failure
#   6: bundle schema-version mismatch
#   7: SIF build unavailable on this host (ADR-19 rootless-fakeroot preflight FAIL)
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
    # BundleSchemaError (a ValueError, not a TRITONSWMMError) MUST precede the
    # Exception catch-all: map_exception_to_exit_code returns the first isinstance
    # match in insertion order, so a post-catch-all entry would be dead code and the
    # schema mismatch would resolve to 10 (Gotcha 27).
    BundleSchemaError: 6,
    # SifBuildUnavailable (a plain Exception, not a TRITONSWMMError) MUST precede the
    # catch-all for the same insertion-order reason as BundleSchemaError. It is a
    # STRUCTURED SIGNAL, not a bug: the documented ADR-19 build-vs-transfer branch.
    # Under exit 10 an operator scripting [Q8] cannot tell "this host cannot build
    # rootlessly, use the ADR-2 transfer" from "unexpected internal error" — and the
    # dedicated except-block in cli.py's build-sif command is exit-code-INERT without
    # this entry (Gotcha 27: add an explicit EXIT_CODE_MAP entry rather than relying on
    # a call-site catch).
    _SifBuildUnavailable: 7,
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
