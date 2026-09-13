"""CLI utility functions for TRITON-SWMM toolkit.

This module provides helper functions for CLI implementation: the exception-to-exit-code
mapping (``EXIT_CODE_MAP`` / ``map_exception_to_exit_code``), the stderr label per exception
class (``label_for``), and the ONE ``WorkflowResult`` reporter both workflow verbs call
(``report_workflow_result``).
"""

from pydantic import ValidationError as _PydanticValidationError
from rich.console import Console
from rich.markup import escape

# ADR-19's build-unavailable signal lives in container_build (not exceptions.py) because
# it is a structured branch signal rather than an error taxonomy member. Imported here so
# EXIT_CODE_MAP can give it an explicit code; container_build imports only
# hhemt._filelock_compat + hhemt.exceptions, so this introduces no cycle (verified).
from .container_build import SifBuildUnavailable as _SifBuildUnavailable
from .exceptions import (
    BundleSchemaError,
    CLIValidationError,
    CompilationError,
    ConfigurationError,
    ProcessingError,
    SimulationError,
    WorkflowError,
    WorkflowPlanningError,
)
from .orchestration import WorkflowResult

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

EXIT_CODE_MAP: dict[type[Exception] | str, int] = {
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
    # pydantic.ValidationError (a ValueError) MUST precede the catch-all for the same
    # insertion-order reason. A schema-invalid system.yaml / analysis.yaml surfaces as this
    # class from every in-process model_validate site, and the published tables promise it
    # exit 2 (configuration), not 10. A plain ValueError still resolves to 10.
    _PydanticValidationError: 2,
    Exception: 10,  # Catch-all for unexpected errors
}

# The stderr label printed before an exception's message, keyed by class in the same
# first-isinstance-match order as EXIT_CODE_MAP. `Argument Error` for CLIValidationError is
# pinned by tests/test_cli_02_exit_codes.py; the rest are the labels `hhemt run` has
# always printed. Anything unmapped is an `Unexpected Error`, matching exit 10.
_LABEL_MAP: dict[type[Exception], str] = {
    CLIValidationError: "Argument Error",
    ConfigurationError: "Configuration Error",
    _PydanticValidationError: "Configuration Error",
    WorkflowPlanningError: "Workflow Error",
    WorkflowError: "Workflow Error",
    CompilationError: "Workflow Error",
    SimulationError: "Simulation Error",
    ProcessingError: "Processing Error",
}


def label_for(exc: Exception) -> str:
    """Return the stderr label for ``exc``'s class (``Unexpected Error`` when unmapped)."""
    for exc_type, label in _LABEL_MAP.items():
        if isinstance(exc, exc_type):
            return label
    return "Unexpected Error"


def report_workflow_result(
    result: WorkflowResult,
    *,
    verb: str,
    dry_run: bool,
    console: Console,
    console_err: Console,
) -> int:
    """Print a ``WorkflowResult``'s verdict and detail lines and return the exit code.

    The ONLY site that prints anything derived from a result, on BOTH ``hhemt run`` and
    ``hhemt run-experiment``, so the two verbs cannot diverge. ``result.success`` is tested
    FIRST on every arm including ``--dry-run``: a refused submit, a Snakemake non-zero exit
    or a rule that failed permanently under ``--keep-going`` is a workflow failure
    (``EXIT_CODE_MAP[WorkflowError]``, 3) and never a green line. The producer owns the
    sentence (``_augment_result_with_partial_failures`` rewrites ``message`` when it flips
    ``success``); this function owns the framing, the per-rule token lines and the two
    detail lines (``SLURM Job ID``, ``Execution time``) that a detached operator needs.
    """
    # Producer text (message, rule tokens, reasons) is escaped: Rich reads `[...]` as markup,
    # so an unescaped `[run_x]` token vanishes and an unescaped `[/x]` raises MarkupError
    # inside this reporter, which the verb's catch-all turns into a second MarkupError and
    # click into a silent exit 1 (measured). The framing tags are ours; the payload is not.
    message = escape(result.message)
    if not result.success:
        console_err.print(f"[bold red]{verb} failed:[/bold red] {message}")
        for record in result.partial_failures:
            token = escape(str(record.get("rule_token", "?")))
            reason = escape(str(record.get("reason", "")))
            console_err.print(f"  failed rule: {token} ({reason})")
        return EXIT_CODE_MAP[WorkflowError]
    if dry_run:
        console.print(f"[bold green]{verb} --dry-run planned cleanly;[/bold green] nothing was executed. {message}")
    else:
        console.print(f"[bold green]{verb} complete.[/bold green] {message}")
    if result.job_id:
        console.print(f"[dim]SLURM Job ID: {result.job_id}[/dim]")
    if result.execution_time:
        console.print(f"[dim]Execution time: {result.execution_time:.1f}s[/dim]")
    return EXIT_CODE_MAP["success"]


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
