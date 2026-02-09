"""Shared orchestration layer for TRITON-SWMM workflows.

This module provides high-level orchestration methods that consolidate
parameter translation logic in a single place. The intent-based API simplifies
both CLI and programmatic usage.

Key components:
- WorkflowResult: Structured result object from workflow execution
- Mode translation: User-friendly modes → low-level workflow parameters
- State detection: Infer what needs to run based on log files
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class WorkflowResult:
    """Structured result from workflow execution.

    This replaces the dict-based return from submit_workflow() with a
    typed, structured object that provides better IDE support and clearer
    semantics.

    Attributes
    ----------
    success : bool
        Whether the workflow completed successfully
    mode : str
        Execution mode used: "local" or "slurm"
    execution_time : Optional[float]
        Total execution time in seconds (None for async SLURM jobs)
    phases_completed : List[str]
        Which workflow phases were executed
        Possible values: ["setup", "prepare", "simulate", "process", "consolidate"]
    events_processed : List[int]
        Event ilocs that were processed
    snakefile_path : Path
        Path to the generated Snakefile
    job_id : Optional[str]
        SLURM job ID (only for SLURM mode)
    message : str
        Human-readable status message

    Examples
    --------
    >>> result = analysis.run(mode="fresh")
    >>> if result.success:
    ...     print(f"Processed {len(result.events_processed)} events")
    >>> if result:  # Truthiness check
    ...     print("Success!")
    """

    success: bool
    mode: str
    execution_time: Optional[float] = None
    phases_completed: List[str] = field(default_factory=list)
    events_processed: List[int] = field(default_factory=list)
    snakefile_path: Optional[Path] = None
    job_id: Optional[str] = None
    message: str = ""

    def __bool__(self) -> bool:
        """Allow truthiness check: if result: ..."""
        return self.success

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        parts = [f"Workflow {status} ({self.mode} mode)"]

        if self.phases_completed:
            parts.append(f"Phases: {', '.join(self.phases_completed)}")

        if self.events_processed:
            n_events = len(self.events_processed)
            parts.append(f"Events: {n_events}")

        if self.execution_time:
            parts.append(f"Time: {self.execution_time:.1f}s")

        if self.job_id:
            parts.append(f"Job: {self.job_id}")

        if self.message:
            parts.append(f"Message: {self.message}")

        return "\n".join(parts)


# Mode translation mapping: user-friendly mode → workflow parameters
MODE_TRANSLATION = {
    "fresh": {
        "from_scratch": True,
        "overwrite_system_inputs": True,
        "recompile_if_already_done_successfully": True,
        "overwrite_scenario": True,
        "overwrite_if_exist": True,
        "pickup_where_leftoff": False,
    },
    "resume": {
        "from_scratch": False,
        "overwrite_system_inputs": False,
        "recompile_if_already_done_successfully": False,
        "overwrite_scenario": False,
        "overwrite_if_exist": False,
        "pickup_where_leftoff": True,
    },
    "overwrite": {
        "from_scratch": False,
        "overwrite_system_inputs": True,
        "recompile_if_already_done_successfully": True,
        "overwrite_scenario": True,
        "overwrite_if_exist": True,
        "pickup_where_leftoff": False,
    },
}


def translate_mode(
    mode: Literal["fresh", "resume", "overwrite"]
) -> dict:
    """Translate user-friendly mode to workflow parameters.

    Parameters
    ----------
    mode : Literal["fresh", "resume", "overwrite"]
        User-specified execution mode

    Returns
    -------
    dict
        Dictionary of workflow parameters for submit_workflow()

    Examples
    --------
    >>> params = translate_mode("fresh")
    >>> params["from_scratch"]
    True
    >>> params["pickup_where_leftoff"]
    False
    """
    return MODE_TRANSLATION[mode].copy()


def translate_phases(
    phases: Optional[List[str]] = None,
) -> dict:
    """Translate phase list to workflow boolean flags.

    Parameters
    ----------
    phases : Optional[List[str]]
        Which phases to run. If None, runs all phases.
        Valid phases: ["setup", "prepare", "simulate", "process", "consolidate"]

    Returns
    -------
    dict
        Dictionary of workflow parameters

    Examples
    --------
    >>> params = translate_phases(["setup", "prepare"])
    >>> params["process_system_level_inputs"]
    True
    >>> params["process_timeseries"]
    False
    """
    # If no phases specified, run everything
    if phases is None:
        return {
            "process_system_level_inputs": True,
            "compile_TRITON_SWMM": True,
            "prepare_scenarios": True,
            "process_timeseries": True,
        }

    # Translate phase names to flags
    params = {
        "process_system_level_inputs": "setup" in phases,
        "compile_TRITON_SWMM": "setup" in phases,
        "prepare_scenarios": "prepare" in phases,
        "process_timeseries": "process" in phases,
    }

    # Note: "simulate" phase is always enabled if scenarios are prepared
    # Note: "consolidate" phase is handled automatically by workflow's consolidate rule

    return params
