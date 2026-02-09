"""Shared orchestration layer for TRITON-SWMM workflows.

This module provides high-level orchestration methods that consolidate
parameter translation logic in a single place. The intent-based API simplifies
both CLI and programmatic usage.

Key components:
- WorkflowResult: Structured result object from workflow execution
- WorkflowStatus: Status report for workflow completion state
- PhaseStatus: Status of individual workflow phases
- Mode translation: User-friendly modes → low-level workflow parameters
- State detection: Infer what needs to run based on log files
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional


@dataclass
class PhaseStatus:
    """Status of a single workflow phase.

    Attributes
    ----------
    name : str
        Phase name: "setup", "preparation", "simulation", "processing", "consolidation"
    complete : bool
        Whether this phase is fully complete
    progress : float
        Completion progress from 0.0 to 1.0
    details : Dict[str, str]
        Phase-specific status details (key: detail_name, value: formatted string)
    failed_items : List[str]
        Items that failed in this phase (e.g., event ilocs, file paths)

    Examples
    --------
    >>> phase = PhaseStatus(name="simulation", complete=False, progress=0.9)
    >>> phase.symbol()
    '⚠'
    """

    name: str
    complete: bool
    progress: float = 0.0
    details: Dict[str, str] = field(default_factory=dict)
    failed_items: List[str] = field(default_factory=list)

    def symbol(self) -> str:
        """Return status symbol for display.

        Returns
        -------
        str
            '✓' if complete, '⚠' if in progress, '✗' if not started
        """
        if self.complete:
            return "✓"
        elif self.progress > 0:
            return "⚠"
        else:
            return "✗"


@dataclass
class WorkflowStatus:
    """Complete workflow status report.

    Provides comprehensive view of workflow completion state across all phases,
    with recommendations for which execution mode to use.

    Attributes
    ----------
    analysis_id : str
        Unique identifier for this analysis
    analysis_dir : Path
        Root directory for analysis outputs
    setup : PhaseStatus
        Status of setup phase (system inputs, compilation)
    preparation : PhaseStatus
        Status of scenario preparation phase
    simulation : PhaseStatus
        Status of simulation execution phase
    processing : PhaseStatus
        Status of output processing phase
    consolidation : PhaseStatus
        Status of analysis-level consolidation phase
    total_simulations : int
        Total number of simulations configured
    simulations_completed : int
        Number of simulations successfully completed
    simulations_failed : int
        Number of simulations that failed
    simulations_pending : int
        Number of simulations not yet attempted
    current_phase : str
        Which phase is currently incomplete
    recommended_mode : str
        Recommended execution mode: "fresh", "resume", or "overwrite"
    recommendation : str
        Human-readable explanation of recommendation

    Examples
    --------
    >>> status = analysis.get_workflow_status()
    >>> print(status)
    >>> if not status.simulation.complete:
    ...     print(f"Retry {len(status.simulation.failed_items)} failed sims")
    """

    analysis_id: str
    analysis_dir: Path

    setup: PhaseStatus
    preparation: PhaseStatus
    simulation: PhaseStatus
    processing: PhaseStatus
    consolidation: PhaseStatus

    total_simulations: int
    simulations_completed: int
    simulations_failed: int = 0
    simulations_pending: int = 0

    current_phase: str = ""
    recommended_mode: str = "resume"
    recommendation: str = ""

    def __str__(self) -> str:
        """Generate formatted status report.

        Returns
        -------
        str
            Multi-line formatted status report with phase details and recommendations
        """
        lines = [
            "",
            "Workflow Status Report",
            "═" * 66,
            f"Analysis: {self.analysis_id}",
            f"Directory: {self.analysis_dir}",
            "",
            "Phase Status:",
        ]

        for phase in [
            self.setup,
            self.preparation,
            self.simulation,
            self.processing,
            self.consolidation,
        ]:
            symbol = phase.symbol()
            progress = (
                f" ({phase.progress*100:.0f}% complete)"
                if 0 < phase.progress < 1
                else ""
            )
            lines.append(f"  {symbol} {phase.name.title()}{progress}")

            for value in phase.details.values():
                lines.append(f"    {value}")

            if phase.failed_items:
                n_show = min(3, len(phase.failed_items))
                lines.append(f"    ✗ {len(phase.failed_items)} failed:")
                for item in phase.failed_items[:n_show]:
                    lines.append(f"      - {item}")
                if len(phase.failed_items) > n_show:
                    lines.append(
                        f"      ... and {len(phase.failed_items) - n_show} more"
                    )

        lines.extend(
            [
                "",
                "Recommendation:",
                f"  {self.recommendation}",
                "═" * 66,
                "",
            ]
        )

        return "\n".join(lines)


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
