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
from typing import TYPE_CHECKING, Literal

from hhemt.exceptions import ConfigurationError

_SLURM_RUN_METHODS: tuple[str, ...] = ("1_job_many_srun_tasks", "batch_job")


def resolve_execution_locus(
    execution_mode: str | None,
    multi_sim_run_method: str | None,
) -> Literal["local", "slurm"]:
    """THE resolver for execution locus. Every site needing a locus calls this.

    An explicit ``execution_mode`` of ``"local"`` or ``"slurm"`` is an OVERRIDE and
    passes through unchanged. That arm carries the ``[Q8]`` shape --
    ``multi_sim_run_method="local"`` plus ``execution_mode="slurm"`` -- and is why a
    caller handed an already-resolved mode needs no second term, unlike
    ``workflow.py``'s report-tail predicate, which must reconstruct the override from
    ``_resolved_execution_locus`` because by that point it is no longer readable off
    the config. ``"auto"`` (or ``None``) falls back to the CONFIG family:
    ``None``/``"local"`` -> ``"local"``; ``"batch_job"``/``"1_job_many_srun_tasks"``
    -> ``"slurm"``. Any other value of either argument raises rather than defaulting.

    Do NOT reintroduce a read of ``analysis.in_slurm`` here or at any caller.
    Promoting a ``local``-family analysis to a slurm workflow because
    ``$SLURM_JOB_ID`` happened to be set is the defect this rule exists to prevent
    (``8249167f``): a synth fixture running inside an array element reached
    ``generate_snakemake_config(mode="slurm")`` and tripped its max_concurrent_jobs
    assert. ``in_slurm`` must ALSO not become config-only -- its live consumers are
    ``resource_management.py`` (allocation-derived SIZING) and ``workflow.py``'s
    ``_generate_submission_script`` assert and tmux module-load gate.
    ``run_simulation.py``'s ``using_srun`` no longer reads it: that predicate CONSUMES
    this function's result through a threaded ``--execution-locus`` and adds a
    fail-safe family term, so it is a consumer, not another resolver.

    A comment is not a gate. ``690cd765`` cited a PREVIOUS form of this text as
    authority for making ``using_srun`` config-only, which stripped srun from the
    local-family + ``execution_mode="slurm"`` path entirely. Verify against the
    predicate, never against this docstring.
    """
    if execution_mode in ("local", "slurm"):
        return execution_mode  # type: ignore[return-value]
    if execution_mode not in (None, "auto"):
        raise ConfigurationError(
            field="execution_mode",
            message=f"Unrecognized execution_mode={execution_mode!r}; expected 'auto', 'local' or 'slurm'.",
        )
    if multi_sim_run_method is None or multi_sim_run_method == "local":
        return "local"
    if multi_sim_run_method in _SLURM_RUN_METHODS:
        return "slurm"
    raise ConfigurationError(
        field="multi_sim_run_method",
        message=f"Unrecognized multi_sim_run_method={multi_sim_run_method!r} for execution-locus resolution",
    )


if TYPE_CHECKING:
    from hhemt.config.analysis import ClearRawValue, ForceRerunValue


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
    details: dict[str, str] = field(default_factory=dict)
    failed_items: list[str] = field(default_factory=list)

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
        Recommended execution mode: "fresh" or "resume" (the modes translate_mode
        accepts). A complete analysis recommends "fresh" (redo from scratch).
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
            progress = f" ({phase.progress*100:.0f}% complete)" if 0 < phase.progress < 1 else ""
            lines.append(f"  {symbol} {phase.name.title()}{progress}")

            for value in phase.details.values():
                lines.append(f"    {value}")

            if phase.failed_items:
                n_show = min(3, len(phase.failed_items))
                lines.append(f"    ✗ {len(phase.failed_items)} failed:")
                for item in phase.failed_items[:n_show]:
                    lines.append(f"      - {item}")
                if len(phase.failed_items) > n_show:
                    lines.append(f"      ... and {len(phase.failed_items) - n_show} more")

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
    partial_failures : List[dict]
        Rules that permanently failed under ``--keep-going`` (the sweep let the
        rest of the DAG complete). Each entry carries at least ``rule_token`` and
        ``reason``. Non-empty implies ``success=False``. Populated only on the
        blocking submit paths (``local`` / ``1_job_many_srun_tasks``); a detached
        ``batch_job`` (tmux) run has no in-process completion point and leaves this
        empty (operator inspects ``_status/_failed/`` post-hoc — captured follow-up).

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
    execution_time: float | None = None
    phases_completed: list[str] = field(default_factory=list)
    events_processed: list[int] = field(default_factory=list)
    snakefile_path: Path | None = None
    job_id: str | None = None
    message: str = ""
    partial_failures: list[dict] = field(default_factory=list)

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

        if self.partial_failures:
            tokens = ", ".join(r.get("rule_token", "?") for r in self.partial_failures)
            parts.append(f"Partial failures ({len(self.partial_failures)}): {tokens}")

        return "\n".join(parts)


@dataclass(frozen=True)
class RunOverrides:
    """Runtime override knobs for a single workflow invocation.

    Each field follows the override-prefix convention: None means
    read-config-when-None (the resolved value comes from cfg_analysis /
    cfg_system); a concrete value overrides the config for this invocation
    only. Carried as one argument through the submit_workflow facade chain
    instead of five individual keyword params.

    ONE FIELD DEPARTS FROM THAT CONVENTION AND THE EXCEPTION IS DELIBERATE.
    ``live_driver`` bridges no config field: it is a per-invocation operator
    ASSERTION that one specific orchestration driver is dead, consumed once by
    ``_acquire_submit_driver_claim`` and never read from config. It rides this
    carrier anyway because D4 made the builder and sensitivity layers
    overrides-only, and adding a sixth individual keyword to the sensitivity
    facade would re-open the shape D4 closed. Carrying the exception in writing
    is preferred to a structural regression that leaves no trace.
    """

    clear_raw: "ClearRawValue | None" = None
    force_rerun: "ForceRerunValue | None" = None
    hpc_total_nodes: int | None = None
    hpc_restart_times_simulate: int | None = None
    hpc_restart_times_other: int | None = None
    live_driver: str | None = None


def translate_mode(mode: Literal["fresh", "resume"]) -> dict:
    """Translate user-friendly mode to workflow parameters.

    Parameters
    ----------
    mode : Literal["fresh", "resume"]
        User-specified execution mode

    Returns
    -------
    dict
        Dictionary of workflow parameters for submit_workflow()

    Examples
    --------
    >>> params = translate_mode("fresh")
    >>> params["overwrite_system_inputs"]
    True
    >>> params["pickup_where_leftoff"]
    False
    """
    MODE_TRANSLATION = {
        "fresh": {
            # "from_scratch" MUST stay commented: the wipe is owned by run(), and
            # submit_workflow() has no from_scratch param (uncommenting -> TypeError).
            # "from_scratch": True,
            "overwrite_system_inputs": True,
            "recompile_if_already_done_successfully": False,
            "overwrite_scenario_if_already_set_up": True,
            "rerun_swmm_hydro_if_outputs_exist": True,
            "pickup_where_leftoff": False,
        },
        "resume": {
            # "from_scratch": False,
            "overwrite_system_inputs": False,
            "recompile_if_already_done_successfully": False,
            "overwrite_scenario_if_already_set_up": False,
            "rerun_swmm_hydro_if_outputs_exist": False,
            "pickup_where_leftoff": True,
        },
    }
    return MODE_TRANSLATION[mode].copy()


def translate_phases(
    phases: list[str] | None = None,
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
    valid_phases = ["setup", "prepare", "process"]
    for phase in phases:
        if phase not in valid_phases:
            raise ValueError(f"Invalid phase specified. Must be one of: {valid_phases}")

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
