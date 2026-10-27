"""Shared vocabulary for launching a TRITON-SWMM workflow and reporting on it.

The ``hhemt run-experiment`` CLI, the ``Toolkit`` facade and the experiment-bundle
runner accept the same run modes, and ``TRITONSWMM_analysis.run()`` and the
``Toolkit`` hand back the same result and status objects. ``run()`` itself takes a
``from_scratch`` flag rather than a mode and expands it with ``translate_mode``.
This module defines each shared piece once, so every layer reads one definition
instead of restating its own:

- ``RunMode`` and ``translate_mode``: the closed set of run modes, and the
  low-level ``submit_workflow`` flags each mode expands to.
- ``resolve_execution_locus``: whether a workflow runs locally or under SLURM.
- ``RunOverrides``: the per-invocation overrides, carried through the
  ``submit_workflow`` layers as one argument.
- ``WorkflowResult``: what one ``run()`` call returns.
- ``WorkflowStatus`` and ``PhaseStatus``: what a status check reports.
- ``translate_phases``: the setup/prepare/process flags for a list of phases.

Nothing here reads logs or the filesystem. ``TRITONSWMM_analysis.get_workflow_status``
inspects an analysis and fills a ``WorkflowStatus``.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from hhemt.exceptions import ConfigurationError

_SLURM_RUN_METHODS: tuple[str, ...] = ("1_job_many_srun_tasks", "batch_job")


class RunMode(StrEnum):
    """The closed set of user-facing run modes.

    ONE declaration read by every layer that accepts a ``mode``: the ``hhemt run-experiment``
    ``--mode`` option (Typer renders the members as the choice set and refuses anything else
    at parse time), ``experiment_bundle.run_experiment``, ``Toolkit.run`` (whose
    ``RunMode(mode)`` call is the runtime guard for a notebook caller passing the bare word),
    ``WorkflowStatus.recommended_mode`` and ``translate_mode``. Members are ``str``, so
    ``mode == "fresh"``, dict lookup by the bare string, f-strings and JSON all keep working.

    ``fresh`` deletes the analysis directory and rebuilds everything (guarded by
    ``assert_wipe_is_deliberate``); ``resume`` continues from the last checkpoint. There is
    no third member: re-running completed scenarios without a wipe lives on the
    ``force_rerun`` axis (``override_force_rerun``), never on a mode value.

    Declared at the top of the module rather than beside ``translate_mode`` because
    ``WorkflowStatus`` is a dataclass whose ``recommended_mode`` default is a member, and a
    dataclass default must be bound before the class body runs.
    """

    fresh = "fresh"
    resume = "resume"


def resolve_execution_locus(
    execution_mode: str | None,
    multi_sim_run_method: str | None,
) -> Literal["local", "slurm"]:
    """THE resolver for execution locus. Every site needing a locus calls this.

    An explicit ``execution_mode`` of ``"local"`` or ``"slurm"`` is an OVERRIDE and
    passes through unchanged. That arm carries a ``local``-family config submitted
    to SLURM -- ``multi_sim_run_method="local"`` plus ``execution_mode="slurm"`` -- and is why a
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
    assert. ``in_slurm`` must ALSO not become config-only -- its live readers are
    ``workflow.py``'s ``_generate_submission_script`` assert and tmux module-load
    gate, and ``resource_management.py`` computes the same ``$SLURM_JOB_ID`` test
    independently for allocation-derived SIZING.
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
        Directory paths of the scenarios not yet complete in this phase. Despite the
        name it holds scenarios never started as well as ones that failed;
        ``get_workflow_status`` fills it for the preparation and simulation phases
        only, and ``WorkflowStatus.simulations_failed`` / ``simulations_pending``
        separate the two for simulation.

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
        Number of scenarios the analysis defines (``n_scenarios``). Despite the field
        name it counts scenarios, not model runs: a scenario with two enabled models
        counts once.
    simulations_completed : int
        Number of scenarios in which every enabled model run completed
    simulations_failed : int
        Number of scenarios with at least one unfinished model run that wrote a
        model log -- a run that started and did not finish
    simulations_pending : int
        Number of scenarios whose unfinished model runs wrote no model log -- runs
        not yet started (a run killed before its first log line also counts here)
    current_phase : str
        The first incomplete phase, or ``"complete"`` when every phase is done
    recommended_mode : RunMode
        Recommended execution mode, a ``RunMode`` member (``fresh`` or ``resume``, the
        modes translate_mode accepts). A complete analysis recommends ``fresh``.
    recommendation : str
        Human-readable explanation of recommendation

    Examples
    --------
    >>> status = analysis.get_workflow_status()
    >>> print(status)
    >>> if not status.simulation.complete:
    ...     print(f"{len(status.simulation.failed_items)} scenarios still to run")
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
    recommended_mode: RunMode = RunMode.resume
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
            progress = f" ({phase.progress * 100:.0f}% complete)" if 0 < phase.progress < 1 else ""
            lines.append(f"  {symbol} {phase.name.title()}{progress}")

            for value in phase.details.values():
                lines.append(f"    {value}")

            if phase.failed_items:
                n_show = min(3, len(phase.failed_items))
                lines.append(f"    ✗ {len(phase.failed_items)} incomplete:")
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
    """Typed result of one ``TRITONSWMM_analysis.run()`` call.

    ``run()`` builds it from the dict that ``submit_workflow`` returns. What
    ``success`` certifies depends on the launch. On a ``batch_job`` tmux launch the
    caller does not wait on -- ``run(wait_for_job_completion=False)``, the default
    on a SLURM locus -- and on the detached SIF-build chain, ``success=True``
    records that the launch was made, not that every rule then succeeded, so read
    ``_status/_failed/`` or the produced outputs for the outcome. A waited tmux
    launch reports whether the session's Snakemake exited 0.

    Attributes
    ----------
    success : bool
        Whether the workflow succeeded -- on an unwaited detached launch, whether the
        launch did
    mode : str
        The launch label the submit layer returned (for example ``local``,
        ``slurm``, ``batch_job``, ``tmux`` or ``detached-build-chain``), falling
        back to the resolved locus, ``local`` or ``slurm``, when it returned none
    execution_time : Optional[float]
        Wall-clock seconds from the start of ``run()``'s pre-submit steps (after any
        SIF build prestep) to its return, set only when ``success`` is True and the
        resolved locus is local; None otherwise
    phases_completed : List[str]
        The phases the call requested, listed only when ``success`` is True. Derived
        from the flags passed to ``submit_workflow``, not observed from outputs.
        Possible values: ["setup", "prepare", "simulate", "process", "consolidate"]
    events_processed : List[int]
        Every event iloc of the analysis when ``success`` is True, empty otherwise --
        the events in scope, not a per-event completion record
    snakefile_path : Optional[Path]
        Path to the generated Snakefile, or None when the submit layer returned none
    job_id : Optional[str]
        SLURM job ID, or the process id of the detached SIF-build chain; None when
        there is neither
    message : str
        Human-readable status message
    partial_failures : List[dict]
        Rules that permanently failed under ``--keep-going`` (the sweep let the
        rest of the DAG complete). An entry is either a parsed ``_status/_failed/``
        marker, whose keys depend on its writer -- a simulation runner's marker
        carries ``slurm_jobid``, ``run_uuid``, ``sa_id``, ``model_type``,
        ``event_id``, ``status`` and ``finished_at``, but neither ``rule_token`` nor
        ``reason`` -- or a record carrying ``rule_token`` and ``reason``, for a
        failure seen only in the Snakemake log or an unreadable marker. Non-empty
        implies ``success=False``. Filled only where the call waited on Snakemake:
        always on ``local``, and on ``1_job_many_srun_tasks`` only when ``run()``
        waits (``wait_for_job_completion=True``). A
        ``batch_job`` (tmux) run leaves it empty whether or not it is waited on;
        nothing collects its failures, so the operator reads ``_status/_failed/``
        after the run.
    retried_to_success : List[dict]
        Jobs Snakemake restarted after a failed attempt and then finished, one
        ``{rule, jobid, attempts}`` per job, where ``attempts`` counts every launch
        including the finishing one. A disclosure only: it never affects ``success``.
        Filled on the same paths as ``partial_failures``.

    Examples
    --------
    >>> result = analysis.run(from_scratch=True)
    >>> if result.success:
    ...     print(f"{len(result.events_processed)} events in scope")
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
    retried_to_success: list[dict] = field(default_factory=list)

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

        if self.retried_to_success:
            retried = ", ".join(
                f"{r.get('rule', '?')}[{r.get('jobid', '?')}] ×{r.get('attempts', '?')}"
                for r in self.retried_to_success
            )
            parts.append(f"Retried to success ({len(self.retried_to_success)}): {retried}")

        return "\n".join(parts)


@dataclass(frozen=True)
class RunOverrides:
    """Runtime override knobs for a single workflow invocation.

    Each field follows the override-prefix convention: None means
    read-config-when-None (the resolved value comes from cfg_analysis); a
    concrete value overrides the config for this invocation
    only. Carried as one argument through the submit_workflow facade chain
    instead of five individual keyword params.

    ONE FIELD DEPARTS FROM THAT CONVENTION AND THE EXCEPTION IS DELIBERATE.
    ``live_driver`` bridges no config field: it is a per-invocation operator
    ASSERTION that one specific orchestration driver is dead, consumed once by
    ``_acquire_submit_driver_claim`` and never read from config. It rides this
    carrier anyway because the builder and sensitivity ``submit_workflow`` layers
    accept overrides ONLY through this carrier. Deliberately NOT a sixth
    individual keyword on the sensitivity facade: that would bring back the
    per-layer threading the carrier replaced, and a layer that omitted it would
    hand ``_acquire_submit_driver_claim`` no assertion, so the gate would raise
    ``WorkflowError`` on the very driver the operator had named as dead.
    """

    clear_raw: "ClearRawValue | None" = None
    force_rerun: "ForceRerunValue | None" = None
    hpc_total_nodes: int | None = None
    hpc_restart_times_simulate: int | None = None
    hpc_restart_times_other: int | None = None
    live_driver: str | None = None


def translate_mode(mode: RunMode | Literal["fresh", "resume"]) -> dict:
    """Translate user-friendly mode to workflow parameters.

    Parameters
    ----------
    mode : RunMode or Literal["fresh", "resume"]
        User-specified execution mode. A bare string is coerced through ``RunMode``,
        so a value outside the set raises ``ValueError`` here rather than ``KeyError``.

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
            "overwrite_system_inputs": False,
            "recompile_if_already_done_successfully": False,
            "overwrite_scenario_if_already_set_up": False,
            "rerun_swmm_hydro_if_outputs_exist": False,
            "pickup_where_leftoff": True,
        },
    }
    return MODE_TRANSLATION[RunMode(mode)].copy()


def translate_phases(
    phases: list[str] | None = None,
) -> dict:
    """Translate phase list to workflow boolean flags.

    Parameters
    ----------
    phases : Optional[List[str]]
        Which phases to run. If None, runs all phases.
        Valid phases: ["setup", "prepare", "process"]. ``"simulate"`` and
        ``"consolidate"`` raise ``ValueError``: simulation follows preparation and
        consolidation follows processing, so neither has a flag of its own.

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

    params = {
        "process_system_level_inputs": "setup" in phases,
        "compile_TRITON_SWMM": "setup" in phases,
        "prepare_scenarios": "prepare" in phases,
        "process_timeseries": "process" in phases,
    }

    return params
