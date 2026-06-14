"""Command-line interface for TRITON-SWMM Toolkit.

Provides a Snakemake-first single-command CLI for running TRITON-SWMM
workflows with support for production, testcase, and case-study profiles.
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .exceptions import (
    CLIValidationError,
    CompilationError,
    ConfigurationError,
    ProcessingError,
    SimulationError,
    WorkflowError,
    WorkflowPlanningError,
)
from .profile_catalog import (
    list_case_studies,
    list_testcases,
    load_profile_catalog,
)


def _parse_override_clear_raw(value: str | None) -> str | list | None:
    """Parse the ``--override-clear-raw`` CLI flag value.

    Accepts ``"all"``, ``"none"``, or a JSON list (e.g. ``'["tritonswmm","swmm"]'``).
    Phase 3 plants this helper; Phase 4 reuses the same shape for
    ``--override-force-rerun``.
    """
    if value is None:
        return None
    if value in ("all", "none"):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"--override-clear-raw expects 'all', 'none', or a JSON list "
            f'like \'["tritonswmm","swmm"]\'; got: {value!r} ({exc})'
        )


def _parse_override_force_rerun(value: str | None) -> str | dict | None:
    """Parse the ``--override-force-rerun`` CLI flag value.

    Accepts ``"all"``, ``"none"``, or a JSON dict with one of
    ``"sa_id"`` / ``"event_iloc"`` keys mapping to a list of values
    (e.g. ``'{"sa_id":[0,5,22]}'`` for sensitivity, or
    ``'{"event_iloc":[3,7]}'`` for non-sensitivity).

    Per cleanup-rerun-delete-redesign Phase 4.
    """
    if value is None:
        return None
    if value in ("all", "none"):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"--override-force-rerun expects 'all', 'none', or a JSON dict "
            f'like \'{{"sa_id":[0,5]}}\'; got: {value!r} ({exc})'
        )


app = typer.Typer(
    name="TRITON-SWMM",
    help="TRITON-SWMM Toolkit: Coupled hydrodynamic-stormwater simulation orchestration",
    no_args_is_help=True,
)
console = Console()
console_err = Console(stderr=True)


@app.command(name="run")
def run_command(
    # ═══════════════════════════════════════════════════════════════
    # Required Arguments (unless using list actions)
    # ═══════════════════════════════════════════════════════════════
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Execution profile: production, testcase, or case-study",
    ),
    system_config: Path | None = typer.Option(
        None,
        "--system-config",
        help="Path to system configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    analysis_config: Path | None = typer.Option(
        None,
        "--analysis-config",
        help="Path to analysis configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    # ═══════════════════════════════════════════════════════════════
    # Execution Control
    # ═══════════════════════════════════════════════════════════════
    from_scratch: bool = typer.Option(
        False,
        "--from-scratch",
        help="Clear run artifacts and execute from fresh state",
    ),
    override_clear_raw: str = typer.Option(
        None,
        "--override-clear-raw",
        help=(
            'Runtime override for cfg_analysis.clear_raw. Accepts "all", "none", '
            'or a JSON list of model types: \'["tritonswmm","swmm"]\'. '
            "When omitted, reads cfg_analysis.clear_raw from the YAML."
        ),
        callback=lambda value: _parse_override_clear_raw(value),
    ),
    override_force_rerun: str = typer.Option(
        None,
        "--override-force-rerun",
        help=(
            'Runtime override for cfg_analysis.force_rerun. Accepts "all", "none", '
            'or a JSON dict: \'{"sa_id":[0,5,22]}\' (sensitivity) or '
            '\'{"event_iloc":[3,7]}\' (non-sensitivity).'
        ),
        callback=lambda value: _parse_override_force_rerun(value),
    ),
    resume: bool = typer.Option(
        True,
        "--resume",
        help="Continue from completed state (default behavior)",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Recreate outputs even if completion logs indicate success",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate configs and show intended workflow without execution",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Model/Processing Scope
    # ═══════════════════════════════════════════════════════════════
    model: str = typer.Option(
        "auto",
        "--model",
        help="Model selection: auto (use config toggles), triton, swmm, tritonswmm",
    ),
    which: str = typer.Option(
        "both",
        "--which",
        help="Processing scope: TRITON, SWMM, or both",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Scenario/Event Selection
    # ═══════════════════════════════════════════════════════════════
    event_ilocs: str | None = typer.Option(
        None,
        "--event-ilocs",
        help="Comma-separated event indices (e.g., '0,1,2,10')",
    ),
    event_range: str | None = typer.Option(
        None,
        "--event-range",
        help="Event range START:END (e.g., '0:100', inclusive start, exclusive end)",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Profile-Specific Options
    # ═══════════════════════════════════════════════════════════════
    testcase: str | None = typer.Option(
        None,
        "--testcase",
        help="Testcase name (required when --profile testcase)",
    ),
    case_study: str | None = typer.Option(
        None,
        "--case-study",
        help="Case study name (required when --profile case-study)",
    ),
    tests_case_config: Path | None = typer.Option(
        None,
        "--tests-case-config",
        help="Path to tests_and_case_studies.yaml profile catalog",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    list_testcases_flag: bool = typer.Option(
        False,
        "--list-testcases",
        help="Print available testcases and exit",
    ),
    list_case_studies_flag: bool = typer.Option(
        False,
        "--list-case-studies",
        help="Print available case studies and exit",
    ),
    status_flag: bool = typer.Option(
        False,
        "--status",
        help="Show workflow status report and exit (no execution)",
    ),
    # ═══════════════════════════════════════════════════════════════
    # HPC Override Options
    # ═══════════════════════════════════════════════════════════════
    platform_config: str | None = typer.Option(None, "--platform-config", help="Platform configuration name"),
    partition: str | None = typer.Option(None, "--partition", help="SLURM partition override"),
    account: str | None = typer.Option(None, "--account", help="SLURM account override"),
    qos: str | None = typer.Option(None, "--qos", help="SLURM QoS override"),
    nodes: int | None = typer.Option(None, "--nodes", help="Number of nodes", min=1),
    ntasks_per_node: int | None = typer.Option(None, "--ntasks-per-node", help="Tasks per node", min=1),
    cpus_per_task: int | None = typer.Option(None, "--cpus-per-task", help="CPUs per task", min=1),
    gpus_per_node: int | None = typer.Option(None, "--gpus-per-node", help="GPUs per node", min=0),
    walltime: str | None = typer.Option(None, "--walltime", help="Walltime limit (HH:MM:SS format)"),
    # ═══════════════════════════════════════════════════════════════
    # Workflow Engine Options
    # ═══════════════════════════════════════════════════════════════
    jobs: int | None = typer.Option(
        None,
        "--jobs",
        "-j",
        help="Parallel jobs for workflow execution",
        min=1,
    ),
    workflow_target: str | None = typer.Option(
        None,
        "--workflow-target",
        help="Explicit Snakemake target/rule group (advanced)",
    ),
    snakemake_args: list[str] | None = typer.Option(
        None,
        "--snakemake-arg",
        help="Pass-through Snakemake flag (repeatable)",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Tool Provisioning
    # ═══════════════════════════════════════════════════════════════
    redownload: str = typer.Option(
        "none",
        "--redownload",
        help="Bootstrap tool binaries: none, triton, swmm, all",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Logging & UX
    # ═══════════════════════════════════════════════════════════════
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-error output",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Python logging level: DEBUG, INFO, WARNING, ERROR",
    ),
):
    """Run TRITON-SWMM workflow with specified profile and configuration.

    This command orchestrates the full TRITON-SWMM workflow including system
    setup, scenario preparation, simulation execution, and output processing.

    Examples:

        # Production run with default resume behavior
        $ triton-swmm run --profile production \\
            --system-config system.yaml --analysis-config analysis.yaml

        # Fresh run with selected events
        $ triton-swmm run --profile production \\
            --system-config system.yaml --analysis-config analysis.yaml \\
            --from-scratch --event-ilocs 0,1,2

        # Testcase with HPC overrides
        $ triton-swmm run --profile testcase --testcase norfolk_smoke \\
            --system-config system.yaml --analysis-config analysis.yaml \\
            --partition debug --walltime 00:20:00

    """
    try:
        # ═══════════════════════════════════════════════════════════════
        # Stage 1: Action Flags (Early Exit)
        # ═══════════════════════════════════════════════════════════════
        if list_testcases_flag:
            _handle_list_testcases(tests_case_config)
            raise typer.Exit(0)

        if list_case_studies_flag:
            _handle_list_case_studies(tests_case_config)
            raise typer.Exit(0)

        # ═══════════════════════════════════════════════════════════════
        # Stage 2: Required Argument Check (for non-list actions)
        # ═══════════════════════════════════════════════════════════════
        if not profile:
            raise CLIValidationError(
                argument="--profile",
                message="--profile is required",
                fix_hint="Specify production, testcase, or case-study",
            )
        if not system_config:
            raise CLIValidationError(
                argument="--system-config",
                message="--system-config is required",
                fix_hint="Provide path to system configuration YAML file",
            )
        if not analysis_config:
            raise CLIValidationError(
                argument="--analysis-config",
                message="--analysis-config is required",
                fix_hint="Provide path to analysis configuration YAML file",
            )

        # ═══════════════════════════════════════════════════════════════
        # Stage 3: Argument Validation
        # ═══════════════════════════════════════════════════════════════
        _validate_cli_arguments(
            profile=profile,
            from_scratch=from_scratch,
            resume=resume,
            verbose=verbose,
            quiet=quiet,
            event_ilocs=event_ilocs,
            event_range=event_range,
            testcase=testcase,
            case_study=case_study,
            model=model,
            which=which,
            redownload=redownload,
            log_level=log_level,
            walltime=walltime,
        )

        # ═══════════════════════════════════════════════════════════════
        # Stage 4: Profile Resolution (if applicable)
        # ═══════════════════════════════════════════════════════════════
        # Note: Profile resolution for testcase/case-study profiles is
        # deferred to future phases. For now, production profiles use
        # config files directly without merging profile catalog entries.

        if profile in ["testcase", "case-study"]:
            # Future: Load catalog, resolve profile entry, merge with configs
            raise CLIValidationError(
                argument="--profile",
                message=f"Profile type '{profile}' not yet implemented",
                fix_hint="Use --profile production for now",
            )

        # ═══════════════════════════════════════════════════════════════
        # Stage 5: Config Loading & System/Analysis Instantiation
        # ═══════════════════════════════════════════════════════════════
        if not quiet:
            console.print("[cyan]Loading configurations...[/cyan]")

        from .analysis import TRITONSWMM_analysis
        from .system import TRITONSWMM_system

        system = TRITONSWMM_system(system_config)
        analysis = TRITONSWMM_analysis(analysis_config, system)
        system._analysis = analysis  # Link back

        if not quiet:
            console.print("[green]✓[/green] Configurations loaded")

        # ═══════════════════════════════════════════════════════════════
        # Stage 6: Preflight Validation
        # ═══════════════════════════════════════════════════════════════
        if not quiet:
            console.print("[cyan]Running preflight validation...[/cyan]")

        validation_result = analysis.validate()

        if validation_result.has_warnings:
            for warning in validation_result.warnings:
                console.print(f"[yellow]Warning:[/yellow] {warning.message}")

        if not validation_result.is_valid:
            console_err.print("[bold red]Validation failed:[/bold red]")
            for error in validation_result.errors:
                console_err.print(f"  • {error.message}")
            raise typer.Exit(2)

        if not quiet:
            console.print("[green]✓[/green] Validation passed")

        # ═══════════════════════════════════════════════════════════════
        # Stage 7a: Status Report (if requested)
        # ═══════════════════════════════════════════════════════════════
        if status_flag:
            status = analysis.get_workflow_status()
            console.print(status)
            raise typer.Exit(0)

        # ═══════════════════════════════════════════════════════════════
        # Stage 7b: Dry-Run Output (if requested)
        # ═══════════════════════════════════════════════════════════════
        if dry_run:
            _print_dry_run_summary(locals())
            console.print("\n[yellow]Dry-run mode: Snakemake DAG will be validated but not executed[/yellow]")
            # Continue to Stage 8 with dry_run=True

        # ═══════════════════════════════════════════════════════════════
        # Stage 8: Workflow Orchestration
        # ═══════════════════════════════════════════════════════════════

        # Determine execution mode from CLI flags
        # Map CLI flags to orchestration layer modes
        if from_scratch:
            run_mode = "fresh"
        elif overwrite:
            run_mode = "overwrite"
        else:
            run_mode = "resume"

        # Determine execution context (local vs SLURM)
        if analysis.in_slurm or analysis.cfg_analysis.multi_sim_run_method == "1_job_many_srun_tasks":
            execution_mode = "slurm"
        else:
            execution_mode = "local"

        if not quiet:
            console.print(f"\n[cyan]Submitting workflow in {execution_mode} mode ({run_mode})...[/cyan]")

        # Execute workflow via high-level orchestration API
        result = analysis.run(
            mode=run_mode,
            phases=None,  # Run all phases
            events=None,  # Process all events
            execution_mode=execution_mode,
            dry_run=dry_run,
            verbose=verbose,
            override_clear_raw=override_clear_raw,
            override_force_rerun=override_force_rerun,
        )

        # Check workflow result
        if not result.success:
            console_err.print(f"[bold red]Workflow Error:[/bold red] {result.message}")
            raise typer.Exit(3)

        if dry_run:
            console.print("\n[bold green]✓ Dry-run validation complete![/bold green]")
            console.print("[dim]No simulations were executed.[/dim]")
        else:
            console.print("[bold green]✓ Workflow complete![/bold green]")
            if execution_mode == "slurm" and result.job_id:
                console.print(f"[dim]SLURM Job ID: {result.job_id}[/dim]")
            if result.execution_time:
                console.print(f"[dim]Execution time: {result.execution_time:.1f}s[/dim]")

        raise typer.Exit(0)

    except typer.Exit:
        # Re-raise Typer exits (clean exits with specific codes)
        raise

    except CLIValidationError as e:
        console_err.print(f"[bold red]Argument Error:[/bold red] {e}")
        raise typer.Exit(2)

    except ConfigurationError as e:
        console_err.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(2)

    except (CompilationError, WorkflowError, WorkflowPlanningError) as e:
        console_err.print(f"[bold red]Workflow Error:[/bold red] {e}")
        raise typer.Exit(3)

    except SimulationError as e:
        console_err.print(f"[bold red]Simulation Error:[/bold red] {e}")
        raise typer.Exit(4)

    except ProcessingError as e:
        console_err.print(f"[bold red]Processing Error:[/bold red] {e}")
        raise typer.Exit(5)

    except Exception as e:
        console_err.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        if verbose:
            import traceback

            console_err.print(traceback.format_exc())
        raise typer.Exit(10)


@app.command(name="cleanup-orphans")
def cleanup_orphans_command(
    system_config: Path = typer.Option(
        ...,
        "--system-config",
        help="Path to system configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    analysis_config: Path = typer.Option(
        ...,
        "--analysis-config",
        help="Path to analysis configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--apply",
        help="List orphan directories without deleting (default) or delete them with --apply",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Required with --apply to actually delete",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print each orphan path and deletion",
    ),
):
    """List or delete sub-analysis directories orphaned by sensitivity CSV edits.

    When sensitivity sub-analyses are removed from the CSV and the workflow is
    re-run, their output directories remain on disk. This command identifies
    those orphans and optionally removes them.

    Examples:

        # List orphans (no deletion)
        $ triton-swmm cleanup-orphans --system-config system.yaml \\
            --analysis-config analysis.yaml

        # Actually delete
        $ triton-swmm cleanup-orphans --system-config system.yaml \\
            --analysis-config analysis.yaml --apply --force
    """
    try:
        from .analysis import TRITONSWMM_analysis
        from .system import TRITONSWMM_system

        system = TRITONSWMM_system(system_config)
        analysis = TRITONSWMM_analysis(analysis_config, system)
        system._analysis = analysis

        if not analysis.cfg_analysis.toggle_sensitivity_analysis:
            console_err.print(
                "[bold red]Error:[/bold red] cleanup-orphans requires a sensitivity analysis "
                "(toggle_sensitivity_analysis=True in analysis config)."
            )
            raise typer.Exit(2)

        if not dry_run and not force:
            console_err.print("[bold red]Error:[/bold red] --apply requires --force to confirm deletion.")
            raise typer.Exit(2)

        result = analysis.sensitivity.cleanup_all_orphans(
            dry_run=dry_run,
            force=force,
            verbose=verbose,
        )
        n_dirs = len(result["dirs"])
        n_flags = len(result["status_flags"])
        n_groups = len(result["datatree_groups"])
        total = n_dirs + n_flags + n_groups

        if total == 0:
            console.print("[green]No orphan sub-analysis artifacts found.[/green]")
        elif dry_run:
            console.print(
                f"[yellow]Found orphans (dry-run; nothing deleted): "
                f"{n_dirs} dir(s), {n_flags} status flag(s), {n_groups} datatree group(s).[/yellow]"
            )
            for p in result["dirs"]:
                console.print(f"  dir: {p}")
            for p in result["status_flags"]:
                console.print(f"  flag: {p}")
            for sa_id in result["datatree_groups"]:
                console.print(f"  datatree-group: sa_{sa_id}")
        else:
            zarr_removed = result.get("sensitivity_datatree_removed", False)
            master_flag_removed = result.get("master_flag_removed", False)
            extras = []
            if zarr_removed:
                extras.append("sensitivity_datatree.zarr")
            if master_flag_removed:
                extras.append("f_consolidate_master_complete.flag")
            extras_msg = f" plus {' and '.join(extras)}" if extras else ""
            console.print(
                f"[green]Deleted {n_dirs} orphan dir(s), {n_flags} status flag(s), "
                f"and {n_groups} datatree group(s){extras_msg}.[/green]"
            )

        raise typer.Exit(0)

    except typer.Exit:
        raise
    except ConfigurationError as e:
        console_err.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(2)
    except Exception as e:
        console_err.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(10)


@app.command(name="reprocess")
def reprocess_command(
    system_config: Path = typer.Option(
        ...,
        "--system-config",
        help="Path to system configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    analysis_config: Path = typer.Option(
        ...,
        "--analysis-config",
        help="Path to analysis configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    start_with: str = typer.Option(
        "consolidate",
        "--start-with",
        help="Stage to re-fire from: process | consolidate | render",
    ),
    execution_mode: str = typer.Option(
        "auto",
        "--execution-mode",
        help="Execution mode: auto | local | slurm",
    ),
    which: str = typer.Option(
        "both",
        "--which",
        help="Processing scope: TRITON | SWMM | both",
    ),
    regenerate_existing: bool = typer.Option(
        False,
        "--regenerate-existing",
        help=(
            "Opt in to deleting and rebuilding already-completed content "
            "(consolidated zarr; at start_with='process', per-scenario "
            "processed/ dirs). Default: preserve completed content, "
            "regenerate report+plots only."
        ),
    ),
    # Phase 3: R8 SLURM-offload toggle for the opt-in deletion.
    delete_via_slurm: bool | None = typer.Option(
        None, "--delete-via-slurm/--no-delete-via-slurm",
        help=("Offload the opt-in (--regenerate-existing) consolidated-zarr + "
              "processed/ deletion to SLURM via the analysis.delete() architecture. "
              "Default (unset): auto — offload when the analysis runs on an HPC "
              "multi_sim_run_method, in-process fast_rmtree on local."),
    ),
    override_clear_raw: str = typer.Option(
        None,
        "--override-clear-raw",
        help=(
            'Runtime override for cfg_analysis.clear_raw. Accepts "all", "none", '
            'or a JSON list of model types: \'["tritonswmm","swmm"]\'. When '
            'omitted, reprocess defaults to "none" (preserves historic semantics: '
            "reprocess never auto-clears raw outputs). When the resolved value "
            'would clear, two guards must pass: every sim\'s c_run_*.flag must '
            "exist and no in-flight _status/_submitted/ sentinel may be present."
        ),
        callback=lambda value: _parse_override_clear_raw(value),
    ),
    override_force_rerun: str = typer.Option(
        None,
        "--override-force-rerun",
        help=(
            'Runtime override for cfg_analysis.force_rerun. Accepts "all", "none", '
            'or a JSON dict: \'{"sa_id":[0,5,22]}\' (sensitivity) or '
            '\'{"event_iloc":[3,7]}\' (non-sensitivity).'
        ),
        callback=lambda value: _parse_override_force_rerun(value),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run snakemake --dry-run only; no execution",
    ),
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        help="Print progress messages",
    ),
):
    """Re-run downstream stages (process / consolidate / render) against existing
    simulation outputs without re-running sims.

    Builds a scope-limited Snakefile at ``{analysis_dir}/Snakefile.reprocess``
    and runs it against a separate ``.snakemake_reprocess/`` working dir so
    the reprocess driver can coexist with a live simulation driver. Runs
    the Phase-1 reconciliation guard before submission.

    Examples:

        # Re-aggregate datatree + render (common case)
        $ triton-swmm reprocess --system-config system.yaml \\
            --analysis-config analysis.yaml --start-with consolidate

        # Re-render report only against existing plots
        $ triton-swmm reprocess --system-config system.yaml \\
            --analysis-config analysis.yaml --start-with render
    """
    try:
        from .analysis import TRITONSWMM_analysis
        from .system import TRITONSWMM_system

        if start_with not in ("process", "consolidate", "render"):
            console_err.print(
                f"[bold red]Error:[/bold red] --start-with must be one of "
                f"'process', 'consolidate', 'render'; got {start_with!r}."
            )
            raise typer.Exit(2)
        if execution_mode not in ("auto", "local", "slurm"):
            console_err.print(
                f"[bold red]Error:[/bold red] --execution-mode must be one of "
                f"'auto', 'local', 'slurm'; got {execution_mode!r}."
            )
            raise typer.Exit(2)

        system = TRITONSWMM_system(system_config)
        analysis = TRITONSWMM_analysis(analysis_config, system)
        system._analysis = analysis

        result = analysis.reprocess(
            start_with=start_with,  # type: ignore[arg-type]
            execution_mode=execution_mode,  # type: ignore[arg-type]
            which=which,  # type: ignore[arg-type]
            regenerate_existing=regenerate_existing,
            delete_via_slurm=delete_via_slurm,
            override_clear_raw=override_clear_raw if override_clear_raw is not None else "none",
            override_force_rerun=override_force_rerun,
            verbose=verbose,
            dry_run=dry_run,
        )

        if result.get("success"):
            console.print(f"[green]Reprocess completed:[/green] {result.get('message', '(no message)')}")
            raise typer.Exit(0)
        else:
            console_err.print(f"[bold red]Reprocess failed:[/bold red] {result.get('message', '(no message)')}")
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except ConfigurationError as e:
        console_err.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(2)
    except (WorkflowError, ProcessingError, SimulationError) as e:
        console_err.print(f"[bold red]Workflow Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console_err.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(10)


@app.command(name="delete")
def delete_command(
    system_config: Path = typer.Option(
        ...,
        "--system-config",
        help="Path to system configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    analysis_config: Path = typer.Option(
        ...,
        "--analysis-config",
        help="Path to analysis configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    override_in_flight: bool = typer.Option(
        False,
        "--override-in-flight",
        help="Bypass the live-SLURM-sentinel refusal guard. Use only when you know "
        "the jobs are dead but reconciliation cannot prove it (e.g., orphaned "
        "sentinels from worker hard-kill).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the interactive confirmation prompt (use for scripted invocation).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would be deleted (count of scenarios / sub-analyses + "
        "disk size estimate) without deleting anything.",
    ),
    skip_preview: bool = typer.Option(
        False,
        "--skip-preview",
        help="Skip the per-sub-analysis disk-utilization preview before "
        "deletion. Useful when the preview's per-sub-analysis `du -sh` "
        "walks dominate runtime on large Lustre trees (~minutes per TiB). "
        "Without the preview the user has no size context at the "
        "confirmation prompt; typically combined with --yes.",
    ),
):
    """Delete an entire analysis tree via distributed Snakemake workflow.

    Generates a Snakefile.delete with per-scenario (regular analysis) or
    per-sub-analysis (sensitivity) delete rules plus an analysis-level
    consolidation rule, then submits the workflow. On full success, the
    orchestrator removes ``analysis_dir/`` atomically; if any per-rule
    sentinel is missing, ``analysis_dir/`` is preserved for debugging.

    Refuses by default when ``_status/_submitted/*.json`` sentinels indicate
    live SLURM jobs. Pass ``--override-in-flight`` to bypass.

    Per cleanup-rerun-delete-redesign Phase 2.

    Examples:

        # Inspect what would be deleted without acting
        $ triton-swmm delete --system-config system.yaml \\
            --analysis-config analysis.yaml --dry-run

        # Delete after dry-run confirmation
        $ triton-swmm delete --system-config system.yaml \\
            --analysis-config analysis.yaml --yes

        # Delete despite orphaned in-flight sentinels (use sparingly)
        $ triton-swmm delete --system-config system.yaml \\
            --analysis-config analysis.yaml --override-in-flight --yes
    """
    try:
        from .analysis import TRITONSWMM_analysis
        from .system import TRITONSWMM_system

        system = TRITONSWMM_system(system_config)
        analysis = TRITONSWMM_analysis(analysis_config, system)
        system._analysis = analysis

        if skip_preview:
            console.print(
                f"[yellow]Preview skipped (--skip-preview). "
                f"Targeting {analysis.analysis_paths.analysis_dir}[/yellow]"
            )
        else:
            _print_delete_dry_run_summary(analysis)

        if dry_run:
            console.print("[yellow]Dry-run only — no deletion performed.[/yellow]")
            raise typer.Exit(0)

        if not yes:
            response = input(
                "Proceed with deletion? Type 'y' or 'yes' to confirm: "
            ).strip().lower()
            if response not in ("y", "yes"):
                console_err.print("[yellow]Aborted.[/yellow]")
                raise typer.Exit(1)

        analysis.delete(override_in_flight=override_in_flight)

        if analysis.analysis_paths.analysis_dir.exists():
            console_err.print(
                "[bold yellow]analysis_dir preserved — see [delete] log "
                "messages above for missing sentinels.[/bold yellow]"
            )
            raise typer.Exit(1)
        console.print("[green]Analysis deleted successfully.[/green]")
        raise typer.Exit(0)

    except typer.Exit:
        raise
    except ConfigurationError as e:
        console_err.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(2)
    except (WorkflowError, ProcessingError, SimulationError) as e:
        console_err.print(f"[bold red]Workflow Error:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console_err.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(10)


def _print_delete_dry_run_summary(analysis) -> None:
    """Print a per-scenario / per-sub-analysis breakdown of what
    ``analysis.delete()`` would remove from disk, plus a total size estimate.

    Per cleanup-rerun-delete-redesign Phase 2.
    """
    analysis_dir = analysis.analysis_paths.analysis_dir
    if not analysis_dir.exists():
        console.print(
            f"[yellow]analysis_dir does not exist: {analysis_dir}[/yellow]"
        )
        return

    def _du(path: Path) -> int:
        total = 0
        if not path.exists():
            return 0
        if path.is_file():
            try:
                return path.stat().st_size
            except OSError:
                return 0
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
        return total

    def _du_via_sentinel(path: Path) -> tuple[int, bool]:
        """Return (disk_utilization_bytes, from_sentinel).

        Falls back to `_du` walk when the `_status/_du.json` sentinel is
        absent and prints a stderr warning naming the missing sentinel.
        """
        import sys

        from TRITON_SWMM_toolkit.du_sentinels import read_du_sentinel

        sentinel_path = path / "_status" / "_du.json"
        payload = read_du_sentinel(sentinel_path)
        if payload is not None:
            return int(payload.get("disk_utilization_bytes", 0)), True
        print(
            f"[delete] DU sentinel absent — walking tree: {sentinel_path}",
            file=sys.stderr,
        )
        return _du(path), False

    def _fmt(size_bytes: int) -> str:
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0  # type: ignore[assignment]
        return f"{size_bytes:.1f} PiB"

    console.print(
        f"[bold]Delete preview for[/bold] {analysis_dir}"
    )
    total = 0
    if analysis.cfg_analysis.toggle_sensitivity_analysis:
        subanalyses_dir = analysis_dir / "subanalyses"
        sa_ids = list(analysis.sensitivity.df_setup.index.astype(str))
        console.print(f"  Sensitivity master with {len(sa_ids)} sub-analyses:")
        for sa_id in sa_ids:
            sa_dir = subanalyses_dir / f"sa_{sa_id}"
            size = _du_via_sentinel(sa_dir)[0]
            total += size
            console.print(f"    sa_{sa_id}: {_fmt(size)}  ({sa_dir})")
    else:
        sims_dir = analysis_dir / "sims"
        scen_dirs = sorted(sims_dir.glob("*")) if sims_dir.exists() else []
        console.print(f"  Regular analysis with {len(scen_dirs)} scenarios:")
        for sd in scen_dirs:
            size = _du_via_sentinel(sd)[0]
            total += size
            console.print(f"    {sd.name}: {_fmt(size)}")

    analysis_total = _du_via_sentinel(analysis_dir)[0]
    analysis_level_size = analysis_total - total
    total_size = analysis_total
    console.print(f"  Analysis-level artifacts: {_fmt(analysis_level_size)}")
    console.print(f"  [bold]Total to be removed:[/bold] {_fmt(total_size)}")


@app.command(name="cleanup-stale-metadata")
def cleanup_stale_metadata_command(
    system_config: Path = typer.Option(
        ...,
        "--system-config",
        help="Path to system configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    analysis_config: Path = typer.Option(
        ...,
        "--analysis-config",
        help="Path to analysis configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--apply",
        help="List orphan metadata records without deleting (default) or delete them with --apply",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Required with --apply to actually invoke snakemake --cleanup-metadata",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print each orphan path",
    ),
):
    """List or delete orphaned ``.snakemake/metadata/`` records left by past rule-output renames.

    The Phase 8 rule-output renames (``.png``/``.svg`` → ``.html`` for
    system_overview, per_sim plots, and sensitivity_benchmarking) leave orphaned
    metadata records that Snakemake will trigger a one-shot full plot rebuild
    against on first post-rename invocation. This command identifies those
    orphans and optionally removes them via ``snakemake --cleanup-metadata <paths>``.

    Examples:

        # List orphans (no deletion)
        $ triton-swmm cleanup-stale-metadata --system-config system.yaml \\
            --analysis-config analysis.yaml

        # Actually delete
        $ triton-swmm cleanup-stale-metadata --system-config system.yaml \\
            --analysis-config analysis.yaml --apply --force
    """
    try:
        from .analysis import TRITONSWMM_analysis
        from .system import TRITONSWMM_system

        system = TRITONSWMM_system(system_config)
        analysis = TRITONSWMM_analysis(analysis_config, system)
        system._analysis = analysis

        if not dry_run and not force:
            console_err.print(
                "[bold red]Error:[/bold red] --apply requires --force to confirm deletion."
            )
            raise typer.Exit(2)

        orphan_paths = analysis._enumerate_stale_metadata_paths()
        n = len(orphan_paths)

        if n == 0:
            console.print("[green]No orphan metadata candidates enumerated.[/green]")
        elif dry_run:
            console.print(
                f"[yellow]Found {n} orphan metadata candidate(s) (dry-run; nothing deleted).[/yellow]"
            )
            for p in orphan_paths:
                console.print(f"  orphan: {p}")
        else:
            if verbose:
                for p in orphan_paths:
                    console.print(f"  orphan: {p}")
            analysis._invoke_snakemake_cleanup_metadata(orphan_paths)
            console.print(
                f"[green]Invoked `snakemake --cleanup-metadata` against {n} orphan path(s).[/green]"
            )

        raise typer.Exit(0)

    except typer.Exit:
        raise
    except ConfigurationError as e:
        console_err.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(2)
    except Exception as e:
        console_err.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(10)


@app.command(name="cleanup-settled-markers")
def cleanup_settled_markers_command(
    system_config: Path = typer.Option(
        ...,
        "--system-config",
        help="Path to system configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    analysis_config: Path = typer.Option(
        ...,
        "--analysis-config",
        help="Path to analysis configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--apply",
        help="List settled markers without deleting (default) or delete them with --apply",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Required with --apply to actually unlink settled markers",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print each settled-marker path",
    ),
):
    """Prune settled ``_status/_completed`` / ``_status/_failed`` markers whose submitted-sentinel is gone.

    Inert hygiene, not correctness.

    A marker is *settled* when its sibling ``_status/_submitted/{token}.json`` is
    absent: the runner's try/finally wrote the terminal marker then deleted the
    submitted-sentinel, so the reconcile will never re-read it. These markers are
    pure accumulation over long resumable campaigns. This command lists them and
    optionally removes them.

    Examples:

        # List settled markers (no deletion)
        $ triton-swmm cleanup-settled-markers --system-config system.yaml \\
            --analysis-config analysis.yaml

        # Actually delete
        $ triton-swmm cleanup-settled-markers --system-config system.yaml \\
            --analysis-config analysis.yaml --apply --force
    """
    try:
        from .analysis import TRITONSWMM_analysis
        from .system import TRITONSWMM_system

        system = TRITONSWMM_system(system_config)
        analysis = TRITONSWMM_analysis(analysis_config, system)
        system._analysis = analysis

        if not dry_run and not force:
            console_err.print("[bold red]Error:[/bold red] --apply requires --force to confirm deletion.")
            raise typer.Exit(2)

        settled = analysis._prune_settled_markers(dry_run=dry_run)
        n = len(settled)

        if n == 0:
            console.print("[green]No settled markers found.[/green]")
        elif dry_run:
            console.print(f"[yellow]Found {n} settled marker(s) (dry-run; nothing deleted).[/yellow]")
            for p in settled:
                console.print(f"  settled: {p}")
        else:
            if verbose:
                for p in settled:
                    console.print(f"  settled: {p}")
            console.print(f"[green]Pruned {n} settled marker(s).[/green]")

        raise typer.Exit(0)

    except typer.Exit:
        raise
    except ConfigurationError as e:
        console_err.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(2)
    except Exception as e:
        console_err.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        raise typer.Exit(10)


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════


def _handle_list_testcases(catalog_path: Path | None) -> None:
    """Print available testcases and exit."""
    try:
        catalog = load_profile_catalog(catalog_path)
        testcases = list_testcases(catalog)

        if not testcases:
            console.print("[yellow]No testcases defined in catalog.[/yellow]")
            return

        table = Table(title="Available Testcases", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("Description")

        for name, description in testcases:
            table.add_row(name, description)

        console.print(table)

    except ConfigurationError as e:
        console_err.print(f"[bold red]Error loading catalog:[/bold red] {e}")
        raise typer.Exit(2)


def _handle_list_case_studies(catalog_path: Path | None) -> None:
    """Print available case studies and exit."""
    try:
        catalog = load_profile_catalog(catalog_path)
        case_studies = list_case_studies(catalog)

        if not case_studies:
            console.print("[yellow]No case studies defined in catalog.[/yellow]")
            return

        table = Table(title="Available Case Studies", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green")
        table.add_column("Description")

        for name, description in case_studies:
            table.add_row(name, description)

        console.print(table)

    except ConfigurationError as e:
        console_err.print(f"[bold red]Error loading catalog:[/bold red] {e}")
        raise typer.Exit(2)


def _validate_cli_arguments(**kwargs) -> None:
    """Validate business logic constraints on CLI arguments.

    Raises CLIValidationError on validation failure.
    """
    # Mutually exclusive: --from-scratch and --resume
    if kwargs["from_scratch"] and kwargs["resume"]:
        raise CLIValidationError(
            argument="--from-scratch/--resume",
            message="Cannot use both --from-scratch and --resume",
            fix_hint="Choose one or omit both (default is --resume)",
        )

    # Mutually exclusive: --event-ilocs and --event-range
    if kwargs["event_ilocs"] and kwargs["event_range"]:
        raise CLIValidationError(
            argument="--event-ilocs/--event-range",
            message="Cannot use both --event-ilocs and --event-range",
            fix_hint="Choose one event selection method",
        )

    # Mutually exclusive: --verbose and --quiet
    if kwargs["verbose"] and kwargs["quiet"]:
        raise CLIValidationError(
            argument="--verbose/--quiet",
            message="Cannot use both --verbose and --quiet",
            fix_hint="Choose one output mode",
        )

    # Profile validation
    valid_profiles = ["production", "testcase", "case-study"]
    if kwargs["profile"] not in valid_profiles:
        raise CLIValidationError(
            argument="--profile",
            message=f"Invalid profile: {kwargs['profile']}",
            fix_hint=f"Must be one of: {', '.join(valid_profiles)}",
        )

    # Conditional requirement: testcase profile requires --testcase
    if kwargs["profile"] == "testcase" and not kwargs["testcase"]:
        raise CLIValidationError(
            argument="--testcase",
            message="--testcase NAME required when --profile testcase",
            fix_hint="Specify testcase name or use --list-testcases to see available options",
        )

    # Conditional requirement: case-study profile requires --case-study
    if kwargs["profile"] == "case-study" and not kwargs["case_study"]:
        raise CLIValidationError(
            argument="--case-study",
            message="--case-study NAME required when --profile case-study",
            fix_hint="Specify case study name or use --list-case-studies to see available options",
        )

    # Model validation
    valid_models = ["auto", "triton", "swmm", "tritonswmm"]
    if kwargs["model"] not in valid_models:
        raise CLIValidationError(
            argument="--model",
            message=f"Invalid model: {kwargs['model']}",
            fix_hint=f"Must be one of: {', '.join(valid_models)}",
        )

    # Which validation
    valid_which = ["TRITON", "SWMM", "both"]
    if kwargs["which"] not in valid_which:
        raise CLIValidationError(
            argument="--which",
            message=f"Invalid which: {kwargs['which']}",
            fix_hint=f"Must be one of: {', '.join(valid_which)}",
        )

    # Redownload validation
    valid_redownload = ["none", "triton", "swmm", "all"]
    if kwargs["redownload"] not in valid_redownload:
        raise CLIValidationError(
            argument="--redownload",
            message=f"Invalid redownload: {kwargs['redownload']}",
            fix_hint=f"Must be one of: {', '.join(valid_redownload)}",
        )

    # Log level validation
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    if kwargs["log_level"] not in valid_log_levels:
        raise CLIValidationError(
            argument="--log-level",
            message=f"Invalid log level: {kwargs['log_level']}",
            fix_hint=f"Must be one of: {', '.join(valid_log_levels)}",
        )

    # Walltime format validation
    if kwargs["walltime"]:
        import re

        if not re.match(r"^\d{2}:\d{2}:\d{2}$", kwargs["walltime"]):
            raise CLIValidationError(
                argument="--walltime",
                message=f"Invalid walltime format: {kwargs['walltime']}",
                fix_hint="Use HH:MM:SS format (e.g., 01:30:00)",
            )


def _print_dry_run_summary(args: dict) -> None:
    """Print dry-run summary showing resolved configuration without execution."""
    console.print("\n[bold cyan]═══ Dry-Run Summary ═══[/bold cyan]\n")

    # Profile information
    console.print(f"[bold]Profile:[/bold] {args['profile']}")
    if args["profile"] == "testcase":
        console.print(f"[bold]Testcase:[/bold] {args['testcase']}")
    elif args["profile"] == "case-study":
        console.print(f"[bold]Case Study:[/bold] {args['case_study']}")

    # Configuration files
    console.print("\n[bold]Configuration Files:[/bold]")
    console.print(f"  System:   {args['system_config']}")
    console.print(f"  Analysis: {args['analysis_config']}")

    # Execution mode
    console.print("\n[bold]Execution Mode:[/bold]")
    console.print(f"  Model:  {args['model']}")
    console.print(f"  Which:  {args['which']}")
    console.print(f"  Resume: {args['resume']}")
    console.print(f"  From scratch: {args['from_scratch']}")

    # Event selection
    if args["event_ilocs"] or args["event_range"]:
        console.print("\n[bold]Event Selection:[/bold]")
        if args["event_ilocs"]:
            console.print(f"  Indices: {args['event_ilocs']}")
        if args["event_range"]:
            console.print(f"  Range: {args['event_range']}")

    # HPC overrides (if any)
    hpc_overrides = {
        k: v
        for k, v in args.items()
        if k in ["partition", "account", "nodes", "walltime", "cpus_per_task"] and v is not None
    }
    if hpc_overrides:
        console.print("\n[bold]HPC Overrides:[/bold]")
        for key, value in hpc_overrides.items():
            console.print(f"  {key}: {value}")

    console.print("\n[yellow]Note: Dry-run mode - no execution performed.[/yellow]\n")


@app.command(name="bundle")
def bundle_command(
    system_config: Path = typer.Option(
        ...,
        "--system-config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to system configuration YAML file",
    ),
    analysis_config: Path = typer.Option(
        ...,
        "--analysis-config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to analysis configuration YAML file",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        help=(
            "Target path for the bundle zip. Defaults to "
            "{analysis_dir}/render_bundle/{analysis_id}_{git_sha}_v{schema}.zip."
        ),
    ),
) -> None:
    """Emit a portable render bundle for local renderer iteration.

    Walks *.manifest.json provenance sidecars under {analysis_dir}/plots/
    and copies the union of declared source paths into a self-contained
    zip with relative-path configs and the HPC-baseline
    analysis_report.{html,zip} under bundle_baseline/.

    Requires render_report() to have been invoked at least once on the
    target analysis (so manifest sidecars exist). Raises FileNotFoundError
    if no manifests are found.
    """
    from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis
    from TRITON_SWMM_toolkit.system import TRITONSWMM_system

    system = TRITONSWMM_system(system_config)
    analysis = TRITONSWMM_analysis(analysis_config, system)
    if getattr(analysis.cfg_analysis, "toggle_sensitivity_analysis", False):
        bundle_path = analysis.sensitivity.bundle_report_data(output)
    else:
        bundle_path = analysis.bundle_report_data(output)
    console.print(f"[green]Bundle emitted:[/green] {bundle_path}")


@app.command(name="report-from-bundle")
def report_from_bundle_command(
    bundle_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=True,
        readable=True,
        help="Path to the bundle zip (or unpacked bundle directory).",
    ),
    format: str = typer.Option(
        "zip",
        "--format",
        help="Output format: 'zip' (single-HTML wrapped in a zip — default) or 'html' (uncompressed single-file).",
    ),
) -> None:
    # Render a fresh analysis_report from a portable render bundle.
    #
    # Per Plan Phase 3's rewire, this is a thin wrapper over
    # Bundle.from_directory(bundle_root).regenerate_report(format=format).
    # The Bundle class derives the static_backend from the bundle's
    # cfg_analysis.yaml (cfg-controlled default 'plotly' per Plan
    # Phase 2 D3 + Decision 4); no static_backend kwarg is threaded
    # through the CLI per Decision 3.3D.
    #
    import zipfile

    from TRITON_SWMM_toolkit.bundle import Bundle, _get_toolkit_git_sha

    if bundle_path.is_file() and bundle_path.suffix == ".zip":
        unpack_dir = bundle_path.parent / bundle_path.stem
        unpack_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_path) as zf:
            zf.extractall(unpack_dir)
        bundle_root = unpack_dir
    elif bundle_path.is_dir():
        bundle_root = bundle_path
    else:
        raise CLIValidationError(
            argument="bundle_path",
            message=f"{bundle_path} is neither a .zip file nor a directory",
            fix_hint=(
                "Pass a path to a bundle.zip produced by "
                "`TRITON_SWMM_toolkit bundle`, or to an unpacked bundle directory."
            ),
        )

    bundle = Bundle.from_directory(bundle_root)

    local_sha = _get_toolkit_git_sha(strict=False)
    bundle_sha = bundle.manifest.get("toolkit_git_sha", "unknown")
    if bundle_sha != "unknown" and local_sha != "unknown" and bundle_sha != local_sha:
        console.print(
            f"[yellow]Toolkit git SHA divergence:[/yellow] bundle={bundle_sha}, "
            f"local={local_sha}. The local re-render uses the locally installed "
            f"toolkit's report templates and post-process surgery; wrapper "
            f"sections may differ from HPC. Compare against bundle_baseline/."
        )

    for fmt in ("html", "zip"):
        prior = bundle_root / f"analysis_report.{fmt}"
        if prior.exists():
            # EXEMPT-DU: bundle-root
            prior.unlink()

    locks_dir = bundle_root / ".snakemake" / "locks"
    if locks_dir.exists() and any(locks_dir.iterdir()):
        console.print(
            f"[yellow]Stale Snakemake locks found at[/yellow] {locks_dir} — "
            f"removing (left behind by an interrupted prior render)."
        )
        from TRITON_SWMM_toolkit.utils import fast_rmtree

        # EXEMPT-DU: lock-file-cleanup
        fast_rmtree(locks_dir)
        locks_dir.mkdir(parents=True, exist_ok=True)

    rendered = bundle.regenerate_report(format=format)
    console.print(f"[green]Report rendered:[/green] {rendered}")


if __name__ == "__main__":
    app()
