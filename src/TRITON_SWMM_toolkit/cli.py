"""Command-line interface for TRITON-SWMM Toolkit.

Provides a Snakemake-first single-command CLI for running TRITON-SWMM
workflows with support for production, testcase, and case-study profiles.
"""

from pathlib import Path
from typing import List, Optional
import sys

import typer
from rich.console import Console
from rich.table import Table

from .cli_utils import map_exception_to_exit_code
from .profile_catalog import (
    load_profile_catalog,
    get_profile_entry,
    list_testcases,
    list_case_studies,
)
from .exceptions import (
    CLIValidationError,
    ConfigurationError,
    CompilationError,
    SimulationError,
    ProcessingError,
    WorkflowError,
    WorkflowPlanningError,
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
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Execution profile: production, testcase, or case-study",
    ),
    system_config: Optional[Path] = typer.Option(
        None,
        "--system-config",
        help="Path to system configuration YAML file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    analysis_config: Optional[Path] = typer.Option(
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
    event_ilocs: Optional[str] = typer.Option(
        None,
        "--event-ilocs",
        help="Comma-separated event indices (e.g., '0,1,2,10')",
    ),
    event_range: Optional[str] = typer.Option(
        None,
        "--event-range",
        help="Event range START:END (e.g., '0:100', inclusive start, exclusive end)",
    ),
    # ═══════════════════════════════════════════════════════════════
    # Profile-Specific Options
    # ═══════════════════════════════════════════════════════════════
    testcase: Optional[str] = typer.Option(
        None,
        "--testcase",
        help="Testcase name (required when --profile testcase)",
    ),
    case_study: Optional[str] = typer.Option(
        None,
        "--case-study",
        help="Case study name (required when --profile case-study)",
    ),
    tests_case_config: Optional[Path] = typer.Option(
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
    # ═══════════════════════════════════════════════════════════════
    # HPC Override Options
    # ═══════════════════════════════════════════════════════════════
    platform_config: Optional[str] = typer.Option(
        None, "--platform-config", help="Platform configuration name"
    ),
    partition: Optional[str] = typer.Option(
        None, "--partition", help="SLURM partition override"
    ),
    account: Optional[str] = typer.Option(
        None, "--account", help="SLURM account override"
    ),
    qos: Optional[str] = typer.Option(
        None, "--qos", help="SLURM QoS override"
    ),
    nodes: Optional[int] = typer.Option(
        None, "--nodes", help="Number of nodes", min=1
    ),
    ntasks_per_node: Optional[int] = typer.Option(
        None, "--ntasks-per-node", help="Tasks per node", min=1
    ),
    cpus_per_task: Optional[int] = typer.Option(
        None, "--cpus-per-task", help="CPUs per task", min=1
    ),
    gpus_per_node: Optional[int] = typer.Option(
        None, "--gpus-per-node", help="GPUs per node", min=0
    ),
    walltime: Optional[str] = typer.Option(
        None, "--walltime", help="Walltime limit (HH:MM:SS format)"
    ),
    # ═══════════════════════════════════════════════════════════════
    # Workflow Engine Options
    # ═══════════════════════════════════════════════════════════════
    jobs: Optional[int] = typer.Option(
        None,
        "--jobs",
        "-j",
        help="Parallel jobs for workflow execution",
        min=1,
    ),
    workflow_target: Optional[str] = typer.Option(
        None,
        "--workflow-target",
        help="Explicit Snakemake target/rule group (advanced)",
    ),
    snakemake_args: Optional[List[str]] = typer.Option(
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
                fix_hint="Specify production, testcase, or case-study"
            )
        if not system_config:
            raise CLIValidationError(
                argument="--system-config",
                message="--system-config is required",
                fix_hint="Provide path to system configuration YAML file"
            )
        if not analysis_config:
            raise CLIValidationError(
                argument="--analysis-config",
                message="--analysis-config is required",
                fix_hint="Provide path to analysis configuration YAML file"
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
        # Stage 3: Profile Resolution (if applicable)
        # ═══════════════════════════════════════════════════════════════
        # TODO: Implement profile resolution and config merging

        # ═══════════════════════════════════════════════════════════════
        # Stage 4: Config Loading
        # ═══════════════════════════════════════════════════════════════
        # TODO: Load system and analysis configs

        # ═══════════════════════════════════════════════════════════════
        # Stage 5: Preflight Validation
        # ═══════════════════════════════════════════════════════════════
        # TODO: Run preflight validation

        # ═══════════════════════════════════════════════════════════════
        # Stage 6: Dry-Run Output (if requested)
        # ═══════════════════════════════════════════════════════════════
        if dry_run:
            _print_dry_run_summary(locals())
            raise typer.Exit(0)

        # ═══════════════════════════════════════════════════════════════
        # Stage 7: Orchestration
        # ═══════════════════════════════════════════════════════════════
        # TODO: Wire to Analysis class and execute workflow

        console.print("[bold green]✓ Workflow complete![/bold green]")
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


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════


def _handle_list_testcases(catalog_path: Optional[Path]) -> None:
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


def _handle_list_case_studies(catalog_path: Optional[Path]) -> None:
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
        if not re.match(r'^\d{2}:\d{2}:\d{2}$', kwargs["walltime"]):
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
    if args['profile'] == 'testcase':
        console.print(f"[bold]Testcase:[/bold] {args['testcase']}")
    elif args['profile'] == 'case-study':
        console.print(f"[bold]Case Study:[/bold] {args['case_study']}")

    # Configuration files
    console.print(f"\n[bold]Configuration Files:[/bold]")
    console.print(f"  System:   {args['system_config']}")
    console.print(f"  Analysis: {args['analysis_config']}")

    # Execution mode
    console.print(f"\n[bold]Execution Mode:[/bold]")
    console.print(f"  Model:  {args['model']}")
    console.print(f"  Which:  {args['which']}")
    console.print(f"  Resume: {args['resume']}")
    console.print(f"  From scratch: {args['from_scratch']}")

    # Event selection
    if args['event_ilocs'] or args['event_range']:
        console.print(f"\n[bold]Event Selection:[/bold]")
        if args['event_ilocs']:
            console.print(f"  Indices: {args['event_ilocs']}")
        if args['event_range']:
            console.print(f"  Range: {args['event_range']}")

    # HPC overrides (if any)
    hpc_overrides = {
        k: v for k, v in args.items()
        if k in ['partition', 'account', 'nodes', 'walltime', 'cpus_per_task']
        and v is not None
    }
    if hpc_overrides:
        console.print(f"\n[bold]HPC Overrides:[/bold]")
        for key, value in hpc_overrides.items():
            console.print(f"  {key}: {value}")

    console.print("\n[yellow]Note: Dry-run mode - no execution performed.[/yellow]\n")


if __name__ == "__main__":
    app()
