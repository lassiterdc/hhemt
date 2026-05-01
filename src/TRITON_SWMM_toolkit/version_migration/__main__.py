"""CLI for the version-migration system.

Usage:
    python -m TRITON_SWMM_toolkit.version_migration migrate {dir} \
        [--target N] [--apply]
    python -m TRITON_SWMM_toolkit.version_migration status {dir}
    python -m TRITON_SWMM_toolkit.version_migration baseline {dir} {version} \
        [--force]
    python -m TRITON_SWMM_toolkit.version_migration verify {dir}
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from TRITON_SWMM_toolkit.version_migration import runner
from TRITON_SWMM_toolkit.version_migration.exceptions import (
    BaselineRequiredError,
    LayoutVersionError,
    MigrationConflictError,
    MigrationError,
    RegistryError,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Exit-code map per cli_utils.py convention: 2=validation/usage, 4=execution.
_VALIDATION_ERRORS: tuple[type[MigrationError], ...] = (
    LayoutVersionError,
    BaselineRequiredError,
    RegistryError,
)


def _exit_code_for(exc: MigrationError) -> int:
    """Map MigrationError subclasses to CLI exit codes per cli_utils.py
    convention."""
    if isinstance(exc, _VALIDATION_ERRORS):
        return 2
    if isinstance(exc, MigrationConflictError):
        return 4
    return 4  # default for unmapped MigrationError subclasses


app = typer.Typer(help="TRITON-SWMM_toolkit version-migration CLI")


@app.command()
def migrate(
    target_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    target: int = typer.Option(None, help="Target layout_version (defaults to LAYOUT_VERSION)"),
    apply: bool = typer.Option(False, help="Apply migrations (default: dry-run)"),
    system_config: Path | None = typer.Option(
        None,
        "--system-config",
        help=("Path to system YAML (required for migrations that need slug derivation, e.g., V0001)"),
    ),
    analysis_config: Path | None = typer.Option(
        None,
        "--analysis-config",
        help=("Path to analysis YAML (required for migrations that need slug derivation, e.g., V0001)"),
    ),
) -> None:
    cfg_paths: dict[str, Path] | None = None
    if system_config is not None and analysis_config is not None:
        cfg_paths = {"system": system_config, "analysis": analysis_config}
    try:
        result = runner.run_migration(target_dir, target=target, apply=apply, cfg_paths=cfg_paths)
    except MigrationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=_exit_code_for(exc)) from exc
    typer.echo(
        f"current={result.current_version}, target={result.target_version}, "
        f"planned={result.migrations_planned}, applied={result.migrations_applied}"
    )


@app.command()
def status(
    target_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    try:
        result = runner.status(target_dir)
    except MigrationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=_exit_code_for(exc)) from exc
    typer.echo(f"current={result.current_version}, target={result.target_version}, planned={result.migrations_planned}")


@app.command()
def baseline(
    target_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    version: int = typer.Argument(...),
    force: bool = typer.Option(False, help="Overwrite existing _version.json"),
) -> None:
    try:
        result = runner.baseline(target_dir, version, force=force)
    except MigrationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=_exit_code_for(exc)) from exc
    typer.echo(f"baselined at layout_version={result.current_version}")


@app.command()
def verify(
    target_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
) -> None:
    ok = runner.verify(target_dir)
    if ok:
        typer.echo("verify: OK")
    else:
        typer.echo(
            "verify: FAIL - current layout does not match LAYOUT_VERSION",
            err=True,
        )
        raise typer.Exit(code=4)


if __name__ == "__main__":  # pragma: no cover
    app()
