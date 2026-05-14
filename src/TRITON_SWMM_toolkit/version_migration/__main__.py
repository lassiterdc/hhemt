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
    MigrationBlockedError,
    MigrationConflictError,
    MigrationError,
    RegistryError,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Exit-code map per cli_utils.py convention: 2=validation/usage, 4=execution.
_VALIDATION_ERRORS: tuple[type[MigrationError], ...] = (
    LayoutVersionError,
    BaselineRequiredError,
    MigrationBlockedError,
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


# ---- V0005 dry-run reconnaissance ----

# Disposition labels used by `dry-run-report`. RECOVERABLE means V0005's
# Snakefile-grep would succeed against the referenced path. The two
# UNRECOVERABLE_* labels mirror V0005's `_recover_source_report_cfg_path`
# failure-mode classification.
_DRYRUN_RECOVERABLE = "RECOVERABLE"
_DRYRUN_NO_FLAG = "UNRECOVERABLE_NO_FLAG"
_DRYRUN_FILE_MISSING = "UNRECOVERABLE_FILE_MISSING"


def _classify_for_dry_run(analysis_dir: Path) -> tuple[str, str]:
    """Run V0005's recovery logic against `analysis_dir` without writing.

    Returns `(label, detail)` where label is one of the _DRYRUN_* constants
    and detail is the underlying reason string (or the recovered source path
    when RECOVERABLE)."""
    from TRITON_SWMM_toolkit.version_migration.versions.V0005__inline_report_config import (
        _recover_source_report_cfg_path,
    )

    src_path, reason = _recover_source_report_cfg_path(analysis_dir)
    if src_path is not None:
        return _DRYRUN_RECOVERABLE, str(src_path)
    if reason and "no Snakefile" in reason:
        return _DRYRUN_NO_FLAG, reason
    if reason and "no `--report-config" in reason:
        return _DRYRUN_NO_FLAG, reason
    return _DRYRUN_FILE_MISSING, reason or "unknown"


def _discover_analysis_dirs(root: Path) -> list[Path]:
    """Find analysis dirs under `root`. An analysis dir is identified by the
    presence of `cfg_analysis.yaml` at its top level. Returns a sorted list
    of unique paths. When `root` itself is an analysis dir, returns
    `[root]`."""
    if not root.exists():
        return []
    if (root / "cfg_analysis.yaml").exists():
        return [root]
    return sorted({p.parent for p in root.rglob("cfg_analysis.yaml")})


@app.command(name="dry-run-report")
def dry_run_report(
    roots: list[Path] = typer.Option(
        ...,
        "--roots",
        help=(
            "One or more analysis-corpus roots to scan. Each is walked for "
            "directories containing cfg_analysis.yaml; V0005's recovery "
            "logic is run against each without writing."
        ),
    ),
) -> None:
    """V0005 dry-run reconnaissance: classify each analysis dir as
    RECOVERABLE / UNRECOVERABLE_NO_FLAG / UNRECOVERABLE_FILE_MISSING and
    print a per-analysis table + summary counts. Exit 0 regardless of the
    recoverability split; this is a read-only reconnaissance surface used
    by the Phase 2 operator step in bundle-cfg-report-canonicalization."""
    counts = {_DRYRUN_RECOVERABLE: 0, _DRYRUN_NO_FLAG: 0, _DRYRUN_FILE_MISSING: 0}
    rows: list[tuple[Path, str, str]] = []
    for root in roots:
        for analysis_dir in _discover_analysis_dirs(root):
            label, detail = _classify_for_dry_run(analysis_dir)
            counts[label] += 1
            rows.append((analysis_dir, label, detail))

    typer.echo("V0005 dry-run report")
    typer.echo("=" * 70)
    for analysis_dir, label, detail in rows:
        typer.echo(f"  {label:<28s}  {analysis_dir}")
        typer.echo(f"      detail: {detail}")
    typer.echo("-" * 70)
    typer.echo(
        f"counts: {_DRYRUN_RECOVERABLE}={counts[_DRYRUN_RECOVERABLE]}, "
        f"{_DRYRUN_NO_FLAG}={counts[_DRYRUN_NO_FLAG]}, "
        f"{_DRYRUN_FILE_MISSING}={counts[_DRYRUN_FILE_MISSING]}"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
