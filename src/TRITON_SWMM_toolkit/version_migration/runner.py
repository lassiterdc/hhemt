"""Orchestrate dry-run / apply / verify of a migration sequence.

Public entry points:
    run_migration(target_dir, *, target=None, apply=False, cfg_paths=None)
    status(target_dir)
    baseline(target_dir, version, *, force=False)
    verify(target_dir)

`MigrationContext` (Phase 2) is the per-migration scratch surface;
runner.py constructs one per migration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from TRITON_SWMM_toolkit.version_migration import registry, state
from TRITON_SWMM_toolkit.version_migration.constants import (
    LAYOUT_VERSION,
    MINIMUM_SUPPORTED_VERSION,
)
from TRITON_SWMM_toolkit.version_migration.exceptions import (
    BaselineRequiredError,
    LayoutVersionError,
    MigrationConflictError,
)

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    target_dir: Path
    current_version: int
    target_version: int
    migrations_planned: list[str]
    migrations_applied: list[str]
    applied: bool


def status(target_dir: Path) -> RunResult:
    """Report current vs LAYOUT_VERSION; do not mutate."""
    current = _resolve_current(target_dir)
    plan_modules = registry.plan(current, LAYOUT_VERSION) if current >= 0 else []
    return RunResult(
        target_dir=target_dir,
        current_version=current,
        target_version=LAYOUT_VERSION,
        migrations_planned=[m.module_name for m in plan_modules],
        migrations_applied=[],
        applied=False,
    )


def baseline(target_dir: Path, version: int, *, force: bool = False) -> RunResult:
    """Stamp _version.json at ``version`` without running migrations."""
    if version < MINIMUM_SUPPORTED_VERSION or version > LAYOUT_VERSION:
        raise LayoutVersionError(
            current=-1,
            target=version,
            reason=(f"baseline {version} outside supported range [{MINIMUM_SUPPORTED_VERSION}, {LAYOUT_VERSION}]"),
        )
    existing = state.read_version_file(target_dir)
    if existing is not None and existing.layout_version != version and not force:
        raise LayoutVersionError(
            current=existing.layout_version,
            target=version,
            reason="refusing to baseline over existing _version.json without --force",
        )
    if existing is None:
        state.stamp_new_target(target_dir, version)
    else:
        if version < existing.layout_version:
            # --force path: dropping below current resets history (semantic
            # mismatch between layout_version and history would otherwise
            # be a logical inconsistency).
            new_state = state.VersionState(
                layout_version=version,
                toolkit_version=existing.toolkit_version,
                created_at=existing.created_at,
                migration_history=[],
            )
        else:
            new_state = _replace_layout(existing, version)
        state.write_version_file(target_dir, new_state)
    return RunResult(
        target_dir=target_dir,
        current_version=version,
        target_version=version,
        migrations_planned=[],
        migrations_applied=[],
        applied=True,
    )


def _replace_layout(existing: state.VersionState, version: int) -> state.VersionState:
    return state.VersionState(
        layout_version=version,
        toolkit_version=existing.toolkit_version,
        created_at=existing.created_at,
        migration_history=existing.migration_history,
    )


def run_migration(
    target_dir: Path,
    *,
    target: int | None = None,
    apply: bool = False,
    cfg_paths: dict[str, Path] | None = None,
) -> RunResult:
    """Plan and (optionally) apply the migration sequence to ``target_dir``.

    Renamed from ``migrate`` to disambiguate from the package name and CLI
    verb.

    ``cfg_paths`` (optional) maps {"system": Path, "analysis": Path} and is
    threaded into every MigrationContext via ``_construct_context``.
    Required by migrations whose ``upgrade()`` calls
    ``ctx.build_expected_slugs_for_current_version()`` (currently V0001).
    Ignored by migrations that do not.
    """
    target = LAYOUT_VERSION if target is None else target
    current = _resolve_current(target_dir)
    if current < MINIMUM_SUPPORTED_VERSION:
        raise LayoutVersionError(
            current=current,
            target=target,
            reason=(
                f"current layout_version={current} below "
                f"MINIMUM_SUPPORTED_VERSION={MINIMUM_SUPPORTED_VERSION}; "
                f"this analysis is too old to migrate"
            ),
        )
    plan_modules = registry.plan(current, target)
    if not plan_modules:
        logger.info(
            "[%s] current=%d, target=%d, no migrations needed",
            target_dir,
            current,
            target,
        )
        return RunResult(
            target_dir=target_dir,
            current_version=current,
            target_version=target,
            migrations_planned=[],
            migrations_applied=[],
            applied=apply,
        )

    applied: list[str] = []
    for m in plan_modules:
        ctx = _construct_context(target_dir, m, dry_run=not apply, cfg_paths=cfg_paths)
        try:
            m.upgrade(ctx)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Narrow wrapping: OSError covers filesystem conflicts;
            # ValueError / KeyError cover primitive argument validation;
            # TypeError catches Pydantic model-class resolution failures.
            # Programming errors (AttributeError, NameError) propagate
            # as-is so they are not mistaken for retryable migration
            # conflicts.
            raise MigrationConflictError(
                version=m.version_to,
                op_index=len(ctx.plan),
                reason=str(exc),
            ) from exc
        for op in ctx.plan:
            print(f"[V{m.version_to:04d}] {op}", flush=True)
        if apply:
            ctx.execute()
            state.record_migration(target_dir, m.version_from, m.version_to, m.module_name)
            applied.append(m.module_name)
    return RunResult(
        target_dir=target_dir,
        current_version=current,
        target_version=target,
        migrations_planned=[m.module_name for m in plan_modules],
        migrations_applied=applied,
        applied=apply,
    )


def _construct_context(
    target_dir: Path,
    m: registry.MigrationModule,
    *,
    dry_run: bool,
    cfg_paths: dict[str, Path] | None = None,
):
    """Construct a MigrationContext.

    Phase 1 defers to the Phase-2 `MigrationContext` when available. Phase 2
    implements the real DSL; Phase 1 has no migrations registered, so this
    path is only exercised when a synthetic migration is wired in for
    testing.
    """
    from TRITON_SWMM_toolkit.version_migration.context import (  # Phase 2
        MigrationContext,
    )

    return MigrationContext(
        target_dir=target_dir,
        dry_run=dry_run,
        migration_id=m.module_name,
        cfg_paths=cfg_paths,
    )


def _resolve_current(target_dir: Path) -> int:
    inferred = state.infer_layout_version(target_dir)
    if inferred is None:
        raise BaselineRequiredError(target_dir)
    return inferred


def verify(target_dir: Path) -> bool:
    """Compare target_dir to schema/layout_schema_v{LAYOUT_VERSION}.json.

    Phase 1 returns True if _version.json shows current LAYOUT_VERSION,
    False otherwise. Phase 5 expands this to full schema-walk comparison.
    """
    st = state.read_version_file(target_dir)
    if st is None:
        return False
    return st.layout_version == LAYOUT_VERSION
