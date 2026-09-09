"""Migration exception hierarchy.

All migration errors derive from ``MigrationError`` which in turn derives
from ``TRITONSWMMError`` (the project's exception root). This lets the
CLI map structured exceptions to documented exit codes.
"""

from __future__ import annotations

from pathlib import Path

from hhemt.exceptions import TRITONSWMMError


class MigrationError(TRITONSWMMError):
    """Base for all version-migration errors."""


class LayoutVersionError(MigrationError):
    """Raised when current/target version state is invalid.

    Covers downgrade attempts, target below the floor, and applying a
    migration against an unexpected on-disk version.
    """

    def __init__(self, current: int, target: int, reason: str) -> None:
        self.current = current
        self.target = target
        self.reason = reason
        super().__init__(f"layout version error: current={current}, target={target}: {reason}")


class BaselineRequiredError(MigrationError):
    """Raised when the detection ladder cannot infer a layout version and
    the user must explicitly baseline."""

    def __init__(self, target_dir: Path) -> None:
        self.target_dir = target_dir
        super().__init__(
            f"cannot infer layout version for {target_dir}; "
            f"use `python -m hhemt.version_migration "
            f"baseline {target_dir} {{N}}`"
        )


class MigrationConflictError(MigrationError):
    """Raised mid-apply when a primitive cannot complete.

    Covers filesystem conflicts, permission errors, and other runtime
    failures surfaced by a primitive.
    """

    def __init__(self, version: int, op_index: int, reason: str) -> None:
        self.version = version
        self.op_index = op_index
        self.reason = reason
        super().__init__(
            f"migration V{version:04d} failed at operation {op_index}; "
            f"partial state at layout_version={version - 1}: {reason}; "
            f"re-run --apply after resolving"
        )


class RegistryError(MigrationError):
    """Raised when migration discovery finds gaps, duplicates, or invalid
    modules."""

    def __init__(self, reason: str, paths: list[Path] | None = None) -> None:
        self.reason = reason
        self.paths = paths or []
        super().__init__(f"registry error: {reason}; paths={self.paths}")


class MigrationBlockedError(MigrationError):
    """Raised when a migration's upgrade() cannot proceed without operator
    remediation. The message names the analysis path and the minimum edit
    the operator must apply before re-running. Mapped to validation exit
    code (2) by the CLI."""


class VersionFileUnreadableError(MigrationError):
    """Raised when `_version.json` EXISTS but does not parse as a layout-version record.

    Distinct from `BaselineRequiredError`, which means the version could not be INFERRED
    from a tree carrying no record. Here a record is present and is broken, so the remedy
    is to inspect or replace that file rather than to infer around it.

    SUBCLASSES `MigrationError` DELIBERATELY, AND NOT `ValueError`. The runner catches
    `(OSError, ValueError, KeyError, TypeError)` around `upgrade()` and re-wraps them as
    `MigrationConflictError`; a `ValueError` subclass would therefore be relabelled and the
    operator would be told a migration conflicted when in fact the tree's own record is
    unreadable. Verified: `issubclass(VersionFileUnreadableError, (OSError, ValueError,
    KeyError, TypeError))` is False, so it reaches the operator unwrapped."""
