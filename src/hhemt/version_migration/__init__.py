"""hhemt version-migration system.

Forward-only migrations of persistent on-disk state in analysis trees and
system directories, authored as numbered Python modules under
``versions/`` and applied via the ``MigrationContext`` primitive DSL.

The public surface:

- ``LAYOUT_VERSION`` is the current canonical layout version.
- ``MINIMUM_SUPPORTED_VERSION`` is the lowest layout version that migration accepts.
- ``run_migration``, ``status``, ``baseline``, and ``verify`` are high-level functions that the CLI also exposes.
- ``MigrationError`` is the exception base.
"""

from __future__ import annotations

from hhemt.version_migration.constants import (
    LAYOUT_VERSION,
    MINIMUM_SUPPORTED_VERSION,
)
from hhemt.version_migration.exceptions import (
    BaselineRequiredError,
    LayoutVersionError,
    MigrationConflictError,
    MigrationError,
    RegistryError,
)
from hhemt.version_migration.runner import (
    baseline,
    run_migration,
    status,
    verify,
)

__all__ = [
    "LAYOUT_VERSION",
    "MINIMUM_SUPPORTED_VERSION",
    "MigrationError",
    "LayoutVersionError",
    "BaselineRequiredError",
    "MigrationConflictError",
    "RegistryError",
    "run_migration",
    "status",
    "baseline",
    "verify",
]
